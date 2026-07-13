"""
50_ism_rnamsm_msa.py — RIBOSCOPE G4 step 6: the THIRD model. In-silico
mutagenesis of disease RNAs through RNA-MSM in its NATIVE MSA input.

Why
---
RNA-FM (structure-naive) is at chance on the U4/RNU4-2 ClinVar pathogenic set;
ErnieRNA (secondary-structure-aware) is significant (AUC 0.69, p=0.004). The
prediction that completes the architecture dissection: the EVOLUTIONARY model
(RNA-MSM, trained on MSAs) should localize the U4 hotspot at least as well,
because that hotspot is defined by population-level conservation — RNA-MSM's
native signal. This script produces RNA-MSM's per-nucleotide criticality map so
49 can test it on the same ClinVar positions: naive < 2ary-structure < MSA?

Method (reuses the validated MSA machinery from 16/17 + MAFFT --add from 33)
---------------------------------------------------------------------------
For each disease RNA with a known Rfam family:
  1. Parse the family's Rfam SEED alignment (gaps kept).
  2. MAFFT --add (NO --keeplength, so ALL query residues are preserved and the
     numbering stays 1:1 with the RefSeq / ClinVar coordinates) to align the
     human disease sequence into the SEED columns. Assert ungap(query)==input.
  3. Build the MSA = [aligned query row] + up to MSA_DEPTH-1 aligned SEED rows.
  4. RNA-MSM MSA forward (model.embeddings -> model.encoder, layer-8 hook); take
     the query row, keep CLS+EOS+non-gap query columns -> WT residue embeddings.
  5. ISM: mutate each query residue (in its alignment column) to the 3 alts,
     re-run, measure the mean L2 change of the query-row layer-8 embedding.
     -> representation-sensitivity criticality, same readout as 47's embed map.

Output schema MATCHES 47 (model="rnamsm") so 48/49 ingest it with no changes.

Requires MAFFT (conda install -c bioconda mafft) + RNA-MSM (multimolecule).

Run with
--------
    cd ~/projects/riboscope
    uv run python 50_ism_rnamsm_msa.py            # RNU4-2 + RMRP
    uv run python 50_ism_rnamsm_msa.py RNU4-2     # one gene

Inputs : sequences/disease_structured_rna.fasta, data/Rfam.seed.gz
Output : outputs/ism_criticality_rnamsm.json
"""

from __future__ import annotations

import gzip
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import torch
    from multimolecule import RnaTokenizer, RnaMsmModel
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

# ============================ CONFIG ============================
FASTA_FILE = Path("sequences/disease_structured_rna.fasta")
RFAM_SEED = Path("data/Rfam.seed.gz")
OUT = Path("outputs/ism_criticality_rnamsm.json")
LAYER = 8                      # most de-collapsed (headline RNA-MSM layer)
MSA_DEPTH = 32
RNG_SEED = 42
BASES = ("A", "C", "G", "U")

# disease RNA -> its Rfam family (so we can build the native MSA)
GENE_RFAM = {
    "RNU4-2": "RF00015",       # U4 snRNA — primary (ReNU)
    "RMRP":   "RF00030",       # RNase MRP
    "RNU6-1": "RF00026",       # U6
    "RN7SL1": "RF00017",       # SRP
    "TERC":   "RF00024",       # telomerase
}
MODEL_NAME = "multimolecule/rnamsm"
FALLBACK_MODELS = ["multimolecule/RNA-MSM", "yikun-zhang/RNA-MSM"]
# ================================================================


def parse_fasta(path: Path) -> dict[str, str]:
    out, gene, buf = {}, None, []
    def flush():
        if gene is not None:
            out[gene] = "".join(buf).upper().replace("T", "U")
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                gene = line[1:].split("|")[0].strip()
                buf = []
            else:
                buf.append(line)
        flush()
    return out


def norm_aligned(s: str) -> str:
    s = s.upper().replace("T", "U").replace("-", ".")
    return "".join(c if c in "ACGU." else "N" for c in s)


def ungap(s: str) -> str:
    return s.replace(".", "")


def parse_seed_rows(path: Path, rfam: str) -> list[str] | None:
    with gzip.open(path, "rt", encoding="latin-1") as f:
        block = []
        for line in f:
            block.append(line)
            if line.startswith("//"):
                fam_id, rows = _parse_block("".join(block))
                block = []
                if fam_id == rfam and rows:
                    widths = {len(r) for r in rows}
                    if len(widths) == 1:
                        return rows
    return None


def _parse_block(text: str):
    bufs, fam_id = {}, None
    for line in text.split("\n"):
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("#=GF AC"):
            fam_id = line.split(maxsplit=2)[2].strip()
        elif line.startswith("#") or line.startswith("//"):
            continue
        else:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                bufs[parts[0]] = bufs.get(parts[0], "") + parts[1]
    return fam_id, [norm_aligned(a) for a in bufs.values()]


def write_fasta(path, records):
    with open(path, "w") as f:
        for n, s in records:
            f.write(f">{n}\n{s}\n")


def mafft_add_full(seed_aln_rows: list[str], query_seq: str) -> dict | None:
    """mafft --add (NO --keeplength) -> {'QUERY':aln, 'SEED_i':aln,...} all same width."""
    with tempfile.TemporaryDirectory() as td:
        seed_fa, add_fa = Path(td) / "seed.fa", Path(td) / "add.fa"
        write_fasta(seed_fa, [(f"SEED_{i}", s.replace(".", "-")) for i, s in enumerate(seed_aln_rows)])
        write_fasta(add_fa, [("QUERY", query_seq)])
        proc = subprocess.run(["mafft", "--add", str(add_fa), "--quiet", str(seed_fa)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"mafft failed: {proc.stderr[:300]}")
        out, name, buf = {}, None, []
        for line in proc.stdout.splitlines():
            if line.startswith(">"):
                if name is not None:
                    out[name] = "".join(buf)
                name, buf = line[1:].strip(), []
            else:
                buf.append(line.strip())
        if name is not None:
            out[name] = "".join(buf)
        return out


def find_encoder_layers(model):
    for path in ("encoder.layer", "bert.encoder.layer", "rnamsm.encoder.layer",
                 "msm.encoder.layer", "model.encoder.layer"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            _ = len(obj)
            return obj
        except (AttributeError, TypeError):
            continue
    raise AttributeError("encoder.layer not found")


def make_hook(captured, layer_idx):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        captured[layer_idx] = output.detach().to(torch.float32).cpu()
    return hook


def msa_forward_qrow(model, tokenizer, rows, layer, captured, device):
    """Forward an MSA (list of equal-width aligned rows, query=row 0); return the
    query-row layer-`layer` embedding [C, D]."""
    enc = [tokenizer(r, return_tensors="pt")["input_ids"][0] for r in rows]
    if len({t.shape[0] for t in enc}) != 1:
        raise ValueError("ragged token lengths within MSA")
    ids = torch.stack(enc, 0).unsqueeze(0).to(device)     # [1,R,C]
    mask = torch.ones_like(ids)
    captured.clear()
    emb = model.embeddings(input_ids=ids, attention_mask=mask)
    _ = model.encoder(emb, attention_mask=mask)
    cap = captured[layer]                                  # [R,C,1,D]
    return cap[0].squeeze(1), ids.shape[2]                 # [C,D], C


def residue_cols(aln_query: str) -> list[int]:
    """Alignment column indices (0-based) of the query's non-gap residues, in order."""
    return [c for c, ch in enumerate(aln_query) if ch != "."]


def kept_mask(aln_query: str, C: int) -> torch.Tensor:
    m = torch.zeros(C, dtype=torch.bool)
    m[0] = True
    m[C - 1] = True
    for c, ch in enumerate(aln_query):
        if ch != ".":
            m[c + 1] = True            # +1 for CLS at front
    return m


def load_model(device):
    for cand in [MODEL_NAME] + FALLBACK_MODELS:
        try:
            tok = RnaTokenizer.from_pretrained(cand)
            model = RnaMsmModel.from_pretrained(cand).eval().to(device)
            print(f"   ✓ loaded {cand}")
            return tok, model
        except Exception as e:  # noqa: BLE001
            print(f"   ✗ {cand}: {type(e).__name__}: {str(e)[:80]}")
    raise SystemExit("❌ could not load RNA-MSM")


def main() -> None:
    want = [a for a in sys.argv[1:] if not a.startswith("-")]
    if shutil.which("mafft") is None:
        print("❌ MAFFT not found. Install: conda install -c bioconda mafft")
        sys.exit(1)
    for p in (FASTA_FILE, RFAM_SEED):
        if not p.exists():
            print(f"❌ Missing {p}")
            sys.exit(1)

    seqs = parse_fasta(FASTA_FILE)
    genes = want or ["RNU4-2", "RMRP"]
    genes = [g for g in genes if g in seqs and g in GENE_RFAM]
    if not genes:
        print(f"❌ No usable genes. Available: {list(seqs)}; mapped: {list(GENE_RFAM)}")
        sys.exit(1)

    print("=" * 80)
    print(f"RIBOSCOPE G4 RNA-MSM MSA in-silico mutagenesis — layer {LAYER}, genes {genes}")
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = load_model(device)
    layers = find_encoder_layers(model)
    captured: dict = {}
    handle = layers[LAYER].register_forward_hook(make_hook(captured, LAYER))
    cfg_maxpos = getattr(model.config, "max_position_embeddings", 1024)

    results = {}
    rng = random.Random(RNG_SEED)
    with torch.no_grad():
        for gene in genes:
            rfam = GENE_RFAM[gene]
            q = "".join(c if c in "ACGU" else "N"
                        for c in seqs[gene].upper().replace("T", "U"))
            L = len(q)
            seed_rows = parse_seed_rows(RFAM_SEED, rfam)
            if not seed_rows:
                print(f"  {gene} ({rfam}): no SEED alignment found — skipping.")
                continue
            ctx = seed_rows[:]
            rng.shuffle(ctx)
            ctx = ctx[: MSA_DEPTH - 1]
            try:
                aligned = mafft_add_full(ctx, q)
            except Exception as e:  # noqa: BLE001
                print(f"  {gene}: MAFFT error: {e}")
                continue
            aln_q = norm_aligned(aligned.get("QUERY", ""))
            if ungap(aln_q) != q:
                print(f"  {gene}: aligned query ungaps to len {len(ungap(aln_q))} != {L} "
                      f"(MAFFT altered residues) — skipping to preserve numbering.")
                continue
            rows = [aln_q] + [norm_aligned(aligned[f"SEED_{i}"]) for i in range(len(ctx))
                              if f"SEED_{i}" in aligned]
            W = len(aln_q)
            if W + 2 > cfg_maxpos:
                print(f"  {gene}: aligned width {W} exceeds model max {cfg_maxpos} — skipping.")
                continue
            cols = residue_cols(aln_q)            # len L, alignment col per residue
            assert len(cols) == L

            # WT embedding
            qrow_wt, C = msa_forward_qrow(model, tok, rows, LAYER, captured, device)
            m = kept_mask(aln_q, C)
            wt = qrow_wt[m]                        # [L+2, D]
            if wt.shape[0] != L + 2:
                print(f"  {gene}: kept {wt.shape[0]} != L+2 {L+2} — skipping.")
                continue
            wt_res = wt[1:L + 1].to(device)        # [L, D] residue embeddings

            crit = [0.0] * L
            for i in tqdm(range(L), desc=f"{gene} rnamsm-ISM", unit="pos", leave=False):
                ci = cols[i]
                wt_b = aln_q[ci]
                alts = [b for b in BASES if b != wt_b]
                acc = 0.0
                for b in alts:
                    mq = aln_q[:ci] + b + aln_q[ci + 1:]
                    mrows = [mq] + rows[1:]
                    qrow_m, Cm = msa_forward_qrow(model, tok, mrows, LAYER, captured, device)
                    mm = kept_mask(mq, Cm)         # gap pattern unchanged by a substitution
                    em = qrow_m[mm][1:L + 1].to(device)
                    if em.shape == wt_res.shape:
                        acc += float((em - wt_res).norm(dim=1).mean().item())
                crit[i] = acc / len(alts)

            mx = max(crit) or 1.0
            results[gene] = {
                "length": L,
                "verdict": "RNA-MSM native-MSA (embed-only)",
                "has_feature_readout": False,
                "readout_features": [],
                "readout_is_specialist": False,
                "fired_features_all": [],
                "primary_map": "embedding",
                "criticality_feature": None,
                "criticality_embedding": [round(c / mx, 4) for c in crit],
                "msa_depth_used": len(rows),
                "aligned_width": W,
            }
            top = sorted(range(L), key=lambda j: crit[j], reverse=True)[:8]
            print(f"  {gene} ({rfam}) len={L}  msa_rows={len(rows)}  width={W}  "
                  f"top-criticality nt: {[p + 1 for p in top]}")

    handle.remove()
    OUT.write_text(json.dumps({"model": "rnamsm", "layer": LAYER, "rnas": results}, indent=2))
    print("\n" + "=" * 80)
    print(f"✅ Saved {OUT}")
    print("   Now re-run 48 and 49 — they auto-ingest rnamsm for the 3-model consensus")
    print("   and the ClinVar enrichment (naive < 2ary-structure < MSA?).")
    print("=" * 80)


if __name__ == "__main__":
    main()
