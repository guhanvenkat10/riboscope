"""
61_rnamsm_pll_localization.py — RIBOSCOPE: the alignment-AWARE RNA-MSM readout.

Why
---
On RNU4-2, RNA-MSM read in its native MSA is *anti*-correlated with the ClinVar
pathogenic set (embedding-sensitivity AUC 0.26; see 50). The paper argues this is
a READOUT artifact, not a failure of the model's knowledge: mutating the query
base at a conserved alignment column perturbs the representation LESS because the
column consensus holds it, so embedding-shift runs inversely to conservation.

This script implements the fair, alignment-aware readout the Discussion flags as
future work: a masked-language pseudo-likelihood (masked-marginal) constraint
score. For each query residue we mask it in the native MSA and read the model's
predicted distribution over {A,C,G,U} for the query row; the score is how strongly
the model prefers wild-type over the alternatives,

    score_i = mean_{b != wt}  [ log p(wt | context) - log p(b | context) ].

High score = the model is confident the position must be wild-type = a constrained
column = expected to be pathogenic-enriched. This tracks constraint POSITIVELY, so
it should remove the 0.26 inversion. We compute BOTH readouts on the SAME MSA
(embedding-shift, for an apples-to-apples contrast, and the PLL score) and report
AUC vs the same live-fetched ClinVar pathogenic positions used everywhere else.

Method reuses the validated MSA machinery from 50 (Rfam SEED + MAFFT --add, no
--keeplength, so query numbering stays 1:1 with RefSeq/ClinVar coordinates) and
the ClinVar/efetch machinery from 51. Self-contained: no files needed except
data/Rfam.seed.gz (+ MAFFT + network for NCBI).

Run with
--------
    cd ~/projects/riboscope
    uv run python 61_rnamsm_pll_localization.py            # RNU4-2 + RNU4ATAC
    uv run python 61_rnamsm_pll_localization.py RNU4-2     # one gene
    uv run python 61_rnamsm_pll_localization.py --probe    # 1 forward, print shapes, exit

Inputs : data/Rfam.seed.gz  (+ NCBI efetch/eutils over the network)
Output : outputs/rnamsm_pll_localization.json
"""

from __future__ import annotations

import gzip
import json
import math
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

try:
    import torch
    import multimolecule
    from multimolecule import RnaTokenizer, RnaMsmModel
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

# The masked-LM head class name has varied across multimolecule versions; resolve
# it tolerantly so a rename is a clear message, not an import crash.
MLM_CLS = None
for _cls_name in ("RnaMsmForMaskedLM", "RnaMsmForPreTraining", "RnaMsmForNucleotidePrediction"):
    MLM_CLS = getattr(multimolecule, _cls_name, None)
    if MLM_CLS is not None:
        break
if MLM_CLS is None:
    print("❌ Could not find an RNA-MSM masked-LM class in multimolecule.")
    print(f"   Tried RnaMsmForMaskedLM / RnaMsmForPreTraining. Available RnaMsm* classes: "
          f"{[n for n in dir(multimolecule) if n.startswith('RnaMsm')]}")
    sys.exit(1)

# ============================ CONFIG ============================
RFAM_SEED = Path("data/Rfam.seed.gz")
OUT = Path("outputs/rnamsm_pll_localization.json")
LAYER = 8                      # headline RNA-MSM layer (for the embedding contrast)
MSA_DEPTH = 32
RNG_SEED = 42
BASES = ("A", "C", "G", "U")

# gene -> (RefSeq accession for sequence+ClinVar, Rfam family for the native MSA)
GENES = {
    "RNU4-2":   ("NR_003137.2", "RF00015"),   # U4  — ReNU  (known: embed AUC 0.26)
    "RNU4ATAC": ("NR_023343.1", "RF00618"),   # U4atac — MOPD1 (replication)
}
MODEL_NAME = "multimolecule/rnamsm"
FALLBACK_MODELS = ["multimolecule/RNA-MSM", "yikun-zhang/RNA-MSM"]

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
from entrez_config import get_entrez_email
TOOL, EMAIL = "riboscope", get_entrez_email()
TIMEOUT, N_RETRIES = 30, 3
# ================================================================


# ---------- NCBI (from 51) ----------
def _get(url: str) -> str:
    last = None
    for attempt in range(1, N_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": f"{TOOL} (mailto:{EMAIL})"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * attempt)
    print(f"   ⚠ fetch failed: {last}")
    return ""


def efetch_seq(accession: str) -> str:
    txt = _get(f"{EUTILS}/efetch.fcgi?db=nuccore&id={accession}&rettype=fasta&retmode=text"
               f"&tool={TOOL}&email={EMAIL}")
    if not txt.startswith(">"):
        return ""
    seq = "".join(l.strip() for l in txt.splitlines()[1:] if l and not l.startswith(">"))
    return "".join(c if c in "ACGU" else "N" for c in seq.upper().replace("T", "U"))


def fetch_clinvar_positions(gene: str, acc_prefix: str) -> list[int]:
    term = urllib.parse.quote(f'{gene}[gene] AND ("pathogenic"[Germline classification] '
                              f'OR "likely pathogenic"[Germline classification])')
    es = _get(f"{EUTILS}/esearch.fcgi?db=clinvar&term={term}&retmax=500&retmode=json"
              f"&tool={TOOL}&email={EMAIL}")
    try:
        ids = json.loads(es)["esearchresult"]["idlist"]
    except Exception:  # noqa: BLE001
        ids = []
    pos: set[int] = set()
    for k in range(0, len(ids), 100):
        time.sleep(0.34)
        summ = _get(f"{EUTILS}/esummary.fcgi?db=clinvar&id={','.join(ids[k:k+100])}"
                    f"&retmode=json&tool={TOOL}&email={EMAIL}")
        try:
            res = json.loads(summ)["result"]
        except Exception:  # noqa: BLE001
            continue
        for uid in res.get("uids", []):
            rec = res.get(uid, {})
            title = rec.get("title", "") or ""
            cls = (rec.get("germline_classification", {}) or {}).get("description", "") \
                or (rec.get("clinical_significance", {}) or {}).get("description", "")
            if "pathogenic" not in cls.lower() or acc_prefix not in title:
                continue
            for m in re.finditer(r"n\.(\d+)", title):
                pos.add(int(m.group(1)))
    return sorted(pos)


# ---------- MSA machinery (from 50) ----------
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
                    if len({len(r) for r in rows}) == 1:
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


def mafft_add_full(seed_aln_rows: list[str], query_seq: str) -> dict:
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


def residue_cols(aln_query: str) -> list[int]:
    return [c for c, ch in enumerate(aln_query) if ch != "."]


def kept_mask(aln_query: str, C: int) -> torch.Tensor:
    m = torch.zeros(C, dtype=torch.bool)
    m[0] = True
    m[C - 1] = True
    for c, ch in enumerate(aln_query):
        if ch != ".":
            m[c + 1] = True
    return m


def find_encoder_layers(model):
    for path in ("encoder.layer", "bert.encoder.layer", "rnamsm.encoder.layer",
                 "msm.encoder.layer", "model.encoder.layer", "rnamsm.encoder.layers"):
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


def msa_ids(tokenizer, rows, device):
    """Tokenize an MSA (list of equal-width aligned rows, query=row 0) -> [1,R,C] ids."""
    enc = [tokenizer(r, return_tensors="pt")["input_ids"][0] for r in rows]
    if len({t.shape[0] for t in enc}) != 1:
        raise ValueError("ragged token lengths within MSA")
    ids = torch.stack(enc, 0).unsqueeze(0).to(device)      # [1,R,C]
    return ids


def locate_qrow_vec(t: torch.Tensor, R: int, C: int, want_last: int | None):
    """From an MLM/hidden tensor of unknown singleton-padded layout, return a view
    indexed as [row, col, feat]. Squeezes size-1 dims, then finds the axis == R
    (rows) and the axis == C (cols); the remaining axis is features/vocab."""
    t = t.squeeze()                                         # drop batch/size-1 dims
    if t.dim() == 2:                                        # [C, V] single-row style
        t = t.unsqueeze(0).expand(R, -1, -1) if t.shape[0] == C else t.unsqueeze(0)
    # now expect 3D; identify dims
    dims = list(t.shape)
    # find col axis (== C) and row axis (== R); prefer exact matches
    col_ax = next((i for i, d in enumerate(dims) if d == C), None)
    row_ax = next((i for i, d in enumerate(dims) if d == R and i != col_ax), None)
    if col_ax is None:
        raise ValueError(f"could not find column axis (C={C}) in logits shape {dims}")
    if row_ax is None:                                     # rows collapsed -> single row
        # insert a length-1 row axis at front
        t = t.unsqueeze(0); dims = list(t.shape); row_ax, col_ax = 0, col_ax + 1
    feat_ax = ({0, 1, 2} - {row_ax, col_ax}).pop()
    t = t.permute(row_ax, col_ax, feat_ax).contiguous()    # [R, C, F]
    return t


def load_base(device):
    for cand in [MODEL_NAME] + FALLBACK_MODELS:
        try:
            tok = RnaTokenizer.from_pretrained(cand)
            model = RnaMsmModel.from_pretrained(cand).eval().to(device)
            print(f"   ✓ loaded base {cand}")
            return tok, model, cand
        except Exception as e:  # noqa: BLE001
            print(f"   ✗ base {cand}: {type(e).__name__}: {str(e)[:80]}")
    raise SystemExit("❌ could not load RNA-MSM base")


def load_mlm(cand, device):
    try:
        model = MLM_CLS.from_pretrained(cand).eval().to(device)
        print(f"   ✓ loaded MLM head {cand} ({MLM_CLS.__name__})")
        return model
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"❌ could not load {MLM_CLS.__name__} ({cand}): {e}")


# ---------- stats (from 51) ----------
def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def auc_enrichment(scores, pos1):
    L = len(scores)
    P = sorted({p - 1 for p in pos1 if 1 <= p <= L})
    n1, n2 = len(P), L - len(P)
    if n1 == 0 or n2 == 0:
        return None
    ranks = rankdata(scores)
    R1 = sum(ranks[i] for i in P)
    U1 = R1 - n1 * (n1 + 1) / 2
    auc = U1 / (n1 * n2)
    mu, sd = n1 * n2 / 2, math.sqrt(n1 * n2 * (L + 1) / 12)
    z = (U1 - mu) / sd if sd > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return {"auc": round(auc, 3), "p_value": float(f"{p:.2e}"), "n_path": n1, "L": L}


def build_msa(gene, rfam, device):
    """Return (query_seq, rows, cols, aln_q) for `gene`, or None on failure."""
    acc = GENES[gene][0]
    q = efetch_seq(acc)
    if not q:
        print(f"  {gene}: sequence fetch failed ({acc}) — skipping.")
        return None
    L = len(q)
    seed_rows = parse_seed_rows(RFAM_SEED, rfam)
    if not seed_rows:
        print(f"  {gene} ({rfam}): no SEED alignment found in {RFAM_SEED} — skipping.")
        return None
    rng = random.Random(RNG_SEED)
    ctx = seed_rows[:]
    rng.shuffle(ctx)
    ctx = ctx[: MSA_DEPTH - 1]
    try:
        aligned = mafft_add_full(ctx, q)
    except Exception as e:  # noqa: BLE001
        print(f"  {gene}: MAFFT error: {e}")
        return None
    aln_q = norm_aligned(aligned.get("QUERY", ""))
    if ungap(aln_q) != q:
        print(f"  {gene}: MAFFT altered query residues (len {len(ungap(aln_q))} != {L}) — skipping.")
        return None
    rows = [aln_q] + [norm_aligned(aligned[f"SEED_{i}"]) for i in range(len(ctx))
                      if f"SEED_{i}" in aligned]
    cols = residue_cols(aln_q)
    assert len(cols) == L
    return q, rows, cols, aln_q


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    probe = "--probe" in sys.argv
    if shutil.which("mafft") is None:
        print("❌ MAFFT not found. Install: conda install -c bioconda mafft  (or: sudo apt install mafft)")
        sys.exit(1)
    if not RFAM_SEED.exists():
        print(f"❌ Missing {RFAM_SEED}")
        sys.exit(1)

    genes = argv or list(GENES)
    genes = [g for g in genes if g in GENES]
    if not genes:
        print(f"❌ No usable genes. Choose from: {list(GENES)}")
        sys.exit(1)

    print("=" * 84)
    print(f"RIBOSCOPE — RNA-MSM alignment-aware (masked-PLL) localization, layer {LAYER}")
    print(f"genes: {genes}")
    print("=" * 84)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, base, cand = load_base(device)
    mlm = load_mlm(cand, device)
    layers = find_encoder_layers(base)
    captured: dict = {}
    handle = layers[LAYER].register_forward_hook(make_hook(captured, LAYER))
    # RnaMsmForMaskedLM.forward can't take the 3D MSA tensor, so we run the base
    # embeddings+encoder (as in 50) and apply the LM head to the final layer's
    # query-row hidden states ourselves.
    final_idx = len(layers) - 1
    handle2 = layers[final_idx].register_forward_hook(make_hook(captured, "final"))
    lm_head = getattr(mlm, "lm_head", None)
    if lm_head is None:
        print("❌ RnaMsmForMaskedLM has no .lm_head attribute; cannot read logits.")
        sys.exit(1)

    # token ids for the four bases (for reading PLL over {A,C,G,U})
    base_ids = {}
    for b in BASES:
        tid = tok.convert_tokens_to_ids(b)
        base_ids[b] = tid
    mask_id = tok.mask_token_id
    if mask_id is None:
        print("❌ tokenizer has no mask token; cannot compute masked PLL.")
        sys.exit(1)
    print(f"   base token ids: {base_ids} | mask id: {mask_id}")

    results = {}
    with torch.no_grad():
        for gene in genes:
            rfam = GENES[gene][1]
            built = build_msa(gene, rfam, device)
            if built is None:
                continue
            q, rows, cols, aln_q = built
            L = len(q)
            ids = msa_ids(tok, rows, device)                # [1,R,C]
            R, C = ids.shape[1], ids.shape[2]

            # ---- one WT forward: layer-8 (embed contrast) + final layer (for lm_head) ----
            captured.clear()
            emb = base.embeddings(input_ids=ids, attention_mask=torch.ones_like(ids))
            _ = base.encoder(emb, attention_mask=torch.ones_like(ids))
            cap = captured[LAYER]                           # [R,C,1,D] (per 50)
            qrow_wt = cap[0].squeeze(1)                     # [C, D]
            km = kept_mask(aln_q, C)
            wt_res = qrow_wt[km][1:L + 1].to(device)        # [L, D]
            qfinal_wt = captured["final"][0].squeeze(1).to(device)      # [C, D]
            # multimolecule heads take a tuple/ModelOutput, not a raw tensor -> wrap it
            _wt_out = lm_head((qfinal_wt.unsqueeze(0),))
            wt_logits = (_wt_out.logits if hasattr(_wt_out, "logits") else _wt_out).squeeze(0)  # [C, V]
            if probe:
                print(f"\n[PROBE] {gene}: ids {tuple(ids.shape)}  hidden8 {tuple(cap.shape)}  "
                      f"final {tuple(captured['final'].shape)}  logits {tuple(wt_logits.shape)}  "
                      f"R={R} C={C} L={L}")
                print("        (share this line if the run errors so shapes can be confirmed)")
                handle.remove(); handle2.remove()
                return

            crit_pll = [0.0] * L
            crit_embed = [0.0] * L
            V = wt_logits.shape[-1]
            for i in tqdm(range(L), desc=f"{gene} PLL", unit="pos", leave=False):
                ci = cols[i]
                tok_pos = ci + 1                            # +1 for CLS
                wt_b = aln_q[ci]
                alts = [b for b in BASES if b != wt_b]

                # --- masked-marginal PLL: mask query base at this column ---
                mids = ids.clone()
                mids[0, 0, tok_pos] = mask_id               # row 0 = query
                captured.clear()
                embm = base.embeddings(input_ids=mids, attention_mask=torch.ones_like(mids))
                _ = base.encoder(embm, attention_mask=torch.ones_like(mids))
                qfinal_m = captured["final"][0].squeeze(1).to(device)      # [C, D]
                _m_out = lm_head((qfinal_m.unsqueeze(0),))
                mrow_logits = (_m_out.logits if hasattr(_m_out, "logits") else _m_out).squeeze(0)  # [C, V]
                qlog = mrow_logits[tok_pos]                 # [V]
                logp = torch.log_softmax(qlog.float(), dim=-1)
                lp_wt = float(logp[base_ids[wt_b]])
                crit_pll[i] = sum(lp_wt - float(logp[base_ids[b]]) for b in alts) / len(alts)

                # --- embedding-shift (the confounded readout), same MSA ---
                acc = 0.0
                for b in alts:
                    mq = aln_q[:ci] + b + aln_q[ci + 1:]
                    mrows = [mq] + rows[1:]
                    ids_b = msa_ids(tok, mrows, device)
                    captured.clear()
                    emb_b = base.embeddings(input_ids=ids_b, attention_mask=torch.ones_like(ids_b))
                    _ = base.encoder(emb_b, attention_mask=torch.ones_like(ids_b))
                    capb = captured[LAYER][0].squeeze(1)
                    kmb = kept_mask(mq, ids_b.shape[2])
                    em = capb[kmb][1:L + 1].to(device)
                    if em.shape == wt_res.shape:
                        acc += float((em - wt_res).norm(dim=1).mean().item())
                crit_embed[i] = acc / len(alts)

            cv = fetch_clinvar_positions(gene, GENES[gene][0].split(".")[0])
            auc_pll = auc_enrichment(crit_pll, cv)
            auc_embed = auc_enrichment(crit_embed, cv)
            mxp = max(crit_pll) or 1.0
            mnp = min(crit_pll)
            mxe = max(crit_embed) or 1.0
            results[gene] = {
                "accession": GENES[gene][0], "rfam": rfam, "L": L,
                "msa_rows": len(rows), "aligned_width": C - 2,
                "n_clinvar": len(cv), "clinvar_positions": cv,
                "auc_pll": auc_pll, "auc_embed": auc_embed,
                "criticality_pll": [round((c - mnp) / (mxp - mnp) if mxp > mnp else 0.0, 4)
                                    for c in crit_pll],
                "criticality_embed": [round(c / mxe, 4) for c in crit_embed],
            }
            ap = auc_pll["auc"] if auc_pll else float("nan")
            pp = auc_pll["p_value"] if auc_pll else float("nan")
            ae = auc_embed["auc"] if auc_embed else float("nan")
            print(f"\n  {gene} ({rfam}) L={L} rows={len(rows)} ClinVar n={len(cv)}")
            print(f"     embedding-shift AUC = {ae:.3f}   (the confounded readout)")
            print(f"     masked-PLL      AUC = {ap:.3f}  (p={pp:.1e})  ← alignment-aware")

    handle.remove()
    handle2.remove()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"model": "rnamsm", "layer": LAYER,
                               "readout": "masked_pseudo_likelihood_marginal",
                               "genes": results}, indent=2))
    print("\n" + "=" * 84)
    print(f"✅ Saved {OUT}")
    print("   Interpretation: if masked-PLL AUC > 0.5 while embedding-shift < 0.5,")
    print("   the RNA-MSM 'inversion' is a readout artifact — the model DID know.")
    print("=" * 84)


if __name__ == "__main__":
    main()
