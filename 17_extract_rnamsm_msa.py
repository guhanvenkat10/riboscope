"""
17_extract_rnamsm_msa.py — RIBOSCOPE: extract RNA-MSM activations under its
NATIVE MSA input, in a drop-in format for the existing SAE pipeline.

Why this exists
---------------
The geometry scan (16_msa_geometry_scan.py) showed that feeding RNA-MSM a real
Rfam SEED alignment (instead of a single sequence) CAUSALLY and PARTIALLY
de-collapses its representation — the strongest effect at layer 8
(var@32 94%->67%, fam_pcs90 ~10->38, eta up ~4x). That is a geometry PROXY for
SAE-viability. The decisive, reviewer-proof test is to actually run the SAE on
the de-collapsed (in-distribution) activations and check for monosemantic
family specialists. This script produces those activations.

It is a PURE INPUT-DISTRIBUTION SWAP relative to the single-sequence v3 path:
  * same model, same query sequences, same layer-6 (+ layer-8) hook,
  * same per-token convention ([CLS, residue_1..residue_L, EOS]),
  * same downstream preprocessing (z-score + position-center, via 18) and
    SAE recipe (08/09).
The ONLY thing that changes is that each query is now embedded inside its
family's MSA, so the model's row/column (axial) attention is in-distribution.

How it works (all MSA machinery reused VERBATIM from 16, which the geometry
scan already validated end-to-end on this exact data):
  1. Parse the FASTA query set (top-100 families, identical to 13/14/15/16).
  2. Parse Rfam.seed.gz keeping gaps; locate each query's aligned seed row.
  3. Build a [1, R, C] MSA tensor (query as row 0 + up to MSA_DEPTH-1 homologs).
  4. Bypass the 2D-only outer forward: model.embeddings -> model.encoder.
     Hooks on encoder.layer[6] and encoder.layer[8] capture [R, C, 1, D].
  5. Take ROW 0 (the query) and keep CLS + EOS + every NON-GAP query column,
     dropping only the alignment's gap columns. This yields exactly the same
     token set a single-sequence forward of the ungapped query would produce
     ([CLS, residue_1..residue_L, EOS]) — so 08's N-mask and 09's family
     tracking line up 1:1 with the matching FASTA written below.
  6. Save per-layer safetensors with keys "{name}__layer{N}" + a matching
     FASTA (sequences/rfam_msa_query.fasta) whose headers/sequences are the
     ungapped queries actually extracted, so set_model.py/08/09 use the right
     metadata for THIS token set (not the 30k single-seq FASTA).

Disk pre-check + post-save verify mirror 11_extract_rnamsm.py.

Run with
--------
    cd ~/projects/riboscope
    uv run python 17_extract_rnamsm_msa.py
    # SMOKE phase (SMOKE_N seqs) prints/asserts shapes first, then full run.

Output
------
    outputs/activations_rnamsm_msa_layer6.safetensors   ({name}__layer6)
    outputs/activations_rnamsm_msa_layer8.safetensors   ({name}__layer8)
    outputs/activations_rnamsm_msa_extract_meta.json
    sequences/rfam_msa_query.fasta   (metadata matched to the extracted tokens)
"""

from __future__ import annotations

import gc
import gzip
import json
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

try:
    import torch
    from multimolecule import RnaTokenizer, RnaMsmModel
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("   Needs: torch, multimolecule (RnaMsmModel), tqdm.")
    sys.exit(1)


# ============================ CONFIG ============================
MODEL_NAME = "multimolecule/rnamsm"
FALLBACK_MODELS = [
    "multimolecule/RNA-MSM",
    "multimolecule/RNAMsm",
    "multimolecule/rna-msm",
    "yikun-zhang/RNA-MSM",
]

FASTA_FILE = Path("sequences/rfam_30k.fasta")              # query set (same as 16)
RFAM_SEED = Path("data/Rfam.seed.gz")                      # gapped seed alignments
OUT_FASTA = Path("sequences/rfam_msa_query.fasta")         # metadata for THIS token set

# Layers to capture in a single forward (the encoder runs all layers regardless,
# so capturing two is free). L8 = most de-collapsed (primary, see 16); L6 = same
# layer as the other two models' SAEs (control).
LAYERS_TO_HOOK = [6, 8]

def out_path_for(layer: int) -> Path:
    return Path(f"outputs/activations_rnamsm_msa_layer{layer}.safetensors")

META_OUT = Path("outputs/activations_rnamsm_msa_extract_meta.json")

# Family / token selection — identical to 13/14/15/16 (apples-to-apples).
TOP_K_FAMILIES = 100

# MSA construction (identical to 16).
MSA_DEPTH = 32             # query row 0 + up to (MSA_DEPTH-1) homolog rows
RNG_SEED = 42             # reproducible homolog subsampling (matches 16)
SMOKE_N = 3              # shape-checked dry run before the full loop; 0 to skip
# ================================================================


# ---------- sequence + Rfam parsing (VERBATIM from 16) ----------
def parse_fasta_with_metadata(path: Path) -> dict:
    """Header: '>{name} | {rfam_id} | {rfam_name} | ...'. T->U applied. Identical
    parser to 08/09/13/14/15/16 so the query set + token indexing match exactly."""
    out: dict[str, dict] = {}
    name = rfam_id = rfam_name = None
    seq_buf: list[str] = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    out[name] = {"sequence": "".join(seq_buf),
                                 "rfam_id": rfam_id, "rfam_name": rfam_name}
                parts = [p.strip() for p in line[1:].split("|")]
                name = parts[0] if len(parts) > 0 else "unknown"
                rfam_id = parts[1] if len(parts) > 1 else None
                rfam_name = parts[2] if len(parts) > 2 else None
                seq_buf = []
            else:
                seq_buf.append(line.replace("T", "U").replace("t", "u"))
        if name is not None:
            out[name] = {"sequence": "".join(seq_buf),
                         "rfam_id": rfam_id, "rfam_name": rfam_name}
    return out


def norm_aligned(s: str) -> str:
    """Normalize an aligned Stockholm row, KEEPING gaps. upper, T->U, '-'->'.',
    non-ACGU/. -> N. Rfam uses '-' and '.' for gaps; multimolecule gap token = '.'."""
    s = s.upper().replace("T", "U").replace("-", ".")
    return "".join(c if c in "ACGU." else "N" for c in s)


def ungap(s: str) -> str:
    return s.replace(".", "")


def parse_rfam_seed_aligned(path: Path, keep_fams: set[str]) -> dict[str, dict]:
    """Parse Rfam.seed.gz, returning ONLY the requested families (gaps kept)."""
    fams: dict[str, dict] = {}
    with gzip.open(path, "rt", encoding="latin-1") as f:
        block: list[str] = []
        for line in f:
            block.append(line)
            if line.startswith("//"):
                fam_id, rows = _parse_block_aligned("".join(block))
                block = []
                if fam_id and fam_id in keep_fams and rows:
                    widths = {len(r) for _, r in rows}
                    if len(widths) != 1:
                        continue
                    ungap2aln: dict[str, str] = {}
                    for _sid, aln in rows:
                        ungap2aln.setdefault(ungap(aln), aln)
                    fams[fam_id] = {
                        "aligned_rows": [aln for _sid, aln in rows],
                        "ungap2aln": ungap2aln,
                        "width": next(iter(widths)),
                    }
    return fams


def _parse_block_aligned(text: str):
    """One Stockholm block -> (fam_id, [(seq_id, normalized_aligned_row), ...])."""
    seq_buffers: dict[str, str] = {}
    fam_id = None
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
                sid, aln = parts
                seq_buffers[sid] = seq_buffers.get(sid, "") + aln
    rows = [(sid, norm_aligned(aln)) for sid, aln in seq_buffers.items()]
    return fam_id, rows


# ---------- MSA tensor construction (VERBATIM from 16) ----------
def build_msa_rows(fam: dict, query_norm: str, depth: int, rng: random.Random):
    """Aligned MSA rows (query first) or None if the query's aligned row can't
    be located in this family's seed alignment."""
    aln_query = fam["ungap2aln"].get(query_norm)
    if aln_query is None:
        return None
    others = [r for r in fam["aligned_rows"] if r != aln_query]
    rng.shuffle(others)
    rows = [aln_query] + others[: max(depth - 1, 0)]
    return rows


def msa_to_tensors(tokenizer, rows: list[str], device):
    """Tokenize each aligned row (adds CLS/EOS); all rows share the alignment
    width so token lengths match. Returns input_ids [1, R, C], mask [1, R, C]."""
    enc = [tokenizer(r, return_tensors="pt")["input_ids"][0] for r in rows]
    lengths = {t.shape[0] for t in enc}
    if len(lengths) != 1:
        raise ValueError(f"ragged token lengths within an MSA: {sorted(lengths)}")
    ids = torch.stack(enc, dim=0).unsqueeze(0).to(device)      # [1, R, C]
    mask = torch.ones_like(ids)                                # [1, R, C]
    return ids, mask


def msa_forward(model, ids3d, mask3d):
    """Drive RNA-MSM with a real MSA by bypassing the 2D-only outer forward."""
    emb = model.embeddings(input_ids=ids3d, attention_mask=mask3d)   # [1, R, C, D]
    _ = model.encoder(emb, attention_mask=mask3d)                    # hooks capture


def residue_col_mask(aln_query: str, n_tokens: int) -> torch.Tensor:
    """Keep CLS(0), EOS(n-1), and every NON-GAP column of the query row; drop
    only gap columns. n_tokens = len(aln_query) + 2. The kept rows, in column
    order, are exactly [CLS, residue_1..residue_L, EOS] for the ungapped query —
    i.e. the SAME token tensor a single-sequence forward of ungap(aln_query)
    would produce. (N residues are kept here, and dropped later by 08/09's
    FASTA-driven N-mask, identical to the single-seq path.)"""
    m = torch.zeros(n_tokens, dtype=torch.bool)
    m[0] = True                 # CLS
    m[n_tokens - 1] = True      # EOS
    for col in range(1, n_tokens - 1):
        if aln_query[col - 1] != ".":     # residue (ACGU or N), not a gap
            m[col] = True
    return m


# ---------- hooks (VERBATIM from 16) ----------
def make_hook(captured: dict, layer_idx: int):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        captured[layer_idx] = output.detach().to(torch.float16).cpu()
    return hook


def find_encoder_layers(model):
    candidates = [
        "encoder.layer", "bert.encoder.layer", "roberta.encoder.layer",
        "rnamsm.encoder.layer", "msm.encoder.layer", "model.encoder.layer",
    ]
    for path in candidates:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            _ = len(obj)
            return obj, path
        except (AttributeError, TypeError):
            continue
    print("❌ Could not find encoder.layer. Top-level modules:")
    for name, _ in model.named_modules():
        if name.count(".") <= 2:
            print(f"   {name}")
    raise AttributeError("encoder.layer not found")


def estimate_output_bytes(d: dict) -> int:
    return sum(t.numel() * 2 for t in d.values())   # fp16 store -> 2 bytes/elem


def verify_saved_file(path: Path, expected: dict, sample_key: str) -> None:
    chk = load_file_local(str(path))
    if len(chk) != len(expected):
        print(f"❌ Verify failed for {path}: saved {len(expected)} but reload sees {len(chk)}.")
        sys.exit(1)
    if chk[sample_key].shape != expected[sample_key].shape:
        print(f"❌ Shape mismatch on {sample_key} in {path}.")
        sys.exit(1)
    if set(chk.keys()) != set(expected.keys()):
        print(f"❌ Key set changed during save of {path}.")
        sys.exit(1)
    print(f"      ✓ Verified {path.name}: {len(chk)} tensors, keys + shapes preserved.")


# safetensors I/O (imported lazily so the dependency error above is the only one)
from safetensors.torch import load_file as load_file_local, save_file as save_file_local


# ---------- main ----------
def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 92)
    print("RIBOSCOPE: RNA-MSM NATIVE-MSA activation extraction (drop-in for 08/09)")
    print("=" * 92)

    for p in (FASTA_FILE, RFAM_SEED):
        if not p.exists():
            print(f"❌ Required file not found: {p}")
            if p == RFAM_SEED:
                print("   Run 06_fetch_rfam_sequences.py first (it downloads Rfam.seed.gz).")
            sys.exit(1)

    for layer in LAYERS_TO_HOOK:
        op = out_path_for(layer)
        if op.exists():
            print(f"⚠ {op} already exists. Refusing to overwrite. Delete it to re-run.")
            sys.exit(1)
    out_path_for(LAYERS_TO_HOOK[0]).parent.mkdir(parents=True, exist_ok=True)
    OUT_FASTA.parent.mkdir(parents=True, exist_ok=True)

    # ---- family selection (identical to 16) ----
    print(f"[1/5] Parsing {FASTA_FILE}; selecting top-{TOP_K_FAMILIES} families...")
    seq_meta = parse_fasta_with_metadata(FASTA_FILE)
    fam_counts = Counter(m["rfam_id"] for m in seq_meta.values() if m["rfam_id"])
    ranked = sorted(fam_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    selected_families = [fid for fid, _ in ranked[:TOP_K_FAMILIES]]
    fam_to_int = {f: i for i, f in enumerate(selected_families)}
    n_classes = len(selected_families)
    selected_seqs = sorted(n for n, m in seq_meta.items() if m["rfam_id"] in fam_to_int)
    print(f"      Total families in FASTA:   {len(fam_counts)}")
    print(f"      Selected families (K):     {n_classes}")
    print(f"      Total selected sequences:  {len(selected_seqs)}")

    # ---- Rfam seed alignments (gaps kept) ----
    print(f"[2/5] Parsing {RFAM_SEED} (keeping gaps) for {n_classes} families...")
    fams = parse_rfam_seed_aligned(RFAM_SEED, set(selected_families))
    got = sum(1 for f in selected_families if f in fams)
    print(f"      Families with seed alignment found: {got}/{n_classes}")
    if got == 0:
        print("❌ No seed alignments matched — aborting.")
        sys.exit(1)

    # ---- load model + hook target layers ----
    print(f"[3/5] Loading RNA-MSM + hooking layers {LAYERS_TO_HOOK}...")
    tokenizer = model = None
    tried = []
    for candidate in [MODEL_NAME] + FALLBACK_MODELS:
        tried.append(candidate)
        try:
            tokenizer = RnaTokenizer.from_pretrained(candidate)
            model = RnaMsmModel.from_pretrained(candidate)
            print(f"      ✓ Loaded {candidate}")
            break
        except Exception as e:
            print(f"      ✗ {candidate}: {type(e).__name__}: {str(e)[:120]}")
    if model is None:
        print(f"❌ Could not load any RNA-MSM checkpoint. Tried: {tried}")
        sys.exit(1)

    model.eval().to(device)
    if not (hasattr(model, "embeddings") and hasattr(model, "encoder")):
        print("❌ model lacks .embeddings/.encoder — MSA bypass not applicable.")
        sys.exit(1)
    layers, path = find_encoder_layers(model)
    n_layers = len(layers)
    hidden_dim = getattr(model.config, "hidden_size", None)
    cfg_maxpos = getattr(model.config, "max_position_embeddings", 1024)
    if max(LAYERS_TO_HOOK) >= n_layers:
        print(f"❌ Requested layer {max(LAYERS_TO_HOOK)} but model has {n_layers} layers.")
        sys.exit(1)
    print(f"      Layers at model.{path}: n_layers={n_layers}, hidden_dim={hidden_dim}")
    print(f"      config.max_position_embeddings={cfg_maxpos}")

    captured: dict[int, torch.Tensor] = {}
    handles = [layers[li].register_forward_hook(make_hook(captured, li)) for li in LAYERS_TO_HOOK]

    # ---- SMOKE phase ----
    if SMOKE_N > 0:
        print(f"[4/5] SMOKE: verifying MSA extraction on {SMOKE_N} sequences...")
        rng = random.Random(RNG_SEED)
        n_checked = 0
        with torch.no_grad():
            for name in selected_seqs:
                fam_id = seq_meta[name]["rfam_id"]
                if fam_id not in fams:
                    continue
                q = seq_meta[name]["sequence"].upper().replace("T", "U")
                q = "".join(c if c in "ACGU" else "N" for c in q)
                rows = build_msa_rows(fams[fam_id], q, MSA_DEPTH, rng)
                if rows is None:
                    continue
                W = len(rows[0])
                if W + 2 > cfg_maxpos or len(rows) > cfg_maxpos:
                    continue
                ids, mask = msa_to_tensors(tokenizer, rows, device)
                captured.clear()
                msa_forward(model, ids, mask)
                C = ids.shape[2]
                m = residue_col_mask(rows[0], C)
                for li in LAYERS_TO_HOOK:
                    cap = captured[li]
                    assert cap.ndim == 4 and cap.shape[1] == C and cap.shape[2] == 1, \
                        f"L{li} unexpected capture shape {tuple(cap.shape)}"
                    qrow = cap[0].squeeze(1)               # [C, D]
                    kept = qrow[m]                          # [L+2, D]
                    assert kept.shape[0] == len(q) + 2, \
                        f"kept {kept.shape[0]} != len(q)+2 ({len(q)+2})"
                    assert int(kept.shape[1]) == hidden_dim
                print(f"      seq={name} fam={fam_id}  C={C}  kept_tokens={int(m.sum())} "
                      f"(== len(q)+2 = {len(q)+2})  layers ok={LAYERS_TO_HOOK}")
                n_checked += 1
                if n_checked >= SMOKE_N:
                    break
        if n_checked == 0:
            print("❌ SMOKE: could not build a single MSA — check Rfam parsing/matching.")
            sys.exit(1)
        print(f"      ✓ SMOKE passed ({n_checked} seqs). Proceeding to full extraction.")

    # ---- full extraction ----
    print(f"[5/5] Extracting row-0 tokens over {len(selected_seqs)} queries (depth<= {MSA_DEPTH})...")
    out_by_layer: dict[int, dict[str, torch.Tensor]] = {li: {} for li in LAYERS_TO_HOOK}
    fasta_entries: list[tuple[str, str, str, str]] = []   # (name, rfam_id, rfam_name, q)
    n_used = n_skip_nofam = n_skip_nomatch = n_skip_toolong = n_skip_oom = n_skip_err = 0
    rng = random.Random(RNG_SEED)
    start = time.perf_counter()

    with torch.no_grad():
        for name in tqdm(selected_seqs, desc="msa-extract", unit="seq"):
            fam_id = seq_meta[name]["rfam_id"]
            fam = fams.get(fam_id)
            if fam is None:
                n_skip_nofam += 1
                continue
            q = seq_meta[name]["sequence"].upper().replace("T", "U")
            q = "".join(c if c in "ACGU" else "N" for c in q)
            rows = build_msa_rows(fam, q, MSA_DEPTH, rng)
            if rows is None:
                n_skip_nomatch += 1
                continue
            W = len(rows[0])
            if W + 2 > cfg_maxpos or len(rows) > cfg_maxpos:
                n_skip_toolong += 1
                continue

            # try at full depth, halve on OOM, then skip (identical policy to 16)
            cur_rows = rows
            ok = False
            for _attempt in range(2):
                try:
                    ids, mask = msa_to_tensors(tokenizer, cur_rows, device)
                    captured.clear()
                    msa_forward(model, ids, mask)
                    ok = True
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if len(cur_rows) > 2:
                        cur_rows = cur_rows[: max(2, len(cur_rows) // 2)]
                    else:
                        break
                except Exception:
                    break
            if not ok:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                n_skip_oom += 1
                continue

            try:
                C = ids.shape[2]
                m = residue_col_mask(cur_rows[0], C)
                if int(m.sum()) != len(q) + 2:
                    # Defensive: should be exact by construction. Skip if not.
                    n_skip_err += 1
                    continue
                for li in LAYERS_TO_HOOK:
                    cap = captured[li]                 # [R, C, 1, D]
                    qrow = cap[0].squeeze(1)           # [C, D]
                    out_by_layer[li][f"{name}__layer{li}"] = qrow[m].clone()  # [L+2, D] fp16
                meta = seq_meta[name]
                fasta_entries.append((name, meta["rfam_id"] or "", meta["rfam_name"] or "", q))
                n_used += 1
            except Exception:
                n_skip_err += 1
                continue

    for h in handles:
        h.remove()
    elapsed = time.perf_counter() - start
    print(f"      Forward pass done in {elapsed:.1f}s. used={n_used}  "
          f"skipped: nofam={n_skip_nofam} nomatch={n_skip_nomatch} "
          f"toolong={n_skip_toolong} oom={n_skip_oom} err={n_skip_err}")
    if n_used == 0:
        print("❌ No sequences produced activations — aborting before save.")
        sys.exit(1)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- disk pre-check (sum across both layer files) ----
    total_est = sum(estimate_output_bytes(out_by_layer[li]) for li in LAYERS_TO_HOOK)
    _, _, free = shutil.disk_usage(out_path_for(LAYERS_TO_HOOK[0]).parent)
    print(f"      Disk check: need ~{total_est / 1e9:.2f} GB, free {free / 1e9:.2f} GB")
    if free < total_est * 1.5:
        print(f"❌ Insufficient disk space (want {total_est * 1.5 / 1e9:.2f} GB headroom).")
        sys.exit(1)

    # ---- save per-layer + verify ----
    total_tokens = 0
    for li in LAYERS_TO_HOOK:
        d_out = out_by_layer[li]
        op = out_path_for(li)
        print(f"      Saving {len(d_out)} tensors → {op}")
        save_file_local(d_out, str(op))
        sample_key = next(iter(d_out))
        verify_saved_file(op, d_out, sample_key)
        if li == LAYERS_TO_HOOK[0]:
            total_tokens = sum(int(t.shape[0]) for t in d_out.values())

    # ---- write matching FASTA (metadata for THIS token set) ----
    print(f"      Writing matched FASTA → {OUT_FASTA}")
    with open(OUT_FASTA, "w") as f:
        for name, rfam_id, rfam_name, q in fasta_entries:
            f.write(f">{name} | {rfam_id} | {rfam_name}\n")
            f.write(q + "\n")

    with open(META_OUT, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "layers": LAYERS_TO_HOOK,
            "hidden_dim": hidden_dim,
            "msa_depth": MSA_DEPTH,
            "rng_seed": RNG_SEED,
            "top_k_families": TOP_K_FAMILIES,
            "n_classes": n_classes,
            "max_position_embeddings": cfg_maxpos,
            "n_used": n_used,
            "n_tokens_per_layer": total_tokens,
            "counts": {
                "n_skip_nofam": n_skip_nofam, "n_skip_nomatch": n_skip_nomatch,
                "n_skip_toolong": n_skip_toolong, "n_skip_oom": n_skip_oom,
                "n_skip_err": n_skip_err,
            },
            "out_files": [str(out_path_for(li)) for li in LAYERS_TO_HOOK],
            "out_fasta": str(OUT_FASTA),
            "note": (
                "Drop-in for 08/09. Per-seq tensor = [CLS, residue_1..residue_L, "
                "EOS] of the query row of its Rfam SEED MSA (gap columns dropped). "
                "Token set identical to a single-seq forward of the ungapped query, "
                "so 08 N-mask + 09 family-tracking work unchanged when FASTA_FILE "
                "points at rfam_msa_query.fasta. The ONLY change vs single-seq v3 is "
                "in-distribution MSA input. Next: 18_prep_msa_acts.py, then set_model."
            ),
        }, f, indent=2)

    print()
    print("=" * 92)
    print("✅ MSA extraction complete.")
    print("=" * 92)
    for li in LAYERS_TO_HOOK:
        op = out_path_for(li)
        print(f"   L{li}: {op} ({op.stat().st_size / 1e9:.2f} GB)")
    print(f"   FASTA: {OUT_FASTA}  ({len(fasta_entries)} sequences)")
    print(f"   Tokens per layer: {total_tokens:,}")
    print()
    print("   Next steps (primary = layer 8, the most de-collapsed layer):")
    print("     uv run python 18_prep_msa_acts.py 8       # normalize + position-center")
    print("     uv run python set_model.py rnamsm_msa     # point 08/09 at L8 MSA acts")
    print("     uv run python 08_train_sae_big.py         # retrain (~2 hr)")
    print("     uv run python 09_inspect_features_big.py  # inspect specialists (~10 min)")
    print("   (Optional same-layer control: 18_prep_msa_acts.py 6; set_model.py rnamsm_msa_l6)")


if __name__ == "__main__":
    main()
