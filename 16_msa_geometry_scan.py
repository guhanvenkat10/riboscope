"""
16_msa_geometry_scan.py — RIBOSCOPE: does RNA-MSM's rank-collapse survive when the
model is fed its NATIVE MSA input (instead of a single sequence)?

Why this exists
---------------
13/14/15 established, for RNA-MSM run at msa_depth=1 (one sequence at a time):
  * family identity IS present and linearly decodable (probe 13: 0.85 acc, 100-way)
  * family signal is NOT buried — it sits fully inside the top-32 PCs (14)
  * BUT every layer is in near rank-32 COLLAPSE (15: var@32 = 92.9-96.0% across all
    10 layers, vs RNA-FM 43% / ErnieRNA 27%), with the family code crushed into
    ~3-11 entangled dims (fam_pcs90 3-11 vs 117 / 232).
You cannot carve 100 monosemantic one-vs-rest SAE specialists out of ~10 entangled
dimensions, which is exactly why the RNA-MSM SAE failed where RNA-FM/ErnieRNA
succeeded. The per-layer scan proved the collapse is GLOBAL — so there is no
"better layer" to iterate on within single-sequence input.

RNA-MSM is an MSA Transformer (ESM-MSA-1b lineage). Running it on one sequence is
out-of-distribution; the model literally warns "Single sequence input detected,
RNA-MSM works best with MSA inputs." The decisive, higher-ceiling question is:

    If we give RNA-MSM a real MSA (the family's Rfam SEED alignment), does the
    representation DE-COLLAPSE — i.e. does effective rank rise and the family code
    spread across many dimensions (toward RNA-FM-like geometry)?

This is a controlled causal experiment: SAME model, SAME families, SAME geometry
metric; the ONLY changed variable is the input distribution (1 seq -> native MSA).
  * If geometry de-collapses  -> ITERATE: extract that layer with MSA input and
    train an SAE there (a third SAE success, and a clean "interpretability is
    conditional on in-distribution input" result).
  * If geometry still collapses -> LOCK RNA-MSM as the mechanistically-explained
    cross-model NEGATIVE: the collapse is intrinsic to the trained weights, not
    just the single-sequence regime. (The cross-model thesis never required all
    three SAEs to succeed; a rigorously-diagnosed negative is publishable.)

Geometry FIRST (one forward pass), SAE only if geometry says go — this gates the
expensive retraining on a cheap, decisive measurement.

How the MSA is built (evidence/sources in code)
-----------------------------------------------
* Data already on disk: data/Rfam.seed.gz (downloaded by 06_fetch_rfam_sequences.py).
  Every Rfam family ships a curated SEED alignment — the exact in-distribution input
  RNA-MSM was trained on. We re-parse it but KEEP the gaps (06 threw them away).
* For each of our top-100-family query sequences (the SAME 2351 used by 13/14/15),
  we reconstruct its MSA: the query's own aligned seed row as ROW 0, plus up to
  MSA_DEPTH-1 homologous rows from the same family's seed alignment (all rows share
  the alignment width, so they stack into a column-aligned MSA).
* Gap handling: multimolecule's RnaTokenizer uses "." as the gap token; Rfam uses
  both "-" and "." for gaps, so we map "-" -> "." (verified in
  multimolecule/tokenisers/rna/ALPHABET.md and tokenization_rna.py doctest).

How the model is driven (evidence/sources in code)
--------------------------------------------------
RnaMsmModel.forward (modeling_rnamsm.py L114-178) hard-codes the SINGLE-sequence
path: L134 `batch_size, seq_length = input_shape` unpacks a 2-tuple, so a 3D MSA
tensor crashes there, and L147-157 unsqueezes 2D input to msa_depth=1. BUT the
real MSA machinery exists one level down:
  * RnaMsmEmbeddings.forward (L576-614) unpacks `_, num_alignments, seq_length`
    (L587) and adds per-row msa_embeddings (L608) — it WANTS 3D [B, R, C].
  * RnaMsmEncoder.forward (L671-699) permutes B x R x C x D -> R x C x B x D
    (L680) and runs axial ROW + COLUMN self-attention across the alignment.
So we bypass the 2D-only outer forward and call model.embeddings + model.encoder
directly with a 3D [1, R, C] tensor. Forward hooks on encoder.layer[i] fire exactly
as before; in the [R, C, B, D] frame `captured[i][0]` is ROW 0 = the query (the
SAME indexing 15 already uses), so the per-token extraction code is reused verbatim.
max_position_embeddings = 1024 (config) caps both width and depth; we guard on it.

Run with
--------
    cd ~/projects/riboscope
    uv run python 16_msa_geometry_scan.py
    # First runs a SMOKE phase (SMOKE_N seqs) printing tensor shapes; if those
    # asserts pass it proceeds automatically to the full scan. Set SMOKE_N=0 to skip.

Output
------
    outputs/msa_geometry_scan.json   (per-layer MSA geometry + single-seq comparison)
    console: per-layer SINGLE-SEQ vs MSA table + automated ITERATE/LOCK verdict
"""

from __future__ import annotations

import gc
import gzip
import json
import random
import sys
import time
from collections import Counter, defaultdict
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

FASTA_FILE = Path("sequences/rfam_30k.fasta")
RFAM_SEED = Path("data/Rfam.seed.gz")                  # written by 06_fetch_rfam_sequences.py
SINGLE_SEQ_JSON = Path("outputs/per_layer_rank_scan.json")  # from 15, for side-by-side
REPORT_OUT = Path("outputs/msa_geometry_scan.json")

# Same K-way problem + token filtering as 13/14/15 (apples-to-apples geometry).
TOP_K_FAMILIES = 100
SKIP_SPECIAL_TOKENS = True
SKIP_N_TOKENS = True

K_BUDGET = 32
RANKS = [1, 5, 10, 32, 64, 128, 256]

# MSA construction
MSA_DEPTH = 32              # query row 0 + up to (MSA_DEPTH-1) homolog rows
RNG_SEED = 42              # reproducible homolog subsampling
MAX_POS = 1024            # config.max_position_embeddings; width+2 and depth must be <= this
SMOKE_N = 3               # verbose shape-checked dry run before the full loop; 0 to skip

# Layer-6 reference values from 14 (single-seq), printed for context.
REF_L6 = {
    "rnafm":     {"eta_family": 0.1461, "var_top32": 0.4334, "fam_top32": 0.6832, "fam_pcs_90": 117},
    "erniarna":  {"eta_family": 0.0925, "var_top32": 0.2656, "fam_top32": 0.4274, "fam_pcs_90": 232},
}

# A layer becomes an SAE-VIABLE candidate (under MSA input) if it breaks the
# rank-32 collapse AND keeps family signal spread + present. Same thresholds as 15.
CAND_VAR_TOP32_MAX = 0.70
CAND_FAM_PCS_90_MIN = 40
CAND_ETA_MIN = 0.08
# ================================================================


# ---------- sequence + Rfam parsing ----------
def parse_fasta_with_metadata(path: Path) -> dict:
    """Header: '>{name} | {rfam_id} | {rfam_name} | {len} nt'. Identical parser to
    08/09/13/14/15 so the query set and token indexing match exactly. T->U applied."""
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
    """Normalize an aligned Stockholm row, KEEPING gaps. Mirrors 06's residue
    normalization (upper, T->U, non-ACGU -> N) but preserves gaps as '.'.
    Rfam uses '-' and '.' for gaps; multimolecule's gap token is '.', so '-'->'.'."""
    s = s.upper().replace("T", "U").replace("-", ".")
    return "".join(c if c in "ACGU." else "N" for c in s)


def ungap(s: str) -> str:
    return s.replace(".", "")


def parse_rfam_seed_aligned(path: Path, keep_fams: set[str]) -> dict[str, dict]:
    """Parse Rfam.seed.gz, returning ONLY the requested families. For each:
        rows        : list of normalized aligned rows (gaps kept), alignment width W
        ungap2aln   : {normalized_ungapped_row -> normalized_aligned_row} for query lookup
    Stockholm parsing identical to 06_fetch_rfam_sequences.py (so fam ids + the
    ungapped forms line up with the FASTA), except we DO NOT discard the gaps."""
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
                    # All rows in a Stockholm alignment share one width; if a family
                    # somehow has ragged rows (wrapped oddly), skip it defensively.
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


# ---------- MSA tensor construction ----------
def build_msa_rows(fam: dict, query_norm: str, depth: int, rng: random.Random):
    """Return aligned MSA rows (query first) or None if the query's aligned row
    can't be located in this family's seed alignment."""
    aln_query = fam["ungap2aln"].get(query_norm)
    if aln_query is None:
        return None
    others = [r for r in fam["aligned_rows"] if r != aln_query]
    rng.shuffle(others)
    rows = [aln_query] + others[: max(depth - 1, 0)]
    return rows


def msa_to_tensors(tokenizer, rows: list[str], device):
    """Tokenize each aligned row (adds CLS/EOS); all rows share the alignment width
    so token lengths match. Returns input_ids [1, R, C] and attention_mask [1, R, C]
    (all ones — gaps are real tokens in an MSA, not padding)."""
    enc = [tokenizer(r, return_tensors="pt")["input_ids"][0] for r in rows]
    lengths = {t.shape[0] for t in enc}
    if len(lengths) != 1:
        raise ValueError(f"ragged token lengths within an MSA: {sorted(lengths)}")
    ids = torch.stack(enc, dim=0).unsqueeze(0).to(device)      # [1, R, C]
    mask = torch.ones_like(ids)                                # [1, R, C]
    return ids, mask


def msa_forward(model, ids3d, mask3d):
    """Drive RNA-MSM with a real MSA by bypassing the 2D-only outer forward.
    Calls embeddings (expects [B,R,C]) then encoder (axial row/col attention).
    Forward hooks on encoder.layer[i] fire here, capturing [R, C, B, D]."""
    emb = model.embeddings(input_ids=ids3d, attention_mask=mask3d)   # [1, R, C, D]
    _ = model.encoder(emb, attention_mask=mask3d)                    # hooks capture per-layer


def aligned_col_mask(aln_query: str, n_tokens: int) -> torch.Tensor:
    """Keep ONLY the query row's real-nucleotide columns (A/C/G/U), dropping CLS,
    EOS, gap columns, and N — identical token set to the single-seq scan's
    real_token_mask for the same query. n_tokens = len(aln_query) + 2 (CLS/EOS)."""
    m = torch.zeros(n_tokens, dtype=torch.bool)
    for col in range(1, n_tokens - 1):           # skip CLS(0) and EOS(n-1)
        ch = aln_query[col - 1]
        keep = ch in "ACGU"
        if SKIP_N_TOKENS and ch == "N":
            keep = False
        if not SKIP_SPECIAL_TOKENS:
            keep = ch in "ACGU"
        m[col] = keep
    return m


# ---------- geometry (VERBATIM from 14/15) ----------
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


def scatter_geometry(X: torch.Tensor, y: torch.Tensor, n_classes: int, device):
    """Family-vs-total variance geometry. VERBATIM from 14/15 (proven)."""
    d = X.shape[1]
    X = X.to(device)
    N = X.shape[0]

    mu = X.mean(dim=0)
    Xc = X - mu
    S_T = (Xc.T @ Xc)
    trace_T = (Xc * Xc).sum()

    csum = torch.zeros(n_classes, d, device=device)
    csum.index_add_(0, y.to(device), X)
    cn = torch.zeros(n_classes, device=device)
    cn.index_add_(0, y.to(device), torch.ones(N, device=device))
    present = cn > 0
    mu_c = torch.zeros_like(csum)
    mu_c[present] = csum[present] / cn[present].unsqueeze(1)

    dev = (mu_c[present] - mu) * cn[present].sqrt().unsqueeze(1)
    S_B = dev.T @ dev
    trace_B = (dev * dev).sum()
    eta_family = float(trace_B / trace_T)

    S_T64 = S_T.double().cpu()
    dev64 = dev.double().cpu()
    trace_T = float(trace_T)
    trace_B = float(trace_B)

    evals, evecs = torch.linalg.eigh(S_T64)
    order = torch.argsort(evals, descending=True)
    lam = evals[order].clamp(min=0.0)
    V = evecs[:, order]

    lam_cum = torch.cumsum(lam, dim=0)
    var_frac_at = {r: float(lam_cum[min(r, d) - 1] / lam.sum()) for r in RANKS if r <= d}

    DV = dev64 @ V
    fam_pc = (DV * DV).sum(dim=0)
    fam_cum = torch.cumsum(fam_pc, dim=0)
    fam_total = float(fam_pc.sum())
    fam_frac_at = {r: float(fam_cum[min(r, d) - 1] / fam_total) for r in RANKS if r <= d}

    def count_for(curve_cum, total, thresh):
        hit = (curve_cum >= thresh * total).nonzero()
        return int(hit[0].item()) + 1 if len(hit) else d
    fam_pcs_90 = count_for(fam_cum, fam_total, 0.90)
    fam_pcs_50 = count_for(fam_cum, fam_total, 0.50)

    return {
        "d": int(d), "n_tokens": int(N),
        "eta_family": eta_family,
        "var_top32": var_frac_at.get(K_BUDGET),
        "fam_top32": fam_frac_at.get(K_BUDGET),
        "fam_pcs_50": fam_pcs_50, "fam_pcs_90": fam_pcs_90,
        "var_frac_cum": var_frac_at, "fam_frac_cum": fam_frac_at,
        "trace_T": trace_T, "trace_B": trace_B,
    }


# ---------- main ----------
def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 92)
    print("RIBOSCOPE: RNA-MSM NATIVE-MSA geometry scan (does MSA input de-collapse the rank?)")
    print("=" * 92)

    for p in (FASTA_FILE, RFAM_SEED):
        if not p.exists():
            print(f"❌ Required file not found: {p}")
            if p == RFAM_SEED:
                print("   Run 06_fetch_rfam_sequences.py first (it downloads Rfam.seed.gz).")
            sys.exit(1)

    # ---- family selection (identical to 13/14/15) ----
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

    # ---- Rfam seed alignments (gaps kept) for the selected families ----
    print(f"[2/5] Parsing {RFAM_SEED} (keeping gaps) for {n_classes} families...")
    fams = parse_rfam_seed_aligned(RFAM_SEED, set(selected_families))
    got = sum(1 for f in selected_families if f in fams)
    depths = [len(fams[f]["aligned_rows"]) for f in fams]
    print(f"      Families with seed alignment found: {got}/{n_classes}")
    if got == 0:
        print("❌ No seed alignments matched the selected families — aborting.")
        sys.exit(1)
    if depths:
        import statistics
        print(f"      Seed rows per family: min={min(depths)} median="
              f"{int(statistics.median(depths))} max={max(depths)} "
              f"(MSA_DEPTH cap = {MSA_DEPTH})")

    # ---- load model + hook EVERY layer ----
    print(f"[3/5] Loading RNA-MSM + hooking all encoder layers...")
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
        print("❌ model lacks .embeddings/.encoder — MSA bypass not applicable here.")
        sys.exit(1)
    layers, path = find_encoder_layers(model)
    n_layers = len(layers)
    hidden_dim = getattr(model.config, "hidden_size", None)
    cfg_maxpos = getattr(model.config, "max_position_embeddings", MAX_POS)
    layer_indices = list(range(n_layers))
    print(f"      Layers at model.{path}: n_layers={n_layers}, hidden_dim={hidden_dim}")
    print(f"      config.max_position_embeddings={cfg_maxpos}; embed_positions_msa="
          f"{getattr(model.config, 'embed_positions_msa', None)}")

    captured: dict[int, torch.Tensor] = {}
    handles = [layers[li].register_forward_hook(make_hook(captured, li)) for li in layer_indices]

    # ---- shape-checked smoke phase ----
    if SMOKE_N > 0:
        print(f"[4/5] SMOKE phase: verifying MSA forward on {SMOKE_N} sequences...")
        rng = random.Random(RNG_SEED)
        n_checked = 0
        with torch.no_grad():
            for name in selected_seqs:
                fam_id = seq_meta[name]["rfam_id"]
                if fam_id not in fams:
                    continue
                q = seq_meta[name]["sequence"].upper().replace("T", "U")
                q = "".join(c if c in "ACGU" else "N" for c in q)  # match FASTA form
                rows = build_msa_rows(fams[fam_id], q, MSA_DEPTH, rng)
                if rows is None:
                    continue
                W = len(rows[0])
                if W + 2 > cfg_maxpos or len(rows) > cfg_maxpos:
                    continue
                ids, mask = msa_to_tensors(tokenizer, rows, device)
                captured.clear()
                msa_forward(model, ids, mask)
                cap = captured[layer_indices[0]]
                R, C = ids.shape[1], ids.shape[2]
                print(f"      seq={name} fam={fam_id}  MSA rows={R} width(tok)={C}  "
                      f"captured[L0].shape={tuple(cap.shape)}")
                assert cap.ndim == 4, f"expected 4D [R,C,B,D], got {tuple(cap.shape)}"
                assert cap.shape[0] == R, f"dim0 should be R={R}, got {cap.shape[0]}"
                assert cap.shape[1] == C, f"dim1 should be C={C}, got {cap.shape[1]}"
                assert cap.shape[2] == 1, f"dim2 (batch) should be 1, got {cap.shape[2]}"
                assert cap.shape[3] == hidden_dim, f"dim3 should be D={hidden_dim}"
                qrow = cap[0]                      # row 0 = query -> [C, 1, D]
                if qrow.ndim == 3 and qrow.shape[1] == 1:
                    qrow = qrow.squeeze(1)         # -> [C, D]
                m = aligned_col_mask(rows[0], C)
                n_keep = int(m.sum())
                n_acgu = sum(1 for ch in rows[0] if ch in "ACGU")
                print(f"          query row -> [C,D]={tuple(qrow.shape)}  kept tokens="
                      f"{n_keep} (ACGU in query={n_acgu})")
                assert n_keep == n_acgu, "column mask must keep exactly the ACGU residues"
                n_checked += 1
                if n_checked >= SMOKE_N:
                    break
        if n_checked == 0:
            print("❌ SMOKE: could not build a single MSA — check Rfam parsing/matching.")
            sys.exit(1)
        print(f"      ✓ SMOKE passed ({n_checked} seqs). Proceeding to full scan.")

    # ---- full forward pass over all queries ----
    print(f"[5/5] MSA forward pass over {len(selected_seqs)} queries "
          f"(depth<= {MSA_DEPTH}); accumulating row-0 real-token acts for all {n_layers} layers...")
    per_layer_chunks: dict[int, list] = {li: [] for li in layer_indices}
    y_chunks: list = []
    n_used = n_skip_nofam = n_skip_nomatch = n_skip_toolong = n_skip_oom = n_skip_err = 0
    rng = random.Random(RNG_SEED)
    start = time.perf_counter()

    with torch.no_grad():
        for name in tqdm(selected_seqs, desc="msa-scan", unit="seq"):
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

            # try at full depth, then halved depth on OOM, then skip
            cur_rows = rows
            ok = False
            for attempt in range(2):
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
                mask_cols = aligned_col_mask(cur_rows[0], C)
                n_keep = int(mask_cols.sum())
                if n_keep == 0:
                    continue
                fam_int = fam_to_int[fam_id]
                for li in layer_indices:
                    cap = captured[li]                 # [R, C, 1, D]
                    qrow = cap[0]                       # row 0 = query -> [C, 1, D]
                    if qrow.ndim == 3 and qrow.shape[1] == 1:
                        qrow = qrow.squeeze(1)          # -> [C, D]
                    per_layer_chunks[li].append(qrow[mask_cols].clone())  # [n_keep, D] fp16
                y_chunks.append(torch.full((n_keep,), fam_int, dtype=torch.long))
                n_used += 1
            except Exception:
                n_skip_err += 1
                continue

    for h in handles:
        h.remove()
    elapsed = time.perf_counter() - start
    print(f"      Done in {elapsed:.1f}s. used={n_used}  "
          f"skipped: nofam={n_skip_nofam} nomatch={n_skip_nomatch} "
          f"toolong={n_skip_toolong} oom={n_skip_oom} err={n_skip_err}")
    if n_used == 0:
        print("❌ No sequences produced activations — aborting before geometry.")
        sys.exit(1)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    y = torch.cat(y_chunks)
    fams_present = len(torch.unique(y))
    print(f"      Tokens accumulated: {int(y.shape[0]):,}  across {fams_present} families")

    # ---- per-layer geometry ----
    print(f"      Computing scatter geometry per layer on device={device}...")
    results: dict[str, dict] = {}
    for li in layer_indices:
        X = torch.cat(per_layer_chunks[li]).float()
        per_layer_chunks[li] = None
        g = scatter_geometry(X, y, n_classes, device)
        del X
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        g["n_seqs_used"] = n_used
        results[str(li)] = g
        print(f"  layer {li:>2}:  eta_family={g['eta_family']*100:6.2f}%   "
              f"var@32={g['var_top32']*100:6.2f}%   fam@32={g['fam_top32']*100:6.2f}%   "
              f"fam_pcs90={g['fam_pcs_90']:>4} (50%: {g['fam_pcs_50']})")

    # ---- load single-seq scan for side-by-side ----
    single = None
    if SINGLE_SEQ_JSON.exists():
        try:
            single = json.loads(SINGLE_SEQ_JSON.read_text())["results_by_layer"]
        except Exception:
            single = None

    # ---- comparison table ----
    print("\n" + "=" * 92)
    print("SINGLE-SEQ (15)  vs  NATIVE-MSA (this)   — RNA-MSM, raw acts (z-score-invariant)")
    print("=" * 92)
    hdr = (f"{'layer':<6}"
           f"{'var@32 ss':>11}{'var@32 msa':>12}"
           f"{'fam_pcs90 ss':>14}{'fam_pcs90 msa':>15}"
           f"{'eta ss':>9}{'eta msa':>9}")
    print(hdr); print("-" * len(hdr))
    for li in layer_indices:
        r = results[str(li)]
        s = single.get(str(li)) if single else None
        ss_var = f"{s['var_top32']*100:.2f}" if s else "  n/a"
        ss_fp = f"{s['fam_pcs_90']}" if s else "n/a"
        ss_eta = f"{s['eta_family']*100:.1f}" if s else " n/a"
        flag = ""
        if (r["var_top32"] <= CAND_VAR_TOP32_MAX and r["fam_pcs_90"] >= CAND_FAM_PCS_90_MIN
                and r["eta_family"] >= CAND_ETA_MIN):
            flag = "  <- SAE-VIABLE"
        print(f"{('L'+str(li)):<6}"
              f"{ss_var:>11}{r['var_top32']*100:>12.2f}"
              f"{ss_fp:>14}{r['fam_pcs_90']:>15}"
              f"{ss_eta:>9}{r['eta_family']*100:>9.1f}{flag}")

    print("\nReference (single-seq layer-6, from 14):")
    for k, v in REF_L6.items():
        print(f"  {k:<10} var@32={v['var_top32']*100:6.2f}%  fam_pcs90={v['fam_pcs_90']:>3}  "
              f"eta={v['eta_family']*100:.1f}%   (SAE SUCCEEDED)")

    # ---- verdict ----
    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    cands = [li for li in layer_indices
             if results[str(li)]["var_top32"] <= CAND_VAR_TOP32_MAX
             and results[str(li)]["fam_pcs_90"] >= CAND_FAM_PCS_90_MIN
             and results[str(li)]["eta_family"] >= CAND_ETA_MIN]
    # did MSA materially de-collapse vs single-seq?
    decollapse = None
    if single:
        drops = []
        for li in layer_indices:
            s = single.get(str(li))
            if s:
                drops.append(s["var_top32"] - results[str(li)]["var_top32"])
        if drops:
            decollapse = max(drops)

    if cands:
        best = sorted(cands, key=lambda li: (-results[str(li)]["fam_pcs_90"],
                                             results[str(li)]["var_top32"]))[0]
        b = results[str(best)]
        print(f"  => ITERATE. With native MSA input, LAYER {best} breaks the rank-32 collapse")
        print(f"     (var@32={b['var_top32']*100:.1f}% vs single-seq ~94%) and spreads the")
        print(f"     family code (fam_pcs90={b['fam_pcs_90']}, eta={b['eta_family']*100:.1f}%, "
              f"fam@32={b['fam_top32']*100:.1f}%).")
        print(f"     NEXT: extract layer {best} with MSA input (adapt 11 -> msa_forward),")
        print(f"     normalize, set_model.py rnamsm, 08 retrain, 09 inspect, 12 cross-model.")
        print(f"     SCIENCE: 'RNA-MSM interpretability is conditional on in-distribution (MSA)")
        print(f"     input' — same model + same SAE recipe flips negative->positive by input alone.")
    else:
        msg_dc = (f"max var@32 drop vs single-seq = {decollapse*100:.1f} pts"
                  if decollapse is not None else "single-seq json not found for delta")
        print(f"  => LOCK as NEGATIVE/CONTRAST. Even with its native MSA, every layer stays")
        print(f"     near rank-32 (no SAE-viable layer; {msg_dc}). The collapse is INTRINSIC to")
        print(f"     the trained RNA-MSM weights, not merely the single-sequence regime.")
        print(f"     SCIENCE: family signal is linearly present (probe 0.85) yet geometrically")
        print(f"     unresolvable into monosemantic SAE features in BOTH input regimes — a")
        print(f"     rigorously-diagnosed cross-model negative (2/3 SAEs succeed; RNA-MSM is")
        print(f"     the mechanistic contrast). No further SAE retraining warranted.")

    # ---- persist ----
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUT, "w") as f:
        json.dump({
            "config": {
                "top_k_families": TOP_K_FAMILIES, "n_classes": n_classes,
                "skip_special_tokens": SKIP_SPECIAL_TOKENS, "skip_n_tokens": SKIP_N_TOKENS,
                "k_budget": K_BUDGET, "ranks": RANKS,
                "msa_depth": MSA_DEPTH, "rng_seed": RNG_SEED,
                "n_layers": n_layers, "hidden_dim": hidden_dim,
                "max_position_embeddings": cfg_maxpos,
                "space": "native-MSA input; raw acts; metrics invariant to global-scalar z-score",
            },
            "counts": {
                "n_used": n_used, "n_skip_nofam": n_skip_nofam,
                "n_skip_nomatch": n_skip_nomatch, "n_skip_toolong": n_skip_toolong,
                "n_skip_oom": n_skip_oom, "n_skip_err": n_skip_err,
                "families_present": int(fams_present),
            },
            "ref_layer6_singleseq_from_14": REF_L6,
            "single_seq_by_layer": single,
            "msa_results_by_layer": results,
            "verdict": {
                "sae_viable_layers": cands,
                "max_var_top32_drop_vs_singleseq": decollapse,
            },
        }, f, indent=2)
    print(f"\n✅ Done. Full metrics -> {REPORT_OUT}")


if __name__ == "__main__":
    main()
