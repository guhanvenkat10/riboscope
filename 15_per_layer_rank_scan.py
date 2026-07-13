"""
15_per_layer_rank_scan.py — RIBOSCOPE: is RNA-MSM's near-rank-32 collapse a
LAYER-6 artifact, or global to every layer (at msa_depth=1)?

Why this exists
---------------
The geometry diagnostic (14) OVERTURNED the burial hypothesis. RNA-MSM layer-6's
family signal is PRESENT (eta_family 12.5% ~ RNA-FM 14.6%, ErnieRNA 9.2%) and
FULLY inside the SAE's top-32 PC budget (fam_top32 97% vs RNA-FM 68%, ErnieRNA
43%). The real pathology is EXTREME ANISOTROPY / near rank-32 collapse: 96% of
ALL variance sits in 32 of 768 dims (var_top32 0.958 vs RNA-FM 0.433, ErnieRNA
0.266), and the family code is crushed into ~13 effective dims (fam_pcs_90 13 vs
117 / 232), co-aligned with the dominant generic-variance PCs. You cannot carve
100 monosemantic one-vs-rest family specialists out of ~13 entangled dimensions:
every top-PC SAE atom is polysemantic by construction, and a handful of high-
variance generic atoms saturate the reconstruction budget. Whitening was RULED
OUT (it smears the clean concentration: fam_top32 0.97 -> 0.69).

That leaves ONE decisive, cheap question before we either ITERATE or lock RNA-MSM
as a cross-model CONTRAST: is the rank collapse specific to layer 6, or does
EVERY layer collapse? If some layer L has RNA-FM-like spread (var_top32 ~ 0.4-0.5)
AND retains family signal (eta_family comparable, family reasonably accessible),
an SAE can plausibly work there -> re-extract + retrain on layer L. If ALL layers
are near rank-32, the collapse is intrinsic to running an MSA Transformer at
msa_depth=1 -> RNA-MSM becomes a rigorously-characterized NEGATIVE/CONTRAST case
(the cross-model thesis never required all three SAEs to succeed).

Method
------
ONE forward pass over the SAME top-100-family sequences used by 13/14, hooking
EVERY encoder layer at once (forward hooks are read-only; hooking all layers does
not change the outputs). For each layer we measure the SAME scale-invariant
scatter geometry as 14 (mean-centered native space):
  eta_family = trace(S_B)/trace(S_T)               family vs total variance
  var_top32  = % total variance  in top-32 PCs     anisotropy / effective rank
  fam_top32  = % family variance in top-32 PCs      accessibility to a top-k SAE
  fam_pcs_50/90 = # PCs for 50/90% of family var    family-code concentration

We run on RAW activations the model emits. normalize_rnamsm.py applies ONLY a
GLOBAL-SCALAR z-score ((x-mean)/std with single scalars) — an affine transform
that leaves every centered-scatter metric above invariant. So LAYER 6 here MUST
reproduce 14's rnamsm_pre row (var_top32~0.94, fam_top32~0.98, fam_pcs_90~9): a
built-in correctness check. RNA-FM / ErnieRNA layer-6 reference numbers are taken
from 14 (outputs/family_geometry_diagnostic.json) and printed for context; this
script is a single-model, single-pass RNA-MSM scan.

Run with
--------
    cd ~/projects/riboscope
    uv run python 15_per_layer_rank_scan.py

Output
------
    outputs/per_layer_rank_scan.json     (full per-layer metrics, auditable)
    console per-layer table + automated layer recommendation
"""

from __future__ import annotations

import gc
import json
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
# Model load — identical fallback chain to 11_extract_rnamsm.py.
MODEL_NAME = "multimolecule/rnamsm"
FALLBACK_MODELS = [
    "multimolecule/RNA-MSM",
    "multimolecule/RNAMsm",
    "multimolecule/rna-msm",
    "yikun-zhang/RNA-MSM",
]

FASTA_FILE = Path("sequences/rfam_30k.fasta")
REPORT_OUT = Path("outputs/per_layer_rank_scan.json")

# Identical K-way problem + token filtering as 13 (probe) and 14 (geometry).
TOP_K_FAMILIES = 100
SKIP_SPECIAL_TOKENS = True   # drop CLS (pos 0) and EOS (pos n-1)
SKIP_N_TOKENS = True         # drop N nucleotide positions

K_BUDGET = 32                # SAE's effective per-token budget (k active feats)
RANKS = [1, 5, 10, 32, 64, 128, 256]
EPS_FRAC = 1e-3              # whitening noise floor (carried for parity with 14)

# Layer-6 reference values from 14 (outputs/family_geometry_diagnostic.json),
# printed for context. RNA-MSM L6 (rnamsm_pre) is also the sanity-check target.
REF_L6 = {
    "rnafm":       {"eta_family": 0.1461, "var_top32": 0.4334, "fam_top32": 0.6832, "fam_pcs_90": 117},
    "erniarna":    {"eta_family": 0.0925, "var_top32": 0.2656, "fam_top32": 0.4274, "fam_pcs_90": 232},
    "rnamsm_L6":   {"eta_family": 0.2044, "var_top32": 0.9410, "fam_top32": 0.9804, "fam_pcs_90": 9},
}

# A layer is an ITERATE CANDIDATE if it is MUCH less anisotropic than L6 AND
# still carries spread, accessible family signal. Thresholds are deliberately
# generous (RNA-FM sits at var_top32 0.43 / fam_pcs_90 117); the printed table is
# the source of truth and the human makes the final call.
CAND_VAR_TOP32_MAX = 0.70    # must be well below L6's 0.94
CAND_FAM_PCS_90_MIN = 40     # family code spread across >=40 dims (vs L6's 9)
CAND_ETA_MIN = 0.08          # family signal at least as present as ErnieRNA
# ================================================================


def parse_fasta_with_metadata(path: Path) -> dict:
    """Parse the Rfam FASTA from 06_fetch_rfam_sequences.py. Header:
    '>{name} | {rfam_id} | {rfam_name} | {len} nt'. Same parser as 08/09/13/14
    so token indexing is identical. T->U applied."""
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


def real_token_mask(n: int, seq_upper: str) -> torch.Tensor:
    """True for tokens to keep. Mirrors 09/13/14: pos 0 = CLS, pos n-1 = EOS,
    pos 1..n-2 = nucleotides (nt_pos = pos-1). Drops special + N tokens."""
    m = torch.ones(n, dtype=torch.bool)
    if SKIP_SPECIAL_TOKENS:
        m[0] = False
        m[n - 1] = False
    if SKIP_N_TOKENS:
        for pos in range(1, n - 1):
            nt = pos - 1
            if nt < len(seq_upper) and seq_upper[nt] == "N":
                m[pos] = False
    return m


def make_hook(captured: dict, layer_idx: int):
    """Forward hook: stash this layer's output as fp16 on CPU. Read-only."""
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        captured[layer_idx] = output.detach().to(torch.float16).cpu()
    return hook


def find_encoder_layers(model):
    """Locate the encoder-layer ModuleList. Same candidate paths as 11."""
    candidates = [
        "encoder.layer",
        "bert.encoder.layer",
        "roberta.encoder.layer",
        "rnamsm.encoder.layer",
        "msm.encoder.layer",
        "model.encoder.layer",
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
    """Family-vs-total variance geometry for one (layer's) activation matrix.
    VERBATIM from 14_family_geometry_diagnostic.py (proven). S_T / S_B built in
    float32 on `device` after centering; eigendecomposition in float64 on CPU."""
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
        "d": int(d),
        "n_tokens": int(N),
        "eta_family": eta_family,
        "var_top32": var_frac_at.get(K_BUDGET),
        "fam_top32": fam_frac_at.get(K_BUDGET),
        "fam_pcs_50": fam_pcs_50,
        "fam_pcs_90": fam_pcs_90,
        "var_frac_cum": var_frac_at,
        "fam_frac_cum": fam_frac_at,
        "trace_T": trace_T,
        "trace_B": trace_B,
    }


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 88)
    print("RIBOSCOPE: RNA-MSM per-layer effective-rank scan (layer-6 collapse: local or global?)")
    print("=" * 88)

    if not FASTA_FILE.exists():
        print(f"❌ FASTA not found: {FASTA_FILE}")
        sys.exit(1)

    # ---- family selection (identical to 13/14) ----
    print(f"[1/4] Parsing {FASTA_FILE}; selecting top-{TOP_K_FAMILIES} families...")
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

    # ---- load model + hook EVERY layer ----
    print(f"[2/4] Loading RNA-MSM + hooking all encoder layers...")
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
    layers, path = find_encoder_layers(model)
    n_layers = len(layers)
    hidden_dim = getattr(model.config, "hidden_size", None)
    layer_indices = list(range(n_layers))
    print(f"      Layers at model.{path}: n_layers={n_layers}, hidden_dim={hidden_dim}")
    print(f"      Hooking layers: {layer_indices}")

    captured: dict[int, torch.Tensor] = {}
    handles = [layers[li].register_forward_hook(make_hook(captured, li)) for li in layer_indices]

    # ---- single forward pass; accumulate per-layer real-token activations ----
    print(f"[3/4] One forward pass over {len(selected_seqs)} sequences "
          f"(accumulating real-token acts for all {n_layers} layers)...")
    per_layer_chunks: dict[int, list] = {li: [] for li in layer_indices}
    y_chunks: list = []
    n_used = 0
    n_skipped = 0
    start = time.perf_counter()

    with torch.no_grad():
        for name in tqdm(selected_seqs, desc="scanning", unit="seq"):
            seq = seq_meta[name]["sequence"]
            seq_upper = seq.upper()
            try:
                inputs = tokenizer(seq, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                _ = model(**inputs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); n_skipped += 1; continue
            except Exception:
                n_skipped += 1; continue

            # Mask is layer-independent (depends only on the sequence).
            ref_act = captured[layer_indices[0]][0]
            if ref_act.ndim == 3 and ref_act.shape[1] == 1:
                ref_act = ref_act.squeeze(1)
            n = ref_act.shape[0]
            mask = real_token_mask(n, seq_upper)
            n_keep = int(mask.sum())
            if n_keep == 0:
                continue

            fam = fam_to_int[seq_meta[name]["rfam_id"]]
            for li in layer_indices:
                act = captured[li][0]
                if act.ndim == 3 and act.shape[1] == 1:
                    act = act.squeeze(1)
                per_layer_chunks[li].append(act[mask].clone())  # [n_keep, d] fp16
            y_chunks.append(torch.full((n_keep,), fam, dtype=torch.long))
            n_used += 1

    for h in handles:
        h.remove()
    elapsed = time.perf_counter() - start
    print(f"      Forward pass done in {elapsed:.1f}s "
          f"({n_used} seqs used, {n_skipped} skipped).")

    # free the model before the linear-algebra phase
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    y = torch.cat(y_chunks)

    # ---- per-layer geometry ----
    print(f"[4/4] Computing scatter geometry per layer on device={device}...")
    results: dict[str, dict] = {}
    for li in layer_indices:
        X = torch.cat(per_layer_chunks[li]).float()
        per_layer_chunks[li] = None  # free as we go
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

    # ---- context table + recommendation ----
    print("\n" + "=" * 88)
    print("PER-LAYER TABLE  (RNA-MSM, raw acts; geometry invariant to the global-scalar z-score)")
    print("=" * 88)
    hdr = f"{'layer':<8}{'eta_fam%':>10}{'var@32%':>10}{'fam@32%':>10}{'fam_pcs90':>11}{'fam_pcs50':>11}"
    print(hdr); print("-" * len(hdr))
    for li in layer_indices:
        r = results[str(li)]
        flag = ""
        if (r["var_top32"] <= CAND_VAR_TOP32_MAX and r["fam_pcs_90"] >= CAND_FAM_PCS_90_MIN
                and r["eta_family"] >= CAND_ETA_MIN):
            flag = "  <- ITERATE candidate"
        print(f"{('L'+str(li)):<8}{r['eta_family']*100:>10.2f}{r['var_top32']*100:>10.2f}"
              f"{r['fam_top32']*100:>10.2f}{r['fam_pcs_90']:>11}{r['fam_pcs_50']:>11}{flag}")

    print("\nReference (layer-6, from 14):")
    for k, v in REF_L6.items():
        print(f"  {k:<11}  eta_fam={v['eta_family']*100:5.2f}%  var@32={v['var_top32']*100:6.2f}%  "
              f"fam@32={v['fam_top32']*100:6.2f}%  fam_pcs90={v['fam_pcs_90']}")
    print("  (SAEs SUCCEEDED on rnafm + erniarna; FAILED on rnamsm at L6.)")

    # ---- sanity check: this scan's layer 6 must match 14's rnamsm_pre ----
    print("\nSANITY CHECK — this scan's layer 6 vs 14's rnamsm_pre (must match):")
    if str(6) in results:
        r6 = results["6"]; ref = REF_L6["rnamsm_L6"]
        dv = abs(r6["var_top32"] - ref["var_top32"])
        df = abs(r6["fam_top32"] - ref["fam_top32"])
        ok = dv < 0.03 and df < 0.03
        print(f"  scan L6: var@32={r6['var_top32']*100:.2f}%  fam@32={r6['fam_top32']*100:.2f}%  "
              f"fam_pcs90={r6['fam_pcs_90']}")
        print(f"  14 pre : var@32={ref['var_top32']*100:.2f}%  fam@32={ref['fam_top32']*100:.2f}%  "
              f"fam_pcs90={ref['fam_pcs_90']}")
        print(f"  -> {'✓ MATCH (pipeline consistent)' if ok else '⚠ MISMATCH — investigate before trusting other layers'}")

    # ---- automated recommendation ----
    print("\n" + "=" * 88)
    print("RECOMMENDATION")
    print("=" * 88)
    cands = [li for li in layer_indices
             if results[str(li)]["var_top32"] <= CAND_VAR_TOP32_MAX
             and results[str(li)]["fam_pcs_90"] >= CAND_FAM_PCS_90_MIN
             and results[str(li)]["eta_family"] >= CAND_ETA_MIN]
    if cands:
        # prefer the most-spread (highest fam_pcs_90), tie-break lowest var_top32
        best = sorted(cands, key=lambda li: (-results[str(li)]["fam_pcs_90"],
                                             results[str(li)]["var_top32"]))[0]
        b = results[str(best)]
        print(f"  => ITERATE on LAYER {best}. It breaks the rank-32 collapse "
              f"(var@32={b['var_top32']*100:.1f}% vs L6 94%) while keeping family")
        print(f"     signal spread + accessible (fam_pcs90={b['fam_pcs_90']}, "
              f"eta_family={b['eta_family']*100:.1f}%, fam@32={b['fam_top32']*100:.1f}%).")
        print(f"     NEXT: set LAYERS_TO_HOOK=[{best}] in 11_extract_rnamsm.py, re-extract,")
        print(f"     normalize_rnamsm.py, set_model.py rnamsm, 08 retrain, 09 inspect.")
    else:
        worst_var = max(results[str(li)]["var_top32"] for li in layer_indices)
        min_var = min(results[str(li)]["var_top32"] for li in layer_indices)
        print(f"  => NO viable layer. Every layer is near rank-32 "
              f"(var@32 range {min_var*100:.1f}-{worst_var*100:.1f}%, all >> RNA-FM 43%).")
        print(f"     The collapse is INTRINSIC to RNA-MSM at msa_depth=1, not a layer-6 quirk.")
        print(f"     DECISION: either (A) feed RNA-MSM its native MSA input (Rfam alignments)")
        print(f"     and re-diagnose, or (B) LOCK RNA-MSM as the cross-model NEGATIVE/CONTRAST")
        print(f"     case — an evolutionary-MSA model run without an MSA collapses to low")
        print(f"     effective rank, so linearly-present family signal cannot resolve into")
        print(f"     monosemantic SAE features. Both are publishable; (A) is the higher ceiling.")

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUT, "w") as f:
        json.dump({
            "config": {
                "top_k_families": TOP_K_FAMILIES, "n_classes": n_classes,
                "skip_special_tokens": SKIP_SPECIAL_TOKENS, "skip_n_tokens": SKIP_N_TOKENS,
                "k_budget": K_BUDGET, "ranks": RANKS,
                "n_layers": n_layers, "hidden_dim": hidden_dim,
                "n_seqs_used": n_used, "n_skipped": n_skipped,
                "space": "raw acts; metrics invariant to global-scalar z-score (normalize_rnamsm.py)",
                "candidate_thresholds": {
                    "var_top32_max": CAND_VAR_TOP32_MAX,
                    "fam_pcs_90_min": CAND_FAM_PCS_90_MIN,
                    "eta_min": CAND_ETA_MIN,
                },
            },
            "ref_layer6_from_14": REF_L6,
            "results_by_layer": results,
        }, f, indent=2)
    print(f"\n✅ Done. Full metrics -> {REPORT_OUT}")


if __name__ == "__main__":
    main()
