"""
14_family_geometry_diagnostic.py — RIBOSCOPE: WHERE does family signal live in
the layer-6 representation, and would whitening surface it for the SAE?

Why this exists
---------------
The linear probe (13) proved family identity is strongly, linearly decodable
from RNA-MSM's layer-6 acts (token top-1 0.85 @ 100-way, chance 0.01; ratio 0.89
vs RNA-FM/ErnieRNA). Yet the v3 SAE's best family-specialist maxes at 0.137 —
40-400x below RNA-FM's healthy specialists (5-65). A 1.13x representation gap
cannot explain a 400x specialist gap. The hypothesis: family signal is PRESENT
but GEOMETRICALLY BURIED — it lives in low-variance / distributed directions,
while dominant nucleotide-identity + residual-position variance soak up the
top-k L2 reconstruction budget. A linear probe doesn't care (it reweights all
dims freely with learned weights); a BatchTopK L2-reconstruction SAE does.

CRITICAL — the space we measure
-------------------------------
The SAE (sae_models.py) subtracts a learned pre-bias b_dec before encoding
(x_centered = x - b_dec) but does NOT z-score per dim and does NOT whiten. So
the geometry the SAE actually "sees" is the MEAN-CENTERED, native-per-dim-scale
activation space. This script therefore does NOT standardize. All scatter is
measured relative to the global mean (which the SAE's b_dec converges to),
faithfully reproducing the variance landscape the reconstruction loss weights.

What it measures (per model, on the SAME top-100 families / token filtering as 13)
----------------------------------------------------------------------------------
Let S_T = total scatter (Σ (x-μ)(x-μ)^T), S_B = between-family scatter
(Σ_c n_c (μ_c-μ)(μ_c-μ)^T). PCA = eigendecomposition of S_T (directions ordered
by total variance = the order an L2 SAE prioritizes).

  1. eta_family = trace(S_B)/trace(S_T)         fraction of total variance that
                                                is between-family. Low = family
                                                signal is a small slice of the
                                                variance the SAE must reconstruct.
  2. var_top32  = % total variance in top-32 PCs   how concentrated the dominant
                                                   (nucleotide+position) variance
                                                   is in the SAE's k=32 budget.
  3. fam_top32  = % FAMILY variance in those same top-32 PCs   THE BURIAL TEST.
                  If fam_top32 << var_top32 (and << the successful models),
                  family signal sits in the low-variance tail the SAE ignores.
  4. fam_pcs_90 = # of (variance-ordered) PCs to capture 90% of family variance
                  high = family signal is distributed across many weak dims.
  5. WHITENING SIMULATION: PCA-whiten (each PC rescaled to unit variance), then
     measure how concentrated family variance becomes in the whitened basis
     (eigenspectrum of the whitened S_B). fam_white_top32 / fam_white_dirs_90.
     If whitening concentrates family signal into few directions, an SAE on
     whitened acts can form specialists -> whitening is the principled fix.

Decision (printed): compares RNA-MSM(v3) against the mean of the two models whose
SAEs SUCCEEDED (RNA-FM, ErnieRNA) and recommends one of:
  - PREPROCESS+RETRAIN with whitening  (buried AND whitening concentrates it)
  - whitening won't help / signal distributed in a way whitening can't fix
  - burial is NOT the mechanism -> the failure is sparse-allocation, not geometry
    (try a feature-frequency penalty / lower k, not whitening)

Run with
--------
    cd ~/projects/riboscope
    uv run python 14_family_geometry_diagnostic.py

Output
------
    outputs/family_geometry_diagnostic.json   (full metrics, auditable)
    console comparison table + recommended preprocessing decision
"""

from __future__ import annotations

import gc
import json
import sys
from collections import Counter
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)


# ============================ CONFIG ============================
LAYER = 6

# SAME activation sources as 13 (the probe), so geometry is directly comparable.
MODELS = [
    {"tag": "rnafm",      "act_file": "outputs/activations_big_layer6_v2.safetensors",
     "note": "RNA-FM native acts (SAE EV=0.882, real specialists)"},
    {"tag": "erniarna",   "act_file": "outputs/activations_erniarna_layer6.safetensors",
     "note": "ErnieRNA native acts (SAE EV=0.679, real specialists)"},
    {"tag": "rnamsm_v3",  "act_file": "outputs/activations_rnamsm_layer6_poscentered.safetensors",
     "note": "RNA-MSM position-centered acts (what the v3 SAE saw)"},
    {"tag": "rnamsm_pre", "act_file": "outputs/activations_rnamsm_layer6.safetensors",
     "note": "RNA-MSM normalized, NOT position-centered (v2 input; control)"},
]

FASTA_FILE = Path("sequences/rfam_30k.fasta")
REPORT_OUT = Path("outputs/family_geometry_diagnostic.json")

# Identical K-way problem as the probe.
TOP_K_FAMILIES = 100

# Token filtering — mirror SAE training / 09 inspection / the probe EXACTLY.
SKIP_SPECIAL_TOKENS = True   # drop CLS (pos 0) and EOS (pos n-1)
SKIP_N_TOKENS = True         # drop N nucleotide positions

# The SAE's effective per-token budget (k active features). We report family
# variance captured within the top-K_BUDGET principal components.
K_BUDGET = 32

# Cumulative-capture report points (PC ranks).
RANKS = [1, 5, 10, 32, 64, 128, 256]

# Whitening numerical floor: PCs with variance < EPS_FRAC * mean(variance) are
# treated as noise and NOT amplified (prevents whitening from blowing up tiny
# directions). Reported so the choice is auditable.
EPS_FRAC = 1e-3
# ================================================================


def parse_fasta_with_metadata(path: Path) -> dict:
    """Parse the Rfam FASTA produced by 06_fetch_rfam_sequences.py.
    Header: ">{name} | {rfam_id} | {rfam_name} | {len} nt". Same parser as
    08 / 09 / 13 so token indexing is identical. T->U applied."""
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
    """True for tokens to keep. Mirrors 09/13: pos 0 = CLS, pos n-1 = EOS,
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


def collect_tokens(acts, selected_seqs, seq_meta, fam_to_int):
    """Stack all real-token activations for the selected families into one
    matrix X [N, d] (float32) with integer family labels y [N]. No split, no
    standardization — descriptive geometry over the full selection."""
    X_chunks, y_chunks = [], []
    n_used = 0
    for name in selected_seqs:
        key = f"{name}__layer{LAYER}"
        if key not in acts:
            continue
        t = acts[key].float()                       # [n, d]
        seq_upper = seq_meta[name]["sequence"].upper()
        rows = t[real_token_mask(t.shape[0], seq_upper)]
        if rows.shape[0] == 0:
            continue
        y = fam_to_int[seq_meta[name]["rfam_id"]]
        X_chunks.append(rows)
        y_chunks.append(torch.full((rows.shape[0],), y, dtype=torch.long))
        n_used += 1
    X = torch.cat(X_chunks)
    y = torch.cat(y_chunks)
    return X, y, n_used


def scatter_geometry(X: torch.Tensor, y: torch.Tensor, n_classes: int, device):
    """Compute the family-vs-total variance geometry for one model.

    Returns a dict of scalar metrics + cumulative curves. All heavy linear
    algebra: S_T / S_B built in float32 on `device` (no catastrophic
    cancellation — we center first), eigendecomposition in float64 on CPU
    (d<=768, milliseconds, full precision)."""
    d = X.shape[1]
    X = X.to(device)
    N = X.shape[0]

    # --- global + class means (centered scatter avoids cancellation) ---
    mu = X.mean(dim=0)                                  # [d]
    Xc = X - mu                                          # centered
    S_T = (Xc.T @ Xc)                                    # [d,d]  total scatter
    trace_T = (Xc * Xc).sum()                            # = trace(S_T)

    # class means via scatter-add
    csum = torch.zeros(n_classes, d, device=device)
    csum.index_add_(0, y.to(device), X)
    cn = torch.zeros(n_classes, device=device)
    cn.index_add_(0, y.to(device), torch.ones(N, device=device))
    present = cn > 0
    mu_c = torch.zeros_like(csum)
    mu_c[present] = csum[present] / cn[present].unsqueeze(1)

    # D rows = sqrt(n_c) * (mu_c - mu) so that S_B = D^T D
    dev = (mu_c[present] - mu) * cn[present].sqrt().unsqueeze(1)   # [Kc, d]
    S_B = dev.T @ dev                                    # [d,d]  between-family
    trace_B = (dev * dev).sum()                          # = trace(S_B)
    eta_family = float(trace_B / trace_T)

    # --- move small d x d / Kc x d matrices to CPU float64 for eig ---
    S_T64 = S_T.double().cpu()
    dev64 = dev.double().cpu()
    trace_T = float(trace_T)
    trace_B = float(trace_B)

    # PCA of total scatter: eigenvectors ordered by DESC variance (the order an
    # L2 SAE prioritizes for reconstruction).
    evals, evecs = torch.linalg.eigh(S_T64)              # ascending
    order = torch.argsort(evals, descending=True)
    lam = evals[order].clamp(min=0.0)                    # [d] desc total-variance
    V = evecs[:, order]                                  # [d,d] columns = PCs

    # total variance captured by top ranks
    lam_cum = torch.cumsum(lam, dim=0)
    var_frac_at = {r: float(lam_cum[min(r, d) - 1] / lam.sum()) for r in RANKS if r <= d}

    # family variance carried by each PC: b_i = ||dev @ v_i||^2
    DV = dev64 @ V                                       # [Kc, d] in PC order
    fam_pc = (DV * DV).sum(dim=0)                         # [d] family var per PC
    fam_cum = torch.cumsum(fam_pc, dim=0)
    fam_total = float(fam_pc.sum())                      # == trace_B (sanity)
    fam_frac_at = {r: float(fam_cum[min(r, d) - 1] / fam_total) for r in RANKS if r <= d}

    def count_for(curve_cum, total, thresh):
        hit = (curve_cum >= thresh * total).nonzero()
        return int(hit[0].item()) + 1 if len(hit) else d
    fam_pcs_90 = count_for(fam_cum, fam_total, 0.90)
    fam_pcs_50 = count_for(fam_cum, fam_total, 0.50)

    # --- WHITENING SIMULATION ---
    # PCA-whiten: each PC scaled to unit variance (variance = lam/N, but scale
    # is a global constant so we use lam directly). Tiny PCs (noise) floored.
    eps = EPS_FRAC * float(lam.mean())
    inv_sqrt = torch.where(lam > eps, lam.clamp(min=eps).rsqrt(),
                           torch.zeros_like(lam))        # noise dirs -> 0 (dropped)
    n_white_dims = int((lam > eps).sum())
    W = V * inv_sqrt.unsqueeze(0)                         # [d,d]  whitening map
    DW = dev64 @ W                                        # [Kc, d] family in white space
    S_Bw = DW.T @ DW
    ew, _ = torch.linalg.eigh(S_Bw)
    ew = ew.clamp(min=0.0).sort(descending=True).values  # family var per white dir
    ew_total = float(ew.sum())
    ew_cum = torch.cumsum(ew, dim=0)
    fam_white_frac_at = {r: float(ew_cum[min(r, d) - 1] / ew_total) for r in RANKS if r <= d}
    fam_white_dirs_90 = count_for(ew_cum, ew_total, 0.90)
    fam_white_dirs_50 = count_for(ew_cum, ew_total, 0.50)

    return {
        "d": int(d),
        "n_tokens": int(N),
        "eta_family": eta_family,
        "var_top32": var_frac_at.get(K_BUDGET),
        "fam_top32": fam_frac_at.get(K_BUDGET),
        "fam_pcs_50": fam_pcs_50,
        "fam_pcs_90": fam_pcs_90,
        "fam_white_top32": fam_white_frac_at.get(K_BUDGET),
        "fam_white_dirs_50": fam_white_dirs_50,
        "fam_white_dirs_90": fam_white_dirs_90,
        "n_white_dims": n_white_dims,
        "var_frac_cum": var_frac_at,
        "fam_frac_cum": fam_frac_at,
        "fam_white_frac_cum": fam_white_frac_at,
        "trace_T": trace_T,
        "trace_B": trace_B,
    }


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 84)
    print("RIBOSCOPE: family-signal geometry diagnostic (is the signal BURIED?)")
    print("=" * 84)

    if not FASTA_FILE.exists():
        print(f"❌ FASTA not found: {FASTA_FILE}")
        sys.exit(1)

    # ---- family selection (ONCE, shared) — identical to the probe ----
    print(f"[1/3] Parsing {FASTA_FILE}; selecting top-{TOP_K_FAMILIES} families...")
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
    print(f"      Space measured:            MEAN-CENTERED native scale "
          f"(matches SAE b_dec; NO z-score, NO whiten)")

    # ---- per-model geometry ----
    print(f"[2/3] Measuring scatter geometry on device={device}...")
    results: dict[str, dict] = {}
    for cfg in MODELS:
        tag, act_path = cfg["tag"], Path(cfg["act_file"])
        print("\n" + "-" * 84)
        print(f"MODEL: {tag}   ({cfg['note']})")
        print(f"  act_file: {act_path}")
        if not act_path.exists():
            print(f"  ⚠ MISSING — skipping {tag}.")
            results[tag] = {"error": "missing_act_file", "act_file": str(act_path)}
            continue

        acts = load_file(str(act_path))
        X, y, n_used = collect_tokens(acts, selected_seqs, seq_meta, fam_to_int)
        del acts
        gc.collect()
        g = scatter_geometry(X, y, n_classes, device)
        del X, y
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        g["n_seqs_used"] = n_used
        results[tag] = g

        print(f"  d={g['d']}  seqs={n_used}  tokens={g['n_tokens']:,}")
        print(f"  eta_family (between-fam / total variance) : {g['eta_family']*100:6.2f} %")
        print(f"  total variance in top-{K_BUDGET} PCs            : {g['var_top32']*100:6.2f} %")
        print(f"  FAMILY variance in top-{K_BUDGET} PCs           : {g['fam_top32']*100:6.2f} %"
              f"   <- burial test")
        print(f"  PCs needed for 90% of family variance     : {g['fam_pcs_90']}  "
              f"(50%: {g['fam_pcs_50']})")
        print(f"  [whiten] FAMILY var in top-{K_BUDGET} white dirs : {g['fam_white_top32']*100:6.2f} %")
        print(f"  [whiten] white dirs for 90% family var    : {g['fam_white_dirs_90']}  "
              f"(50%: {g['fam_white_dirs_50']};  usable dims: {g['n_white_dims']}/{g['d']})")

    # ---- comparison + recommendation ----
    print("\n" + "=" * 84)
    print("[3/3] COMPARISON")
    print("=" * 84)
    hdr = (f"{'model':<13}{'eta_fam%':>9}{'var@32%':>9}{'fam@32%':>9}"
           f"{'fam_pcs90':>10}{'whtfam@32%':>11}{'whtdirs90':>10}")
    print(hdr)
    print("-" * len(hdr))
    for tag, r in results.items():
        if "error" in r:
            print(f"{tag:<13}  MISSING ({r['act_file']})")
            continue
        print(f"{tag:<13}"
              f"{r['eta_family']*100:>9.2f}"
              f"{r['var_top32']*100:>9.2f}"
              f"{r['fam_top32']*100:>9.2f}"
              f"{r['fam_pcs_90']:>10}"
              f"{r['fam_white_top32']*100:>11.2f}"
              f"{r['fam_white_dirs_90']:>10}")

    # ---- automated recommendation ----
    def ok(tag):
        return tag in results and "error" not in results[tag]
    print("\nRECOMMENDED PREPROCESSING DECISION")
    print("-" * 84)
    if not (ok("rnamsm_v3") and (ok("rnafm") or ok("erniarna"))):
        print("  Incomplete — need rnamsm_v3 plus at least one of rnafm/erniarna.")
    else:
        refs = [results[t] for t in ("rnafm", "erniarna") if ok(t)]
        ref_eta = sum(r["eta_family"] for r in refs) / len(refs)
        ref_famtop = sum(r["fam_top32"] for r in refs) / len(refs)
        ref_pcs90 = sum(r["fam_pcs_90"] for r in refs) / len(refs)
        ms = results["rnamsm_v3"]
        print(f"  reference (mean RNA-FM/ErnieRNA): eta_family={ref_eta*100:.2f}%  "
              f"fam@32={ref_famtop*100:.2f}%  fam_pcs90={ref_pcs90:.0f}")
        print(f"  RNA-MSM(v3)                     : eta_family={ms['eta_family']*100:.2f}%  "
              f"fam@32={ms['fam_top32']*100:.2f}%  fam_pcs90={ms['fam_pcs_90']}")

        # Buried = family variance disproportionately OUT of the SAE's top-32
        # budget vs the successful models, OR spread across many more dims.
        buried = (ms["fam_top32"] < 0.7 * ref_famtop) or (ms["fam_pcs_90"] > 1.5 * ref_pcs90)
        # Whitening helps only if it CONCENTRATES family signal into the budget
        # well beyond the pre-whitening accessibility.
        whiten_helps = ms["fam_white_top32"] >= max(0.5, 1.5 * ms["fam_top32"])

        print()
        if buried and whiten_helps:
            print("  => PREPROCESS + RETRAIN. Family signal is geometrically buried in")
            print("     RNA-MSM (low-variance / distributed directions the top-k L2 SAE")
            print("     under-weights), and PCA-whitening concentrates it back into the")
            print(f"     k={K_BUDGET} budget (fam@32 {ms['fam_top32']*100:.1f}% -> whitened "
                  f"{ms['fam_white_top32']*100:.1f}%). Plan: (1) full-length centering")
            print("     (relax compute_position_means MIN_CONTRIBUTORS so the 445-511 tail")
            print("     is centered too), (2) PCA-whiten the centered acts, (3) retrain SAE,")
            print("     (4) re-run 09 — specialist max should climb off 0.137.")
        elif buried and not whiten_helps:
            print("  => BURIED, but whitening alone won't fix it. Family variance is")
            print("     distributed in a way PCA-whitening does not concentrate into the")
            print(f"     k={K_BUDGET} budget (whitened fam@32 only {ms['fam_white_top32']*100:.1f}%).")
            print("     Consider: larger k, a feature-frequency penalty to free budget from")
            print("     generic detectors, or a supervised/steered objective. Re-evaluate")
            print("     before a blind retrain.")
        else:
            print("  => BURIAL is NOT the mechanism. RNA-MSM's family variance is about as")
            print("     accessible within the top-32 budget as in the models whose SAEs")
            print("     succeeded. The v3 failure is sparse-ALLOCATION, not geometry:")
            print("     generic high-frequency detectors are winning the budget. Fix = a")
            print("     feature-frequency / activation-density penalty (discourage features")
            print("     firing on >X% of tokens) or lower k — NOT whitening. Retrain with")
            print("     that change and re-run 09.")

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUT, "w") as f:
        json.dump({
            "config": {
                "top_k_families": TOP_K_FAMILIES, "n_classes": n_classes,
                "skip_special_tokens": SKIP_SPECIAL_TOKENS, "skip_n_tokens": SKIP_N_TOKENS,
                "k_budget": K_BUDGET, "ranks": RANKS, "eps_frac": EPS_FRAC,
                "space": "mean-centered native scale (matches SAE b_dec; no z-score/whiten)",
            },
            "results": results,
        }, f, indent=2)
    print(f"\n✅ Done. Full metrics -> {REPORT_OUT}")


if __name__ == "__main__":
    main()
