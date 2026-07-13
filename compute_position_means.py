"""
compute_position_means.py — RIBOSCOPE v3: per-position mean of RNA-MSM acts.

Why this exists
---------------
RNA-MSM's layer-6 SAE has failed twice (v1 dict=8192, v2 dict=4096 + N-mask),
both hitting EV ≈ 0.998 while learning ZERO biology: the top features are
per-position constants (fire on token position 0/1/197/232-260 across every
Rfam family), and specialists max out at 0.5-0.9 vs RNA-FM's 5-65. The
dictionary collapses onto a positional signal rather than nucleotide motifs.

Verified mechanism (config.json + multimolecule modeling source, 2026-05-28):
all three models we use ADD a position term to the residual stream at the input
layer — RNA-FM and RNA-MSM use learned absolute embeddings, ErnieRNA sinusoidal.
(The earlier "RNA-FM/ErnieRNA use RoPE that never enters the residual" claim was
wrong and is retired; see position_variance_report.py for the quantitative
contrast.) The distinguishing fact is not presence vs absence but DEGREE of
dominance: RNA-MSM is an MSA Transformer (ESM-MSA-1b lineage) run out of
distribution on single sequences (msa_depth=1), so its row/MSA attention
degenerates and the representation collapses onto the per-position term.

The fix: estimate the per-position mean activation across all 30k sequences and
subtract it before SAE training, so the SAE can no longer "explain variance" by
memorizing position. This script computes and saves that per-position mean; the
sibling subtract_position_means.py applies it.

What this script does
---------------------
1. Loads the normalized RNA-MSM activations (output of normalize_rnamsm.py).
2. Pass 1: finds max token length across all sequences.
3. Pass 2 (fp64 accumulators): per-position sum S_p and contributor count n_p,
   plus the global second moment needed for the variance-explained diagnostic.
4. Builds position_mean[p] = S_p / n_p, then applies a CONTRIBUTOR GUARD:
   positions with n_p < MIN_CONTRIBUTORS are zeroed (so subtraction is a no-op
   there — see "Why the guard" below).
5. Saves position_mean [max_seq_len, hidden] and position_count [max_seq_len].
6. Computes the fraction of total activation variance explained by position
   (one-way ANOVA on position, pooled over dims) — the headline diagnostic.
7. Writes a JSON sidecar with per-position n_p, the variance fraction, and
   provenance so every downstream decision is auditable.

Why the guard (MIN_CONTRIBUTORS)
--------------------------------
Sequences are len ∈ [50, 510]. Low token positions appear in nearly all 30k
sequences (n_p ≈ 30k) — these carry the dominant positional artifact and are
estimated with tiny error. High positions (≈450-510) appear in only a handful
of long sequences; their per-position mean is dominated by sampling noise
(estimator variance ~ σ²/n_p). Subtracting a noisy mean there would inject
noise instead of removing structure, and those positions contribute very few
tokens to training anyway. So we only subtract where n_p ≥ MIN_CONTRIBUTORS and
leave the sparse tail untouched. n_p is logged per position so the cut is
auditable.

Why this is scientifically OK
-----------------------------
Cross-model agreement (12_cross_model_agreement.py) compares features by their
Rfam-family firing patterns (relative within-feature magnitude). It does NOT
require identical preprocessing across models — the whole premise is "do
biological features replicate across architectures AND preprocessing." Removing
a position-only term cannot create spurious family structure; it can only stop
position-detector features from crowding out biology. Documented as a methods
detail in the writeup.

Run with
--------
    cd ~/projects/riboscope
    uv run python compute_position_means.py            # normal
    uv run python compute_position_means.py --overwrite  # force re-run

Output
------
    outputs/activations_rnamsm_layer6_posmean.safetensors
    outputs/activations_rnamsm_layer6_posmean_meta.json
"""

import json
import sys
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file, save_file
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)


# ============================ CONFIG ============================
SRC = Path("outputs/activations_rnamsm_layer6.safetensors")          # normalized acts
POSMEAN = Path("outputs/activations_rnamsm_layer6_posmean.safetensors")
META = Path("outputs/activations_rnamsm_layer6_posmean_meta.json")

# Contributor guard: only trust (and later subtract) a per-position mean that
# was estimated from at least this many sequences. Below this, the estimate is
# noise-dominated and the position is left untouched. See "Why the guard".
MIN_CONTRIBUTORS = 100
# ================================================================


def main() -> None:
    overwrite = "--overwrite" in sys.argv[1:]

    print("=" * 70)
    print("RIBOSCOPE v3: RNA-MSM per-position mean computation")
    print("=" * 70)

    if not SRC.exists():
        print(f"❌ {SRC} not found.")
        print(f"   Expected the NORMALIZED RNA-MSM activations (run normalize_rnamsm.py).")
        sys.exit(1)

    if POSMEAN.exists() and not overwrite:
        print(f"⚠ {POSMEAN} already exists.")
        print(f"  Refusing to overwrite. Re-run with --overwrite to recompute.")
        sys.exit(1)

    POSMEAN.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------- #
    # Load
    # --------------------------------------------------------------- #
    print(f"[1/5] Loading {SRC}...")
    d = load_file(str(SRC))
    keys = sorted(k for k in d.keys())
    if not keys:
        print(f"❌ No tensors found in {SRC}.")
        sys.exit(1)

    sample_shape = d[keys[0]].shape
    if len(sample_shape) != 2:
        print(f"❌ Expected 2D tensors [seq_len, hidden], got {tuple(sample_shape)}.")
        print(f"   Did 11_extract_rnamsm.py apply the [L, hidden] squeeze?")
        sys.exit(1)
    hidden_dim = int(sample_shape[1])
    print(f"      {len(keys)} tensors, hidden_dim={hidden_dim}")

    # --------------------------------------------------------------- #
    # Pass 1: max token length
    # --------------------------------------------------------------- #
    print(f"[2/5] Pass 1: finding max token length...")
    max_len = 0
    for k in keys:
        L = int(d[k].shape[0])
        if d[k].shape[1] != hidden_dim:
            print(f"❌ {k} has hidden dim {int(d[k].shape[1])} != {hidden_dim}.")
            sys.exit(1)
        if L > max_len:
            max_len = L
    print(f"      max token length = {max_len}")

    # --------------------------------------------------------------- #
    # Pass 2: per-position sum + count, plus global moments (fp64)
    # --------------------------------------------------------------- #
    print(f"[3/5] Pass 2: accumulating per-position sums (fp64)...")
    pos_sum = torch.zeros((max_len, hidden_dim), dtype=torch.float64)   # S_p (vector per position)
    pos_count = torch.zeros(max_len, dtype=torch.int64)                 # n_p
    global_sumsq = 0.0   # Σ_i ||x_i||^2   (scalar, over all elements)
    n_tokens_total = 0

    for k in keys:
        t = d[k].to(torch.float64)         # [L, hidden]
        L = t.shape[0]
        pos_sum[:L] += t
        pos_count[:L] += 1
        global_sumsq += float((t * t).sum().item())
        n_tokens_total += L

    # global sum vector = Σ_p S_p ; global mean vector μ = (Σ S_p) / N_tokens
    global_sum = pos_sum.sum(dim=0)                          # [hidden]
    global_mean = global_sum / max(n_tokens_total, 1)        # [hidden]

    # --------------------------------------------------------------- #
    # Per-position mean + contributor guard
    # --------------------------------------------------------------- #
    print(f"[4/5] Building guarded per-position mean (MIN_CONTRIBUTORS={MIN_CONTRIBUTORS})...")
    safe_count = pos_count.clamp(min=1).unsqueeze(1).to(torch.float64)
    pos_mean_raw = pos_sum / safe_count                     # μ_p for all positions
    # zero where n_p == 0 (no contributors at all)
    pos_mean_raw[pos_count == 0] = 0.0

    # Guard: positions below threshold are left untouched downstream (mean→0).
    guard_keep = pos_count >= MIN_CONTRIBUTORS
    pos_mean_guarded = pos_mean_raw.clone()
    pos_mean_guarded[~guard_keep] = 0.0

    n_kept = int(guard_keep.sum().item())
    n_zeroed = int((~guard_keep).sum().item())
    print(f"      positions with n_p >= {MIN_CONTRIBUTORS}: {n_kept} (subtracted)")
    print(f"      positions with n_p <  {MIN_CONTRIBUTORS}: {n_zeroed} (left untouched)")
    if n_kept > 0:
        kept_idx = guard_keep.nonzero(as_tuple=False).flatten()
        print(f"      kept position range: {int(kept_idx.min())}..{int(kept_idx.max())}")

    # --------------------------------------------------------------- #
    # Variance-explained-by-position diagnostic (uses UNGUARDED means —
    # it's a descriptive stat over the full data, independent of the guard).
    #   SS_total   = Σ_i ||x_i - μ||^2 = Σ||x_i||^2 - (1/N)||Σ x_i||^2
    #   SS_between = Σ_p n_p ||μ_p - μ||^2 = Σ_p ||S_p||^2 / n_p - (1/N)||Σ x_i||^2
    #   fraction   = SS_between / SS_total   ∈ [0, 1]
    # --------------------------------------------------------------- #
    correction = float((global_sum * global_sum).sum().item()) / max(n_tokens_total, 1)
    ss_total = global_sumsq - correction
    pc = pos_count.clamp(min=1).to(torch.float64)
    ss_between = float(((pos_sum * pos_sum).sum(dim=1) / pc).sum().item()) - correction
    var_fraction = ss_between / ss_total if ss_total > 0 else 0.0
    print(f"      SS_total           = {ss_total:.4e}")
    print(f"      SS_between(pos)     = {ss_between:.4e}")
    print(f"      variance explained by position = {var_fraction:.4f}")

    # --------------------------------------------------------------- #
    # Save (fp32 means; int64 counts)
    # --------------------------------------------------------------- #
    print(f"[5/5] Saving → {POSMEAN}")
    out = {
        "position_mean": pos_mean_guarded.to(torch.float32),   # what subtract uses
        "position_count": pos_count,                           # n_p per position
        "position_mean_unguarded": pos_mean_raw.to(torch.float32),  # for transparency
    }
    save_file(out, str(POSMEAN))

    # Post-save verification (catches truncated writes)
    print(f"      Verifying saved file...")
    chk = load_file(str(POSMEAN))
    if chk["position_mean"].shape != (max_len, hidden_dim):
        print(f"❌ position_mean shape mismatch after reload: {tuple(chk['position_mean'].shape)}")
        sys.exit(1)
    if int(chk["position_count"].sum().item()) != n_tokens_total:
        print(f"❌ position_count sum mismatch after reload.")
        sys.exit(1)
    print(f"      ✓ Verified: position_mean {tuple(chk['position_mean'].shape)}, "
          f"counts sum to {n_tokens_total:,} tokens.")

    with open(META, "w") as f:
        json.dump(
            {
                "src": str(SRC),
                "hidden_dim": hidden_dim,
                "max_token_len": max_len,
                "n_sequences": len(keys),
                "n_tokens_total": n_tokens_total,
                "min_contributors": MIN_CONTRIBUTORS,
                "n_positions_subtracted": n_kept,
                "n_positions_untouched": n_zeroed,
                "variance_explained_by_position": var_fraction,
                "ss_total": ss_total,
                "ss_between_position": ss_between,
                "position_count": pos_count.tolist(),
                "note": (
                    "position_mean is the GUARDED mean (zeroed where n_p < "
                    "min_contributors). Apply x[p] - position_mean[p] to center; "
                    "guarded positions are unchanged. variance_explained_by_position "
                    "is computed on unguarded means over the full data."
                ),
            },
            f,
            indent=2,
        )

    size_mb = POSMEAN.stat().st_size / 1e6
    print()
    print("=" * 70)
    print("✅ Per-position mean computed.")
    print("=" * 70)
    print(f"   Position mean: {POSMEAN} ({size_mb:.1f} MB)")
    print(f"   Meta:          {META}")
    print(f"   Variance explained by position: {var_fraction:.4f}")
    print()
    print("   Next:")
    print("     uv run python subtract_position_means.py")
    print("     uv run python set_model.py rnamsm")
    print("     uv run python 08_train_sae_big.py")


if __name__ == "__main__":
    main()
