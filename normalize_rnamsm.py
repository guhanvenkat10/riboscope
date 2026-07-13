"""
normalize_rnamsm.py — Standardize RNA-MSM layer-6 activations to unit scale.

Why this exists
---------------
RNA-MSM layer-6 hidden states have std ≈ 32 (range ±1100), ~90× the scale of
ErnieRNA's (std ≈ 0.36) and ~6× RNA-FM's (std ≈ 5.5). The first RNA-MSM SAE
training run at that scale catastrophically failed (final eval EV = -0.118 —
worse than just predicting the mean) because the SAE's architecture and
hyperparameters (unit-norm decoder rows, k=32 TopK, AdamW LR=4e-4) are tuned
for the typical-transformer activation range. RNA-FM (std=5.5) was at the
upper edge but still trained fine (EV = 0.882). RNA-MSM is outside the
operating envelope.

What this script does
---------------------
1. Loads the raw RNA-MSM activations.
2. Computes the global per-element mean and std across all 30k sequences
   in fp64 (for numerical stability — the values are large).
3. Moves the raw file aside as activations_rnamsm_layer6_raw.safetensors.
4. Writes the z-scored (mean=0, std=1) version back to the original filename
   so set_model.py / 08 / 09 / 12 work unchanged.
5. Saves the (mean, std) to activations_rnamsm_layer6_normstats.json
   so we can un-normalize for reconstruction comparisons later if needed.

Why this is scientifically OK
-----------------------------
Cross-model agreement (12_cross_model_agreement.py) compares features by
their Rfam-family firing patterns with a per-feature relative magnitude
threshold (25% of feature's own max). Family identity and relative
within-feature magnitudes are invariant to a global affine transform of
the inputs, so the cross-model Jaccard remains valid. We document the
preprocessing as a methods detail in the writeup.

Other models don't need this — RNA-FM trained to EV=0.882 and ErnieRNA to
EV=0.674 at their native scales. Only RNA-MSM gets normalized.

Run with
--------
    cd ~/projects/riboscope
    uv run python normalize_rnamsm.py

Then resume the chain at SAE training:
    uv run python set_model.py rnamsm
    uv run python 08_train_sae_big.py
    uv run python 09_inspect_features_big.py
    uv run python 12_cross_model_agreement.py
"""

import json
import math
import shutil
import sys
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file, save_file
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)


SRC = Path("outputs/activations_rnamsm_layer6.safetensors")
RAW = Path("outputs/activations_rnamsm_layer6_raw.safetensors")
STATS = Path("outputs/activations_rnamsm_layer6_normstats.json")


def main() -> None:
    print("=" * 70)
    print("RIBOSCOPE: RNA-MSM activation normalization (z-score)")
    print("=" * 70)

    if not SRC.exists():
        print(f"❌ {SRC} not found.")
        sys.exit(1)

    if RAW.exists():
        print(f"⚠ {RAW} already exists — normalize_rnamsm.py appears to have been")
        print(f"  run before. Refusing to overwrite to avoid double-normalization.")
        print(f"  Delete {RAW} to force re-run from current SRC.")
        sys.exit(1)

    print(f"[1/4] Loading {SRC}...")
    d = load_file(str(SRC))
    keys = sorted(d.keys())
    print(f"      {len(keys)} tensors loaded")

    # Sanity-check the input shape
    sample_shape = d[keys[0]].shape
    if len(sample_shape) != 2:
        print(f"❌ Expected 2D tensors [seq_len, hidden], got {tuple(sample_shape)}")
        print(f"   The shape patch in 11_extract_rnamsm.py may not have been applied.")
        sys.exit(1)
    hidden_dim = sample_shape[1]
    print(f"      Sample tensor: {keys[0]} shape={tuple(sample_shape)}")
    print(f"      Hidden dim:    {hidden_dim}")

    # Pass 1: global mean + std in fp64 for numerical stability
    print(f"[2/4] Computing global mean + std across all elements (fp64 accumulator)...")
    sum_x = 0.0
    sum_x2 = 0.0
    n_total = 0
    for k in keys:
        t = d[k].to(torch.float64)
        sum_x += float(t.sum().item())
        sum_x2 += float((t * t).sum().item())
        n_total += t.numel()

    mean = sum_x / n_total
    var = max(sum_x2 / n_total - mean * mean, 0.0)
    std = math.sqrt(var)
    print(f"      mean        = {mean:+.6f}")
    print(f"      std         = {std:.6f}")
    print(f"      n_elements  = {n_total:,}")

    if std < 1e-6:
        print(f"❌ std is essentially zero — refusing to normalize (would divide by zero).")
        sys.exit(1)

    # Move raw aside BEFORE writing — guards against losing data if save fails
    print(f"[3/4] Moving raw activations aside → {RAW}")
    shutil.move(str(SRC), str(RAW))

    # Pass 2: build standardized dict and save (per-tensor, fp32 math, fp16 store)
    print(f"[4/4] Writing standardized activations → {SRC}")
    d_norm: dict[str, torch.Tensor] = {}
    for k in keys:
        t = d[k].to(torch.float32)
        t_norm = ((t - mean) / std).to(torch.float16)
        d_norm[k] = t_norm

    save_file(d_norm, str(SRC))

    # Post-save verification (catches the same truncated-write bug we saw earlier)
    print(f"      Verifying saved file...")
    d_check = load_file(str(SRC))
    if len(d_check) != len(d):
        print(f"❌ Verification failed: saved {len(d)} tensors but reload sees {len(d_check)}")
        sys.exit(1)
    if d_check[keys[0]].shape != d[keys[0]].shape:
        print(f"❌ Shape mismatch on sample tensor")
        sys.exit(1)
    print(f"      ✓ Verified: {len(d_check)} tensors load cleanly.")

    # Confirm the post-normalize stats look right (~mean=0, ~std=1)
    sample = torch.cat([d_check[k].to(torch.float32) for k in keys[:200]], dim=0)
    post_mean = sample.mean().item()
    post_std = sample.std().item()
    print(f"      Post-normalize sample stats (first 200 sequences):")
    print(f"        mean = {post_mean:+.4f}  (expect ~0)")
    print(f"        std  = {post_std:.4f}  (expect ~1)")
    if abs(post_mean) > 0.1 or abs(post_std - 1.0) > 0.1:
        print(f"⚠ Post-normalize stats look off — investigate before training.")

    # Save the stats so we can un-normalize for reconstruction comparisons
    with open(STATS, "w") as f:
        json.dump(
            {
                "mean": mean,
                "std": std,
                "n_elements": n_total,
                "note": (
                    "Apply (x - mean) / std to convert raw → normalized; apply "
                    "x * std + mean to convert normalized → raw."
                ),
            },
            f,
            indent=2,
        )

    norm_size = SRC.stat().st_size / 1e9
    raw_size = RAW.stat().st_size / 1e9
    print()
    print("=" * 70)
    print("✅ RNA-MSM normalization complete.")
    print("=" * 70)
    print(f"   Standardized: {SRC} ({norm_size:.2f} GB)")
    print(f"   Raw (kept):   {RAW} ({raw_size:.2f} GB)")
    print(f"   Stats:        {STATS}")
    print()
    print("   Next steps (retrain RNA-MSM SAE on standardized inputs):")
    print("     uv run python set_model.py rnamsm")
    print("     uv run python 08_train_sae_big.py")
    print("     uv run python 09_inspect_features_big.py")
    print("     uv run python 12_cross_model_agreement.py")


if __name__ == "__main__":
    main()
