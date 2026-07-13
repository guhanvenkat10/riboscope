"""
subtract_position_means.py — RIBOSCOPE v3: position-center RNA-MSM acts.

Why this exists
---------------
Second half of the RNA-MSM v3 fix (see compute_position_means.py for the full
diagnosis). RNA-MSM's SAE collapsed onto a per-position signal twice. This
script subtracts the per-position mean (computed by compute_position_means.py)
from every token so the SAE can no longer "explain variance" by memorizing
position, forcing it to learn nucleotide motifs instead.

This is the ONLY variable that changes between RNA-MSM v2 (failed, EV=0.998,
zero biology) and v3. The SAE hyperparameters in 08_train_sae_big.py
(DICT_SIZE=4096, K=32, LR=4e-4, MASK_N_TOKENS=True) are held fixed, so v2 vs v3
is a clean A/B that attributes any improvement specifically to position-
centering. (v2 itself is the negative control — it was trained on globally
z-scored but NOT position-centered activations.)

What this script does
---------------------
1. Loads the normalized RNA-MSM activations and the guarded per-position mean.
2. For each sequence tensor [L, hidden], subtracts position_mean[0:L].
   Positions the guard left at 0 are unchanged (no-op subtraction).
3. Preserves the EXACT {seqname}__layer{LAYER} keys and per-key shapes that
   08/09/12 depend on (the N-mask routine reads raw[key].shape[0]).
4. Writes activations_rnamsm_layer6_poscentered.safetensors (fp16) + meta.
5. Disk-space precheck, post-save verify (reload + count + shape), and a
   post-subtraction sanity check that high-n_p positions now have ~0 mean.

Run with
--------
    cd ~/projects/riboscope
    uv run python subtract_position_means.py            # normal
    uv run python subtract_position_means.py --overwrite  # force re-run

Output
------
    outputs/activations_rnamsm_layer6_poscentered.safetensors
    outputs/activations_rnamsm_layer6_poscentered_meta.json
"""

import json
import shutil
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
OUT = Path("outputs/activations_rnamsm_layer6_poscentered.safetensors")
META = Path("outputs/activations_rnamsm_layer6_poscentered_meta.json")
# ================================================================


def estimate_output_bytes(d: dict) -> int:
    # fp16 store → 2 bytes/element
    return sum(t.numel() * 2 for t in d.values())


def main() -> None:
    overwrite = "--overwrite" in sys.argv[1:]

    print("=" * 70)
    print("RIBOSCOPE v3: RNA-MSM position-centering")
    print("=" * 70)

    if not SRC.exists():
        print(f"❌ {SRC} not found (run normalize_rnamsm.py).")
        sys.exit(1)
    if not POSMEAN.exists():
        print(f"❌ {POSMEAN} not found (run compute_position_means.py first).")
        sys.exit(1)
    if OUT.exists() and not overwrite:
        print(f"⚠ {OUT} already exists.")
        print(f"  Refusing to overwrite. Re-run with --overwrite to regenerate.")
        sys.exit(1)

    OUT.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------- #
    # Load
    # --------------------------------------------------------------- #
    print(f"[1/4] Loading activations + position mean...")
    d = load_file(str(SRC))
    keys = sorted(d.keys())
    pm_data = load_file(str(POSMEAN))
    position_mean = pm_data["position_mean"].to(torch.float32)   # [max_len, hidden]
    position_count = pm_data["position_count"]
    max_len, hidden_dim = position_mean.shape
    print(f"      {len(keys)} sequences, hidden_dim={hidden_dim}, posmean rows={max_len}")

    # Shape sanity vs activations
    samp = d[keys[0]].shape
    if len(samp) != 2 or int(samp[1]) != hidden_dim:
        print(f"❌ Activation/posmean hidden-dim mismatch: act {tuple(samp)} vs posmean {hidden_dim}.")
        sys.exit(1)

    # --------------------------------------------------------------- #
    # Subtract (fp32 math, fp16 store), preserving keys + shapes
    # --------------------------------------------------------------- #
    print(f"[2/4] Subtracting per-position mean (preserving keys/shapes)...")
    d_centered: dict[str, torch.Tensor] = {}
    n_too_long = 0
    for k in keys:
        t = d[k].to(torch.float32)          # [L, hidden]
        L = t.shape[0]
        if L > max_len:
            # Should never happen (posmean built from the same file), but guard:
            # only center the covered prefix, leave any overflow untouched.
            n_too_long += 1
            t[:max_len] = t[:max_len] - position_mean[:max_len]
        else:
            t = t - position_mean[:L]
        d_centered[k] = t.to(torch.float16)

    if n_too_long:
        print(f"      ⚠ {n_too_long} sequences exceeded posmean rows — prefix-centered only.")

    # --------------------------------------------------------------- #
    # Disk-space precheck
    # --------------------------------------------------------------- #
    est = estimate_output_bytes(d_centered)
    total, used, free = shutil.disk_usage(OUT.parent)
    print(f"[3/4] Disk check: need ~{est / 1e9:.2f} GB, free {free / 1e9:.2f} GB")
    if free < est * 1.5:
        print(f"❌ Insufficient disk space (want {est * 1.5 / 1e9:.2f} GB headroom).")
        sys.exit(1)

    # --------------------------------------------------------------- #
    # Save + verify
    # --------------------------------------------------------------- #
    print(f"[4/4] Saving → {OUT}")
    save_file(d_centered, str(OUT))

    print(f"      Verifying saved file...")
    chk = load_file(str(OUT))
    if len(chk) != len(d_centered):
        print(f"❌ Verification failed: saved {len(d_centered)} but reload sees {len(chk)}.")
        sys.exit(1)
    if chk[keys[0]].shape != d[keys[0]].shape:
        print(f"❌ Shape mismatch on {keys[0]}: {tuple(chk[keys[0]].shape)} vs {tuple(d[keys[0]].shape)}.")
        sys.exit(1)
    # Key-set identity (downstream depends on exact keys)
    if set(chk.keys()) != set(d.keys()):
        print(f"❌ Key set changed during centering — downstream scripts would break.")
        sys.exit(1)
    print(f"      ✓ Verified: {len(chk)} tensors, keys + shapes preserved.")

    # --------------------------------------------------------------- #
    # Sanity: at high-n_p positions, the new per-position mean should be ~0.
    # Recompute mean at a few well-covered positions across all sequences.
    # --------------------------------------------------------------- #
    well_covered = (position_count >= 1000).nonzero(as_tuple=False).flatten().tolist()
    check_positions = well_covered[:5]
    if check_positions:
        sums = {p: torch.zeros(hidden_dim, dtype=torch.float64) for p in check_positions}
        cnts = {p: 0 for p in check_positions}
        for k in keys:
            t = chk[k].to(torch.float64)
            L = t.shape[0]
            for p in check_positions:
                if p < L:
                    sums[p] += t[p]
                    cnts[p] += 1
        print(f"      Post-centering mean magnitude at well-covered positions (expect ~0):")
        for p in check_positions:
            if cnts[p] > 0:
                m = (sums[p] / cnts[p]).abs().mean().item()
                print(f"        pos {p:>4} (n={cnts[p]:>6}): mean|·| = {m:.4f}")

    with open(META, "w") as f:
        json.dump(
            {
                "src": str(SRC),
                "posmean": str(POSMEAN),
                "out": str(OUT),
                "n_sequences": len(keys),
                "hidden_dim": hidden_dim,
                "max_token_len": max_len,
                "store_dtype": "float16",
                "note": (
                    "Position-centered RNA-MSM activations: x[p] - guarded "
                    "position_mean[p]. Keys/shapes identical to the input so "
                    "08/09/12 work unchanged. This is the v3 input."
                ),
            },
            f,
            indent=2,
        )

    out_gb = OUT.stat().st_size / 1e9
    print()
    print("=" * 70)
    print("✅ Position-centering complete.")
    print("=" * 70)
    print(f"   Output: {OUT} ({out_gb:.2f} GB)")
    print(f"   Meta:   {META}")
    print()
    print("   Next steps (train v3 RNA-MSM SAE on position-centered inputs):")
    print("     uv run python set_model.py rnamsm")
    print("     uv run python 08_train_sae_big.py")
    print("     uv run python 09_inspect_features_big.py")
    print("     uv run python 12_cross_model_agreement.py")


if __name__ == "__main__":
    main()
