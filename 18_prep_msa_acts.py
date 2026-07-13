"""
18_prep_msa_acts.py — RIBOSCOPE: preprocess RNA-MSM *MSA* activations for the SAE.

Why this exists
---------------
17_extract_rnamsm_msa.py produced RNA-MSM activations under its NATIVE MSA input
(outputs/activations_rnamsm_msa_layer{N}.safetensors). Before the SAE can train
on them they need the EXACT SAME preprocessing the single-sequence v3 path used,
so that the only difference between the failed single-seq v3 SAE and this MSA SAE
is the input distribution (single seq -> in-distribution MSA), not preprocessing.

The single-seq v3 recipe was three scripts run in sequence:
    normalize_rnamsm.py        (global z-score: (x - mean)/std)
    compute_position_means.py  (per-position mean, MIN_CONTRIBUTORS guard)
    subtract_position_means.py (x[p] - guarded position_mean[p])
This script does all three, in one pass, on the MSA file for a chosen layer,
replicating their math BIT-FOR-BIT (same fp64 accumulators, same fp16 store
between stages, same MIN_CONTRIBUTORS=100 guard, same ANOVA diagnostic).

It is parameterized by layer so we can prep the primary layer (8 = most
de-collapsed per 16_msa_geometry_scan.py) and, optionally, a same-layer control
(6 = the layer the other two models' SAEs use).

What this script does
---------------------
  [1] Load outputs/activations_rnamsm_msa_layer{N}.safetensors.
  [2] Global z-score: fp64 accumulate sum_x, sum_x2 over ALL elements;
      mean = sum_x/n; std = sqrt(max(sum_x2/n - mean^2, 0)); store fp16.
      (Identical to normalize_rnamsm.py. Save normstats json.)
  [3] Per-position mean from the z-scored fp16 values cast to fp64; guarded by
      MIN_CONTRIBUTORS=100; plus the variance-explained-by-position ANOVA.
      (Identical to compute_position_means.py.)
  [4] Subtract guarded position_mean[:L] per sequence (fp32 math, fp16 store);
      preserve EXACT keys + shapes so 08/09 read it unchanged.
      (Identical to subtract_position_means.py.)
  [5] Disk precheck, post-save verify (count + shape + key-set identity), and a
      post-centering sanity check that well-covered positions now have ~0 mean.

Run with
--------
    cd ~/projects/riboscope
    uv run python 18_prep_msa_acts.py 8        # primary: layer 8
    uv run python 18_prep_msa_acts.py 6        # optional control: layer 6
    uv run python 18_prep_msa_acts.py 8 --overwrite

Output (for layer N)
--------------------
    outputs/activations_rnamsm_msa_layer{N}_poscentered.safetensors   <- SAE input
    outputs/activations_rnamsm_msa_layer{N}_normstats.json
    outputs/activations_rnamsm_msa_layer{N}_poscentered_meta.json

Then:
    uv run python set_model.py rnamsm_msa        # (rnamsm_msa_l6 for the control)
    uv run python 08_train_sae_big.py
    uv run python 09_inspect_features_big.py
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


# ============================ CONFIG ============================
DEFAULT_LAYER = 8

# Contributor guard — identical to compute_position_means.py. Only subtract a
# per-position mean estimated from >= this many sequences; below it the estimate
# is noise-dominated and the position is left untouched (mean -> 0 there).
MIN_CONTRIBUTORS = 100
# ================================================================


def src_path(layer: int, stem: str = "msa") -> Path:
    return Path(f"outputs/activations_rnamsm_{stem}_layer{layer}.safetensors")


def out_path(layer: int, stem: str = "msa") -> Path:
    return Path(f"outputs/activations_rnamsm_{stem}_layer{layer}_poscentered.safetensors")


def normstats_path(layer: int, stem: str = "msa") -> Path:
    return Path(f"outputs/activations_rnamsm_{stem}_layer{layer}_normstats.json")


def meta_path(layer: int, stem: str = "msa") -> Path:
    return Path(f"outputs/activations_rnamsm_{stem}_layer{layer}_poscentered_meta.json")


def estimate_output_bytes(d: dict) -> int:
    return sum(t.numel() * 2 for t in d.values())   # fp16 store -> 2 bytes/elem


def parse_args(argv: list[str]) -> tuple[int, bool, str]:
    overwrite = "--overwrite" in argv
    # --stem <name> selects which activation family to prep:
    #   "msa"    (default) -> activations_rnamsm_msa_layer{N}      (the MSA run)
    #   "single"           -> activations_rnamsm_single_layer{N}   (matched control)
    stem = "msa"
    if "--stem" in argv:
        i = argv.index("--stem")
        if i + 1 >= len(argv):
            print(f"❌ --stem requires a value (e.g. --stem single)")
            sys.exit(1)
        stem = argv[i + 1]
    flag_values = {"--stem"}
    skip = set()
    for f in flag_values:
        if f in argv:
            skip.add(argv.index(f) + 1)
    positional = [a for i, a in enumerate(argv)
                  if not a.startswith("--") and i not in skip]
    if len(positional) == 0:
        layer = DEFAULT_LAYER
    elif len(positional) == 1:
        try:
            layer = int(positional[0])
        except ValueError:
            print(f"❌ layer must be an integer, got {positional[0]!r}")
            print(f"   Usage: uv run python 18_prep_msa_acts.py [LAYER] [--stem msa|single] [--overwrite]")
            sys.exit(1)
    else:
        print(f"❌ Too many positional args: {positional}")
        print(f"   Usage: uv run python 18_prep_msa_acts.py [LAYER] [--stem msa|single] [--overwrite]")
        sys.exit(1)
    return layer, overwrite, stem


def main() -> None:
    layer, overwrite, stem = parse_args(sys.argv[1:])

    SRC = src_path(layer, stem)
    OUT = out_path(layer, stem)
    NORMSTATS = normstats_path(layer, stem)
    META = meta_path(layer, stem)

    print("=" * 74)
    print(f"RIBOSCOPE: RNA-MSM activation preprocessing for SAE (stem={stem}, layer {layer})")
    print(f"  global z-score -> per-position mean (guard={MIN_CONTRIBUTORS}) -> subtract")
    print("=" * 74)

    if not SRC.exists():
        print(f"❌ {SRC} not found.")
        if stem == "msa":
            print(f"   Run 17_extract_rnamsm_msa.py first (it writes layer 6 and 8).")
        elif stem == "single":
            print(f"   Run 19_extract_rnamsm_single_subset.py first (it writes layer 6 and 8).")
        sys.exit(1)
    if OUT.exists() and not overwrite:
        print(f"⚠ {OUT} already exists.")
        print(f"  Refusing to overwrite. Re-run with --overwrite to regenerate.")
        sys.exit(1)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------- #
    # Load
    # --------------------------------------------------------------- #
    print(f"[1/5] Loading {SRC}...")
    d = load_file(str(SRC))
    keys = sorted(d.keys())
    if not keys:
        print(f"❌ No tensors found in {SRC}.")
        sys.exit(1)
    sample_shape = d[keys[0]].shape
    if len(sample_shape) != 2:
        print(f"❌ Expected 2D tensors [seq_len, hidden], got {tuple(sample_shape)}.")
        sys.exit(1)
    hidden_dim = int(sample_shape[1])
    print(f"      {len(keys)} tensors, hidden_dim={hidden_dim}")

    # --------------------------------------------------------------- #
    # [2] Global z-score  (identical math to normalize_rnamsm.py)
    # --------------------------------------------------------------- #
    print(f"[2/5] Global z-score (fp64 accumulators over all elements)...")
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
    print(f"      mean       = {mean:+.6f}")
    print(f"      std        = {std:.6f}")
    print(f"      n_elements = {n_total:,}")
    if std < 1e-6:
        print(f"❌ std is essentially zero — refusing to normalize (divide by zero).")
        sys.exit(1)

    # Build z-scored dict, fp16 store (so position-mean is computed from the SAME
    # fp16 values the single-seq v3 path used — bit-for-bit recipe match).
    d_norm: dict[str, torch.Tensor] = {}
    for k in keys:
        t = d[k].to(torch.float32)
        d_norm[k] = ((t - mean) / std).to(torch.float16)
    del d

    with open(NORMSTATS, "w") as f:
        json.dump(
            {
                "layer": layer,
                "mean": mean,
                "std": std,
                "n_elements": n_total,
                "note": (
                    "Apply (x - mean)/std to convert raw->normalized; x*std+mean to "
                    "invert. Same global-affine z-score as normalize_rnamsm.py."
                ),
            },
            f,
            indent=2,
        )

    # --------------------------------------------------------------- #
    # [3] Per-position mean + guard  (identical to compute_position_means.py)
    # --------------------------------------------------------------- #
    print(f"[3/5] Per-position mean (fp64) + ANOVA diagnostic...")
    max_len = max(int(d_norm[k].shape[0]) for k in keys)
    print(f"      max token length = {max_len}")

    pos_sum = torch.zeros((max_len, hidden_dim), dtype=torch.float64)   # S_p
    pos_count = torch.zeros(max_len, dtype=torch.int64)                 # n_p
    global_sumsq = 0.0
    n_tokens_total = 0
    for k in keys:
        t = d_norm[k].to(torch.float64)        # [L, hidden]
        L = t.shape[0]
        pos_sum[:L] += t
        pos_count[:L] += 1
        global_sumsq += float((t * t).sum().item())
        n_tokens_total += L

    global_sum = pos_sum.sum(dim=0)                       # [hidden]

    safe_count = pos_count.clamp(min=1).unsqueeze(1).to(torch.float64)
    pos_mean_raw = pos_sum / safe_count
    pos_mean_raw[pos_count == 0] = 0.0

    guard_keep = pos_count >= MIN_CONTRIBUTORS
    position_mean = pos_mean_raw.clone()
    position_mean[~guard_keep] = 0.0

    n_kept = int(guard_keep.sum().item())
    n_zeroed = int((~guard_keep).sum().item())
    print(f"      positions n_p >= {MIN_CONTRIBUTORS}: {n_kept} (subtracted)")
    print(f"      positions n_p <  {MIN_CONTRIBUTORS}: {n_zeroed} (left untouched)")
    if n_kept > 0:
        kept_idx = guard_keep.nonzero(as_tuple=False).flatten()
        print(f"      kept position range: {int(kept_idx.min())}..{int(kept_idx.max())}")

    # variance-explained-by-position (one-way ANOVA over position, unguarded means)
    correction = float((global_sum * global_sum).sum().item()) / max(n_tokens_total, 1)
    ss_total = global_sumsq - correction
    pc = pos_count.clamp(min=1).to(torch.float64)
    ss_between = float(((pos_sum * pos_sum).sum(dim=1) / pc).sum().item()) - correction
    var_fraction = ss_between / ss_total if ss_total > 0 else 0.0
    print(f"      variance explained by position = {var_fraction:.4f}")
    print(f"        (single-seq v3 had a HIGH value here; MSA input should be lower)")

    # --------------------------------------------------------------- #
    # [4] Subtract guarded mean  (identical to subtract_position_means.py)
    # --------------------------------------------------------------- #
    print(f"[4/5] Subtracting per-position mean (preserving keys/shapes)...")
    pm32 = position_mean.to(torch.float32)
    d_out: dict[str, torch.Tensor] = {}
    for k in keys:
        t = d_norm[k].to(torch.float32)        # [L, hidden]
        L = t.shape[0]
        t = t - pm32[:L]
        d_out[k] = t.to(torch.float16)

    # --------------------------------------------------------------- #
    # Disk precheck
    # --------------------------------------------------------------- #
    est = estimate_output_bytes(d_out)
    _, _, free = shutil.disk_usage(OUT.parent)
    print(f"[5/5] Disk check: need ~{est / 1e9:.2f} GB, free {free / 1e9:.2f} GB")
    if free < est * 1.5:
        print(f"❌ Insufficient disk space (want {est * 1.5 / 1e9:.2f} GB headroom).")
        sys.exit(1)

    print(f"      Saving → {OUT}")
    save_file(d_out, str(OUT))

    # --------------------------------------------------------------- #
    # Post-save verify: count + shape + key-set identity
    # --------------------------------------------------------------- #
    print(f"      Verifying saved file...")
    chk = load_file(str(OUT))
    if len(chk) != len(d_out):
        print(f"❌ Verify failed: saved {len(d_out)} but reload sees {len(chk)}.")
        sys.exit(1)
    if chk[keys[0]].shape != d_out[keys[0]].shape:
        print(f"❌ Shape mismatch on {keys[0]}.")
        sys.exit(1)
    if set(chk.keys()) != set(d_norm.keys()):
        print(f"❌ Key set changed during preprocessing — downstream would break.")
        sys.exit(1)
    print(f"      ✓ Verified: {len(chk)} tensors, keys + shapes preserved.")

    # --------------------------------------------------------------- #
    # Sanity: at high-n_p positions the new per-position mean should be ~0.
    # --------------------------------------------------------------- #
    well_covered = (pos_count >= 1000).nonzero(as_tuple=False).flatten().tolist()
    if not well_covered:  # MSA set is smaller; fall back to the guard threshold
        well_covered = guard_keep.nonzero(as_tuple=False).flatten().tolist()
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
                "layer": layer,
                "src": str(SRC),
                "out": str(OUT),
                "normstats": str(NORMSTATS),
                "n_sequences": len(keys),
                "hidden_dim": hidden_dim,
                "max_token_len": max_len,
                "n_tokens_total": n_tokens_total,
                "min_contributors": MIN_CONTRIBUTORS,
                "n_positions_subtracted": n_kept,
                "n_positions_untouched": n_zeroed,
                "variance_explained_by_position": var_fraction,
                "global_mean": mean,
                "global_std": std,
                "store_dtype": "float16",
                "stem": stem,
                "note": (
                    f"RNA-MSM '{stem}' activations preprocessed with the EXACT single-seq "
                    "v3 recipe (global z-score then guarded per-position-mean subtraction). "
                    "stem='msa' = native-MSA input; stem='single' = single-sequence "
                    "(msa_depth=1) control over the SAME 2327-seq subset. The two differ "
                    "ONLY in input distribution. Keys/shapes identical to the input; "
                    "drop-in for 08/09."
                ),
            },
            f,
            indent=2,
        )

    out_gb = OUT.stat().st_size / 1e9
    print()
    print("=" * 74)
    print(f"✅ MSA preprocessing complete (layer {layer}).")
    print("=" * 74)
    print(f"   SAE input: {OUT} ({out_gb:.2f} GB)")
    print(f"   Normstats: {NORMSTATS}")
    print(f"   Meta:      {META}")
    print(f"   variance explained by position = {var_fraction:.4f}")
    print()
    if stem == "msa":
        cfg = "rnamsm_msa" if layer == 8 else ("rnamsm_msa_l6" if layer == 6 else f"<add a set_model config for msa layer {layer}>")
    elif stem == "single":
        cfg = "rnamsm_single" if layer == 8 else ("rnamsm_single_l6" if layer == 6 else f"<add a set_model config for single layer {layer}>")
    else:
        cfg = f"<add a set_model config for stem={stem} layer {layer}>"
    print("   Next steps:")
    print(f"     uv run python set_model.py {cfg}")
    print(f"     uv run python 08_train_sae_big.py        # retrain (~2 hr)")
    print(f"     uv run python 09_inspect_features_big.py # inspect specialists (~10 min)")


if __name__ == "__main__":
    main()
