"""
35_snorna_rnamsm_features.py — RIBOSCOPE steps R3+R4: preprocess snoRNA RNA-MSM
activations with the EXACT training normalization, then SAE-encode.

The locked RNA-MSM-MSA SAE was trained on activations normalized by 18_prep:
global z-score (training mean/std) then guarded per-position-mean subtraction.
To apply that SAE to NEW snoRNA activations in-distribution we must use the
TRAINING statistics, not stats recomputed on the snoRNAs.

The training position-mean tensor wasn't saved, so we recompute it from the raw
training activations (18's exact math). To prove the recomputation is correct, we
first REBUILD the training poscentered file and assert it matches the locked one
(activations_rnamsm_msa_layer8_poscentered.safetensors). Only if that gate passes
do we preprocess the snoRNA activations and encode them.

Run with
--------
    cd ~/projects/riboscope
    uv run python 35_snorna_rnamsm_features.py

Inputs : activations_rnamsm_msa_layer8.safetensors            (raw training acts)
         activations_rnamsm_msa_layer8_poscentered.safetensors (locked, for the gate)
         activations_rnamsm_msa_layer8_normstats.json          (training mean/std)
         activations_rnamsm_msa_snorna_layer8.safetensors      (raw snoRNA acts, from R2)
         sae_rnamsm_msa_layer8_v1.safetensors                  (locked SAE)
Output : outputs/snodb_cd_features_rnamsm.safetensors  (sae_max, embed_mean)
         outputs/snodb_cd_rnamsm_meta.tsv               (snodb_id per matrix row)
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file, save_file, safe_open
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

from sae_models import BatchTopKSAE

TRAIN_RAW = Path("outputs/activations_rnamsm_msa_layer8.safetensors")
TRAIN_POSC = Path("outputs/activations_rnamsm_msa_layer8_poscentered.safetensors")
NORMSTATS = Path("outputs/activations_rnamsm_msa_layer8_normstats.json")
SNO_RAW = Path("outputs/activations_rnamsm_msa_snorna_layer8.safetensors")
SAE_FILE = Path("outputs/sae_rnamsm_msa_layer8_v1.safetensors")
OUT_FEATS = Path("outputs/snodb_cd_features_rnamsm.safetensors")
OUT_META = Path("outputs/snodb_cd_rnamsm_meta.tsv")
MIN_CONTRIBUTORS = 100
GATE_SAMPLE = 60
GATE_TOL = 2e-3   # fp16 round-trip tolerance


def compute_position_mean(d_norm: dict):
    max_len = max(int(v.shape[0]) for v in d_norm.values())
    hidden = int(next(iter(d_norm.values())).shape[1])
    pos_sum = torch.zeros((max_len, hidden), dtype=torch.float64)
    pos_count = torch.zeros(max_len, dtype=torch.int64)
    for v in d_norm.values():
        t = v.to(torch.float64)
        L = t.shape[0]
        pos_sum[:L] += t
        pos_count[:L] += 1
    safe = pos_count.clamp(min=1).unsqueeze(1).to(torch.float64)
    pm = pos_sum / safe
    pm[pos_count == 0] = 0.0
    pm[pos_count < MIN_CONTRIBUTORS] = 0.0
    return pm.to(torch.float32)


def main():
    for p in (TRAIN_RAW, TRAIN_POSC, NORMSTATS, SNO_RAW, SAE_FILE):
        if not p.exists():
            print(f"❌ Missing {p}")
            sys.exit(1)

    print("=" * 74)
    print("RIBOSCOPE R3+R4: snoRNA RNA-MSM preprocessing (training stats) + SAE")
    print("=" * 74)

    stats = json.loads(NORMSTATS.read_text())
    mean, std = float(stats["mean"]), float(stats["std"])
    print(f"[1/5] Training z-score stats: mean={mean:+.6f} std={std:.6f}")

    print("[2/5] Recomputing training position-mean from raw training acts ...")
    train_raw = load_file(str(TRAIN_RAW))
    d_norm = {k: ((v.to(torch.float32) - mean) / std).to(torch.float16) for k, v in train_raw.items()}
    del train_raw
    pm32 = compute_position_mean(d_norm)
    pm_len = pm32.shape[0]
    print(f"      position-mean shape: {tuple(pm32.shape)}")

    # ---- GATE: rebuild training poscentered, compare to locked ----
    print(f"[3/5] GATE: verifying reproduction on {GATE_SAMPLE} training sequences ...")
    keys = sorted(d_norm.keys())
    sample = keys[:: max(1, len(keys) // GATE_SAMPLE)][:GATE_SAMPLE]
    max_abs = 0.0
    with safe_open(str(TRAIN_POSC), framework="pt") as f:
        locked_keys = set(f.keys())
        for k in sample:
            if k not in locked_keys:
                continue
            t = d_norm[k].to(torch.float32)
            L = t.shape[0]
            lim = min(L, pm_len)
            t[:lim] -= pm32[:lim]
            mine = t.to(torch.float16)
            locked = f.get_tensor(k)
            max_abs = max(max_abs, float((mine.float() - locked.float()).abs().max().item()))
    print(f"      max abs diff vs locked poscentered: {max_abs:.2e}  (tol {GATE_TOL:.0e})")
    if max_abs > GATE_TOL:
        print("      ❌ GATE FAILED — preprocessing does NOT reproduce the locked file. STOP.")
        sys.exit(1)
    print("      ✓ GATE PASSED — preprocessing matches training. Safe to apply to snoRNAs.")
    del d_norm

    # ---- preprocess snoRNA acts with the SAME stats ----
    print("[4/5] Preprocessing snoRNA acts + SAE-encoding ...")
    sae_state = load_file(str(SAE_FILE))
    d_input, d_dict = sae_state["W_enc"].shape[0], sae_state["W_enc"].shape[1]
    sae = BatchTopKSAE(d_input=d_input, d_dict=d_dict, k=32)
    sae.load_state_dict(sae_state)
    sae.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = sae.to(device)

    sno = load_file(str(SNO_RAW))
    ids = sorted(k.replace("__layer8", "") for k in sno)
    sae_max_rows, embed_rows, kept_ids = [], [], []
    with torch.no_grad():
        for sid in ids:
            raw = sno[f"{sid}__layer8"].to(torch.float32)     # [Ltok, 768]
            Lt = raw.shape[0]
            if Lt <= 2:
                continue
            zn = (raw - mean) / std
            lim = min(Lt, pm_len)
            pc = zn.clone()
            pc[:lim] -= pm32[:lim]
            feats = sae.encode(pc.to(device))                 # [Ltok, d_dict]
            real = feats[1:Lt - 1]                            # drop CLS/EOS
            sae_max_rows.append(real.max(dim=0).values.cpu())
            embed_rows.append(raw[1:Lt - 1].mean(dim=0))      # raw embedding baseline
            kept_ids.append(sid)

    save_file({"sae_max": torch.stack(sae_max_rows).contiguous(),
               "embed_mean": torch.stack(embed_rows).contiguous()}, str(OUT_FEATS))
    with open(OUT_META, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["row", "snodb_id"])
        for i, sid in enumerate(kept_ids):
            w.writerow([i, sid])

    print(f"[5/5] Saved {OUT_FEATS}  (sae_max [{len(kept_ids)},{d_dict}], embed_mean [{len(kept_ids)},{d_input}])")
    print(f"      and {OUT_META} (snodb_id per row).")
    print("\n✓ RNA-MSM snoRNA features ready. Next: R5 — 3-model functional axis + agreement.")
    print("  Run ~/projects/riboscope/sync_to_windows.sh.")


if __name__ == "__main__":
    main()
