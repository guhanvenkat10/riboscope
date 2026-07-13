"""
09_inspect_features_big.py — RIBOSCOPE Phase 11: Streaming feature inspection (v2).

v2 changes (memory-efficient)
-----------------------------
- Top-K *within chunk first*, then merge two small [K, d_dict] buffers.
  This replaces the v1 pattern of concatenating a full [chunk, d_dict]
  index tensor (which materialized ~2 GB per chunk).
- Vectorized family tracking via a torch boolean matrix
  family_presence[d_dict, n_families]. Replaces v1's Python for-loop
  over millions of nonzero (token, feature) pairs.
- Smaller default CHUNK_SIZE (10k) — tighter per-chunk memory ceiling.
- Aggressive `del` of intermediate tensors per chunk.
- Prints memory checkpoints so you can see what's happening.

Note: requires WSL to have at least ~16 GB available. If your WSL has
the default ~50% cap, raise it via ~/.wslconfig (see PHASE 11 doc).

Run with
--------
    cd ~/projects/riboscope
    uv run python 09_inspect_features_big.py
"""

from __future__ import annotations

import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

from sae_models import BatchTopKSAE


# ============================ CONFIG ============================
LAYER = 6

# How many top features to display in detail
N_FEATURES_TO_SHOW = 30

# How many top-activating tokens to keep per feature
TOP_K_PER_FEATURE = 8

# Token-chunk size for streaming computation. Lower = less memory peak;
# higher = fewer iterations. 10k is a safe middle ground for ~32GB systems.
CHUNK_SIZE = 10_000

# Context window (nt) on each side of the focal nucleotide
CONTEXT_WINDOW = 7

# Phase 12 NEW: magnitude threshold for "fires meaningfully on a family".
# A family X is counted as fired-on by feature F only if some token in X
# has activation ≥ MAGNITUDE_THRESHOLD_FRAC * (max activation of feature F).
# This replaces the v1 metric (any positive activation), which over-counted
# tiny tail activations and made all features look like generalists.
MAGNITUDE_THRESHOLD_FRAC = 0.25

# Phase 12.1 NEW: skip CLS and EOS tokens during inspection. These special
# tokens have very high-magnitude embeddings that dominate the top-K of
# many features, drowning out real biological signals. Excluding them
# during inspection (NOT during training) makes the biological features
# rise to the top of the report. The SAE itself still uses these tokens
# during training; we just don't show them as "interesting" tokens.
SKIP_SPECIAL_TOKENS = True

# Phase 13b NEW: skip N (masked/unknown) nucleotide tokens during inspection.
# RNA-MSM was pretrained on MSAs with N gap tokens at high frequency, so its
# per-position embeddings for N tokens are distinct and dominate sparse
# decomposition. RNA-FM/ErnieRNA were trained without significant N exposure,
# so this filter is a no-op for them; safe to leave True for all models.
# Pairs with MASK_N_TOKENS in 08_train_sae_big.py — if you trained with the
# N mask on, inspect with this filter on too for consistency.
SKIP_N_TOKENS = True

# I/O paths — Phase 12 (v2)
ACT_FILE = Path("outputs/activations_big_layer6_v2.safetensors")
SAE_FILE = Path(f"outputs/sae_big_layer{LAYER}_v2.safetensors")
FASTA_FILE = Path("sequences/rfam_30k.fasta")
REPORT_OUT = Path(f"outputs/inspection_big_layer{LAYER}_v2.json")
# ================================================================


def parse_fasta_with_metadata(path: Path) -> dict:
    """Parse the Rfam FASTA produced by 06_fetch_rfam_sequences.py."""
    out: dict[str, dict] = {}
    name = None
    rfam_id = None
    rfam_name = None
    seq_buf: list[str] = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    out[name] = {
                        "sequence": "".join(seq_buf),
                        "rfam_id": rfam_id,
                        "rfam_name": rfam_name,
                    }
                parts = [p.strip() for p in line[1:].split("|")]
                name = parts[0] if len(parts) > 0 else "unknown"
                rfam_id = parts[1] if len(parts) > 1 else None
                rfam_name = parts[2] if len(parts) > 2 else None
                seq_buf = []
            else:
                seq_buf.append(line.replace("T", "U").replace("t", "u"))
        if name is not None:
            out[name] = {
                "sequence": "".join(seq_buf),
                "rfam_id": rfam_id,
                "rfam_name": rfam_name,
            }
    return out


def format_context(seq_str: str, nt_pos: int, win: int) -> str:
    if nt_pos < 0 or nt_pos >= len(seq_str):
        return "[OUT-OF-RANGE]"
    lo = max(0, nt_pos - win)
    hi = min(len(seq_str), nt_pos + win + 1)
    pre = seq_str[lo:nt_pos]
    focal = seq_str[nt_pos]
    post = seq_str[nt_pos + 1:hi]
    pre_pad = " " * (win - len(pre))
    post_pad = " " * (win - len(post))
    return f"{pre_pad}{pre}[{focal}]{post}{post_pad}"


def get_mem_gb() -> str:
    """Return current process RSS in GB, for diagnostic printing."""
    try:
        import resource
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return f"{rss_kb / 1e6:.2f} GB"
    except Exception:
        return "?"


def main() -> None:
    print("=" * 80)
    print(f"RIBOSCOPE Phase 11: Streaming feature inspection v2 (layer {LAYER})")
    print("=" * 80)

    # --------------------------------------------------------------- #
    # 1. FASTA metadata
    # --------------------------------------------------------------- #
    print(f"[1/5] Loading FASTA metadata from {FASTA_FILE}...")
    seq_meta = parse_fasta_with_metadata(FASTA_FILE)
    print(f"      Sequences: {len(seq_meta)}    [RSS: {get_mem_gb()}]")

    # --------------------------------------------------------------- #
    # 2. Activations
    # --------------------------------------------------------------- #
    print(f"[2/5] Loading activations from {ACT_FILE}...")
    raw = load_file(str(ACT_FILE))
    layer_keys = sorted(k for k in raw.keys() if k.endswith(f"__layer{LAYER}"))
    print(f"      Layer-{LAYER} tensors: {len(layer_keys)}    [RSS: {get_mem_gb()}]")

    # Build per-token metadata (light, just dicts) and the big concat
    token_meta_seq_name: list[str] = []
    token_meta_tok_pos: list[int] = []
    token_meta_nt_pos: list[int] = []
    token_meta_is_special: list[bool] = []
    token_meta_is_n: list[bool] = []  # Phase 13b NEW
    token_meta_rfam_id: list[str | None] = []

    activation_chunks: list[torch.Tensor] = []
    for key in layer_keys:
        seq_name = key.replace(f"__layer{LAYER}", "")
        chunk = raw[key]  # [seq_len, hidden_dim], fp16
        n = chunk.shape[0]
        meta = seq_meta.get(seq_name, {"sequence": "", "rfam_id": None, "rfam_name": None})
        rfam_id = meta["rfam_id"]
        seq_upper = meta["sequence"].upper() if meta["sequence"] else ""
        for pos in range(n):
            is_special = pos == 0 or pos == n - 1
            nt_pos = -1 if is_special else pos - 1
            is_n = (not is_special) and (0 <= nt_pos < len(seq_upper)) and (seq_upper[nt_pos] == "N")
            token_meta_seq_name.append(seq_name)
            token_meta_tok_pos.append(pos)
            token_meta_nt_pos.append(nt_pos)
            token_meta_is_special.append(is_special)
            token_meta_is_n.append(is_n)
            token_meta_rfam_id.append(rfam_id)
        activation_chunks.append(chunk)

    activations = torch.cat(activation_chunks, dim=0)  # [n_tokens, hidden_dim], fp16
    n_tokens, d_input = activations.shape
    del raw, activation_chunks
    gc.collect()
    print(f"      Total tokens: {n_tokens:,}    d_input: {d_input}    "
          f"[RSS: {get_mem_gb()}]")

    # --------------------------------------------------------------- #
    # 3. SAE
    # --------------------------------------------------------------- #
    print(f"[3/5] Loading SAE from {SAE_FILE}...")
    sae_state = load_file(str(SAE_FILE))
    d_dict = sae_state["W_enc"].shape[1]
    sae = BatchTopKSAE(d_input=d_input, d_dict=d_dict, k=32)
    sae.load_state_dict(sae_state)
    sae.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = sae.to(device)
    print(f"      SAE: d_input={d_input}, d_dict={d_dict}, device={device}")

    # --------------------------------------------------------------- #
    # 3b. Pre-build family integer mapping (for vectorized family tracking)
    # --------------------------------------------------------------- #
    unique_families = sorted({fid for fid in token_meta_rfam_id if fid})
    family_to_int = {f: i for i, f in enumerate(unique_families)}
    int_to_family = {i: f for f, i in family_to_int.items()}
    n_families = len(unique_families)

    token_family_int = torch.tensor(
        [family_to_int.get(fid, -1) for fid in token_meta_rfam_id],
        dtype=torch.long,
    )  # [n_tokens]; -1 = special token or no family

    # Phase 12.1 / 13b: build the "inspectable token" mask.
    # `is_real_token` is True for tokens that should be included in top-K and
    # family-tracking. Two independent filters compose into it:
    #   - SKIP_SPECIAL_TOKENS drops CLS/EOS
    #   - SKIP_N_TOKENS drops N nucleotide positions (matches v2 SAE training)
    # The chunk loop below applies the mask whenever ANY filter is enabled.
    is_special_t = torch.tensor(token_meta_is_special, dtype=torch.bool)
    is_n_t = torch.tensor(token_meta_is_n, dtype=torch.bool)
    is_real_token = torch.ones(len(token_meta_is_special), dtype=torch.bool)
    if SKIP_SPECIAL_TOKENS:
        is_real_token = is_real_token & ~is_special_t
    if SKIP_N_TOKENS:
        is_real_token = is_real_token & ~is_n_t
    apply_mask = SKIP_SPECIAL_TOKENS or SKIP_N_TOKENS  # used by chunk loop
    n_real = int(is_real_token.sum().item())
    n_special = int(is_special_t.sum().item())
    n_n_tokens = int(is_n_t.sum().item())
    print(f"      Distinct Rfam families: {n_families}    [RSS: {get_mem_gb()}]")
    print(f"      Real tokens: {n_real:,}    Special tokens: {n_special:,}    "
          f"N tokens: {n_n_tokens:,}")
    print(f"      Skip special: {SKIP_SPECIAL_TOKENS}    "
          f"Skip N: {SKIP_N_TOKENS}")

    # --------------------------------------------------------------- #
    # 4. Streaming top-K + family tracking (memory-bounded)
    # --------------------------------------------------------------- #
    print(f"[4/5] Streaming inspection in chunks of {CHUNK_SIZE} tokens...")

    # Running buffers — small ([d_dict, K])
    top_vals = torch.full((d_dict, TOP_K_PER_FEATURE), float("-inf"))
    top_idxs = torch.zeros((d_dict, TOP_K_PER_FEATURE), dtype=torch.long)

    n_active_per_feature = torch.zeros(d_dict, dtype=torch.long)

    # Phase 12 NEW: track the MAXIMUM activation per (feature, family) pair.
    # At the end of the streaming pass, we apply a per-feature magnitude
    # threshold (e.g., 25% of max) to derive the actual breadth.
    # 10240 features * ~3500 families * 4 bytes = ~140 MB. Fits in RAM.
    max_act_per_feature_family = torch.zeros((d_dict, n_families), dtype=torch.float32)

    n_chunks = (n_tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
    pbar = tqdm(range(n_chunks), unit="chunk")

    with torch.no_grad():
        for ci in pbar:
            start = ci * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, n_tokens)
            chunk_size = end - start

            # Move to GPU as fp32, encode
            chunk = activations[start:end].float().to(device, non_blocking=True)
            chunk_feats = sae.encode(chunk)  # [chunk_size, d_dict] on GPU

            # Phase 12.1 / 13b: mask uninspectable tokens before top-K so
            # CLS/EOS (and optionally N) don't dominate the top-activating-
            # tokens report. We set their activations to -inf so topk skips them.
            if apply_mask:
                chunk_is_real = is_real_token[start:end].to(device)
                # Broadcast: [chunk_size] → [chunk_size, 1] → [chunk_size, d_dict]
                mask_broadcast = chunk_is_real.unsqueeze(1)
                chunk_feats_for_topk = torch.where(
                    mask_broadcast, chunk_feats, torch.full_like(chunk_feats, float("-inf"))
                )
            else:
                chunk_feats_for_topk = chunk_feats

            # ----- Top-K within this chunk (on GPU, fast) ----- #
            k_eff = min(TOP_K_PER_FEATURE, chunk_size)
            chunk_top_vals_gpu, chunk_top_pos_gpu = chunk_feats_for_topk.topk(k_eff, dim=0)

            # Move to CPU for the merge with running buffers
            chunk_top_vals = chunk_top_vals_gpu.cpu()
            chunk_top_idxs = chunk_top_pos_gpu.cpu() + start  # global indices

            # ----- Merge with running top-K (small tensors) ----- #
            # top_vals: [d_dict, K] → top_vals.T: [K, d_dict]
            # chunk_top_vals: [k_eff, d_dict]
            merged_vals = torch.cat([top_vals.T, chunk_top_vals], dim=0)  # [K + k_eff, d_dict]
            merged_idxs = torch.cat([top_idxs.T, chunk_top_idxs], dim=0)

            new_top_vals, new_top_pos = merged_vals.topk(TOP_K_PER_FEATURE, dim=0)
            new_top_idxs = merged_idxs.gather(0, new_top_pos)
            top_vals = new_top_vals.T.contiguous()
            top_idxs = new_top_idxs.T.contiguous()

            # ----- Active count update ----- #
            chunk_feats_cpu = chunk_feats.cpu()
            del chunk_feats  # free GPU
            n_active_per_feature += (chunk_feats_cpu > 0).sum(dim=0).long()

            # Phase 12 NEW: update per-(feature, family) max activation.
            # Phase 12.1 NEW: also exclude special tokens from family tracking
            # if SKIP_SPECIAL_TOKENS is set (otherwise CLS-detector features
            # appear to fire on every family).
            chunk_fams = token_family_int[start:end]
            chunk_is_real_cpu = is_real_token[start:end] if apply_mask else None
            unique_fams_in_chunk = chunk_fams.unique()
            for fam in unique_fams_in_chunk.tolist():
                if fam < 0:
                    continue
                fam_mask = (chunk_fams == fam)
                if apply_mask:
                    fam_mask = fam_mask & chunk_is_real_cpu
                if fam_mask.any():
                    fam_max = chunk_feats_cpu[fam_mask].max(dim=0).values  # [d_dict]
                    max_act_per_feature_family[:, fam] = torch.maximum(
                        max_act_per_feature_family[:, fam], fam_max
                    )

            # Cleanup
            del chunk, chunk_feats_cpu
            del chunk_top_vals_gpu, chunk_top_pos_gpu, chunk_top_vals, chunk_top_idxs
            del merged_vals, merged_idxs, new_top_vals, new_top_pos, new_top_idxs

            if ci % 10 == 0:
                pbar.set_postfix(rss=get_mem_gb())

    print(f"      Streaming done.    [RSS: {get_mem_gb()}]")

    # --------------------------------------------------------------- #
    # 5. Report
    # --------------------------------------------------------------- #
    print(f"[5/5] Building report...")

    n_alive = int((n_active_per_feature > 0).sum().item())
    n_dead = d_dict - n_alive

    # Phase 12 NEW: magnitude-aware breadth.
    # Derive a per-feature magnitude threshold = MAGNITUDE_THRESHOLD_FRAC * max,
    # then count families that have at least one token at or above that threshold.
    max_per_feature = max_act_per_feature_family.max(dim=1).values  # [d_dict]
    threshold_per_feature = max_per_feature * MAGNITUDE_THRESHOLD_FRAC  # [d_dict]
    above_threshold = max_act_per_feature_family >= threshold_per_feature.unsqueeze(1)
    family_presence = above_threshold  # [d_dict, n_families] bool
    breadth = family_presence.sum(dim=1).tolist()

    # Also report breadth at the v1 (>0) threshold for comparison
    breadth_loose = (max_act_per_feature_family > 0).sum(dim=1).tolist()

    n_specialists = sum(1 for b in breadth if 1 <= b <= 2)
    n_moderate = sum(1 for b in breadth if 3 <= b <= 9)
    n_generalists = sum(1 for b in breadth if b >= 10)
    n_zero_breadth = sum(1 for b in breadth if b == 0)
    print(f"      Magnitude threshold:    {MAGNITUDE_THRESHOLD_FRAC:.0%} of per-feature max")

    print(f"\n      Alive features:                {n_alive} / {d_dict}  "
          f"({100 * n_alive / d_dict:.1f}%)")
    print(f"      Dead features:                 {n_dead} / {d_dict}  "
          f"({100 * n_dead / d_dict:.1f}%)")
    print(f"      Specialists (≤2 families):     {n_specialists}")
    print(f"      Moderate (3-9 families):       {n_moderate}")
    print(f"      Generalists (≥10 families):    {n_generalists}")
    print(f"      Below-threshold (0 families):  {n_zero_breadth}  "
          f"(features that never reached threshold magnitude in any family)")

    # Top-N features by max activation
    max_per_feature = top_vals[:, 0]
    selected = max_per_feature.topk(N_FEATURES_TO_SHOW)
    selected_idxs = selected.indices.tolist()
    selected_vals = selected.values.tolist()

    print(f"\n      Top {N_FEATURES_TO_SHOW} features by max activation strength:")
    print("=" * 80)

    report_data: list[dict] = []
    for rank, (feat_idx, max_act) in enumerate(zip(selected_idxs, selected_vals), start=1):
        feat_breadth = int(breadth[feat_idx])
        feat_top_vals = top_vals[feat_idx].tolist()
        feat_top_idxs = top_idxs[feat_idx].tolist()

        feature_record = {
            "rank": rank,
            "feature_idx": feat_idx,
            "max_activation": float(max_act),
            "n_active_tokens": int(n_active_per_feature[feat_idx].item()),
            "n_families": feat_breadth,
            "top_tokens": [],
        }

        print(f"\nFeature #{feat_idx:5d}  "
              f"max={max_act:7.3f}  "
              f"active_on_{n_active_per_feature[feat_idx]}_tokens  "
              f"fires_on_{feat_breadth}_families  "
              f"(rank {rank}/{N_FEATURES_TO_SHOW})")
        print(f"{'  rk':<4}{'act':>8}  {'rfam':<10}{'sequence':<22}{'tok':>5}{'nt':>5}  context (±{CONTEXT_WINDOW})")
        print(f"  {'-' * 2}  {'-' * 7}  {'-' * 9}  {'-' * 21}  {'-' * 4} {'-' * 4}  {'-' * (2 * CONTEXT_WINDOW + 3)}")

        for r, (tok_idx, act) in enumerate(zip(feat_top_idxs, feat_top_vals), start=1):
            if act == float("-inf"):
                continue
            seq_name = token_meta_seq_name[tok_idx]
            tok_pos = token_meta_tok_pos[tok_idx]
            nt_pos = token_meta_nt_pos[tok_idx]
            is_special = token_meta_is_special[tok_idx]
            rfam_id = token_meta_rfam_id[tok_idx] or "?"
            if is_special:
                context = "  [SPECIAL TOKEN — CLS or EOS]"
            else:
                seq_str = seq_meta.get(seq_name, {}).get("sequence", "")
                context = format_context(seq_str, nt_pos, CONTEXT_WINDOW)

            print(f"  {r:>2}  {act:>7.3f}  {rfam_id:<9}  "
                  f"{seq_name:<21}  {tok_pos:>4} {nt_pos:>4}  {context}")

            feature_record["top_tokens"].append({
                "rank": r,
                "activation": float(act),
                "rfam_id": rfam_id,
                "seq_name": seq_name,
                "tok_pos": tok_pos,
                "nt_pos": nt_pos,
                "is_special": is_special,
                "context": context,
            })

        # Top-8 family list for this feature
        if feat_breadth > 0:
            fam_int_indices = family_presence[feat_idx].nonzero(as_tuple=False).flatten().tolist()
            fam_list = [int_to_family[i] for i in fam_int_indices][:8]
            print(f"     fires_on_families: {fam_list}{' ...' if feat_breadth > 8 else ''}")
            feature_record["families_sample"] = fam_list

        report_data.append(feature_record)

    # --------------------------------------------------------------- #
    # 5b. Phase 13 ADDED: dump top specialists and top moderate features
    #     ranked by max activation *within their breadth bucket*.
    #     This surfaces the biology buried below the position-detector
    #     and N-token artifacts that dominate the absolute top-N.
    # --------------------------------------------------------------- #
    def _dump_bucket(
        bucket_name: str,
        bucket_predicate,
        n_show: int,
    ) -> list[dict]:
        breadth_t = torch.tensor(breadth)
        active_mask = n_active_per_feature > 0
        bucket_mask = bucket_predicate(breadth_t) & active_mask
        bucket_idxs_all = bucket_mask.nonzero(as_tuple=False).flatten()
        if len(bucket_idxs_all) == 0:
            print(f"\n      No features matched bucket: {bucket_name}")
            return []
        bucket_maxes = top_vals[:, 0][bucket_idxs_all]
        k_eff = min(n_show, len(bucket_maxes))
        bk = bucket_maxes.topk(k_eff)
        sel_idxs = bucket_idxs_all[bk.indices].tolist()
        sel_vals = bk.values.tolist()

        print(f"\n      Top {k_eff} {bucket_name} by max activation:")
        print("=" * 80)
        out: list[dict] = []
        for rank, (feat_idx, max_act) in enumerate(zip(sel_idxs, sel_vals), start=1):
            feat_breadth = int(breadth[feat_idx])
            feat_top_vals = top_vals[feat_idx].tolist()
            feat_top_idxs = top_idxs[feat_idx].tolist()
            rec: dict = {
                "rank": rank,
                "feature_idx": int(feat_idx),
                "max_activation": float(max_act),
                "n_active_tokens": int(n_active_per_feature[feat_idx].item()),
                "n_families": feat_breadth,
                "bucket": bucket_name,
                "top_tokens": [],
            }
            print(f"\n{bucket_name} Feature #{int(feat_idx):5d}  "
                  f"max={max_act:7.3f}  "
                  f"active_on_{n_active_per_feature[feat_idx]}_tokens  "
                  f"fires_on_{feat_breadth}_families  "
                  f"(rank {rank}/{k_eff})")
            print(f"{'  rk':<4}{'act':>8}  {'rfam':<10}{'sequence':<22}{'tok':>5}{'nt':>5}  context (±{CONTEXT_WINDOW})")
            print(f"  {'-' * 2}  {'-' * 7}  {'-' * 9}  {'-' * 21}  {'-' * 4} {'-' * 4}  {'-' * (2 * CONTEXT_WINDOW + 3)}")
            for r, (tok_idx, act) in enumerate(zip(feat_top_idxs, feat_top_vals), start=1):
                if act == float("-inf"):
                    continue
                seq_name = token_meta_seq_name[tok_idx]
                tok_pos = token_meta_tok_pos[tok_idx]
                nt_pos = token_meta_nt_pos[tok_idx]
                is_special = token_meta_is_special[tok_idx]
                rfam_id = token_meta_rfam_id[tok_idx] or "?"
                if is_special:
                    context = "  [SPECIAL TOKEN — CLS or EOS]"
                else:
                    seq_str = seq_meta.get(seq_name, {}).get("sequence", "")
                    context = format_context(seq_str, nt_pos, CONTEXT_WINDOW)
                print(f"  {r:>2}  {act:>7.3f}  {rfam_id:<9}  "
                      f"{seq_name:<21}  {tok_pos:>4} {nt_pos:>4}  {context}")
                rec["top_tokens"].append({
                    "rank": r,
                    "activation": float(act),
                    "rfam_id": rfam_id,
                    "seq_name": seq_name,
                    "tok_pos": tok_pos,
                    "nt_pos": nt_pos,
                    "is_special": is_special,
                    "context": context,
                })
            if feat_breadth > 0:
                fam_int_indices = family_presence[feat_idx].nonzero(as_tuple=False).flatten().tolist()
                fam_list = [int_to_family[i] for i in fam_int_indices][:8]
                print(f"     fires_on_families: {fam_list}{' ...' if feat_breadth > 8 else ''}")
                rec["families_sample"] = fam_list
            out.append(rec)
        return out

    specialists_data = _dump_bucket(
        "SPECIALIST(≤2-fam)",
        lambda b: (b >= 1) & (b <= 2),
        N_FEATURES_TO_SHOW,
    )
    moderate_data = _dump_bucket(
        "MODERATE(3-9-fam)",
        lambda b: (b >= 3) & (b <= 9),
        N_FEATURES_TO_SHOW,
    )

    # Save JSON
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUT, "w") as f:
        json.dump({
            "summary": {
                "n_alive": n_alive,
                "n_dead": n_dead,
                "n_specialists_le2_families": n_specialists,
                "n_moderate_3to9_families": n_moderate,
                "n_generalists_ge10_families": n_generalists,
                "n_below_threshold_zero_families": n_zero_breadth,
                "magnitude_threshold_frac": MAGNITUDE_THRESHOLD_FRAC,
                "n_features_total": d_dict,
                "n_tokens_inspected": n_tokens,
                "n_distinct_families_in_data": n_families,
            },
            "top_features": report_data,
            "specialist_features": specialists_data,
            "moderate_features": moderate_data,
        }, f, indent=2)

    print()
    print("=" * 80)
    print(f"✅ Inspection complete.")
    print(f"   Console report:   above")
    print(f"   JSON report:      {REPORT_OUT}")
    print(f"   Final RSS:        {get_mem_gb()}")
    print()
    print(f"   Phase 12 reading guide:")
    print(f"   - Magnitude threshold: feature fires meaningfully on a family only if")
    print(f"     some token in that family has activation ≥ {MAGNITUDE_THRESHOLD_FRAC:.0%} of the")
    print(f"     feature's overall max activation. This corrects v1's overcounting.")
    print("   - True specialists (≤2 families) are now meaningful — features that have")
    print("     learned family-specific motifs (snoRNA C-box, Y-RNA bulge, riboswitch")
    print("     aptamers, etc.).")
    print("   - Generalists (≥10 families) likely encode broadly biological patterns")
    print("     (poly-U terminators, GC-rich stems, AU-rich elements, position markers).")
    print("   - Below-threshold features have no strong family signal — noise-dominated;")
    print("     many will become productive when we scale to more tokens.")


if __name__ == "__main__":
    main()
