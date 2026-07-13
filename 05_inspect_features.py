"""
05_inspect_features.py — RIBOSCOPE Phase 10: Look inside the trained SAE.

What this does
--------------
Takes the SAE trained in Phase 9 and answers, for each interesting feature:
  "Which (sequence, position) pairs make this feature fire most strongly?"

This is the foundational interpretability operation. Once you know what
inputs activate a feature, you start to know what the feature represents.

What we report
--------------
1. Summary stats:
   - How many features are 'alive' (fire on at least one token)?
   - How many fire on only 1 sequence (specialists) vs many sequences
     (generalists)?
   - Distribution of max-activation magnitudes across features.
2. Top-N features by max activation strength.
3. For each shown feature: top-5 activating tokens with sequence name,
   nucleotide position, and a ±5-nt context window.

Important caveat (read this!)
-----------------------------
The SAE was trained on 736 tokens — far too few for biological features
to emerge. Most features will be SEQUENCE-SPECIFIC (a feature that only
fires on tokens from one of our 12 test sequences). That's overfitting.
The point of Phase 10 is NOT to discover biology — it's to verify the
inspection workflow works end-to-end so we can use it in Phase 11+ when
we have real data.

Run with
--------
    cd ~/projects/riboscope
    uv run python 05_inspect_features.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

from sae_models import BatchTopKSAE


# ============================ CONFIG ============================
LAYER = 6

# How many features to display in the per-feature top-token report
N_FEATURES_TO_SHOW = 15

# How many top-activating tokens to show per feature
TOP_TOKENS_PER_FEATURE = 5

# Window size (in nucleotides) on each side of the activating token
CONTEXT_WINDOW = 5

# I/O paths
ACT_FILE = Path("outputs/activations_rnafm_test.safetensors")
SAE_FILE = Path(f"outputs/sae_layer{LAYER}_v1.safetensors")
FASTA_FILE = Path("sequences/test_rnas.fasta")
# ================================================================


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    """Identical to the Phase 8 parser — small enough to duplicate."""
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found at {path}")
    sequences: list[tuple[str, str]] = []
    name: str | None = None
    seq: list[str] = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    sequences.append((name, "".join(seq)))
                name = line[1:].split("|")[0].strip()
                seq = []
            else:
                seq.append(line.replace("T", "U").replace("t", "u"))
        if name is not None:
            sequences.append((name, "".join(seq)))
    return sequences


def format_context(seq_str: str, nt_pos: int, win: int) -> str:
    """
    Make a ±win nucleotide context string with the focal nucleotide
    bracketed. nt_pos is 0-indexed into seq_str (i.e., the actual
    nucleotide index, NOT the token index).
    """
    if nt_pos < 0 or nt_pos >= len(seq_str):
        return "[OUT-OF-RANGE]"
    lo = max(0, nt_pos - win)
    hi = min(len(seq_str), nt_pos + win + 1)
    pre = seq_str[lo:nt_pos]
    focal = seq_str[nt_pos]
    post = seq_str[nt_pos + 1:hi]
    # Pad pre/post so the bracket lines up visually across rows
    pre_pad = " " * (win - len(pre))
    post_pad = " " * (win - len(post))
    return f"{pre_pad}{pre}[{focal}]{post}{post_pad}"


def main() -> None:
    print("=" * 70)
    print(f"RIBOSCOPE Phase 10: Inspect SAE features (layer {LAYER})")
    print("=" * 70)

    # --------------------------------------------------------------- #
    # 1. Load activations + sequences
    # --------------------------------------------------------------- #
    print(f"[1/5] Loading activations from {ACT_FILE}...")
    if not ACT_FILE.exists():
        print(f"❌ Run Phase 8 first.")
        sys.exit(1)

    all_acts = load_file(str(ACT_FILE))
    sequences = parse_fasta(FASTA_FILE)
    seq_strs = {name: seq for name, seq in sequences}

    layer_keys = sorted(k for k in all_acts.keys() if k.endswith(f"__layer{LAYER}"))

    # Build a flat list of (sequence_name, token_position) tuples in the
    # same order we'll concat the activations.
    token_metadata: list[dict] = []
    activation_chunks: list[torch.Tensor] = []
    for key in layer_keys:
        seq_name = key.replace(f"__layer{LAYER}", "")
        chunk = all_acts[key].float()  # [seq_len_with_special, hidden_dim]
        n_tokens_in_seq = chunk.shape[0]
        for pos in range(n_tokens_in_seq):
            # Token positions: 0 = [CLS], last = [EOS], 1..(n-2) = nucleotides 0..(n-3)
            is_special = pos == 0 or pos == n_tokens_in_seq - 1
            nt_pos = -1 if is_special else pos - 1
            token_metadata.append({
                "seq_name": seq_name,
                "tok_pos": pos,
                "nt_pos": nt_pos,
                "is_special": is_special,
            })
        activation_chunks.append(chunk)

    activations = torch.cat(activation_chunks, dim=0)
    n_tokens, d_input = activations.shape
    print(f"      Total tokens (incl. special): {n_tokens}")

    # --------------------------------------------------------------- #
    # 2. Load the trained SAE
    # --------------------------------------------------------------- #
    print(f"[2/5] Loading SAE from {SAE_FILE}...")
    if not SAE_FILE.exists():
        print(f"❌ Run Phase 9 first.")
        sys.exit(1)

    sae_state = load_file(str(SAE_FILE))

    # Reconstruct the SAE architecture matching the saved weights
    d_dict = sae_state["W_enc"].shape[1]
    sae = BatchTopKSAE(d_input=d_input, d_dict=d_dict, k=32)  # k value doesn't matter for inspection
    sae.load_state_dict(sae_state)
    sae.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = sae.to(device)
    activations = activations.to(device)

    print(f"      SAE: d_input={d_input}, d_dict={d_dict}")

    # --------------------------------------------------------------- #
    # 3. Compute per-feature activations on every token
    # --------------------------------------------------------------- #
    # Note: we use the ENCODER output (post-ReLU, pre-BatchTopK) for
    # inspection, NOT the sparsity-gated output. For interpretation we
    # want to know which tokens cause feature i to fire MOST, even if
    # BatchTopK might have suppressed the feature on some of those
    # tokens during training. This is standard practice in the SAE
    # interpretability literature.
    print(f"[3/5] Encoding all tokens through the SAE...")
    with torch.no_grad():
        feat_acts = sae.encode(activations)  # [n_tokens, d_dict]

    # --------------------------------------------------------------- #
    # 4. Summary stats: feature aliveness and breadth
    # --------------------------------------------------------------- #
    print(f"[4/5] Computing feature summary stats...")
    # Max activation per feature (across all tokens)
    max_per_feature = feat_acts.max(dim=0).values  # [d_dict]
    n_alive = (max_per_feature > 0).sum().item()
    n_dead = d_dict - n_alive

    # For each feature, how many DISTINCT sequences does it fire on?
    # We need to collapse tokens by sequence_name, then count.
    seq_idx_per_token = torch.tensor(
        [hash(m["seq_name"]) % 10**9 for m in token_metadata],
        device=device,
    )
    # For each (feature, sequence) pair, did the feature fire on any token in that sequence?
    fires_per_feature_per_seq: dict[int, set] = defaultdict(set)
    nonzero_token_mask = feat_acts > 0  # [n_tokens, d_dict]
    nonzero_indices = nonzero_token_mask.nonzero(as_tuple=False).cpu().numpy()
    for tok_idx, feat_idx in nonzero_indices:
        fires_per_feature_per_seq[int(feat_idx)].add(token_metadata[int(tok_idx)]["seq_name"])

    breadth_counts = [len(fires_per_feature_per_seq.get(f, set())) for f in range(d_dict)]
    n_specialists = sum(1 for b in breadth_counts if b == 1)
    n_generalists = sum(1 for b in breadth_counts if b >= 5)

    print(f"      Alive features:       {n_alive} / {d_dict}  ({100 * n_alive / d_dict:.1f}%)")
    print(f"      Dead features:        {n_dead} / {d_dict}  ({100 * n_dead / d_dict:.1f}%)")
    print(f"      Specialists (1 seq):  {n_specialists}  "
          f"(features that only fire within a single sequence)")
    print(f"      Generalists (≥5 seq): {n_generalists}  "
          f"(features that fire on 5 or more distinct sequences)")

    # --------------------------------------------------------------- #
    # 5. Per-feature top-activating tokens
    # --------------------------------------------------------------- #
    print(f"\n[5/5] Top {N_FEATURES_TO_SHOW} features by max activation strength:")
    print("=" * 70)

    # Pick top features by max activation magnitude
    top_feat_acts, top_feat_indices = max_per_feature.topk(N_FEATURES_TO_SHOW)
    top_feat_indices = top_feat_indices.tolist()
    top_feat_acts = top_feat_acts.tolist()

    for rank, (feat_idx, max_act) in enumerate(zip(top_feat_indices, top_feat_acts), start=1):
        breadth = len(fires_per_feature_per_seq.get(feat_idx, set()))
        feat_col = feat_acts[:, feat_idx]
        # Top tokens for this feature
        top_token_acts, top_token_idx = feat_col.topk(TOP_TOKENS_PER_FEATURE)
        top_token_idx = top_token_idx.tolist()
        top_token_acts = top_token_acts.tolist()

        print(f"\nFeature #{feat_idx:5d}  "
              f"max_act={max_act:7.3f}  "
              f"fires_on_{breadth}_sequences  "
              f"(rank {rank}/{N_FEATURES_TO_SHOW})")
        print(f"{'  rank':<5}  {'act':>7}  {'sequence':<24}  {'tok_pos':>7}  {'nt_pos':>6}  context (±{CONTEXT_WINDOW} nt)")
        print(f"  {'-' * 4}  {'-' * 7}  {'-' * 24}  {'-' * 7}  {'-' * 6}  {'-' * (2 * CONTEXT_WINDOW + 3)}")

        for r, (tok_idx, act) in enumerate(zip(top_token_idx, top_token_acts), start=1):
            meta = token_metadata[tok_idx]
            seq_name = meta["seq_name"]
            tok_pos = meta["tok_pos"]
            nt_pos = meta["nt_pos"]
            if meta["is_special"]:
                context = "  [SPECIAL TOKEN — CLS or EOS]"
            else:
                context = format_context(seq_strs[seq_name], nt_pos, CONTEXT_WINDOW)
            print(f"  {r:>4}  {act:>7.3f}  {seq_name:<24}  {tok_pos:>7d}  {nt_pos:>6d}  {context}")

    print("\n" + "=" * 70)
    print("Interpretation guide for what you're looking at")
    print("=" * 70)
    print(
        "- 'fires_on_N_sequences' is the breadth: how many of the 12 test\n"
        "  sequences contain at least one token where this feature fires.\n"
        "  N=1 → specialist (overfit signature). N≥5 → generalist (encouraging).\n"
        "- 'context' shows the focal nucleotide [in brackets] surrounded by\n"
        "  ±5 nt of context. Look for visual patterns: does the feature\n"
        "  consistently fire next to G's? On poly-U regions? On CLS tokens?\n"
        "- With only 736 training tokens, EXPECT most top features to be\n"
        "  specialists. That's overfitting, not a bug. Phase 11+ scales the\n"
        "  data and the picture changes dramatically.\n"
    )


if __name__ == "__main__":
    main()
