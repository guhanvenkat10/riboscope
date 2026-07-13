"""
position_variance_report.py — RIBOSCOPE v3: cross-model position dominance.

Why this exists
---------------
The original handoff explained RNA-MSM's SAE collapse with the claim that
"RNA-FM and ErnieRNA use RoPE that never enters the residual stream, while
RNA-MSM uses absolute embeddings that do." That claim is FALSE for the
multimolecule implementations we actually use:

  config.json (fetched 2026-05-28):
    multimolecule/rnamsm   : position_embedding_type = "absolute" (learned)
    multimolecule/rnafm    : position_embedding_type = "absolute" (learned)
    multimolecule/ernierna : position_embedding_type = "sinusoidal"
  modeling source (multimolecule GitHub master):
    all three do `embeddings = inputs_embeds + position_embeddings`
    → every model adds a position term to the residual stream.

So the difference is not presence vs absence — it's DEGREE OF DOMINANCE. This
script measures that degree directly and reproducibly, so the writeup can cite
a number instead of a wrong architectural label.

What it measures
----------------
For each model's layer-6 activations, the fraction of total activation variance
explained by token position — a one-way ANOVA with position as the factor,
pooled over hidden dims:

    SS_total   = Σ_i ||x_i - μ||^2
    SS_between = Σ_p n_p ||μ_p - μ||^2          (μ_p = mean over tokens at pos p)
    fraction   = SS_between / SS_total           ∈ [0, 1]

A high fraction means the representation is largely a function of position; the
SAE can then hit high explained-variance by memorizing position rather than
learning biology. Prediction (to be confirmed by running): RNA-MSM ≫ RNA-FM,
ErnieRNA.

Scale invariance
----------------
The fraction is invariant to a global affine transform (x → (x-a)/b scales
SS_between and SS_total identically), so comparing RNA-MSM's normalized
activations against RNA-FM/ErnieRNA's native-scale activations is valid. Caveat:
models with very different per-dim scaling weight dims differently in the pooled
ratio; we report it as a descriptive comparison, not an exact apples-to-apples
statistic, and note this in the writeup.

Run with
--------
    cd ~/projects/riboscope
    uv run python position_variance_report.py

Output
------
    outputs/position_variance_report.json
"""

import json
import sys
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)


# ============================ CONFIG ============================
LAYER = 6

MODELS = [
    {
        "tag": "rnafm",
        "label": "RNA-FM",
        "act_file": Path("outputs/activations_big_layer6_v2.safetensors"),
        "pos_emb": "learned absolute (added to residual)",
        "normalized": False,
    },
    {
        "tag": "erniarna",
        "label": "ErnieRNA",
        "act_file": Path("outputs/activations_erniarna_layer6.safetensors"),
        "pos_emb": "sinusoidal (added to residual)",
        "normalized": False,
    },
    {
        "tag": "rnamsm",
        "label": "RNA-MSM",
        "act_file": Path("outputs/activations_rnamsm_layer6.safetensors"),
        "pos_emb": "learned absolute + MSA-row (added to residual)",
        "normalized": True,  # z-scored by normalize_rnamsm.py; fraction is scale-invariant
    },
]

REPORT_OUT = Path("outputs/position_variance_report.json")
# ================================================================


def position_variance_fraction(act_file: Path) -> dict:
    """One-way ANOVA on token position, pooled over hidden dims (fp64)."""
    d = load_file(str(act_file))
    keys = sorted(k for k in d.keys() if k.endswith(f"__layer{LAYER}"))
    if not keys:
        # fall back to all keys (RNA-MSM normalized file uses the same suffix)
        keys = sorted(d.keys())

    hidden_dim = int(d[keys[0]].shape[1])
    max_len = max(int(d[k].shape[0]) for k in keys)

    pos_sum = torch.zeros((max_len, hidden_dim), dtype=torch.float64)
    pos_count = torch.zeros(max_len, dtype=torch.int64)
    global_sumsq = 0.0
    n_tokens = 0

    for k in keys:
        t = d[k].to(torch.float64)
        L = t.shape[0]
        pos_sum[:L] += t
        pos_count[:L] += 1
        global_sumsq += float((t * t).sum().item())
        n_tokens += L

    global_sum = pos_sum.sum(dim=0)
    correction = float((global_sum * global_sum).sum().item()) / max(n_tokens, 1)
    ss_total = global_sumsq - correction
    pc = pos_count.clamp(min=1).to(torch.float64)
    ss_between = float(((pos_sum * pos_sum).sum(dim=1) / pc).sum().item()) - correction
    fraction = ss_between / ss_total if ss_total > 0 else 0.0

    return {
        "n_sequences": len(keys),
        "n_tokens": n_tokens,
        "hidden_dim": hidden_dim,
        "max_token_len": max_len,
        "ss_total": ss_total,
        "ss_between_position": ss_between,
        "variance_explained_by_position": fraction,
    }


def main() -> None:
    print("=" * 70)
    print("RIBOSCOPE v3: cross-model position-variance report")
    print("=" * 70)
    print()
    print("Fraction of layer-6 activation variance explained by token position.")
    print("Higher → representation is more position-dominated → SAE more prone to")
    print("collapsing onto position instead of biology.")
    print()

    results = {}
    for cfg in MODELS:
        if not cfg["act_file"].exists():
            print(f"⚠ {cfg['label']:>10}: {cfg['act_file']} missing — skipped.")
            results[cfg["tag"]] = {"status": "missing", "act_file": str(cfg["act_file"])}
            continue
        print(f"  {cfg['label']:>10}: loading {cfg['act_file'].name}...")
        stats = position_variance_fraction(cfg["act_file"])
        stats["pos_emb"] = cfg["pos_emb"]
        stats["normalized_input"] = cfg["normalized"]
        stats["label"] = cfg["label"]
        results[cfg["tag"]] = stats
        print(f"  {cfg['label']:>10}: variance explained by position = "
              f"{stats['variance_explained_by_position']:.4f}  "
              f"({stats['n_sequences']} seqs, {stats['n_tokens']:,} tokens, "
              f"pos_emb: {cfg['pos_emb']})")

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUT, "w") as f:
        json.dump(
            {
                "layer": LAYER,
                "metric": "SS_between(position) / SS_total, pooled over hidden dims",
                "note": (
                    "All three multimolecule models add a position term to the "
                    "residual stream (RNA-FM/RNA-MSM: learned absolute; ErnieRNA: "
                    "sinusoidal) — verified from config.json + modeling source. "
                    "The earlier RoPE-vs-absolute framing is retired; this metric "
                    "quantifies position DOMINANCE instead. Fraction is invariant "
                    "to a global affine, so RNA-MSM's normalized input is comparable."
                ),
                "models": results,
            },
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print(f"✅ Report written: {REPORT_OUT}")
    print("=" * 70)
    print("Use the RNA-MSM vs RNA-FM/ErnieRNA gap as the evidence motivating")
    print("per-position-mean subtraction for RNA-MSM only.")


if __name__ == "__main__":
    main()
