"""
37_snorna_mechanism.py — RIBOSCOPE: mechanistic interpretation of the non-canonical axis.

What biology does the RNA-FM non-canonical axis actually track? For each C/D snoRNA
we compute interpretable sequence properties:
  - C-box match (consensus RUGAUGA, scanned in the 5' region)
  - D-box match (consensus CUGA, scanned in the 3' region)
  - length, GC content
Then:
  [1] BIOLOGY (ground truth): how do canonical vs non-canonical snoRNAs differ on
      these properties? (Mann-Whitney effect, no model involved.)
  [2] MODEL: does RNA-FM's P(non-canonical) (out-of-fold) track the SAME
      properties? (Spearman.) This says, mechanistically, what the learned axis
      keys on — and honestly separates the length/composition component from the
      box-architecture component.

Run with
--------
    cd ~/projects/riboscope
    uv run python 37_snorna_mechanism.py
Output: outputs/snodb_mechanism.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

try:
    import numpy as np
    from safetensors.torch import load_file
    from scipy.stats import spearmanr, mannwhitneyu
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
except ImportError as e:
    print(f"❌ Missing dependency: {e}  (uv pip install scikit-learn scipy)")
    sys.exit(1)

META = Path("outputs/snodb_cd_metadata.tsv")
FASTA = Path("sequences/snodb_cd.fasta")
RNAFM = Path("outputs/snodb_cd_features_rnafm.safetensors")


def load_meta():
    with open(META, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_fasta():
    seqs, name, buf = {}, None, []
    with open(FASTA) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(buf)
                name = line[1:].split("|")[0]
                buf = []
            else:
                buf.append(line)
        if name:
            seqs[name] = "".join(buf)
    return seqs


def best_match(seq, consensus, region):
    """Max fraction of matched positions of `consensus` (R=A/G) over `region` of seq."""
    if len(region) < len(consensus):
        region = seq
    best = 0.0
    L = len(consensus)
    for i in range(0, max(1, len(region) - L + 1)):
        win = region[i:i + L]
        if len(win) < L:
            break
        m = 0
        for a, b in zip(win, consensus):
            if b == "R":
                m += a in "AG"
            else:
                m += a == b
        best = max(best, m / L)
    return best


def props(seq):
    seq = seq.upper().replace("T", "U")
    n = len(seq)
    gc = (seq.count("G") + seq.count("C")) / n if n else 0.0
    c_region = seq[: min(n, 30)]          # C-box near 5'
    d_region = seq[max(0, n - 15):]       # D-box near 3'
    return {
        "length": float(n),
        "gc": gc,
        "c_box": best_match(seq, "RUGAUGA", c_region),
        "d_box": best_match(seq, "CUGA", d_region),
    }


def main():
    for p in (META, FASTA, RNAFM):
        if not p.exists():
            print(f"❌ Missing {p}")
            sys.exit(1)

    meta = load_meta()
    seqs = load_fasta()
    sae = load_file(str(RNAFM))["sae_max"].float().numpy()

    canon = np.array([int(m["label_canonical_rrna"]) == 1 and int(m["label_orphan"]) == 0 for m in meta])
    noncanon = np.array([int(m["label_noncanonical"]) == 1 for m in meta])
    contrast = canon | noncanon
    y = noncanon[contrast].astype(int)  # 1 = non-canonical
    idx = np.where(contrast)[0]

    P = [props(seqs.get(meta[i]["snodb_id"], "")) for i in idx]
    keys = ["length", "gc", "c_box", "d_box"]
    X = {k: np.array([p[k] for p in P]) for k in keys}

    print("=" * 70)
    print("RIBOSCOPE: mechanistic interpretation of the non-canonical axis")
    print(f"  canonical n={int((1-y).sum())}   non-canonical n={int(y.sum())}")
    print("=" * 70)

    # [1] biology: property differences (ground truth)
    print(f"\n[1] BIOLOGY — property by class (mean) + Mann-Whitney")
    print(f"    {'property':<10}{'canonical':>11}{'non-canon':>11}{'p-value':>11}")
    bio = {}
    for k in keys:
        a = X[k][y == 0]; b = X[k][y == 1]
        try:
            _, pv = mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            pv = float("nan")
        bio[k] = {"canonical_mean": float(a.mean()), "noncanonical_mean": float(b.mean()), "mannwhitney_p": float(pv)}
        print(f"    {k:<10}{a.mean():>11.3f}{b.mean():>11.3f}{pv:>11.1e}")

    # [2] model: out-of-fold RNA-FM P(non-canonical), correlate with properties
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced"))
    oof = cross_val_predict(clf, sae[contrast], y,
                            cv=StratifiedKFold(5, shuffle=True, random_state=0),
                            method="predict_proba")[:, 1]
    print(f"\n[2] MODEL — Spearman(RNA-FM P(non-canonical), property)  [out-of-fold]")
    model = {}
    for k in keys:
        rho, pv = spearmanr(oof, X[k])
        model[k] = {"spearman_rho": float(rho), "p": float(pv)}
        print(f"    P ~ {k:<10} rho={rho:+.3f}  (p={pv:.1e})")

    print("\nInterpretation: properties that differ by class AND track the model's P")
    print("are what the axis mechanistically keys on. Compare the length/GC")
    print("(trivial) correlations against the box-architecture (c_box/d_box) ones to")
    print("see how much is sequence-composition vs snoRNA functional architecture.")

    with open("outputs/snodb_mechanism.json", "w") as f:
        json.dump({"n_canonical": int((1 - y).sum()), "n_noncanonical": int(y.sum()),
                   "biology": bio, "model_correlation": model}, f, indent=2)
    print("\nSaved outputs/snodb_mechanism.json.  (sync to bring back)")


if __name__ == "__main__":
    main()
