"""
26_snodb_grouped_cv.py — RIBOSCOPE G3 (snoRNA discovery) step 2c: leakage control.

Step 2b found the FM/SAE adds real value over trivial features, strongest on
non-canonical-vs-canonical (+0.10 AUC). But that positive set is likely
dominated by a few large paralog clusters (e.g., the SNORD115/116 cluster that
targets HTR2C). Under random k-fold CV, near-identical paralogs leak between
train and test and inflate AUC.

This script repeats the comparison with GROUPED CV — entire gene-name families
are held out together, so no paralog is ever in both train and test. It also
prints the family composition of the positive set so we can see cluster
dominance directly.

Verdict:
  - If SAE value-add over TRIVIAL SURVIVES grouped CV (still >= ~+0.03) → the
    signal generalizes across snoRNA families → real → proceed to discovery.
  - If it collapses toward 0 → it was paralog memorization → honest negative.

Run with
--------
    cd ~/projects/riboscope
    uv run python 26_snodb_grouped_cv.py            # rnafm

Output: outputs/snodb_grouped_cv_{model}.json
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from itertools import product
from pathlib import Path

try:
    import numpy as np
    from safetensors.torch import load_file
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(f"❌ Missing dependency: {e}  (uv pip install scikit-learn)")
    sys.exit(1)

META = Path("outputs/snodb_cd_metadata.tsv")
FASTA = Path("sequences/snodb_cd.fasta")
N_SPLITS = 5
SEED = 0


def load_meta():
    with open(META, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_fasta(path):
    seqs, name, buf = {}, None, []
    with open(path) as f:
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


def kmer_features(seqs, k=3):
    kmers = ["".join(p) for p in product("ACGU", repeat=k)]
    idx = {km: i for i, km in enumerate(kmers)}
    X = np.zeros((len(seqs), len(kmers)))
    for r, s in enumerate(seqs):
        n = 0
        for i in range(len(s) - k + 1):
            j = idx.get(s[i:i + k])
            if j is not None:
                X[r, j] += 1
                n += 1
        if n:
            X[r] /= n
    return X


def family_of(name: str, fallback: str) -> str:
    """Collapse paralog copies to a family stem: SNORD115-12 -> SNORD115."""
    if not name or not name.strip():
        return fallback
    return re.sub(r"[-_]\d+[A-Za-z]?$", "", name.strip()) or fallback


def grouped_auc(X, y, groups):
    n_g = len(set(groups))
    splits = min(N_SPLITS, n_g)
    gkf = GroupKFold(n_splits=splits)
    aucs = []
    for tr, te in gkf.split(X, y, groups):
        if len(set(y[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return (float(np.mean(aucs)), float(np.std(aucs)), len(aucs)) if aucs else (float("nan"), float("nan"), 0)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "rnafm"
    feat_file = Path(f"outputs/snodb_cd_features_{model}.safetensors")
    for p in (META, FASTA, feat_file):
        if not p.exists():
            print(f"❌ Missing {p}.")
            sys.exit(1)

    print("=" * 76)
    print(f"RIBOSCOPE G3 step 2c: grouped-CV leakage control — model={model}")
    print("=" * 76)

    meta = load_meta()
    feats = load_file(str(feat_file))
    sae = feats["sae_max"].float().numpy()
    seqs_by_id = load_fasta(FASTA)

    canon = np.array([int(m["label_canonical_rrna"]) == 1 for m in meta])
    noncanon = np.array([int(m["label_noncanonical"]) == 1 for m in meta])

    pos_mask, neg_mask = noncanon, (canon & ~noncanon)
    keep = pos_mask | neg_mask
    idx = np.where(keep)[0]
    y = pos_mask[keep].astype(int)
    groups = np.array([family_of(meta[i]["gene_name"], meta[i]["snodb_id"]) for i in idx])

    # composition of the positive (non-canonical) set
    pos_fams = Counter(groups[i] for i in range(len(y)) if y[i] == 1)
    print(f"\nContrast B: non-canonical (pos={int(y.sum())}) vs canonical (neg={int((1-y).sum())})")
    print(f"Distinct families among positives: {len(pos_fams)}")
    print("Top positive families (family : #copies):")
    for fam, n in pos_fams.most_common(8):
        print(f"   {fam:<16} {n}")
    n_groups = len(set(groups))
    print(f"Total groups (families) in contrast: {n_groups}")

    seqs = [seqs_by_id.get(meta[i]["snodb_id"], "") for i in idx]
    lencons = np.array([[float(meta[i]["length"]),
                         float(meta[i]["conservation_phastcons"] or 0.0)] for i in idx])
    trivial = np.hstack([lencons, kmer_features(seqs, 3)])
    trivial_sae = np.hstack([trivial, sae[keep]])

    # random (leaky) vs grouped (honest)
    def random_auc(X):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        a = []
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
            a.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
        return float(np.mean(a)), float(np.std(a))

    rt = random_auc(trivial); rts = random_auc(trivial_sae)
    gt = grouped_auc(trivial, y, groups); gts = grouped_auc(trivial_sae, y, groups)

    print(f"\n{'':<16}{'TRIVIAL':>18}{'TRIVIAL+SAE':>18}{'SAE add':>10}")
    print("-" * 62)
    print(f"{'random CV':<16}{rt[0]:>10.3f}±{rt[1]:.3f}{rts[0]:>10.3f}±{rts[1]:.3f}{rts[0]-rt[0]:>10.3f}")
    print(f"{'grouped CV':<16}{gt[0]:>10.3f}±{gt[1]:.3f}{gts[0]:>10.3f}±{gts[1]:.3f}{gts[0]-gt[0]:>10.3f}")

    grouped_add = gts[0] - gt[0]
    print("\n--- VERDICT (grouped CV is the honest one) ---")
    if not np.isnan(grouped_add) and grouped_add >= 0.03:
        print(f"✓ SURVIVES: SAE adds {grouped_add:+.3f} over trivial under family-held-out CV "
              f"→ generalizes across snoRNA families. Proceed to cross-model + discovery.")
    elif not np.isnan(grouped_add) and grouped_add >= 0.01:
        print(f"~ MARGINAL: SAE adds {grouped_add:+.3f} under grouped CV — weak but nonzero. Interpret cautiously.")
    else:
        print(f"✗ COLLAPSES: SAE adds {grouped_add:+.3f} under grouped CV → the step-2b signal was "
              f"largely paralog memorization. Honest negative; pivot to methods-centerpiece.")

    out = Path(f"outputs/snodb_grouped_cv_{model}.json")
    with open(out, "w") as f:
        json.dump({
            "model": model,
            "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
            "n_families": n_groups,
            "top_positive_families": pos_fams.most_common(12),
            "random_cv": {"trivial": rt, "trivial_sae": rts},
            "grouped_cv": {"trivial": gt, "trivial_sae": gts},
            "grouped_sae_value_add": grouped_add,
        }, f, indent=2)
    print(f"\nSaved {out}.  (sync to bring it back)")


if __name__ == "__main__":
    main()
