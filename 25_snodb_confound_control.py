"""
25_snodb_confound_control.py — RIBOSCOPE G3 (snoRNA discovery) step 2b.

Step 2 showed canonical-vs-orphan is highly separable (AUC 0.94) but the
TRIVIAL baselines (length+conservation 0.857, 3-mer 0.895) were already high and
SAE barely beat the raw embedding. This script answers the decisive question:

  Does the interpretable SAE feature set add ANYTHING on top of trivial sequence
  properties (length + conservation + 3-mer composition)?

It runs a NESTED comparison (5-fold CV ROC-AUC):
    TRIVIAL          = LENCONS + KMER
    TRIVIAL + EMBED  = + raw layer-6 embedding
    TRIVIAL + SAE    = + interpretable SAE features
and reports the value-add (delta over TRIVIAL) on two contrasts:
    A) canonical vs orphan      (confounded by annotation status)
    B) non-canonical vs canonical  (both characterized — the cleaner, biological one)

Verdict logic:
  - If SAE adds >= +0.03 AUC over TRIVIAL on a contrast → the FM/SAE carries
    unique signal worth pursuing for discovery.
  - If SAE adds < ~+0.01 → FM/SAE is redundant with trivial features here; the
    interpretable-discovery angle is weak and we should pivot. State it honestly.

Run with
--------
    cd ~/projects/riboscope
    uv run python 25_snodb_confound_control.py            # rnafm (trusted)

Output: outputs/snodb_confound_control_{model}.json
"""

from __future__ import annotations

import csv
import json
import sys
from itertools import product
from pathlib import Path

try:
    import numpy as np
    from safetensors.torch import load_file
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
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


def cv_auc(X, y, seed=SEED):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "rnafm"
    feat_file = Path(f"outputs/snodb_cd_features_{model}.safetensors")
    for p in (META, FASTA, feat_file):
        if not p.exists():
            print(f"❌ Missing {p}.")
            sys.exit(1)

    print("=" * 76)
    print(f"RIBOSCOPE G3 step 2b: confound-controlled value-add — model={model}")
    print("=" * 76)

    meta = load_meta()
    feats = load_file(str(feat_file))
    sae = feats["sae_max"].float().numpy()
    emb = feats["embed_mean"].float().numpy()
    seqs_by_id = load_fasta(FASTA)

    canon = np.array([int(m["label_canonical_rrna"]) == 1 for m in meta])
    orph = np.array([int(m["label_orphan"]) == 1 for m in meta])
    noncanon = np.array([int(m["label_noncanonical"]) == 1 for m in meta])

    def block_features(keep):
        idx = np.where(keep)[0]
        seqs = [seqs_by_id.get(meta[i]["snodb_id"], "") for i in idx]
        lencons = np.array([[float(meta[i]["length"]),
                             float(meta[i]["conservation_phastcons"] or 0.0)] for i in idx])
        kmer = kmer_features(seqs, 3)
        trivial = np.hstack([lencons, kmer])
        return {
            "TRIVIAL": trivial,
            "TRIVIAL+EMBED": np.hstack([trivial, emb[keep]]),
            "TRIVIAL+SAE": np.hstack([trivial, sae[keep]]),
            "SAE_only": sae[keep],
        }

    contrasts = {
        "A_canonical_vs_orphan": (canon & ~noncanon, orph & ~canon),
        "B_noncanonical_vs_canonical": (noncanon, canon & ~noncanon),
    }

    report = {}
    for cname, (pos_mask, neg_mask) in contrasts.items():
        keep = pos_mask | neg_mask
        y = pos_mask[keep].astype(int)
        if y.sum() < 20 or (1 - y).sum() < 20:
            print(f"\n[{cname}] too few samples (pos={int(y.sum())}, neg={int((1-y).sum())}) — skipped.")
            continue
        blocks = block_features(keep)
        print(f"\n[{cname}]  pos={int(y.sum())}  neg={int((1 - y).sum())}")
        print(f"  {'features':<16}{'dim':>7}{'AUC':>9}{'±std':>8}{'Δ vs TRIVIAL':>14}")
        print("  " + "-" * 54)
        base_auc = None
        rep = {}
        for name in ("TRIVIAL", "TRIVIAL+EMBED", "TRIVIAL+SAE", "SAE_only"):
            X = blocks[name]
            m, s = cv_auc(X, y)
            if name == "TRIVIAL":
                base_auc = m
            delta = "" if base_auc is None or name == "TRIVIAL" else f"{m - base_auc:+.3f}"
            print(f"  {name:<16}{X.shape[1]:>7}{m:>9.3f}{s:>8.3f}{delta:>14}")
            rep[name] = {"auc": m, "std": s, "dim": X.shape[1]}
        sae_add = rep["TRIVIAL+SAE"]["auc"] - rep["TRIVIAL"]["auc"]
        emb_add = rep["TRIVIAL+EMBED"]["auc"] - rep["TRIVIAL"]["auc"]
        rep["sae_value_add"] = sae_add
        rep["embed_value_add"] = emb_add
        report[cname] = rep
        verdict = ("REAL FM/SAE signal" if sae_add >= 0.03 else
                   "marginal" if sae_add >= 0.01 else "REDUNDANT with trivial features")
        print(f"  → SAE value-add over trivial: {sae_add:+.3f}  ({verdict})")

    out = Path(f"outputs/snodb_confound_control_{model}.json")
    with open(out, "w") as f:
        json.dump({"model": model, "contrasts": report}, f, indent=2)
    print(f"\nSaved {out}.")
    print("\nRead: SAE value-add >= +0.03 on a contrast = interpretable features carry")
    print("unique signal → pursue discovery. < +0.01 = redundant → pivot honestly.")


if __name__ == "__main__":
    main()
