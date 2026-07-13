"""
24_snodb_functional_axis.py — RIBOSCOPE G3 (snoRNA discovery) step 2.

THE GATE. Tests whether the interpretable SAE-feature profile of a C/D-box
snoRNA separates CANONICAL (rRNA-guiding) from ORPHAN (no known target)
snoRNAs on held-out data, and whether it beats fair baselines:

  representations compared (5-fold stratified CV, ROC-AUC):
    SAE        - sae_max interpretable features        (our method)
    EMBED      - mean-pooled raw layer-6 embedding     (model baseline)
    KMER       - 3-mer composition of the sequence     (sequence baseline)
    LENCONS    - [length, phastcons conservation]      (confound baseline)
    SAE-SHUF   - SAE features with labels shuffled      (negative control → ~0.5)

Interpretation gates:
  - SAE-SHUF must be ~0.5 (else the high-dim fit is cheating).
  - SAE must beat LENCONS clearly (else it's just length/conservation).
  - SAE vs EMBED tells us whether the *interpretable* features add value over
    the raw embedding (the novelty hook), or merely match it.
  If SAE AUC is not clearly > 0.5 and > baselines, the direction is weak — we say so.

Trusted models only: RNA-FM. (ErnieRNA's structure-attention weights failed to
load after the multimolecule upgrade — its features are set aside until fixed.)

Run with
--------
    uv pip install scikit-learn        # if not already present (leaf dep; no -U)
    cd ~/projects/riboscope
    uv run python 24_snodb_functional_axis.py            # rnafm
    uv run python 24_snodb_functional_axis.py erniarna   # only after ErnieRNA load is fixed

Inputs : outputs/snodb_cd_metadata.tsv, outputs/snodb_cd_features_{model}.safetensors,
         sequences/snodb_cd.fasta
Output : outputs/snodb_functional_axis_{model}.json
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
    print(f"❌ Missing dependency: {e}")
    print("   Install scikit-learn:  uv pip install scikit-learn")
    sys.exit(1)

META = Path("outputs/snodb_cd_metadata.tsv")
FASTA = Path("sequences/snodb_cd.fasta")
N_SPLITS = 5
SEED = 0


def load_meta() -> list[dict]:
    with open(META, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_fasta(path: Path) -> dict:
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


def kmer_features(seqs: list[str], k: int = 3) -> np.ndarray:
    alpha = "ACGU"
    kmers = ["".join(p) for p in product(alpha, repeat=k)]
    idx = {km: i for i, km in enumerate(kmers)}
    X = np.zeros((len(seqs), len(kmers)), dtype=np.float64)
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


def cv_auc(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "rnafm"
    feat_file = Path(f"outputs/snodb_cd_features_{model}.safetensors")
    for p in (META, FASTA, feat_file):
        if not p.exists():
            print(f"❌ Missing {p}. Run steps 1a/1b first.")
            sys.exit(1)

    print("=" * 74)
    print(f"RIBOSCOPE G3 step 2: functional-axis test — model={model}")
    print("=" * 74)
    if model == "erniarna":
        print("⚠ ErnieRNA features are suspect (structure-attention weights didn't load")
        print("  after the multimolecule upgrade). Treat results as provisional.\n")

    meta = load_meta()
    feats = load_file(str(feat_file))
    sae = feats["sae_max"].float().numpy()
    emb = feats["embed_mean"].float().numpy()
    assert sae.shape[0] == len(meta), f"row mismatch: {sae.shape[0]} vs {len(meta)}"
    seqs_by_id = load_fasta(FASTA)

    # Contrast: canonical (rRNA target, not orphan) vs orphan (no target, no rRNA)
    canon = np.array([int(m["label_canonical_rrna"]) == 1 and int(m["label_orphan"]) == 0 for m in meta])
    orph = np.array([int(m["label_orphan"]) == 1 and int(m["label_canonical_rrna"]) == 0 for m in meta])
    keep = canon | orph
    y = canon[keep].astype(int)  # 1 = canonical, 0 = orphan
    idx_keep = np.where(keep)[0]
    print(f"Contrast set: canonical={int(y.sum())}  orphan={int((1 - y).sum())}  total={len(y)}")

    seqs = [seqs_by_id.get(meta[i]["snodb_id"], "") for i in idx_keep]
    lengths = np.array([[float(meta[i]["length"]),
                         float(meta[i]["conservation_phastcons"] or 0.0)] for i in idx_keep])

    reps = {
        "SAE": sae[keep],
        "EMBED": emb[keep],
        "KMER": kmer_features(seqs, 3),
        "LENCONS": lengths,
    }

    print(f"\n{'representation':<12}{'dim':>7}{'AUC':>9}{'±std':>8}")
    print("-" * 38)
    results = {}
    for name, X in reps.items():
        m, s = cv_auc(X, y)
        results[name] = {"auc": m, "std": s, "dim": X.shape[1]}
        print(f"{name:<12}{X.shape[1]:>7}{m:>9.3f}{s:>8.3f}")

    # negative control: shuffle labels, re-test SAE
    rng = np.random.default_rng(SEED)
    y_shuf = y.copy()
    rng.shuffle(y_shuf)
    ms, ss = cv_auc(reps["SAE"], y_shuf)
    results["SAE_SHUFFLED"] = {"auc": ms, "std": ss, "dim": reps["SAE"].shape[1]}
    print(f"{'SAE-SHUF':<12}{reps['SAE'].shape[1]:>7}{ms:>9.3f}{ss:>8.3f}   (negative control → want ~0.50)")

    # top discriminative SAE features (full-data fit; for interpretability cross-ref)
    sc = StandardScaler().fit(reps["SAE"])
    clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced").fit(sc.transform(reps["SAE"]), y)
    coef = clf.coef_[0]
    top = np.argsort(-np.abs(coef))[:15]
    print("\nTop 15 discriminative SAE features (idx : signed weight; + = canonical):")
    for fi in top:
        print(f"  feature {int(fi):>5} : {coef[fi]:+.3f}")
    results["top_features"] = [{"feature_idx": int(fi), "weight": float(coef[fi])} for fi in top]

    # verdict
    sae_auc = results["SAE"]["auc"]
    best_baseline = max(results["EMBED"]["auc"], results["KMER"]["auc"], results["LENCONS"]["auc"])
    print("\n--- VERDICT ---")
    if ms > 0.6:
        print(f"⚠ shuffled-label AUC = {ms:.3f} is not ~0.5 — high-dim leakage suspected; interpret cautiously.")
    if sae_auc >= 0.70 and sae_auc > best_baseline + 0.03 and ms < 0.6:
        print(f"✓ SIGNAL: SAE AUC {sae_auc:.3f} clearly separates canonical vs orphan and beats baselines "
              f"(best baseline {best_baseline:.3f}). Proceed to cross-model + discovery.")
    elif sae_auc >= 0.70 and ms < 0.6:
        print(f"~ PARTIAL: SAE AUC {sae_auc:.3f} separates, but does NOT clearly beat a baseline "
              f"({best_baseline:.3f}) — the signal may be embedding/length/composition, not interpretable features.")
    else:
        print(f"✗ WEAK: SAE AUC {sae_auc:.3f} — no strong functional axis in C/D canonical-vs-orphan. "
              f"Honest negative; reconsider the framing before more compute.")

    out = Path(f"outputs/snodb_functional_axis_{model}.json")
    with open(out, "w") as f:
        json.dump({"model": model, "n_canonical": int(y.sum()), "n_orphan": int((1 - y).sum()),
                   "results": results}, f, indent=2)
    print(f"\nSaved {out}.  Run ~/projects/riboscope/sync_to_windows.sh to bring results back.")


if __name__ == "__main__":
    main()
