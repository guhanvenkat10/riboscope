"""
31_cross_model_snorna_agreement.py — RIBOSCOPE: cross-model agreement on the
snoRNA non-canonical axis (RNA-FM vs ErnieRNA).

Both architectures independently show the functional axis (steps 25/26). This
asks the sharper question: do they AGREE per-snoRNA?

  1. Train the non-canonical(1)-vs-canonical(0) classifier on each model's SAE
     features (labeled set only).
  2. Predict P(non-canonical) for the 708 HELD-OUT orphans with each model.
  3. Spearman-correlate the two models' orphan scores + top-orphan overlap
     (do they rank the same snoRNAs as non-canonical-looking?).
  4. HELD-OUT REDISCOVERY: U3, U8, SNORD97 are orphans (never in training) —
     report both models' P for them. Both scoring them high = two independent
     AIs rediscovering known non-canonical biology.

Requires snodb_cd_features_rnafm.safetensors AND _erniarna.safetensors
(ErnieRNA must be the FIXED extraction, step 23 patched).

Run with
--------
    cd ~/projects/riboscope
    uv run python 31_cross_model_snorna_agreement.py
Output: outputs/snodb_cross_model_agreement.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

try:
    import numpy as np
    from safetensors.torch import load_file
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(f"❌ Missing dependency: {e}  (uv pip install scikit-learn scipy)")
    sys.exit(1)

META = Path("outputs/snodb_cd_metadata.tsv")
MODELS = ["rnafm", "erniarna"]


def load_meta():
    with open(META, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    for m in MODELS:
        if not Path(f"outputs/snodb_cd_features_{m}.safetensors").exists():
            print(f"❌ Missing features for {m}.")
            sys.exit(1)

    print("=" * 72)
    print("RIBOSCOPE: cross-model snoRNA agreement (RNA-FM vs ErnieRNA)")
    print("=" * 72)

    meta = load_meta()
    canon = np.array([int(x["label_canonical_rrna"]) == 1 for x in meta])
    noncanon = np.array([int(x["label_noncanonical"]) == 1 for x in meta])
    orphan = np.array([int(x["label_orphan"]) == 1 for x in meta])

    train = noncanon | (canon & ~noncanon)
    y = noncanon[train].astype(int)
    orph = orphan & ~noncanon & ~canon
    orph_idx = np.where(orph)[0]

    p_orph = {}
    for m in MODELS:
        sae = load_file(f"outputs/snodb_cd_features_{m}.safetensors")["sae_max"].float().numpy()
        sc = StandardScaler().fit(sae[train])
        clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced").fit(sc.transform(sae[train]), y)
        p_orph[m] = clf.predict_proba(sc.transform(sae[orph]))[:, 1]

    rho, pval = spearmanr(p_orph["rnafm"], p_orph["erniarna"])
    print(f"\nHeld-out orphans scored by both models: {len(orph_idx)}")
    print(f"Spearman correlation of P(non-canonical) across orphans: rho={rho:.3f} (p={pval:.1e})")

    # top-orphan overlap
    for topn in (25, 50, 100):
        a = set(np.argsort(-p_orph["rnafm"])[:topn])
        b = set(np.argsort(-p_orph["erniarna"])[:topn])
        jac = len(a & b) / len(a | b)
        print(f"  top-{topn} orphan overlap: {len(a & b)}/{topn}  (Jaccard {jac:.2f})")

    # held-out rediscovery
    print("\nHELD-OUT REDISCOVERY (orphans never seen in training):")
    print(f"  {'snoRNA':<22}{'RNA-FM P':>10}{'ErnieRNA P':>12}")
    targets = [("U3 (mean of copies)", lambda x: x["gene_name"].strip() == "U3"),
               ("U8 (mean of copies)", lambda x: x["gene_name"].strip() == "U8"),
               ("SNORD97 (snoDB0492)", lambda x: x["snodb_id"] == "snoDB0492")]
    id_to_orphpos = {gi: k for k, gi in enumerate(orph_idx)}
    rediscovery = {}
    for label, pred in targets:
        members = [gi for gi in orph_idx if pred(meta[gi])]
        if not members:
            print(f"  {label:<22}{'(not in orphan set)':>22}")
            continue
        rf = float(np.mean([p_orph["rnafm"][id_to_orphpos[gi]] for gi in members]))
        er = float(np.mean([p_orph["erniarna"][id_to_orphpos[gi]] for gi in members]))
        rediscovery[label] = {"rnafm": rf, "erniarna": er, "n": len(members)}
        print(f"  {label:<22}{rf:>10.2f}{er:>12.2f}")

    out = Path("outputs/snodb_cross_model_agreement.json")
    with open(out, "w") as f:
        json.dump({
            "n_orphans": int(orph.sum()),
            "spearman_rho": float(rho), "spearman_p": float(pval),
            "rediscovery": rediscovery,
        }, f, indent=2)
    print(f"\nSaved {out}.")
    print("\nRead: high Spearman + both models flagging U3/U8/SNORD97 = two independent")
    print("architectures independently agree on snoRNA non-canonical function.")


if __name__ == "__main__":
    main()
