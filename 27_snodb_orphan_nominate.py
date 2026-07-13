"""
27_snodb_orphan_nominate.py — RIBOSCOPE G3 (snoRNA discovery) step 3a.

The functional axis passed its gates (FM features add a generalizable +0.095 AUC
on non-canonical-vs-canonical under family-held-out CV). This script turns that
into candidate discoveries:

  1. Train a non-canonical(1) vs canonical(0) classifier on RNA-FM SAE features.
  2. Apply it to the ORPHAN C/D snoRNAs (no known target) → P(non-canonical).
  3. Rank orphans: high P = "resembles functional non-canonical snoRNAs" =
     candidate uncharacterized functional/non-canonical snoRNA.
  4. Surface the interpretable SAE features driving the call (cross-referenced to
     their Rfam families/motifs from the inspection JSON) = the mechanism.
  5. Sanity: family composition of top hits (are they all SNORD115-like?) +
     conservation, so we can prioritize conserved, non-cluster candidates.

THIS IS A HYPOTHESIS GENERATOR. Single model, moderate AUC — nominations are
CANDIDATES, not findings. They must clear cross-model replication (step 3b) and a
per-candidate novelty + evidence audit (step 3c) before any claim.

Run with
--------
    cd ~/projects/riboscope
    uv run python 27_snodb_orphan_nominate.py            # rnafm

Outputs: outputs/snodb_orphan_nominations_{model}.tsv   (full ranking)
         outputs/snodb_orphan_nominations_{model}.json  (summary + driver features)
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

try:
    import numpy as np
    from safetensors.torch import load_file
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(f"❌ Missing dependency: {e}  (uv pip install scikit-learn)")
    sys.exit(1)

META = Path("outputs/snodb_cd_metadata.tsv")
INSPECTION = Path("outputs/inspection_big_layer6_v3.json")
N_SHOW = 25


def load_meta():
    with open(META, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def family_of(name, fallback):
    if not name or not name.strip():
        return fallback
    return re.sub(r"[-_]\d+[A-Za-z]?$", "", name.strip()) or fallback


def load_feature_names(path: Path) -> dict:
    """feat_idx -> 'RF000xx[,RF..]' from the inspection buckets (for interpretability)."""
    if not path.exists():
        return {}
    insp = json.loads(path.read_text())
    out = {}
    for bucket in ("specialist_features", "moderate_features", "top_features"):
        for rec in insp.get(bucket, []):
            fi = int(rec["feature_idx"])
            if fi not in out:
                fams = rec.get("families_sample", [])
                out[fi] = ",".join(fams[:3]) if fams else f"(breadth {rec.get('n_families','?')})"
    return out


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "rnafm"
    feat_file = Path(f"outputs/snodb_cd_features_{model}.safetensors")
    for p in (META, feat_file):
        if not p.exists():
            print(f"❌ Missing {p}.")
            sys.exit(1)

    print("=" * 76)
    print(f"RIBOSCOPE G3 step 3a: orphan nomination — model={model}")
    print("=" * 76)

    meta = load_meta()
    sae = load_file(str(feat_file))["sae_max"].float().numpy()
    fnames = load_feature_names(INSPECTION)

    canon = np.array([int(m["label_canonical_rrna"]) == 1 for m in meta])
    noncanon = np.array([int(m["label_noncanonical"]) == 1 for m in meta])
    orphan = np.array([int(m["label_orphan"]) == 1 for m in meta])

    train = noncanon | (canon & ~noncanon)
    y = noncanon[train].astype(int)
    nominate = orphan & ~noncanon & ~canon
    print(f"Train: non-canonical={int(y.sum())} vs canonical={int((1-y).sum())}")
    print(f"Nominate over: {int(nominate.sum())} pure-orphan C/D snoRNAs")

    sc = StandardScaler().fit(sae[train])
    clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced").fit(sc.transform(sae[train]), y)
    p_orphan = clf.predict_proba(sc.transform(sae[nominate]))[:, 1]

    idx = np.where(nominate)[0]
    rows = []
    for k, i in enumerate(idx):
        m = meta[i]
        rows.append({
            "snodb_id": m["snodb_id"],
            "gene_name": m["gene_name"],
            "family": family_of(m["gene_name"], m["snodb_id"]),
            "p_noncanonical": round(float(p_orphan[k]), 4),
            "length": m["length"],
            "conservation_phastcons": m["conservation_phastcons"],
            "host_gene": m.get("host_biotype", ""),
            "host_function": m.get("host_function", ""),
        })
    rows.sort(key=lambda r: r["p_noncanonical"], reverse=True)

    # full ranking TSV
    tsv = Path(f"outputs/snodb_orphan_nominations_{model}.tsv")
    with open(tsv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    print(f"\nTop {N_SHOW} orphan nominations (high P = resembles non-canonical snoRNAs):")
    print(f"  {'snodb_id':<12}{'gene_name':<16}{'P':>6}{'cons':>7}{'len':>6}  host_function")
    print("  " + "-" * 72)
    for r in rows[:N_SHOW]:
        cons = r["conservation_phastcons"] or "?"
        print(f"  {r['snodb_id']:<12}{(r['gene_name'] or '-')[:15]:<16}{r['p_noncanonical']:>6.2f}"
              f"{str(cons)[:5]:>7}{str(r['length']):>6}  {(r['host_function'] or '')[:28]}")

    # family composition of top hits (watch cluster dominance)
    topfam = Counter(r["family"] for r in rows[:N_SHOW])
    print(f"\n  Family composition of top {N_SHOW}: ", dict(topfam.most_common(8)))
    n_conserved = sum(1 for r in rows[:N_SHOW] if r["conservation_phastcons"] and float(r["conservation_phastcons"]) >= 0.5)
    print(f"  Of top {N_SHOW}: {n_conserved} have phastcons >= 0.5 (stronger candidates)")

    # driver features (interpretability)
    coef = clf.coef_[0]
    top_pos = np.argsort(-coef)[:12]
    print("\n  Top SAE features pushing toward NON-CANONICAL (idx : weight : Rfam families):")
    drivers = []
    for fi in top_pos:
        nm = fnames.get(int(fi), "?")
        drivers.append({"feature_idx": int(fi), "weight": float(coef[fi]), "families": nm})
        print(f"    {int(fi):>5} : {coef[fi]:+.3f} : {nm}")

    out = Path(f"outputs/snodb_orphan_nominations_{model}.json")
    with open(out, "w") as f:
        json.dump({
            "model": model,
            "n_train_noncanonical": int(y.sum()),
            "n_train_canonical": int((1 - y).sum()),
            "n_orphans_scored": int(nominate.sum()),
            "top_nominations": rows[:N_SHOW],
            "top_family_composition": topfam.most_common(),
            "driver_features": drivers,
        }, f, indent=2)
    print(f"\nSaved {tsv} and {out}.")
    print("Reminder: CANDIDATES only — pending cross-model replication + per-candidate")
    print("novelty/evidence audit. Run sync_to_windows.sh to bring results back.")


if __name__ == "__main__":
    main()
