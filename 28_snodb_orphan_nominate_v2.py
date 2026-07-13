"""
28_snodb_orphan_nominate_v2.py — RIBOSCOPE G3 step 3a (refined).

Step 3a's naive ranking was dominated by U3/U8 (known non-canonical processing
snoRNAs) and their low-conservation copies. This refines the nomination to
surface NOVEL candidates:

  - collapse orphan copies to gene-name FAMILIES (one entry per family)
  - flag/set aside KNOWN non-canonical/processing families (U3, U8, U13, U17, ...)
  - rank remaining families by P(non-canonical) AND report conservation
  - a strong candidate = unknown family, high P, and reasonably conserved

Still single-model (RNA-FM) and hypothesis-generating only.

Run with
--------
    cd ~/projects/riboscope
    uv run python 28_snodb_orphan_nominate_v2.py
Output: outputs/snodb_orphan_families_{model}.tsv / .json
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
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
# Known non-canonical / processing C/D snoRNAs (not 2'-O-me guides) — not novel.
KNOWN_NONCANON = {"U3", "U8", "U13", "U14", "U17", "U22", "SNORD3", "SNORD118", "SNORD13"}
N_SHOW = 25


def load_meta():
    with open(META, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def family_of(name, fallback):
    if not name or not name.strip():
        return fallback
    return re.sub(r"[-_]\d+[A-Za-z]?$", "", name.strip()) or fallback


def is_known(fam: str) -> bool:
    f = fam.upper()
    return any(f == k or f.startswith(k) for k in KNOWN_NONCANON)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "rnafm"
    feat_file = Path(f"outputs/snodb_cd_features_{model}.safetensors")
    if not feat_file.exists() or not META.exists():
        print("❌ Missing inputs.")
        sys.exit(1)

    print("=" * 76)
    print(f"RIBOSCOPE G3 step 3a refined: family-level orphan nomination — {model}")
    print("=" * 76)

    meta = load_meta()
    sae = load_file(str(feat_file))["sae_max"].float().numpy()
    canon = np.array([int(m["label_canonical_rrna"]) == 1 for m in meta])
    noncanon = np.array([int(m["label_noncanonical"]) == 1 for m in meta])
    orphan = np.array([int(m["label_orphan"]) == 1 for m in meta])

    train = noncanon | (canon & ~noncanon)
    y = noncanon[train].astype(int)
    sc = StandardScaler().fit(sae[train])
    clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced").fit(sc.transform(sae[train]), y)

    nominate = orphan & ~noncanon & ~canon
    idx = np.where(nominate)[0]
    p = clf.predict_proba(sc.transform(sae[nominate]))[:, 1]

    fam = defaultdict(lambda: {"n": 0, "ps": [], "cons": [], "ids": []})
    for k, i in enumerate(idx):
        m = meta[i]
        f = family_of(m["gene_name"], m["snodb_id"])
        d = fam[f]
        d["n"] += 1
        d["ps"].append(float(p[k]))
        try:
            d["cons"].append(float(m["conservation_phastcons"]))
        except (ValueError, TypeError):
            pass
        d["ids"].append(m["snodb_id"])

    fams = []
    for f, d in fam.items():
        fams.append({
            "family": f,
            "n_copies": d["n"],
            "max_p": round(max(d["ps"]), 4),
            "median_p": round(float(np.median(d["ps"])), 4),
            "max_cons": round(max(d["cons"]), 3) if d["cons"] else None,
            "known": is_known(f),
            "example_id": d["ids"][int(np.argmax(d["ps"]))],
        })
    fams.sort(key=lambda r: r["max_p"], reverse=True)

    novel = [r for r in fams if not r["known"]]
    conserved_novel = [r for r in novel if r["max_cons"] is not None and r["max_cons"] >= 0.5]

    print(f"\nOrphan families scored: {len(fams)}  (known non-canonical: {sum(r['known'] for r in fams)})")
    print(f"\nTop {N_SHOW} NOVEL (non-U3/U8) orphan families by max P(non-canonical):")
    print(f"  {'family':<16}{'n':>4}{'maxP':>7}{'medP':>7}{'maxCons':>9}   example")
    print("  " + "-" * 60)
    for r in novel[:N_SHOW]:
        cons = "?" if r["max_cons"] is None else f"{r['max_cons']:.2f}"
        print(f"  {r['family'][:15]:<16}{r['n_copies']:>4}{r['max_p']:>7.2f}{r['median_p']:>7.2f}{cons:>9}   {r['example_id']}")

    print(f"\n  CONSERVED novel candidates (maxCons >= 0.5): {len(conserved_novel)}")
    for r in conserved_novel[:15]:
        print(f"    {r['family']:<16} n={r['n_copies']}  maxP={r['max_p']:.2f}  maxCons={r['max_cons']:.2f}  {r['example_id']}")

    tsv = Path(f"outputs/snodb_orphan_families_{model}.tsv")
    with open(tsv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fams[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(fams)
    out = Path(f"outputs/snodb_orphan_families_{model}.json")
    with open(out, "w") as f:
        json.dump({"model": model, "n_families": len(fams),
                   "n_known": sum(r["known"] for r in fams),
                   "top_novel": novel[:N_SHOW],
                   "conserved_novel": conserved_novel}, f, indent=2)
    print(f"\nSaved {tsv} and {out}. (sync to bring back)")
    print("\nRead: a short list of CONSERVED, high-P, non-U3/U8 families = a real candidate")
    print("payload worth cross-model hardening. If it's empty/weak, we lean on the")
    print("validated functional-axis + methods result instead.")


if __name__ == "__main__":
    main()
