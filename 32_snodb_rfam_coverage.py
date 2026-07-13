"""
32_snodb_rfam_coverage.py — scope the RNA-MSM (MSA) leg of the cross-model panel.

RNA-MSM needs native-MSA input, so it can only score snoRNAs for which we can
build a real alignment — practically, snoRNAs that belong to an Rfam family
(whose SEED alignment we already have in data/Rfam.seed.gz). This reports how
many of our C/D-box snoRNAs have an Rfam id, broken down by functional label,
so we know RNA-MSM's achievable coverage before building the pipeline.

Run with
--------
    cd ~/projects/riboscope
    uv run python 32_snodb_rfam_coverage.py
"""

import csv
import sys
from collections import Counter
from pathlib import Path

SNODB = Path("data/snodb_all.tsv")
MIN_LEN, MAX_LEN = 50, 510


def nonempty(v):
    return bool(v and str(v).strip() and str(v).strip().lower() != "nan")


def main():
    if not SNODB.exists():
        print(f"❌ {SNODB} not found (run 22_fetch_snodb.py).")
        sys.exit(1)
    with open(SNODB, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    cd = []
    for r in rows:
        if r.get("box_type", "").strip() != "C/D":
            continue
        seq = (r.get("sequence", "") or "").strip()
        if MIN_LEN <= len(seq) <= MAX_LEN:
            cd.append(r)

    def label(r):
        if nonempty(r.get("rrna_targets")):
            return "canonical"
        targets = ["rrna_targets", "snrna_targets", "lncrna_targets", "protein_coding_targets",
                   "snorna_targets", "mirna_targets", "trna_targets", "ncrna_targets",
                   "pseudogene_targets", "other_targets"]
        if not (nonempty(r.get("target_count")) or any(nonempty(r.get(c)) for c in targets)):
            return "orphan"
        return "other/non-canonical"

    n = len(cd)
    with_rfam = [r for r in cd if nonempty(r.get("rfam_id"))]
    print("=" * 64)
    print("snoDB C/D-box — Rfam coverage (RNA-MSM/MSA feasibility)")
    print("=" * 64)
    print(f"C/D snoRNAs (len {MIN_LEN}-{MAX_LEN}): {n}")
    print(f"  with an Rfam id:  {len(with_rfam)}  ({100*len(with_rfam)/n:.0f}%)")
    print(f"  distinct Rfam families: {len(set(r['rfam_id'].strip() for r in with_rfam))}")

    print("\nRfam coverage by functional label:")
    by_label = Counter(label(r) for r in cd)
    by_label_rfam = Counter(label(r) for r in with_rfam)
    for lab in ("canonical", "orphan", "other/non-canonical"):
        tot = by_label[lab]
        cov = by_label_rfam[lab]
        print(f"  {lab:<22} {cov:>4}/{tot:<4}  ({100*cov/tot:.0f}% have Rfam)" if tot else f"  {lab}: 0")

    print("\nTop Rfam families among C/D snoRNAs (acc : #snoRNAs):")
    fam_counts = Counter(r["rfam_id"].strip() for r in with_rfam)
    for acc, c in fam_counts.most_common(12):
        print(f"   {acc:<10} {c}")
    print(f"\nRead: high coverage (esp. of orphan/non-canonical) = RNA-MSM/MSA leg is")
    print("worth building. Low coverage = report RNA-MSM on the covered subset only.")


if __name__ == "__main__":
    main()
