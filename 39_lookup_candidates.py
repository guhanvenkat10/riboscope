"""
39_lookup_candidates.py — dump full snoDB records for the top consensus candidates
so we can identify them (Rfam family, RNAcentral id, locus, host gene) and
literature-vet them.

Run with
--------
    cd ~/projects/riboscope
    uv run python 39_lookup_candidates.py
"""

import csv
import sys
from pathlib import Path

SNODB = Path("data/snodb_all.tsv")
CANDIDATES = ["snoDB0781", "snoDB0263", "snoDB0190", "snoDB0161",
              "snoDB0911", "snoDB0129", "snoDB0486", "snoDB0340"]
SHOW = ["snodb_id", "gene_name", "rfam_id", "rna_central_id", "ensembl_id",
        "chr", "start", "end", "strand", "length", "box_type",
        "conservation_phastcons", "conservation_snorna_atlas",
        "host_gene_name", "host_biotype", "host_function",
        "target_count", "target_biotypes", "rrna_targets", "snrna_targets",
        "protein_coding_targets", "sequence"]


def main():
    if not SNODB.exists():
        print(f"❌ {SNODB} not found.")
        sys.exit(1)
    with open(SNODB, encoding="utf-8") as f:
        rows = {r["snodb_id"]: r for r in csv.DictReader(f, delimiter="\t")}

    print("=" * 74)
    print("snoDB records for top 3-model consensus candidates")
    print("=" * 74)
    for cid in CANDIDATES:
        r = rows.get(cid)
        print(f"\n### {cid} ###")
        if r is None:
            print("   (not found)")
            continue
        for k in SHOW:
            v = (r.get(k) or "").strip()
            if v:
                print(f"   {k:<24} {v}")


if __name__ == "__main__":
    main()
