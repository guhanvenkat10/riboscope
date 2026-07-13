"""
43_mine_cross_model_connections.py — RIBOSCOPE: mine the models' SURPRISING
cross-family connections (the "what new connections does the model form?" hunt).

Every SAE feature that fires on families A and B asserts "A and B share something."
Most are within-class (known). A cross-model-agreed feature that links families
from DIFFERENT Rfam classes (e.g. snoRNA + riboswitch) is the model seeing a
relationship not in the textbooks — a candidate novel connection.

This first pass mines the EXISTING cross_model_agreement.json (no recompute),
annotates each shared family with its Rfam class (from Rfam.seed.gz #=GF TP),
and surfaces the matches whose family group spans >= 2 distinct classes, ranked
by Jaccard and cross-model recurrence.

Run with
--------
    cd ~/projects/riboscope
    uv run python 43_mine_cross_model_connections.py
    ~/projects/riboscope/sync_to_windows.sh

Inputs : outputs/cross_model_agreement.json, data/Rfam.seed.gz
Output : outputs/cross_model_surprising_connections.json
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

AGREE = Path("outputs/cross_model_agreement.json")
RFAM_SEED = Path("data/Rfam.seed.gz")


def parse_rfam_types(path):
    types = {}
    acc = tp = None
    with gzip.open(path, "rt", encoding="latin-1") as f:
        for line in f:
            if line.startswith("#=GF AC"):
                acc = line.split(maxsplit=2)[2].strip()
            elif line.startswith("#=GF TP"):
                tp = line.split(maxsplit=2)[2].strip()
            elif line.startswith("//"):
                if acc:
                    types[acc] = tp or ""
                acc = tp = None
    return types


def simple_class(tp: str) -> str:
    t = (tp or "").lower()
    for key in ("snorna", "riboswitch", "rrna", "trna", "mirna", "snrna",
                "ribozyme", "lncrna", "crispr", "thermoregulator", "frameshift",
                "ires", "leader", "srp", "rnase p", "telomerase", "antisense", "srna"):
        if key in t:
            return key
    if "cis-reg" in t:
        return "cis-reg"
    if "gene" in t:
        return "gene-other"
    return (tp.split(";")[0].strip().lower() if tp else "unknown") or "unknown"


def main():
    for p in (AGREE, RFAM_SEED):
        if not p.exists():
            print(f"❌ Missing {p}")
            sys.exit(1)
    types = parse_rfam_types(RFAM_SEED)
    agree = json.loads(AGREE.read_text())

    print("=" * 84)
    print("RIBOSCOPE: surprising cross-class connections the models agree on")
    print("=" * 84)

    surprising = []
    group_support = defaultdict(set)  # frozenset(families) -> set(model-pairs)
    for pair in agree.get("pairs", []):
        pa, pb = pair["model_a"], pair["model_b"]
        # skip the single-seq OOD control pairs
        if "single" in pa or "single" in pb:
            continue
        for m in pair.get("matches", []):
            fams = [f for f in m.get("shared_families_sample", []) if f]
            if len(fams) < 2:
                continue
            classes = {simple_class(types.get(f, "")) for f in fams}
            classes.discard("unknown")
            group_support[frozenset(fams)].add(f"{pa}~{pb}")
            if len(classes) >= 2:
                surprising.append({
                    "pair": f"{pa}~{pb}", "jaccard": round(m.get("jaccard", 0), 3),
                    "families": fams,
                    "classes": sorted(classes),
                    "fam_classes": {f: simple_class(types.get(f, "")) for f in fams},
                })

    # de-dup surprising groups, attach cross-model support
    seen = {}
    for s in surprising:
        key = frozenset(s["families"])
        if key not in seen or s["jaccard"] > seen[key]["jaccard"]:
            seen[key] = s
    out = sorted(seen.values(), key=lambda s: (len(group_support[frozenset(s["families"])]), s["jaccard"]),
                 reverse=True)

    print(f"\nSurprising cross-class connections found: {len(out)}")
    print(f"(of these, replicated across >1 model-pair = strongest leads)\n")
    print(f"  {'pair':<22}{'J':>5}  classes -> families")
    print("  " + "-" * 78)
    for s in out[:40]:
        n_support = len(group_support[frozenset(s["families"])])
        star = "★" if n_support > 1 else " "
        fc = ", ".join(f"{f}({c})" for f, c in s["fam_classes"].items())
        print(f" {star}{s['pair']:<22}{s['jaccard']:>5}  {'+'.join(s['classes'])}: {fc[:60]}")

    with open("outputs/cross_model_surprising_connections.json", "w") as f:
        json.dump({"n_surprising": len(out),
                   "connections": [{**s, "n_model_pairs": len(group_support[frozenset(s["families"])])}
                                   for s in out]}, f, indent=2)
    print(f"\nSaved outputs/cross_model_surprising_connections.json.")
    print("★ = replicated across >1 model-pair. Those are the candidate novel relationships to vet.")
    print("If the list is empty/all-explainable, I'll run the COMPREHENSIVE mine (all features,")
    print("full family profiles per model) rather than just the pre-computed matches.")


if __name__ == "__main__":
    main()
