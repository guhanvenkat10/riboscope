"""
44_length_control_connections.py — RIBOSCOPE: confound-control the "surprising
cross-class connections" from 43 by family length.

The dominant confound is length/structure-complexity: the models cluster LONG
RNAs together regardless of class. A genuinely interesting connection links
families of DIFFERENT size/composition (which length can't explain). This
annotates each surprising connection with its families' mean lengths + GC and
flags the ones that are NOT explainable by "all big and similar."

Run with
--------
    cd ~/projects/riboscope
    uv run python 44_length_control_connections.py

Inputs : outputs/cross_model_surprising_connections.json,
         sequences/rfam_30k.fasta, sequences/rfam_msa_query.fasta
Output : outputs/cross_model_connections_lengthcontrolled.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

CONN = Path("outputs/cross_model_surprising_connections.json")
FASTAS = [Path("sequences/rfam_30k.fasta"), Path("sequences/rfam_msa_query.fasta")]


def parse_family_stats(paths):
    by_fam = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        name = fam = None
        buf = []

        def flush():
            if fam:
                seq = "".join(buf).upper().replace("T", "U")
                if seq:
                    gc = (seq.count("G") + seq.count("C")) / len(seq)
                    by_fam[fam].append((len(seq), gc))
        with open(path) as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    flush()
                    parts = [p.strip() for p in line[1:].split("|")]
                    fam = parts[1] if len(parts) > 1 else None
                    buf = []
                else:
                    buf.append(line)
            flush()
    stats = {}
    for fam, vals in by_fam.items():
        lens = [v[0] for v in vals]
        gcs = [v[1] for v in vals]
        stats[fam] = {"mean_len": sum(lens) / len(lens), "mean_gc": sum(gcs) / len(gcs), "n": len(vals)}
    return stats


def main():
    if not CONN.exists():
        print(f"❌ Missing {CONN} (run 43 first).")
        sys.exit(1)
    stats = parse_family_stats(FASTAS)
    conns = json.loads(CONN.read_text())["connections"]

    annotated = []
    for c in conns:
        fams = c["families"]
        lens = [stats[f]["mean_len"] for f in fams if f in stats]
        gcs = [stats[f]["mean_gc"] for f in fams if f in stats]
        if len(lens) < 2:
            continue
        lo, hi = min(lens), max(lens)
        len_ratio = hi / max(lo, 1)
        all_long = lo >= 200          # "all big" = classic confound
        len_diverse = len_ratio >= 1.8  # spans very different sizes
        gc_spread = (max(gcs) - min(gcs)) if gcs else 0.0
        c2 = {**c, "n_model_pairs": c.get("n_model_pairs", 1),
              "lens": {f: round(stats[f]["mean_len"], 0) for f in fams if f in stats},
              "len_ratio": round(len_ratio, 2), "all_long": all_long,
              "len_diverse": len_diverse, "gc_spread": round(gc_spread, 3)}
        annotated.append(c2)

    # A "length-independent" lead = cross-class, NOT all-long, replicated, and
    # ideally length-diverse (model links different-sized families) with similar GC
    # (so it's not just composition either).
    leads = [c for c in annotated
             if (not c["all_long"]) and c["n_model_pairs"] >= 2]
    leads.sort(key=lambda c: (c["n_model_pairs"], c["jaccard"]), reverse=True)

    print("=" * 86)
    print("RIBOSCOPE: length-controlled cross-class connections")
    print("=" * 86)
    print(f"Total surprising connections: {len(annotated)}")
    print(f"  all-long (length-confound likely): {sum(c['all_long'] for c in annotated)}")
    print(f"  replicated >1 model-pair:          {sum(c['n_model_pairs']>=2 for c in annotated)}")
    print(f"\nLENGTH-INDEPENDENT LEADS (cross-class, NOT all-long, replicated): {len(leads)}")
    print(f"  {'pair':<20}{'J':>5}{'np':>3}{'lenR':>6}  classes : families(len)")
    print("  " + "-" * 72)
    for c in leads[:30]:
        famlen = ", ".join(f"{f}({c['fam_classes'][f]},{int(c['lens'].get(f,0))})" for f in c["families"])
        print(f"  {c['pair']:<20}{c['jaccard']:>5}{c['n_model_pairs']:>3}{c['len_ratio']:>6}  "
              f"{'+'.join(c['classes'])}: {famlen[:54]}")

    with open("outputs/cross_model_connections_lengthcontrolled.json", "w") as f:
        json.dump({"n_total": len(annotated), "n_leads": len(leads), "leads": leads}, f, indent=2)
    print(f"\nSaved outputs/cross_model_connections_lengthcontrolled.json.")
    if not leads:
        print("→ No length-independent cross-class connections. The 'connections' are the length")
        print("  confound. That's a definitive (honest) negative for this avenue.")
    else:
        print("→ These survive the length control — the real candidates to literature-vet.")


if __name__ == "__main__":
    main()
