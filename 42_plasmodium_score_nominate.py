"""
42_plasmodium_score_nominate.py — RIBOSCOPE (malaria) step P3: score + rank
P. falciparum snoRNAs by the validated functional axis.

We transfer the human-trained non-canonical-vs-canonical RNA-FM axis to the
Plasmodium snoRNAs (a cross-species transfer — nominations are CANDIDATES, to be
confirmed by essentiality + conservation in the next steps). High score = looks
functionally distinctive (non-canonical), i.e. doing something beyond standard
2'-O-methylation guidance — the more interesting kind for a drug target.

Run with
--------
    cd ~/projects/riboscope
    uv run python 42_plasmodium_score_nominate.py
    ~/projects/riboscope/sync_to_windows.sh

Inputs : outputs/plasmodium_features_rnafm.safetensors, outputs/plasmodium_feasibility_gate.json,
         sequences/plasmodium_ncrna.fasta, outputs/snodb_cd_features_rnafm.safetensors,
         outputs/snodb_cd_metadata.tsv
Output : outputs/plasmodium_snorna_scores.tsv / .json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

try:
    import numpy as np
    from safetensors.torch import load_file
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

GATE = Path("outputs/plasmodium_feasibility_gate.json")
PLAS_FEATS = Path("outputs/plasmodium_features_rnafm.safetensors")
FASTA = Path("sequences/plasmodium_ncrna.fasta")
HUMAN_FEATS = Path("outputs/snodb_cd_features_rnafm.safetensors")
HUMAN_META = Path("outputs/snodb_cd_metadata.tsv")


def main():
    for p in (GATE, PLAS_FEATS, FASTA, HUMAN_FEATS, HUMAN_META):
        if not p.exists():
            print(f"❌ Missing {p}")
            sys.exit(1)

    gate = json.loads(GATE.read_text())
    ids, types = gate["ids"], gate["types"]
    plas = load_file(str(PLAS_FEATS))["sae_max"].float().numpy()
    assert plas.shape[0] == len(ids), "row/id mismatch"

    # descriptions/lengths from the FASTA
    desc, seqlen = {}, {}
    name = None
    buf = []
    for line in open(FASTA):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if name:
                seqlen[name.split("|")[0]] = len("".join(buf))
            parts = line[1:].split("|")
            name = line[1:]
            acc = parts[0]
            desc[acc] = parts[2] if len(parts) > 2 else ""
            buf = []
        elif line:
            buf.append(line.strip())
    if name:
        seqlen[name.split("|")[0]] = len("".join(buf))

    # train human non-canonical axis
    hu = load_file(str(HUMAN_FEATS))["sae_max"].float().numpy()
    meta = list(csv.DictReader(open(HUMAN_META, encoding="utf-8"), delimiter="\t"))
    canon = np.array([int(m["label_canonical_rrna"]) == 1 and int(m["label_orphan"]) == 0 for m in meta])
    noncanon = np.array([int(m["label_noncanonical"]) == 1 for m in meta])
    contrast = canon | noncanon
    y = noncanon[contrast].astype(int)
    sc = StandardScaler().fit(hu[contrast])
    clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced").fit(sc.transform(hu[contrast]), y)

    # score Plasmodium snoRNAs
    P = clf.predict_proba(sc.transform(plas))[:, 1]
    rows = []
    for i, acc in enumerate(ids):
        if types[i] != "snoRNA":
            continue
        rows.append({"accession": acc, "p_noncanonical": round(float(P[i]), 3),
                     "length": seqlen.get(acc, ""), "description": desc.get(acc, "")[:70]})
    rows.sort(key=lambda r: r["p_noncanonical"], reverse=True)

    print("=" * 78)
    print(f"RIBOSCOPE P3: P. falciparum snoRNA functional-axis scores (n={len(rows)})")
    print("=" * 78)
    print(f"  {'accession':<16}{'P_noncanon':>11}{'len':>6}  description")
    print("  " + "-" * 64)
    for r in rows[:20]:
        print(f"  {r['accession']:<16}{r['p_noncanonical']:>11.2f}{str(r['length']):>6}  {r['description'][:40]}")
    hi = sum(1 for r in rows if r["p_noncanonical"] >= 0.5)
    print(f"\n  {hi}/{len(rows)} parasite snoRNAs score >=0.5 (functionally distinctive candidates).")

    with open("outputs/plasmodium_snorna_scores.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(rows)
    with open("outputs/plasmodium_snorna_scores.json", "w") as f:
        json.dump({"n_snorna": len(rows), "n_high": hi, "ranked": rows}, f, indent=2)
    print("\nSaved outputs/plasmodium_snorna_scores.tsv/.json.")
    print("CANDIDATES only — next: essentiality (piggyBac/Zhang2018) + conservation cross-reference.")


if __name__ == "__main__":
    main()
