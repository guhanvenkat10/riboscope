"""
38_consensus_novel_candidates.py — RIBOSCOPE: 3-model CONSENSUS novel-orphan hunt.

The single-model nomination kept surfacing known snoRNAs. Now that all three
models score the orphans, the strongest possible novel lead is an orphan that
ALL THREE independent architectures agree looks non-canonical, is conserved, and
is NOT one of the already-characterized ones.

For each shared orphan we take each model's P(non-canonical) and rank by the
MIN across the three (strict consensus — all three must agree). We exclude known
non-canonical families and report conservation + host context for vetting.

Run with
--------
    cd ~/projects/riboscope
    uv run python 38_consensus_novel_candidates.py
    ~/projects/riboscope/sync_to_windows.sh
Output: outputs/snodb_consensus_candidates.tsv / .json
"""

from __future__ import annotations

import csv
import json
import re
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

META = Path("outputs/snodb_cd_metadata.tsv")
RNAMSM_META = Path("outputs/snodb_cd_rnamsm_meta.tsv")
FEATS = {"RNA-FM": "outputs/snodb_cd_features_rnafm.safetensors",
         "ErnieRNA": "outputs/snodb_cd_features_erniarna.safetensors",
         "RNA-MSM": "outputs/snodb_cd_features_rnamsm.safetensors"}
# Already-characterized non-canonical / processing C/D snoRNAs — not novel.
KNOWN = {"U3", "U8", "U13", "U14", "U17", "U22", "SNORD3", "SNORD118", "SNORD13",
         "SNORD115", "SNORD116", "SNORD97", "SNORD113", "SNORD114"}
N_SHOW = 30


def load_meta():
    with open(META, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def family_of(name, fallback):
    if not name or not name.strip():
        return fallback
    return re.sub(r"[-_]\d+[A-Za-z]?$", "", name.strip()) or fallback


def is_known(fam):
    f = fam.upper()
    return any(f == k or f.startswith(k) for k in KNOWN)


def main():
    for p in [META, RNAMSM_META] + [Path(v) for v in FEATS.values()]:
        if not Path(p).exists():
            print(f"❌ Missing {p}")
            sys.exit(1)

    meta = load_meta()
    id2row = {m["snodb_id"]: i for i, m in enumerate(meta)}
    rnamsm_ids = [r["snodb_id"] for r in csv.DictReader(open(RNAMSM_META), delimiter="\t")]
    rnamsm_pos = {s: i for i, s in enumerate(rnamsm_ids)}
    shared = [s for s in rnamsm_ids if s in id2row]

    feats = {n: load_file(p) for n, p in FEATS.items()}
    def row(n, s):
        return (feats[n]["sae_max"][rnamsm_pos[s]] if n == "RNA-MSM"
                else feats[n]["sae_max"][id2row[s]]).numpy()
    sae = {n: np.stack([row(n, s) for s in shared]) for n in FEATS}

    M = [meta[id2row[s]] for s in shared]
    canon = np.array([int(m["label_canonical_rrna"]) == 1 and int(m["label_orphan"]) == 0 for m in M])
    noncanon = np.array([int(m["label_noncanonical"]) == 1 for m in M])
    orphan = np.array([int(m["label_orphan"]) == 1 for m in M])
    contrast = canon | noncanon
    y = noncanon[contrast].astype(int)

    P = {}
    for n in FEATS:
        sc = StandardScaler().fit(sae[n][contrast])
        clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced").fit(sc.transform(sae[n][contrast]), y)
        P[n] = clf.predict_proba(sc.transform(sae[n][orphan]))[:, 1]

    orph_idx = np.where(orphan)[0]
    rows = []
    for k, gi in enumerate(orph_idx):
        m = M[gi]
        fam = family_of(m["gene_name"], m["snodb_id"])
        ps = {n: float(P[n][k]) for n in FEATS}
        rows.append({
            "snodb_id": m["snodb_id"], "gene_name": m["gene_name"], "family": fam,
            "known": is_known(fam),
            "P_min": round(min(ps.values()), 3), "P_mean": round(sum(ps.values()) / 3, 3),
            "P_rnafm": round(ps["RNA-FM"], 3), "P_ernie": round(ps["ErnieRNA"], 3), "P_rnamsm": round(ps["RNA-MSM"], 3),
            "conservation": m["conservation_phastcons"], "host_function": m.get("host_function", ""),
        })
    rows.sort(key=lambda r: r["P_min"], reverse=True)
    novel = [r for r in rows if not r["known"]]

    print("=" * 84)
    print(f"RIBOSCOPE: 3-model CONSENSUS novel-orphan candidates (shared orphans = {len(rows)})")
    print("=" * 84)
    print(f"Top {N_SHOW} NOVEL (non-known) orphans by 3-model consensus (min P across models):")
    print(f"  {'snodb_id':<11}{'family':<13}{'Pmin':>6}{'FM':>6}{'Ern':>6}{'MSM':>6}{'cons':>6}  host_function")
    print("  " + "-" * 78)
    for r in novel[:N_SHOW]:
        cons = r["conservation"] or "?"
        print(f"  {r['snodb_id']:<11}{r['family'][:12]:<13}{r['P_min']:>6.2f}{r['P_rnafm']:>6.2f}"
              f"{r['P_ernie']:>6.2f}{r['P_rnamsm']:>6.2f}{str(cons)[:5]:>6}  {(r['host_function'] or '')[:26]}")

    # conserved + strong-consensus shortlist
    strong = [r for r in novel if r["P_min"] >= 0.5 and r["conservation"]
              and r["conservation"] != "" and float(r["conservation"]) >= 0.5]
    print(f"\n  STRONG LEADS — all 3 models P>=0.5 AND phastcons>=0.5 (non-known): {len(strong)}")
    for r in strong[:20]:
        print(f"    {r['snodb_id']:<11} {r['family']:<14} Pmin={r['P_min']:.2f} "
              f"(FM {r['P_rnafm']:.2f}/Ern {r['P_ernie']:.2f}/MSM {r['P_rnamsm']:.2f}) cons={r['conservation']}")

    with open("outputs/snodb_consensus_candidates.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(rows)
    with open("outputs/snodb_consensus_candidates.json", "w") as f:
        json.dump({"n_orphans": len(rows), "top_novel": novel[:N_SHOW], "strong_leads": strong}, f, indent=2)
    print(f"\nSaved outputs/snodb_consensus_candidates.tsv/.json.")
    print("STRONG LEADS = the best novel candidates we can produce (3-model consensus +")
    print("conserved + not already known). These are what we literature-vet next.")


if __name__ == "__main__":
    main()
