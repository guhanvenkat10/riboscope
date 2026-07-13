"""
54_triage_pathogenic_benign.py — RIBOSCOPE G4 Phase 2a: the clinical VUS-triage demo.

The real-world question a genetics lab faces: given a variant in RNU4-2, is it
pathogenic or benign? We show our UNSUPERVISED criticality separates:
  PATHOGENIC = ClinVar pathogenic/likely-pathogenic variants
  BENIGN     = variants observed in healthy population cohorts (All of Us / UK
               Biobank allele count > 0), i.e. tolerated in people
…and compare to CADD (a supervised baseline) and to the SGE function score
(the wet-lab gold-standard ceiling). RNA-FM = structure-naive control.

This is NON-CIRCULAR: the labels are clinical + population frequency, independent
of both our model and the SGE assay. Criticality is per-position (n.), so each
variant inherits its position's ErnieRNA/RNA-FM criticality.

Both label sources come straight from the same Supplementary Table 1 we already
downloaded (AoU_AC / UKBiobank_AC columns) + a live ClinVar query.

Run with
--------
    cd ~/projects/riboscope
    uv run python 54_triage_pathogenic_benign.py

Inputs : data/sge_rnu4-2.xlsx, outputs/ism_criticality_{erniarna,rnafm}.json
Output : outputs/triage_pathogenic_benign.json
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

# ============================ CONFIG ============================
GENE = "RNU4-2"
L_REF = 141
SGE_FILE = "data/sge_rnu4-2.xlsx"
HEADER_ROW = 1
POS_COL = "position_oligonucleotide"
SCORE_COL = "function_score"
CADD_COL = "CADD_score"
AOU_COL = "AoU_AC"
UKB_COL = "UKBiobank_AC"
TYPE_COL = "Type"
EXCLUDE_TYPE_SUBSTR = ("control", "insertion", "deletion", "indel")
CRIT_FILES = {"erniarna": "outputs/ism_criticality_erniarna.json",
              "rnafm": "outputs/ism_criticality_rnafm.json"}
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
from entrez_config import get_entrez_email
TOOL, EMAIL = "riboscope", get_entrez_email()
# ================================================================


def is_number(v):
    try:
        return not math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def load_rows(path):
    import pandas as pd
    df = pd.read_excel(path, sheet_name=0, header=HEADER_ROW)
    df.columns = [str(c).strip() for c in df.columns]
    return df.to_dict("records")


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def auc_mwu(scores, labels):
    """AUC = P(score[pathogenic] > score[benign]); + two-sided MWU p. label 1=pathogenic."""
    pos = [i for i, l in enumerate(labels) if l == 1]
    neg = [i for i, l in enumerate(labels) if l == 0]
    n1, n2 = len(pos), len(neg)
    if n1 == 0 or n2 == 0:
        return None
    r = rankdata(scores)
    U = sum(r[i] for i in pos) - n1 * (n1 + 1) / 2
    a = U / (n1 * n2)
    mu, sd = n1 * n2 / 2, math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (U - mu) / sd if sd > 0 else 0.0
    return {"auc": round(a, 3), "p": float(f"{math.erfc(abs(z)/math.sqrt(2)):.2e}"), "n_path": n1, "n_benign": n2}


def clinvar_path_positions():
    def _get(u):
        try:
            req = urllib.request.Request(u, headers={"User-Agent": f"{TOOL} (mailto:{EMAIL})"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return ""
    term = urllib.parse.quote(f'{GENE}[gene] AND ("pathogenic"[Germline classification] '
                              f'OR "likely pathogenic"[Germline classification])')
    try:
        ids = json.loads(_get(f"{EUTILS}/esearch.fcgi?db=clinvar&term={term}&retmax=500&retmode=json"
                              f"&tool={TOOL}&email={EMAIL}"))["esearchresult"]["idlist"]
    except Exception:  # noqa: BLE001
        return set()
    pos = set()
    for k in range(0, len(ids), 100):
        time.sleep(0.34)
        try:
            res = json.loads(_get(f"{EUTILS}/esummary.fcgi?db=clinvar&id={','.join(ids[k:k+100])}"
                                  f"&retmode=json&tool={TOOL}&email={EMAIL}"))["result"]
        except Exception:  # noqa: BLE001
            continue
        for uid in res.get("uids", []):
            t = res.get(uid, {}).get("title", "") or ""
            if "NR_003137" in t:
                for m in re.finditer(r"n\.(\d+)", t):
                    pos.add(int(m.group(1)))
    return pos


def load_crit(model):
    p = Path(CRIT_FILES[model])
    if not p.exists():
        return None
    rec = json.loads(p.read_text())["rnas"].get(GENE)
    return rec.get("criticality_embedding") if rec else None


def main():
    if not Path(SGE_FILE).exists():
        print(f"❌ {SGE_FILE} not found — run 53 first (it downloads the table).")
        sys.exit(1)
    print("=" * 82)
    print("RIBOSCOPE G4 Phase 2a — clinical VUS triage: pathogenic vs population-benign")
    print("=" * 82)
    rows = load_rows(SGE_FILE)
    cv = clinvar_path_positions()
    print(f"[1] ClinVar pathogenic positions (n.): {len(cv)}")
    crit = {m: load_crit(m) for m in CRIT_FILES}
    for m, c in crit.items():
        print(f"    {m} criticality: {'loaded ('+str(len(c))+' nt)' if c else 'MISSING'}")

    # build labeled variant set (SNVs only)
    variants = []  # (pos, label, {predictor: score})
    n_pop = n_path = n_both = 0
    for r in rows:
        t = (str(r.get(TYPE_COL, "")) + " " + str(r.get("Type_expanded_further", ""))).lower()
        if any(s in t for s in EXCLUDE_TYPE_SUBSTR):
            continue
        if not is_number(r.get(POS_COL)):
            continue
        p = int(round(float(r[POS_COL])))
        if not (1 <= p <= L_REF):
            continue
        aou = float(r[AOU_COL]) if is_number(r.get(AOU_COL)) else 0.0
        ukb = float(r[UKB_COL]) if is_number(r.get(UKB_COL)) else 0.0
        is_pop = (aou > 0) or (ukb > 0)
        is_path_pos = p in cv
        if is_path_pos and is_pop:
            n_both += 1
            continue                     # ambiguous — drop
        if is_path_pos:
            label = 1; n_path += 1
        elif is_pop:
            label = 0; n_pop += 1
        else:
            continue                     # neither labeled — skip (VUS / unobserved)
        preds = {}
        for m in CRIT_FILES:
            if crit[m]:
                preds[m] = crit[m][p - 1]
        if is_number(r.get(CADD_COL)):
            preds["CADD"] = float(r[CADD_COL])
        if is_number(r.get(SCORE_COL)):
            preds["SGE_ceiling"] = -float(r[SCORE_COL])   # lower SGE = more pathogenic → negate
        variants.append((p, label, preds))

    print(f"[2] Labeled variants — pathogenic(ClinVar)={n_path}  benign(population)={n_pop}  "
          f"dropped-ambiguous={n_both}")
    if n_path < 5 or n_pop < 5:
        print("❌ Too few in a class for a meaningful AUC. Aborting.")
        sys.exit(1)

    print("\n[3] How well does each score separate pathogenic from benign? (AUC, 0.5=chance)")
    print(f"    {'predictor':<14}{'AUC':>7}{'p':>11}{'n_path':>8}{'n_benign':>9}   note")
    print("    " + "-" * 70)
    notes = {"SGE_ceiling": "wet-lab gold-standard CEILING",
             "erniarna": "OUR unsupervised structure-aware method",
             "rnafm": "structure-naive control",
             "CADD": "supervised baseline tool"}
    out = {"gene": GENE, "n_path": n_path, "n_benign": n_pop, "predictors": {}}
    order = ["SGE_ceiling", "erniarna", "CADD", "rnafm"]
    present = [k for k in order if any(k in v[2] for v in variants)]
    for pred in present:
        sub = [(v[2][pred], v[1]) for v in variants if pred in v[2]]
        res = auc_mwu([s for s, _ in sub], [l for _, l in sub])
        if res:
            out["predictors"][pred] = res
            print(f"    {pred:<14}{res['auc']:>7.3f}{res['p']:>11.1e}{res['n_path']:>8}{res['n_benign']:>9}   {notes.get(pred,'')}")

    Path("outputs/triage_pathogenic_benign.json").write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 82)
    print("✅ Saved outputs/triage_pathogenic_benign.json")
    print("   READ: ErnieRNA AUC well above 0.5 = our unsupervised map triages real variants;")
    print("   compare to CADD (supervised baseline) and the SGE wet-lab ceiling. RNA-FM≈chance")
    print("   = the architecture contrast. This is the clinical real-world-value demonstration.")
    print("=" * 82)


if __name__ == "__main__":
    main()
