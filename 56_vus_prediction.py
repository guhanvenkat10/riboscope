"""
56_vus_prediction.py — RIBOSCOPE G4: the honest novel-prediction deliverable.

Takes the variants in RNU4-2 that ClinVar currently classifies as VARIANTS OF
UNCERTAIN SIGNIFICANCE (VUS) — i.e. nobody currently knows if they cause disease
— ranks them by our UNSUPERVISED ErnieRNA criticality, and backs each with its
EXACT per-variant saturation-genome-editing (SGE) function score. Output: a ranked
list of currently-uncertain variants we predict are pathogenic, with experimental
support — a concrete, falsifiable contribution to variant interpretation.

Honest framing: these are computational predictions with experimental (SGE)
support that NOMINATE variants for reclassification review. They are not, by
themselves, a clinical reclassification (that needs ACMG/clinical review).

Run with
--------
    cd ~/projects/riboscope
    uv run python 56_vus_prediction.py

Inputs : data/sge_rnu4-2.xlsx (from 53), outputs/ism_criticality_erniarna.json
Output : outputs/vus_predictions_rnu4-2.json
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

GENE = "RNU4-2"
L_REF = 141
SGE_FILE = "data/sge_rnu4-2.xlsx"
HEADER_ROW = 1
HGVS_COL = "HGVS"
SCORE_COL = "function_score"
CRIT_FILE = "outputs/ism_criticality_erniarna.json"
SGE_DAMAGING = -0.5            # function_score below this = clearly depleted/damaging
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
from entrez_config import get_entrez_email
TOOL, EMAIL = "riboscope", get_entrez_email()


def _get(u):
    for attempt in range(3):
        try:
            req = urllib.request.Request(u, headers={"User-Agent": f"{TOOL} (mailto:{EMAIL})"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            time.sleep(1.2 * (attempt + 1))
    return ""


def is_number(v):
    try:
        return not math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def load_sge_variant_scores():
    """Exact per-variant SGE scores: {(pos,ref,alt): function_score}."""
    import pandas as pd
    df = pd.read_excel(SGE_FILE, sheet_name=0, header=HEADER_ROW)
    df.columns = [str(c).strip() for c in df.columns]
    out = {}
    for _, r in df.iterrows():
        m = re.search(r"n\.(\d+)([ACGT])>([ACGT])", str(r.get(HGVS_COL, "")))
        s = r.get(SCORE_COL)
        if m and is_number(s):
            out[(int(m.group(1)), m.group(2), m.group(3))] = float(s)
    return out


def fetch_clinvar_vus():
    """ClinVar VUS SNVs for RNU4-2 → set of (pos, ref, alt)."""
    used_filter = True
    term = urllib.parse.quote(f'{GENE}[gene] AND "uncertain significance"[Germline classification]')
    es = _get(f"{EUTILS}/esearch.fcgi?db=clinvar&term={term}&retmax=800&retmode=json"
              f"&tool={TOOL}&email={EMAIL}")
    try:
        ids = json.loads(es)["esearchresult"]["idlist"]
    except Exception:  # noqa: BLE001
        ids = []
    if not ids:
        used_filter = False
        es = _get(f"{EUTILS}/esearch.fcgi?db=clinvar&term={urllib.parse.quote(GENE + '[gene]')}"
                  f"&retmax=800&retmode=json&tool={TOOL}&email={EMAIL}")
        try:
            ids = json.loads(es)["esearchresult"]["idlist"]
        except Exception:  # noqa: BLE001
            return set()
    variants = set()
    for k in range(0, len(ids), 100):
        time.sleep(0.34)
        try:
            res = json.loads(_get(f"{EUTILS}/esummary.fcgi?db=clinvar&id={','.join(ids[k:k+100])}"
                                  f"&retmode=json&tool={TOOL}&email={EMAIL}"))["result"]
        except Exception:  # noqa: BLE001
            continue
        for uid in res.get("uids", []):
            rec = res.get(uid, {})
            if not used_filter:
                cls = (rec.get("germline_classification", {}) or {}).get("description", "") \
                    or (rec.get("clinical_significance", {}) or {}).get("description", "")
                if "uncertain" not in cls.lower():
                    continue
            t = rec.get("title", "") or ""
            if "NR_003137" not in t:
                continue
            m = re.search(r"n\.(\d+)([ACGT])>([ACGT])", t)
            if m:
                variants.add((int(m.group(1)), m.group(2), m.group(3)))
    return variants


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs); i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        a = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = a
        i = j + 1
    return r


def auc(scores, labels):
    pos = [i for i, l in enumerate(labels) if l == 1]; n1 = len(pos); n2 = len(labels) - n1
    if n1 == 0 or n2 == 0:
        return None
    r = rankdata(scores); U = sum(r[i] for i in pos) - n1 * (n1 + 1) / 2
    return U / (n1 * n2)


def main():
    if not Path(SGE_FILE).exists():
        print(f"❌ {SGE_FILE} missing — run 53 first."); sys.exit(1)
    crit_rec = json.loads(Path(CRIT_FILE).read_text())["rnas"].get(GENE) if Path(CRIT_FILE).exists() else None
    if not crit_rec or not crit_rec.get("criticality_embedding"):
        print(f"❌ no ErnieRNA criticality for {GENE} in {CRIT_FILE}."); sys.exit(1)
    crit = crit_rec["criticality_embedding"]

    print("=" * 82)
    print(f"RIBOSCOPE — VUS prediction for {GENE} (rank uncertain variants by criticality, back with SGE)")
    print("=" * 82)
    sge = load_sge_variant_scores()
    vus = fetch_clinvar_vus()
    print(f"[1] ClinVar VUS SNVs in {GENE}: {len(vus)}   |   SGE-scored variants available: {len(sge)}")

    rows = []
    for (pos, ref, alt) in vus:
        if not (1 <= pos <= min(L_REF, len(crit))):
            continue
        rows.append({"variant": f"n.{pos}{ref}>{alt}", "pos": pos,
                     "criticality": round(crit[pos - 1], 4),
                     "sge_score": (round(sge[(pos, ref, alt)], 4) if (pos, ref, alt) in sge else None)})
    rows.sort(key=lambda r: -r["criticality"])
    n_with_sge = sum(1 for r in rows if r["sge_score"] is not None)
    print(f"[2] VUS in range with criticality: {len(rows)}   (of these, {n_with_sge} also have an exact SGE score)")

    # how well does criticality flag the experimentally-damaging VUS?
    with_sge = [r for r in rows if r["sge_score"] is not None]
    a = None
    if with_sge:
        labels = [1 if r["sge_score"] < SGE_DAMAGING else 0 for r in with_sge]
        a = auc([r["criticality"] for r in with_sge], labels)
        n_dmg = sum(labels)
        print(f"[3] Among VUS with SGE: {n_dmg} are experimentally damaging (score < {SGE_DAMAGING}). "
              f"Criticality ranks them: AUC={a:.3f}" if a is not None else "    (AUC n/a)")

    # the deliverable: top VUS predicted pathogenic, SGE-supported
    print("\n[4] TOP VUS PREDICTED PATHOGENIC (high criticality), with experimental check:")
    print(f"    {'variant':<14}{'criticality':>12}{'SGE_score':>11}   experimental verdict")
    print("    " + "-" * 64)
    nominated = []
    for r in rows[:20]:
        s = r["sge_score"]
        if s is None:
            verdict = "no SGE coverage"
        elif s < SGE_DAMAGING:
            verdict = "✓ SGE-confirmed DAMAGING"
            nominated.append(r)
        elif s < 0:
            verdict = "~ mildly depleted"
        else:
            verdict = "✗ SGE-neutral (likely tolerated)"
        sstr = f"{s:>11.3f}" if s is not None else f"{'—':>11}"
        print(f"    {r['variant']:<14}{r['criticality']:>12.3f}{sstr}   {verdict}")

    out = {"gene": GENE, "n_vus": len(vus), "n_vus_in_range": len(rows),
           "n_vus_with_sge": n_with_sge, "auc_crit_vs_sge_damaging": round(a, 3) if a is not None else None,
           "sge_damaging_threshold": SGE_DAMAGING,
           "nominated_pathogenic_sge_confirmed": nominated, "all_vus_ranked": rows}
    Path("outputs/vus_predictions_rnu4-2.json").write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 82)
    print(f"✅ {len(nominated)} currently-uncertain variants are HIGH-criticality AND SGE-confirmed damaging")
    print("   → our nominated reclassification candidates. Saved outputs/vus_predictions_rnu4-2.json")
    print("   Honest framing: predictions with experimental support, NOT clinical reclassification.")
    print("=" * 82)


if __name__ == "__main__":
    main()
