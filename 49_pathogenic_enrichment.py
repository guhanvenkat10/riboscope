"""
49_pathogenic_enrichment.py — RIBOSCOPE G4 step 5: turn the RMRP/U4 single-
position result into a STATISTIC.

Question
--------
Is each model's unsupervised criticality map enriched at the FULL set of known
pathogenic positions (not just the one major variant)? If the structure-aware
model's criticality ranks pathogenic nucleotides above background while the
structure-naive model's does not, the architecture-dependent localization is a
population-level result, not a lucky single hit.

Method
------
1. Pull ClinVar pathogenic / likely-pathogenic variants for RMRP and RNU4-2 via
   NCBI E-utilities (same urllib pattern that works for efetch in 45). Parse the
   1-based n. position from each variant title that is written on OUR RefSeq
   (NR_003051 / NR_003137 -> same numbering as our sequence).
2. For each model + map, compute how well criticality discriminates pathogenic
   positions from the rest: AUC (= P(crit_pathogenic > crit_background)), the
   mean percentile of pathogenic positions, and a Mann-Whitney rank-sum p-value.
   AUC 0.5 = chance; ->1.0 = criticality concentrates at pathogenic sites.

A curated, literature-verified fallback set is used if ClinVar returns nothing,
so the test still runs (and prints which source was used — never silently fake).

Run with
--------
    cd ~/projects/riboscope
    uv run python 49_pathogenic_enrichment.py

Inputs : outputs/ism_criticality_{rnafm,erniarna,rnamsm}.json
Output : outputs/pathogenic_enrichment.json
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

MODELS = ["rnafm", "erniarna", "rnamsm"]
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
from entrez_config import get_entrez_email
TOOL, EMAIL = "riboscope", get_entrez_email()
TIMEOUT, N_RETRIES = 30, 3

# gene -> (RefSeq accession prefix that fixes n. numbering, ISM gene key)
GENES = {
    "RMRP":   ("NR_003051", "RMRP"),
    "RNU4-2": ("NR_003137", "RNU4-2"),
}

# Literature-verified fallback pathogenic positions (1-based on the RefSeq),
# used only if ClinVar fetch/parse yields nothing. Sources in the handoff.
FALLBACK = {
    "RMRP":   [72, 197],            # n.72A>G major; n.197C>T Brazilian founder
    "RNU4-2": [64, 65, 77, 78],     # n.64_65insT; n.77_78insT
}


def _get(url: str) -> str:
    last = None
    for attempt in range(1, N_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": f"{TOOL} (mailto:{EMAIL})"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * attempt)
    print(f"   ⚠ fetch failed: {last}")
    return ""


def fetch_clinvar_positions(gene: str, acc_prefix: str) -> list[int]:
    """ClinVar pathogenic/likely-pathogenic n. positions on the given RefSeq."""
    term = urllib.parse.quote(f'{gene}[gene] AND ("pathogenic"[Germline classification] '
                              f'OR "likely pathogenic"[Germline classification])')
    es = _get(f"{EUTILS}/esearch.fcgi?db=clinvar&term={term}&retmax=500&retmode=json"
              f"&tool={TOOL}&email={EMAIL}")
    ids = []
    try:
        ids = json.loads(es)["esearchresult"]["idlist"]
    except Exception:  # noqa: BLE001
        # fall back to a non-filtered search; we filter by title classification later
        es = _get(f"{EUTILS}/esearch.fcgi?db=clinvar&term={urllib.parse.quote(gene + '[gene]')}"
                  f"&retmax=500&retmode=json&tool={TOOL}&email={EMAIL}")
        try:
            ids = json.loads(es)["esearchresult"]["idlist"]
        except Exception:  # noqa: BLE001
            return []
    positions: set[int] = set()
    for k in range(0, len(ids), 100):
        batch = ",".join(ids[k:k + 100])
        time.sleep(0.4)
        summ = _get(f"{EUTILS}/esummary.fcgi?db=clinvar&id={batch}&retmode=json"
                    f"&tool={TOOL}&email={EMAIL}")
        try:
            res = json.loads(summ)["result"]
        except Exception:  # noqa: BLE001
            continue
        for uid in res.get("uids", []):
            rec = res.get(uid, {})
            title = rec.get("title", "") or ""
            cls = (rec.get("germline_classification", {}) or {}).get("description", "") \
                or (rec.get("clinical_significance", {}) or {}).get("description", "")
            if "pathogenic" not in cls.lower():
                continue
            if acc_prefix not in title:
                continue
            # parse the n. position(s) on our RefSeq, e.g. NR_003051.4(RMRP):n.72A>G
            for m in re.finditer(r"n\.(\d+)", title):
                positions.add(int(m.group(1)))
    return sorted(positions)


def rankdata(xs: list[float]) -> list[float]:
    """Average ranks (1 = smallest), ties averaged."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # average of 1-based ranks i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def enrichment(scores: list[float], pos1: list[int]):
    """AUC, mean percentile, and Mann-Whitney p for pathogenic vs background."""
    L = len(scores)
    P = sorted({p - 1 for p in pos1 if 1 <= p <= L})
    n1 = len(P)
    n2 = L - n1
    if n1 == 0 or n2 == 0:
        return None
    ranks = rankdata(scores)
    R1 = sum(ranks[i] for i in P)
    U1 = R1 - n1 * (n1 + 1) / 2
    auc = U1 / (n1 * n2)
    mean_pct = sum(ranks[i] / L for i in P) / n1
    mu = n1 * n2 / 2
    sd = math.sqrt(n1 * n2 * (L + 1) / 12)
    z = (U1 - mu) / sd if sd > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))      # two-sided
    return {"n_pathogenic": n1, "auc": round(auc, 3),
            "mean_percentile": round(mean_pct, 3), "p_value": float(f"{p:.2e}")}


def main() -> None:
    print("=" * 80)
    print("RIBOSCOPE G4 pathogenic-position enrichment (criticality vs ClinVar)")
    print("=" * 80)

    # 1) gather pathogenic positions
    path_pos = {}
    for gene, (acc, _) in GENES.items():
        pos = fetch_clinvar_positions(gene, acc)
        src = "ClinVar"
        if not pos:
            pos = FALLBACK[gene]
            src = "FALLBACK (literature)"
        path_pos[gene] = pos
        print(f"  {gene:<8} pathogenic positions ({src}, n={len(pos)}): "
              f"{pos[:25]}{' ...' if len(pos) > 25 else ''}")

    # 2) load ISM maps
    loaded = {}
    for m in MODELS:
        p = Path(f"outputs/ism_criticality_{m}.json")
        if p.exists():
            loaded[m] = json.loads(p.read_text())["rnas"]
    if not loaded:
        print("❌ No ism_criticality_*.json found. Run 47 first.")
        return

    out = {"pathogenic_positions": path_pos, "results": {}}
    for gene, (_, ism_key) in GENES.items():
        P = path_pos[gene]
        print(f"\n■ {gene}  (pathogenic n={len(P)})")
        out["results"][gene] = {}
        for m, rnas in loaded.items():
            rec = rnas.get(ism_key)
            if not rec:
                continue
            row = {}
            for label, key in (("embed", "criticality_embedding"),
                               ("feat", "criticality_feature")):
                scores = rec.get(key)
                if scores is None:
                    continue
                e = enrichment(scores, P)
                if e:
                    row[label] = e
                    print(f"   [{m:<8} {label:<5}] AUC={e['auc']:.3f}  "
                          f"mean-pctile={e['mean_percentile']:.2f}  p={e['p_value']:.1e}  "
                          f"(n_path={e['n_pathogenic']})")
            out["results"][gene][m] = row

    with open("outputs/pathogenic_enrichment.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "=" * 80)
    print("✅ Saved outputs/pathogenic_enrichment.json")
    print("   AUC 0.5 = chance; >0.5 = criticality concentrates at pathogenic positions.")
    print("   Compare structure-aware (erniarna) vs structure-naive (rnafm) per gene.")
    print("=" * 80)


if __name__ == "__main__":
    main()
