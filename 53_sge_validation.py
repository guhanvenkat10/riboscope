"""
53_sge_validation.py — RIBOSCOPE G4 Phase 1: the gold-standard validation.

Does our UNSUPERVISED criticality map recapitulate the RNU4-2 saturation-genome-
editing (SGE) experimental functional map? This is the centerpiece test: if our
ErnieRNA criticality correlates with the experimentally-measured functional
effect of every variant — and the structure-naive RNA-FM does not — then the
disease half of the project is validated against the strongest possible ground
truth (a wet-lab measurement of all variants), not just ClinVar labels.

Coordinate note (verified 2026-06-08): the SGE reference is NR_003137.3 (145 nt);
our criticality maps are on NR_003137.2 (141 nt). .3 == .2 + a 4-nt 3' extension,
so positions n.1..141 are IDENTICALLY numbered — SGE n.X maps to criticality[X-1]
for X in 1..141. We drop SGE positions 142-145 and the distal SNVs (no criticality).

DATA YOU MUST DOWNLOAD FIRST (one file, free):
  Open the medRxiv preprint supplementary material:
    https://www.medrxiv.org/content/10.1101/2025.04.08.25325442v1.supplementary-material
  Download "Supplementary Table 1" (the SGE function scores for all variants) and
  save it in the repo as:   data/sge_rnu4-2.xlsx   (or .csv)
  (Nature version: https://www.nature.com/articles/s41586-026-10334-9 — Supplementary information.)

This script is built to be SAFE on an unknown file: it ALWAYS prints the sheet
names, columns, dtypes and first rows, auto-detects the variant-position and
function-score columns, and prints what it chose. If auto-detection misses, set
POS_COL / SCORE_COL / VARIANT_COL / SHEET below and re-run — it will not silently
misread the table.

Run with
--------
    cd ~/projects/riboscope
    uv run python 53_sge_validation.py
    # if needed after seeing the printout: edit the CONFIG constants, re-run.

Inputs : data/sge_rnu4-2.{xlsx,csv}, outputs/ism_criticality_{erniarna,rnafm}.json
Output : outputs/sge_validation.json
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
L_REF = 141                      # our criticality length (NR_003137.2); SGE n.1..141 align
SGE_CANDIDATES = ["data/sge_rnu4-2.xlsx", "data/sge_rnu4-2.csv",
                  "data/SupplementaryTable1.xlsx", "data/supp_table1.xlsx"]
# Open-access medRxiv Supplementary Table 1 (SGE function scores, all variants).
# Auto-downloaded if absent (same urllib pattern as our NCBI/ClinVar fetches).
SGE_URL = ("https://www.medrxiv.org/content/medrxiv/early/2025/04/10/"
           "2025.04.08.25325442/DC1/embed/media-1.xlsx?download=true")
SGE_DEST = "data/sge_rnu4-2.xlsx"
# Leave as None to auto-detect; set explicitly (a column header string) if needed:
SHEET = None
HEADER_ROW = 1                   # the xlsx has a title in row 0; real headers are row 1
# Columns LOCKED from the verified medRxiv Supplementary Table 1 structure (2026-06-08):
VARIANT_COL = "HGVS"             # values like "n.64A>G" / "n.35_36insT"
POS_COL = "position_oligonucleotide"   # transcript n. position (1..~145), matches HGVS
SCORE_COL = "function_score"     # SGE function score; NEGATIVE = depleted/loss-of-function
CADD_COL = "CADD_score"          # built-in baseline VEP tool to compare against
TYPE_COL = "Type"                # to restrict to SNVs (exclude insertion/control variants)
EXCLUDE_TYPE_SUBSTR = ("control", "insertion", "deletion", "indel")
TOP_FRAC = 0.15                  # "high criticality" cut for novel nominations
DISRUPTIVE_FRAC = 0.30           # bottom fraction of SGE scores treated as "functional/disruptive"
CRIT_FILES = {"erniarna": "outputs/ism_criticality_erniarna.json",
              "rnafm": "outputs/ism_criticality_rnafm.json"}
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
from entrez_config import get_entrez_email
TOOL, EMAIL = "riboscope", get_entrez_email()
# ================================================================


def load_table(path: Path):
    """Return (sheet_name, list-of-dict rows, columns) using pandas if available,
    else csv stdlib for .csv. Prints structure for transparency."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        try:
            import pandas as pd
        except ImportError:
            print("❌ Reading .xlsx needs pandas+openpyxl. Install:")
            print("   uv pip install pandas openpyxl")
            print("   (or re-save the table as data/sge_rnu4-2.csv and re-run)")
            sys.exit(1)
        xl = pd.ExcelFile(path)
        print(f"   sheets: {xl.sheet_names}")
        sheet = SHEET or xl.sheet_names[0]
        # if multiple sheets, pick the one with the most rows unless SHEET set
        if SHEET is None and len(xl.sheet_names) > 1:
            best, best_n = sheet, -1
            for s in xl.sheet_names:
                n = len(pd.read_excel(path, sheet_name=s, header=HEADER_ROW))
                if n > best_n:
                    best, best_n = s, n
            sheet = best
        df = pd.read_excel(path, sheet_name=sheet, header=HEADER_ROW)
        df.columns = [str(c).strip() for c in df.columns]
        return sheet, df.to_dict("records"), list(df.columns)
    else:
        import csv
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        cols = list(rows[0].keys()) if rows else []
        return path.name, rows, cols


def is_number(v):
    try:
        f = float(v)
        return not math.isnan(f)
    except (TypeError, ValueError):
        return False


def detect_columns(rows, cols):
    """Heuristic detection of (variant_col, pos_col, score_col)."""
    sample = rows[: min(200, len(rows))]
    var_c, pos_c, score_c = VARIANT_COL, POS_COL, SCORE_COL
    # variant column: values look like HGVS n. substitutions
    hgvs = re.compile(r"n?\.?-?\d+[ACGTU]\s*>\s*[ACGTU]", re.I)
    if var_c is None:
        for c in cols:
            hits = sum(1 for r in sample if isinstance(r.get(c), str) and hgvs.search(r.get(c, "")))
            if hits >= max(5, len(sample) // 5):
                var_c = c; break
    # position column: the TRANSCRIPT (n.) position — an integer column whose values
    # run ~1..145 (the RNU4-2 transcript length). This uniquely flags the n. column
    # even if unlabeled, and avoids genomic (~1.2e8) or oligo-design positions.
    if pos_c is None:
        best, best_score = None, -1.0
        for c in cols:
            vals = [float(r[c]) for r in sample if is_number(r.get(c))]
            if len(vals) < len(sample) // 2 or not vals:
                continue
            ints = sum(1 for v in vals if abs(v - round(v)) < 1e-9)
            inrange = sum(1 for v in vals if 1 <= v <= 150)
            if ints < 0.8 * len(vals) or inrange < 0.7 * len(vals):
                continue
            mx = max(vals)
            # score: reward a max near the transcript length (~145) + name hint
            name_hint = any(k in c.lower() for k in ("pos", "coord", "n.", "rnu", "transcript", "nt"))
            s = inrange / len(vals) + (1.0 if 140 <= mx <= 146 else 0.0) + (0.3 if name_hint else 0.0)
            if s > best_score:
                best, best_score = c, s
        pos_c = best
    # score column: numeric, name hints; else widest-range numeric that isn't pos
    if score_c is None:
        hint = re.compile(r"score|function|sge|fitness|deplet|lfc|log2|effect", re.I)
        cand = []
        for c in cols:
            if c in (pos_c,):
                continue
            vals = [float(r[c]) for r in sample if is_number(r.get(c))]
            if len(vals) >= len(sample) // 2 and vals:
                rng = max(vals) - min(vals)
                cand.append((bool(hint.search(c)), rng, c))
        cand.sort(key=lambda t: (t[0], t[1]), reverse=True)
        if cand:
            score_c = cand[0][2]
    return var_c, pos_c, score_c


def parse_pos(row, var_c, pos_c):
    if pos_c is not None and is_number(row.get(pos_c)):
        return int(round(float(row[pos_c])))
    if var_c is not None and isinstance(row.get(var_c), str):
        m = re.search(r"n?\.?(-?\d+)[ACGTU]\s*>\s*[ACGTU]", row[var_c], re.I)
        if m:
            return int(m.group(1))
    return None


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


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = sum((a - mx) ** 2 for a in x); sy = sum((b - my) ** 2 for b in y)
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / math.sqrt(sx * sy)


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def auc(scores, positive_idx):
    L = len(scores); P = set(positive_idx); n1 = len(P); n2 = L - n1
    if n1 == 0 or n2 == 0:
        return None
    r = rankdata(scores)
    U = sum(r[i] for i in P) - n1 * (n1 + 1) / 2
    return U / (n1 * n2)


def clinvar_positions(gene, acc_prefix):
    def _get(u):
        try:
            req = urllib.request.Request(u, headers={"User-Agent": f"{TOOL} (mailto:{EMAIL})"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return ""
    term = urllib.parse.quote(f'{gene}[gene] AND ("pathogenic"[Germline classification] '
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
            if acc_prefix in t:
                for m in re.finditer(r"n\.(\d+)", t):
                    pos.add(int(m.group(1)))
    return pos


def main():
    sge_path = next((Path(p) for p in SGE_CANDIDATES if Path(p).exists()), None)
    print("=" * 82)
    print("RIBOSCOPE G4 Phase 1 — SGE gold-standard validation (RNU4-2)")
    print("=" * 82)
    if sge_path is None:
        dest = Path(SGE_DEST)
        print(f"[0] SGE table not found locally — downloading from medRxiv → {dest}")
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(SGE_URL, headers={"User-Agent": f"{TOOL} (mailto:{EMAIL})"})
            with urllib.request.urlopen(req, timeout=60) as r:
                blob = r.read()
            if len(blob) < 2000:
                raise RuntimeError(f"downloaded file suspiciously small ({len(blob)} bytes)")
            dest.write_bytes(blob)
            sge_path = dest
            print(f"    ✓ downloaded {len(blob):,} bytes")
        except Exception as e:  # noqa: BLE001
            print(f"    ❌ auto-download failed ({type(e).__name__}: {e}).")
            print("    Manual fallback: open the medRxiv supplementary page, download")
            print("    'Supplementary Table 1' (media-1.xlsx), and save it as one of:")
            for p in SGE_CANDIDATES:
                print(f"       {p}")
            print("    page: https://www.medrxiv.org/content/10.1101/2025.04.08.25325442v1.supplementary-material")
            sys.exit(1)
    print(f"[1] Loading {sge_path} (header row = {HEADER_ROW})")
    sheet, rows, cols = load_table(sge_path)
    print(f"    sheet='{sheet}'  rows={len(rows)}  n_columns={len(cols)}")
    print("    PER-COLUMN PROFILE (name | n_numeric | range | sample values):")
    sample = rows[: min(300, len(rows))]
    for c in cols:
        nums = [float(r[c]) for r in sample if is_number(r.get(c))]
        rng = f"[{min(nums):.3g},{max(nums):.3g}]" if nums else "—"
        svals = []
        for r in sample:
            v = r.get(c)
            if v is not None and str(v).strip() and str(v).strip().lower() != "nan":
                svals.append(str(v)[:18])
            if len(svals) >= 3:
                break
        print(f"      {str(c)[:42]:<44} num={len(nums):>3} {rng:<18} e.g. {svals}")

    var_c, pos_c, score_c = detect_columns(rows, cols)
    print(f"\n[2] Auto-detected → variant_col={var_c!r}  pos_col={pos_c!r}  score_col={score_c!r}")
    print("    (if these are wrong, set VARIANT_COL/POS_COL/SCORE_COL in CONFIG from the profile above)")
    if score_c is None or (var_c is None and pos_c is None):
        print("❌ Could not auto-detect the needed columns. Set VARIANT_COL/POS_COL/SCORE_COL")
        print("   (and SHEET) in the CONFIG block from the columns printed above, then re-run.")
        sys.exit(1)

    # per-position aggregation (SNVs only; exclude insertion/control rows so the
    # comparison matches our substitution-based ISM). Also collect CADD baseline.
    by_pos: dict[int, list[float]] = {}
    cadd_pos: dict[int, list[float]] = {}
    n_kept = 0
    for r in rows:
        t = (str(r.get(TYPE_COL, "")) + " " + str(r.get("Type_expanded_further", ""))).lower()
        if any(sub in t for sub in EXCLUDE_TYPE_SUBSTR):
            continue
        p = parse_pos(r, var_c, pos_c)
        s = r.get(score_c)
        if p is None or not is_number(s):
            continue
        n_kept += 1
        by_pos.setdefault(p, []).append(float(s))
        if is_number(r.get(CADD_COL)):
            cadd_pos.setdefault(p, []).append(float(r.get(CADD_COL)))
    in_range = {p: v for p, v in by_pos.items() if 1 <= p <= L_REF}
    print(f"[3] SNV rows kept: {n_kept}; SGE positions {len(by_pos)} total, {len(in_range)} within n.1-{L_REF}")
    if len(in_range) < 30:
        print("❌ Too few in-range positions — check column detection / numbering. Aborting.")
        sys.exit(1)
    # per-position summary: mean score, and min (most disruptive allele)
    pos_mean = {p: sum(v) / len(v) for p, v in in_range.items()}
    pos_min = {p: min(v) for p, v in in_range.items()}
    cadd_mean = {p: sum(v) / len(v) for p, v in cadd_pos.items() if 1 <= p <= L_REF}
    allmeans = sorted(pos_mean.values())
    print(f"    SGE per-position mean score range [{allmeans[0]:.3f}, {allmeans[-1]:.3f}] "
          f"(lower usually = more functionally disruptive)")

    # ClinVar positions (for novel-prediction filter)
    cv = clinvar_positions(GENE, "NR_003137")
    print(f"[4] ClinVar pathogenic positions (n.): {len(cv)}")

    out = {"gene": GENE, "n_positions": len(in_range), "n_clinvar": len(cv), "models": {}}
    print("\n[5] Correlation of unsupervised criticality vs SGE function score")
    print(f"    {'model':<10}{'spearman(crit,SGE)':>20}{'spearman(crit,-min)':>22}{'AUC(disruptive)':>17}")
    print("    " + "-" * 70)
    # define experimentally "disruptive/functional" positions = most-depleted DISRUPTIVE_FRAC by mean
    cut = allmeans[max(0, int(DISRUPTIVE_FRAC * len(allmeans)) - 1)]
    disruptive = {p for p, m in pos_mean.items() if m <= cut}

    for model, cf in CRIT_FILES.items():
        cp = Path(cf)
        if not cp.exists():
            print(f"    {model:<10} (no {cf})")
            continue
        rec = json.loads(cp.read_text())["rnas"].get(GENE)
        if not rec or not rec.get("criticality_embedding"):
            print(f"    {model:<10} (no RNU4-2 criticality_embedding)")
            continue
        crit = rec["criticality_embedding"]            # length 141, normalized
        common = sorted(p for p in in_range if 1 <= p <= len(crit))
        x = [crit[p - 1] for p in common]
        y_mean = [pos_mean[p] for p in common]
        y_min = [pos_min[p] for p in common]
        rho_mean = spearman(x, y_mean)                 # expect NEGATIVE (high crit ↔ low score)
        rho_negmin = spearman(x, [-v for v in y_min])  # expect POSITIVE
        pos_idx = [i for i, p in enumerate(common) if p in disruptive]
        a = auc(x, pos_idx)
        print(f"    {model:<10}{rho_mean:>20.3f}{rho_negmin:>22.3f}{(a if a is not None else float('nan')):>17.3f}")
        out["models"][model] = {"n": len(common), "spearman_crit_vs_score": round(rho_mean, 3),
                                "spearman_crit_vs_negmin": round(rho_negmin, 3),
                                "auc_disruptive": round(a, 3) if a is not None else None}

    # BASELINE: CADD vs SGE (the supervised tool our unsupervised method should match/beat).
    # CADD high = deleterious; SGE low = depleted → expect NEGATIVE spearman (same sign as ours).
    common_c = sorted(p for p in in_range if p in cadd_mean)
    if len(common_c) >= 30:
        rho_cadd = spearman([cadd_mean[p] for p in common_c], [pos_mean[p] for p in common_c])
        print(f"    {'CADD*':<10}{rho_cadd:>20.3f}{'':>22}{'':>17}   (*supervised baseline vs same SGE truth)")
        out["baseline_CADD_spearman_vs_sge"] = round(rho_cadd, 3)
        out["baseline_CADD_n"] = len(common_c)

    # novel predictions from the localizer (ErnieRNA): high criticality + SGE-disruptive + not in ClinVar
    er = json.loads(Path(CRIT_FILES["erniarna"]).read_text())["rnas"].get(GENE) \
        if Path(CRIT_FILES["erniarna"]).exists() else None
    novel = []
    if er and er.get("criticality_embedding"):
        crit = er["criticality_embedding"]
        order = sorted(range(len(crit)), key=lambda i: crit[i], reverse=True)
        topk = set(order[:max(1, int(TOP_FRAC * len(crit)))])
        for i in order:
            p = i + 1
            if i in topk and p in disruptive and p not in cv:
                novel.append({"pos1": p, "criticality": round(crit[i], 3),
                              "sge_mean": round(pos_mean[p], 3)})
            if len(novel) >= 15:
                break
    out["novel_predictions"] = novel
    print(f"\n[6] Novel predictions (ErnieRNA high-criticality + SGE-disruptive + NOT in ClinVar): {len(novel)}")
    for n in novel[:10]:
        print(f"      n.{n['pos1']}  crit={n['criticality']}  SGE_mean={n['sge_mean']}")

    Path("outputs/sge_validation.json").write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 82)
    print("✅ Saved outputs/sge_validation.json")
    print("   READ: a NEGATIVE spearman(crit,SGE) [equivalently POSITIVE crit-vs-(-min)] for")
    print("   ErnieRNA, stronger than RNA-FM, = our unsupervised map recapitulates the wet-lab")
    print("   functional map. AUC(disruptive) >0.5 = criticality ranks experimentally-disruptive")
    print("   positions high. Novel predictions = SGE-confirmed functional nt not yet in ClinVar.")
    print("=" * 82)


if __name__ == "__main__":
    main()
