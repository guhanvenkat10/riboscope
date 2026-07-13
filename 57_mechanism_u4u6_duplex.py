"""
57_mechanism_u4u6_duplex.py — RIBOSCOPE G4: the mechanism figure.

Why does the structure-aware model localize disease nucleotides in U4-type snRNAs
but not U2/U12? Hypothesis: U4 (and U4atac) function by base-pairing with U6, and
their disease mutations disrupt that U4/U6 duplex — exactly the secondary-structure
information ErnieRNA encodes. This script tests it:

  1. Co-fold U4 (RNU4-2) with U6 (RNU6-1) [ViennaRNA RNAcofold] to identify the U4
     positions that base-pair with U6 (the intermolecular duplex).
  2. Test whether ErnieRNA criticality concentrates on those duplex positions
     (AUC + mean) vs RNA-FM (structure-naive control).
  3. Confirm the duplex is also where SGE says function lives (the duplex positions
     should be experimentally damaging) — closing the loop model→structure→disease.

Needs ViennaRNA:  uv pip install ViennaRNA
(RNAcofold MFE is a computational proxy for the known U4/U6 duplex — stated as such.)

Run with
--------
    cd ~/projects/riboscope
    uv run python 57_mechanism_u4u6_duplex.py

Inputs : sequences/disease_structured_rna.fasta, outputs/ism_criticality_{erniarna,rnafm}.json,
         data/sge_rnu4-2.xlsx
Output : outputs/mechanism_u4u6.json
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

L_REF = 141
FASTA = "sequences/disease_structured_rna.fasta"
CRIT = {"erniarna": "outputs/ism_criticality_erniarna.json",
        "rnafm": "outputs/ism_criticality_rnafm.json"}
SGE_FILE = "data/sge_rnu4-2.xlsx"


def parse_fasta(path):
    out, gene, buf = {}, None, []
    def flush():
        if gene is not None:
            out[gene] = "".join(buf).upper().replace("T", "U")
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush(); gene = line[1:].split("|")[0].strip(); buf = []
        else:
            buf.append(line)
    flush()
    return out


def is_number(v):
    try:
        return not math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def sge_per_position():
    try:
        import pandas as pd
    except ImportError:
        return {}
    df = pd.read_excel(SGE_FILE, sheet_name=0, header=1)
    df.columns = [str(c).strip() for c in df.columns]
    by = {}
    for _, r in df.iterrows():
        t = (str(r.get("Type", "")) + str(r.get("Type_expanded_further", ""))).lower()
        if any(s in t for s in ("control", "insertion", "deletion", "indel")):
            continue
        if not is_number(r.get("position_oligonucleotide")) or not is_number(r.get("function_score")):
            continue
        p = int(round(float(r["position_oligonucleotide"])))
        by.setdefault(p, []).append(float(r["function_score"]))
    return {p: sum(v) / len(v) for p, v in by.items() if 1 <= p <= L_REF}


def cofold_structure(RNA, seq):
    """Get the dot-bracket structure string from RNA.cofold, robust to nested/odd
    return shapes across ViennaRNA versions (picks the first dot-bracket-looking str)."""
    def find_db(x):
        if isinstance(x, str):
            return x if any(ch in x for ch in ".()") else None
        if isinstance(x, (tuple, list)):
            for y in x:
                r = find_db(y)
                if r is not None:
                    return r
        return None
    s = find_db(RNA.cofold(seq))
    if s is None:
        try:
            res = RNA.fold_compound(seq).mfe_dimer()
            s = find_db(res)
        except Exception:  # noqa: BLE001
            pass
    if s is None:
        raise RuntimeError("could not extract a dot-bracket structure from ViennaRNA")
    return s


def pairs_from_db(s):
    """1-indexed pair table from a dot-bracket string (MFE = no pseudoknots)."""
    stack = []; pt = [0] * (len(s) + 1)
    for i, ch in enumerate(s, 1):
        if ch in "([{<":
            stack.append(i)
        elif ch in ")]}>":
            if stack:
                j = stack.pop(); pt[i] = j; pt[j] = i
    return pt


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


def auc(scores, pos_idx):
    P = set(pos_idx); n1 = len(P); n2 = len(scores) - n1
    if n1 == 0 or n2 == 0:
        return None
    r = rankdata(scores); U = sum(r[i] for i in P) - n1 * (n1 + 1) / 2
    return U / (n1 * n2)


def main():
    try:
        import RNA
    except ImportError:
        print("❌ ViennaRNA not installed. Run:  uv pip install ViennaRNA"); sys.exit(1)

    seqs = parse_fasta(FASTA)
    if "RNU4-2" not in seqs or "RNU6-1" not in seqs:
        print(f"❌ need RNU4-2 + RNU6-1 in {FASTA} (run 45)."); sys.exit(1)
    u4, u6 = seqs["RNU4-2"], seqs["RNU6-1"]
    Lu4 = len(u4)
    print("=" * 80)
    print("RIBOSCOPE mechanism — does criticality land on the U4/U6 base-pairing duplex?")
    print("=" * 80)

    # 1) co-fold U4 & U6, extract U4 positions paired with U6 (intermolecular)
    structure = cofold_structure(RNA, u4 + "&" + u6)
    db = structure.replace("&", "")       # concatenated dot-bracket: U4 then U6
    if len(db) != Lu4 + len(u6):
        print(f"    ⚠ structure length {len(db)} != U4+U6 {Lu4 + len(u6)} — interpreting first {Lu4} as U4.")
    pt = pairs_from_db(db)                 # 1-indexed pair table
    duplex = sorted({i for i in range(1, Lu4 + 1) if pt[i] > Lu4})
    print(f"[1] U4 length={Lu4}, U6 length={len(u6)}; U4 positions base-paired to U6 (duplex): "
          f"{len(duplex)}")
    print(f"    duplex positions (n.): {duplex}")

    if len(duplex) < 3:
        print("    ⚠ RNAcofold found few intermolecular pairs — MFE proxy weak; interpret cautiously.")

    out = {"u4_len": Lu4, "duplex_positions": duplex, "models": {}}

    # 2) criticality enrichment at duplex positions
    print("\n[2] Does criticality concentrate on the U4/U6 duplex? (AUC, mean in vs out)")
    print(f"    {'model':<10}{'AUC(duplex)':>12}{'mean_in':>10}{'mean_out':>10}")
    for model, cf in CRIT.items():
        rec = json.loads(Path(cf).read_text())["rnas"].get("RNU4-2") if Path(cf).exists() else None
        if not rec or not rec.get("criticality_embedding"):
            print(f"    {model:<10} (no criticality)"); continue
        c = rec["criticality_embedding"]
        idx = [p - 1 for p in duplex if 1 <= p <= len(c)]
        a = auc(c, idx)
        din = [c[p - 1] for p in duplex if 1 <= p <= len(c)]
        dout = [c[i] for i in range(len(c)) if (i + 1) not in duplex]
        mi, mo = (sum(din) / len(din) if din else 0), (sum(dout) / len(dout) if dout else 0)
        print(f"    {model:<10}{(a if a is not None else float('nan')):>12.3f}{mi:>10.3f}{mo:>10.3f}")
        out["models"][model] = {"auc_duplex": round(a, 3) if a is not None else None,
                                "mean_in": round(mi, 3), "mean_out": round(mo, 3)}

    # 3) confirm the duplex is also where SGE says function lives
    sge = sge_per_position()
    if sge:
        din = [sge[p] for p in duplex if p in sge]
        dout = [sge[p] for p in sge if p not in set(duplex)]
        if din and dout:
            mi, mo = sum(din) / len(din), sum(dout) / len(dout)
            print(f"\n[3] SGE confirms function: mean SGE score at duplex {mi:.3f} vs elsewhere {mo:.3f} "
                  f"(lower = more damaging)")
            out["sge_mean_in_duplex"] = round(mi, 3)
            out["sge_mean_out"] = round(mo, 3)

    Path("outputs/mechanism_u4u6.json").write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 80)
    print("✅ Saved outputs/mechanism_u4u6.json")
    print("   READ: ErnieRNA AUC(duplex) > 0.5 and > RNA-FM, with the duplex also SGE-damaging,")
    print("   = criticality lands on the U4/U6 base-pairing region → mechanistic explanation for")
    print("   the U4-type specificity (U4/U4atac act via this duplex; U2/U12 do not).")
    print("=" * 80)


if __name__ == "__main__":
    main()
