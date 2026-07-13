"""
48_consensus_and_validation.py — RIBOSCOPE G4 step 4: cross-model consensus +
the corrected disease-hotspot validation + novel-site nominations.

Reads the per-position criticality maps already saved by 47 (no model re-run) and:
  1. Recomputes the disease-hotspot enrichment against AUTHORITATIVE, literature-
     verified coordinates (NR RefSeq numbering = HGVS n. numbering):
       - RNU4-2 (U4): ReNU critical region n.62-79 (T-loop + Stem III); recurrent
         insertions n.64_65insT and n.77_78insT.  [Chen/Greene 2024, Nature]
       - RMRP: major cartilage-hair-hypoplasia variant n.72A>G (~90% of cases);
         Brazilian founder n.197C>T.  [Ridanpaa 2001; PMC11166637]
  2. Builds a CROSS-MODEL CONSENSUS map (mean of the available embedding maps —
     the fair, feature-independent readout every architecture produces) and tests
     it the same way.
  3. Nominates NOVEL high-consensus-criticality positions OUTSIDE annotated
     hotspots as testable hypotheses.

This is the validation + discovery read-out. The embedding ("representation-
sensitivity") map is the cross-architecture-comparable readout; the feature map
(when present) is the interpretable family-recognition layer.

Run with
--------
    cd ~/projects/riboscope
    uv run python 48_consensus_and_validation.py

Inputs : outputs/ism_criticality_{rnafm,erniarna,rnamsm}.json (whichever exist)
Output : outputs/disease_validation_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

MODELS = ["rnafm", "erniarna", "rnamsm"]
TOP_FRAC = 0.15
NOVEL_TOP_FRAC = 0.10      # consensus positions in the top 10% ...
NOVEL_PAD = 2              # ... and >NOVEL_PAD nt from any annotated hotspot = novel lead

# Authoritative, literature-verified disease coordinates (1-based on the RefSeq
# used in 45). Numbering checked against the fetched sequence on 2026-06-04.
HOTSPOTS = {
    "RNU4-2": {
        "disease": "ReNU syndrome (neurodevelopmental disorder)",
        "region_pos1": [62, 79],                  # T-loop + Stem III critical region
        "recurrent_pos1": [64, 65, 77, 78],       # n.64_65insT and n.77_78insT
        "ref": "Chen/Greene 2024 Nature s41586-024-07773-7; saturation editing PMC12036422",
    },
    "RMRP": {
        "disease": "Cartilage-hair hypoplasia",
        "region_pos1": None,
        "recurrent_pos1": [72, 197],              # n.72A>G (major ~90%); n.197C>T (Brazilian)
        "ref": "Ridanpaa 2001 (5200824); Brazilian founder PMC11166637",
    },
    "TERC": {
        "disease": "Dyskeratosis congenita (approx regions — secondary)",
        "region_pos1": None,
        "recurrent_pos1": None,
        "ref": "spread variant spectrum; not used as a primary gate",
    },
}


def region_enrichment(scores, region_pos1, L, top_frac):
    if not region_pos1 or len(region_pos1) != 2:
        return None
    lo, hi = max(0, region_pos1[0] - 1), min(L, region_pos1[1])
    if hi <= lo:
        return None
    inreg = [scores[j] for j in range(lo, hi)]
    outreg = [scores[j] for j in range(L) if not (lo <= j < hi)]
    if not inreg or not outreg:
        return None
    mi, mo = sum(inreg) / len(inreg), sum(outreg) / len(outreg)
    order = sorted(range(L), key=lambda j: scores[j], reverse=True)
    top_k = max(1, int(round(top_frac * L)))
    in_top = sum(1 for j in order[:top_k] if lo <= j < hi)
    region_frac = (hi - lo) / L
    hit_rate = in_top / top_k
    return {
        "region_pos1": [region_pos1[0], region_pos1[1]],
        "enrichment_ratio": round(mi / mo, 3) if mo > 0 else None,
        "region_frac_of_len": round(region_frac, 3),
        "top_hit_rate_in_region": round(hit_rate, 3),
        "fold_over_chance": round(hit_rate / region_frac, 2) if region_frac > 0 else None,
    }


def variant_ranks(scores, recur_pos1, L):
    if not recur_pos1:
        return None
    order = sorted(range(L), key=lambda j: scores[j], reverse=True)
    out = {}
    for p in recur_pos1:
        if 1 <= p <= L:
            r = order.index(p - 1) + 1
            out[str(p)] = round(1 - (r - 1) / L, 3)     # percentile (1.0 = most critical)
    return out or None


def mean_maps(maps):
    """Element-wise mean of equal-length per-position maps."""
    L = len(maps[0])
    if any(len(m) != L for m in maps):
        return None
    return [sum(m[j] for m in maps) / len(maps) for j in range(L)]


def fmt_enr(e):
    if not e:
        return "n/a"
    return (f"ratio={e['enrichment_ratio']}  top-{int(TOP_FRAC*100)}%-in-region="
            f"{e['top_hit_rate_in_region']*100:.0f}% ({e['fold_over_chance']}x chance)")


def fmt_ranks(r):
    if not r:
        return "n/a"
    return ", ".join(f"nt{p}={v*100:.0f}%" for p, v in r.items())


def main() -> None:
    loaded = {}
    for m in MODELS:
        p = Path(f"outputs/ism_criticality_{m}.json")
        if p.exists():
            loaded[m] = json.loads(p.read_text())["rnas"]
    if not loaded:
        print("❌ No ism_criticality_*.json found. Run 47 first.")
        return
    print("=" * 84)
    print(f"RIBOSCOPE G4 consensus + disease validation — models: {list(loaded)}")
    print("=" * 84)

    genes = sorted({g for r in loaded.values() for g in r})
    summary = {}
    for gene in genes:
        hs = HOTSPOTS.get(gene, {})
        region = hs.get("region_pos1")
        recur = hs.get("recurrent_pos1")
        per_model = {}
        embed_maps = []
        L_ref = None
        for m, rnas in loaded.items():
            if gene not in rnas:
                continue
            rec = rnas[gene]
            ce = rec.get("criticality_embedding")
            cf = rec.get("criticality_feature")
            if ce is None:
                continue
            L = len(ce)
            L_ref = L_ref or L
            entry = {
                "has_feature": rec.get("has_feature_readout", cf is not None),
                "embed_region": region_enrichment(ce, region, L, TOP_FRAC),
                "embed_variant_pct": variant_ranks(ce, recur, L),
            }
            if cf is not None:
                entry["feature_region"] = region_enrichment(cf, region, L, TOP_FRAC)
                entry["feature_variant_pct"] = variant_ranks(cf, recur, L)
            per_model[m] = entry
            if L == L_ref:
                embed_maps.append(ce)

        consensus = mean_maps(embed_maps) if len(embed_maps) >= 2 else None
        cons_stats = None
        novel = []
        if consensus:
            cons_stats = {
                "region": region_enrichment(consensus, region, L_ref, TOP_FRAC),
                "variant_pct": variant_ranks(consensus, recur, L_ref),
            }
            # novel nominations: top consensus positions far from any annotated hotspot
            order = sorted(range(L_ref), key=lambda j: consensus[j], reverse=True)
            n_top = max(1, int(round(NOVEL_TOP_FRAC * L_ref)))
            annotated = set()
            if region:
                annotated |= set(range(region[0] - 1, region[1]))
            if recur:
                for p in recur:
                    annotated |= set(range(p - 1 - NOVEL_PAD, p + NOVEL_PAD))
            for j in order[:n_top]:
                if all(abs(j - a) > NOVEL_PAD for a in annotated) if annotated else True:
                    novel.append({"pos1": j + 1, "consensus_criticality": round(consensus[j], 4)})
                if len(novel) >= 10:
                    break

        summary[gene] = {"disease": hs.get("disease"), "ref": hs.get("ref"),
                         "per_model": per_model, "consensus": cons_stats,
                         "novel_nominations": novel}

        # ---- console ----
        print(f"\n■ {gene}  ({hs.get('disease', '—')})")
        if region:
            print(f"   critical region n.{region[0]}-{region[1]}; recurrent {recur}")
        elif recur:
            print(f"   recurrent disease variants: {recur}")
        for m, e in per_model.items():
            tag = "feat+embed" if e["has_feature"] else "embed-only"
            print(f"   [{m:<8} {tag}] embed: {fmt_enr(e['embed_region'])}"
                  f"{'' if not e['embed_variant_pct'] else '  variants ' + fmt_ranks(e['embed_variant_pct'])}")
            if e.get("feature_variant_pct"):
                print(f"   {'':<20} feat:  variants {fmt_ranks(e['feature_variant_pct'])}")
        if cons_stats:
            print(f"   [CONSENSUS embed] {fmt_enr(cons_stats['region'])}"
                  f"{'' if not cons_stats['variant_pct'] else '  variants ' + fmt_ranks(cons_stats['variant_pct'])}")
        if novel:
            tops = ", ".join(f"nt{n['pos1']}" for n in novel[:6])
            print(f"   novel high-criticality (off-hotspot) leads: {tops}")

    with open("outputs/disease_validation_summary.json", "w") as f:
        json.dump({"models": list(loaded), "top_frac": TOP_FRAC,
                   "hotspots": HOTSPOTS, "genes": summary}, f, indent=2)
    print("\n" + "=" * 84)
    print("✅ Saved outputs/disease_validation_summary.json")
    print("   Read: variant percentile near 100% = the unsupervised map ranks that exact")
    print("   disease nucleotide among the most functionally critical. CONSENSUS = cross-model.")
    print("=" * 84)


if __name__ == "__main__":
    main()
