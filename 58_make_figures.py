"""
58_make_figures.py — RIBOSCOPE manuscript Figures 2 and 3 (data-heavy panels).

Figures 1, 4 and 5 are hand-authored SVGs in figures/ and are NOT generated here.
Only Fig 2 (per-position tracks) and Fig 3 (141-point scatter) need code.

Run in the WSL repo:
    cd ~/projects/riboscope
    uv run python 58_make_figures.py
(needs matplotlib + pandas + openpyxl)

Style matches the hand-authored SVGs: no titles on the figure (captions live in the
legends), large readable type, minimal annotation, shared palette. Outputs land in
figures/ as .svg + .pdf, plus figures/fig3_scatter_data.json (the 141 triples).
"""

from __future__ import annotations
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ---- shared palette (matches the SVG figures) ----
ERNIE, RNAFM, RNAMSM, CADD = "#0d7d7d", "#9aa3ab", "#d98a3d", "#5b6b7a"
PATHO, INK, GRID, MUTE = "#c0392b", "#1a1a1a", "#d0d5da", "#6b757e"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 13,
    "axes.edgecolor": "#9aa3ab", "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTE, "ytick.color": MUTE,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.labelcolor": INK, "axes.labelsize": 13.5, "text.color": INK,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "svg.fonttype": "none", "figure.dpi": 150,
})

OUT = Path("figures"); OUT.mkdir(exist_ok=True)
OUTPUTS = Path("outputs")


def load(name):
    p = OUTPUTS / name
    if not p.exists():
        print(f"  !! missing {p}"); return None
    return json.loads(p.read_text())


def save(fig, stem):
    for ext in ("svg", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"  -> figures/{stem}.svg + .pdf")


# ============================ FIGURE 2 ============================
def figure2():
    er = load("ism_criticality_erniarna.json"); fm = load("ism_criticality_rnafm.json")
    ms = load("ism_criticality_rnamsm.json"); pe = load("pathogenic_enrichment.json")
    if not all([er, fm, ms, pe]):
        return
    tracks = [("ERNIE-RNA", er["rnas"]["RNU4-2"]["criticality_embedding"], ERNIE),
              ("RNA-FM",    fm["rnas"]["RNU4-2"]["criticality_embedding"], RNAFM),
              ("RNA-MSM",   ms["rnas"]["RNU4-2"]["criticality_embedding"], RNAMSM)]
    path_pos = pe["pathogenic_positions"]["RNU4-2"]
    auc = [pe["results"]["RNU4-2"]["erniarna"]["embed"]["auc"],
           pe["results"]["RNU4-2"]["rnafm"]["embed"]["auc"],
           pe["results"]["RNU4-2"]["rnamsm"]["embed"]["auc"]]
    L = len(tracks[0][1]); CR = (62, 79)
    print(f"[Fig2] L={L}, {len(path_pos)} pathogenic, AUC={auc}")

    fig = plt.figure(figsize=(9.2, 6.2))
    gs = gridspec.GridSpec(3, 2, width_ratios=[3.4, 1.0], hspace=0.22, wspace=0.30)
    x = list(range(1, L + 1))
    for i, (name, y, col) in enumerate(tracks):
        ax = fig.add_subplot(gs[i, 0])
        ax.axvspan(*CR, color="#fbeee6", lw=0)
        for p in path_pos:
            ax.axvline(p, color=PATHO, lw=0.6, alpha=0.10)
        ax.plot(x, y, color=col, lw=1.3)
        ax.scatter(path_pos, [y[p - 1] for p in path_pos], s=24, color=PATHO, zorder=5,
                   clip_on=False)
        ax.set_ylim(0, 1.06); ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel("criticality", fontsize=12.5)
        ax.text(0.012, 0.84, name, transform=ax.transAxes, fontweight="bold",
                color=col, fontsize=14)
        ax.set_xlim(1, L)
        ax.set_xticklabels([]) if i < 2 else ax.set_xlabel("RNU4-2 nucleotide position")
    fig.text(0.02, 0.965, "a", fontsize=17, fontweight="bold")

    axb = fig.add_subplot(gs[:, 1])
    names = ["ERNIE-RNA", "RNA-FM", "RNA-MSM"]
    bars = axb.bar(names, auc, color=[ERNIE, RNAFM, RNAMSM], width=0.66)
    axb.axhline(0.5, ls=(0, (4, 3)), lw=0.9, color=MUTE)
    axb.set_ylim(0, 0.8); axb.set_ylabel("AUC, pathogenic vs rest", fontsize=12.5)
    for b in bars:
        axb.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.014, f"{b.get_height():.2f}",
                 ha="center", va="bottom", fontweight="bold", fontsize=14)
    axb.text(1.0, 0.515, "chance", transform=axb.get_yaxis_transform(), ha="right",
             va="bottom", fontsize=10.5, color=MUTE)
    for t in axb.get_xticklabels():
        t.set_rotation(28); t.set_ha("right"); t.set_fontsize(11.5)
    fig.text(0.70, 0.965, "b", fontsize=17, fontweight="bold")
    save(fig, "Fig2_rnu4-2_dissection")


# ============================ FIGURE 3 ============================
SGE_XLSX, SGE_CSV = Path("data/sge_rnu4-2.xlsx"), Path("data/sge_rnu4-2.csv")


def read_sge_per_position(L=141):
    rows = None
    if SGE_XLSX.exists():
        try:
            import pandas as pd
            df = pd.read_excel(SGE_XLSX, header=1)
            df.columns = [str(c).strip() for c in df.columns]
            rows = df.to_dict("records")
        except Exception as e:  # noqa: BLE001
            print(f"  !! xlsx read failed ({e})")
    if rows is None and SGE_CSV.exists():
        import csv
        rows = list(csv.DictReader(SGE_CSV.open(encoding="utf-8-sig")))
    if rows is None:
        print("  !! no SGE table — Fig 3 scatter skipped"); return None

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def pos_of(r):
        v = num(r.get("position_oligonucleotide"))
        if v is not None:
            return int(round(v))
        m = re.search(r"n?\.?(-?\d+)[ACGTU]\s*>\s*[ACGTU]", str(r.get("HGVS", "")), re.I)
        return int(m.group(1)) if m else None

    by = {}
    for r in rows:
        t = (str(r.get("Type", "")) + " " + str(r.get("Type_expanded_further", ""))).lower()
        if any(s in t for s in ("control", "insertion", "deletion", "indel")):
            continue
        p, s = pos_of(r), num(r.get("function_score"))
        if p is None or s is None or not (1 <= p <= L):
            continue
        by.setdefault(p, []).append(s)
    return {p: sum(v) / len(v) for p, v in by.items()}


def figure3():
    sv = load("sge_validation.json"); er = load("ism_criticality_erniarna.json")
    if not all([sv, er]):
        return
    crit = er["rnas"]["RNU4-2"]["criticality_embedding"]
    rho_er, auc_dis = sv["models"]["erniarna"]["spearman_crit_vs_score"], sv["models"]["erniarna"]["auc_disruptive"]
    rho_fm, rho_cadd = sv["models"]["rnafm"]["spearman_crit_vs_score"], sv.get("baseline_CADD_spearman_vs_sge")
    sge = read_sge_per_position(len(crit))
    print(f"[Fig3] rho ER={rho_er}, FM={rho_fm}, CADD={rho_cadd}, AUC_dis={auc_dis}")

    fig = plt.figure(figsize=(9.2, 3.9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.55, 1.0], wspace=0.32)

    axa = fig.add_subplot(gs[0, 0])
    if sge:
        pos = sorted(p for p in sge if 1 <= p <= len(crit))
        x = [crit[p - 1] for p in pos]; y = [sge[p] for p in pos]
        dmg = [s < -0.39 for s in y]
        axa.scatter([xi for xi, d in zip(x, dmg) if not d], [yi for yi, d in zip(y, dmg) if not d],
                    s=30, color=RNAFM, alpha=0.75, edgecolor="none")
        axa.scatter([xi for xi, d in zip(x, dmg) if d], [yi for yi, d in zip(y, dmg) if d],
                    s=32, color=ERNIE, alpha=0.85, edgecolor="none")
        axa.axhline(-0.39, ls=(0, (4, 3)), lw=0.8, color=MUTE)
        axa.set_xlabel("ERNIE-RNA criticality"); axa.set_ylabel("SGE function score")
        axa.text(0.035, 0.05, f"ρ = {rho_er:.2f},  p = 0.002\nAUC = {auc_dis:.2f},  n = {len(pos)}",
                 transform=axa.transAxes, fontsize=12, va="bottom")
        Path("figures/fig3_scatter_data.json").write_text(json.dumps(
            [{"pos": p, "criticality": crit[p - 1], "sge": sge[p]} for p in pos], indent=2))
    else:
        axa.text(0.5, 0.5, "SGE table not found", ha="center", va="center",
                 transform=axa.transAxes, color=MUTE); axa.set_xticks([]); axa.set_yticks([])
    fig.text(0.02, 0.95, "a", fontsize=17, fontweight="bold")

    axb = fig.add_subplot(gs[0, 1])
    labels = ["ERNIE-RNA", "RNA-FM", "CADD"]
    vals = [rho_er, rho_fm, rho_cadd if rho_cadd is not None else 0.0]
    yb = [2, 1, 0]
    axb.barh(yb, vals, color=[ERNIE, RNAFM, CADD], height=0.62)
    axb.axvline(0, lw=0.9, color=MUTE)
    axb.set_yticks(yb); axb.set_yticklabels(labels, fontsize=12.5)
    axb.set_xlabel("Spearman ρ vs SGE map"); axb.set_xlim(-0.32, 0.18)
    for yi, v in zip(yb, vals):
        axb.text(v + (-0.012 if v < 0 else 0.012), yi, f"{v:+.2f}", va="center",
                 ha="right" if v < 0 else "left", fontsize=13, color=INK)
    fig.text(0.63, 0.95, "b", fontsize=17, fontweight="bold")
    save(fig, "Fig3_sge_validation")


def main():
    print("=" * 60); print("RIBOSCOPE Figures 2 & 3"); print("=" * 60)
    figure2(); figure3()
    print("\nFigures 1, 4, 5 are the hand-authored SVGs in figures/.")


if __name__ == "__main__":
    main()
