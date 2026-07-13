"""
59_render_svg_figures.py — emit Figures 2 and 3 as hand-styled SVG.

Writes clean, paper-idiom SVG (matching the hand-authored figures/Fig1,4,5) but
computes every coordinate in code from the locked result JSONs, so there is no
hand-transcription and every point is exact.

Run in the WSL repo:
    cd ~/projects/riboscope
    uv run python 59_render_svg_figures.py
(needs pandas + openpyxl only for Fig 3's SGE table; no matplotlib)

Outputs: figures/Fig2_rnu4-2_dissection.svg, figures/Fig3_sge_validation.svg
"""
from __future__ import annotations
import json, re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

# palette (identical to the hand-authored SVGs)
ERNIE, RNAFM, RNAMSM, CADD = "#0d7d7d", "#9aa3ab", "#d98a3d", "#5b6b7a"
PATHO, INK, GRID, MUTE, CRBG = "#c0392b", "#1a1a1a", "#d0d5da", "#6b757e", "#fbeee6"
FONT = "Helvetica Neue, Arial, sans-serif"
OUT = Path("figures"); OUT.mkdir(exist_ok=True)
OUTPUTS = Path("outputs")


def load(n):
    p = OUTPUTS / n
    return json.loads(p.read_text()) if p.exists() else None


def f(x):  # short float
    return f"{x:.2f}".rstrip("0").rstrip(".")


def rho_lbl(v):  # round half up to 2 dp so -0.105 -> -0.11 (matches manuscript)
    return f'{Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP):+.2f}'


def txt(x, y, s, size=13, fill=INK, anchor="start", weight="normal", italic=False, rot=None):
    st = f'font-style="italic" ' if italic else ""
    tr = f'transform="rotate({rot} {x} {y})" ' if rot is not None else ""
    return (f'<text x="{f(x)}" y="{f(y)}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" {st}{tr}>{s}</text>')


def line(x1, y1, x2, y2, stroke=MUTE, w=0.9, dash=None, op=1):
    d = f'stroke-dasharray="{dash}" ' if dash else ""
    return (f'<line x1="{f(x1)}" y1="{f(y1)}" x2="{f(x2)}" y2="{f(y2)}" '
            f'stroke="{stroke}" stroke-width="{w}" {d}opacity="{op}"/>')


# ============================ FIGURE 2 ============================
def figure2():
    er = load("ism_criticality_erniarna.json"); fm = load("ism_criticality_rnafm.json")
    ms = load("ism_criticality_rnamsm.json"); pe = load("pathogenic_enrichment.json")
    if not all([er, fm, ms, pe]):
        print("  Fig2: missing inputs"); return
    tracks = [("ERNIE-RNA", er["rnas"]["RNU4-2"]["criticality_embedding"], ERNIE),
              ("RNA-FM",    fm["rnas"]["RNU4-2"]["criticality_embedding"], RNAFM),
              ("RNA-MSM",   ms["rnas"]["RNU4-2"]["criticality_embedding"], RNAMSM)]
    path_pos = pe["pathogenic_positions"]["RNU4-2"]
    auc = [pe["results"]["RNU4-2"]["erniarna"]["embed"]["auc"],
           pe["results"]["RNU4-2"]["rnafm"]["embed"]["auc"],
           pe["results"]["RNU4-2"]["rnamsm"]["embed"]["auc"]]
    L = len(tracks[0][1]); CR = (62, 79)

    W, H = 940, 600
    PX0, PX1 = 92, 660            # track plot x-range
    def X(p): return PX0 + (p - 1) / (L - 1) * (PX1 - PX0)
    TOP = [70, 240, 410]; TH = 140   # track tops, height
    def Y(v, top): return top + (1 - v) * TH

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="{FONT}">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         txt(24, 40, "a", 17, INK, weight="700")]

    for (name, y, col), top in zip(tracks, TOP):
        # CR shading
        s.append(f'<rect x="{f(X(CR[0]))}" y="{top}" width="{f(X(CR[1])-X(CR[0]))}" height="{TH}" fill="{CRBG}"/>')
        # pathogenic vertical lines
        for p in path_pos:
            s.append(line(X(p), top, X(p), top + TH, PATHO, 0.7, op=0.12))
        # axes
        s.append(line(PX0, top, PX0, top + TH, MUTE, 0.9))
        s.append(line(PX0, top + TH, PX1, top + TH, MUTE, 0.9))
        for v in (0, 0.5, 1):
            s.append(line(PX0 - 4, Y(v, top), PX0, Y(v, top), MUTE, 0.9))
            s.append(txt(PX0 - 8, Y(v, top) + 4, f(v), 11.5, MUTE, "end"))
        s.append(txt(PX0 - 40, top + TH / 2, "criticality", 12.5, INK, "middle", rot=-90))
        # criticality polyline (non-scaling stroke keeps it crisp)
        pts = " ".join(f"{f(X(i+1))},{f(Y(val, top))}" for i, val in enumerate(y))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.4"/>')
        # red dots at pathogenic positions
        for p in path_pos:
            s.append(f'<circle cx="{f(X(p))}" cy="{f(Y(y[p-1], top))}" r="3.6" fill="{PATHO}"/>')
        s.append(f'<text x="{f(PX0+8)}" y="{f(top+20)}" font-size="14" fill="{col}" '
                 f'font-weight="700" stroke="#ffffff" stroke-width="3.4" paint-order="stroke" '
                 f'stroke-linejoin="round">{name}</text>')
    # x-axis labels under bottom track
    bt = TOP[2] + TH
    for p in (1, 25, 50, 75, 100, 125, L):
        s.append(line(X(p), bt, X(p), bt + 4, MUTE, 0.9))
        s.append(txt(X(p), bt + 18, str(p), 11.5, MUTE, "middle"))
    s.append(txt((PX0 + PX1) / 2, bt + 38, "RNU4-2 nucleotide position", 13.5, INK, "middle"))
    # pathogenic legend cue
    s.append(f'<circle cx="{PX1-150}" cy="{TOP[0]-18}" r="3.6" fill="{PATHO}"/>')
    s.append(txt(PX1 - 142, TOP[0] - 14, "ClinVar-pathogenic position", 11.5, MUTE))

    # ---- AUC bar panel (right) ----
    BX0, BX1 = 730, 910; BBOT, BTOP = 510, 70; SC = 0.8
    s.append(txt(700, 40, "b", 17, INK, weight="700"))
    s.append(line(BX0, BTOP, BX0, BBOT, MUTE, 0.9))
    s.append(line(BX0, BBOT, BX1, BBOT, MUTE, 0.9))
    for v in (0, 0.4, 0.8):
        yy = BBOT - v / SC * (BBOT - BTOP)
        s.append(line(BX0 - 4, yy, BX0, yy, MUTE, 0.9))
        s.append(txt(BX0 - 8, yy + 4, f(v), 11.5, MUTE, "end"))
    s.append(txt(BX0 - 44, (BTOP + BBOT) / 2, "AUC, pathogenic vs rest", 12.5, INK, "middle", rot=-90))
    ch = BBOT - 0.5 / SC * (BBOT - BTOP)
    s.append(line(BX0, ch, BX1, ch, MUTE, 0.9, dash="4 3", op=0.7))
    s.append(txt(BX1, ch - 5, "chance", 11, MUTE, "end"))
    names = ["ERNIE-RNA", "RNA-FM", "RNA-MSM"]; cols = [ERNIE, RNAFM, RNAMSM]
    bw, slot = 40, (BX1 - BX0) / 3
    for i, (n, a, c) in enumerate(zip(names, auc, cols)):
        cx = BX0 + slot * (i + 0.5); h = a / SC * (BBOT - BTOP)
        s.append(f'<rect x="{f(cx-bw/2)}" y="{f(BBOT-h)}" width="{bw}" height="{f(h)}" fill="{c}"/>')
        s.append(txt(cx, BBOT - h - 8, f"{a:.2f}", 14, c, "middle", weight="700"))
        s.append(txt(cx, BBOT + 18, n, 11, c if c != RNAFM else INK, "middle", weight="700", rot=-18))
    s.append("</svg>")
    (OUT / "Fig2_rnu4-2_dissection.svg").write_text("\n".join(s))
    print(f"  -> figures/Fig2_rnu4-2_dissection.svg  (AUC {auc}, {len(path_pos)} pathogenic)")


# ============================ FIGURE 3 ============================
def read_sge(L=141):
    import pandas as pd
    df = pd.read_excel("data/sge_rnu4-2.xlsx", header=1)
    df.columns = [str(c).strip() for c in df.columns]
    by = {}
    for r in df.to_dict("records"):
        t = (str(r.get("Type", "")) + str(r.get("Type_expanded_further", ""))).lower()
        if any(k in t for k in ("control", "insertion", "deletion", "indel")):
            continue
        try:
            p = int(round(float(r["position_oligonucleotide"]))); v = float(r["function_score"])
        except (TypeError, ValueError, KeyError):
            continue
        if 1 <= p <= L:
            by.setdefault(p, []).append(v)
    return {p: sum(v) / len(v) for p, v in by.items()}


def figure3():
    sv = load("sge_validation.json"); er = load("ism_criticality_erniarna.json")
    if not all([sv, er]):
        print("  Fig3: missing inputs"); return
    crit = er["rnas"]["RNU4-2"]["criticality_embedding"]
    rho_er = sv["models"]["erniarna"]["spearman_crit_vs_score"]
    auc_d = sv["models"]["erniarna"]["auc_disruptive"]
    rho_fm = sv["models"]["rnafm"]["spearman_crit_vs_score"]
    rho_c = sv.get("baseline_CADD_spearman_vs_sge", 0.0)
    try:
        sge = read_sge(len(crit))
    except Exception as e:  # noqa: BLE001
        print(f"  Fig3: SGE table unreadable ({e})"); return
    pos = sorted(p for p in sge if 1 <= p <= len(crit))
    xs = [crit[p - 1] for p in pos]; ys = [sge[p] for p in pos]
    ymin, ymax = min(ys) - 0.1, max(ys) + 0.1

    W, H = 900, 420
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="{FONT}">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    # ---- panel a: scatter ----
    AX0, AX1, AY0, AY1 = 80, 470, 50, 330
    XLO = min(0.6, min(xs) - 0.02)
    def SX(c): return AX0 + (c - XLO) / (1.0 - XLO) * (AX1 - AX0)
    def SY(v): return AY1 - (v - ymin) / (ymax - ymin) * (AY1 - AY0)
    s.append(txt(20, 38, "a", 17, INK, weight="700"))
    s.append(line(AX0, AY0, AX0, AY1, MUTE, 0.9)); s.append(line(AX0, AY1, AX1, AY1, MUTE, 0.9))
    for c in [t for t in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0) if t >= XLO - 1e-9]:
        s.append(line(SX(c), AY1, SX(c), AY1 + 4, MUTE, 0.9)); s.append(txt(SX(c), AY1 + 18, f(c), 11.5, MUTE, "middle"))
    for v in (0, -0.5, -1, -1.5, -2):
        if ymin <= v <= ymax:
            s.append(line(AX0 - 4, SY(v), AX0, SY(v), MUTE, 0.9)); s.append(txt(AX0 - 8, SY(v) + 4, str(v), 11.5, MUTE, "end"))
    s.append(line(AX0, SY(-0.39), AX1, SY(-0.39), MUTE, 0.8, dash="4 3", op=0.7))
    s.append(txt(AX1, SY(-0.39) - 5, "depleted", 10.5, MUTE, "end"))
    for c, v in zip(xs, ys):
        col = ERNIE if v < -0.39 else RNAFM
        s.append(f'<circle cx="{f(SX(c))}" cy="{f(SY(v))}" r="4.2" fill="{col}" opacity="0.8"/>')
    s.append(txt((AX0 + AX1) / 2, AY1 + 38, "ERNIE-RNA criticality", 13.5, INK, "middle"))
    s.append(txt(AX0 - 46, (AY0 + AY1) / 2, "SGE function score", 13.5, INK, "middle", rot=-90))
    s.append(txt(AX0 + 14, AY0 + 22, f"ρ = {rho_er:.2f},  p = 0.002", 13, INK))
    s.append(txt(AX0 + 14, AY0 + 40, f"AUC = {auc_d:.2f},  n = {len(pos)}", 13, INK))

    # ---- panel b: rho vs SGE truth (vertical bars, robust layout) ----
    s.append(txt(540, 38, "b", 17, INK, weight="700"))
    PBX0, PBX1, PBTOP, PBBOT = 600, 860, 70, 290
    RHI, RLO = 0.2, -0.3
    def RY(r): return PBTOP + (RHI - r) / (RHI - RLO) * (PBBOT - PBTOP)
    zy = RY(0)
    s.append(line(PBX0 - 4, PBTOP, PBX0 - 4, PBBOT, MUTE, 0.9))
    for r in (0.2, 0.1, 0.0, -0.1, -0.2, -0.3):
        s.append(line(PBX0 - 8, RY(r), PBX0 - 4, RY(r), MUTE, 0.9))
        s.append(txt(PBX0 - 12, RY(r) + 4, ("0" if r == 0 else f"{r:+.1f}"), 11, MUTE, "end"))
    s.append(line(PBX0 - 4, zy, PBX1, zy, MUTE, 0.9))
    s.append(txt(PBX0 - 48, (PBTOP + PBBOT) / 2, "Spearman ρ vs SGE map", 12.5, INK, "middle", rot=-90))
    labels = ["ERNIE-RNA", "RNA-FM", "CADD"]; vals = [rho_er, rho_fm, rho_c]; cols = [ERNIE, RNAFM, CADD]
    slot = (PBX1 - PBX0) / 3; bw = 46
    for i, (lab, v, c) in enumerate(zip(labels, vals, cols)):
        cx = PBX0 + slot * (i + 0.5)
        ytop = min(zy, RY(v)); h = abs(RY(v) - zy)
        s.append(f'<rect x="{f(cx-bw/2)}" y="{f(ytop)}" width="{bw}" height="{f(h)}" fill="{c}"/>')
        vy = RY(v) - 8 if v >= 0 else RY(v) + 18
        s.append(txt(cx, vy, rho_lbl(v), 13.5, (INK if c == RNAFM else c), "middle", weight="700"))
        s.append(txt(cx, PBBOT + 22, lab, 12, (INK if c == RNAFM else c), "middle", weight="700"))
    s.append(txt((PBX0 + PBX1) / 2, PBBOT + 44, "negative ρ = criticality tracks depletion", 11.5, MUTE, "middle"))
    s.append("</svg>")
    (OUT / "Fig3_sge_validation.svg").write_text("\n".join(s))
    print(f"  -> figures/Fig3_sge_validation.svg  (rho_er {rho_er}, n {len(pos)})")


if __name__ == "__main__":
    print("Rendering Fig 2 and Fig 3 as SVG ...")
    figure2(); figure3()
    print("Done. Fig 1, 4, 5 are the existing hand-authored SVGs.")
