"""
60_build_paper.py — assemble the manuscript + 5 figures into one printable file.

Produces RIBOSCOPE_paper.html with the figures embedded inline (self-contained),
styled like a paper. Open it in any browser and File -> Print -> Save as PDF.
This avoids LaTeX/weasyprint system dependencies entirely.

Setup (one time): make sure all five figure SVGs sit in ./figures/ next to the
manuscript. Fig1/4/5 are the hand-authored SVGs; copy Fig2/3 in from the WSL run:
    cp ~/projects/riboscope/figures/Fig2_rnu4-2_dissection.svg \
       ~/projects/riboscope/figures/Fig3_sge_validation.svg ./figures/

Run (from the folder that holds RIBOSCOPE_manuscript.md):
    uv pip install markdown
    uv run python code/60_build_paper.py
Then open RIBOSCOPE_paper.html and print to PDF.
"""
from __future__ import annotations
import re
from pathlib import Path

try:
    import markdown
except ImportError:
    raise SystemExit("Need the markdown library:  uv pip install markdown")

MD = Path("RIBOSCOPE_manuscript.md")
FIGDIR = Path("figures")
OUT = Path("RIBOSCOPE_paper.html")
FIGS = {1: "Fig1_method.svg", 2: "Fig2_rnu4-2_dissection.svg",
        3: "Fig3_sge_validation.svg", 4: "Fig4_triage.svg",
        5: "Fig5_panel_specificity.svg"}

if not MD.exists():
    raise SystemExit(f"manuscript not found: {MD.resolve()} (run from the folder holding it)")

html_body = markdown.markdown(MD.read_text(encoding="utf-8"),
                              extensions=["tables", "sane_lists"])

# inline each figure SVG just before its legend paragraph (<strong>Figure N.)
for n, fn in FIGS.items():
    p = FIGDIR / fn
    if not p.exists():
        print(f"  !! missing {p} — figure {n} will be absent from the PDF")
        continue
    svg = re.sub(r"<\?xml.*?\?>", "", p.read_text(encoding="utf-8"), flags=re.S).strip()
    block = f'<figure class="fig">{svg}</figure>'
    marker = f"<p><strong>Figure {n}."
    i = html_body.find(marker)
    html_body = (html_body[:i] + block + html_body[i:]) if i != -1 else (html_body + block)

CSS = """
@page { size: A4; margin: 2cm; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.5pt; line-height: 1.5;
       color: #1a1a1a; max-width: 760px; margin: 0 auto; padding: 24px; }
h1 { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 20pt; line-height: 1.25;
     margin: 0 0 4px; }
h2 { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 13.5pt; margin: 1.6em 0 0.4em;
     padding-bottom: 3px; border-bottom: 1px solid #d8dde1; }
h3 { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 11.5pt; margin: 1.2em 0 0.3em; }
p { margin: 0 0 0.7em; text-align: justify; }
em { color: #333; }
sup { font-size: 0.72em; }
a { color: #0d6e6e; text-decoration: none; }
figure.fig { margin: 1.3em 0; text-align: center; page-break-inside: avoid; }
figure.fig svg { max-width: 100%; height: auto; }
table { border-collapse: collapse; width: 100%; font-size: 8.6pt; margin: 0.6em 0 1em;
        page-break-inside: avoid; }
th, td { border: 1px solid #cfd6da; padding: 3px 6px; text-align: left; vertical-align: top; }
th { background: #f3f6f7; font-family: 'Helvetica Neue', Arial, sans-serif; }
hr { border: none; border-top: 1px solid #e3e8ea; margin: 1.4em 0; }
code { font-family: 'SF Mono', Consolas, monospace; font-size: 0.9em; }
h2:first-of-type { page-break-before: avoid; }
"""

html = ("<!doctype html><html><head><meta charset='utf-8'>"
        "<title>RIBOSCOPE</title><style>" + CSS + "</style></head><body>"
        + html_body + "</body></html>")
OUT.write_text(html, encoding="utf-8")
print(f"Wrote {OUT.resolve()}")
print("Open it in a browser and File -> Print -> Save as PDF.")
