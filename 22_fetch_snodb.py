"""
22_fetch_snodb.py — RIBOSCOPE G3 (snoRNA discovery) step 1a: fetch + inspect snoDB.

Downloads the full snoDB 2.0 human-snoRNA table and PRINTS ITS SCHEMA so we
parse it correctly (the columns aren't visible from outside WSL). Run this
first; paste the output back, and the feature-extraction script (step 1b) will
be written against the exact column names.

snoDB 2.0 (Scott Lab, U. Sherbrooke) is freely/publicly available — no
permission walls (satisfies the project's data-access rule).

Run with
--------
    cd ~/projects/riboscope
    uv run python 22_fetch_snodb.py

Output
------
    data/snodb_all.tsv         (raw download)
    + console schema report (columns, row count, sample rows, key value counts)
"""

import csv
import io
import ssl
import sys
import urllib.request
from collections import Counter
from pathlib import Path

URL = "https://bioinfo-scottgroup.med.usherbrooke.ca/snoDB/download_all/"
OUT = Path("data/snodb_all.tsv")
from entrez_config import get_entrez_email
UA = f"Mozilla/5.0 (X11; Linux x86_64) riboscope-research (mailto:{get_entrez_email()})"
TIMEOUT = 60


def _ssl_contexts():
    """Try a verified context (certifi, then system), then an unverified fallback.

    Unverified is acceptable here: the data is public and read-only, and we
    sanity-check the downloaded content (columns/rows) before using it.
    """
    ctxs = []
    try:
        import certifi
        ctxs.append(("certifi CA bundle", ssl.create_default_context(cafile=certifi.where())))
    except Exception:  # noqa: BLE001
        pass
    try:
        ctxs.append(("system default CAs", ssl.create_default_context()))
    except Exception:  # noqa: BLE001
        pass
    unver = ssl.create_default_context()
    unver.check_hostname = False
    unver.verify_mode = ssl.CERT_NONE
    ctxs.append(("UNVERIFIED fallback", unver))
    return ctxs


def download() -> str:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for label, ctx in _ssl_contexts():
        try:
            req = urllib.request.Request(URL, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=TIMEOUT, context=ctx) as resp:
                raw = resp.read()
            text = raw.decode("utf-8", errors="replace")
            if text.strip():
                if "UNVERIFIED" in label:
                    print(f"⚠ TLS chain didn't verify; downloaded via {label} "
                          f"(ok for public read-only data; content is checked below).")
                else:
                    print(f"   TLS via {label}.")
                OUT.write_text(text, encoding="utf-8")
                return text
            last = f"empty response ({label})"
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__} via {label}: {e}"
    print(f"❌ Download failed: {last}")
    print(f"   Try opening {URL} in a browser to confirm it serves a TSV.")
    sys.exit(1)


def sniff_delimiter(header_line: str) -> str:
    return "\t" if header_line.count("\t") >= header_line.count(",") else ","


def main() -> None:
    print("=" * 74)
    print("RIBOSCOPE G3: fetch + inspect snoDB 2.0")
    print("=" * 74)
    text = download()
    lines = text.splitlines()
    if not lines:
        print("❌ Downloaded file is empty.")
        sys.exit(1)

    delim = sniff_delimiter(lines[0])
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    rows = list(reader)
    header, data = rows[0], rows[1:]

    print(f"\nSaved: {OUT}  ({len(text)/1e6:.2f} MB)")
    print(f"Delimiter: {'TAB' if delim == chr(9) else 'COMMA'}")
    print(f"Rows (excl header): {len(data):,}   Columns: {len(header)}")

    print(f"\n--- COLUMNS ({len(header)}) ---")
    for i, c in enumerate(header):
        print(f"  [{i:>2}] {c}")

    print("\n--- FIRST 2 DATA ROWS (col: value) ---")
    for r in data[:2]:
        print("  " + "-" * 50)
        for c, v in zip(header, r):
            vv = (v[:70] + "…") if len(v) > 70 else v
            print(f"    {c}: {vv}")

    # Value counts for columns likely to carry box-type / target / function info
    keys = ("box", "type", "class", "target", "rrna", "orphan", "modif", "function", "gene_biotype")
    print("\n--- VALUE COUNTS for function/type-relevant columns ---")
    for i, c in enumerate(header):
        if any(k in c.lower() for k in keys):
            vals = [r[i] if i < len(r) else "" for r in data]
            cnt = Counter(v if v.strip() else "<empty>" for v in vals)
            print(f"\n  [{i}] {c}  ({len(cnt)} distinct):")
            for val, n in cnt.most_common(10):
                vv = (val[:50] + "…") if len(val) > 50 else val
                print(f"      {n:>6}  {vv}")

    # Flag candidate sequence + id columns
    seq_cols = [c for c in header if "seq" in c.lower()]
    id_cols = [c for c in header if c.lower() in ("id", "snodb_id", "gene_id", "ensembl_id", "symbol", "gene_name", "name")]
    print(f"\n  candidate SEQUENCE columns: {seq_cols}")
    print(f"  candidate ID/NAME columns:  {id_cols}")
    print("\n✅ Done. Paste this schema report back so step 1b is written to the exact columns.")


if __name__ == "__main__":
    main()
