"""
40_fetch_plasmodium_ncrna.py — RIBOSCOPE (malaria scope) step P1: fetch
P. falciparum structured ncRNAs via NCBI E-utilities (reliable; the RNAcentral
API hung on filtered queries).

esearch (nuccore) for P. falciparum structured ncRNAs -> efetch FASTA in batches.
We keep short structured RNAs (tRNA/snoRNA/snRNA/etc.) — tRNAs double as a great
in-distribution positive control for the next step's feasibility gate (the models
definitely know tRNA). lncRNAs are excluded by the length filter.

Run with
--------
    cd ~/projects/riboscope
    uv run python 40_fetch_plasmodium_ncrna.py

Output: sequences/plasmodium_ncrna.fasta   (>{accession}|{type}|{description})
        outputs/plasmodium_ncrna_manifest.json
"""

from __future__ import annotations

import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TERM = ('"Plasmodium falciparum"[Organism] AND ('
        'biomol_snorna[prop] OR biomol_snrna[prop] OR biomol_trna[prop] OR biomol_rrna[prop] '
        'OR biomol_ncrna[prop] OR snoRNA[Title] OR "small nucleolar RNA"[Title] '
        'OR "spliceosomal RNA"[Title] OR "transfer RNA"[Title])')
RETMAX = 800
BATCH = 100
MIN_LEN, MAX_LEN = 40, 510
OUT_FASTA = Path("sequences/plasmodium_ncrna.fasta")
MANIFEST = Path("outputs/plasmodium_ncrna_manifest.json")
from entrez_config import get_entrez_email
TOOL, EMAIL = "riboscope", get_entrez_email()
UA = f"{TOOL} (mailto:{EMAIL})"


def ssl_ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        c = ssl.create_default_context(); c.check_hostname = False; c.verify_mode = ssl.CERT_NONE
        return c


CTX = ssl_ctx()


def eutils(endpoint, params):
    params = {**params, "tool": TOOL, "email": EMAIL}
    url = f"{EUTILS}/{endpoint}?{urllib.parse.urlencode(params)}"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(2.0 * (attempt + 1))


def infer_type(desc: str) -> str:
    d = desc.lower()
    if "small nucleolar" in d or "snorna" in d:
        return "snoRNA"
    if "spliceosomal" in d or "snrna" in d or " u1 " in d or " u2 " in d or " u4 " in d or " u5 " in d or " u6 " in d:
        return "snRNA"
    if "transfer rna" in d or "trna" in d or "tRNA" in desc:
        return "tRNA"
    if "ribosomal" in d or "rrna" in d:
        return "rRNA"
    if "srp" in d or "signal recognition" in d:
        return "SRP_RNA"
    if "rnase p" in d or "ribonuclease p" in d:
        return "RNase_P"
    return "ncRNA"


def parse_multifasta(text):
    recs, name, buf = [], None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if name is not None:
                recs.append((name, "".join(buf)))
            name = line[1:].strip()
            buf = []
        else:
            buf.append(line.strip())
    if name is not None:
        recs.append((name, "".join(buf)))
    return recs


def main():
    OUT_FASTA.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("RIBOSCOPE P1: fetch P. falciparum structured ncRNAs (NCBI E-utilities)")
    print("=" * 72)

    print("[1/3] esearch ...")
    s = eutils("esearch.fcgi", {"db": "nuccore", "term": TERM, "retmax": RETMAX, "retmode": "json"})
    try:
        ids = json.loads(s)["esearchresult"]["idlist"]
    except Exception:  # noqa: BLE001
        print(f"❌ Could not parse esearch result. First 300 chars:\n{s[:300]}")
        sys.exit(1)
    print(f"      esearch returned {len(ids)} records")
    if not ids:
        print("⚠ No records — the term may be too strict. We'll broaden it / try PlasmoDB.")
        sys.exit(1)

    print(f"[2/3] efetch FASTA in batches of {BATCH} ...")
    records = []
    for i in range(0, len(ids), BATCH):
        txt = eutils("efetch.fcgi", {"db": "nuccore", "id": ",".join(ids[i:i + BATCH]),
                                     "rettype": "fasta", "retmode": "text"})
        records.extend(parse_multifasta(txt))
        time.sleep(0.4)
    print(f"      fetched {len(records)} sequences")

    print("[3/3] filtering + writing ...")
    kept = []
    for header, seq in records:
        seq = seq.upper().replace("T", "U")
        seq = "".join(c if c in "ACGUN" else "N" for c in seq)
        if not (MIN_LEN <= len(seq) <= MAX_LEN):
            continue
        acc = header.split()[0]
        desc = header[len(acc):].strip()
        kept.append((acc, infer_type(desc), desc[:80], seq))

    with open(OUT_FASTA, "w") as f:
        for acc, rtype, desc, seq in kept:
            f.write(f">{acc}|{rtype}|{desc}\n{seq}\n")

    by_type = Counter(r[1] for r in kept)
    print(f"\nKept {len(kept)} structured ncRNAs (len {MIN_LEN}-{MAX_LEN}). By type:")
    for t, n in by_type.most_common():
        print(f"   {t:<10} {n}")
    with open(MANIFEST, "w") as f:
        json.dump({"term": TERM, "n_esearch": len(ids), "n_fetched": len(records),
                   "n_kept": len(kept), "by_type": dict(by_type)}, f, indent=2)
    print(f"\nWrote {OUT_FASTA} and {MANIFEST}.")
    if len(kept) < 15:
        print("⚠ Few sequences — tell me and I'll broaden the query or add PlasmoDB.")
    else:
        print("Next: P2 feasibility gate — do the models recognize these AT-rich Plasmodium ncRNAs?")


if __name__ == "__main__":
    main()
