"""
20_fetch_disease_lncrna.py — RIBOSCOPE G3 (disease-discovery) step 1.

Downloads the disease-associated human lncRNA transcripts we will scan with
the locked SAEs, and writes a single FASTA in the same shape the rest of the
pipeline expects.

Why these three (the "let the data decide" screen)
--------------------------------------------------
  - MALAT1   (NR_002819.5, ~8779 nt) — cancer-metastasis lncRNA, drug target.
               Has KNOWN 3' elements (mascRNA = tRNA-like, + triple helix/ENE)
               that serve as built-in POSITIVE CONTROLS for the scanner.
  - NEAT1_2  (NR_131012.1, ~22.7 kb) — paraspeckle scaffold; cancer + ALS.
               (Long isoform; NEAT1_1 is just its 5' subset.)
  - BACE1-AS (NR_024766.1, ~?)       — Alzheimer's antisense lncRNA.

These are LONG and OUT-OF-DISTRIBUTION relative to the Rfam 50-510 nt training
families. That is exactly why the scanner (21_scan_disease_rna.py) tiles them
into in-distribution windows. Do NOT feed full length to the models.

Output
------
    sequences/disease_lncrna.fasta
      header convention:  >{GENE}|{accession}|{description}
      sequence stored as-is from RefSeq (T's); downstream parsers convert T->U.

Run with
--------
    cd ~/projects/riboscope
    uv run python 20_fetch_disease_lncrna.py

Notes
-----
  - Uses NCBI E-utilities efetch over HTTPS (no Biopython dependency).
  - If an accession 404s or NCBI changes a version suffix, the script reports
    it and continues with the others; just swap the accession below and re-run.
  - Prints each fetched header + length so you can eyeball that the accession
    really is the gene you expect (verification, per project rules).
"""

import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

# ============================ CONFIG ============================
# gene label -> RefSeq accession.  Edit here if an accession needs swapping.
ACCESSIONS = {
    "MALAT1":   "NR_002819.5",   # verified: MALAT1 variant 1, ~7.5 kb
    "NEAT1_2":  "NR_131012.1",   # verified: NEAT1 variant MENbeta (long isoform), ~22.7 kb
    "BACE1-AS": "NR_037803.2",   # verified: BACE1 antisense RNA, ~2 kb (was wrongly NR_024766 = bacterial 16S)
}

OUTPUT_FILE = Path("sequences/disease_lncrna.fasta")

# NCBI etiquette: identify the tool + a contact email, throttle requests.
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TOOL = "riboscope"
from entrez_config import get_entrez_email
EMAIL = get_entrez_email()
SLEEP_BETWEEN = 0.5     # seconds; stays under NCBI's 3 req/s unauthenticated cap
N_RETRIES = 3
TIMEOUT = 30
# ================================================================


def efetch_fasta(accession: str) -> str:
    """Fetch a single FASTA record from NCBI nuccore. Returns raw FASTA text."""
    params = {
        "db": "nuccore",
        "id": accession,
        "rettype": "fasta",
        "retmode": "text",
        "tool": TOOL,
        "email": EMAIL,
    }
    url = f"{EUTILS}?{urllib.parse.urlencode(params)}"
    last_err = None
    for attempt in range(1, N_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": f"{TOOL} (mailto:{EMAIL})"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                text = resp.read().decode("utf-8", errors="replace").strip()
            if text.startswith(">"):
                return text
            last_err = f"unexpected response (first 80 chars): {text[:80]!r}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
        if attempt < N_RETRIES:
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"efetch failed for {accession} after {N_RETRIES} tries: {last_err}")


def parse_single_fasta(text: str) -> tuple[str, str]:
    """Parse one FASTA record. Returns (header_without_'>', sequence_no_whitespace)."""
    lines = text.splitlines()
    header = lines[0][1:].strip()
    seq = "".join(l.strip() for l in lines[1:] if l and not l.startswith(">"))
    return header, seq


def main() -> None:
    print("=" * 70)
    print("RIBOSCOPE G3: fetch disease lncRNA transcripts")
    print("=" * 70)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    records: list[tuple[str, str, str, str]] = []  # (gene, accession, description, sequence)
    for gene, acc in ACCESSIONS.items():
        print(f"\n[fetch] {gene}  ({acc}) ...")
        try:
            raw = efetch_fasta(acc)
        except Exception as e:  # noqa: BLE001
            print(f"   ❌ {e}")
            print(f"      Skipping {gene}. Fix the accession in ACCESSIONS and re-run.")
            continue
        header, seq = parse_single_fasta(raw)
        seq_clean = seq.upper().replace(" ", "")
        print(f"   header: {header}")
        print(f"   length: {len(seq_clean):,} nt")
        # Sanity flag: does the header mention the gene we asked for?
        gene_root = gene.split("_")[0].split("-")[0].upper()
        if gene_root not in header.upper():
            print(f"   ⚠ header does not contain {gene_root!r} — VERIFY this accession is correct.")
        records.append((gene, acc, header, seq_clean))
        time.sleep(SLEEP_BETWEEN)

    if not records:
        print("\n❌ No records fetched. Check network / accessions and re-run.")
        sys.exit(1)

    with open(OUTPUT_FILE, "w") as f:
        for gene, acc, header, seq in records:
            desc = header.replace("|", " ").strip()
            f.write(f">{gene}|{acc}|{desc}\n")
            for i in range(0, len(seq), 70):
                f.write(seq[i:i + 70] + "\n")

    print("\n" + "=" * 70)
    print(f"✅ Wrote {len(records)} sequence(s) to {OUTPUT_FILE}")
    for gene, acc, _, seq in records:
        print(f"   {gene:<10} {acc:<14} {len(seq):>7,} nt")
    print("=" * 70)
    print("\nNext: run the scanner for each model:")
    print("   uv run python 21_scan_disease_rna.py rnafm")
    print("   uv run python 21_scan_disease_rna.py erniarna")


if __name__ == "__main__":
    main()
