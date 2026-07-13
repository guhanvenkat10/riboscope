"""
45_fetch_disease_structured_rna.py — RIBOSCOPE G4 step 1: fetch the disease-
associated STRUCTURED ncRNAs that the functional-nucleotide ISM program targets.

Why these (vs the G3 lncRNAs, which were OOD and failed)
--------------------------------------------------------
The G3 lncRNA scan went null precisely because lncRNAs (MALAT1/NEAT1/BACE1-AS,
8-22 kb) lack the compact Rfam-like motifs our features detect. G4 deliberately
picks COMPACT, structured, in-distribution (<=510 nt) disease ncRNAs that ARE
the kind of thing the models recognize — each with a KNOWN disease-variant
hotspot we can later validate an unsupervised criticality map against:

  - RNU4-2  (NR_003137.2, 141 nt)  U4 snRNA. ReNU syndrome — de novo variants
              cause ~0.4% of ALL neurodevelopmental disorders (Greene/Chen 2024).
              Hotspot = 18-bp T-loop + Stem III; recurrent n.64_65insT. HEADLINE.
  - RMRP    (NR_003051.4, ~268 nt) RNase MRP RNA. Cartilage-hair hypoplasia;
              133+ pathogenic variants clustering in the conserved P3 domain.
  - TERC    (NR_001566.1, 451 nt)  Telomerase RNA. Dyskeratosis congenita.
  - RPPH1   (NR_002312.1, ~340 nt) RNase P RNA (H1). Structured ribozyme RNA.
  - RN7SL1  (NR_002715.1, ~299 nt) SRP RNA. Structured, deeply conserved.
  - RNU6-1  (NR_004394.1, 106 nt)  U6 snRNA. Base-pairs U4; spliceosome core.

All are single-sequence IN-DISTRIBUTION for RNA-FM and ErnieRNA (Rfam 50-510 nt
training window). RNA-MSM is handled later with native per-RNA MSAs.

Output
------
    sequences/disease_structured_rna.fasta   (header: >{GENE}|{accession}|{desc})
    outputs/disease_rna_hotspots.json        (known functional hotspots; the
                                              validation ground truth for step 47)

Run with
--------
    cd ~/projects/riboscope
    uv run python 45_fetch_disease_structured_rna.py

Notes
-----
  - Uses NCBI E-utilities efetch over HTTPS (same proven pattern as 20).
  - Accessions + expected lengths were verified against NCBI on 2026-06-04.
    A length mismatch only WARNS (NCBI may bump a version); a wrong-organism
    accession is caught by the header-contains-gene check + the length sanity.
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

# ============================ CONFIG ============================
# gene -> (accession, expected_length_nt, one-line disease note)
TARGETS = {
    "RNU4-2": ("NR_003137.2", 141, "U4 snRNA; ReNU neurodevelopmental syndrome"),
    "RMRP":   ("NR_003051.4", 268, "RNase MRP RNA; cartilage-hair hypoplasia"),
    "TERC":   ("NR_001566.1", 451, "Telomerase RNA; dyskeratosis congenita"),
    "RPPH1":  ("NR_002312.1", 340, "RNase P RNA H1; structured ribozyme"),
    "RN7SL1": ("NR_002715.1", 299, "SRP 7SL RNA; signal recognition particle"),
    "RNU6-1": ("NR_004394.1", 106, "U6 snRNA; spliceosome catalytic core"),
}

OUTPUT_FASTA = Path("sequences/disease_structured_rna.fasta")
OUTPUT_HOTSPOTS = Path("outputs/disease_rna_hotspots.json")

# Known functional hotspots = the ground truth the step-47 validation tests the
# unsupervised criticality map against. Coordinates are 1-based on the RefSeq
# above. Where an exact region is not yet pinned, the recurrent variant anchors
# it and the region is marked for refinement at step 47 (do NOT overclaim).
HOTSPOTS = {
    "RNU4-2": {
        "disease": "ReNU syndrome (neurodevelopmental disorder)",
        "recurrent_variant": "n.64_65insT (T-loop) and n.77_78insT (Stem III)",
        "recurrent_pos1": [64, 65, 77, 78],
        "critical_region_desc": "18-bp critical region: T-loop + Stem III of the U4/U6 duplex",
        "critical_region_pos1_approx": [62, 79],
        "ground_truth_ref": "Chen/Greene 2024 Nature s41586-024-07773-7; "
                            "2025 RNU4-2 saturation genome editing (PMC12036422)",
        "note": "Region n.62-79 verified vs literature 2026-06-04.",
    },
    "RMRP": {
        "disease": "Cartilage-hair hypoplasia",
        "recurrent_variant": "n.72A>G (major, ~90%); n.197C>T (Brazilian founder)",
        "recurrent_pos1": [72, 197],
        "critical_region_desc": "conserved core; major variant n.72A>G",
        "critical_region_pos1_approx": None,
        "ground_truth_ref": "Ridanpaa 2001 (5200824); Brazilian founder PMC11166637",
        "note": "n.72=A and n.197=C verified vs fetched RefSeq 2026-06-04.",
    },
    "TERC": {
        "disease": "Dyskeratosis congenita / telomere biology disorders",
        "recurrent_variant": "multiple; CR4/CR5, pseudoknot/P6.1, template region",
        "recurrent_pos1": None,
        "critical_region_desc": "pseudoknot (P2b/P3) + CR4/CR5 protein-binding domain",
        "critical_region_pos1_approx": None,
        "ground_truth_ref": "Telomerase RNA disease-variant literature",
        "note": "Map exact pathogenic-variant positions from ClinVar/literature at step 47.",
    },
    "RPPH1":  {"disease": "(control — structured ribozyme, no single hotspot)"},
    "RN7SL1": {"disease": "(control — deeply conserved structured RNA)"},
    "RNU6-1": {"disease": "(partner of U4; spliceosome core)"},
}

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TOOL = "riboscope"
from entrez_config import get_entrez_email
EMAIL = get_entrez_email()
SLEEP_BETWEEN = 0.5
N_RETRIES = 3
TIMEOUT = 30
MAX_INDIST_LEN = 510   # Rfam training upper bound; >this is OOD for single-seq SAEs
# ================================================================


def efetch_fasta(accession: str) -> str:
    params = {
        "db": "nuccore", "id": accession, "rettype": "fasta", "retmode": "text",
        "tool": TOOL, "email": EMAIL,
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
    lines = text.splitlines()
    header = lines[0][1:].strip()
    seq = "".join(l.strip() for l in lines[1:] if l and not l.startswith(">"))
    return header, seq


def main() -> None:
    print("=" * 74)
    print("RIBOSCOPE G4 step 1: fetch disease STRUCTURED ncRNAs")
    print("=" * 74)
    OUTPUT_FASTA.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HOTSPOTS.parent.mkdir(parents=True, exist_ok=True)

    records: list[tuple[str, str, str, str]] = []
    problems = 0
    for gene, (acc, exp_len, note) in TARGETS.items():
        print(f"\n[fetch] {gene:<8} {acc:<14} — {note}")
        try:
            raw = efetch_fasta(acc)
        except Exception as e:  # noqa: BLE001
            print(f"   ❌ {e}\n      Skipping {gene}; fix the accession and re-run.")
            problems += 1
            continue
        header, seq = parse_single_fasta(raw)
        seq_clean = seq.upper().replace(" ", "")
        L = len(seq_clean)
        print(f"   header: {header}")
        print(f"   length: {L} nt (expected ~{exp_len})")
        # sanity checks (warn, do not abort — but flag loudly)
        gene_root = gene.split("-")[0].split("_")[0].upper()
        if gene_root not in header.upper():
            print(f"   ⚠ header lacks {gene_root!r} — VERIFY accession.")
            problems += 1
        if abs(L - exp_len) > 5:
            print(f"   ⚠ length {L} differs from expected {exp_len} by >5 — VERIFY accession.")
            problems += 1
        if L > MAX_INDIST_LEN:
            print(f"   ⚠ {L} nt > {MAX_INDIST_LEN}: OUT-OF-DISTRIBUTION for single-seq SAEs.")
        if any(c not in "ACGTU" for c in seq_clean):
            bad = sorted(set(c for c in seq_clean if c not in "ACGTU"))
            print(f"   ⚠ non-ACGTU characters present: {bad}")
        records.append((gene, acc, header, seq_clean))
        time.sleep(SLEEP_BETWEEN)

    if not records:
        print("\n❌ No records fetched. Check network/accessions and re-run.")
        sys.exit(1)

    with open(OUTPUT_FASTA, "w") as f:
        for gene, acc, header, seq in records:
            desc = header.replace("|", " ").strip()
            f.write(f">{gene}|{acc}|{desc}\n")
            for i in range(0, len(seq), 70):
                f.write(seq[i:i + 70] + "\n")

    # hotspots sidecar (only for genes we actually fetched)
    hs = {g: HOTSPOTS.get(g, {}) for g, *_ in [(r[0],) for r in records]}
    with open(OUTPUT_HOTSPOTS, "w") as f:
        json.dump({"genes": hs, "coord_system": "1-based on the fetched RefSeq",
                   "note": "Hotspots are the step-47 validation ground truth."}, f, indent=2)

    print("\n" + "=" * 74)
    print(f"✅ Wrote {len(records)} sequence(s) → {OUTPUT_FASTA}")
    for gene, acc, _, seq in records:
        print(f"   {gene:<8} {acc:<14} {len(seq):>5} nt")
    print(f"✅ Wrote hotspots → {OUTPUT_HOTSPOTS}")
    if problems:
        print(f"\n⚠ {problems} sanity warning(s) above — eyeball them before proceeding.")
    print("\nNext: fingerprint which RNAs the models recognize:")
    print("   uv run python 46_recon_disease_features.py rnafm")
    print("   uv run python 46_recon_disease_features.py erniarna")
    print("=" * 74)


if __name__ == "__main__":
    main()
