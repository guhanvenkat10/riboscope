"""
33_build_snorna_msa.py — RIBOSCOPE step R1: align human snoRNAs into their Rfam
families so RNA-MSM can be run on them in-distribution (native MSA input).

Our snoDB human snoRNAs are NOT rows of the Rfam SEED alignments, so we align
each into its family's SEED columns with MAFFT `--add --keeplength` (adds new
sequences to an existing alignment, preserving the SEED's column coordinates so
the input distribution matches the headline MSA run).

Output: one aligned FASTA per family in outputs/snorna_msa/<rfam>.fasta, with
the SEED rows (>SEED_k) followed by the human queries (>{snodb_id}|{gene}), all
at the SEED width, gaps written as '.' (RNA-MSM convention). Step R2 builds each
snoRNA's MSA as [its query row] + SEED rows.

Requires MAFFT:  conda install -c bioconda mafft   (or: sudo apt install mafft)

Run with
--------
    cd ~/projects/riboscope
    uv run python 33_build_snorna_msa.py --pilot     # 3 families first (validate)
    uv run python 33_build_snorna_msa.py             # all families

Inputs : data/snodb_all.tsv, data/Rfam.seed.gz
Output : outputs/snorna_msa/<rfam>.fasta  +  outputs/snorna_msa_manifest.json
"""

from __future__ import annotations

import csv
import gzip
import json
import random
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

SNODB = Path("data/snodb_all.tsv")
RFAM_SEED = Path("data/Rfam.seed.gz")
OUT_DIR = Path("outputs/snorna_msa")
MANIFEST = Path("outputs/snorna_msa_manifest.json")
MIN_LEN, MAX_LEN = 50, 510
MSA_DEPTH = 32          # query + up to (MSA_DEPTH-1) SEED rows (matches training)
RNG_SEED = 42
PILOT_FAMILIES = 3      # in --pilot, do this many largest families


def nonempty(v):
    return bool(v and str(v).strip() and str(v).strip().lower() != "nan")


def norm_aligned(s: str) -> str:
    """Aligned Stockholm row -> upper, T->U, gaps '-'/'.'->'.', non-ACGU/. -> N."""
    s = s.upper().replace("T", "U").replace("-", ".")
    return "".join(c if c in "ACGU." else "N" for c in s)


def ungap(s: str) -> str:
    return s.replace(".", "")


def parse_rfam_seed_aligned(path: Path, keep: set[str]) -> dict[str, list[str]]:
    """Return {rfam_id: [aligned_row,...]} (gaps kept, '.'), only for `keep`."""
    fams: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="latin-1") as f:
        block = []
        for line in f:
            block.append(line)
            if line.startswith("//"):
                fam_id, rows = _parse_block("".join(block))
                block = []
                if fam_id and fam_id in keep and rows:
                    widths = {len(r) for _, r in rows}
                    if len(widths) == 1:
                        fams[fam_id] = [r for _, r in rows]
    return fams


def _parse_block(text: str):
    bufs: dict[str, str] = {}
    fam_id = None
    for line in text.split("\n"):
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("#=GF AC"):
            fam_id = line.split(maxsplit=2)[2].strip()
        elif line.startswith("#") or line.startswith("//"):
            continue
        else:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                bufs[parts[0]] = bufs.get(parts[0], "") + parts[1]
    return fam_id, [(sid, norm_aligned(aln)) for sid, aln in bufs.items()]


def read_snodb_cd():
    """C/D snoRNAs in length window, with an Rfam id -> {rfam: [(id, gene, ungapped_seq)]}."""
    by_fam = defaultdict(list)
    with open(SNODB, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("box_type", "").strip() != "C/D":
                continue
            seq = (r.get("sequence", "") or "").strip().upper().replace("T", "U")
            seq = "".join(c if c in "ACGUN" else "N" for c in seq)
            if not (MIN_LEN <= len(seq) <= MAX_LEN):
                continue
            rfam = (r.get("rfam_id", "") or "").strip()
            if not nonempty(rfam):
                continue
            by_fam[rfam].append((r["snodb_id"], r.get("gene_name", ""), seq))
    return by_fam


def write_fasta(path, records):
    with open(path, "w") as f:
        for name, seq in records:
            f.write(f">{name}\n{seq}\n")


def mafft_add(seed_aln_records, human_records):
    """mafft --add human --keeplength seed_aln ; returns {name: aligned_seq} for ALL."""
    with tempfile.TemporaryDirectory() as td:
        seed_fa = Path(td) / "seed.fasta"
        add_fa = Path(td) / "add.fasta"
        # MAFFT wants '-' gaps + uppercase letters
        write_fasta(seed_fa, [(n, s.replace(".", "-")) for n, s in seed_aln_records])
        write_fasta(add_fa, [(n, s) for n, s in human_records])
        proc = subprocess.run(
            ["mafft", "--add", str(add_fa), "--keeplength", "--quiet", str(seed_fa)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"mafft failed: {proc.stderr[:300]}")
        # parse FASTA from stdout
        out = {}
        name = None
        buf = []
        for line in proc.stdout.splitlines():
            if line.startswith(">"):
                if name is not None:
                    out[name] = "".join(buf)
                name = line[1:].strip()
                buf = []
            else:
                buf.append(line.strip())
        if name is not None:
            out[name] = "".join(buf)
        return out


def main():
    pilot = "--pilot" in sys.argv
    if shutil.which("mafft") is None:
        print("❌ MAFFT not found. Install it:")
        print("   conda install -c bioconda mafft     (or: sudo apt install mafft)")
        sys.exit(1)
    for p in (SNODB, RFAM_SEED):
        if not p.exists():
            print(f"❌ Missing {p}")
            sys.exit(1)

    print("=" * 72)
    print(f"RIBOSCOPE R1: build snoRNA MSAs (MAFFT --add) {'[PILOT]' if pilot else ''}")
    print("=" * 72)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    by_fam = read_snodb_cd()
    fams_sorted = sorted(by_fam.items(), key=lambda kv: -len(kv[1]))
    if pilot:
        fams_sorted = fams_sorted[:PILOT_FAMILIES]
    keep = {f for f, _ in fams_sorted}
    print(f"Families to process: {len(keep)}  (snoRNAs: {sum(len(v) for _, v in fams_sorted)})")

    print(f"Parsing {RFAM_SEED} for SEED alignments...")
    seeds = parse_rfam_seed_aligned(RFAM_SEED, keep)
    print(f"  SEED alignments found: {len(seeds)}/{len(keep)}")

    rng = random.Random(RNG_SEED)
    manifest = {"families": {}, "n_snorna_aligned": 0, "n_snorna_total": 0, "pilot": pilot}
    for fam, humans in fams_sorted:
        manifest["n_snorna_total"] += len(humans)
        seed_rows = seeds.get(fam)
        if not seed_rows:
            manifest["families"][fam] = {"status": "no_seed", "n_human": len(humans)}
            continue
        # cap SEED context to MSA_DEPTH-1 rows (random, reproducible)
        ctx = seed_rows[:]
        rng.shuffle(ctx)
        ctx = ctx[: MSA_DEPTH - 1]
        seed_recs = [(f"SEED_{i}", s) for i, s in enumerate(ctx)]
        human_recs = [(sid, seq) for sid, gene, seq in humans]
        try:
            aligned = mafft_add(seed_recs, human_recs)
        except Exception as e:  # noqa: BLE001
            print(f"  {fam}: MAFFT error: {e}")
            manifest["families"][fam] = {"status": "mafft_error", "n_human": len(humans)}
            continue

        width = len(next(iter(aligned.values())))
        # write family MSA file: SEED rows + human query rows, '.' gaps, uppercase
        recs = []
        for i, _ in enumerate(seed_recs):
            recs.append((f"SEED_{i}", aligned[f"SEED_{i}"].upper().replace("-", ".")))
        n_q = 0
        id2gene = {sid: gene for sid, gene, _ in humans}
        for sid, gene, _ in humans:
            if sid in aligned:
                row = aligned[sid].upper().replace("-", ".")
                recs.append((f"{sid}|{id2gene[sid]}", row))
                n_q += 1
        write_fasta(OUT_DIR / f"{fam}.fasta", recs)
        manifest["families"][fam] = {"status": "ok", "width": width,
                                     "n_seed_ctx": len(seed_recs), "n_human_aligned": n_q,
                                     "n_human": len(humans)}
        manifest["n_snorna_aligned"] += n_q
        print(f"  {fam}: width={width}  seed_ctx={len(seed_recs)}  human_aligned={n_q}/{len(humans)}")

    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nAligned {manifest['n_snorna_aligned']} / {manifest['n_snorna_total']} snoRNAs "
          f"across {sum(1 for v in manifest['families'].values() if v.get('status')=='ok')} families.")
    print(f"Wrote per-family MSAs to {OUT_DIR}/ and {MANIFEST}.")
    if pilot:
        print("\nPILOT ok? Inspect a file (head outputs/snorna_msa/*.fasta) — SEED rows + human")
        print("rows should all be the same width. If good, run without --pilot, then step R2.")


if __name__ == "__main__":
    main()
