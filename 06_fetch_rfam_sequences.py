"""
06_fetch_rfam_sequences.py — RIBOSCOPE Phase 11/12: Fetch RNA sequences from Rfam.

Phase 12 update: bumped per-family cap from 10 → 25 and target total from
10000 → 30000. Output filename changed to rfam_30k.fasta so it doesn't
overwrite Phase 11's rfam_10k.fasta (kept as the v1 baseline).

Run with
--------
    cd ~/projects/riboscope
    uv run python 06_fetch_rfam_sequences.py
"""

import gzip
import random
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

# ============================ CONFIG ============================
RFAM_URL = "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.seed.gz"
LOCAL_RFAM = Path("data/Rfam.seed.gz")
OUTPUT_FASTA = Path("sequences/rfam_30k.fasta")  # Phase 12: was rfam_10k.fasta

MIN_LEN = 50
MAX_LEN = 510

N_PER_FAMILY = 25      # Phase 12: was 10
TARGET_TOTAL = 30000   # Phase 12: was 10000
MAX_AMBIGUITY_FRAC = 0.05
RNG_SEED = 42
# ================================================================


def download_rfam() -> None:
    if LOCAL_RFAM.exists():
        size_mb = LOCAL_RFAM.stat().st_size / 1e6
        print(f"      Using cached {LOCAL_RFAM} ({size_mb:.1f} MB)")
        return

    LOCAL_RFAM.parent.mkdir(parents=True, exist_ok=True)
    print(f"      Downloading from {RFAM_URL}...")
    print(f"      (~50 MB; this may take 1-2 minutes on a typical connection)")
    try:
        urllib.request.urlretrieve(RFAM_URL, LOCAL_RFAM)
    except urllib.error.URLError as e:
        print(f"❌ Download failed: {e}")
        print(f"   Try manually: curl -o {LOCAL_RFAM} {RFAM_URL}")
        sys.exit(1)
    size_mb = LOCAL_RFAM.stat().st_size / 1e6
    print(f"      Saved to {LOCAL_RFAM} ({size_mb:.1f} MB)")


def parse_stockholm_block(text: str):
    seq_buffers: dict[str, str] = {}
    family_id: str | None = None
    family_name: str | None = None
    for line in text.split("\n"):
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("#=GF AC"):
            family_id = line.split(maxsplit=2)[2].strip()
        elif line.startswith("#=GF ID"):
            family_name = line.split(maxsplit=2)[2].strip()
        elif line.startswith("#") or line.startswith("//"):
            continue
        else:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                seq_id, aligned_seq = parts
                seq_buffers[seq_id] = seq_buffers.get(seq_id, "") + aligned_seq

    cleaned = []
    for seq_id, aligned in seq_buffers.items():
        ungapped = aligned.replace("-", "").replace(".", "").upper()
        ungapped = ungapped.replace("T", "U")
        cleaned.append((seq_id, ungapped))
    return family_id, family_name, cleaned


def iter_rfam_seed(path: Path):
    with gzip.open(path, "rt", encoding="latin-1") as f:
        block: list[str] = []
        for line in f:
            block.append(line)
            if line.startswith("//"):
                fam_id, fam_name, seqs = parse_stockholm_block("\n".join(block))
                if fam_id and seqs:
                    yield fam_id, fam_name, seqs
                block = []


def is_valid_sequence(seq: str) -> bool:
    if not (MIN_LEN <= len(seq) <= MAX_LEN):
        return False
    n_acgu = sum(1 for c in seq if c in "ACGU")
    if n_acgu / len(seq) < 1 - MAX_AMBIGUITY_FRAC:
        return False
    return True


def normalize_sequence(seq: str) -> str:
    return "".join(c if c in "ACGU" else "N" for c in seq)


def main() -> None:
    print("=" * 70)
    print(f"RIBOSCOPE Phase 12: Fetch ~{TARGET_TOTAL} RNA sequences from Rfam")
    print("=" * 70)

    print("[1/4] Acquiring Rfam.seed.gz...")
    download_rfam()

    print("[2/4] Parsing Rfam seed alignments...")
    family_buckets: list[tuple[str, str, list[tuple[str, str]]]] = []
    n_total_raw = 0
    for fam_id, fam_name, sequences in iter_rfam_seed(LOCAL_RFAM):
        n_total_raw += len(sequences)
        valid = [(sid, normalize_sequence(s)) for sid, s in sequences if is_valid_sequence(s)]
        if valid:
            family_buckets.append((fam_id, fam_name, valid))

    print(f"      Families parsed:                       {len(family_buckets)}")
    print(f"      Raw sequences (all families):          {n_total_raw}")
    n_after_filter = sum(len(s) for _, _, s in family_buckets)
    print(f"      Sequences after length+ambiguity:      {n_after_filter}")

    print(f"[3/4] Sampling up to {N_PER_FAMILY} sequences per family...")
    rng = random.Random(RNG_SEED)
    candidates: list[tuple[str, str, str, str]] = []
    for fam_id, fam_name, sequences in family_buckets:
        rng.shuffle(sequences)
        for i, (seq_id, seq) in enumerate(sequences[:N_PER_FAMILY]):
            display_name = f"{fam_id}_{i:02d}"
            candidates.append((display_name, fam_id, fam_name, seq))
    print(f"      Candidates after per-family cap:       {len(candidates)}")

    seen: set[str] = set()
    deduped: list[tuple[str, str, str, str]] = []
    for cand in candidates:
        seq = cand[3]
        if seq in seen:
            continue
        seen.add(seq)
        deduped.append(cand)
    print(f"      After exact-sequence dedup:            {len(deduped)}")

    if len(deduped) > TARGET_TOTAL:
        rng.shuffle(deduped)
        deduped = deduped[:TARGET_TOTAL]
    print(f"      After capping at {TARGET_TOTAL}:           {len(deduped)}")

    print(f"[4/4] Writing FASTA to {OUTPUT_FASTA}...")
    OUTPUT_FASTA.parent.mkdir(parents=True, exist_ok=True)
    family_counts: dict[str, int] = defaultdict(int)
    length_total = 0
    with open(OUTPUT_FASTA, "w") as f:
        for display_name, fam_id, fam_name, seq in deduped:
            f.write(f">{display_name} | {fam_id} | {fam_name} | {len(seq)} nt\n")
            f.write(seq + "\n")
            family_counts[fam_id] += 1
            length_total += len(seq)

    avg_len = length_total / max(len(deduped), 1)
    n_unique_families = len(family_counts)

    print()
    print("=" * 70)
    print("✅ Sequence fetch complete!")
    print("=" * 70)
    print(f"   Output file:           {OUTPUT_FASTA}")
    print(f"   Total sequences:       {len(deduped)}")
    print(f"   Distinct families:     {n_unique_families}")
    print(f"   Total tokens (~):      {length_total + 2 * len(deduped):,}")
    print(f"   Average length:        {avg_len:.1f} nt")
    print(f"   File size:             {OUTPUT_FASTA.stat().st_size / 1e6:.2f} MB")
    print()
    print(f"   Top 10 most-represented families:")
    for fam_id, count in sorted(family_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"     {fam_id}: {count} sequences")


if __name__ == "__main__":
    main()
