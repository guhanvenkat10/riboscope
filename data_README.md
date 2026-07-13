# Data provenance

All inputs are **public** and fetched programmatically — nothing here is
permission-gated. The `data/` folder is git-ignored (files are large and
regenerable); this note (shipped at the repo root) records where each input
comes from so the pipeline is reproducible.

| File (in `data/`) | Source | Fetched by |
|---|---|---|
| `Rfam.seed.gz` | Rfam (SEED alignments, all families) | `06_fetch_rfam_sequences.py` |
| `snodb_all.tsv` | snoDB 2.0 (human snoRNAs) | `22_fetch_snodb.py` |
| `sge_rnu4-2.xlsx` | RNU4-2 saturation genome editing, **Supplementary Table 1**, medRxiv `2025.04.08.25325442` (auto-downloaded from the DC1 media link) | `53_sge_validation.py` |

Fetched live (not stored in `data/`):

- **RefSeq ncRNA sequences** — NCBI E-utilities `efetch` (e.g. RNU4-2 `NR_003137`,
  RNU4ATAC `NR_023343`, …). Scripts `45`, `51`, `52`, `55`.
- **ClinVar** classifications (pathogenic / benign / uncertain) — NCBI `esearch` +
  `esummary`. Scripts `49`, `51`, `52`, `53`, `54`, `55`, `56`.
- **Mitochondrial tRNA** coordinates — region `efetch` on `NC_012920.1` with a
  reference-base self-check. Script `52`.

Raw saturation-editing FASTQs (not needed here): ENA accession `PRJEB87505`.

Models are downloaded from Hugging Face on first use (`multimolecule/rnafm`,
`multimolecule/ernierna`, `multimolecule/rnamsm`) and cached outside the repo.
