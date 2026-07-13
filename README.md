# RIBOSCOPE

**Mechanistic interpretability of RNA foundation models, applied to disease-variant interpretation in spliceosomal snRNAs.**

RIBOSCOPE trains sparse autoencoders (SAEs) on the hidden states of three RNA
foundation models, reads out per-nucleotide *functional criticality* by in-silico
mutagenesis, and shows that a **secondary-structure-aware** model localizes the
disease-causing nucleotides of **U4-type spliceosomal snRNA disorders** — validated
against a gold-standard saturation-genome-editing experiment, out-performing the
standard clinical variant tool, and used to nominate currently-uncertain variants
for reclassification review.

> Research code accompanying a manuscript in preparation (ISEF 2027, Computational
> Biology). Results are reported honestly with their bounds; see [Limitations](#limitations).

---

## The result in one minute

- **Method.** SAEs over RNA-FM (structure-naive), ErnieRNA (secondary-structure-aware)
  and RNA-MSM (evolutionary/MSA) yield interpretable biological features. A key
  finding: **SAE interpretability is conditional on in-distribution model input**
  (RNA-MSM gives 0 usable family specialists on single sequences, 376 on its native
  MSA — same weights, only the input changed).
- **Disease localization.** Reading per-nucleotide criticality by in-silico
  mutagenesis, **ErnieRNA localizes ClinVar-pathogenic nucleotides in U4-type snRNAs**:
  - *RNU4-2 / ReNU syndrome:* recapitulates the saturation-editing functional map
    (Spearman rho = -0.26, AUC 0.71 for experimentally-disruptive positions); triages
    pathogenic-vs-population-benign variants at **AUC 0.72, beating CADD (0.60)** with
    the structure-naive model at chance (0.55).
  - *RNU4ATAC / MOPD1:* independent replication (localization 0.68, triage 0.65).
  - Architecture-specific: structure-naive (RNA-FM) and MSA (RNA-MSM) models do not localize.
- **Predictions.** 5 currently-uncertain RNU4-2 variants are high-criticality **and**
  saturation-editing-confirmed damaging — nominated for reclassification review
  (`n.44A>T, n.78A>C, n.7G>C, n.27C>G, n.121T>A`).

---

## Repository layout

```
riboscope/
├── README.md              <- you are here
├── PIPELINE.md            <- script-by-script index, grouped by phase
├── requirements.txt       <- dependencies (PIN multimolecule -- see note)
├── LICENSE                <- MIT (code)
├── CITATION.cff           <- how to cite this repo
├── .gitignore
├── sae_models.py          <- the BatchTopK sparse autoencoder
├── entrez_config.py       <- reads NCBI_EMAIL env var (NCBI etiquette; see below)
├── 01..63_*.py            <- numbered pipeline scripts (run from repo root)
├── set_model.py           <- switches the SAE pipeline between models
├── data_README.md         <- provenance of every downloaded input
├── sequences/             <- fetched FASTA (small; some ship, rest regenerated)
├── figures/               <- the 5 manuscript figures (SVG)
├── outputs_mirror/        <- curated REFERENCE results (JSONs) — read-only mirror
└── outputs/               <- created by a fresh run; large files git-ignored
```

Fresh runs write to `outputs/`; the shipped JSON results live in `outputs_mirror/`
so you can compare a re-run against the values in the paper. Large activation and
SAE-weight files (`*.safetensors`, ~430 MB) are **not** in git — they are archived
on Zenodo (DOI in the manuscript's Data Availability) and most are regenerable.

Scripts are **flat at the repo root by design** — they import `sae_models` and
read/write `outputs/`, `sequences/`, `data/` by relative path, so run them from
the repository root (`cd riboscope && uv run python 53_sge_validation.py`).

---

## Quickstart (reproduce the headline disease result)

The disease-validation half (scripts 45–57) is light: it needs only forward passes
of two ~100M-param models plus public data fetched live. No SAE training required.

```bash
# 1. environment (uv recommended; see requirements.txt -- DO NOT upgrade multimolecule)
uv venv && uv pip install -r requirements.txt

# 1b. NCBI E-utilities require a contact email. Set yours (used only in request
#     headers, per NCBI etiquette; nothing is sent anywhere else):
export NCBI_EMAIL="you@example.com"          # bash / zsh
#   $env:NCBI_EMAIL = "you@example.com"       # PowerShell

# 2. fetch the disease structured ncRNAs + recognise which the models read
uv run python 45_fetch_disease_structured_rna.py
uv run python 46_recon_disease_features.py rnafm
uv run python 46_recon_disease_features.py erniarna

# 3. per-nucleotide criticality (in-silico mutagenesis), both models, all RNAs
uv run python 47_ism_functional_criticality.py rnafm    --all
uv run python 47_ism_functional_criticality.py erniarna --all

# 4. the gold-standard validation + clinical triage + novel predictions
uv run python 53_sge_validation.py            # vs saturation-editing map
uv run python 54_triage_pathogenic_benign.py  # pathogenic vs benign, vs CADD
uv run python 56_vus_prediction.py            # nominate uncertain variants
```

Re-building the SAEs and the full 3-model panel (scripts 06–18) requires GPU
training (~2 h per SAE) and ~5 GB activation files; see `PIPELINE.md`.

---

## Data & provenance

All inputs are **public** and fetched programmatically by the scripts (no
permission-gated data). See `data/README.md`. Sources:

- **Rfam** SEED alignments / family sequences (`Rfam.seed.gz`).
- **NCBI E-utilities** — RefSeq ncRNA sequences (efetch).
- **ClinVar** — pathogenic / benign / uncertain variant classifications (esearch/esummary).
- **snoDB 2.0** — human C/D-box snoRNAs (snoRNA phase).
- **RNU4-2 saturation genome editing** — function scores, medRxiv `2025.04.08.25325442`
  Supplementary Table 1 (auto-downloaded by `53_sge_validation.py`); ENA `PRJEB87505`.

Models (Hugging Face, via `multimolecule`): `multimolecule/rnafm`,
`multimolecule/ernierna`, `multimolecule/rnamsm`.

---

## Environment note (important)

A blind upgrade of `multimolecule` (0.0.9 -> 0.2.0) silently broke ErnieRNA's
structure-attention loading (renamed `pairwise_bias_proj` keys). **Pin
`multimolecule==0.0.9`** (or rely on the remap baked into the `load_model` of
scripts 23/46/47/53/54/55). Commit your exact `uv.lock` / `pip freeze` for full
reproducibility.

---

## Limitations (read before citing)

- The disease-localization effect is **specific to U4-type snRNAs** (RNU4-2,
  RNU4ATAC). It does **not** extend to U2/U12 or non-spliceosomal RNAs.
- The effect is **modest** (AUCs ~0.6–0.72): a variant-**prioritization** signal,
  not a diagnostic. VUS "nominations" are computational predictions with
  experimental support, **not** clinical reclassifications (ACMG review required).
- Criticality is **per-position** (per-variant resolution is future work).
- The "criticality concentrates on the U4/U6 duplex" mechanism was tested and
  **not supported** (the structure-naive model concentrates there too); the
  defensible mechanism is functional-map correlation, not structural-region hits.
- Negative results are reported as such (broad disease screen, connection-mining,
  RNA-MSM single-seq, mito-tRNAs).

---

## License & citation

Code: MIT (`LICENSE`). Manuscript in preparation — citation to be added on preprint.
If you use this code, please cite the preprint (forthcoming) and the upstream
models, Rfam, ClinVar, and the RNU4-2 saturation-editing study.
