# RIBOSCOPE pipeline index

Every script has a module docstring with its exact inputs/outputs and run command.
Scripts run from the **repo root**. Below they are grouped by phase, in run order.
`★` = part of the locked result; `(explored)` = ran, informative, but not in the
headline (kept for transparency / honesty).

---

## 0. Setup & SAE core

| Script | Purpose |
|---|---|
| `sae_models.py` | BatchTopK sparse autoencoder (the SAE used throughout) |
| `01_check_gpu.py`, `02_smoke_test_rnafm.py` | environment / forward-pass smoke tests |
| `03–05_*` | early small-scale prototype (legacy) |
| `06_fetch_rfam_sequences.py` | build the family-balanced 30k Rfam FASTA |
| `07_extract_activations_big.py` | ★ RNA-FM layer-6 activations |
| `10_extract_erniarna.py` | ★ ErnieRNA activations (`10_extract_omnigenome.py` = abandoned pivot) |
| `11_extract_rnamsm.py` | ★ RNA-MSM activations, single-seq (`11_extract_rinalmo.py` = abandoned) |
| `normalize_rnamsm.py` | z-score RNA-MSM activations |
| `08_train_sae_big.py` | ★ train BatchTopK SAE (dict 8192, k 32) |
| `09_inspect_features_big.py` | ★ inspect features → specialists + motifs |
| `set_model.py` | repoints 08/09 between models/configs |
| `12_cross_model_agreement.py` | ★ cross-model specialist replication |

## 1. RNA-MSM diagnosis → the input-conditional finding

| Script | Purpose |
|---|---|
| `position_variance_report.py`, `compute_position_means.py`, `subtract_position_means.py` | diagnose + fix position-term dominance |
| `13_linear_probe_family.py`, `14_family_geometry_diagnostic.py`, `15_per_layer_rank_scan.py` | diagnostics (signal present, geometry collapsed) |
| `16_msa_geometry_scan.py` | native-MSA partially de-collapses geometry |
| `17_extract_rnamsm_msa.py` | ★ RNA-MSM activations under **native MSA** input |
| `18_prep_msa_acts.py`, `19_extract_rnamsm_single_subset.py` | ★ matched-control preprocessing |

→ Headline methods result: **interpretability is conditional on in-distribution input**
(single-seq: ~0 specialists; native MSA: 376).

## 2. snoRNA functional axis (explored — validation, rediscovers known biology)

`20–21` disease-lncRNA scan (null) · `22–28` snoDB C/D-box functional axis + grouped CV ·
`29` ErnieRNA load repair · `31–32` cross-model agreement / coverage ·
`33–36` RNA-MSM-MSA on snoRNAs (3-model panel) · `37` mechanism · `38–39` consensus hunt.

## 3. Cross-species + connection mining (explored — ruled out)

`40–42` *P. falciparum* ncRNA (in-distribution, rediscovers U3) ·
`43–44` cross-model "connections" + length control → confound-dominated, **closed**.

## 4. ★ Disease-variant interpretation (the locked result — scripts 45–57)

| Script | Purpose |
|---|---|
| `45_fetch_disease_structured_rna.py` | fetch disease structured ncRNAs (NCBI) + hotspot annotations |
| `46_recon_disease_features.py` | recognition gate: which RNAs the SAEs read, via which feature |
| `47_ism_functional_criticality.py` | per-nt criticality by in-silico mutagenesis (feature + representation maps) |
| `48_consensus_and_validation.py` | cross-model consensus + hotspot enrichment |
| `49_pathogenic_enrichment.py` | criticality vs full ClinVar pathogenic set (AUC + MWU) |
| `50_ism_rnamsm_msa.py` | RNA-MSM MSA in-silico mutagenesis (3rd model; readout confound documented) |
| `51_benchmark_localization.py` | nuclear-ncRNA benchmark (does it generalize?) |
| `52_benchmark_mito_trna.py` | mito-tRNA expansion (coordinate self-check) — narrows scope |
| `53_sge_validation.py` | ★ vs RNU4-2 saturation-genome-editing map (gold standard) + CADD baseline |
| `54_triage_pathogenic_benign.py` | ★ clinical VUS triage: pathogenic vs population-benign, vs CADD/SGE |
| `55_spliceosome_panel.py` | spliceosomal-snRNA panel (auto-resolves accessions) — U4-type specificity |
| `56_vus_prediction.py` | ★ nominate currently-uncertain RNU4-2 variants, SGE-backed |
| `57_mechanism_u4u6_duplex.py` | U4/U6-duplex mechanism test (honest negative — duplex claim dropped) |

**Minimal path to the headline figures:** 45 → 46 → 47 (`--all`) → 53 → 54 → 56.

## 5. Masked-PLL readout + second-gene scan (scripts 61–63) and figure/paper build

| Script | Purpose |
|---|---|
| `61_rnamsm_pll_localization.py` | ★ alignment-aware **masked pseudo-likelihood** readout — localizes the disease positions (readout-driven, complements the criticality result) |
| `63_singleseq_pll_control.py` | ★ single-sequence PLL control — masked-PLL localizes across architectures, so the signal is readout-driven, not architecture-specific |
| `62_candidate_gene_scan.py` | scan for a **second** positive disease-gene class (null so far — reported honestly) |
| `58_make_figures.py`, `59_render_svg_figures.py` | build Figures 2–3 as paper-idiom SVG (Figs 1/4/5 are hand-authored in `figures/`) |
| `60_build_paper.py` | assemble manuscript + 5 figures into a self-contained `RIBOSCOPE_paper.html` |
