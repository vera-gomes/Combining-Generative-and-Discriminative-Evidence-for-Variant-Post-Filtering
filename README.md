# Combining Generative and Discriminative Evidence for Variant Post-Filtering

[![tests](https://github.com/vera-gomes/Combining-Generative-and-Discriminative-Evidence-for-Variant-Post-Filtering/actions/workflows/tests.yml/badge.svg)](https://github.com/vera-gomes/Combining-Generative-and-Discriminative-Evidence-for-Variant-Post-Filtering/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Code, configuration and result files for:

> Pinto, V., Sousa, L. and Silva, C. *Combining Generative and Discriminative Evidence for Variant Post-Filtering.* Frontiers in Bioinformatics (under revision).

Germline variant calling produces millions of candidate calls per genome, some of which are artefacts. A **post-filter** scores each call from its annotations so that unreliable calls can be removed. This repository contains a post-filtering pipeline that

1. **trains** generative, discriminative and hybrid post-filters on a benchmark sample with a known truth set (NA12878, GIAB v4.2.1), and
2. **applies** them, without retraining, to a real whole-genome sample (NA06985) and evaluates them against partial external labels from population databases and ClinVar.

It extends the benchmark pipeline of Pinto et al. (under review), which compared post-filters across four variant callers.

## Contents

- [Models](#models)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Reproducing the paper](#reproducing-the-paper)
- [Repository structure](#repository-structure)
- [Data sources](#data-sources)
- [Known limitations](#known-limitations)
- [Citation](#citation)

## Models

Every call is described by eight GATK HaplotypeCaller INFO annotations: `QD`, `MQ`, `FS`, `SOR`, `MQRankSum`, `ReadPosRankSum`, `BaseQRankSum` and `DP`. The pipeline reads any VCF and a list of annotation names, so other callers can be used if they emit comparable annotations.

| Name in the paper | Name in the code | Description |
|---|---|---|
| `GM`, `BGM` | `GM`, `BGM` | Class-conditional Gaussian / Bayesian Gaussian mixtures (5 components), fitted separately to true and false calls; score = sigmoid of the log-likelihood ratio (a VQSLOD-like score) |
| `LogReg` | `LogReg` | Logistic regression |
| `RF` | `RandForest` | Random forest (50 trees, depth 10) |
| `LGB` | `LGB` | LightGBM with default settings |
| `LGB_Bayes` | `LGB_Bayes` | LightGBM tuned by Bayesian optimization (Optuna TPE, 25 trials, ROC AUC) |
| `LGB_MultiObj` | `LGB_MultiObj` | LightGBM tuned with the multi-objective criterion J = λ·PR_AUC − (1−λ)·min(t/T, 1), λ = 0.85, T = 5 s |
| `Hybrid_α` | `Hybrid_LGB`, `Hybrid_LGB_Bayes`, `Hybrid_LGB_MultiObj`, `Hybrid_LGB_alpha1` | α·p_LGB + (1−α)·p_GM with α chosen by nested 3-fold CV (selected α = 1, so these equal their LightGBM base) |
| `Hybrid_0.8`, `Hybrid_0.7` | `Hybrid_LGB_alpha08`, `Hybrid_LGB_alpha07` | Fixed-weight blends with 20% and 30% mixture evidence |

Implementation details that matter for correctness:

- **Labels are allele-aware.** A call is true only if chromosome, position, REF and ALT all match the truth set.
- **No leakage.** Missing values are imputed with the training mean, and `GM`, `BGM` and `LogReg` use standardized features. Imputation and scaling are fitted inside each CV fold, and the fitted objects are saved and reused in apply mode. Tree models use unscaled features.
- **Grouped CV.** 5-fold cross-validation uses 1 Mb genomic blocks (`StratifiedGroupKFold`), so neighbouring correlated calls never fall in both training and validation folds.

## Installation

Python ≥ 3.10. `pysam` needs a POSIX system (Linux or macOS).

```bash
git clone https://github.com/vera-gomes/Combining-Generative-and-Discriminative-Evidence-for-Variant-Post-Filtering.git
cd Combining-Generative-and-Discriminative-Evidence-for-Variant-Post-Filtering
python -m venv .venv && source .venv/bin/activate
pip install -e ".[full]"        # pipeline + optional extras (scikit-optimize fallback, Excel export, memory monitor)
pip install -e ".[dev]"         # adds pytest, to run the tests
```

or directly from GitHub:

```bash
pip install "postfilter-pipeline[full] @ git+https://github.com/vera-gomes/Combining-Generative-and-Discriminative-Evidence-for-Variant-Post-Filtering.git"
```

Check the installation:

```bash
postfilter-pipeline --help
pytest -m "not slow"            # unit tests (seconds)
pytest -m slow                  # train + apply on a small synthetic dataset (~3 min)
```

## Quick start

The pipeline is driven by a YAML file. Copy a template, edit the paths, and run:

```bash
cp configs/train.example.yaml configs/train.local.yaml   # *.local.yaml is git-ignored
cp configs/apply.example.yaml configs/apply.local.yaml

postfilter-pipeline --config configs/train.local.yaml    # fit and cross-validate all models
postfilter-pipeline --config configs/apply.local.yaml    # score a new sample with the saved models
```

Command-line flags override paths in the config:

```bash
postfilter-pipeline --config configs/apply.local.yaml \
    --apply-vcf /path/to/sample.vcf.gz \
    --trained-model-dir output/train_YYYYMMDD_HHMMSS \
    --output-dir output/apply_sample2
```

**Train mode** writes the fitted models, imputer and scaler (`models/`), per-fold and summary CV metrics, out-of-fold scores, LightGBM tuning and hybrid-α summaries, and diagnostic plots.
**Apply mode** writes one score per call and model (`real_data_scores.csv`; with population databases configured, also `scored_variants_with_external_validation.csv`, which adds the PopDB label) and diagnostic plots.

Main config sections:

| Section | Purpose |
|---|---|
| `paths` | input VCF(s), truth VCF, trained-model directory, output directory |
| `features` | INFO annotations used as features |
| `models` | mixture, LightGBM, Optuna search space, multi-objective (`lambda`, `runtime_budget_sec`) and hybrid (`n_alphas`, `cv_splits`) settings |
| `cv` | number of folds, block CV and block size |
| `evaluation.external_validation` | ClinVar VCF and population-database sources (apply mode) |
| `memory_safety` | memory ceiling and batch size for large callsets |

## Reproducing the paper

The full workflow is described in [`docs/reproducing_the_paper.md`](docs/reproducing_the_paper.md). In short:

| Step | Command | Output |
|---|---|---|
| 1. Train on NA12878 | `postfilter-pipeline --config configs/train.local.yaml` | models, CV results (`results/train/`) |
| 2. Score NA06985 | `postfilter-pipeline --config configs/apply.local.yaml` | `scored_variants_with_external_validation.csv` (5,069,016 calls) |
| 3. External validation | `python scripts/reeval_external.py --scores <scores.csv> --clinvar clinvar.vcf.gz --out results/external_validation --n-boot 1000 --chroms primary` | summaries, pairwise tests and operating points for each evaluation regime (chr1–22 and chrX, 4,963,081 calls) |
| 4. Tables and figures | `python scripts/make_results.py --reeval results/external_validation --train results/train --clinvar-records results/external_validation/clinvar_records_in_callset.csv.gz` | main-text tables, Supplementary Tables S1, S2 and S7–S16, heatmaps |
| 5. Score distributions | `python scripts/plot_scores.py --scores <scores.csv> --out fig_score_distributions` | Figure 2 |
| 6. Stacked hybrid and α analyses | `scripts/stacked_hybrid.py`, `scripts/alpha_convergence.py`, `scripts/alpha_curve_external.py`, `scripts/hybrid_alpha_simulation.py` | `results/stacked_hybrid/`, `results/alpha_*/` |
| 7. Remaining supplement | `python scripts/make_supp_extra.py` and `python scripts/fig_alpha_three.py` | Supplementary Tables S3–S6 and Figure S2 |

The `results/` folder holds the result files of the published run, so steps 4 and 7 regenerate every table without rerunning the other steps. The external-validation statistics in the paper come from `scripts/reeval_external.py` (exact DeLong tests with Holm correction, and full-data paired bootstraps with 1,000 replicates), not from the summaries that apply mode writes itself; see [Known limitations](#known-limitations).

## Repository structure

```
├── src/postfilter_pipeline/   installable package
│   ├── cli.py                 command-line entry point (postfilter-pipeline)
│   ├── config.py              YAML configuration and defaults
│   ├── data_prep.py           annotation extraction, allele-aware labelling, genomic blocks
│   ├── models.py              imputer, scaler, mixture (VQSLOD) and hybrid scoring
│   ├── training.py            cross-validation, LightGBM tuning, hybrid-α selection, final fit
│   ├── metrics.py             thresholds and classification metrics
│   ├── stats.py               DeLong, McNemar, bootstrap, Holm and Benjamini–Hochberg
│   ├── variant_io.py          VCF I/O, population-database and ClinVar access
│   ├── external_validation.py apply-mode scoring and built-in external evaluation
│   ├── plotting.py            diagnostic figures
│   ├── memory_utils.py        memory monitoring
│   └── pipeline.py            train and apply workflows
├── scripts/                   analysis scripts used for the paper
│   ├── reeval_external.py     external-validation statistics (all regimes; --chroms primary|all)
│   ├── make_results.py        main-text tables, most supplementary tables, heatmaps
│   ├── make_supp_extra.py     Supplementary Tables S3–S6 (stacked hybrid, all-calls analysis)
│   ├── plot_scores.py         score-distribution figure (Figure 2)
│   ├── stacked_hybrid.py      stacked hybrid: GM log-likelihood ratio as a LightGBM feature
│   ├── alpha_convergence.py   hybrid weight α: benchmark block CV over seeds
│   ├── alpha_curve_external.py  hybrid weight α: NA06985 after transfer, per chromosome
│   ├── hybrid_alpha_simulation.py  hybrid weight α: illustrative simulation
│   ├── fig_alpha_three.py     Supplementary Figure S2
│   └── count_clinvar_pathogenic.py  lists ClinVar P/LP records present in a callset
├── results/                   result files of the published run
│   ├── train/                 CV metrics, LightGBM tuning, hybrid-α selection
│   ├── external_validation/   main analysis (chr1–22 and chrX): per-regime summaries, pairwise tests, operating points, ClinVar records
│   ├── external_validation_allcalls/  the same on all 5,069,016 calls (sensitivity analysis)
│   ├── stacked_hybrid/        stacked hybrid: benchmark CV, feature importance, NA06985 re-evaluation
│   └── alpha_simulation/, alpha_convergence/, alpha_external/  analyses of the hybrid weight α
├── configs/                   train and apply templates
├── docs/                      reproduction guide, population-database URLs
├── tests/                     unit tests and an end-to-end smoke test
├── AGENTS.md                  module map and conventions for contributors
├── CHANGELOG.md
├── CITATION.cff
└── pyproject.toml
```

## Data sources

All data are public; none are stored in this repository.

| Data | Source |
|---|---|
| NA12878 reads (training) | Illumina TruSeq WGS, HiSeq 2000, ~50×, Illumina BaseSpace; aligned to hg38 with Isaac 04.16.09.24; called with GATK HaplotypeCaller 4.2.0.0 |
| Truth set | GIAB HG001 v4.2.1 GRCh38 benchmark, chromosomes 1–22 |
| NA06985 reads (application) | 1000 Genomes high-coverage (30×) set, New York Genome Center, run ERR3239276, via [IGSR](https://www.internationalgenome.org/data-portal/sample/NA06985); called with GATK HaplotypeCaller 4.6.1.0 against `GRCh38_full_analysis_set_plus_decoy_hla` |
| Population databases | gnomAD v3.1.2 genome sites and the 1000 Genomes 20190312 GRCh38 biallelic SNV/indel release; URLs in [`docs/population_db_sources.md`](docs/population_db_sources.md) |
| ClinVar | NCBI ClinVar GRCh38 VCF, release of 4 October 2026 (`fileDate` 20261004) |

## Known limitations

- **Built-in apply-mode statistics.** For compatibility with the original submission, `external_validation.py` keeps its original behaviour. It counts every expert-reviewed ClinVar record as a positive, whatever its classification, and its bootstrap subsamples the data (`apply_stats_max_n`). Use `scripts/reeval_external.py` for analyses: it separates pathogenic from benign records, filters by review status, and bootstraps over all calls.
- **Single caller and features.** The models were trained and applied on GATK HaplotypeCaller annotations only. Other callers need comparable annotations.
- **Benchmark regions.** Training calls are not restricted to the GIAB high-confidence regions, so real calls outside them are labelled false.
- **Multi-allelic records.** Only the first alternative allele of a multi-allelic record is scored.

## Citation

If you use this code, please cite the paper (see [`CITATION.cff`](CITATION.cff)) and, for the benchmark pipeline and caller comparison:

- Pinto, V., Sousa, L. and Silva, C. A caller-agnostic variant post-filtering pipeline. Manuscript under review.
- Pinto, V., Sousa, L. and Silva, C. (2026). Variant calling in genomics: a comparative performance analysis and decision guide. *PLOS ONE* 21, 1–22. doi:[10.1371/journal.pone.0339891](https://doi.org/10.1371/journal.pone.0339891)

## Funding

FCT – Fundação para a Ciência e a Tecnologia, I.P.: doctoral grant UI/BD/153743/2022, CEAUL Research Unit UID/00006/2025 and project UID/PRR/00006/2025 (European Union – NextGenerationEU).

## License

MIT — see [`LICENSE`](LICENSE).
