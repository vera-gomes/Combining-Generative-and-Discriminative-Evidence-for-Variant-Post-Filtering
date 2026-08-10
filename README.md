# Combining Generative and Discriminative Evidence for Variant Post-Filtering

Code and configuration accompanying:

> Pinto, V., Sousa, L., and Silva, C. *Combining Generative and Discriminative Evidence for Variant Post-Filtering.* Frontiers in Bioinformatics (submitted).

This repository contains the deployment-oriented post-filtering pipeline used to train benchmark post-filtering models and apply them, without retraining, to a real whole-genome sequencing (WGS) sample (NA06985), evaluated against ClinVar-supported and population-database-supported (PopDB) external validation regimes.

This is a deployment-oriented extension of an earlier caller-agnostic benchmark pipeline (Pinto et al., 2026a).

## Authors

- **Vera Pinto** — DCM/CEAUL, Faculdade de Ciências, Universidade de Lisboa — vgpinto@ciencias.ulisboa.pt
- **Lisete Sousa** — DCM/CEAUL, Faculdade de Ciências, Universidade de Lisboa
- **Carina Silva** — CEAUL, Faculdade de Ciências, Universidade de Lisboa / Escola Superior de Saúde de Lisboa, Instituto Politécnico de Lisboa

## Overview

The pipeline treats variant post-filtering as a statistical classification problem operating on standardized per-variant feature summaries (MQ, QD, FS, SOR, MQRankSum, ReadPosRankSum, BaseQRankSum, DP) extracted from VCF files. It compares two families of evidence:

- **Generative** models — Gaussian Mixture (GM), Bayesian Gaussian Mixture (BGM) — which characterize the feature-space structure without using benchmark labels.
- **Discriminative** models — Logistic Regression, Random Forest, LightGBM — which learn a direct label-supervised decision boundary, along with two LightGBM extensions:
  - **LGB_Bayes** — Optuna-based Bayesian hyperparameter search.
  - **LGB_MultiObj** — multi-objective Bayesian optimization trading discrimination against measured runtime.
- **Hybrid** scores blending generative and discriminative evidence:
  - **Hybrid_α** — data-driven blend weight selected by nested cross-validation.
  - **Hybrid_0.8** — fixed-weight blend (0.80 × LightGBM + 0.20 × GM), grounded in the cross-validation convergence/variance analysis (Figures 3–5 of the paper).

Benchmark-trained models are applied in **apply mode**, without retraining, to a GATK HaplotypeCaller callset from NA06985 and scored under two external, partially labeled validation regimes:

- **ClinVar-supported pathogenic enrichment** — 360 positives vs. 5,068,656 background variants.
- **PopDB-supported population enrichment** — 4,859,857 positives vs. 209,159 background variants.

Evaluation includes AUC, precision–recall summaries, sensitivity-tranche operating points, paired DeLong AUC tests, McNemar and (Bayesian) bootstrap comparisons, all with Holm's correction for multiple comparisons.

## Repository structure

```
.
├── README.md
├── pipeline/
│   └── try_live27.py            # main pipeline script (train + apply modes)
├── configs/
│   ├── pipeline_config_train.yaml
│   └── pipeline_config_apply.yaml
├── models/                      # trained model artifacts (produced by train mode)
├── output/                      # timestamped run outputs (produced by apply mode)
├── figures/                     # manuscript figures (Figures 1–7)
└── supplementary/                # Supplementary Material tables and PDF
```

## Requirements

- Python ≥ 3.10
- Core dependencies: `numpy`, `pandas`, `scikit-learn`, `lightgbm`, `optuna`, `pysam`, `scipy`, `matplotlib`
- `htslib`/`pysam` for VCF/BCF I/O, including remote-index (`.tbi`) fetching over HTTPS for remote population-database VCFs

Install with your environment manager of choice (`pip`/`conda`), then verify `pysam` can open a remote, indexed VCF before running apply mode against the gnomAD/1000 Genomes URLs listed in the config.

## Configuration

The pipeline is driven entirely by a YAML config (see `configs/`). Key sections:

- `mode`: `train` or `apply`
- `paths`: input VCF, trained-model directory, output directory
- `features`: the eight feature columns used by all models
- `evaluation`: target sensitivities, bootstrap subsample size, minimum class size for bootstrap, and `external_validation` block (ClinVar VCF URL, list of population-database VCF/BCF sources, per-chunk fetch parameters)
- `statistics`: which pairwise model comparisons to run (empty list = all-vs-all), bootstrap draws/seeds, probability threshold, log-loss mode
- `memory_safety`: memory ceiling, batch size, and LightGBM memory-safe mode for large-scale apply-mode scoring

## Pipeline logic (pseudocode)

The actual implementation is in `pipeline/try_live27.py`; the logic below is a structural summary rather than the literal code.

### Train mode

```
FUNCTION run_train_pipeline(config):
    load benchmark VCF(s), extract feature matrix X and truth labels y
    build genomic-block groups for leakage-aware, block-wise CV splitting
    impute missing feature values within each CV fold only (no leakage)

    FOR each of 5 CV folds:
        fit generative models:      GM, BGM                      (unsupervised)
        fit discriminative models:  LogReg, RF, LightGBM          (supervised)
        fit LightGBM extensions:
            LGB_Bayes    <- Optuna search over LightGBM hyperparameters,
                             maximizing ROC_AUC
            LGB_MultiObj <- Bayesian multi-objective search,
                             maximizing  λ * PR_AUC - (1-λ) * runtime
        select hybrid weight α via nested CV grid search over
            p_H(x; α) = α * p_LGB(x) + (1-α) * p_GM(x)
        record out-of-fold scores for every model + Hybrid_0.8 comparator

    aggregate per-fold metrics (AUC, PR_AUC, threshold metrics)
    run paired statistical tests across all model pairs:
        DeLong AUC test, McNemar, Bayesian McNemar, paired bootstrap
        apply Holm's correction across all pairwise comparisons

    refit each model family on the full benchmark data  -> final_models
    save final_models, imputer, and diagnostic plots to trained_model_dir
```

### Apply mode

```
FUNCTION run_apply_pipeline(config):
    load trained_models and imputer from trained_model_dir
    stream-parse the target WGS VCF (apply_input_vcf) in batches
    extract the same eight-feature matrix used in training
    apply the leakage-aware imputer (fit only on benchmark data)

    FOR each trained model (GM, BGM, LogReg, RF, LightGBM,
                            LGB_Bayes, LGB_MultiObj, Hybrid_α, Hybrid_0.8):
        score every variant  -> per-model score vector
    write per-variant scores to a scores CSV

    IF external_validation.clinvar.enabled:
        load ClinVar pathogenic positions -> positive label set
        label scored variants (positive vs. background)
        FOR each model:
            compute AUC, PR summaries, tranche precision/recall/F1
                at target sensitivities (0.90, 0.99, ...)
        run pairwise DeLong / McNemar / bootstrap comparisons (Holm-adjusted)
        compute probability-of-best via (Bayesian) bootstrap

    IF external_validation.population_dbs is non-empty:
        FOR each population-database source (gnomAD / 1000 Genomes, chunked
                                              by popdb_chunk_bp, with retry):
            fetch overlapping records and mark variants as PopDB-supported
        label scored variants (PopDB-supported vs. novel/background)
        repeat the same AUC / tranche / pairwise-comparison analysis
            as for ClinVar, using a much larger positive class

    generate apply-mode diagnostic plots:
        score CDFs and density histograms (Figures 1-2)
        pairwise ΔAUC forest plots (Figures 6-7)
    write all tables, figures, and a timestamped run summary to output_dir
```

### Hybrid score definitions

```
p_H(x; α)   = α * p_LGB(x) + (1 - α) * p_GM(x)      # Hybrid_α, α chosen by CV
p_H(x; 0.8) = 0.80 * p_LGB(x) + 0.20 * p_GM(x)       # Hybrid_0.8, fixed weight
```

`p_GM(x)` is the monotone-mapped Gaussian-mixture evidence rescaled to [0, 1]; `p_LGB(x)` is the LightGBM class probability.

## Running the pipeline

```
# Train benchmark models
python pipeline/try_live27.py --config configs/pipeline_config_train.yaml

# Apply trained models to a new WGS sample and run external validation
python pipeline/try_live27.py --config configs/pipeline_config_apply.yaml
```

Apply-mode runs against the full gnomAD v3.1.2 + 1000 Genomes PopDB source list are I/O- and compute-intensive (the PopDB bootstrap alone is on the order of hours at the configured subsample size); see `memory_safety` and `evaluation.apply_stats_max_n` in the apply config to tune resource usage.

## Data availability

- **NA06985** WGS sample: 1000 Genomes Project / International Genome Sample Resource (IGSR).
- **ClinVar**: NCBI ClinVar VCF (GRCh38).
- **Population databases**: gnomAD v3.1.2 genome sites VCFs; 1000 Genomes Project phase 3 biallelic SNV/INDEL VCFs (GRCh38).

Trained model artifacts and full output tables are available at: `[NAME OF REPOSITORY] [LINK]`.

## Citation

If you use this code, please cite:

```
Pinto, V., Sousa, L., and Silva, C. Combining Generative and Discriminative
Evidence for Variant Post-Filtering. Frontiers in Bioinformatics (submitted).
```

and the related benchmark pipeline:

```
Pinto, V., Sousa, L., and Silva, C. (2026a). A caller-agnostic variant
post-filtering pipeline. Manuscript under review.

Pinto, V., Sousa, L., and Silva, C. (2026b). Variant calling in genomics:
A comparative performance analysis and decision guide. PLOS ONE 21, 1-22.
doi:10.1371/journal.pone.0339891
```

## Funding

This work is funded by national funds through FCT – Fundação para a Ciência e a Tecnologia, I.P., under the CEAUL Research Unit (UID/00006/2025), the project UID/PRR/00006/2025, and the doctoral grant UI/BD/153743/2022.

