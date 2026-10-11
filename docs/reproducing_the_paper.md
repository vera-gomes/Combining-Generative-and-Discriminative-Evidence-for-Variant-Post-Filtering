# Reproducing the paper

This guide goes from public data to every table and figure of *Combining Generative and Discriminative Evidence for Variant Post-Filtering*. Steps 1–3 need the raw data and several hours of compute. Step 4 needs only this repository, because `results/` already holds the outputs of steps 1–3.

## 0. Install

```bash
pip install -e ".[full]"
```

## Software versions used for the paper

Python 3.12 with scikit-learn 1.4.1, LightGBM 4.6.0, Optuna 4.8.0, pysam 0.23.3, NumPy 1.26.4, pandas 2.3.3 and SciPy 1.16.3. Other versions satisfying `pyproject.toml` should give the same results up to small numerical differences. To match exactly:

```bash
pip install scikit-learn==1.4.1.post1 lightgbm==4.6.0 optuna==4.8.0 pysam==0.23.3 numpy==1.26.4 pandas==2.3.3 scipy==1.16.3
```

## 1. Train on NA12878

Inputs:
- the GATK HaplotypeCaller 4.2.0.0 callset of NA12878 (Illumina TruSeq WGS, HiSeq 2000, ~50×, aligned to hg38 with Isaac 04.16.09.24);
- the GIAB HG001 v4.2.1 GRCh38 benchmark VCF (chromosomes 1–22).

```bash
cp configs/train.example.yaml configs/train.local.yaml
# set paths.train_input_vcf and paths.truth_vcf
postfilter-pipeline --config configs/train.local.yaml
```

The published run extracted 4,930,670 calls, of which 3,839,408 (77.9%) matched the benchmark. Its outputs used in the paper are in `results/train/`:

| File | Content |
|---|---|
| `cv_metrics_per_fold.csv` | AUC, precision, recall, F1 and accuracy per model and fold |
| `cv_evaluation_summary.csv` | means over the five folds (Table 2) |
| `lightgbm_tuning_summary.json` | selected hyperparameters of `LGB`, `LGB_Bayes`, `LGB_MultiObj` (Table S1) |
| `hybrid_tuning_summary.json` | nested-CV AUC for each α and base model (Table S2, Figure S1) |

## 2. Score NA06985

Input: the 1000 Genomes high-coverage NA06985 alignment (run ERR3239276, [IGSR](https://www.internationalgenome.org/data-portal/sample/NA06985)), called with GATK HaplotypeCaller 4.6.1.0 against `GRCh38_full_analysis_set_plus_decoy_hla` (5,069,016 calls).

```bash
cp configs/apply.example.yaml configs/apply.local.yaml
# set paths.apply_input_vcf and paths.trained_model_dir (the train output directory)
postfilter-pipeline --config configs/apply.local.yaml
```

The population-database sources in the template (gnomAD v3.1.2 and 1000 Genomes) mark each call as PopDB-supported or not. The result is `scored_variants_with_external_validation.csv`, with one score column per model and the PopDB label in `IsExternalVariant`. Fetching the remote databases is the slow part; local copies can be listed instead.

## 3. External-validation statistics

```bash
python scripts/reeval_external.py \
    --scores output/apply/scored_variants_with_external_validation.csv \
    --clinvar clinvar.vcf.gz \
    --out results/external_validation --n-boot 1000 --seed 2026
```

The paper used the ClinVar GRCh38 VCF of 4 October 2026. The script retrains nothing. It recomputes, for each evaluation regime:

| Prefix | Positives | Background |
|---|---|---|
| `popdb` | PopDB-supported calls (4,859,857) | other calls (209,159) |
| `clinvar_expert_B_vs_bg` | benign/likely benign records reviewed by an expert panel (363) | calls absent from ClinVar (5,044,742) |
| `clinvar_2star_B_vs_bg` | benign/likely benign records with ≥2 review stars (24,270) | calls absent from ClinVar |
| `clinvar_*_vs_popdbbg` | as above | PopDB-supported calls absent from ClinVar (sensitivity analysis) |
| `clinvar_legacy_expertPB_vs_bg` | every expert-panel record (the definition of the original submission) | calls absent from ClinVar |

For each regime it writes `_summary.csv` (AUC, AP, F1 and log-loss with bootstrap intervals and probability of being best), `_pairwise.csv` (exact DeLong tests, Holm-adjusted over the 36 pairs of the nine distinct models, with bootstrap intervals of the differences) and `_tranches.csv` (operating points at fixed sensitivity). `clinvar_records_in_callset.csv` lists every ClinVar record found in the callset, with its classification, review status and the score of each model. `run_info.json` records the counts, the ClinVar release and which hybrids were identical to their base model. The four pathogenic records can also be listed with `scripts/count_clinvar_pathogenic.py`.

## 4. Tables and figures

```bash
python scripts/make_results.py \
    --reeval results/external_validation --train results/train \
    --clinvar-records results/external_validation/clinvar_records_in_callset.csv.gz \
    --tables tables --figures figures
python scripts/plot_scores.py \
    --scores output/apply/scored_variants_with_external_validation.csv \
    --out figures/fig_score_distributions
```

`make_results.py` writes every table of the main text and the Supplementary Material as LaTeX (`tables/*.tex`) and the AUC heatmaps and α-selection figure (`figures/`). The continuous-integration workflow checks that these tables regenerate from `results/`. `plot_scores.py` needs the per-call scores file from step 2 and produces Figure 2.
