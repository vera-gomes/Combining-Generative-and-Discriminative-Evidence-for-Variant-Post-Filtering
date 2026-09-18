# Agent-facing notes

This file is a structural summary of the pipeline for coding agents (and
humans skimming the codebase) working on this repository. It is not
user-facing documentation — see `README.md` for that.

## Package layout

The implementation lives in `src/postfilter_pipeline/`, split by concern:

| Module | Contents |
|---|---|
| `config.py` | `PipelineConfig` (YAML-backed config object) and pairwise-comparison resolution helpers |
| `models.py` | Hybrid-score constants (`HYBRID_BASE_MODELS`, `FIXED_ALPHA_HYBRIDS`), `LeakageAwareImputer`, VQSLOD/hybrid scoring functions, `score_model_instance` |
| `variant_io.py` | VCF/BCF I/O, remote-index handling, `PopulationDatabaseManager`, ClinVar loading |
| `metrics.py` | Threshold selection and basic classification metrics |
| `stats.py` | Multiple-testing correction (Holm, BH), DeLong test, McNemar/Bayesian McNemar, paired/Bayesian bootstrap, bootstrap CI metrics |
| `memory_utils.py` | Memory monitoring / memory-safety skip logic |
| `data_prep.py` | VCF annotation extraction, truth labeling, genomic block grouping/splitting |
| `training.py` | LightGBM hyperparameter search (Optuna/skopt), hybrid-α nested CV selection, cross-validated benchmark training, final-model fitting/saving |
| `external_validation.py` | ClinVar- and population-database-supported apply-mode evaluation, comprehensive external validation, apply-mode scoring |
| `plotting.py` | All `plot_*` diagnostic figures used by train/apply modes |
| `pipeline.py` | `run_train_pipeline`, `run_apply_pipeline` — the two top-level orchestration entry points |
| `cli.py` | `main()` — argparse CLI, wired to the `postfilter-pipeline` console script |

Import graph (no cycles): `models` → `config` → `variant_io` / `memory_utils`;
`metrics` → `stats`; `data_prep` (standalone) → `training` → `external_validation`
→ `plotting` → `pipeline` → `cli`.

## Pipeline logic (pseudocode)

The actual implementation is in `src/postfilter_pipeline/pipeline.py` (plus
the modules it calls into, above); the logic below is a structural summary
rather than the literal code.

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

`p_GM(x)` is the monotone-mapped Gaussian-mixture evidence rescaled to [0, 1];
`p_LGB(x)` is the LightGBM class probability.

## Known naming inconsistency

`models.FIXED_ALPHA_HYBRIDS` still uses the internal names
`Hybrid_LGB_alpha1` / `Hybrid_LGB_alpha08` / `Hybrid_LGB_alpha07`, which don't
line up 1:1 with the manuscript's `Hybrid_α` / `Hybrid_0.8` naming. This is a
pre-existing naming choice in the pipeline code, not something introduced by
the package split — flagged here for whoever next touches hybrid-model naming,
not acted on automatically since it would change output file/column names.

## Working conventions for agents

- Preserve function bodies verbatim when moving code between modules; this is
  scientific-analysis code backing a submitted manuscript, so mechanical
  refactors should not change numerical behavior.
- After any change touching more than one module, sanity-check the import
  graph above still holds (no new cycles) and that
  `python3 -m py_compile src/postfilter_pipeline/*.py` passes.
- Config templates (`configs/*.example.yaml`) are tracked; `configs/*.local.yaml`
  is gitignored and is where real, machine-specific paths belong.
