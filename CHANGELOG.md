# Changelog

## 0.2.0 — revision (October 2026)

Changes relative to the code of the original submission (`try_live27.py`), made while revising the manuscript, several prompted by the reviewers.

**Fixes that change results**
- Labels are allele-aware: calls are matched to the truth set on chromosome, position, REF and ALT (previously chromosome and position only).
- Features are standardized for `GM`, `BGM` and `LogReg`, with the scaler fitted on training data only (inside each CV fold) and saved for apply mode.

**Fixes to the package**
- Restored `metrics.py`, which had been deleted and made the package fail on import.
- Fixed a `KeyError` in cross-validation (`LR`/`RF` vs `LogReg`/`RandForest` keys) introduced when the monolithic script was split into modules. The package now matches, function by function, the script that produced the published results.
- `optuna` is now a core dependency, because the default `LGB_Bayes` and `LGB_MultiObj` searches use it.
- The train template now lists the eight features used in the paper (it previously included `ClippingRankSum`).

**Additions**
- `scripts/reeval_external.py`: external-validation statistics used in the paper. It separates ClinVar pathogenic and benign records, filters by review status, runs exact DeLong tests with Holm correction over the distinct model pairs, and uses full-data paired (standard and Bayesian) bootstraps.
- `scripts/make_results.py`, `scripts/plot_scores.py`, `scripts/count_clinvar_pathogenic.py`.
- `results/`: the result files behind every table of the paper.
- Unit tests, an end-to-end smoke test and GitHub Actions CI. CI also checks that the paper's tables regenerate from `results/`.
- `CITATION.cff`, `docs/reproducing_the_paper.md`. The README was rewritten to match the repository.
- `reeval_external.py --chroms primary|all`. The main analysis of the revised paper is restricted to chr1–22 and chrX; `results/external_validation/` holds it and `results/external_validation_allcalls/` the analysis of all calls.
- Stacked hybrid (`scripts/stacked_hybrid.py`, `results/stacked_hybrid/`), suggested by Reviewer 3: the Gaussian-mixture log-likelihood ratio is a cross-fitted ninth LightGBM feature.
- Analyses of the hybrid weight α: `scripts/alpha_convergence.py` (benchmark), `scripts/alpha_curve_external.py` (NA06985), `scripts/hybrid_alpha_simulation.py` (illustrative simulation), with results in `results/alpha_*/`.
- `scripts/make_supp_extra.py` and `scripts/fig_alpha_three.py` (Supplementary Tables S3–S6 and Figure S2); `plot_scores.py` uses the primary chromosomes by default.

## 0.1.0 — original submission (August 2026)

Single script `try_live27.py`.
