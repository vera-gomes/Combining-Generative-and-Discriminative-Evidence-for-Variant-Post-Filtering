"""Top-level train/apply pipeline orchestration."""

import json
import logging
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import PipelineConfig, _resolve_pairwise_comparisons
from .data_prep import (
    build_genomic_groups, extract_annotations, extract_truth_positions,
    label_variants, label_variants_with_external_set,
)
from .external_validation import (
    clinvar_based_model_comparison, population_db_enrichment_comparison,
    run_comprehensive_external_validation, score_real_data_fixed_complete,
)
from .memory_utils import monitor_memory
from .metrics import precision_at_sensitivity
from .plotting import (
    generate_all_train_plots, plot_apply_score_boxplots,
    plot_apply_score_distributions,
)
from .stats import delong_roc_test, holm_bonferroni, paired_bootstrap_delta
from .training import (
    cross_validation_evaluation_fixed, generate_tranche_analysis,
    run_paired_tests, run_variant_classification_counts,
    train_final_models_fixed,
)
from .variant_io import PopulationDatabaseManager


def run_train_pipeline(config: PipelineConfig, output_dir: str):
    start_time = time.time()
    # output_dir is created and logging is configured by main() before this call.

    logging.info("=" * 80)
    logging.info("STARTING TRAINING PIPELINE")
    logging.info("=" * 80)

    if config._raw_get('memory_safety.monitor_memory', True):
        monitor_memory("Pipeline start")

    batch_size = config._raw_get('memory_safety.batch_size', 100000)
    logging.info("Step 1: Extracting annotations...")
    df_annotations = extract_annotations(
        config._raw_get('paths.train_input_vcf'),
        config._raw_get('features', []),
        batch_size=batch_size,
    )
    if config._raw_get('memory_safety.monitor_memory', True):
        monitor_memory("After annotation extraction")

    feature_keys = [k for k in config._raw_get('features', [])
                    if k in df_annotations.columns and df_annotations[k].notna().any()]
    logging.info(f"Using features: {feature_keys}")

    logging.info("Step 2: Labelling variants...")
    truth_positions = extract_truth_positions(config._raw_get('paths.truth_vcf'))
    df_labeled = label_variants(df_annotations, truth_positions)
    df_labeled.to_csv(os.path.join(output_dir, "annotations_labeled.csv"), index=False)

    X = df_labeled[feature_keys].to_numpy(dtype=np.float32)
    y = df_labeled["TruthLabel"].to_numpy(dtype=int)

    if config._raw_get('memory_safety.monitor_memory', True):
        monitor_memory("Before cross-validation")

    logging.info("Step 3: Cross-validation evaluation...")
    groups = None
    if config._raw_get('cv.use_block_cv', True):
        groups = build_genomic_groups(df_labeled, config._raw_get('cv.block_size_bp', 1_000_000))

    avg_metrics, all_metrics, oof_scores = cross_validation_evaluation_fixed(
        X, y, groups=groups,
        n_splits=config._raw_get('cv.n_splits', 5),
        random_state=config._raw_get('models.random_state', 42),
        config=config,
    )

    pd.DataFrame(avg_metrics).to_csv(os.path.join(output_dir, "cv_evaluation_summary.csv"), index=False)

    per_fold_records = []
    for mname, mlist in all_metrics.items():
        for fi, m in enumerate(mlist, start=1):
            per_fold_records.append({"Model": mname, "Fold": fi,
                                     "AUC": m.get("auc", np.nan), "Precision": m.get("precision", np.nan),
                                     "Recall": m.get("recall", np.nan), "F1": m.get("f1", np.nan),
                                     "Accuracy": m.get("accuracy", np.nan)})
    per_fold_df = pd.DataFrame(per_fold_records)
    per_fold_df.to_csv(os.path.join(output_dir, "cv_metrics_per_fold.csv"), index=False)
    run_paired_tests(per_fold_df, os.path.join(output_dir, "cv_paired_tests.txt"))

    logging.info("Step 4: Training final models...")
    final_models, imputer, scaler = train_final_models_fixed(X, y, feature_keys, config, output_dir)
    X_imputed = imputer.transform(X)
    X_imputed_scaled = scaler.transform(X_imputed)

    logging.info("Step 5: Generating evaluations...")
    generate_tranche_analysis(final_models, X_imputed, y, output_dir, config,
                              X_scaled=X_imputed_scaled)
    run_variant_classification_counts(
        df_labeled, X_imputed, y, final_models, output_dir,
        target_sens=config._raw_get('evaluation.variant_classification_target_sens', 0.99),
        X_scaled=X_imputed_scaled,
    )

    if config._raw_get('evaluation.external_validation.enabled', False):
        logging.info("Step 6: Running comprehensive external validation...")
        run_comprehensive_external_validation(final_models, df_labeled, feature_keys, output_dir, config,
                                              imputer=imputer, scaler=scaler)

    logging.info("Step 7: Statistical comparisons...")
    oof_df = df_labeled[["CHROM", "POS", "REF", "ALT"]].copy()
    oof_df["y_true"] = y
    for mname, scores in oof_scores.items():
        if scores is not None and not np.all(np.isnan(scores)):
            oof_df[f"{mname}_oof"] = scores
    oof_df.to_csv(os.path.join(output_dir, "cv_oof_scores.csv"), index=False)

    pairwise_stats = []
    oof_score_cols = [c for c in oof_df.columns if c.endswith("_oof")]
    oof_model_names = [c.replace("_oof", "") for c in oof_score_cols]
    # Build pairs from OOF columns that actually have non-NaN values
    valid_oof_cols = [c for c in oof_score_cols
                     if not np.all(np.isnan(oof_df[c].values))]
    valid_model_names = [c.replace("_oof", "") for c in valid_oof_cols]
    comparisons = _resolve_pairwise_comparisons(
        config,
        [f"{m}_score" for m in valid_model_names]   # helper expects _score suffix
    )
    # Remap back to _oof suffix for actual lookup
    for model_A, model_B in comparisons:
        col_A, col_B = f"{model_A}_oof", f"{model_B}_oof"
        if col_A not in oof_df.columns or col_B not in oof_df.columns:
            missing = [c for c in [col_A, col_B] if c not in oof_df.columns]
            logging.warning(
                f"Skipping pairwise comparison {model_A} vs {model_B}: "
                f"OOF column(s) missing or all-NaN: {missing}. "
                f"Check that LightGBM is installed and cv_oof_scores.csv has non-NaN values."
            )
            continue
        sA, sB = oof_df[col_A].values, oof_df[col_B].values
        mask = ~np.isnan(sA) & ~np.isnan(sB)
        if mask.sum() == 0:
            logging.warning(
                f"Skipping pairwise comparison {model_A} vs {model_B}: "
                f"no overlapping non-NaN OOF values after masking."
            )
            continue
        dl = delong_roc_test(y[mask], sA[mask], sB[mask])
        primary_sens = config._raw_get('evaluation.primary_sensitivity', 0.999)
        for mname, mfn in [
            ("ROC_AUC", lambda yt, s: roc_auc_score(yt, s)),
            ("PR_AUC", lambda yt, s: average_precision_score(yt, s)),
            (f"Precision@Sens{primary_sens}", lambda yt, s: precision_at_sensitivity(yt, s, primary_sens)),
        ]:
            boot = paired_bootstrap_delta(
                y[mask], sA[mask], sB[mask], mfn,
                n_boot=config._raw_get('statistics.pairwise_bootstrap_n', 1000),
                seed=config._raw_get('statistics.pairwise_bootstrap_seed', 123),
            )
            pairwise_stats.append({"Comparison": f"{model_A}_vs_{model_B}", "Metric": mname,
                                   "A": boot["A"], "B": boot["B"],
                                   "Delta": boot["Delta_point"],
                                   "CI_low": boot["CI_low"], "CI_high": boot["CI_high"],
                                   "p_two_sided": boot["p_two_sided"],
                                   "DeLong_p": dl["p_value"] if mname == "ROC_AUC" else np.nan})

    # Always write the file — even if empty — so its absence never silently hides issues.
    df_pw = pd.DataFrame(pairwise_stats)
    if df_pw.empty:
        logging.warning(
            "pairwise_statistical_tests.csv will be written but is EMPTY. "
            "Likely cause: LightGBM is not installed or all LGB/Hybrid models failed "
            "every CV fold. The always-available GM/BGM/LR/RF pairs should still appear "
            "unless those models also failed."
        )
    else:
        df_pw["p_two_sided_Holm"] = np.nan
        for mn in df_pw["Metric"].dropna().unique():
            rows = df_pw["Metric"] == mn
            df_pw.loc[rows, "p_two_sided_Holm"] = holm_bonferroni(df_pw.loc[rows, "p_two_sided"].tolist())
        auc_rows = df_pw["Metric"] == "ROC_AUC"
        if auc_rows.any():
            df_pw.loc[auc_rows, "DeLong_p_Holm"] = holm_bonferroni(df_pw.loc[auc_rows, "DeLong_p"].tolist())
    df_pw.to_csv(os.path.join(output_dir, "pairwise_statistical_tests.csv"), index=False)
    logging.info(
        f"Pairwise statistical tests saved ({len(df_pw)} rows, "
        f"{df_pw['Comparison'].nunique() if not df_pw.empty else 0} comparisons)."
    )

    # ---- Step 8: Visualisations ----
    logging.info("Step 8: Generating visualisations...")
    generate_all_train_plots(
        per_fold_df=per_fold_df,
        y_true=y,
        oof_scores=oof_scores,
        final_models=final_models,
        feature_keys=feature_keys,
        pairwise_stats=pairwise_stats,
        output_dir=output_dir,
    )

    elapsed = time.time() - start_time
    logging.info("=" * 80)
    logging.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logging.info(f"Total time: {elapsed:.2f}s  |  Output: {output_dir}")
    logging.info("=" * 80)
    return output_dir


def run_apply_pipeline(config: PipelineConfig, output_dir: str):
    start_time = time.time()
    # output_dir is created and logging is configured by main() before this call.
    trained_model_dir = config._raw_get('paths.trained_model_dir', '')
    if not trained_model_dir or not os.path.isdir(trained_model_dir):
        raise ValueError(
            "apply mode requires 'paths.trained_model_dir' to point to a completed "
            f"training output directory. Got: '{trained_model_dir}'"
        )

    logging.info("=" * 80)
    logging.info("STARTING APPLY PIPELINE")
    logging.info(f"  trained_model_dir : {trained_model_dir}")
    logging.info(f"  apply output_dir  : {output_dir}")
    logging.info("=" * 80)

    logging.info("Step 1: Scoring real data...")
    scores_df = score_real_data_fixed_complete(
        config._raw_get('paths.apply_input_vcf'),
        trained_model_dir,
        output_dir,
        config,
    )

    logging.info("Step 2: Generating score summaries...")
    score_cols = [c for c in scores_df.columns if c.endswith("_score")]
    summary_records = []
    for col in score_cols:
        s = pd.to_numeric(scores_df[col], errors='coerce')
        summary_records.append({"Model": col.replace("_score", ""),
                                 "Mean": s.mean(), "Std": s.std(),
                                 "Min": s.min(), "25%": s.quantile(0.25),
                                 "Median": s.median(), "75%": s.quantile(0.75),
                                 "Max": s.max(), "N": s.count()})
    pd.DataFrame(summary_records).to_csv(
        os.path.join(output_dir, "score_summary_statistics.csv"), index=False)

    if config._raw_get('evaluation.external_validation.enabled', False):
        logging.info("Step 3: External database validation...")
        popdb_manager = PopulationDatabaseManager(config)
        external_set = popdb_manager.load_population_variants_for_chromosomes(scores_df)

        # Build a boolean array aligned to scores_df for reuse in enrichment comparison
        # (avoids re-scanning all population VCF files a second time).
        precomputed_is_supported: Optional[np.ndarray] = None
        if external_set:
            chrom_norm = scores_df["CHROM"].astype(str).str.replace("^chr", "", regex=True)
            pos_arr = pd.to_numeric(scores_df["POS"], errors="coerce").fillna(-1).astype(int)
            ref_arr = scores_df["REF"].astype(str)
            alt_arr = scores_df["ALT"].astype(str)
            precomputed_is_supported = np.array(
                [1 if (c, p, r, a) in external_set else 0
                 for c, p, r, a in zip(chrom_norm, pos_arr, ref_arr, alt_arr)],
                dtype=bool,
            )
            df_labeled = label_variants_with_external_set(scores_df, external_set, "IsExternalVariant")
            scores_df["IsExternalVariant"] = df_labeled["IsExternalVariant"]
            scores_df.to_csv(os.path.join(output_dir, "scored_variants_with_external_validation.csv"),
                             index=False)
            ext_count = int(scores_df["IsExternalVariant"].sum())
            total = len(scores_df)
            logging.info(f"Found {ext_count}/{total} ({100*ext_count/total:.1f}%) variants in external DBs")
            with open(os.path.join(output_dir, "external_validation_summary.json"), 'w') as f:
                json.dump({"Total_Variants": total, "External_Variants": ext_count,
                           "Novel_Variants": total - ext_count,
                           "Percent_External": 100.0 * ext_count / total}, f, indent=2)

        scores_csv = os.path.join(output_dir, "real_data_scores.csv")
        if config._raw_get('evaluation.external_validation.clinvar.enabled', False):
            clinvar_vcf = config._raw_get('evaluation.external_validation.clinvar.vcf')
            if clinvar_vcf:
                try:
                    clinvar_based_model_comparison(scores_csv, clinvar_vcf, output_dir, config)
                except Exception as e:
                    logging.warning(f"ClinVar evaluation failed: {e}")

        pop_paths = config._raw_get('evaluation.external_validation.population_dbs', []) or []
        if pop_paths:
            try:
                population_db_enrichment_comparison(
                    scores_csv=scores_csv, db_vcf_paths=pop_paths,
                    output_dir=output_dir, config=config,
                    chunk_bp=int(config._raw_get('evaluation.external_validation.popdb_chunk_bp', 1_000_000)),
                    fetch_retries=int(config._raw_get('evaluation.external_validation.popdb_fetch_retries', 3)),
                    precomputed_is_supported=precomputed_is_supported,  # reuse first scan
                )
            except Exception as e:
                logging.warning(f"PopDB enrichment evaluation failed: {e}")

    # ---- Apply-mode visualisations ----
    logging.info("Generating apply-mode visualisations...")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_apply_score_distributions(scores_df, plots_dir)
    plot_apply_score_boxplots(scores_df, plots_dir)

    # ---- OOF-based pairwise statistical tests from training run ----
    # OOF scores do not exist for new apply data (no ground-truth labels).
    # However the training run saved cv_oof_scores.csv in trained_model_dir.
    # We load those scores, re-run the full pairwise bootstrap + DeLong tests,
    # and write the results into the apply output directory so it is self-contained.
    logging.info("Loading training OOF scores for pairwise statistical tests...")
    oof_csv_src = os.path.join(trained_model_dir, "cv_oof_scores.csv")
    if os.path.exists(oof_csv_src):
        try:
            import shutil
            # Copy the raw OOF CSV so it is available alongside apply outputs
            shutil.copy(oof_csv_src, os.path.join(output_dir, "training_cv_oof_scores.csv"))
            # Copy other training summaries for completeness
            for fname in ["cv_evaluation_summary.csv", "cv_metrics_per_fold.csv",
                          "cv_paired_tests.txt"]:
                src = os.path.join(trained_model_dir, fname)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(output_dir, f"training_{fname}"))

            oof_df = pd.read_csv(oof_csv_src)
            y_oof = oof_df["y_true"].to_numpy(dtype=int) if "y_true" in oof_df.columns else None

            if y_oof is not None and len(np.unique(y_oof)) > 1:
                oof_score_cols = [c for c in oof_df.columns if c.endswith("_oof")]
                # Build oof_scores dict (model_name → array), filtering all-NaN columns
                oof_scores_map = {}
                for col in oof_score_cols:
                    arr = pd.to_numeric(oof_df[col], errors="coerce").to_numpy(dtype=float)
                    if not np.all(np.isnan(arr)):
                        oof_scores_map[col.replace("_oof", "")] = arr

                # Add non-NaN OOF columns to a working DataFrame for pairwise tests
                oof_work = oof_df[["y_true"]].copy()
                for mname, arr in oof_scores_map.items():
                    oof_work[f"{mname}_oof"] = arr

                # Build all-vs-all pairs from models that actually have OOF scores
                valid_oof_score_cols = [f"{m}_score" for m in oof_scores_map]
                comparisons_oof = _resolve_pairwise_comparisons(config, valid_oof_score_cols)
                primary_sens = config._raw_get('evaluation.primary_sensitivity', 0.999)
                pairwise_stats_oof = []

                for model_A, model_B in comparisons_oof:
                    col_A, col_B = f"{model_A}_oof", f"{model_B}_oof"
                    if col_A not in oof_work.columns or col_B not in oof_work.columns:
                        continue
                    sA = oof_work[col_A].to_numpy(dtype=float)
                    sB = oof_work[col_B].to_numpy(dtype=float)
                    mask = ~np.isnan(sA) & ~np.isnan(sB)
                    if mask.sum() == 0:
                        continue
                    try:
                        dl = delong_roc_test(y_oof[mask], sA[mask], sB[mask])
                    except Exception:
                        dl = {"p_value": np.nan, "auc1": np.nan, "auc2": np.nan,
                              "delta": np.nan, "z": np.nan, "se": np.nan}
                    for mname_metric, mfn in [
                        ("ROC_AUC",      lambda yt, s: roc_auc_score(yt, s)),
                        ("PR_AUC",       lambda yt, s: average_precision_score(yt, s)),
                        (f"Precision@Sens{primary_sens}",
                                         lambda yt, s: precision_at_sensitivity(yt, s, primary_sens)),
                    ]:
                        try:
                            boot = paired_bootstrap_delta(
                                y_oof[mask], sA[mask], sB[mask], mfn,
                                n_boot=config._raw_get('statistics.pairwise_bootstrap_n', 1000),
                                seed=config._raw_get('statistics.pairwise_bootstrap_seed', 123),
                            )
                        except Exception:
                            boot = {"A": np.nan, "B": np.nan, "Delta_point": np.nan,
                                    "CI_low": np.nan, "CI_high": np.nan, "p_two_sided": np.nan}
                        pairwise_stats_oof.append({
                            "Comparison": f"{model_A}_vs_{model_B}",
                            "Metric": mname_metric,
                            "A": boot["A"], "B": boot["B"],
                            "Delta": boot["Delta_point"],
                            "CI_low": boot["CI_low"], "CI_high": boot["CI_high"],
                            "p_two_sided": boot["p_two_sided"],
                            "DeLong_p": dl["p_value"] if mname_metric == "ROC_AUC" else np.nan,
                            "DeLong_AUC_A": dl.get("auc1", np.nan) if mname_metric == "ROC_AUC" else np.nan,
                            "DeLong_AUC_B": dl.get("auc2", np.nan) if mname_metric == "ROC_AUC" else np.nan,
                            "DeLong_z":    dl.get("z",   np.nan)   if mname_metric == "ROC_AUC" else np.nan,
                            "DeLong_SE":   dl.get("se",  np.nan)   if mname_metric == "ROC_AUC" else np.nan,
                        })

                df_oof_pw = pd.DataFrame(pairwise_stats_oof)
                if not df_oof_pw.empty:
                    df_oof_pw["p_two_sided_Holm"] = np.nan
                    for mn in df_oof_pw["Metric"].dropna().unique():
                        rows = df_oof_pw["Metric"] == mn
                        df_oof_pw.loc[rows, "p_two_sided_Holm"] = holm_bonferroni(
                            df_oof_pw.loc[rows, "p_two_sided"].tolist())
                    auc_rows = df_oof_pw["Metric"] == "ROC_AUC"
                    if auc_rows.any():
                        df_oof_pw.loc[auc_rows, "DeLong_p_Holm"] = holm_bonferroni(
                            df_oof_pw.loc[auc_rows, "DeLong_p"].tolist())
                df_oof_pw.to_csv(
                    os.path.join(output_dir, "oof_pairwise_statistical_tests.csv"), index=False)
                logging.info(
                    f"OOF pairwise statistical tests saved "
                    f"({len(df_oof_pw)} rows, "
                    f"{df_oof_pw['Comparison'].nunique() if not df_oof_pw.empty else 0} pairs). "
                    f"NOTE: these tests are on the TRAINING data OOF scores, not the apply VCF."
                )
            else:
                logging.warning("OOF CSV has no y_true column or single class — skipping OOF pairwise tests.")
        except Exception as e:
            logging.warning(f"OOF pairwise statistical tests failed: {e}", exc_info=True)
    else:
        logging.info(
            f"No cv_oof_scores.csv found in trained_model_dir ({trained_model_dir}); "
            f"skipping OOF pairwise statistical tests."
        )

    elapsed = time.time() - start_time

    # ── Model inventory summary ────────────────────────────────────────────────
    all_model_names = [c.replace("_score", "") for c in score_cols]
    logging.info("=" * 80)
    logging.info(f"MODELS EVALUATED IN THIS RUN ({len(all_model_names)} total):")
    for i, mname in enumerate(all_model_names, 1):
        logging.info(f"  {i:>2}. {mname}")
    logging.info("=" * 80)

    logging.info("=" * 80)
    logging.info("APPLY PIPELINE COMPLETED SUCCESSFULLY")
    logging.info(f"Total time: {elapsed:.2f}s  |  Output: {output_dir}")
    logging.info("=" * 80)
    return output_dir


