"""ClinVar- and population-database-based external validation, and apply-mode
VCF scoring using previously trained/saved models."""

import itertools
import json
import logging
import os
import time
from typing import Dict, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)

from .config import PipelineConfig
from .data_prep import extract_annotations, label_variants_with_external_set
from .metrics import find_threshold_at_sensitivity
from .models import LeakageAwareImputer, LeakageAwareScaler, score_model_instance
from .stats import delong_roc_test, holm_bonferroni, run_threshold_and_bootstrap_reports
from .variant_io import (
    PopulationDatabaseManager, _choose_db_contig_name, _parse_chrom_from_popdb_path,
    load_clinvar_dict, open_variantfile_with_optional_local_index,
)


def clinvar_based_model_comparison(scores_csv, clinvar_vcf, output_dir, config):
    out_csv = os.path.join(output_dir, "clinvar_model_eval.csv")
    tranche_csv = os.path.join(output_dir, "clinvar_tranche_metrics.csv")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(scores_csv):
        logging.warning(f"Scores CSV not found: {scores_csv}. Skipping ClinVar evaluation.")
        return

    scores_df = pd.read_csv(scores_csv)
    if scores_df.empty:
        logging.warning("Scores CSV is empty; skipping ClinVar evaluation.")
        return

    scores_df["CHROM_NORM"] = scores_df["CHROM"].astype(str).str.replace("^chr", "", regex=True)
    clinvar_dict = load_clinvar_dict(clinvar_vcf)
    if not clinvar_dict:
        logging.warning("No high-confidence ClinVar variants loaded; skipping.")
        return

    # Vectorised ClinVar labelling
    keys = list(zip(scores_df["CHROM_NORM"],
                    scores_df["POS"].astype(int),
                    scores_df["REF"].astype(str),
                    scores_df["ALT"].astype(str)))
    is_pos = np.array([key in clinvar_dict for key in keys], dtype=bool)
    n_pos = int(is_pos.sum())
    if n_pos < 20:
        logging.warning(f"Only {n_pos} ClinVar-overlapping variants; skipping.")
        return
    if (~is_pos).sum() == 0:
        logging.warning("No background variants for ClinVar comparison; skipping.")
        return

    df_eval = scores_df.copy()
    df_eval["ClinVarLabel"] = is_pos.astype(int)
    y = df_eval["ClinVarLabel"].to_numpy()
    score_cols = [c for c in df_eval.columns if c.endswith("_score")]
    if not score_cols:
        logging.warning("No *_score columns; skipping ClinVar evaluation.")
        return

    results, tranche_records = [], []
    tranche_sensitivities = [1.0, 0.999, 0.99, 0.90]
    for col in score_cols:
        mname = col.replace("_score", "")
        scores = pd.to_numeric(df_eval[col], errors="coerce").to_numpy(dtype=float)
        try:
            auc = roc_auc_score(y, scores)
        except ValueError:
            continue
        median_pct = float(df_eval[col].rank(pct=True)[df_eval["ClinVarLabel"] == 1].median())
        results.append({"Model": mname, "N_ClinVar_Pos": n_pos,
                         "N_Background_Neg": int((~is_pos).sum()),
                         "AUC_ClinVar_vs_BG": float(auc),
                         "Median_ClinVar_ScorePercentile": median_pct})
        for sens in tranche_sensitivities:
            thr = find_threshold_at_sensitivity(y, scores, sens)
            if np.isnan(thr):
                continue
            preds = (scores >= thr).astype(int)
            tranche_records.append({"Model": mname, "Sensitivity": sens, "Threshold": thr,
                                     "Precision": float(precision_score(y, preds, zero_division=0)),
                                     "Recall": float(recall_score(y, preds, zero_division=0)),
                                     "F1": float(f1_score(y, preds, zero_division=0)),
                                     "FDR": float(1.0 - precision_score(y, preds, zero_division=0))})

    pd.DataFrame(results).to_csv(out_csv, index=False)
    if tranche_records:
        pd.DataFrame(tranche_records).to_csv(tranche_csv, index=False)

    try:
        plt.figure(figsize=(8, 6))
        for col in score_cols:
            mname = col.replace("_score", "")
            scores = pd.to_numeric(df_eval[col], errors="coerce").to_numpy(dtype=float)
            fpr, tpr, _ = roc_curve(y, scores)
            auc = roc_auc_score(y, scores)
            plt.plot(fpr, tpr, label=f"{mname} (AUC={auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--", linewidth=0.7)
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ClinVar ROC Curves")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "clinvar_roc_curves.png"), dpi=300)
        plt.close()
    except Exception as e:
        logging.warning(f"ClinVar ROC plot failed: {e}")

    # ── DeLong pairwise tests — all-vs-all from score columns actually present ──
    raw_user_pairs = config._raw_get('statistics.pairwise_comparisons', [])
    if raw_user_pairs:
        clinvar_pw_comparisons = [
            [a, b] for a, b in raw_user_pairs
            if f"{a}_score" in df_eval.columns and f"{b}_score" in df_eval.columns
        ]
        if not clinvar_pw_comparisons:
            logging.warning(
                "None of the user-specified pairwise_comparisons matched score columns "
                "in df_eval (ClinVar). Falling back to all-vs-all from actual score columns."
            )
            clinvar_pw_comparisons = [list(p) for p in itertools.combinations(
                [c.replace("_score", "") for c in score_cols], 2)]
    else:
        clinvar_pw_comparisons = [list(p) for p in itertools.combinations(
            [c.replace("_score", "") for c in score_cols], 2)]

    pairwise_stats = []
    for model_A, model_B in clinvar_pw_comparisons:
        col_A, col_B = f"{model_A}_score", f"{model_B}_score"
        if col_A not in df_eval.columns or col_B not in df_eval.columns:
            continue
        sA = pd.to_numeric(df_eval[col_A], errors="coerce").to_numpy(dtype=float)
        sB = pd.to_numeric(df_eval[col_B], errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(sA) & ~np.isnan(sB)
        if mask.sum() == 0:
            continue
        try:
            dl = delong_roc_test(y[mask], sA[mask], sB[mask])
        except Exception:
            dl = {"auc1": np.nan, "auc2": np.nan, "delta": np.nan,
                  "z": np.nan, "se": np.nan,
                  "p_value": np.nan,
                  "p_one_sided_auc1_lt_auc2": np.nan,
                  "p_one_sided_auc1_gt_auc2": np.nan}
        pairwise_stats.append({
            "Scenario": "ClinVar",
            "Comparison": f"{model_A}_vs_{model_B}",
            "Metric": "ROC_AUC",
            "AUC_A": dl.get("auc1", np.nan),
            "AUC_B": dl.get("auc2", np.nan),
            "Delta_AUC_A_minus_B": dl.get("delta", np.nan),
            "DeLong_z": dl.get("z", np.nan),
            "DeLong_SE": dl.get("se", np.nan),
            "DeLong_p_two_sided": dl.get("p_value", np.nan),
            "DeLong_p_one_sided_A_lt_B": dl.get("p_one_sided_auc1_lt_auc2", np.nan),
            "DeLong_p_one_sided_A_gt_B": dl.get("p_one_sided_auc1_gt_auc2", np.nan),
            "N_pos": int(y[mask].sum()),
            "N_neg": int((1 - y[mask]).sum()),
        })

    df_pw = pd.DataFrame(pairwise_stats)
    if not df_pw.empty:
        df_pw["DeLong_p_two_sided_Holm"] = holm_bonferroni(df_pw["DeLong_p_two_sided"].tolist())
        df_pw["DeLong_p_one_sided_A_lt_B_Holm"] = holm_bonferroni(
            df_pw["DeLong_p_one_sided_A_lt_B"].tolist())
    df_pw.to_csv(os.path.join(output_dir, "clinvar_pairwise_statistical_tests.csv"), index=False)
    logging.info(
        f"ClinVar pairwise statistical tests saved "
        f"({len(df_pw)} rows, {df_pw['Comparison'].nunique() if not df_pw.empty else 0} pairs)."
    )

    try:
        run_threshold_and_bootstrap_reports(df_eval, "ClinVarLabel", score_cols,
                                            output_dir, "ClinVar", config)
    except Exception as e:
        logging.warning(f"ClinVar additional statistical reports failed: {e}")


def population_db_enrichment_comparison(scores_csv, db_vcf_paths, output_dir, config,
                                        chunk_bp=1_000_000, fetch_retries=3,
                                        precomputed_is_supported: Optional[np.ndarray] = None):
    out_csv = os.path.join(output_dir, "popdb_model_eval.csv")
    tranche_csv = os.path.join(output_dir, "popdb_tranche_metrics.csv")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(scores_csv):
        logging.warning(f"Scores CSV not found: {scores_csv}. Skipping PopDB evaluation.")
        return
    if not db_vcf_paths:
        logging.info("No population DB VCF paths provided; skipping.")
        return

    scores_df = pd.read_csv(scores_csv)
    if scores_df.empty:
        logging.warning("Scores CSV empty; skipping PopDB evaluation.")
        return

    scores_df["CHROM_NORM"] = scores_df["CHROM"].astype(str).str.replace("^chr", "", regex=True)
    scores_df["POS"] = pd.to_numeric(scores_df["POS"], errors="coerce").astype("Int64")
    scores_df["REF"] = scores_df["REF"].astype(str)
    scores_df["ALT"] = scores_df["ALT"].astype(str)

    needed_chroms = sorted(scores_df["CHROM_NORM"].dropna().unique())
    max_pos_by_chrom = (scores_df.dropna(subset=["CHROM_NORM", "POS"])
                        .groupby("CHROM_NORM")["POS"].max().to_dict())

    # ── Use pre-computed labels if provided (avoids re-scanning all VCF files) ──
    if precomputed_is_supported is not None:
        logging.info(
            "population_db_enrichment_comparison: using pre-computed PopDB labels "
            "(skipping VCF re-scan — saves hours of I/O)."
        )
        is_supported = np.asarray(precomputed_is_supported, dtype=bool)
        if len(is_supported) != len(scores_df):
            logging.warning(
                f"precomputed_is_supported length {len(is_supported)} != "
                f"scores_df length {len(scores_df)}; falling back to VCF scan."
            )
            is_supported = None
    else:
        is_supported = None

    if is_supported is None:
        # ── Full VCF scan (slow path) ──────────────────────────────────────────
        from collections import defaultdict
        key_to_indices: Dict = defaultdict(list)
        for idx, row in scores_df.iterrows():
            if pd.isna(row["CHROM_NORM"]) or pd.isna(row["POS"]):
                continue
            key_to_indices[(str(row["CHROM_NORM"]), int(row["POS"]),
                            str(row["REF"]), str(row["ALT"]))].append(idx)

        is_supported = np.zeros(len(scores_df), dtype=bool)
        for vcf_path in db_vcf_paths:
            chrom_token = _parse_chrom_from_popdb_path(vcf_path)
            if chrom_token is None or chrom_token not in max_pos_by_chrom:
                continue
            try:
                vcf = open_variantfile_with_optional_local_index(vcf_path, search_dirs=[os.getcwd(), output_dir])
            except Exception as e:
                logging.warning(f"Could not open {vcf_path}: {e}")
                continue
            contig = _choose_db_contig_name(vcf, chrom_token)
            if contig is None:
                try:
                    vcf.close()
                except Exception:
                    pass
                continue
            max_pos = int(max_pos_by_chrom[chrom_token])
            consecutive_failures = 0
            failed_regions: list = []
            start0 = 0
            vcf_ok = True
            while start0 < max_pos and vcf_ok:
                end0 = min(start0 + chunk_bp, max_pos)
                chunk_ok = False
                for attempt in range(1, fetch_retries + 1):
                    try:
                        for rec in vcf.fetch(contig, start0, end0):
                            if rec.alts is None:
                                continue
                            c = str(rec.chrom).replace("chr", "")
                            for alt in rec.alts:
                                key = (c, int(rec.pos), str(rec.ref), str(alt))
                                for i in key_to_indices.get(key, []):
                                    is_supported[i] = True
                        chunk_ok = True
                        break
                    except Exception as e:
                        logging.warning(
                            f"[PopDB fetch fail] {vcf_path} {contig}:{start0}-{end0}"
                            f" (attempt {attempt}/{fetch_retries}): {e}"
                        )
                        if attempt < fetch_retries:
                            time.sleep(min(2 ** (attempt - 1), 5))
                if chunk_ok:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    failed_regions.append(f"{contig}:{start0}-{end0}")
                    if consecutive_failures >= 3:
                        logging.warning(
                            f"Aborting {vcf_path} after {consecutive_failures} consecutive "
                            f"chunk failures — file likely truncated. "
                            f"Data lost from position {start0} to end of chromosome."
                        )
                        vcf_ok = False
                start0 = end0
            if failed_regions:
                logging.warning(
                    f"{vcf_path}: {len(failed_regions)} chunk(s) failed permanently: "
                    + ", ".join(failed_regions[:5])
                    + (" ..." if len(failed_regions) > 5 else "")
                )
            try:
                vcf.close()
            except Exception as close_err:
                logging.warning(f"Could not close {vcf_path} cleanly (likely truncated): {close_err}")

    n_pos = int(is_supported.sum())
    if n_pos < 20:
        logging.warning(f"Only {n_pos} PopDB-supported variants found; skipping.")
        return
    if (~is_supported).sum() == 0:
        logging.warning("No background variants; cannot build PopDB comparison.")
        return

    df_eval = scores_df.copy()
    df_eval["PopDBLabel"] = is_supported.astype(int)
    y = df_eval["PopDBLabel"].to_numpy()
    score_cols = [c for c in df_eval.columns if c.endswith("_score")]
    if not score_cols:
        logging.warning("No *_score columns for PopDB evaluation.")
        return

    results, tranche_records = [], []
    tranche_sensitivities = [1.0, 0.999, 0.99, 0.90]

    for col in score_cols:
        mname = col.replace("_score", "")
        scores = pd.to_numeric(df_eval[col], errors="coerce").to_numpy(dtype=float)
        try:
            auc = roc_auc_score(y, scores)
        except ValueError:
            continue
        results.append({"Model": mname, "N_PopDB_Pos": n_pos,
                         "N_Background_Neg": int((~is_supported).sum()),
                         "AUC_PopDB_vs_BG": float(auc),
                         "Median_PopDB_ScorePercentile": float(
                             df_eval[col].rank(pct=True)[df_eval["PopDBLabel"] == 1].median())})

        for sens in tranche_sensitivities:
            thr = find_threshold_at_sensitivity(y, scores, sens)
            if np.isnan(thr):
                continue
            preds = (scores >= thr).astype(int)
            tranche_records.append({
                "Model": mname,
                "Sensitivity": sens,
                "Threshold": float(thr),
                "Precision": float(precision_score(y, preds, zero_division=0)),
                "Recall": float(recall_score(y, preds, zero_division=0)),
                "F1": float(f1_score(y, preds, zero_division=0)),
                "FDR": float(1.0 - precision_score(y, preds, zero_division=0)),
            })

    try:
        plt.figure(figsize=(8, 6))
        for col in score_cols:
            mname = col.replace("_score", "")
            scores = pd.to_numeric(df_eval[col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(scores)
            if mask.sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y[mask], scores[mask])
            auc = roc_auc_score(y[mask], scores[mask])
            plt.plot(fpr, tpr, label=f"{mname} (AUC={auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--", linewidth=0.7)
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("PopDB ROC Curves")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "popdb_roc.png"), dpi=300)
        plt.close()
    except Exception as e:
        logging.warning(f"PopDB ROC plot failed: {e}")

    pd.DataFrame(results).to_csv(out_csv, index=False)

    if tranche_records:
        pd.DataFrame(tranche_records).to_csv(tranche_csv, index=False)

    # ── Pairwise DeLong statistical tests (PopDB label as ground truth) ────────
    # Always build pairs from the score columns that actually exist in df_eval.
    # We do NOT rely on config.get('statistics.pairwise_comparisons') here because
    # the default list uses legacy short names (LR, RF) that may not match the
    # final model save names (LogReg, RandForest) written to the scores CSV.
    raw_user_pairs = config._raw_get('statistics.pairwise_comparisons', [])
    if raw_user_pairs:
        # User explicitly listed specific pairs — respect them, but filter to
        # those where both score columns actually exist.
        pw_comparisons = [
            [a, b] for a, b in raw_user_pairs
            if f"{a}_score" in df_eval.columns and f"{b}_score" in df_eval.columns
        ]
        if not pw_comparisons:
            logging.warning(
                "None of the user-specified pairwise_comparisons matched score columns "
                f"in df_eval. Falling back to all-combinations from actual score columns."
            )
            pw_comparisons = [list(p) for p in itertools.combinations(
                [c.replace("_score", "") for c in score_cols], 2)]
    else:
        # Default: compare every model pair that actually has scores.
        pw_comparisons = [list(p) for p in itertools.combinations(
            [c.replace("_score", "") for c in score_cols], 2)]

    popdb_pw_rows = []
    for model_A, model_B in pw_comparisons:
        col_A, col_B = f"{model_A}_score", f"{model_B}_score"
        if col_A not in df_eval.columns or col_B not in df_eval.columns:
            continue
        sA = pd.to_numeric(df_eval[col_A], errors="coerce").to_numpy(dtype=float)
        sB = pd.to_numeric(df_eval[col_B], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(sA) & np.isfinite(sB)
        if mask.sum() == 0:
            continue
        try:
            dl = delong_roc_test(y[mask], sA[mask], sB[mask])
        except Exception:
            dl = {"auc1": np.nan, "auc2": np.nan, "delta": np.nan,
                  "p_value": np.nan,
                  "p_one_sided_auc1_lt_auc2": np.nan,
                  "p_one_sided_auc1_gt_auc2": np.nan}
        popdb_pw_rows.append({
            "Scenario": "PopDB",
            "Comparison": f"{model_A}_vs_{model_B}",
            "Metric": "ROC_AUC",
            "AUC_A": dl.get("auc1", np.nan),
            "AUC_B": dl.get("auc2", np.nan),
            "Delta_AUC_A_minus_B": dl.get("delta", np.nan),
            "DeLong_z": dl.get("z", np.nan),
            "DeLong_SE": dl.get("se", np.nan),
            "DeLong_p_two_sided": dl.get("p_value", np.nan),
            "DeLong_p_one_sided_A_lt_B": dl.get("p_one_sided_auc1_lt_auc2", np.nan),
            "DeLong_p_one_sided_A_gt_B": dl.get("p_one_sided_auc1_gt_auc2", np.nan),
            "N_pos": int(y[mask].sum()),
            "N_neg": int((1 - y[mask]).sum()),
        })

    df_popdb_pw = pd.DataFrame(popdb_pw_rows)
    if not df_popdb_pw.empty:
        df_popdb_pw["DeLong_p_two_sided_Holm"] = holm_bonferroni(
            df_popdb_pw["DeLong_p_two_sided"].tolist())
        df_popdb_pw["DeLong_p_one_sided_A_lt_B_Holm"] = holm_bonferroni(
            df_popdb_pw["DeLong_p_one_sided_A_lt_B"].tolist())
    df_popdb_pw.to_csv(
        os.path.join(output_dir, "popdb_pairwise_statistical_tests.csv"), index=False)
    logging.info(
        f"PopDB pairwise statistical tests saved "
        f"({len(df_popdb_pw)} rows, {df_popdb_pw['Comparison'].nunique() if not df_popdb_pw.empty else 0} pairs)."
    )

    try:
        run_threshold_and_bootstrap_reports(df_eval, "PopDBLabel", score_cols,
                                            output_dir, "PopDB", config)
    except Exception as e:
        logging.warning(f"PopDB additional statistical reports failed: {e}")


# ============================================================================
# EXTERNAL VALIDATION
# ============================================================================


def run_comprehensive_external_validation(models, df_variants, feature_keys, output_dir, config,
                                          imputer: Optional[LeakageAwareImputer] = None,
                                          scaler: Optional[LeakageAwareScaler] = None):
    """imputer/scaler must be the SAME fitted objects final_models were
    trained with (train_final_models_fixed's returned imputer/scaler) —
    scoring against raw, un-imputed, un-scaled features would silently
    corrupt GM/BGM/LogReg scores and can propagate NaNs through predict_proba."""
    logging.info("Running comprehensive external validation with population databases...")
    popdb_manager = PopulationDatabaseManager(config)
    external_set = popdb_manager.load_population_variants_for_chromosomes(df_variants)
    if not external_set:
        logging.warning("No external variants loaded for validation")
        return None

    df_labeled = label_variants_with_external_set(df_variants, external_set, "IsExternalVariant")
    X = df_labeled[feature_keys].to_numpy(dtype=np.float32)
    y = df_labeled["IsExternalVariant"].to_numpy(dtype=int)

    if imputer is not None:
        X_imputed = imputer.transform(X)
    else:
        logging.warning("run_comprehensive_external_validation called without an imputer; "
                        "scoring on raw (possibly NaN-containing) features.")
        X_imputed = X
    if scaler is not None:
        X_imputed_scaled = scaler.transform(X_imputed)
    else:
        logging.warning("run_comprehensive_external_validation called without a scaler; "
                        "GM/BGM/LogReg scores will be computed on unscaled features.")
        X_imputed_scaled = X_imputed

    results = []
    detailed = df_labeled[["CHROM", "POS", "REF", "ALT", "IsExternalVariant"]].copy()
    for name, model in models.items():
        try:
            scores = score_model_instance(name, model, X_imputed, X_scaled=X_imputed_scaled)
        except Exception as e:
            logging.warning(f"Could not evaluate model {name}: {e}")
            continue
        auc = roc_auc_score(y, scores)
        thr = find_threshold_at_sensitivity(y, scores, 0.99)
        preds = (scores >= thr).astype(int) if not np.isnan(thr) else np.zeros_like(y)
        results.append({'Model': name, 'AUC': auc, 'Threshold': thr,
                         'Precision': precision_score(y, preds, zero_division=0),
                         'Recall': recall_score(y, preds, zero_division=0),
                         'F1': f1_score(y, preds, zero_division=0),
                         'TP': int(((y == 1) & (preds == 1)).sum()),
                         'FP': int(((y == 0) & (preds == 1)).sum()),
                         'TN': int(((y == 0) & (preds == 0)).sum()),
                         'FN': int(((y == 1) & (preds == 0)).sum())})
        detailed[f"{name}_Score"] = scores
        if not np.isnan(thr):
            detailed[f"{name}_Prediction"] = preds
            detailed[f"{name}_Threshold"] = thr

    if results:
        pd.DataFrame(results).to_csv(os.path.join(output_dir, "external_validation_results.csv"), index=False)
        detailed.to_csv(os.path.join(output_dir, "external_validation_detailed.csv"), index=False)

    logging.info(f"External validation complete. Results saved to {output_dir}")
    return df_labeled


# ============================================================================
# APPLY MODE — SCORING
# ============================================================================


def score_real_data_fixed_complete(vcf_path: str, model_dir_root: str,
                                   output_dir: str, config: PipelineConfig) -> pd.DataFrame:
    """Score a VCF using models stored in model_dir_root (a previous train output dir)."""
    logging.info(f"Scoring real data: {vcf_path}")
    model_dir = os.path.join(model_dir_root, "models")
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    imputer = LeakageAwareImputer.load(metadata["imputer_path"])
    scaler = LeakageAwareScaler.load(metadata["scaler_path"])
    batch_size = config._raw_get('memory_safety.batch_size', 100000)
    df_annotations = extract_annotations(vcf_path, config._raw_get('features', []),
                                         batch_size=batch_size)

    feature_keys = config._raw_get('features', [])
    for col in feature_keys:
        if col not in df_annotations.columns:
            df_annotations[col] = np.nan
            logging.warning(f"Feature {col} not found in VCF")

    X = df_annotations[feature_keys].to_numpy(dtype=np.float32)
    X_imputed = imputer.transform(X)
    X_imputed_scaled = scaler.transform(X_imputed)
    scores_df = df_annotations[["CHROM", "POS", "REF", "ALT"]].copy()

    for model_name, model_path in metadata["model_paths"].items():
        try:
            if model_name in ("GM", "BGM"):
                if isinstance(model_path, (list, tuple)) and len(model_path) == 2:
                    model_obj = (joblib.load(model_path[0]), joblib.load(model_path[1]))
                else:
                    logging.warning(f"Unexpected GM/BGM path format for {model_name}; skipping")
                    continue
            else:
                model_obj = joblib.load(model_path)
            # score_model_instance returns (0,1) for all model types
            scores_df[f"{model_name}_score"] = score_model_instance(
                model_name, model_obj, X_imputed, X_scaled=X_imputed_scaled)
        except Exception as e:
            logging.error(f"Failed to score with model {model_name}: {e}")
            scores_df[f"{model_name}_score"] = np.nan

    scores_csv = os.path.join(output_dir, "real_data_scores.csv")
    scores_df.to_csv(scores_csv, index=False)
    logging.info(f"Real data scores saved to {scores_csv}")
    return scores_df


