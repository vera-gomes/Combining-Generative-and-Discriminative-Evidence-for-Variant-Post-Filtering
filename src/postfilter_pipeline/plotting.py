"""All plot-generating functions for both train and apply pipeline modes.

Each plot function accepts the data it needs directly (no side-effects on
pipeline state), saves PNGs to plots_dir, and is wrapped in try/except so a
plotting failure never aborts the pipeline.
"""

import logging
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .models import HYBRID_BASE_MODELS, is_hybrid_model_name, is_hybrid_model_object


# ============================================================================
# VISUALISATION MODULE
# ============================================================================
#
# All plot functions follow the same contract:
#   - Accept the data they need directly (no side-effects on pipeline state)
#   - Save PNGs to plots_dir; return nothing
#   - Wrapped in try/except so a plotting failure never aborts the pipeline
# ============================================================================

# Colour palette — consistent across all plots
_MODEL_PALETTE = [
    "#2196F3",  # blue        GM
    "#03A9F4",  # light-blue  BGM
    "#9C27B0",  # purple      LR
    "#E91E63",  # pink        RF
    "#FF5722",  # deep-orange LGB
    "#FF9800",  # orange      LGB_Bayes
    "#FFC107",  # amber       LGB_MultiObj
    "#4CAF50",  # green       Hybrid_LGB
    "#00BCD4",  # cyan        Hybrid_LGB_Bayes
    "#009688",  # teal        Hybrid_LGB_MultiObj
    "#795548",  # brown       Hybrid_LGB_alpha1  (pure LGB)
    "#FF8F00",  # amber       Hybrid_LGB_alpha08 (80% LGB + 20% GMM)
    "#607D8B",  # blue-grey   Hybrid_LGB_alpha07 (70% LGB + 30% GMM)
]


def _model_color(name: str, all_names: List[str]) -> str:
    try:
        return _MODEL_PALETTE[all_names.index(name) % len(_MODEL_PALETTE)]
    except ValueError:
        return "#607D8B"


def _savefig(fig, path: str):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved plot: {path}")


# ---------------------------------------------------------------------------
# 1. CV metric bar chart — average ± std across folds for every model
# ---------------------------------------------------------------------------

def plot_cv_metric_bars(per_fold_df: pd.DataFrame, plots_dir: str):
    """Grouped bar chart: mean ± 1 std of AUC, F1, Precision, Recall per model."""
    try:
        metrics = ["AUC", "F1", "Precision", "Recall"]
        models = per_fold_df["Model"].unique().tolist()
        # drop models with all-NaN
        models = [m for m in models if per_fold_df.loc[per_fold_df["Model"] == m, "AUC"].notna().any()]
        if not models:
            return

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5), sharey=False)
        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            means, stds = [], []
            for mname in models:
                vals = per_fold_df.loc[per_fold_df["Model"] == mname, metric].dropna().values
                means.append(float(np.mean(vals)) if len(vals) else np.nan)
                stds.append(float(np.std(vals)) if len(vals) else 0.0)

            x = np.arange(len(models))
            colors = [_model_color(m, models) for m in models]
            bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors,
                          edgecolor='white', linewidth=0.5, error_kw={"elinewidth": 1.2})
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=40, ha='right', fontsize=7)
            ax.set_title(metric, fontsize=9, fontweight='bold')
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=0.3, linewidth=0.5)
            ax.spines[['top', 'right']].set_visible(False)
            # annotate mean
            for bar, mean in zip(bars, means):
                if not np.isnan(mean):
                    ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.01,
                            f"{mean:.3f}", ha='center', va='bottom', fontsize=6)

        fig.suptitle("Cross-Validation Metrics (mean ± std across folds)", fontsize=10, y=1.01)
        fig.tight_layout()
        _savefig(fig, os.path.join(plots_dir, "cv_metric_bars.png"))
    except Exception as e:
        logging.warning(f"plot_cv_metric_bars failed: {e}")


# ---------------------------------------------------------------------------
# 2. CV fold stability heatmap (models × folds, AUC)
# ---------------------------------------------------------------------------

def plot_cv_fold_heatmap(per_fold_df: pd.DataFrame, plots_dir: str):
    """Heatmap of per-fold AUC for every model — reveals unstable folds."""
    try:
        pivot = per_fold_df.pivot_table(index="Model", columns="Fold", values="AUC")
        if pivot.empty:
            return
        # sort by mean AUC descending
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

        fig, ax = plt.subplots(figsize=(max(5, pivot.shape[1] * 1.1), max(4, pivot.shape[0] * 0.55)))
        im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn', vmin=0.45, vmax=1.0)
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels([f"Fold {c}" for c in pivot.columns], fontsize=8)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index.tolist(), fontsize=8)
        ax.set_title("Per-Fold AUC Heatmap", fontsize=10, fontweight='bold')
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha='center', va='center',
                            fontsize=6.5, color='black' if 0.55 < val < 0.9 else 'white')
        plt.colorbar(im, ax=ax, label="AUC", fraction=0.03, pad=0.02)
        fig.tight_layout()
        _savefig(fig, os.path.join(plots_dir, "cv_fold_heatmap.png"))
    except Exception as e:
        logging.warning(f"plot_cv_fold_heatmap failed: {e}")


# ---------------------------------------------------------------------------
# 3 & 4. OOF ROC and Precision-Recall curves
# ---------------------------------------------------------------------------

def plot_oof_roc_and_pr(y_true: np.ndarray, oof_scores: Dict[str, np.ndarray],
                         plots_dir: str):
    """ROC and PR curves built from out-of-fold predictions (unbiased estimate)."""
    try:
        from sklearn.metrics import roc_curve, precision_recall_curve, auc as sk_auc
        models = [m for m, s in oof_scores.items()
                  if s is not None and not np.all(np.isnan(s))]
        if not models:
            return

        for fname, curve_fn, xlabel, ylabel, title, diag in [
            ("oof_roc_curves.png",
             lambda y, s: roc_curve(y, s),
             "False Positive Rate", "True Positive Rate",
             "Out-of-Fold ROC Curves", True),
            ("oof_pr_curves.png",
             lambda y, s: precision_recall_curve(y, s),
             "Recall", "Precision",
             "Out-of-Fold Precision-Recall Curves", False),
        ]:
            fig, ax = plt.subplots(figsize=(7, 6))
            for mname in models:
                s = oof_scores[mname]
                mask = ~np.isnan(s)
                if mask.sum() < 2 or len(np.unique(y_true[mask])) < 2:
                    continue
                try:
                    result = curve_fn(y_true[mask], s[mask])
                    if len(result) == 3:
                        x_vals, y_vals, _ = result
                        area = roc_auc_score(y_true[mask], s[mask])
                    else:
                        y_vals, x_vals, _ = result   # PR: precision, recall, thresholds
                        area = sk_auc(x_vals, y_vals)
                    ax.plot(x_vals, y_vals,
                            color=_model_color(mname, models), lw=1.5,
                            label=f"{mname} ({area:.3f})")
                except Exception:
                    continue
            if diag:
                ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
            else:
                base = float(y_true.mean())
                ax.axhline(base, color='k', lw=0.8, ls='--', alpha=0.5,
                           label=f"Baseline ({base:.3f})")
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.legend(fontsize=7, loc='lower right' if diag else 'upper right',
                      framealpha=0.85)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.2)
            ax.spines[['top', 'right']].set_visible(False)
            fig.tight_layout()
            _savefig(fig, os.path.join(plots_dir, fname))
    except Exception as e:
        logging.warning(f"plot_oof_roc_and_pr failed: {e}")


# ---------------------------------------------------------------------------
# 5. Score distribution — positives vs negatives per model
# ---------------------------------------------------------------------------

def plot_score_distributions(y_true: np.ndarray, oof_scores: Dict[str, np.ndarray],
                              plots_dir: str):
    """KDE/histogram of scores split by true label for each model."""
    try:
        models = [m for m, s in oof_scores.items()
                  if s is not None and not np.all(np.isnan(s))]
        if not models:
            return

        ncols = min(3, len(models))
        nrows = int(np.ceil(len(models) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(5 * ncols, 3.2 * nrows), squeeze=False)

        for idx, mname in enumerate(models):
            ax = axes[idx // ncols][idx % ncols]
            s = oof_scores[mname]
            mask = ~np.isnan(s)
            s_pos = s[mask & (y_true == 1)]
            s_neg = s[mask & (y_true == 0)]
            bins = np.linspace(0, 1, 40)
            ax.hist(s_neg, bins=bins, density=True, alpha=0.55,
                    color='#F44336', label='Negative', edgecolor='none')
            ax.hist(s_pos, bins=bins, density=True, alpha=0.55,
                    color='#2196F3', label='Positive', edgecolor='none')
            ax.set_title(mname, fontsize=8, fontweight='bold')
            ax.set_xlabel("Score", fontsize=7)
            ax.set_ylabel("Density", fontsize=7)
            ax.legend(fontsize=6)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=6)

        # hide unused axes
        for idx in range(len(models), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle("Score Distributions: Positives vs Negatives (OOF)",
                     fontsize=10, y=1.01, fontweight='bold')
        fig.tight_layout()
        _savefig(fig, os.path.join(plots_dir, "score_distributions.png"))
    except Exception as e:
        logging.warning(f"plot_score_distributions failed: {e}")


# ---------------------------------------------------------------------------
# 6. Calibration curves (reliability diagrams)
# ---------------------------------------------------------------------------

def plot_calibration_curves(y_true: np.ndarray, oof_scores: Dict[str, np.ndarray],
                             plots_dir: str, n_bins: int = 10):
    """Reliability diagrams: mean predicted probability vs fraction of positives."""
    try:
        models = [m for m, s in oof_scores.items()
                  if s is not None and not np.all(np.isnan(s))]
        if not models:
            return

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label="Perfect calibration")

        for mname in models:
            s = oof_scores[mname]
            mask = ~np.isnan(s)
            if mask.sum() < 20:
                continue
            s_m = s[mask]
            y_m = y_true[mask]
            bin_edges = np.linspace(0, 1, n_bins + 1)
            mean_pred, frac_pos = [], []
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                in_bin = (s_m >= lo) & (s_m < hi)
                if in_bin.sum() == 0:
                    continue
                mean_pred.append(float(s_m[in_bin].mean()))
                frac_pos.append(float(y_m[in_bin].mean()))
            if len(mean_pred) < 2:
                continue
            ax.plot(mean_pred, frac_pos,
                    color=_model_color(mname, models), lw=1.5, marker='o',
                    markersize=4, label=mname)

        ax.set_xlabel("Mean Predicted Probability", fontsize=9)
        ax.set_ylabel("Fraction of Positives", fontsize=9)
        ax.set_title("Calibration Curves (Reliability Diagrams)", fontsize=10, fontweight='bold')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc='upper left', framealpha=0.85)
        ax.grid(alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        _savefig(fig, os.path.join(plots_dir, "calibration_curves.png"))
    except Exception as e:
        logging.warning(f"plot_calibration_curves failed: {e}")


# ---------------------------------------------------------------------------
# 7. Feature importance — LGB models and RF
# ---------------------------------------------------------------------------

def plot_feature_importance(final_models: Dict, feature_keys: List[str], plots_dir: str):
    """Horizontal bar chart of feature importances for tree-based models."""
    try:
        importance_models = {}
        for name, model in final_models.items():
            if name in HYBRID_BASE_MODELS and hasattr(model, 'feature_importances_'):
                importance_models[name] = model.feature_importances_
            elif name == 'RandForest' and hasattr(model, 'feature_importances_'):
                importance_models[name] = model.feature_importances_
        if not importance_models:
            return

        n = len(importance_models)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, max(4, len(feature_keys) * 0.4)))
        if n == 1:
            axes = [axes]

        for ax, (mname, imp) in zip(axes, importance_models.items()):
            # normalise
            imp_norm = np.asarray(imp, dtype=float)
            if imp_norm.sum() > 0:
                imp_norm = imp_norm / imp_norm.sum()
            order = np.argsort(imp_norm)
            feat_sorted = [feature_keys[i] for i in order]
            imp_sorted = imp_norm[order]
            colors = plt.cm.Blues(np.linspace(0.35, 0.9, len(feat_sorted)))
            ax.barh(feat_sorted, imp_sorted, color=colors, edgecolor='white')
            ax.set_xlabel("Relative Importance", fontsize=8)
            ax.set_title(mname, fontsize=9, fontweight='bold')
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=7)
            for i, v in enumerate(imp_sorted):
                ax.text(v + 0.002, i, f"{v:.3f}", va='center', fontsize=6)

        fig.suptitle("Feature Importances", fontsize=10, fontweight='bold', y=1.01)
        fig.tight_layout()
        _savefig(fig, os.path.join(plots_dir, "feature_importance.png"))
    except Exception as e:
        logging.warning(f"plot_feature_importance failed: {e}")


# ---------------------------------------------------------------------------
# 8. Hybrid alpha selection curves
# ---------------------------------------------------------------------------

def plot_hybrid_alpha_curves(final_models: Dict, plots_dir: str):
    """Alpha vs inner-CV AUC for each hybrid model — shows how well-constrained alpha is."""
    try:
        hybrid_names = [n for n in final_models if is_hybrid_model_name(n)
                        and is_hybrid_model_object(final_models[n])
                        and 'alpha_cv_scores' in final_models[n]]
        if not hybrid_names:
            return

        fig, axes = plt.subplots(1, len(hybrid_names),
                                  figsize=(4.5 * len(hybrid_names), 4), squeeze=False)
        for ax, hname in zip(axes[0], hybrid_names):
            info = final_models[hname]
            pairs = info['alpha_cv_scores']  # list of (alpha, auc)
            if not pairs:
                continue
            alphas = [p[0] for p in pairs]
            aucs = [p[1] for p in pairs]
            ax.plot(alphas, aucs, color='#2196F3', lw=2, marker='o', markersize=4)
            best_alpha = float(info['best_alpha'])
            best_auc = info.get('val_auc', np.nan)
            ax.axvline(best_alpha, color='#F44336', lw=1.5, ls='--',
                       label=f"Best α={best_alpha:.2f}")
            ax.set_xlabel("Alpha (weight of LGB)", fontsize=8)
            ax.set_ylabel("Inner-CV AUC", fontsize=8)
            ax.set_title(f"{hname}\nval AUC={best_auc:.4f}" if not np.isnan(best_auc) else hname,
                         fontsize=8, fontweight='bold')
            ax.legend(fontsize=7)
            ax.set_xlim(0, 1)
            ax.grid(alpha=0.25)
            ax.spines[['top', 'right']].set_visible(False)

        fig.suptitle("Hybrid Alpha Selection (LGB weight vs inner-CV AUC)",
                     fontsize=10, fontweight='bold', y=1.02)
        fig.tight_layout()
        _savefig(fig, os.path.join(plots_dir, "hybrid_alpha_curves.png"))
    except Exception as e:
        logging.warning(f"plot_hybrid_alpha_curves failed: {e}")


# ---------------------------------------------------------------------------
# 9. Pairwise delta heatmap (from pairwise_stats df)
# ---------------------------------------------------------------------------

def plot_pairwise_delta_heatmap(pairwise_stats: List[Dict], plots_dir: str,
                                 metric: str = "ROC_AUC"):
    """Square heatmap: Δmetric(A − B) for each ordered pair.
    Cells are coloured by delta magnitude; * marks p < 0.05."""
    try:
        rows = [r for r in pairwise_stats if r.get("Metric") == metric]
        if not rows:
            return

        # Collect unique model names preserving order
        seen: Dict[str, int] = {}
        for r in rows:
            a, b = r["Comparison"].split("_vs_")
            for x in (a, b):
                if x not in seen:
                    seen[x] = len(seen)
        models = list(seen.keys())
        n = len(models)
        if n < 2:
            return

        mat = np.full((n, n), np.nan)
        sig = np.full((n, n), False)
        for r in rows:
            a, b = r["Comparison"].split("_vs_")
            if a not in seen or b not in seen:
                continue
            i, j = seen[a], seen[b]
            delta = r.get("Delta", np.nan)
            p = r.get("p_two_sided_Holm", r.get("p_two_sided", np.nan))
            mat[i, j] = delta
            mat[j, i] = -delta if not np.isnan(delta) else np.nan
            sig[i, j] = (not np.isnan(p)) and p < 0.05
            sig[j, i] = sig[i, j]

        vmax = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 0.1
        fig, ax = plt.subplots(figsize=(max(6, n * 0.8), max(5, n * 0.75)))
        im = ax.imshow(mat, cmap='RdBu', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(models, fontsize=7)
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                if not np.isnan(v):
                    star = "*" if sig[i, j] else ""
                    ax.text(j, i, f"{v:+.3f}{star}", ha='center', va='center',
                            fontsize=5.5,
                            color='white' if abs(v) > 0.6 * vmax else 'black')
        plt.colorbar(im, ax=ax, label=f"Δ {metric} (row − col)", fraction=0.035, pad=0.02)
        ax.set_title(f"Pairwise Δ{metric} (A − B)\n* = Holm-corrected p < 0.05",
                     fontsize=9, fontweight='bold')
        fig.tight_layout()
        _savefig(fig, os.path.join(plots_dir, f"pairwise_delta_{metric.lower()}.png"))
    except Exception as e:
        logging.warning(f"plot_pairwise_delta_heatmap failed: {e}")


# ---------------------------------------------------------------------------
# 10. Variant classification stacked bar (SNP / INDEL breakdown)
# ---------------------------------------------------------------------------

def plot_variant_classification_bars(output_dir: str, plots_dir: str):
    """TP/FP/FN/TN counts by variant type, one group of bars per model."""
    try:
        csv_path = os.path.join(output_dir, "variant_classification_summary.csv")
        if not os.path.exists(csv_path):
            return
        df = pd.read_csv(csv_path)
        if df.empty:
            return

        models = df["Model"].unique().tolist()
        var_types = df["VariantType"].unique().tolist()
        statuses = ["TP", "FP", "FN", "TN"]
        status_colors = {"TP": "#4CAF50", "FP": "#F44336", "FN": "#FF9800", "TN": "#2196F3"}

        fig, axes = plt.subplots(1, len(var_types),
                                  figsize=(5 * len(var_types), 5), squeeze=False)
        for ax, vtype in zip(axes[0], var_types):
            sub = df[df["VariantType"] == vtype]
            x = np.arange(len(models))
            bottoms = np.zeros(len(models))
            for status in statuses:
                counts = []
                for m in models:
                    row = sub[(sub["Model"] == m) & (sub["Status"] == status)]
                    counts.append(int(row["Count"].values[0]) if len(row) else 0)
                ax.bar(x, counts, bottom=bottoms, label=status,
                       color=status_colors[status], edgecolor='white', linewidth=0.4)
                bottoms += np.array(counts, dtype=float)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=40, ha='right', fontsize=7)
            ax.set_title(f"Variant type: {vtype}", fontsize=9, fontweight='bold')
            ax.set_ylabel("Count", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(axis='y', alpha=0.25)
            ax.spines[['top', 'right']].set_visible(False)

        fig.suptitle("Variant Classification Counts by Type", fontsize=10,
                     fontweight='bold', y=1.01)
        fig.tight_layout()
        _savefig(fig, os.path.join(plots_dir, "variant_classification_bars.png"))
    except Exception as e:
        logging.warning(f"plot_variant_classification_bars failed: {e}")


# ---------------------------------------------------------------------------
# 11. Apply-mode: score distribution histograms
# ---------------------------------------------------------------------------

def plot_apply_score_distributions(scores_df: pd.DataFrame, plots_dir: str):
    """Overlapping histograms of all model scores on the scored VCF."""
    try:
        score_cols = [c for c in scores_df.columns if c.endswith("_score")]
        if not score_cols:
            return
        model_names = [c.replace("_score", "") for c in score_cols]

        # Combined overlay
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(0, 1, 50)
        for col, mname in zip(score_cols, model_names):
            s = pd.to_numeric(scores_df[col], errors='coerce').dropna()
            if len(s) == 0:
                continue
            ax.hist(s, bins=bins, density=True, alpha=0.45,
                    label=mname, histtype='stepfilled',
                    color=_model_color(mname, model_names))
        ax.set_xlabel("Score", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title("Score Distributions — All Models (Apply Mode)", fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, framealpha=0.85)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        _savefig(fig, os.path.join(plots_dir, "apply_score_distributions.png"))

        # Per-model CDF
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for col, mname in zip(score_cols, model_names):
            s = pd.to_numeric(scores_df[col], errors='coerce').dropna().sort_values()
            if len(s) == 0:
                continue
            cdf = np.arange(1, len(s) + 1) / len(s)
            ax2.plot(s, cdf, lw=1.5, label=mname,
                     color=_model_color(mname, model_names))
        ax2.set_xlabel("Score", fontsize=9)
        ax2.set_ylabel("Cumulative Fraction of Variants", fontsize=9)
        ax2.set_title("Score CDFs — All Models (Apply Mode)", fontsize=10, fontweight='bold')
        ax2.legend(fontsize=7, framealpha=0.85)
        ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
        ax2.grid(alpha=0.2)
        ax2.spines[['top', 'right']].set_visible(False)
        fig2.tight_layout()
        _savefig(fig2, os.path.join(plots_dir, "apply_score_cdfs.png"))
    except Exception as e:
        logging.warning(f"plot_apply_score_distributions failed: {e}")


# ---------------------------------------------------------------------------
# 12. Apply-mode: per-model score percentile comparison (box-and-whisker)
# ---------------------------------------------------------------------------

def plot_apply_score_boxplots(scores_df: pd.DataFrame, plots_dir: str):
    """Side-by-side boxplots of scores for quick sanity check of score spread."""
    try:
        score_cols = [c for c in scores_df.columns if c.endswith("_score")]
        if not score_cols:
            return
        model_names = [c.replace("_score", "") for c in score_cols]
        data = [pd.to_numeric(scores_df[c], errors='coerce').dropna().values for c in score_cols]
        data = [d for d in data if len(d) > 0]
        names = [n for n, d in zip(model_names, data) if len(d) > 0]  # type: ignore
        if not names:
            return

        fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.9), 5))
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color='white', linewidth=2))
        for patch, mname in zip(bp['boxes'], names):
            patch.set_facecolor(_model_color(mname, names))
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, rotation=40, ha='right', fontsize=8)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title("Score Distributions per Model (Apply Mode)", fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        _savefig(fig, os.path.join(plots_dir, "apply_score_boxplots.png"))
    except Exception as e:
        logging.warning(f"plot_apply_score_boxplots failed: {e}")


# ---------------------------------------------------------------------------
# 13. Logistic regression feature coefficients
# ---------------------------------------------------------------------------

def plot_logreg_coefficients(final_models: Dict, feature_keys: List[str], plots_dir: str):
    """Bar chart of LogReg coefficients — shows feature direction and magnitude."""
    try:
        lr = final_models.get('LogReg')
        if lr is None or not hasattr(lr, 'coef_'):
            return
        coef = lr.coef_[0]
        order = np.argsort(np.abs(coef))[::-1]
        feat_s = [feature_keys[i] for i in order]
        coef_s = coef[order]
        colors = ['#F44336' if c < 0 else '#2196F3' for c in coef_s]

        fig, ax = plt.subplots(figsize=(7, max(4, len(feat_s) * 0.45)))
        y_pos = np.arange(len(feat_s))
        ax.barh(y_pos, coef_s, color=colors, edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_s, fontsize=8)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel("Coefficient Value", fontsize=9)
        ax.set_title("Logistic Regression Coefficients\n(blue = positive, red = negative)",
                     fontsize=9, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        _savefig(fig, os.path.join(plots_dir, "logreg_coefficients.png"))
    except Exception as e:
        logging.warning(f"plot_logreg_coefficients failed: {e}")


# ---------------------------------------------------------------------------
# Master dispatcher for train-mode plots
# ---------------------------------------------------------------------------

def generate_all_train_plots(
    per_fold_df: pd.DataFrame,
    y_true: np.ndarray,
    oof_scores: Dict[str, np.ndarray],
    final_models: Dict,
    feature_keys: List[str],
    pairwise_stats: List[Dict],
    output_dir: str,
):
    """Call every train-mode plot function. All failures are caught internally."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    logging.info("Generating visualisations...")

    plot_cv_metric_bars(per_fold_df, plots_dir)
    plot_cv_fold_heatmap(per_fold_df, plots_dir)
    plot_oof_roc_and_pr(y_true, oof_scores, plots_dir)
    plot_score_distributions(y_true, oof_scores, plots_dir)
    plot_calibration_curves(y_true, oof_scores, plots_dir)
    plot_feature_importance(final_models, feature_keys, plots_dir)
    plot_hybrid_alpha_curves(final_models, plots_dir)
    plot_pairwise_delta_heatmap(pairwise_stats, plots_dir, metric="ROC_AUC")
    plot_pairwise_delta_heatmap(pairwise_stats, plots_dir, metric="PR_AUC")
    plot_variant_classification_bars(output_dir, plots_dir)
    plot_logreg_coefficients(final_models, feature_keys, plots_dir)
    logging.info(f"Visualisations saved to {plots_dir}")


# ============================================================================
# PIPELINE ORCHESTRATION
# ============================================================================


