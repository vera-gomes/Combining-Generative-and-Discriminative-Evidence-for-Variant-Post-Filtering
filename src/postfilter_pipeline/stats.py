"""Statistical testing utilities: multiple-testing correction, DeLong test,
McNemar tests, paired/Bayesian bootstrap deltas, and the ClinVar/PopDB
threshold-and-bootstrap report generator."""

import logging
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score

from .config import PipelineConfig, _resolve_pairwise_comparisons
from .metrics import compute_threshold_metrics, youden_threshold


def holm_bonferroni(pvals: list) -> list:
    pvals = np.asarray(pvals, dtype=float)
    n = int(np.sum(~np.isnan(pvals)))
    adj = np.full_like(pvals, np.nan, dtype=float)
    if n == 0:
        return adj.tolist()
    idx = np.where(~np.isnan(pvals))[0]
    p = pvals[idx]
    order = np.argsort(p)
    p_sorted = p[order]
    adj_sorted = np.zeros(len(p_sorted))
    for i, pv in enumerate(p_sorted):
        adj_sorted[i] = min(1.0, (n - i) * pv)
    for i in range(1, len(adj_sorted)):
        adj_sorted[i] = max(adj_sorted[i], adj_sorted[i - 1])
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(len(order))
    adj[idx] = adj_sorted[inv_order]
    return adj.tolist()


def benjamini_hochberg(pvals: list) -> list:
    pvals = np.asarray(pvals, dtype=float)
    q = np.full_like(pvals, np.nan, dtype=float)
    idx = np.where(~np.isnan(pvals))[0]
    if len(idx) == 0:
        return q.tolist()
    p = pvals[idx]
    order = np.argsort(p)
    p_sorted = p[order]
    m = len(p_sorted)
    q_sorted = np.array([(m / (i + 1)) * pv for i, pv in enumerate(p_sorted)])
    for i in range(m - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(m)
    q[idx] = q_sorted[inv_order]
    return q.tolist()


def paired_bootstrap_delta(y_true, scores_A, scores_B, metric_fn, n_boot: int = 1000, seed: int = 123):
    """Paired bootstrap CI for delta = metric(A) - metric(B)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    sA = np.asarray(scores_A, dtype=float)
    sB = np.asarray(scores_B, dtype=float)
    n = len(y_true)
    idx_all = np.arange(n)
    try:
        point_A = metric_fn(y_true, sA)
        point_B = metric_fn(y_true, sB)
        point_delta = point_A - point_B
    except Exception:
        point_A = point_B = point_delta = np.nan

    deltas = []
    for _ in range(n_boot):
        samp = rng.choice(idx_all, size=n, replace=True)
        try:
            da = metric_fn(y_true[samp], sA[samp])
            db = metric_fn(y_true[samp], sB[samp])
            deltas.append(da - db)
        except Exception:
            deltas.append(np.nan)

    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[~np.isnan(deltas)]
    if len(deltas) == 0:
        return {"A": point_A, "B": point_B, "Delta_point": point_delta,
                "Delta_boot_mean": np.nan, "CI_low": np.nan, "CI_high": np.nan,
                "p_one_sided_A_gt_B": np.nan, "p_two_sided": np.nan}

    ci_low = float(np.quantile(deltas, 0.025))
    ci_high = float(np.quantile(deltas, 0.975))
    boot_mean = float(deltas.mean())
    p_left = float(np.mean(deltas <= 0.0))
    p_right = float(np.mean(deltas >= 0.0))
    p_two = float(min(1.0, 2.0 * min(p_left, p_right)))
    return {
        "A": float(point_A) if point_A is not None else np.nan,
        "B": float(point_B) if point_B is not None else np.nan,
        "Delta_point": float(point_delta) if point_delta is not None else np.nan,
        "Delta_boot_mean": boot_mean,
        "CI_low": ci_low, "CI_high": ci_high,
        "p_one_sided_A_gt_B": p_left,   # P(delta <= 0) == one-sided test A > B
        "p_two_sided": p_two,
    }


# ============================================================================
# DELONG TEST
# ============================================================================


def _compute_midrank(x):
    x = np.asarray(x)
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        mid = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = mid
        i = j
    return ranks


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]
    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]
    tx = np.zeros((k, m))
    ty = np.zeros((k, n))
    tz = np.zeros((k, m + n))
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    s = sx / m + sy / n
    return aucs, s


def delong_roc_test(y_true, pred1, pred2) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    pred1 = np.asarray(pred1, dtype=float)
    pred2 = np.asarray(pred2, dtype=float)
    order = np.argsort(-y_true)
    y_sorted = y_true[order]
    preds = np.vstack([pred1[order], pred2[order]])
    m = int(y_sorted.sum())
    nan_result = {"auc1": np.nan, "auc2": np.nan, "delta": np.nan,
                  "z": np.nan, "se": np.nan, "var": np.nan, "p_value": np.nan,
                  "p_one_sided_auc1_lt_auc2": np.nan, "p_one_sided_auc1_gt_auc2": np.nan}
    if m == 0 or m == len(y_sorted):
        return nan_result
    aucs, cov = _fast_delong(preds, m)
    delta = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    if var <= 0 or not np.isfinite(var):
        return {**nan_result, "auc1": float(aucs[0]), "auc2": float(aucs[1]),
                "delta": delta, "var": var}
    se = float(np.sqrt(var))
    z = float(delta / se)
    try:
        from scipy.stats import norm
        tiny = np.finfo(float).tiny
        p_two = max(float(2.0 * norm.sf(abs(z))), tiny)
        p_lt = max(float(norm.cdf(z)), tiny)
        p_gt = max(float(norm.sf(z)), tiny)
    except Exception:
        import math
        p_two = max(float(math.erfc(abs(z) / math.sqrt(2.0))), np.finfo(float).tiny)
        p_lt = p_gt = np.nan
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "delta": delta,
            "z": z, "se": se, "var": var, "p_value": p_two,
            "p_one_sided_auc1_lt_auc2": p_lt, "p_one_sided_auc1_gt_auc2": p_gt}


# ============================================================================
# MCNEMAR / BOOTSTRAP TESTS
# ============================================================================


def _safe_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -60.0, 60.0)))


def _as_proba_for_logloss(scores: np.ndarray, mode: str = "auto") -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    if mode == "sigmoid":
        p = _safe_sigmoid(s)
    elif mode == "clip":
        p = s
    else:
        p = s if (np.nanmin(s) >= 0.0 and np.nanmax(s) <= 1.0) else _safe_sigmoid(s)
    return np.clip(p, 1e-15, 1.0 - 1e-15)


def _percentile_ci(samples: np.ndarray, alpha: float = 0.05):
    return float(np.nanpercentile(samples, 100.0 * alpha / 2.0)), float(np.nanpercentile(samples, 100.0 * (1.0 - alpha / 2.0)))


def mcnemar_test_pvalue(y_true, predA, predB) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(predA, dtype=int)
    b = np.asarray(predB, dtype=int)
    b_cnt = int(np.sum((a == y) & (b != y)))
    c_cnt = int(np.sum((a != y) & (b == y)))
    n = b_cnt + c_cnt
    if n == 0:
        return {"b": b_cnt, "c": c_cnt, "n": 0, "p_value": np.nan}
    try:
        from scipy.stats import binom
        p = min(float(2.0 * binom.cdf(min(b_cnt, c_cnt), n, 0.5)), 1.0)
    except Exception:
        chi2 = ((abs(b_cnt - c_cnt) - 1.0) ** 2) / max(n, 1)
        try:
            from scipy.stats import chi2 as chi2dist
            p = float(chi2dist.sf(chi2, df=1))
        except Exception:
            p = np.nan
    return {"b": b_cnt, "c": c_cnt, "n": n, "p_value": p}


def bayesian_mcnemar(y_true, predA, predB, alpha: float = 1.0, beta: float = 1.0) -> Dict[str, Any]:
    res = mcnemar_test_pvalue(y_true, predA, predB)
    b_cnt, c_cnt, n = int(res["b"]), int(res["c"]), int(res["n"])
    if n == 0:
        return {"b": b_cnt, "c": c_cnt, "n": 0, "post_alpha": np.nan, "post_beta": np.nan,
                "P_A_better": np.nan, "p_ci_low": np.nan, "p_ci_high": np.nan}
    a_post = alpha + b_cnt
    b_post = beta + c_cnt
    try:
        from scipy.stats import beta as betadist
        P_A_better = float(betadist.sf(0.5, a_post, b_post))
        ci_low = float(betadist.ppf(0.025, a_post, b_post))
        ci_high = float(betadist.ppf(0.975, a_post, b_post))
    except Exception:
        P_A_better = ci_low = ci_high = np.nan
    return {"b": b_cnt, "c": c_cnt, "n": n, "post_alpha": float(a_post), "post_beta": float(b_post),
            "P_A_better": P_A_better, "p_ci_low": ci_low, "p_ci_high": ci_high}


def _subsample_balanced(y, *arrays, max_n: int = 50000, seed: int = 0):
    """Stratified subsample of (y, *arrays) to at most max_n samples."""
    rng = np.random.default_rng(seed)
    if len(y) <= max_n:
        return (y,) + arrays
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    target_pos = min(n_pos, int(round(max_n * n_pos / (n_pos + n_neg))))
    target_neg = min(n_neg, max_n - target_pos)
    sel = np.concatenate([rng.choice(pos_idx, target_pos, replace=False),
                          rng.choice(neg_idx, target_neg, replace=False)])
    rng.shuffle(sel)
    logging.info(f"Bootstrap subsampling: {len(y)} -> {len(sel)} variants")
    return (y[sel],) + tuple(a[sel] for a in arrays)


def paired_bootstrap_deltas(y, scoresA, scoresB, threshold, n_boot=2000, seed=13,
                             logloss_proba_mode="auto", max_n=50000) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    y, scoresA, scoresB = _subsample_balanced(
        np.asarray(y, dtype=int), np.asarray(scoresA, dtype=float), np.asarray(scoresB, dtype=float),
        max_n=max_n, seed=seed,
    )
    n = len(y)
    deltas: Dict[str, list] = {"ROC_AUC": [], "AP": [], "F1": [], "LogLoss": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, a, b = y[idx], scoresA[idx], scoresB[idx]
        if len(np.unique(yb)) < 2:
            for k in deltas:
                deltas[k].append(np.nan)
            continue
        try:
            deltas["ROC_AUC"].append(float(roc_auc_score(yb, a) - roc_auc_score(yb, b)))
        except Exception:
            deltas["ROC_AUC"].append(np.nan)
        try:
            deltas["AP"].append(float(average_precision_score(yb, a) - average_precision_score(yb, b)))
        except Exception:
            deltas["AP"].append(np.nan)
        pA = _as_proba_for_logloss(a, logloss_proba_mode)
        pB = _as_proba_for_logloss(b, logloss_proba_mode)
        try:
            deltas["F1"].append(float(f1_score(yb, (pA >= threshold).astype(int)) -
                                      f1_score(yb, (pB >= threshold).astype(int))))
        except Exception:
            deltas["F1"].append(np.nan)
        try:
            deltas["LogLoss"].append(float(log_loss(yb, pA, labels=[0, 1]) -
                                           log_loss(yb, pB, labels=[0, 1])))
        except Exception:
            deltas["LogLoss"].append(np.nan)

    out = {}
    for k, arr in deltas.items():
        arr = np.asarray(arr, dtype=float)
        lo, hi = _percentile_ci(arr)
        out[k] = {"delta_mean": float(np.nanmean(arr)), "delta_median": float(np.nanmedian(arr)),
                  "ci_low": lo, "ci_high": hi,
                  "P_delta_gt_0": float(np.nanmean(arr > 0.0)),
                  "P_delta_lt_0": float(np.nanmean(arr < 0.0))}
    return out


def bayesian_bootstrap_deltas(y, scoresA, scoresB, threshold, n_draws=4000, seed=13,
                               logloss_proba_mode="auto", max_n=50000) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    y, scoresA, scoresB = _subsample_balanced(
        np.asarray(y, dtype=int), np.asarray(scoresA, dtype=float), np.asarray(scoresB, dtype=float),
        max_n=max_n, seed=seed,
    )
    n = len(y)
    deltas: Dict[str, list] = {"ROC_AUC": [], "AP": [], "F1": [], "LogLoss": []}
    for _ in range(n_draws):
        w = rng.dirichlet(np.ones(n, dtype=float))
        if len(np.unique(y)) < 2:
            for k in deltas:
                deltas[k].append(np.nan)
            continue
        try:
            deltas["ROC_AUC"].append(float(roc_auc_score(y, scoresA, sample_weight=w) -
                                           roc_auc_score(y, scoresB, sample_weight=w)))
        except Exception:
            deltas["ROC_AUC"].append(np.nan)
        try:
            deltas["AP"].append(float(average_precision_score(y, scoresA, sample_weight=w) -
                                      average_precision_score(y, scoresB, sample_weight=w)))
        except Exception:
            deltas["AP"].append(np.nan)
        pA = _as_proba_for_logloss(scoresA, logloss_proba_mode)
        pB = _as_proba_for_logloss(scoresB, logloss_proba_mode)
        try:
            deltas["F1"].append(float(f1_score(y, (pA >= threshold).astype(int), sample_weight=w) -
                                      f1_score(y, (pB >= threshold).astype(int), sample_weight=w)))
        except Exception:
            deltas["F1"].append(np.nan)
        try:
            deltas["LogLoss"].append(float(log_loss(y, pA, labels=[0, 1], sample_weight=w) -
                                           log_loss(y, pB, labels=[0, 1], sample_weight=w)))
        except Exception:
            deltas["LogLoss"].append(np.nan)

    out = {}
    for k, arr in deltas.items():
        arr = np.asarray(arr, dtype=float)
        lo, hi = _percentile_ci(arr)
        out[k] = {"delta_mean": float(np.nanmean(arr)), "delta_median": float(np.nanmedian(arr)),
                  "ci_low": lo, "ci_high": hi, "P_delta_gt_0": float(np.nanmean(arr > 0.0))}
    return out


def model_best_probabilities(y, score_dict, threshold, metric="ROC_AUC",
                              n_draws=3000, seed=13, bayesian=False,
                              logloss_proba_mode="auto") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    n = len(y)
    models = list(score_dict.keys())
    best_counts = {m: 0 for m in models}
    valid = 0
    for _ in range(n_draws):
        if bayesian:
            w = rng.dirichlet(np.ones(n, dtype=float))
            idx = None
        else:
            idx = rng.integers(0, n, size=n)
            w = None
        vals = {}
        for m in models:
            s = np.asarray(score_dict[m], dtype=float)
            try:
                if idx is not None:
                    yy, ss = y[idx], s[idx]
                    if len(np.unique(yy)) < 2:
                        vals[m] = np.nan
                        continue
                    kw = {}
                else:
                    yy, ss, kw = y, s, {"sample_weight": w}
                if metric == "ROC_AUC":
                    vals[m] = roc_auc_score(yy, ss, **kw)
                elif metric == "AP":
                    vals[m] = average_precision_score(yy, ss, **kw)
                elif metric == "F1":
                    pp = _as_proba_for_logloss(ss, logloss_proba_mode)
                    vals[m] = f1_score(yy, (pp >= threshold).astype(int), **kw)
                elif metric == "LogLoss":
                    pp = _as_proba_for_logloss(ss, logloss_proba_mode)
                    vals[m] = log_loss(yy, pp, labels=[0, 1], **kw)
                else:
                    vals[m] = np.nan
            except Exception:
                vals[m] = np.nan
        if all(not np.isfinite(v) for v in vals.values()):
            continue
        valid += 1
        if metric == "LogLoss":
            best = min(vals, key=lambda k: np.inf if not np.isfinite(vals[k]) else vals[k])
        else:
            best = max(vals, key=lambda k: -np.inf if not np.isfinite(vals[k]) else vals[k])
        best_counts[best] += 1
    return pd.DataFrame([{"Model": m, "Metric": metric, "Bayesian": bool(bayesian),
                           "P_best": (best_counts[m] / valid) if valid else np.nan}
                          for m in models])


def bootstrap_ci_metrics(y_true, scores, n_boot=500, seed=42, max_n=200000) -> Dict:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    y_true, scores = _subsample_balanced(y_true, scores, max_n=max_n, seed=seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    _nan = {k: np.nan for k in [
        "N_used_for_bootstrap",
        "ROC_AUC_mean", "ROC_AUC_ci_low", "ROC_AUC_ci_high",
        "PR_AUC_mean", "PR_AUC_ci_low", "PR_AUC_ci_high",
        "MCC_mean", "MCC_ci_low", "MCC_ci_high",
        "BalancedAccuracy_mean", "BalancedAccuracy_ci_low", "BalancedAccuracy_ci_high",
        "Specificity_mean", "Specificity_ci_low", "Specificity_ci_high",
        "Precision_mean", "Precision_ci_low", "Precision_ci_high",
        "Recall_mean", "Recall_ci_low", "Recall_ci_high",
        "F1_mean", "F1_ci_low", "F1_ci_high",
    ]}
    _nan["N_used_for_bootstrap"] = len(y_true)
    if len(pos_idx) < 2 or len(neg_idx) < 2:
        return _nan

    roc_vals, pr_vals, mcc_vals, bal_vals = [], [], [], []
    spec_vals, prec_vals, rec_vals, f1_vals = [], [], [], []
    for _ in range(n_boot):
        samp = np.concatenate([rng.choice(pos_idx, len(pos_idx), replace=True),
                               rng.choice(neg_idx, len(neg_idx), replace=True)])
        yb, sb = y_true[samp], scores[samp]
        try:
            roc_vals.append(roc_auc_score(yb, sb))
        except Exception:
            roc_vals.append(np.nan)
        try:
            pr_vals.append(average_precision_score(yb, sb))
        except Exception:
            pr_vals.append(np.nan)
        try:
            thr = youden_threshold(yb, sb)
            tm = compute_threshold_metrics(yb, sb, thr)
            mcc_vals.append(tm["MCC"])
            bal_vals.append(tm["BalancedAccuracy"])
            spec_vals.append(tm["Specificity"])
            prec_vals.append(tm["Precision"])
            rec_vals.append(tm["Recall"])
            f1_vals.append(tm["F1"])
        except Exception:
            for lst in (mcc_vals, bal_vals, spec_vals, prec_vals, rec_vals, f1_vals):
                lst.append(np.nan)

    def _summ(x):
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return np.nan, np.nan, np.nan
        return float(x.mean()), float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))

    roc_m, roc_lo, roc_hi = _summ(roc_vals)
    pr_m, pr_lo, pr_hi = _summ(pr_vals)
    mcc_m, mcc_lo, mcc_hi = _summ(mcc_vals)
    bal_m, bal_lo, bal_hi = _summ(bal_vals)
    spec_m, spec_lo, spec_hi = _summ(spec_vals)
    prec_m, prec_lo, prec_hi = _summ(prec_vals)
    rec_m, rec_lo, rec_hi = _summ(rec_vals)
    f1_m, f1_lo, f1_hi = _summ(f1_vals)
    return {
        "N_used_for_bootstrap": int(len(y_true)),
        "ROC_AUC_mean": roc_m, "ROC_AUC_ci_low": roc_lo, "ROC_AUC_ci_high": roc_hi,
        "PR_AUC_mean": pr_m, "PR_AUC_ci_low": pr_lo, "PR_AUC_ci_high": pr_hi,
        "MCC_mean": mcc_m, "MCC_ci_low": mcc_lo, "MCC_ci_high": mcc_hi,
        "BalancedAccuracy_mean": bal_m, "BalancedAccuracy_ci_low": bal_lo, "BalancedAccuracy_ci_high": bal_hi,
        "Specificity_mean": spec_m, "Specificity_ci_low": spec_lo, "Specificity_ci_high": spec_hi,
        "Precision_mean": prec_m, "Precision_ci_low": prec_lo, "Precision_ci_high": prec_hi,
        "Recall_mean": rec_m, "Recall_ci_low": rec_lo, "Recall_ci_high": rec_hi,
        "F1_mean": f1_m, "F1_ci_low": f1_lo, "F1_ci_high": f1_hi,
    }


# ============================================================================
# THRESHOLD-AND-BOOTSTRAP REPORT (ClinVar / PopDB scenarios)
# ============================================================================


def run_threshold_and_bootstrap_reports(df_eval, y_col, score_cols, output_dir,
                                        scenario, config: PipelineConfig) -> None:
    os.makedirs(output_dir, exist_ok=True)
    threshold = float(config._raw_get("statistics.prob_threshold",
                                      config._raw_get("statistics.decision_threshold", 0.5)))
    n_boot = int(config._raw_get("statistics.pairwise_bootstrap_n", 1000))
    seed = int(config._raw_get("statistics.bootstrap.seed",
                               config._raw_get("statistics.pairwise_bootstrap_seed", 123)))
    n_draws = int(config._raw_get("statistics.bayes_bootstrap.n_draws", 4000))
    logloss_mode = str(config._raw_get("statistics.logloss_proba_mode", "auto"))
    max_n = int(config._raw_get('evaluation.apply_stats_max_n', 50000))
    # Minimum minority-class count below which bootstrap is skipped entirely.
    # With e.g. 170 ClinVar-positive variants the bootstrap produces noise and
    # would run for days with no scientific value.
    min_class_n = int(config._raw_get('evaluation.min_class_n_for_bootstrap', 500))
    comparisons = _resolve_pairwise_comparisons(config, score_cols)

    y_all = pd.to_numeric(df_eval[y_col], errors="coerce").to_numpy(dtype=int)
    n_pos = int((y_all == 1).sum())
    n_neg = int((y_all == 0).sum())
    minority_n = min(n_pos, n_neg)

    # ── Skip bootstrap entirely when positive or negative class is too small ──
    if minority_n < min_class_n:
        logging.warning(
            f"[{scenario}] Skipping bootstrap/McNemar reports: minority class has only "
            f"{minority_n} samples (threshold={min_class_n}). "
            f"Increase 'evaluation.min_class_n_for_bootstrap' in your config to override. "
            f"Pos={n_pos}, Neg={n_neg}."
        )
        # Still write empty placeholder CSVs so downstream code doesn't crash on missing files.
        for suffix in ["pairwise_mcnemar", "pairwise_bootstrap_deltas",
                       "pairwise_bayes_mcnemar", "pairwise_bayesboot_deltas"]:
            pd.DataFrame().to_csv(
                os.path.join(output_dir, f"{scenario.lower()}_{suffix}.csv"), index=False)
        return

    mcnemar_rows, boot_rows, bayes_mcnemar_rows, bayesboot_rows = [], [], [], []

    for model_A, model_B in comparisons:
        col_A, col_B = f"{model_A}_score", f"{model_B}_score"
        if col_A not in df_eval.columns or col_B not in df_eval.columns:
            continue
        sA = pd.to_numeric(df_eval[col_A], errors="coerce").to_numpy(dtype=float)
        sB = pd.to_numeric(df_eval[col_B], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(sA) & np.isfinite(sB)
        if mask.sum() == 0:
            continue
        y, a, b = y_all[mask], sA[mask], sB[mask]
        pA_dec = _as_proba_for_logloss(a, logloss_mode)
        pB_dec = _as_proba_for_logloss(b, logloss_mode)
        predA = (pA_dec >= threshold).astype(int)
        predB = (pB_dec >= threshold).astype(int)
        mc = mcnemar_test_pvalue(y, predA, predB)
        mcnemar_rows.append({"Scenario": scenario, "Comparison": f"{model_A}_vs_{model_B}",
                              "Threshold": threshold, "b_A_correct_B_wrong": mc["b"],
                              "c_A_wrong_B_correct": mc["c"], "n_disagreements": mc["n"],
                              "McNemar_p_two_sided": mc["p_value"]})
        bmc = bayesian_mcnemar(y, predA, predB)
        bayes_mcnemar_rows.append({"Scenario": scenario, "Comparison": f"{model_A}_vs_{model_B}",
                                   "Threshold": threshold, "b_A_correct_B_wrong": bmc["b"],
                                   "c_A_wrong_B_correct": bmc["c"], "n_disagreements": bmc["n"],
                                   "Posterior_alpha": bmc["post_alpha"], "Posterior_beta": bmc["post_beta"],
                                   "P_A_better_given_disagreement": bmc["P_A_better"],
                                   "p_win_CI95_low": bmc["p_ci_low"], "p_win_CI95_high": bmc["p_ci_high"]})
        bd = paired_bootstrap_deltas(y, a, b, threshold=threshold, n_boot=n_boot,
                                     seed=seed, logloss_proba_mode=logloss_mode, max_n=max_n)
        for metric, st in bd.items():
            boot_rows.append({"Scenario": scenario, "Comparison": f"{model_A}_vs_{model_B}",
                               "Metric": metric, "Delta_A_minus_B_mean": st["delta_mean"],
                               "Delta_A_minus_B_median": st["delta_median"],
                               "CI95_low": st["ci_low"], "CI95_high": st["ci_high"],
                               "P_delta_gt_0": st["P_delta_gt_0"], "P_delta_lt_0": st["P_delta_lt_0"],
                               "n_boot": n_boot, "Threshold": threshold})
        bbd = bayesian_bootstrap_deltas(y, a, b, threshold=threshold, n_draws=n_draws,
                                        seed=seed, logloss_proba_mode=logloss_mode, max_n=max_n)
        for metric, st in bbd.items():
            bayesboot_rows.append({"Scenario": scenario, "Comparison": f"{model_A}_vs_{model_B}",
                                   "Metric": metric, "Delta_A_minus_B_mean": st["delta_mean"],
                                   "Delta_A_minus_B_median": st["delta_median"],
                                   "CI95_low": st["ci_low"], "CI95_high": st["ci_high"],
                                   "P_delta_gt_0": st["P_delta_gt_0"], "n_draws": n_draws,
                                   "Threshold": threshold})

    if mcnemar_rows:
        df_mc = pd.DataFrame(mcnemar_rows)
        df_mc["McNemar_p_two_sided_Holm"] = holm_bonferroni(df_mc["McNemar_p_two_sided"].tolist())
        df_mc.to_csv(os.path.join(output_dir, f"{scenario.lower()}_pairwise_mcnemar.csv"), index=False)
    if boot_rows:
        pd.DataFrame(boot_rows).to_csv(
            os.path.join(output_dir, f"{scenario.lower()}_pairwise_bootstrap_deltas.csv"), index=False)
    if bayes_mcnemar_rows:
        pd.DataFrame(bayes_mcnemar_rows).to_csv(
            os.path.join(output_dir, f"{scenario.lower()}_pairwise_bayes_mcnemar.csv"), index=False)
    if bayesboot_rows:
        pd.DataFrame(bayesboot_rows).to_csv(
            os.path.join(output_dir, f"{scenario.lower()}_pairwise_bayesboot_deltas.csv"), index=False)

    score_dict = {c.replace("_score", ""): pd.to_numeric(df_eval[c], errors="coerce").to_numpy(dtype=float)
                  for c in score_cols}
    mat = np.vstack(list(score_dict.values()))
    mask_all = np.all(np.isfinite(mat), axis=0)
    if mask_all.sum() > 0:
        y_rank = y_all[mask_all]
        score_dict_rank = {m: score_dict[m][mask_all] for m in score_dict}
        for metric in ["ROC_AUC", "AP", "F1", "LogLoss"]:
            model_best_probabilities(y_rank, score_dict_rank, threshold=threshold, metric=metric,
                                     n_draws=max(1000, n_boot), seed=seed, bayesian=False,
                                     logloss_proba_mode=logloss_mode).to_csv(
                os.path.join(output_dir, f"{scenario.lower()}_prob_best_bootstrap_{metric}.csv"), index=False)
            model_best_probabilities(y_rank, score_dict_rank, threshold=threshold, metric=metric,
                                     n_draws=max(1000, n_draws), seed=seed, bayesian=True,
                                     logloss_proba_mode=logloss_mode).to_csv(
                os.path.join(output_dir, f"{scenario.lower()}_prob_best_bayesboot_{metric}.csv"), index=False)


# ============================================================================
# MEMORY MONITORING
# ============================================================================


