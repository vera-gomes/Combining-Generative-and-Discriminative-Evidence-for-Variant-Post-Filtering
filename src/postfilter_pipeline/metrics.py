"""Threshold-finding and basic classification metric helpers."""

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score, roc_curve,
)


def find_threshold_at_sensitivity(y_true, scores, target_sens: float) -> float:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    pos_scores = np.sort(scores[y_true == 1])[::-1]
    if len(pos_scores) == 0:
        return np.nan
    k = max(0, min(int(np.ceil(target_sens * len(pos_scores))) - 1, len(pos_scores) - 1))
    return float(pos_scores[k])


def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising Youden's J (TPR - FPR)."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    fpr, tpr, thr = roc_curve(y_true, scores)
    j = tpr - fpr
    if len(j) == 0:
        return np.nan
    return float(thr[int(np.nanargmax(j))])


def compute_threshold_metrics(y_true: np.ndarray, scores: np.ndarray, thr: float) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(scores).astype(float) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    f1_ = f1_score(y_true, pred, zero_division=0)
    acc = accuracy_score(y_true, pred)
    spec = float(tn / (tn + fp)) if (tn + fp) else np.nan
    npv = float(tn / (tn + fn)) if (tn + fn) else np.nan
    fpr_r = float(fp / (fp + tn)) if (fp + tn) else np.nan
    fnr = float(fn / (fn + tp)) if (fn + tp) else np.nan
    bal_acc = balanced_accuracy_score(y_true, pred)
    mcc = matthews_corrcoef(y_true, pred) if (tp + fp) > 0 and (tp + fn) > 0 and (tn + fp) > 0 and (tn + fn) > 0 else np.nan
    return {
        "Threshold": float(thr),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Precision": float(prec), "Recall": float(rec), "F1": float(f1_),
        "Accuracy": float(acc),
        "Specificity": spec, "NPV": npv, "FPR": fpr_r, "FNR": fnr,
        "BalancedAccuracy": float(bal_acc),
        "MCC": float(mcc) if not np.isnan(mcc) else np.nan,
    }


def compute_basic_model_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    preds = (scores >= threshold).astype(int)
    return {
        'auc': float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else np.nan,
        'precision': float(precision_score(y_true, preds, zero_division=0)),
        'recall': float(recall_score(y_true, preds, zero_division=0)),
        'f1': float(f1_score(y_true, preds, zero_division=0)),
        'accuracy': float(accuracy_score(y_true, preds)),
    }


def _nan_metric_dict() -> Dict[str, float]:
    return {'auc': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'accuracy': np.nan}


def _metric_dict_for_model(model_name: str, y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    # All models now produce (0,1) scores; uniform threshold of 0.5 is fair.
    return compute_basic_model_metrics(y_true, scores, threshold=0.5)


def precision_at_sensitivity(y_true, scores, target_sens: float) -> float:
    thr = find_threshold_at_sensitivity(y_true, scores, target_sens)
    if np.isnan(thr):
        return np.nan
    y_pred = (np.asarray(scores) >= thr).astype(int)
    return precision_score(np.asarray(y_true).astype(int), y_pred, zero_division=0)


def top_fraction_enrichment(y_true: np.ndarray, scores: np.ndarray, frac: float = 0.01) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    n = len(scores)
    if n == 0:
        return {"N_Top": 0, "PosRate_TopFrac": np.nan, "PosRate_Overall": np.nan, "FoldEnrichment": np.nan}
    k = max(1, int(np.floor(frac * n)))
    top = np.argsort(scores)[::-1][:k]
    pos_rate_top = float(y_true[top].mean())
    pos_rate_all = float(y_true.mean())
    fold = (pos_rate_top / pos_rate_all) if pos_rate_all > 0 else np.nan
    return {"N_Top": k, "PosRate_TopFrac": pos_rate_top, "PosRate_Overall": pos_rate_all,
            "FoldEnrichment": fold}


