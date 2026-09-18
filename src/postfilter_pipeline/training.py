"""LightGBM hyperparameter tuning, hybrid alpha selection, cross-validation
evaluation, final-model training, model persistence and post-training
analysis/reporting (tranche analysis, variant classification, paired tests)."""

import gc
import json
import logging
import os
import time
from typing import Any, Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, log_loss, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.model_selection import StratifiedKFold, train_test_split

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    logging.warning("LightGBM not available. LGBM models will be skipped.")

try:
    import optuna
    OPTUNA_AVAILABLE = True
    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception:
        pass
except ImportError:
    OPTUNA_AVAILABLE = False
    # Warning is intentionally NOT emitted here — in apply mode no training occurs
    # so the absence of Optuna is irrelevant.  The warning is logged inside
    # optimize_lightgbm_params() the first time it is actually called.

try:
    from skopt import BayesSearchCV as _BayesSearchCV
    from skopt.space import Integer as _SkInteger
    from skopt.space import Real as _SkReal
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

from .config import PipelineConfig
from .data_prep import make_block_splitter
from .memory_utils import _check_memory_skip, monitor_memory
from .metrics import (
    _metric_dict_for_model, _nan_metric_dict, find_threshold_at_sensitivity,
)
from .models import (
    FIXED_ALPHA_HYBRIDS, HYBRID_BASE_MODELS, LeakageAwareImputer, LeakageAwareScaler,
    compute_hybrid_scores, compute_vqslod_prob, get_hybrid_name,
    is_hybrid_model_name, score_model_instance,
)


def get_lightgbm_base_params(config: PipelineConfig, random_state: int) -> Dict[str, Any]:
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'n_estimators': int(config._raw_get('models.lightgbm.n_estimators', 100)),
        'learning_rate': float(config._raw_get('models.lightgbm.learning_rate', 0.05)),
        'max_depth': int(config._raw_get('models.lightgbm.max_depth', 7)),
        'num_leaves': int(config._raw_get('models.lightgbm.num_leaves', 31)),
        'min_child_samples': int(config._raw_get('models.lightgbm.min_child_samples', 20)),
        'subsample': float(config._raw_get('models.lightgbm.subsample', 0.8)),
        'colsample_bytree': float(config._raw_get('models.lightgbm.colsample_bytree', 0.8)),
        'reg_alpha': float(config._raw_get('models.lightgbm.reg_alpha', 0.0)),
        'reg_lambda': float(config._raw_get('models.lightgbm.reg_lambda', 0.0)),
        'n_jobs': 2 if config._raw_get('memory_safety.lightgbm_memory_safe', True) else -1,
        'random_state': int(random_state),
        'verbose': -1,
    }
    if config._raw_get('memory_safety.lightgbm_memory_safe', True):
        params['force_row_wise'] = True
    return params


def _truncate_num_leaves(params: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(params)
    md = int(params.get('max_depth', -1))
    if md > 0:
        params['num_leaves'] = int(min(int(params.get('num_leaves', 31)), max(2, 2 ** md)))
    return params


def _score_probability_metric(y_true, y_prob, metric_name='ROC_AUC') -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    mn = str(metric_name).upper()
    if mn in {'ROC_AUC', 'AUC'}:
        return float(roc_auc_score(y_true, y_prob))
    if mn in {'PR_AUC', 'AP', 'PRAUC'}:
        return float(average_precision_score(y_true, y_prob))
    if mn == 'LOGLOSS':
        return float(-log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15), labels=[0, 1]))
    raise ValueError(f"Unsupported LightGBM tuning metric: {metric_name}")


def _suggest_lightgbm_params(trial, config: PipelineConfig, random_state: int) -> Dict[str, Any]:
    params = get_lightgbm_base_params(config, random_state)
    params.update({
        'n_estimators': trial.suggest_int('n_estimators',
            int(config._raw_get('models.lightgbm_optuna.n_estimators_min', 50)),
            int(config._raw_get('models.lightgbm_optuna.n_estimators_max', 300))),
        'learning_rate': trial.suggest_float('learning_rate',
            float(config._raw_get('models.lightgbm_optuna.learning_rate_min', 1e-2)),
            float(config._raw_get('models.lightgbm_optuna.learning_rate_max', 2e-1)), log=True),
        'max_depth': trial.suggest_int('max_depth',
            int(config._raw_get('models.lightgbm_optuna.max_depth_min', 3)),
            int(config._raw_get('models.lightgbm_optuna.max_depth_max', 12))),
        'num_leaves': trial.suggest_int('num_leaves',
            int(config._raw_get('models.lightgbm_optuna.num_leaves_min', 16)),
            int(config._raw_get('models.lightgbm_optuna.num_leaves_max', 255))),
        'min_child_samples': trial.suggest_int('min_child_samples',
            int(config._raw_get('models.lightgbm_optuna.min_child_samples_min', 5)),
            int(config._raw_get('models.lightgbm_optuna.min_child_samples_max', 100))),
        'subsample': trial.suggest_float('subsample',
            float(config._raw_get('models.lightgbm_optuna.subsample_min', 0.5)),
            float(config._raw_get('models.lightgbm_optuna.subsample_max', 1.0))),
        'colsample_bytree': trial.suggest_float('colsample_bytree',
            float(config._raw_get('models.lightgbm_optuna.colsample_bytree_min', 0.5)),
            float(config._raw_get('models.lightgbm_optuna.colsample_bytree_max', 1.0))),
        'reg_alpha': trial.suggest_float('reg_alpha',
            float(config._raw_get('models.lightgbm_optuna.reg_alpha_min', 1e-8)),
            float(config._raw_get('models.lightgbm_optuna.reg_alpha_max', 10.0)), log=True),
        'reg_lambda': trial.suggest_float('reg_lambda',
            float(config._raw_get('models.lightgbm_optuna.reg_lambda_min', 1e-8)),
            float(config._raw_get('models.lightgbm_optuna.reg_lambda_max', 10.0)), log=True),
        'min_split_gain': trial.suggest_float('min_split_gain',
            float(config._raw_get('models.lightgbm_optuna.min_split_gain_min', 0.0)),
            float(config._raw_get('models.lightgbm_optuna.min_split_gain_max', 1.0))),
    })
    return _truncate_num_leaves(params)


def _cv_bayes_lightgbm_params(X_train, y_train, config: PipelineConfig,
                               random_state: int, mode: str = 'bayes') -> Tuple[Dict, Dict]:
    """CV-based hyperparameter search used when Optuna is not available.

    Priority:
      1. BayesSearchCV  (scikit-optimize)  — true Bayesian optimisation over CV folds
      2. RandomizedSearchCV (sklearn)      — random search over CV folds
    Both avoid the held-out single-split bias of the Optuna path and do not
    require Optuna to be installed.
    """
    base_params = get_lightgbm_base_params(config, random_state)
    n_iter = int(config._raw_get('models.lightgbm_optuna.n_trials', 25))
    cv_splits = int(config._raw_get('models.hybrid.cv_splits', 3))
    scoring = 'average_precision' if mode == 'multiobj' else 'roc_auc'

    cv = StratifiedKFold(n_splits=min(cv_splits, 3), shuffle=True,
                         random_state=int(random_state))

    if SKOPT_AVAILABLE:
        # ── BayesSearchCV (scikit-optimize) ──────────────────────────────────
        search_space = {
            'n_estimators':      _SkInteger(
                int(config._raw_get('models.lightgbm_optuna.n_estimators_min', 50)),
                int(config._raw_get('models.lightgbm_optuna.n_estimators_max', 300))),
            'learning_rate':     _SkReal(
                float(config._raw_get('models.lightgbm_optuna.learning_rate_min', 0.01)),
                float(config._raw_get('models.lightgbm_optuna.learning_rate_max', 0.2)),
                prior='log-uniform'),
            'max_depth':         _SkInteger(
                int(config._raw_get('models.lightgbm_optuna.max_depth_min', 3)),
                int(config._raw_get('models.lightgbm_optuna.max_depth_max', 12))),
            'num_leaves':        _SkInteger(
                int(config._raw_get('models.lightgbm_optuna.num_leaves_min', 16)),
                int(config._raw_get('models.lightgbm_optuna.num_leaves_max', 255))),
            'min_child_samples': _SkInteger(
                int(config._raw_get('models.lightgbm_optuna.min_child_samples_min', 5)),
                int(config._raw_get('models.lightgbm_optuna.min_child_samples_max', 100))),
            'subsample':         _SkReal(
                float(config._raw_get('models.lightgbm_optuna.subsample_min', 0.5)),
                float(config._raw_get('models.lightgbm_optuna.subsample_max', 1.0))),
            'colsample_bytree':  _SkReal(
                float(config._raw_get('models.lightgbm_optuna.colsample_bytree_min', 0.5)),
                float(config._raw_get('models.lightgbm_optuna.colsample_bytree_max', 1.0))),
            'reg_alpha':         _SkReal(1e-8, 10.0, prior='log-uniform'),
            'reg_lambda':        _SkReal(1e-8, 10.0, prior='log-uniform'),
        }
        lgbm = LGBMClassifier(**base_params)
        search = _BayesSearchCV(
            lgbm, search_space, n_iter=n_iter, cv=cv,
            scoring=scoring, n_jobs=1, random_state=int(random_state),
            refit=True, verbose=0,
        )
        search.fit(X_train, y_train)
        best_params = _truncate_num_leaves({**base_params, **search.best_params_})
        optimizer_name = 'cv_bayes_skopt'
        best_score = float(search.best_score_)
    else:
        # ── RandomizedSearchCV (sklearn, always available) ────────────────────
        param_dist = {
            'n_estimators':      [50, 100, 150, 200, 300],
            'learning_rate':     [0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
            'max_depth':         [3, 5, 7, 9, 12],
            'num_leaves':        [16, 31, 63, 127, 255],
            'min_child_samples': [5, 10, 20, 50, 100],
            'subsample':         [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree':  [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha':         [0.0, 0.001, 0.01, 0.1, 1.0, 10.0],
            'reg_lambda':        [0.0, 0.001, 0.01, 0.1, 1.0, 10.0],
        }
        from sklearn.model_selection import RandomizedSearchCV
        lgbm = LGBMClassifier(**base_params)
        search = RandomizedSearchCV(
            lgbm, param_dist, n_iter=n_iter, cv=cv,
            scoring=scoring, n_jobs=1, random_state=int(random_state),
            refit=True,
        )
        search.fit(X_train, y_train)
        best_params = _truncate_num_leaves({**base_params, **search.best_params_})
        optimizer_name = 'cv_random_search'
        best_score = float(search.best_score_)

    summary = {
        'optimizer': optimizer_name, 'mode': mode,
        'n_iter': n_iter, 'best_value': best_score,
        'best_params': best_params,
        'primary_metric_name': scoring,
        'primary_metric_value': best_score,
    }
    logging.info(f"LGB_Bayes CV opt ({optimizer_name}, mode={mode}): best {scoring}={best_score:.4f}")
    return best_params, summary


def optimize_lightgbm_params(X_train, y_train, X_valid, y_valid, config, random_state, mode='bayes'):
    base_params = get_lightgbm_base_params(config, random_state)
    mode = str(mode).lower()
    metric_name = str(config._raw_get('models.lightgbm_optuna.metric', 'ROC_AUC'))
    n_trials = int(config._raw_get('models.lightgbm_optuna.n_trials', 25))
    timeout_sec = config._raw_get('models.lightgbm_optuna.timeout_sec', None)
    lambda_weight = float(config._raw_get('models.lightgbm_multiobj.lambda', 0.85))
    runtime_budget_sec = float(config._raw_get('models.lightgbm_multiobj.runtime_budget_sec', 5.0))

    if not LGBM_AVAILABLE:
        return base_params, {'optimizer': 'none', 'mode': mode, 'reason': 'lightgbm_unavailable'}
    if not OPTUNA_AVAILABLE:
        logging.warning(
            f"Optuna not available — LGB_{mode} will use CV-based Bayesian/random search "
            f"({'BayesSearchCV via scikit-optimize' if SKOPT_AVAILABLE else 'RandomizedSearchCV via sklearn'})."
        )
        return _cv_bayes_lightgbm_params(X_train, y_train, config, random_state, mode)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_valid)) < 2:
        logging.warning(f"Single-class split for LGB_{mode} — falling back to CV-based search.")
        return _cv_bayes_lightgbm_params(X_train, y_train, config, random_state, mode)

    sampler = optuna.samplers.TPESampler(seed=int(random_state), multivariate=True)
    study = optuna.create_study(direction='maximize', sampler=sampler)

    def objective(trial):
        params = _suggest_lightgbm_params(trial, config, random_state)
        t0 = time.perf_counter()
        model = LGBMClassifier(**params).fit(X_train, y_train)
        y_prob = model.predict_proba(X_valid)[:, 1]
        total_time = time.perf_counter() - t0
        if mode == 'multiobj':
            primary = _score_probability_metric(y_valid, y_prob, 'PR_AUC')
            runtime_norm = min(total_time / max(runtime_budget_sec, 1e-9), 1.0)
            score = float(lambda_weight * primary - (1.0 - lambda_weight) * runtime_norm)
            trial.set_user_attr('primary_metric_name', 'PR_AUC')
        else:
            primary = _score_probability_metric(y_valid, y_prob, metric_name)
            score = float(primary)
            trial.set_user_attr('primary_metric_name', metric_name)
        trial.set_user_attr('primary_metric_value', float(primary))
        trial.set_user_attr('runtime_sec', total_time)
        return score

    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=False)
    tuned_params = _truncate_num_leaves({**base_params, **study.best_params})
    bt = study.best_trial
    summary = {
        'optimizer': 'optuna_tpe', 'mode': mode,
        'n_trials': len(study.trials), 'best_value': float(bt.value),
        'best_params': tuned_params,
        'primary_metric_name': bt.user_attrs.get('primary_metric_name', metric_name),
        'primary_metric_value': bt.user_attrs.get('primary_metric_value', np.nan),
        'runtime_sec': bt.user_attrs.get('runtime_sec', np.nan),
    }
    logging.info(f"LightGBM opt complete mode={mode}: best={summary['best_value']:.4f}")
    return tuned_params, summary


def fit_lightgbm_variants(X_train, y_train, X_val, y_val, config, random_state):
    models, scores, tuning_info = {}, {}, {}
    if not LGBM_AVAILABLE:
        return models, scores, tuning_info

    default_params = get_lightgbm_base_params(config, random_state)
    lgb_model = LGBMClassifier(**default_params).fit(X_train, y_train)
    prob_lgb = lgb_model.predict_proba(X_val)[:, 1]
    models['LGB'] = lgb_model
    scores['LGB'] = prob_lgb
    tuning_info['LGB'] = {
        'optimizer': 'none', 'mode': 'default', 'best_params': default_params,
        'primary_metric_name': 'ROC_AUC',
        'primary_metric_value': float(roc_auc_score(y_val, prob_lgb)) if len(np.unique(y_val)) > 1 else np.nan,
    }

    val_frac = float(config._raw_get('models.lightgbm_optuna.validation_fraction', 0.2))
    if len(np.unique(y_train)) > 1 and len(y_train) >= 20:
        X_bo_tr, X_bo_val, y_bo_tr, y_bo_val = train_test_split(
            X_train, y_train, test_size=val_frac, stratify=y_train, random_state=int(random_state))
    else:
        X_bo_tr, X_bo_val, y_bo_tr, y_bo_val = X_train, X_val, y_train, y_val

    for variant, mode, rs_offset in [('LGB_Bayes', 'bayes', 0), ('LGB_MultiObj', 'multiobj', 17)]:
        params, summary = optimize_lightgbm_params(
            X_bo_tr, y_bo_tr, X_bo_val, y_bo_val, config=config,
            random_state=int(random_state) + rs_offset, mode=mode)
        m = LGBMClassifier(**params).fit(X_train, y_train)
        p = m.predict_proba(X_val)[:, 1]
        models[variant] = m
        scores[variant] = p
        tuning_info[variant] = summary

    return models, scores, tuning_info


# ============================================================================
# HYBRID ALPHA SELECTION
# ============================================================================


def select_hybrid_alpha_nested(X_train, y_train, config, random_state, lgb_params,
                                n_alphas=11, cv_splits=3):
    """Leakage-aware alpha selection via nested inner CV."""
    y_train = np.asarray(y_train).astype(int)
    X_train = np.asarray(X_train, dtype=np.float32)
    if len(np.unique(y_train)) < 2:
        return 0.5, [(0.5, np.nan)]
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    safe_splits = min(cv_splits, n_pos, n_neg)
    if safe_splits < 2:
        return 0.5, [(0.5, np.nan)]

    inner_cv = StratifiedKFold(n_splits=safe_splits, shuffle=True, random_state=int(random_state))
    alphas = np.linspace(0.0, 1.0, int(n_alphas))
    alpha_scores = []
    for alpha in alphas:
        fold_scores = []
        for tr_idx, val_idx in inner_cv.split(X_train, y_train):
            Xtr, ytr = X_train[tr_idx], y_train[tr_idx]
            Xval, yval = X_train[val_idx], y_train[val_idx]
            if len(np.unique(ytr)) < 2 or len(np.unique(yval)) < 2:
                continue
            try:
                # Fit a fresh scaler on this inner-fold's training split only —
                # never reuse the outer fold's scaler here, and never fit on Xval.
                inner_scaler = LeakageAwareScaler()
                Xtr_scaled = inner_scaler.fit_transform(Xtr)
                Xval_scaled = inner_scaler.transform(Xval)

                gm_g = GaussianMixture(n_components=config._raw_get('models.n_components', 5),
                                       max_iter=config._raw_get('models.max_iter_gm', 500),
                                       random_state=int(random_state)).fit(Xtr_scaled[ytr == 1])
                gm_b = GaussianMixture(n_components=config._raw_get('models.n_components', 5),
                                       max_iter=config._raw_get('models.max_iter_gm', 500),
                                       random_state=int(random_state)).fit(Xtr_scaled[ytr == 0])
                lgb_inner = LGBMClassifier(**dict(lgb_params)).fit(Xtr, ytr)  # unscaled, unchanged
                hs = compute_hybrid_scores(gm_g, gm_b, lgb_inner, float(alpha),
                                           X_gm=Xval_scaled, X_lgb=Xval)
                fold_scores.append(float(roc_auc_score(yval, hs)))
            except Exception:
                continue
        alpha_scores.append(float(np.mean(fold_scores)) if fold_scores else np.nan)

    arr = np.asarray(alpha_scores, dtype=float)
    if np.all(np.isnan(arr)):
        return 0.5, list(zip(alphas.tolist(), arr.tolist()))
    best_alpha = float(alphas[int(np.nanargmax(arr))])
    return best_alpha, list(zip(alphas.tolist(), arr.tolist()))


# ============================================================================
# CROSS-VALIDATION EVALUATION
# ============================================================================


def cross_validation_evaluation_fixed(X_all, y_all, groups=None, n_splits=5,
                                       random_state=42, config=None):
    """CV with LightGBM variants and one hybrid per LightGBM variant.
    All scores are normalised to (0,1) before metric computation."""
    logging.info("Starting cross-validation with leakage control and LightGBM hybrids...")
    X_all = np.asarray(X_all, dtype=np.float32)
    y_all = np.asarray(y_all).astype(int)

    if groups is not None and config._raw_get('cv.use_block_cv', True):
        splitter, sname = make_block_splitter(n_splits, random_state)
        logging.info(f"Using grouped CV: {sname}")
        split_iter = splitter.split(X_all, y_all, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        logging.info("Using random StratifiedKFold CV")
        split_iter = splitter.split(X_all, y_all)

    model_order = ["GM", "BGM", "LogReg", "RandForest",
                   "LGB", "LGB_Bayes", "LGB_MultiObj",
                   "Hybrid_LGB", "Hybrid_LGB_Bayes", "Hybrid_LGB_MultiObj",
                   "Hybrid_LGB_alpha1", "Hybrid_LGB_alpha08", "Hybrid_LGB_alpha07"]
    metrics_all = {m: [] for m in model_order}
    oof_scores = {m: np.full(len(y_all), np.nan, dtype=float) for m in model_order}

    for fold, (train_idx, val_idx) in enumerate(split_iter, start=1):
        logging.info(f"--- Fold {fold} ---")
        skip_intensive = _check_memory_skip(config)

        X_train_raw, X_val_raw = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]

        imputer = LeakageAwareImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train_raw)
        X_val = imputer.transform(X_val_raw)

        # Per-fold-fit standardization ahead of GM, BGM and Logistic Regression
        # (scale-sensitive). RF and LightGBM are scale-invariant by construction
        # and stay on the raw imputed X_train/X_val.
        scaler = LeakageAwareScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # ---- GM ----
        gm_good_cv = GaussianMixture(
            n_components=config._raw_get('models.n_components', 5),
            max_iter=config._raw_get('models.max_iter_gm', 500),
            random_state=int(random_state) + fold,
        ).fit(X_train_scaled[y_train == 1])
        gm_bad_cv = GaussianMixture(
            n_components=config._raw_get('models.n_components', 5),
            max_iter=config._raw_get('models.max_iter_gm', 500),
            random_state=int(random_state) + fold,
        ).fit(X_train_scaled[y_train == 0])
        gm_scores = compute_vqslod_prob(gm_good_cv, gm_bad_cv, X_val_scaled)  # sigmoid-normalised
        metrics_all["GM"].append(_metric_dict_for_model("GM", y_val, gm_scores))
        oof_scores["GM"][val_idx] = gm_scores

        # ---- BGM ----
        if not skip_intensive:
            try:
                bgm_g = BayesianGaussianMixture(
                    n_components=config._raw_get('models.n_components', 5),
                    max_iter=config._raw_get('models.max_iter_bgm', 1000),
                    random_state=int(random_state) + fold,
                    weight_concentration_prior=1e-2,
                ).fit(X_train_scaled[y_train == 1])
                bgm_b = BayesianGaussianMixture(
                    n_components=config._raw_get('models.n_components', 5),
                    max_iter=config._raw_get('models.max_iter_bgm', 1000),
                    random_state=int(random_state) + fold,
                    weight_concentration_prior=1e-2,
                ).fit(X_train_scaled[y_train == 0])
                bgm_scores = compute_vqslod_prob(bgm_g, bgm_b, X_val_scaled)  # sigmoid-normalised
                metrics_all["BGM"].append(_metric_dict_for_model("BGM", y_val, bgm_scores))
                oof_scores["BGM"][val_idx] = bgm_scores
            except Exception as e:
                logging.warning(f"BGM fold {fold} failed: {e}")
                metrics_all["BGM"].append(_nan_metric_dict())
        else:
            metrics_all["BGM"].append(_nan_metric_dict())

        # ---- LR ----
        lr_cv = LogisticRegression(max_iter=1000, random_state=int(random_state) + fold).fit(X_train_scaled, y_train)
        lr_scores = lr_cv.predict_proba(X_val_scaled)[:, 1]
        metrics_all["LR"].append(_metric_dict_for_model("LR", y_val, lr_scores))
        oof_scores["LogReg"][val_idx] = lr_scores

        # ---- RF ----
        if not skip_intensive:
            try:
                rf_cv = RandomForestClassifier(
                    n_estimators=50, max_depth=10, min_samples_split=10,
                    n_jobs=2, random_state=int(random_state) + fold,
                ).fit(X_train, y_train)
                rf_scores = rf_cv.predict_proba(X_val)[:, 1]
                metrics_all["RF"].append(_metric_dict_for_model("RF", y_val, rf_scores))
                oof_scores["RandForest"][val_idx] = rf_scores
            except Exception as e:
                logging.warning(f"RF fold {fold} failed: {e}")
                metrics_all["RF"].append(_nan_metric_dict())
        else:
            metrics_all["RF"].append(_nan_metric_dict())

        # ---- LGB variants ----
        if LGBM_AVAILABLE:
            lgb_models, lgb_val_scores, _ = fit_lightgbm_variants(
                X_train, y_train, X_val, y_val,
                config=config, random_state=int(random_state) + fold,
            )
            for base_name in HYBRID_BASE_MODELS:
                if base_name in lgb_val_scores:
                    bs = np.asarray(lgb_val_scores[base_name], dtype=float)
                    metrics_all[base_name].append(_metric_dict_for_model(base_name, y_val, bs))
                    oof_scores[base_name][val_idx] = bs
                else:
                    metrics_all[base_name].append(_nan_metric_dict())

            # ---- Hybrids ----
            for hidx, base_name in enumerate(HYBRID_BASE_MODELS):
                hname = get_hybrid_name(base_name)
                if base_name not in lgb_models:
                    metrics_all[hname].append(_nan_metric_dict())
                    continue
                try:
                    best_alpha, _ = select_hybrid_alpha_nested(
                        X_train, y_train, config,
                        random_state=int(random_state) + fold * 100 + hidx,
                        lgb_params=lgb_models[base_name].get_params(),
                        n_alphas=int(config._raw_get('models.hybrid.n_alphas', 11)),
                        cv_splits=int(config._raw_get('models.hybrid.cv_splits', 3)),
                    )
                    hs = compute_hybrid_scores(gm_good_cv, gm_bad_cv, lgb_models[base_name],
                                              float(best_alpha), X_gm=X_val_scaled, X_lgb=X_val)
                    metrics_all[hname].append(_metric_dict_for_model(hname, y_val, hs))
                    oof_scores[hname][val_idx] = hs
                    logging.info(f"Fold {fold} {hname}: alpha={best_alpha:.3f}")
                except Exception as e:
                    logging.warning(f"Hybrid fold {fold} {hname} failed: {e}")
                    metrics_all[hname].append(_nan_metric_dict())

            # ---- Fixed-alpha hybrids (LGB base only) ----
            for fa_name, fa_alpha in FIXED_ALPHA_HYBRIDS:
                if "LGB" not in lgb_models:
                    metrics_all[fa_name].append(_nan_metric_dict())
                    continue
                try:
                    hs = compute_hybrid_scores(gm_good_cv, gm_bad_cv,
                                               lgb_models["LGB"], fa_alpha,
                                               X_gm=X_val_scaled, X_lgb=X_val)
                    metrics_all[fa_name].append(_metric_dict_for_model(fa_name, y_val, hs))
                    oof_scores[fa_name][val_idx] = hs
                    logging.info(f"Fold {fold} {fa_name}: alpha={fa_alpha:.2f} (fixed)")
                except Exception as e:
                    logging.warning(f"Fixed-alpha hybrid fold {fold} {fa_name} failed: {e}")
                    metrics_all[fa_name].append(_nan_metric_dict())
        else:
            for base_name in HYBRID_BASE_MODELS:
                metrics_all[base_name].append(_nan_metric_dict())
                metrics_all[get_hybrid_name(base_name)].append(_nan_metric_dict())
            for fa_name, _ in FIXED_ALPHA_HYBRIDS:
                metrics_all[fa_name].append(_nan_metric_dict())

        gc.collect()
        if config._raw_get('memory_safety.monitor_memory', True):
            monitor_memory(f"Fold {fold} end")

    avg_metrics = []
    for mname, mlist in metrics_all.items():
        vm = {k: [m[k] for m in mlist if not np.isnan(m[k])]
              for k in ["auc", "precision", "recall", "f1", "accuracy"]}
        avg_metrics.append({
            "Model": mname,
            "Average_AUC": float(np.mean(vm["auc"])) if vm["auc"] else np.nan,
            "Average_Precision": float(np.mean(vm["precision"])) if vm["precision"] else np.nan,
            "Average_Recall": float(np.mean(vm["recall"])) if vm["recall"] else np.nan,
            "Average_F1": float(np.mean(vm["f1"])) if vm["f1"] else np.nan,
            "Average_Accuracy": float(np.mean(vm["accuracy"])) if vm["accuracy"] else np.nan,
        })

    logging.info("Cross-validation evaluation complete")
    return avg_metrics, metrics_all, oof_scores


# ============================================================================
# FINAL MODEL TRAINING
# ============================================================================


def train_final_models_fixed(X_all, y_all, feature_keys, config, output_dir):
    """Train final models (including all hybrids) on the full dataset."""
    logging.info("Training final models with LightGBM variants and per-model hybrids...")
    if config._raw_get('memory_safety.monitor_memory', True):
        monitor_memory("Before final model training")

    rs = int(config._raw_get('models.random_state', 42))
    X_temp, X_val, y_temp, y_val = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=rs)

    imputer = LeakageAwareImputer(strategy='mean')
    X_temp_imp = imputer.fit_transform(X_temp)
    X_val_imp = imputer.transform(X_val)

    scaler = LeakageAwareScaler()
    X_temp_scaled = scaler.fit_transform(X_temp_imp)
    X_val_scaled = scaler.transform(X_val_imp)

    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)

    gm_good = GaussianMixture(n_components=config._raw_get('models.n_components', 5),
                               max_iter=config._raw_get('models.max_iter_gm', 500),
                               random_state=rs).fit(X_temp_scaled[y_temp == 1])
    gm_bad = GaussianMixture(n_components=config._raw_get('models.n_components', 5),
                              max_iter=config._raw_get('models.max_iter_gm', 500),
                              random_state=rs).fit(X_temp_scaled[y_temp == 0])

    bgm_good = bgm_bad = None
    allow_bgm = True
    try:
        if config._raw_get('memory_safety.monitor_memory', True):
            import psutil
            mem_mb = psutil.Process().memory_info().rss / 1024 ** 2
            allow_bgm = mem_mb < config._raw_get('memory_safety.max_memory_mb', 8000) * 0.7
    except Exception:
        pass
    if allow_bgm:
        try:
            bgm_good = BayesianGaussianMixture(n_components=config._raw_get('models.n_components', 5),
                                               max_iter=config._raw_get('models.max_iter_bgm', 1000),
                                               random_state=rs, weight_concentration_prior=1e-2,
                                               ).fit(X_temp_scaled[y_temp == 1])
            bgm_bad = BayesianGaussianMixture(n_components=config._raw_get('models.n_components', 5),
                                              max_iter=config._raw_get('models.max_iter_bgm', 1000),
                                              random_state=rs, weight_concentration_prior=1e-2,
                                              ).fit(X_temp_scaled[y_temp == 0])
        except Exception as e:
            logging.warning(f"BGM training skipped: {e}")

    lr_model = LogisticRegression(max_iter=1000, random_state=rs).fit(X_temp_scaled, y_temp)
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10,
                                      n_jobs=2, random_state=rs).fit(X_temp_imp, y_temp)

    lgb_models, _, lgb_tuning_info = fit_lightgbm_variants(
        X_temp_imp, y_temp, X_val_imp, y_val, config=config, random_state=rs)

    hybrid_summary: Dict[str, Any] = {}
    for idx, base_name in enumerate(HYBRID_BASE_MODELS):
        if base_name not in lgb_models:
            continue
        try:
            best_alpha, alpha_cv = select_hybrid_alpha_nested(
                X_temp_imp, y_temp, config, random_state=rs + 100 + idx,
                lgb_params=lgb_models[base_name].get_params(),
                n_alphas=int(config._raw_get('models.hybrid.n_alphas', 11)),
                cv_splits=int(config._raw_get('models.hybrid.cv_splits', 3)),
            )
            val_scores = compute_hybrid_scores(gm_good, gm_bad, lgb_models[base_name], best_alpha,
                                              X_gm=X_val_scaled, X_lgb=X_val_imp)
            val_auc = float(roc_auc_score(y_val, val_scores)) if len(np.unique(y_val)) > 1 else np.nan
            hybrid_summary[get_hybrid_name(base_name)] = {
                'base_model_name': base_name, 'best_alpha': float(best_alpha),
                'val_auc': val_auc, 'alpha_cv_scores': alpha_cv,
            }
            logging.info(f"{get_hybrid_name(base_name)}: alpha={best_alpha:.3f}, val_auc={val_auc:.4f}")
        except Exception as e:
            logging.warning(f"Could not tune hybrid for {base_name}: {e}")

    logging.info("Retraining on full dataset...")
    X_full = imputer.fit_transform(X_all)
    X_full_scaled = scaler.fit_transform(X_full)

    final_models: Dict[str, Any] = {}
    final_models['GM'] = (
        GaussianMixture(**gm_good.get_params()).fit(X_full_scaled[y_all == 1]),
        GaussianMixture(**gm_bad.get_params()).fit(X_full_scaled[y_all == 0]),
    )
    if bgm_good is not None and bgm_bad is not None:
        final_models['BGM'] = (
            BayesianGaussianMixture(**bgm_good.get_params()).fit(X_full_scaled[y_all == 1]),
            BayesianGaussianMixture(**bgm_bad.get_params()).fit(X_full_scaled[y_all == 0]),
        )
    else:
        final_models['BGM'] = (None, None)
        logging.warning("BGM not included in final models due to memory constraints.")

    final_models['LogReg'] = LogisticRegression(**lr_model.get_params()).fit(X_full_scaled, y_all)
    final_models['RandForest'] = RandomForestClassifier(**rf_model.get_params()).fit(X_full, y_all)

    for base_name in HYBRID_BASE_MODELS:
        if base_name in lgb_models:
            final_models[base_name] = LGBMClassifier(
                **dict(lgb_tuning_info[base_name]['best_params'])).fit(X_full, y_all)

    for hname, info in hybrid_summary.items():
        bname = info['base_model_name']
        if bname not in final_models:
            continue
        final_models[hname] = {
            'gm_good': final_models['GM'][0],
            'gm_bad': final_models['GM'][1],
            'lgb_model': final_models[bname],
            'best_alpha': float(info['best_alpha']),
            'base_model_name': bname,
            'val_auc': info['val_auc'],
            'alpha_cv_scores': info['alpha_cv_scores'],
        }

    # ---- Fixed-alpha hybrids (built on LGB base, no tuning needed) ----
    if 'LGB' in final_models:
        for fa_name, fa_alpha in FIXED_ALPHA_HYBRIDS:
            final_models[fa_name] = {
                'gm_good': final_models['GM'][0],
                'gm_bad': final_models['GM'][1],
                'lgb_model': final_models['LGB'],
                'best_alpha': float(fa_alpha),
                'base_model_name': 'LGB',
                'val_auc': float(roc_auc_score(y_val,
                    compute_hybrid_scores(gm_good, gm_bad, final_models['LGB'],
                                         fa_alpha, X_gm=X_val_scaled, X_lgb=X_val_imp)))
                           if len(np.unique(y_val)) > 1 else np.nan,
                'alpha_cv_scores': [],
            }
            logging.info(f"{fa_name}: alpha={fa_alpha:.2f} (fixed)")

    save_all_models(final_models, output_dir, imputer, scaler)

    with open(os.path.join(output_dir, 'models', 'lightgbm_tuning_summary.json'), 'w') as f:
        json.dump(lgb_tuning_info, f, indent=2)
    with open(os.path.join(output_dir, 'models', 'hybrid_tuning_summary.json'), 'w') as f:
        json.dump(hybrid_summary, f, indent=2)

    if config._raw_get('memory_safety.monitor_memory', True):
        monitor_memory("After final model training")

    logging.info("Final models trained and saved.")
    return final_models, imputer, scaler


def save_all_models(models: Dict, output_dir: str, imputer: LeakageAwareImputer,
                    scaler: LeakageAwareScaler):
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_paths: Dict[str, Any] = {}

    for name, model in models.items():
        if name in ("GM", "BGM"):
            pg = os.path.join(model_dir, f"{name.lower()}_good.pkl")
            pb = os.path.join(model_dir, f"{name.lower()}_bad.pkl")
            if model[0] is not None:
                joblib.dump(model[0], pg)
            if model[1] is not None:
                joblib.dump(model[1], pb)
            model_paths[name] = [pg, pb]
        elif is_hybrid_model_name(name):
            path = os.path.join(model_dir, f"{name.lower()}_model.pkl")
            joblib.dump(model, path)
            model_paths[name] = path
        elif model is not None:
            path = os.path.join(model_dir, f"{name.lower()}.pkl")
            joblib.dump(model, path)
            model_paths[name] = path

    imputer_path = os.path.join(model_dir, "feature_imputer.pkl")
    joblib.dump(imputer, imputer_path)
    scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    metadata = {'model_paths': model_paths, 'imputer_path': imputer_path,
                'scaler_path': scaler_path,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(model_dir, "model_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"All models saved to {model_dir}")


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================


def generate_tranche_analysis(models, X, y, output_dir, config, X_scaled=None):
    logging.info("Generating tranche analysis...")
    tranche_sensitivities = [1.0, 0.999, 0.99, 0.90]
    records = []
    for name, model in models.items():
        try:
            scores = score_model_instance(name, model, X, X_scaled=X_scaled)
        except Exception as e:
            logging.warning(f"Could not compute scores for {name}: {e}")
            continue
        for sens in tranche_sensitivities:
            thr = find_threshold_at_sensitivity(y, scores, sens)
            if np.isnan(thr):
                continue
            preds = (scores >= thr).astype(int)
            records.append({'Model': name, 'Sensitivity': sens, 'Threshold': thr,
                             'Precision': precision_score(y, preds, zero_division=0),
                             'Recall': recall_score(y, preds, zero_division=0),
                             'F1': f1_score(y, preds, zero_division=0),
                             'FDR': 1.0 - precision_score(y, preds, zero_division=0)})

    df_tranches = pd.DataFrame(records)
    df_tranches.to_csv(os.path.join(output_dir, "tranche_metrics.csv"), index=False)

    if not df_tranches.empty:
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        for col, label, marker in [("Precision", "Precision", "o"), ("F1", "F1 Score", "s")]:
            plt.figure(figsize=(10, 6))
            for mname in df_tranches["Model"].unique():
                sub = df_tranches[df_tranches["Model"] == mname]
                plt.plot(sub["Sensitivity"], sub[col], marker=marker, label=mname)
            plt.xlabel("Sensitivity")
            plt.ylabel(label)
            plt.title(f"{label} at Fixed Sensitivities")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{col.lower()}_vs_sensitivity.png"), dpi=300)
            plt.close()

        plt.figure(figsize=(10, 8))
        for name, model in models.items():
            try:
                scores = score_model_instance(name, model, X, X_scaled=X_scaled)
                fpr, tpr, _ = roc_curve(y, scores)
                auc = roc_auc_score(y, scores)
                plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
            except Exception:
                continue
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves - All Models")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "roc_curves.png"), dpi=300)
        plt.close()

    logging.info(f"Tranche analysis saved to {output_dir}")


def run_variant_classification_counts(df_variants, X, y, models, output_dir,
                                       target_sens=0.99, X_scaled=None):
    logging.info("Running variant classification analysis...")
    df_base = df_variants[["CHROM", "POS", "REF", "ALT"]].copy()
    df_base["POS"] = pd.to_numeric(df_base["POS"], errors="coerce").fillna(-1).astype(int)
    ref_len = df_base["REF"].astype(str).str.len()
    alt_len = df_base["ALT"].astype(str).str.len()
    df_base["VariantType"] = np.where((ref_len == 1) & (alt_len == 1), "SNP",
                              np.where(ref_len != alt_len, "INDEL", "OTHER"))
    detailed_records = []
    for name, model in models.items():
        try:
            scores = score_model_instance(name, model, X, X_scaled=X_scaled)
        except Exception as e:
            logging.warning(f"Could not compute scores for {name}: {e}")
            continue
        thr = find_threshold_at_sensitivity(y, scores, target_sens)
        if np.isnan(thr):
            continue
        preds = (scores >= thr).astype(int)
        status = np.empty_like(preds, dtype=object)
        status[(y == 1) & (preds == 1)] = "TP"
        status[(y == 0) & (preds == 1)] = "FP"
        status[(y == 1) & (preds == 0)] = "FN"
        status[(y == 0) & (preds == 0)] = "TN"
        df_out = df_base.copy()
        df_out["TruthLabel"] = y
        df_out["Model"] = name
        df_out["Threshold"] = thr
        df_out["TargetSensitivity"] = target_sens
        df_out["Score"] = scores
        df_out["Prediction"] = preds
        df_out["Status"] = status
        detailed_records.append(df_out)
        counts = {s: int((status == s).sum()) for s in ["TP", "FP", "FN", "TN"]}
        logging.info(f"{name}: thr={thr:.4f}, " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    if not detailed_records:
        logging.warning("No classification results produced")
        return

    df_all = pd.concat(detailed_records, ignore_index=True)
    df_all.to_csv(os.path.join(output_dir, "variant_classification_detailed.csv"), index=False)
    df_all.groupby(["Model", "Status", "VariantType"]).size().reset_index(name="Count").to_csv(
        os.path.join(output_dir, "variant_classification_summary.csv"), index=False)

    pivot_data = df_all.pivot_table(
        index=["CHROM", "POS", "REF", "ALT", "VariantType", "TruthLabel"],
        columns="Model", values="Status", aggfunc="first").reset_index()

    unique_results = {}
    model_names = df_all["Model"].unique().tolist()
    for mname in model_names:
        for status in ["TP", "TN"]:
            mask = pivot_data[mname] == status
            for other in model_names:
                if other != mname:
                    mask &= (pivot_data[other] != status)
            uv = pivot_data[mask].copy()
            if len(uv) > 0:
                unique_results[f"{mname}_{status}"] = uv

    if unique_results:
        with pd.ExcelWriter(os.path.join(output_dir, "unique_tp_tn_variants.xlsx"),
                            engine='openpyxl') as writer:
            for sheet_name, data in unique_results.items():
                data.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    logging.info("Variant classification analysis complete")


def run_paired_tests(per_fold_df: pd.DataFrame, output_path: str):
    metric_cols = ["AUC", "Precision", "Recall", "F1", "Accuracy"]
    # Build comparison pairs dynamically from the models that have at least some data,
    # so we get output even when LightGBM is unavailable.
    available_models = [
        m for m in per_fold_df["Model"].unique()
        if per_fold_df.loc[per_fold_df["Model"] == m, "AUC"].notna().any()
    ]
    # Preferred ordered pairs — only included when both sides exist
    preferred_pairs = [
        ("GM", "BGM"), ("GM", "LR"), ("GM", "RF"), ("LR", "RF"),
        ("LGB", "LGB_Bayes"), ("LGB", "LGB_MultiObj"), ("LGB_Bayes", "LGB_MultiObj"),
        ("LGB", "GM"), ("LGB", "LR"), ("LGB", "RF"),
        ("LGB", "Hybrid_LGB"), ("LGB_Bayes", "Hybrid_LGB_Bayes"),
        ("LGB_MultiObj", "Hybrid_LGB_MultiObj"),
        ("Hybrid_LGB", "Hybrid_LGB_Bayes"), ("Hybrid_LGB", "Hybrid_LGB_MultiObj"),
        ("Hybrid_LGB_Bayes", "Hybrid_LGB_MultiObj"),
    ]
    seen_pairs: set = set()
    comparison_pairs = []
    for a, b in preferred_pairs:
        if a in available_models and b in available_models and (a, b) not in seen_pairs:
            comparison_pairs.append((a, b))
            seen_pairs.add((a, b))
    # Add any remaining pairs not yet covered
    for i, a in enumerate(available_models):
        for b in available_models[i + 1:]:
            if (a, b) not in seen_pairs:
                comparison_pairs.append((a, b))
                seen_pairs.add((a, b))

    lines = []
    if not comparison_pairs:
        lines.append("[WARN] No model pairs available for paired tests "
                     "(no models produced valid AUC values).\n")
    for model_A, model_B in comparison_pairs:
        subA = per_fold_df[per_fold_df["Model"] == model_A]
        subB = per_fold_df[per_fold_df["Model"] == model_B]
        merged = pd.merge(subA[["Fold"] + metric_cols], subB[["Fold"] + metric_cols],
                          on="Fold", suffixes=("_A", "_B"), how="inner").dropna()
        if merged.empty:
            lines.append(f"[WARN] No paired folds for {model_A} vs {model_B}.\n")
            continue
        lines.append("=" * 80 + "\n")
        lines.append(f"Paired tests: {model_A} vs {model_B} (n={len(merged)} folds)\n")
        lines.append("=" * 80 + "\n")
        for m in metric_cols:
            x = merged[f"{m}_A"].values
            y = merged[f"{m}_B"].values
            diff = y - x
            try:
                t_stat, p_t = ttest_rel(y, x)
            except Exception:
                t_stat, p_t = np.nan, np.nan
            try:
                w_stat, p_w = wilcoxon(y, x, zero_method="wilcox", alternative="two-sided")
            except Exception:
                w_stat, p_w = np.nan, np.nan
            lines.append(f"Metric: {m}\n")
            lines.append(f"  mean {model_A}: {x.mean():.6f}\n")
            lines.append(f"  mean {model_B}: {y.mean():.6f}\n")
            lines.append(f"  mean Δ (B - A): {diff.mean():.6f}\n")
            lines.append(f"  Paired t-test: t = {t_stat:.3f}, p = {p_t:.3e}\n")
            lines.append(f"  Wilcoxon test: W = {w_stat:.3f}, p = {p_w:.3e}\n\n")

    report = "".join(lines)
    with open(output_path, 'w') as f:
        f.write(report)
    logging.info(f"Paired tests saved to {output_path}")
    return report


# ============================================================================
# CLINVAR-BASED EVALUATION
# ============================================================================


