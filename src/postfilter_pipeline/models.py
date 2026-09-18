"""Core model primitives: leakage-aware imputer, hybrid-model helpers and scorers."""

from typing import Any, List, Tuple

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


HYBRID_BASE_MODELS: List[str] = ["LGB", "LGB_Bayes", "LGB_MultiObj"]

# Fixed-alpha hybrids built on the LGB base only.
# Each entry: (model_name, alpha)
#   alpha=1.0  → pure LGB (GMM weight = 0)
#   alpha=0.7  → 70 % LGB + 30 % GMM
FIXED_ALPHA_HYBRIDS: List[Tuple[str, float]] = [
    ("Hybrid_LGB_alpha1",   1.0),
    ("Hybrid_LGB_alpha08",  0.8),
    ("Hybrid_LGB_alpha07",  0.7),
]


class LeakageAwareImputer:
    """Safe imputation that prevents data leakage."""

    def __init__(self, strategy: str = 'mean'):
        self.imputer = SimpleImputer(strategy=strategy, copy=True)
        self.fitted = False

    def fit(self, X: np.ndarray) -> 'LeakageAwareImputer':
        self.imputer.fit(X)
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Imputer must be fitted before transform")
        return self.imputer.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str):
        joblib.dump(self.imputer, path)

    @classmethod
    def load(cls, path: str) -> 'LeakageAwareImputer':
        obj = cls()
        obj.imputer = joblib.load(path)
        obj.fitted = True
        return obj


class LeakageAwareScaler:
    """Safe standardization that prevents data leakage. Mirrors
    LeakageAwareImputer's fit/transform/save/load contract exactly, so
    apply-mode scoring can load it the same way it loads the imputer."""

    def __init__(self):
        self.scaler = StandardScaler(copy=True)
        self.fitted = False

    def fit(self, X: np.ndarray) -> 'LeakageAwareScaler':
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str):
        joblib.dump(self.scaler, path)

    @classmethod
    def load(cls, path: str) -> 'LeakageAwareScaler':
        obj = cls()
        obj.scaler = joblib.load(path)
        obj.fitted = True
        return obj


# ============================================================================
# HYBRID HELPERS
# ============================================================================


def get_hybrid_name(base_model_name: str) -> str:
    return f"Hybrid_{base_model_name}"


def is_hybrid_model_name(model_name: str) -> bool:
    return isinstance(model_name, str) and model_name.startswith("Hybrid_")


def is_hybrid_model_object(model: Any) -> bool:
    return isinstance(model, dict) and all(
        k in model for k in ["gm_good", "gm_bad", "lgb_model", "best_alpha"]
    )


# ============================================================================
# CORE SCORING PRIMITIVES
# ============================================================================


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid with overflow protection."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def compute_vqslod(model_good, model_bad, X: np.ndarray) -> np.ndarray:
    """Raw VQSLOD log-likelihood ratio (unbounded)."""
    return model_good.score_samples(X) - model_bad.score_samples(X)


def compute_vqslod_prob(model_good, model_bad, X: np.ndarray) -> np.ndarray:
    """VQSLOD mapped to (0, 1) via sigmoid — comparable to classifier probabilities."""
    return sigmoid(compute_vqslod(model_good, model_bad, X))


def compute_hybrid_scores(gm_good, gm_bad, lgb_model, alpha: float,
                          X_gm: np.ndarray, X_lgb: np.ndarray = None) -> np.ndarray:
    """Hybrid score: alpha * LGB + (1-alpha) * sigmoid(VQSLOD).

    X_gm must be the SCALED feature matrix (GM/BGM were fit on scaled
    features). X_lgb must be the raw/imputed (unscaled) feature matrix
    LightGBM was fit on. If X_lgb is omitted, X_gm is reused for backward
    compatibility, but every call site in this codebase should now pass
    both explicitly.
    """
    if X_lgb is None:
        X_lgb = X_gm
    mix_prob = compute_vqslod_prob(gm_good, gm_bad, X_gm)
    clf_prob = lgb_model.predict_proba(X_lgb)[:, 1]
    return alpha * clf_prob + (1.0 - alpha) * mix_prob


# Models fit on standardized features (see LeakageAwareScaler). Everything
# else (RF, LGB and its variants) is scale-invariant and stays on the raw
# imputed matrix.
SCALE_SENSITIVE_MODELS = {"GM", "BGM", "LogReg"}


def score_model_instance(name: str, model: Any, X: np.ndarray,
                         X_scaled: np.ndarray = None) -> np.ndarray:
    """Return scores in (0, 1) for any model type.

    X must be the raw/imputed (unscaled) feature matrix. X_scaled must be
    the same rows run through the same LeakageAwareScaler the model was
    trained with — required for GM, BGM, LogReg, and the GM half of every
    hybrid. If omitted, X is reused for backward compatibility, but every
    caller in this codebase should now pass both.

    GM/BGM: sigmoid(VQSLOD)  — standardised to probability range.
    Hybrid: alpha*LGB + (1-alpha)*sigmoid(VQSLOD).
    Others: predict_proba[:, 1].
    """
    if X_scaled is None:
        X_scaled = X
    if name in ("GM", "BGM"):
        if model is None or model[0] is None or model[1] is None:
            raise ValueError(f"Model {name} is unavailable")
        return compute_vqslod_prob(model[0], model[1], X_scaled)
    if is_hybrid_model_name(name) or is_hybrid_model_object(model):
        if not is_hybrid_model_object(model):
            raise ValueError(f"Hybrid model {name} is malformed")
        return compute_hybrid_scores(
            model["gm_good"], model["gm_bad"], model["lgb_model"],
            float(model["best_alpha"]), X_gm=X_scaled, X_lgb=X,
        )
    if model is None:
        raise ValueError(f"Model {name} is unavailable")
    if name in SCALE_SENSITIVE_MODELS:  # LogReg
        return model.predict_proba(X_scaled)[:, 1]
    return model.predict_proba(X)[:, 1]  # RF, LGB, LGB_Bayes, LGB_MultiObj


# ============================================================================
# EXTERNAL VALIDATION HELPERS
# ============================================================================


