"""Core model primitives: leakage-aware imputer, hybrid-model helpers and scorers."""

from typing import Any, List, Tuple

import joblib
import numpy as np
from sklearn.impute import SimpleImputer


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


def compute_hybrid_scores(gm_good, gm_bad, lgb_model, alpha: float, X: np.ndarray) -> np.ndarray:
    """Hybrid score: alpha * LGB + (1-alpha) * sigmoid(VQSLOD)."""
    mix_prob = compute_vqslod_prob(gm_good, gm_bad, X)
    clf_prob = lgb_model.predict_proba(X)[:, 1]
    return alpha * clf_prob + (1.0 - alpha) * mix_prob


def score_model_instance(name: str, model: Any, X: np.ndarray) -> np.ndarray:
    """Return scores in (0, 1) for any model type.

    GM/BGM: sigmoid(VQSLOD)  — standardised to probability range.
    Hybrid: alpha*LGB + (1-alpha)*sigmoid(VQSLOD).
    Others: predict_proba[:, 1].
    """
    if name in ("GM", "BGM"):
        if model is None or model[0] is None or model[1] is None:
            raise ValueError(f"Model {name} is unavailable")
        return compute_vqslod_prob(model[0], model[1], X)
    if is_hybrid_model_name(name) or is_hybrid_model_object(model):
        if not is_hybrid_model_object(model):
            raise ValueError(f"Hybrid model {name} is malformed")
        return compute_hybrid_scores(
            model["gm_good"], model["gm_bad"], model["lgb_model"],
            float(model["best_alpha"]), X,
        )
    if model is None:
        raise ValueError(f"Model {name} is unavailable")
    return model.predict_proba(X)[:, 1]


# ============================================================================
# EXTERNAL VALIDATION HELPERS
# ============================================================================


