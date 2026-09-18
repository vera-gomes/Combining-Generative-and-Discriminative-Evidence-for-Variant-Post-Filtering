"""Combining generative and discriminative evidence for variant post-filtering.

Public API: build a `PipelineConfig` (from a dict or YAML file) and run either
`run_train_pipeline` or `run_apply_pipeline`, or use the `cli.main()` entry point.
"""

from .config import PipelineConfig
from .models import (
    FIXED_ALPHA_HYBRIDS,
    HYBRID_BASE_MODELS,
    LeakageAwareImputer,
    compute_hybrid_scores,
    compute_vqslod,
    compute_vqslod_prob,
    get_hybrid_name,
    is_hybrid_model_name,
    is_hybrid_model_object,
    score_model_instance,
    sigmoid,
)
from .pipeline import run_apply_pipeline, run_train_pipeline

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "PipelineConfig",
    "run_train_pipeline",
    "run_apply_pipeline",
    "LeakageAwareImputer",
    "HYBRID_BASE_MODELS",
    "FIXED_ALPHA_HYBRIDS",
    "get_hybrid_name",
    "is_hybrid_model_name",
    "is_hybrid_model_object",
    "sigmoid",
    "compute_vqslod",
    "compute_vqslod_prob",
    "compute_hybrid_scores",
    "score_model_instance",
]
