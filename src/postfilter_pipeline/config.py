"""Pipeline configuration: pairwise-comparison resolution and PipelineConfig."""

import itertools
import json
import logging
import os
from datetime import datetime
from typing import Any, List, Optional

from .models import HYBRID_BASE_MODELS, get_hybrid_name

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logging.warning("PyYAML not available, using JSON for config instead.")


def _resolve_pairwise_comparisons(config: Any,
                                   score_cols: List[str]) -> List[List[str]]:
    """Return the list of model pairs for pairwise statistical tests.

    - If the user explicitly listed pairs in config (non-empty YAML list):
      use those, filtered to pairs where both score columns exist.
    - Otherwise (empty list = default): compare EVERY model against EVERY
      other model using whatever score columns are actually present.

    This replaces the old config.get() pattern that returned hardcoded names
    (LR, RF, LGB_Bayes…) not matching real column names (LogReg, RandForest…)
    and silently produced only 2 pairs instead of all-vs-all.
    """
    raw_user_pairs = config._raw_get('statistics.pairwise_comparisons', [])
    model_names = [c.replace("_score", "") for c in score_cols]

    if raw_user_pairs:
        valid = [
            [a, b] for a, b in raw_user_pairs
            if a in model_names and b in model_names and a != b
        ]
        if not valid:
            logging.warning(
                "None of the user-specified pairwise_comparisons matched available "
                f"score columns {model_names}. Falling back to all-vs-all."
            )
            return [list(p) for p in itertools.combinations(model_names, 2)]
        return valid

    # Default: all-vs-all from whatever models are actually present.
    pairs = [list(p) for p in itertools.combinations(model_names, 2)]
    logging.info(
        f"Pairwise comparisons: {len(pairs)} pairs from {len(model_names)} models "
        f"({model_names})"
    )
    return pairs


def _get_default_pairwise_comparisons() -> List[List[str]]:
    # Model names MUST match the keys used in final_models dict (training) and
    # therefore the score column names written to real_data_scores.csv (apply).
    # Final model save names: GM, BGM, LogReg, RandForest, LGB, LGB_Bayes,
    #   LGB_MultiObj, Hybrid_LGB, Hybrid_LGB_Bayes, Hybrid_LGB_MultiObj.
    # Legacy short names (LR, RF) are kept as aliases for OOF-based comparisons
    # but pairs using LogReg/RandForest are added so apply-mode always gets them.
    return [
        # Always-available (Gaussian + linear + tree, no LightGBM needed)
        ["GM", "BGM"],
        ["GM", "LogReg"],    ["GM", "LR"],
        ["GM", "RandForest"],["GM", "RF"],
        ["BGM", "LogReg"],   ["BGM", "LR"],
        ["BGM", "RandForest"],["BGM", "RF"],
        ["LogReg", "RandForest"], ["LR", "RF"],
        # LightGBM vs always-available
        ["LGB", "GM"],
        ["LGB", "LogReg"],   ["LGB", "LR"],
        ["LGB", "RandForest"],["LGB", "RF"],
        ["LGB_Bayes", "GM"],
        ["LGB_Bayes", "LogReg"],
        ["LGB_Bayes", "RandForest"],
        # LightGBM variant pairs
        ["LGB", "LGB_Bayes"],
        ["LGB", "LGB_MultiObj"],
        ["LGB_Bayes", "LGB_MultiObj"],
        # Hybrid vs base LGB
        ["LGB", "Hybrid_LGB"],
        ["LGB_Bayes", "Hybrid_LGB_Bayes"],
        ["LGB_MultiObj", "Hybrid_LGB_MultiObj"],
        # Hybrid vs hybrid
        ["Hybrid_LGB", "Hybrid_LGB_Bayes"],
        ["Hybrid_LGB", "Hybrid_LGB_MultiObj"],
        ["Hybrid_LGB_Bayes", "Hybrid_LGB_MultiObj"],
        # GM vs Hybrid
        ["GM", "Hybrid_LGB"],
        ["GM", "Hybrid_LGB_Bayes"],
        ["GM", "Hybrid_LGB_MultiObj"],
    ]


def _normalize_pairwise_comparisons(comparisons: Any) -> List[List[str]]:
    """Expand legacy Hybrid alias names; deduplicate.  If comparisons is empty
    the default list is returned.  Otherwise the user list is expanded and
    returned WITHOUT silently merging the defaults."""
    if not comparisons:
        return _get_default_pairwise_comparisons()

    normalized: List[List[str]] = []
    _legacy = {"Hybrid", "Hybrid_Global", "Hybrid_AdaptiveDP"}

    for pair in comparisons:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        a, b = str(pair[0]), str(pair[1])
        a_leg = a in _legacy
        b_leg = b in _legacy
        if a_leg and b_leg:
            normalized.extend([
                ["Hybrid_LGB", "Hybrid_LGB_Bayes"],
                ["Hybrid_LGB", "Hybrid_LGB_MultiObj"],
                ["Hybrid_LGB_Bayes", "Hybrid_LGB_MultiObj"],
            ])
        elif a_leg:
            normalized.extend([[b, get_hybrid_name(base)] for base in HYBRID_BASE_MODELS])
        elif b_leg:
            normalized.extend([[a, get_hybrid_name(base)] for base in HYBRID_BASE_MODELS])
        else:
            normalized.append([a, b])

    seen: set = set()
    deduped: List[List[str]] = []
    for a, b in normalized:
        if a == b:
            continue
        key = (a, b)
        if key not in seen:
            deduped.append([a, b])
            seen.add(key)
    return deduped if deduped else _get_default_pairwise_comparisons()


class PipelineConfig:
    """Configuration class for pipeline parameters."""

    def __init__(self, config_dict=None):
        self.config = config_dict or {
            'mode': 'train',
            'paths': {
                'output_dir': 'output',
                'train_input_vcf': 'train.vcf.gz',
                'truth_vcf': 'truth.vcf.gz',
                'apply_input_vcf': 'apply.vcf.gz',
                # For apply mode: directory produced by a previous train run
                'trained_model_dir': '',
            },
            'features': [
                'DP', 'QD', 'FS', 'SOR', 'MQ', 'MQRankSum',
                'ReadPosRankSum', 'BaseQRankSum', 'ClippingRankSum',
                'ExcessHet', 'InbreedingCoeff',
            ],
            'models': {
                'n_components': 5,
                'max_iter_gm': 500,
                'max_iter_bgm': 1000,
                'random_state': 42,
            },
            'cv': {
                'n_splits': 5,
                'use_block_cv': True,
                'block_size_bp': 1000000,
            },
            'evaluation': {
                'primary_sensitivity': 0.999,
                'variant_classification_target_sens': 0.99,
                # Bootstrap is skipped if the minority class has fewer than this many samples.
                # Protects against e.g. ClinVar with only ~170 positives running for days.
                'min_class_n_for_bootstrap': 500,
                # Maximum sample size for apply-mode bootstrap (subsample if larger).
                'apply_stats_max_n': 50000,
                'external_validation': {
                    'enabled': False,
                    'clinvar': {
                        'enabled': False,
                        'vcf': 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz',
                    },
                    'population_dbs': [],
                    'popdb_chunk_bp': 1000000,
                    'popdb_fetch_retries': 3,
                },
            },
            'statistics': {
                'pairwise_bootstrap_n': 1000,
                'pairwise_bootstrap_seed': 123,
                'pairwise_comparisons': [],  # empty → use defaults
            },
            'memory_safety': {
                'monitor_memory': True,
                'max_memory_mb': 8000,
                'skip_memory_intensive_models': True,
                'lightgbm_memory_safe': True,
                'batch_size': 100000,
            },
        }

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def _raw_get(self, key: str, default=None):
        """Get config value using dot notation, returning raw stored value."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def get(self, key: str, default=None):
        """Get config value; pairwise_comparisons are auto-normalised."""
        value = self._raw_get(key, default)
        if key == 'statistics.pairwise_comparisons':
            return _normalize_pairwise_comparisons(value)
        return value

    def set(self, key: str, value):
        """Set config value using dot notation."""
        keys = key.split('.')
        d = self.config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_yaml(self, path: str):
        if YAML_AVAILABLE:
            with open(path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=2)

    @classmethod
    def from_yaml(cls, path: str) -> 'PipelineConfig':
        if YAML_AVAILABLE:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(path, 'r') as f:
                config_dict = json.load(f)
        return cls(config_dict)

    # ------------------------------------------------------------------
    # Directory management
    # ------------------------------------------------------------------

    def initialize_paths(self, add_timestamp: Optional[bool] = None) -> str:
        """Create output directory and return its path."""
        base_dir = self._raw_get('paths.output_dir', 'output')
        if add_timestamp is None:
            add_timestamp = (self._raw_get('mode', 'train') == 'train')
        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"{base_dir}_{timestamp}"
        else:
            output_dir = base_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        return output_dir

