#!/usr/bin/env python3
"""
Variant post-filtering pipeline (caller-agnostic, generative + discriminative + hybrid).

Supports two modes, selected via the `mode` key in the YAML config (or --mode):

  train  — cross-validated benchmark training of GM, BGM, Logistic Regression,
           Random Forest, LightGBM, LGB_Bayes, LGB_MultiObj, and per-model
           hybrid (generative + discriminative) scores; saves final models,
           CV diagnostics, and plots to the output directory.

  apply  — loads a previously trained model directory (`paths.trained_model_dir`)
           and scores a new VCF without retraining; optionally runs ClinVar-
           and/or population-database-supported external validation.

All paths, features, and run parameters are supplied via YAML config — see
configs/train.example.yaml and configs/apply.example.yaml.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import gc
import numpy as np
import pandas as pd
import pysam
import joblib
import logging
import os
import re
import time
import json
import argparse
import itertools
from urllib.parse import urlparse
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_score, recall_score,
    f1_score, accuracy_score, confusion_matrix, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef, log_loss,
)
from scipy.stats import ttest_rel, wilcoxon

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    logging.warning("LightGBM not available. LGBM models will be skipped.")

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logging.warning("PyYAML not available, using JSON for config instead.")

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
    from skopt.space import Real as _SkReal, Integer as _SkInteger
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

# ============================================================================
# CONSTANTS
# ============================================================================

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

# ============================================================================
# CONFIGURATION CLASS
# ============================================================================


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


# ============================================================================
# LEAKAGE-AWARE IMPUTER
# ============================================================================


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


def _is_remote_path(p: str) -> bool:
    return isinstance(p, str) and p.startswith(("http://", "https://", "ftp://"))


def _find_local_index_for_remote(remote_path: str, search_dirs=None) -> Optional[str]:
    if not _is_remote_path(remote_path):
        return None
    if search_dirs is None:
        search_dirs = [os.getcwd()]
    base = os.path.basename(urlparse(remote_path).path)
    for ext in (".tbi", ".csi"):
        for d in search_dirs:
            cand = os.path.join(d, base + ext)
            if os.path.exists(cand):
                return cand
    return None


def open_variantfile_with_optional_local_index(vcf_path: str, search_dirs=None) -> pysam.VariantFile:
    idx = _find_local_index_for_remote(vcf_path, search_dirs=search_dirs)
    if idx:
        logging.info(f"Using existing local index for remote VCF: {idx}")
        return pysam.VariantFile(vcf_path, index_filename=idx)
    return pysam.VariantFile(vcf_path)


_CHROM_RE = re.compile(r'\.?chr(\d{1,2}|X|Y)\b')


def _parse_chrom_from_popdb_path(path: str) -> Optional[str]:
    """Extract chromosome token from a PopDB file path using regex."""
    if not isinstance(path, str):
        return None
    m = _CHROM_RE.search(path)
    return m.group(1) if m else None


def _choose_db_contig_name(vcf: pysam.VariantFile, chrom_norm: str) -> Optional[str]:
    contigs = set(vcf.header.contigs)
    for cand in (f"chr{chrom_norm}", chrom_norm):
        if cand in contigs:
            return cand
    return None


def _iter_fetch_with_retry(vcf: pysam.VariantFile, vcf_path: str, contig: str,
                           start0: int, end0: int, max_attempts: int = 3):
    for attempt in range(1, max_attempts + 1):
        try:
            yield from vcf.fetch(contig, start0, end0)
            return
        except Exception as e:
            logging.warning(
                f"[PopDB fetch fail] {vcf_path} {contig}:{start0}-{end0} "
                f"(attempt {attempt}/{max_attempts}): {e}"
            )
            if attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 5))


# ============================================================================
# POPULATION DATABASE MANAGER
# ============================================================================


class PopulationDatabaseManager:
    def __init__(self, config: PipelineConfig, search_dirs: Optional[List[str]] = None):
        self.config = config
        self.population_dbs = config._raw_get('evaluation.external_validation.population_dbs', []) or []
        self.chunk_bp = int(config._raw_get('evaluation.external_validation.popdb_chunk_bp', 1_000_000))
        self.fetch_retries = int(config._raw_get('evaluation.external_validation.popdb_fetch_retries', 3))
        self.search_dirs = search_dirs or [os.getcwd(), config._raw_get('paths.output_dir', 'output')]

    def load_population_variants_for_chromosomes(self, df_variants: pd.DataFrame) -> set:
        if df_variants is None or df_variants.empty:
            logging.warning("PopulationDatabaseManager: input variants are empty.")
            return set()
        if not self.population_dbs:
            logging.info("PopulationDatabaseManager: no population DB VCFs configured.")
            return set()

        df = df_variants.copy()
        df["CHROM_NORM"] = df["CHROM"].astype(str).str.replace("^chr", "", regex=True)
        df["POS"] = pd.to_numeric(df["POS"], errors="coerce").astype("Int64")

        # Vectorised key set
        valid = df[["CHROM_NORM", "POS", "REF", "ALT"]].dropna()
        sample_keys = set(
            zip(valid["CHROM_NORM"].astype(str),
                valid["POS"].astype(int),
                valid["REF"].astype(str),
                valid["ALT"].astype(str))
        )
        if not sample_keys:
            logging.warning("PopulationDatabaseManager: no valid sample keys after normalisation.")
            return set()

        needed_chroms = {k[0] for k in sample_keys} & {str(i) for i in range(1, 23)} | {"X", "Y"}
        max_pos_by_chrom = (
            df.dropna(subset=["CHROM_NORM", "POS"])
            .groupby("CHROM_NORM")["POS"].max().to_dict()
        )

        db_paths = [
            p for p in self.population_dbs
            if (ct := _parse_chrom_from_popdb_path(p)) is None or ct in needed_chroms
        ]
        if not db_paths:
            logging.info("PopulationDatabaseManager: no matching PopDB paths to query.")
            return set()

        found: set = set()
        for vcf_path in db_paths:
            chrom_token = _parse_chrom_from_popdb_path(vcf_path)
            if chrom_token is None or chrom_token not in max_pos_by_chrom:
                continue
            max_pos = int(max_pos_by_chrom[chrom_token])
            if max_pos <= 0:
                continue
            try:
                vcf = open_variantfile_with_optional_local_index(vcf_path, search_dirs=self.search_dirs)
            except Exception as e:
                logging.warning(f"Could not open {vcf_path}: {e}")
                continue
            contig = _choose_db_contig_name(vcf, chrom_token)
            if contig is None:
                logging.warning(f"DB header has no contig for chrom {chrom_token}; skipping {vcf_path}")
                continue
            consecutive_failures = 0
            failed_regions: list = []
            start0 = 0
            vcf_ok = True
            while start0 < max_pos and vcf_ok:
                end0 = min(start0 + self.chunk_bp, max_pos)
                chunk_ok = False
                for attempt in range(1, self.fetch_retries + 1):
                    try:
                        for rec in vcf.fetch(contig, start0, end0):
                            if rec.alts is None:
                                continue
                            c = str(rec.chrom).replace("chr", "")
                            for alt in rec.alts:
                                key = (c, int(rec.pos), str(rec.ref), str(alt))
                                if key in sample_keys:
                                    found.add(key)
                        chunk_ok = True
                        break
                    except Exception as e:
                        logging.warning(
                            f"[PopDB fetch fail] {vcf_path} {contig}:{start0}-{end0}"
                            f" (attempt {attempt}/{self.fetch_retries}): {e}"
                        )
                        if attempt < self.fetch_retries:
                            time.sleep(min(2 ** (attempt - 1), 5))
                if chunk_ok:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    failed_regions.append(f"{contig}:{start0}-{end0}")
                    if consecutive_failures >= 3:
                        logging.warning(
                            f"Aborting {vcf_path} after {consecutive_failures} consecutive "
                            f"chunk failures — file is likely truncated or corrupted. "
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

        logging.info(f"PopulationDatabaseManager: found {len(found)} sample variants in population DBs.")
        return found


# ============================================================================
# CLINVAR HELPERS
# ============================================================================


def load_clinvar_dict(clinvar_vcf: str) -> Dict[Tuple[str, int, str, str], str]:
    logging.info(f"Loading ClinVar VCF: {clinvar_vcf}")
    try:
        vcf = open_variantfile_with_optional_local_index(clinvar_vcf, search_dirs=[os.getcwd()])
    except Exception:
        vcf = pysam.VariantFile(clinvar_vcf)

    has_clnsig = "CLNSIG" in set(vcf.header.info)
    has_clnrev = "CLNREVSTAT" in set(vcf.header.info)
    if not has_clnsig:
        logging.warning("ClinVar VCF has no CLNSIG INFO field; cannot filter.")
        return {}

    def classify_record(rec) -> Optional[str]:
        clnsig = rec.info.get("CLNSIG", [])
        if not isinstance(clnsig, (list, tuple)):
            clnsig = [clnsig]
        sig_str = ",".join(map(str, clnsig)).upper()
        if has_clnrev:
            clnrev = rec.info.get("CLNREVSTAT", [])
            if not isinstance(clnrev, (list, tuple)):
                clnrev = [clnrev]
            rev_str = ",".join(map(str, clnrev)).upper()
            if not any(x in rev_str for x in ["PRACTICE_GUIDELINE", "REVIEWED_BY_EXPERT_PANEL"]):
                return None
        is_path = any(x in sig_str for x in ["PATHOGENIC", "LIKELY_PATHOGENIC"])
        is_benign = any(x in sig_str for x in ["BENIGN", "LIKELY_BENIGN"])
        if is_path and not is_benign:
            return "P"
        if is_benign and not is_path:
            return "B"
        return None

    cv_dict: Dict[Tuple[str, int, str, str], str] = {}
    for rec in vcf:
        label = classify_record(rec)
        if label is None or rec.alts is None:
            continue
        c = str(rec.chrom).replace("chr", "")
        for alt in rec.alts:
            cv_dict[(c, int(rec.pos), str(rec.ref), str(alt))] = label

    logging.info(f"Loaded {len(cv_dict)} high-confidence ClinVar variants (P/B).")
    return cv_dict


# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================


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


def monitor_memory(step_name: str, log_warning: bool = False) -> Tuple[Optional[float], Optional[float]]:
    try:
        import psutil
        info = psutil.Process().memory_info()
        rss_mb = info.rss / 1024 ** 2
        vms_mb = info.vms / 1024 ** 2
        fn = logging.warning if log_warning else logging.info
        fn(f"[Memory{'Warning' if log_warning else ''}] {step_name}: RSS={rss_mb:.1f}MB, VMS={vms_mb:.1f}MB")
        return rss_mb, vms_mb
    except ImportError:
        msg = f"[Memory{'Warning' if log_warning else ''}] {step_name}: psutil not available"
        (logging.warning if log_warning else logging.info)(msg)
        return None, None


def _check_memory_skip(config: PipelineConfig) -> bool:
    """Return True if memory-intensive models should be skipped this fold."""
    if not config._raw_get('memory_safety.monitor_memory', True):
        return False
    rss_mb, _ = monitor_memory("fold memory check")
    if rss_mb is None:  # psutil unavailable — safe default
        return False
    max_mb = config._raw_get('memory_safety.max_memory_mb', 8000)
    if rss_mb > max_mb and config._raw_get('memory_safety.skip_memory_intensive_models', True):
        logging.warning(f"High memory ({rss_mb:.0f}MB > {max_mb}MB): skipping BGM/RF this fold")
        return True
    return False


# ============================================================================
# VCF PARSING HELPERS
# ============================================================================


def extract_annotations(vcf_path: str, desired_keys: List[str], batch_size: int = 100000) -> pd.DataFrame:
    logging.info(f"Reading VCF: {vcf_path}")
    vcf = pysam.VariantFile(vcf_path)
    header_info_keys = set(vcf.header.info)
    usable_keys = [k for k in desired_keys if k in header_info_keys]
    missing_keys = sorted(set(desired_keys) - set(usable_keys))
    if missing_keys:
        logging.warning(f"Skipping INFO tags not in header: {missing_keys}")

    try:
        import psutil
        MEMORY_MONITOR = True
    except ImportError:
        MEMORY_MONITOR = False

    all_batches = []
    batch = []
    for i, rec in enumerate(vcf):
        row: Dict[str, Any] = {
            "CHROM": rec.chrom,
            "POS": rec.pos,
            "REF": str(rec.ref),
            "ALT": str(rec.alts[0]) if rec.alts else "N",
        }
        for key in usable_keys:
            val = rec.info.get(key)
            if val is None:
                row[key] = np.nan
            else:
                if isinstance(val, (list, tuple)) and len(val) == 1:
                    val = val[0]
                try:
                    row[key] = float(val)
                except (TypeError, ValueError):
                    row[key] = np.nan
        for key in missing_keys:
            row[key] = np.nan
        batch.append(row)
        if len(batch) >= batch_size:
            all_batches.append(pd.DataFrame(batch))
            batch = []
            if MEMORY_MONITOR and i % (batch_size * 2) == 0:
                mem_mb = psutil.Process().memory_info().rss / 1024 ** 2
                logging.info(f"Processed {i + 1} variants, memory: {mem_mb:.1f}MB")
            gc.collect()

    if batch:
        all_batches.append(pd.DataFrame(batch))
    df = pd.concat(all_batches, ignore_index=True) if all_batches else pd.DataFrame(
        columns=["CHROM", "POS", "REF", "ALT"] + usable_keys + missing_keys)
    gc.collect()
    logging.info(f"Extracted {len(df)} variants with {len(usable_keys)} features")
    return df


def extract_truth_positions(vcf_path: str) -> set:
    logging.info(f"Extracting truth positions from: {vcf_path}")
    truth_set = set()
    for rec in pysam.VariantFile(vcf_path):
        truth_set.add((rec.chrom.replace("chr", ""), rec.pos))
    logging.info(f"Extracted {len(truth_set)} truth positions")
    return truth_set


def label_variants(df: pd.DataFrame, truth_set: set, label_col: str = "TruthLabel") -> pd.DataFrame:
    """Label each variant as 1 if (chrom_norm, pos) is in truth_set, else 0 — vectorised."""
    df = df.copy()
    chrom_norm = df["CHROM"].astype(str).str.replace("^chr", "", regex=True)
    pos = pd.to_numeric(df["POS"], errors="coerce").fillna(-1).astype(int)
    df[label_col] = [1 if (c, p) in truth_set else 0
                     for c, p in zip(chrom_norm, pos)]
    return df


def label_variants_with_external_set(df: pd.DataFrame, external_variant_set: set,
                                     label_col: str = "IsExternalVariant") -> pd.DataFrame:
    """Label each variant as 1 if (chrom_norm, pos, ref, alt) in external_set — vectorised."""
    df = df.copy()
    chrom_norm = df["CHROM"].astype(str).str.replace("^chr", "", regex=True)
    pos = pd.to_numeric(df["POS"], errors="coerce").fillna(-1).astype(int)
    ref = df["REF"].astype(str)
    alt = df["ALT"].astype(str)
    df[label_col] = [1 if (c, p, r, a) in external_variant_set else 0
                     for c, p, r, a in zip(chrom_norm, pos, ref, alt)]
    n = int(df[label_col].sum())
    logging.info(f"Labelled variants: {n} external, {len(df) - n} novel")
    return df


def build_genomic_groups(df: pd.DataFrame, block_size_bp: int = 1_000_000) -> Optional[np.ndarray]:
    if df is None or df.empty:
        return None
    chrom_norm = df["CHROM"].astype(str).str.replace("^chr", "", regex=True)
    pos = pd.to_numeric(df["POS"], errors="coerce").fillna(-1).astype(int)
    block = pos // int(block_size_bp)
    return (chrom_norm + ":" + block.astype(str)).to_numpy()


def make_block_splitter(n_splits: int, random_state: int = 42):
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state), "StratifiedGroupKFold"
    except ImportError:
        return GroupKFold(n_splits=n_splits), "GroupKFold"


# ============================================================================
# LIGHTGBM HELPERS
# ============================================================================


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
                gm_g = GaussianMixture(n_components=config._raw_get('models.n_components', 5),
                                       max_iter=config._raw_get('models.max_iter_gm', 500),
                                       random_state=int(random_state)).fit(Xtr[ytr == 1])
                gm_b = GaussianMixture(n_components=config._raw_get('models.n_components', 5),
                                       max_iter=config._raw_get('models.max_iter_gm', 500),
                                       random_state=int(random_state)).fit(Xtr[ytr == 0])
                lgb_inner = LGBMClassifier(**dict(lgb_params)).fit(Xtr, ytr)
                hs = compute_hybrid_scores(gm_g, gm_b, lgb_inner, float(alpha), Xval)
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

        # ---- GM ----
        gm_good_cv = GaussianMixture(
            n_components=config._raw_get('models.n_components', 5),
            max_iter=config._raw_get('models.max_iter_gm', 500),
            random_state=int(random_state) + fold,
        ).fit(X_train[y_train == 1])
        gm_bad_cv = GaussianMixture(
            n_components=config._raw_get('models.n_components', 5),
            max_iter=config._raw_get('models.max_iter_gm', 500),
            random_state=int(random_state) + fold,
        ).fit(X_train[y_train == 0])
        gm_scores = compute_vqslod_prob(gm_good_cv, gm_bad_cv, X_val)  # sigmoid-normalised
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
                ).fit(X_train[y_train == 1])
                bgm_b = BayesianGaussianMixture(
                    n_components=config._raw_get('models.n_components', 5),
                    max_iter=config._raw_get('models.max_iter_bgm', 1000),
                    random_state=int(random_state) + fold,
                    weight_concentration_prior=1e-2,
                ).fit(X_train[y_train == 0])
                bgm_scores = compute_vqslod_prob(bgm_g, bgm_b, X_val)  # sigmoid-normalised
                metrics_all["BGM"].append(_metric_dict_for_model("BGM", y_val, bgm_scores))
                oof_scores["BGM"][val_idx] = bgm_scores
            except Exception as e:
                logging.warning(f"BGM fold {fold} failed: {e}")
                metrics_all["BGM"].append(_nan_metric_dict())
        else:
            metrics_all["BGM"].append(_nan_metric_dict())

        # ---- LR ----
        lr_cv = LogisticRegression(max_iter=1000, random_state=int(random_state) + fold).fit(X_train, y_train)
        lr_scores = lr_cv.predict_proba(X_val)[:, 1]
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
                                              float(best_alpha), X_val)
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
                                               lgb_models["LGB"], fa_alpha, X_val)
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

    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)

    gm_good = GaussianMixture(n_components=config._raw_get('models.n_components', 5),
                               max_iter=config._raw_get('models.max_iter_gm', 500),
                               random_state=rs).fit(X_temp_imp[y_temp == 1])
    gm_bad = GaussianMixture(n_components=config._raw_get('models.n_components', 5),
                              max_iter=config._raw_get('models.max_iter_gm', 500),
                              random_state=rs).fit(X_temp_imp[y_temp == 0])

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
                                               ).fit(X_temp_imp[y_temp == 1])
            bgm_bad = BayesianGaussianMixture(n_components=config._raw_get('models.n_components', 5),
                                              max_iter=config._raw_get('models.max_iter_bgm', 1000),
                                              random_state=rs, weight_concentration_prior=1e-2,
                                              ).fit(X_temp_imp[y_temp == 0])
        except Exception as e:
            logging.warning(f"BGM training skipped: {e}")

    lr_model = LogisticRegression(max_iter=1000, random_state=rs).fit(X_temp_imp, y_temp)
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
            val_scores = compute_hybrid_scores(gm_good, gm_bad, lgb_models[base_name], best_alpha, X_val_imp)
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

    final_models: Dict[str, Any] = {}
    final_models['GM'] = (
        GaussianMixture(**gm_good.get_params()).fit(X_full[y_all == 1]),
        GaussianMixture(**gm_bad.get_params()).fit(X_full[y_all == 0]),
    )
    if bgm_good is not None and bgm_bad is not None:
        final_models['BGM'] = (
            BayesianGaussianMixture(**bgm_good.get_params()).fit(X_full[y_all == 1]),
            BayesianGaussianMixture(**bgm_bad.get_params()).fit(X_full[y_all == 0]),
        )
    else:
        final_models['BGM'] = (None, None)
        logging.warning("BGM not included in final models due to memory constraints.")

    final_models['LogReg'] = LogisticRegression(**lr_model.get_params()).fit(X_full, y_all)
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
                                         fa_alpha, X_val_imp)))
                           if len(np.unique(y_val)) > 1 else np.nan,
                'alpha_cv_scores': [],
            }
            logging.info(f"{fa_name}: alpha={fa_alpha:.2f} (fixed)")

    save_all_models(final_models, output_dir, imputer)

    with open(os.path.join(output_dir, 'models', 'lightgbm_tuning_summary.json'), 'w') as f:
        json.dump(lgb_tuning_info, f, indent=2)
    with open(os.path.join(output_dir, 'models', 'hybrid_tuning_summary.json'), 'w') as f:
        json.dump(hybrid_summary, f, indent=2)

    if config._raw_get('memory_safety.monitor_memory', True):
        monitor_memory("After final model training")

    logging.info("Final models trained and saved.")
    return final_models


def save_all_models(models: Dict, output_dir: str, imputer: LeakageAwareImputer):
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
    metadata = {'model_paths': model_paths, 'imputer_path': imputer_path,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(model_dir, "model_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"All models saved to {model_dir}")


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================


def generate_tranche_analysis(models, X, y, output_dir, config):
    logging.info("Generating tranche analysis...")
    tranche_sensitivities = [1.0, 0.999, 0.99, 0.90]
    records = []
    for name, model in models.items():
        try:
            scores = score_model_instance(name, model, X)
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
                scores = score_model_instance(name, model, X)
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


def run_variant_classification_counts(df_variants, X, y, models, output_dir, target_sens=0.99):
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
            scores = score_model_instance(name, model, X)
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


def run_comprehensive_external_validation(models, df_variants, feature_keys, output_dir, config):
    logging.info("Running comprehensive external validation with population databases...")
    popdb_manager = PopulationDatabaseManager(config)
    external_set = popdb_manager.load_population_variants_for_chromosomes(df_variants)
    if not external_set:
        logging.warning("No external variants loaded for validation")
        return None

    df_labeled = label_variants_with_external_set(df_variants, external_set, "IsExternalVariant")
    X = df_labeled[feature_keys].to_numpy(dtype=np.float32)
    y = df_labeled["IsExternalVariant"].to_numpy(dtype=int)

    results = []
    detailed = df_labeled[["CHROM", "POS", "REF", "ALT", "IsExternalVariant"]].copy()
    for name, model in models.items():
        try:
            scores = score_model_instance(name, model, X)
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
            scores_df[f"{model_name}_score"] = score_model_instance(model_name, model_obj, X_imputed)
        except Exception as e:
            logging.error(f"Failed to score with model {model_name}: {e}")
            scores_df[f"{model_name}_score"] = np.nan

    scores_csv = os.path.join(output_dir, "real_data_scores.csv")
    scores_df.to_csv(scores_csv, index=False)
    logging.info(f"Real data scores saved to {scores_csv}")
    return scores_df


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
    final_models = train_final_models_fixed(X, y, feature_keys, config, output_dir)

    logging.info("Step 5: Generating evaluations...")
    generate_tranche_analysis(final_models, X, y, output_dir, config)
    run_variant_classification_counts(
        df_labeled, X, y, final_models, output_dir,
        target_sens=config._raw_get('evaluation.variant_classification_target_sens', 0.99),
    )

    if config._raw_get('evaluation.external_validation.enabled', False):
        logging.info("Step 6: Running comprehensive external validation...")
        run_comprehensive_external_validation(final_models, df_labeled, feature_keys, output_dir, config)

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


# ============================================================================
# ENTRY POINT
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Variant Quality Scoring Pipeline")
    parser.add_argument("--config", "-c", default="pipeline_config.yaml",
                        help="Path to YAML/JSON config file")
    parser.add_argument("--mode", choices=["train", "apply"],
                        help="Override pipeline mode")
    parser.add_argument("--output-dir", help="Override output directory base name")
    parser.add_argument("--train-vcf", help="Override training VCF path")
    parser.add_argument("--truth-vcf", help="Override truth VCF path")
    parser.add_argument("--apply-vcf", help="Override apply VCF path")
    parser.add_argument("--trained-model-dir",
                        help="Override path to training output dir (apply mode)")
    args = parser.parse_args()

    config = PipelineConfig.from_yaml(args.config) if os.path.exists(args.config) else PipelineConfig()
    if not os.path.exists(args.config):
        # Can't log yet — print directly so the warning isn't silently lost
        print(f"[WARNING] Config file {args.config} not found, using defaults", flush=True)

    if args.mode:
        config.set('mode', args.mode)
    if args.output_dir:
        config.set('paths.output_dir', args.output_dir)
    if args.train_vcf:
        config.set('paths.train_input_vcf', args.train_vcf)
    if args.truth_vcf:
        config.set('paths.truth_vcf', args.truth_vcf)
    if args.apply_vcf:
        config.set('paths.apply_input_vcf', args.apply_vcf)
    if args.trained_model_dir:
        config.set('paths.trained_model_dir', args.trained_model_dir)

    # ── Create output directory ONCE here ──────────────────────────────────────
    # run_train_pipeline / run_apply_pipeline receive this dir and do NOT call
    # initialize_paths() again.  Previously each pipeline called initialize_paths()
    # independently, producing a second timestamped directory whose path differed
    # from the one used for the log file → log file ended up in the wrong folder.
    output_dir = config.initialize_paths()

    # ── Configure logging IMMEDIATELY after output_dir exists ──────────────────
    # All subsequent log calls — including those inside the pipeline — go here.
    log_file = os.path.join(output_dir, "pipeline.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()],
    )

    # Save final resolved config next to the log
    config.to_yaml(os.path.join(output_dir, "final_config.yaml"))
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Pipeline mode: {config._raw_get('mode')}")

    try:
        mode = config._raw_get('mode')
        if mode == 'train':
            run_train_pipeline(config, output_dir)
        elif mode == 'apply':
            run_apply_pipeline(config, output_dir)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
