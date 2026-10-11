"""Unit tests for the statistical and labelling building blocks."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

from postfilter_pipeline.data_prep import build_genomic_groups, label_variants
from postfilter_pipeline.metrics import find_threshold_at_sensitivity
from postfilter_pipeline.models import (LeakageAwareImputer, LeakageAwareScaler,
                                        compute_hybrid_scores, sigmoid)
from postfilter_pipeline.stats import delong_roc_test, holm_bonferroni


def test_labelling_is_allele_aware():
    truth = {("1", 100, "A", "G"), ("2", 200, "C", "T")}
    df = pd.DataFrame({"CHROM": ["chr1", "1", "2"], "POS": [100, 100, 200],
                       "REF": ["A", "A", "C"], "ALT": ["G", "T", "T"]})
    out = label_variants(df, truth)
    # same position, different allele -> not a true call
    assert out["TruthLabel"].tolist() == [1, 0, 1]


def test_genomic_blocks_group_neighbouring_calls():
    df = pd.DataFrame({"CHROM": ["chr1", "1", "1", "2"],
                       "POS": [10, 999_999, 1_000_001, 10]})
    g = build_genomic_groups(df, block_size_bp=1_000_000)
    assert g[0] == g[1] and g[1] != g[2] and g[0] != g[3]


def test_imputer_and_scaler_use_training_statistics_only(tmp_path):
    Xtr = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 8.0]])
    Xte = np.array([[100.0, np.nan]])
    imp = LeakageAwareImputer().fit(Xtr)
    assert imp.transform(Xte)[0, 1] == pytest.approx(6.0)  # training mean
    sc = LeakageAwareScaler().fit(imp.transform(Xtr))
    z = sc.transform(imp.transform(Xtr))
    assert np.allclose(z.mean(axis=0), 0) and np.allclose(z.std(axis=0), 1)
    sc.save(tmp_path / "s.joblib")
    Xte_imp = imp.transform(Xte)
    assert np.allclose(LeakageAwareScaler.load(tmp_path / "s.joblib").transform(Xte_imp),
                       sc.transform(Xte_imp))
    with pytest.raises(ValueError):
        LeakageAwareScaler().transform(Xte)


def test_delong_matches_sklearn_auc_and_detects_differences():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    good = y + rng.normal(0, 0.8, y.size)
    weak = y + rng.normal(0, 2.5, y.size)
    r = delong_roc_test(y, good, weak)
    assert r["auc1"] == pytest.approx(roc_auc_score(y, good))
    assert r["auc2"] == pytest.approx(roc_auc_score(y, weak))
    assert r["p_value"] < 1e-10
    same = delong_roc_test(y, good, good + 1e-12 * rng.normal(size=y.size))
    assert np.isnan(same["p_value"]) or same["p_value"] > 0.05


def test_holm_adjustment():
    adj = holm_bonferroni([0.01, 0.04, 0.03, np.nan])
    assert adj[:3] == pytest.approx([0.03, 0.06, 0.06])
    assert np.isnan(adj[3])


def test_threshold_at_sensitivity_keeps_target_fraction_of_positives():
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(1000), np.zeros(1000)].astype(int)
    s = rng.random(2000)
    for target in (0.9, 0.99, 1.0):
        t = find_threshold_at_sensitivity(y, s, target)
        assert (s[y == 1] >= t).mean() >= target


def test_hybrid_is_convex_blend_of_lightgbm_and_gm():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 3))

    class FixedClf:
        def predict_proba(self, X):
            p = np.full(len(X), 0.9)
            return np.c_[1 - p, p]

    g = GaussianMixture(2, random_state=0).fit(X[:150])
    b = GaussianMixture(2, random_state=0).fit(X[150:] + 1)
    gm = sigmoid(g.score_samples(X) - b.score_samples(X))
    for alpha in (0.0, 0.7, 0.8, 1.0):
        h = compute_hybrid_scores(g, b, FixedClf(), alpha, X, X)
        assert np.allclose(h, alpha * 0.9 + (1 - alpha) * gm)
