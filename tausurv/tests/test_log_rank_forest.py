from __future__ import annotations

import numpy as np
import pytest

from tausurv.core import fit_log_rank_forest, fit_log_rank_tree


@pytest.fixture
def two_group_data():
    rng = np.random.default_rng(42)
    n, p = 300, 5
    X = rng.normal(size=(n, p))
    rate = np.where(X[:, 0] > 0, 1.0, 0.2)
    T = rng.exponential(1.0 / rate)
    C = rng.exponential(scale=3.0, size=n)
    event_time = np.minimum(T, C)
    event_indicator = (T <= C).astype(np.int8)
    return X, event_time, event_indicator


def _fit_forest(X, event_time, event_indicator, **kwargs):
    return fit_log_rank_forest(
        X=np.asfortranarray(X.astype(np.float64)),
        event_time=event_time.astype(np.float64),
        event_indicator=event_indicator.astype(np.uint8),
        **kwargs,
    )


def test_fits_with_basic_params(two_group_data):
    """Smoke: forest fits and reports expected metadata (no recovery
    assertion — that lives in scripts/log_rank_tree_parity.py)."""
    X, event_time, event_indicator = two_group_data
    forest = _fit_forest(
        X,
        event_time,
        event_indicator,
        n_trees=20,
        min_samples_leaf=15,
        max_features="sqrt",
        seed=42,
    )
    assert forest.n_trees == 20
    assert forest.n_features == X.shape[1]
    assert forest.n_train_samples == X.shape[0]


def test_deterministic_under_seed(two_group_data):
    X, event_time, event_indicator = two_group_data
    a = _fit_forest(X, event_time, event_indicator, n_trees=10, seed=7)
    b = _fit_forest(X, event_time, event_indicator, n_trees=10, seed=7)
    X_f = np.asfortranarray(X.astype(np.float64))
    times = np.linspace(0.1, 5.0, 8, dtype=np.float64)
    np.testing.assert_array_equal(
        a.predict_cumulative_hazard(X=X_f, times=times),
        b.predict_cumulative_hazard(X=X_f, times=times),
    )


def test_no_resample_no_random_features_matches_single_tree(two_group_data):
    """With bootstrap off, fraction=1.0, max_features='all', honesty off,
    every tree is identical to the single-tree fit on the full sample.
    Forest prediction equals single-tree prediction exactly."""
    X, event_time, event_indicator = two_group_data
    common = dict(min_samples_leaf=15, max_features="all", seed=0)
    forest = _fit_forest(
        X,
        event_time,
        event_indicator,
        n_trees=5,
        bootstrap=False,
        subsample_fraction=1.0,
        **common,
    )
    tree = fit_log_rank_tree(
        X=np.asfortranarray(X.astype(np.float64)),
        event_time=event_time.astype(np.float64),
        event_indicator=event_indicator.astype(np.uint8),
        **common,
    )
    X_f = np.asfortranarray(X.astype(np.float64))
    times = np.linspace(0.1, 5.0, 9, dtype=np.float64)
    H_forest = forest.predict_cumulative_hazard(X=X_f, times=times)
    H_tree = tree.predict_cumulative_hazard(X=X_f, times=times)
    np.testing.assert_allclose(H_forest, H_tree, rtol=1e-12, atol=1e-12)


def test_honesty_propagates_to_forest(two_group_data):
    X, event_time, event_indicator = two_group_data
    non = _fit_forest(X, event_time, event_indicator, n_trees=10, honesty=False, seed=0)
    hon = _fit_forest(X, event_time, event_indicator, n_trees=10, honesty=True, seed=0)
    X_f = np.asfortranarray(X.astype(np.float64))
    times = np.array([1.0, 5.0, 10.0])
    H_non = non.predict_cumulative_hazard(X=X_f, times=times)
    H_hon = hon.predict_cumulative_hazard(X=X_f, times=times)
    assert np.max(np.abs(H_hon - H_non)) > 1e-6


def test_rejects_zero_trees(two_group_data):
    X, event_time, event_indicator = two_group_data
    with pytest.raises(ValueError, match="n_trees"):
        _fit_forest(X, event_time, event_indicator, n_trees=0)


def test_rejects_invalid_subsample_fraction(two_group_data):
    X, event_time, event_indicator = two_group_data
    with pytest.raises(ValueError, match="subsample_fraction"):
        _fit_forest(X, event_time, event_indicator, subsample_fraction=0.0)
    with pytest.raises(ValueError, match="subsample_fraction"):
        _fit_forest(X, event_time, event_indicator, subsample_fraction=1.5)


def test_forest_weights_shape_and_nonnegative(two_group_data):
    X, event_time, event_indicator = two_group_data
    forest = _fit_forest(
        X,
        event_time,
        event_indicator,
        n_trees=10,
        min_samples_leaf=10,
        max_features="all",
        seed=0,
    )
    X_f = np.asfortranarray(X.astype(np.float64))
    W = forest.forest_weights(X_f)
    assert W.shape == (X.shape[0], forest.n_train_samples)
    assert (W >= 0).all()


def test_forest_weights_sum_to_one_in_no_subsample_mode(two_group_data):
    """Without subsampling or honesty, every training sample is in every
    tree's estimation set and every tree's target leaf is non-empty, so
    weights sum to exactly 1.0 per query."""
    X, event_time, event_indicator = two_group_data
    forest = _fit_forest(
        X,
        event_time,
        event_indicator,
        n_trees=10,
        min_samples_leaf=10,
        max_features="all",
        bootstrap=False,
        subsample_fraction=1.0,
        honesty=False,
        seed=0,
    )
    X_f = np.asfortranarray(X.astype(np.float64))
    W = forest.forest_weights(X_f)
    np.testing.assert_allclose(W.sum(axis=1), 1.0, rtol=1e-12, atol=1e-12)


def test_forest_weights_one_matches_batch_row(two_group_data):
    X, event_time, event_indicator = two_group_data
    forest = _fit_forest(
        X,
        event_time,
        event_indicator,
        n_trees=8,
        min_samples_leaf=10,
        seed=3,
    )
    X_f = np.asfortranarray(X.astype(np.float64))
    W_batch = forest.forest_weights(X_f)
    for i in (0, 50, 150):
        w = forest.forest_weights_one(X_f[i].copy())
        np.testing.assert_array_equal(W_batch[i], w)


def test_forest_weights_self_weight_positive_under_no_subsample(two_group_data):
    """Without subsampling, training sample i is in every tree's
    estimation set; querying at X[i] must place positive weight on i."""
    X, event_time, event_indicator = two_group_data
    forest = _fit_forest(
        X,
        event_time,
        event_indicator,
        n_trees=5,
        min_samples_leaf=10,
        max_features="all",
        bootstrap=False,
        subsample_fraction=1.0,
        honesty=False,
        seed=0,
    )
    X_f = np.asfortranarray(X.astype(np.float64))
    for i in (0, 100, 200):
        w = forest.forest_weights_one(X_f[i].copy())
        assert w[i] > 0, f"self-weight for sample {i} should be positive, got {w[i]}"
