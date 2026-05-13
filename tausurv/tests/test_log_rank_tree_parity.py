from __future__ import annotations

import numpy as np
import pytest

from tausurv.core import fit_log_rank_tree
from tausurv.trees.survival_tree import SurvivalTree


@pytest.fixture
def two_group_data():
    """Continuous covariates, group label hidden in feature 0 drives hazard.

    No exact duplicates in feature 0, so log-rank tie-breaking is
    well-defined; both implementations must pick the same splits.
    """
    rng = np.random.default_rng(42)
    n, p = 200, 4
    X = rng.normal(size=(n, p))
    rate = np.where(X[:, 0] > 0, 1.0, 0.2)
    T = rng.exponential(1.0 / rate)
    C = rng.exponential(scale=3.0, size=n)
    event_time = np.minimum(T, C)
    event_indicator = (T <= C).astype(np.int8)
    return X, event_time, event_indicator


def _fit_rust(X, event_time, event_indicator, **kwargs):
    return fit_log_rank_tree(
        X=np.asfortranarray(X.astype(np.float64)),
        event_time=event_time.astype(np.float64),
        event_indicator=event_indicator.astype(np.uint8),
        **kwargs,
    )


def test_predicted_cumulative_hazard_matches_reference(two_group_data):
    X, event_time, event_indicator = two_group_data
    times = np.quantile(event_time[event_indicator == 1], np.linspace(0.1, 0.9, 9))

    reference = SurvivalTree(
        min_samples_leaf=15, max_features=None, seed=0
    ).fit(X, event_time, event_indicator)
    rust = _fit_rust(
        X, event_time, event_indicator,
        min_samples_leaf=15, max_features="all", seed=0,
    )

    H_ref = reference.predict_cumulative_hazard(X, times=times)
    H_rust = rust.predict_cumulative_hazard(
        X=np.asfortranarray(X.astype(np.float64)),
        times=times.astype(np.float64),
    )
    np.testing.assert_allclose(H_rust, H_ref, rtol=1e-12, atol=1e-12)


def test_tree_structure_matches_reference(two_group_data):
    X, event_time, event_indicator = two_group_data
    reference = SurvivalTree(
        min_samples_leaf=15, max_features=None, seed=0
    ).fit(X, event_time, event_indicator)
    rust = _fit_rust(
        X, event_time, event_indicator,
        min_samples_leaf=15, max_features="all", seed=0,
    )

    def count_leaves(node):
        if node.is_leaf:
            return 1
        return count_leaves(node.left) + count_leaves(node.right)

    assert rust.n_leaves == count_leaves(reference._root)


def test_deterministic_under_seed(two_group_data):
    X, event_time, event_indicator = two_group_data
    a = _fit_rust(X, event_time, event_indicator, seed=7)
    b = _fit_rust(X, event_time, event_indicator, seed=7)
    times = np.linspace(0.1, 5.0, 8, dtype=np.float64)
    X_f = np.asfortranarray(X.astype(np.float64))
    np.testing.assert_array_equal(
        a.predict_cumulative_hazard(X=X_f, times=times),
        b.predict_cumulative_hazard(X=X_f, times=times),
    )


def test_rejects_non_fortran_input(two_group_data):
    X, event_time, event_indicator = two_group_data
    X_c = np.ascontiguousarray(X.astype(np.float64))  # C-contiguous, not F
    with pytest.raises(ValueError, match="Fortran-contiguous"):
        fit_log_rank_tree(
            X=X_c,
            event_time=event_time.astype(np.float64),
            event_indicator=event_indicator.astype(np.uint8),
        )


def test_rejects_shape_mismatch(two_group_data):
    X, event_time, event_indicator = two_group_data
    with pytest.raises(ValueError, match="rows"):
        fit_log_rank_tree(
            X=np.asfortranarray(X.astype(np.float64)),
            event_time=event_time[:50].astype(np.float64),
            event_indicator=event_indicator.astype(np.uint8),
        )


def test_honest_mode_separates_groups(two_group_data):
    X, event_time, event_indicator = two_group_data
    tree = _fit_rust(
        X, event_time, event_indicator,
        min_samples_leaf=15, honesty=True, honesty_fraction=0.5, seed=42,
    )
    X_f = np.asfortranarray(X.astype(np.float64))
    H = tree.predict_cumulative_hazard(X=X_f, times=np.array([5.0]))
    group_a = X[:, 0] > 0  # higher hazard
    group_b = ~group_a
    assert H[group_a, 0].mean() > H[group_b, 0].mean()


def test_honest_mode_is_deterministic(two_group_data):
    X, event_time, event_indicator = two_group_data
    a = _fit_rust(X, event_time, event_indicator, honesty=True, seed=7)
    b = _fit_rust(X, event_time, event_indicator, honesty=True, seed=7)
    X_f = np.asfortranarray(X.astype(np.float64))
    times = np.linspace(0.1, 5.0, 8, dtype=np.float64)
    np.testing.assert_array_equal(
        a.predict_cumulative_hazard(X=X_f, times=times),
        b.predict_cumulative_hazard(X=X_f, times=times),
    )


def test_honest_differs_from_non_honest(two_group_data):
    X, event_time, event_indicator = two_group_data
    non = _fit_rust(X, event_time, event_indicator, honesty=False, seed=7)
    hon = _fit_rust(X, event_time, event_indicator, honesty=True, seed=7)
    X_f = np.asfortranarray(X.astype(np.float64))
    times = np.linspace(0.1, 5.0, 8, dtype=np.float64)
    H_non = non.predict_cumulative_hazard(X=X_f, times=times)
    H_hon = hon.predict_cumulative_hazard(X=X_f, times=times)
    # Disjoint estimation set ≠ identical predictions to non-honest.
    assert np.max(np.abs(H_hon - H_non)) > 1e-6


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_rejects_invalid_honesty_fraction(two_group_data, bad):
    X, event_time, event_indicator = two_group_data
    with pytest.raises(ValueError, match="honesty_fraction"):
        _fit_rust(
            X, event_time, event_indicator,
            honesty=True, honesty_fraction=bad,
        )
