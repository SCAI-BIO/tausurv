from __future__ import annotations

import numpy as np
import pytest

from tausurv.core import fit_log_rank_tree


@pytest.fixture
def two_group_data():
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
    # The honesty flag must actually do something.
    assert np.max(np.abs(H_hon - H_non)) > 1e-6


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_rejects_invalid_honesty_fraction(two_group_data, bad):
    X, event_time, event_indicator = two_group_data
    with pytest.raises(ValueError, match="honesty_fraction"):
        _fit_rust(
            X,
            event_time,
            event_indicator,
            honesty=True,
            honesty_fraction=bad,
        )
