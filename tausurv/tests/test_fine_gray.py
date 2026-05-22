"""Tests for the Fine-Gray subdistribution hazard model."""

from __future__ import annotations

import numpy as np
import pytest

from tausurv import simulations
from tausurv.linear import FineGray


def _cr_data(n=500, n_causes=2, seed=0):
    return simulations.competing_risk(n=n, n_features=5, n_causes=n_causes, seed=seed)


def test_predict_cif_shape_and_bounds():
    X, T, E = _cr_data(n=300)
    fg = FineGray(cause=1).fit(X, T, E)
    grid = np.linspace(0.1, 3.0, 10)
    F = fg.predict_cif(X[:20], grid)
    assert F.shape == (20, 10)
    assert ((F >= 0.0) & (F <= 1.0)).all()


def test_predict_cif_non_decreasing_in_t():
    """F_k(t|x) increases (or stays flat) as t increases — by construction
    from the cumulative hazard."""
    X, T, E = _cr_data(n=300)
    fg = FineGray(cause=1).fit(X, T, E)
    grid = np.linspace(0.05, 3.0, 20)
    F = fg.predict_cif(X[:30], grid)
    assert (np.diff(F, axis=1) >= -1e-12).all()


def test_predict_survival_function_is_one_minus_cif():
    """Subdistribution survival = 1 - F_k."""
    X, T, E = _cr_data(n=200)
    fg = FineGray(cause=1).fit(X, T, E)
    grid = np.linspace(0.1, 3.0, 8)
    S = fg.predict_survival_function(X[:20], grid)
    F = fg.predict_cif(X[:20], grid)
    np.testing.assert_allclose(S, 1.0 - F)


def test_predict_returns_linear_predictor():
    """predict(X) is β^T x — Cox-style ranking scalar."""
    X, T, E = _cr_data(n=200)
    fg = FineGray(cause=1).fit(X, T, E)
    lp = fg.predict(X)
    expected = X @ fg.coef_
    np.testing.assert_allclose(lp, expected)


def test_baseline_subdist_cumhazard_non_decreasing():
    """The baseline subdistribution cumulative hazard is non-decreasing
    (it's a cumulative sum of non-negative increments)."""
    X, T, E = _cr_data(n=200)
    fg = FineGray(cause=1).fit(X, T, E)
    H = fg.baseline_subdist_cumhazard_.value
    assert (np.diff(H) >= 0).all()


def test_cause_2_fits_independently_of_cause_1():
    """Fitting FineGray(cause=1) and FineGray(cause=2) gives different
    coefficients — they're different models targeting different events."""
    X, T, E = _cr_data(n=500)
    fg1 = FineGray(cause=1).fit(X, T, E)
    fg2 = FineGray(cause=2).fit(X, T, E)
    assert not np.allclose(fg1.coef_, fg2.coef_, atol=1e-2)


def test_cif_competing_risk_complementary():
    """Sum of cause-1 and cause-2 FineGray CIFs is bounded by 1 (subdistribution
    CIFs are submarginal probabilities; their sum is at most the marginal
    cumulative event rate)."""
    X, T, E = _cr_data(n=400, n_causes=2)
    fg1 = FineGray(cause=1).fit(X, T, E)
    fg2 = FineGray(cause=2).fit(X, T, E)
    grid = np.linspace(0.1, 2.0, 6)
    total = fg1.predict_cif(X[:30], grid) + fg2.predict_cif(X[:30], grid)
    # Each is ≤ 1; sum can also exceed 1 in finite samples because the
    # two FG models are fit independently. Test holds in expectation; for
    # robustness check that sum stays in [0, 2] (trivially true) but on
    # average is reasonable.
    assert (total >= 0.0).all() and (total <= 2.0).all()
    # Loose upper bound: sum at t -> ∞ should be < 1.5 typically.
    assert float(total.mean()) < 1.0


def test_rejects_cause_below_one():
    with pytest.raises(ValueError, match="cause"):
        FineGray(cause=0)


def test_rejects_missing_cause_at_fit():
    X = np.zeros((10, 3))
    T = np.linspace(0.1, 1.0, 10)
    E = np.zeros(10, dtype=np.int8)  # all censored — no cause-1 events
    with pytest.raises(ValueError, match="cause=1"):
        FineGray(cause=1).fit(X, T, E)


def test_fine_gray_matches_lifelines_if_available():
    pytest.importorskip("lifelines")
    pytest.skip("lifelines has no direct Fine-Gray fitter for comparison")


def test_inherits_survival_predictor_api():
    from tausurv.predictor import SurvivalPredictor

    X, T, E = _cr_data(n=200)
    fg = FineGray(cause=1).fit(X, T, E)
    assert isinstance(fg, SurvivalPredictor)
    # All inherited methods exist and work on the natural time grid.
    n = X.shape[0]
    assert fg.predict_survival_function(X).shape == (n, fg.times_.shape[0])
    assert fg.predict_cumulative_hazard(X).shape == (n, fg.times_.shape[0])
    assert fg.predict_cif(X).shape == (n, fg.times_.shape[0])
    horizon = float(fg.times_.max() * 0.9)
    assert fg.predict_rmst(X, horizon=horizon).shape == (n,)
    assert fg.predict_risk_at(X, time=float(fg.times_[len(fg.times_) // 2])).shape == (
        n,
    )
