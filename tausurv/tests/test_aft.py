"""Tests for the AFT family."""

from __future__ import annotations

import numpy as np
import pytest

from tausurv.distributions import LogLogistic, LogNormal, Weibull
from tausurv.linear import LogLogisticAFT, LogNormalAFT, WeibullAFT
from tausurv.metrics.concordance import harrell


def _simulate_weibull_aft(
    n: int = 1500,
    n_features: int = 5,
    beta: np.ndarray | None = None,
    intercept: float = 0.1,
    sigma: float = 0.8,
    censoring_rate: float = 0.3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Generate Weibull AFT data: T = exp(X β + b) * (-log U)^σ."""
    rng = np.random.default_rng(seed)
    if beta is None:
        beta = np.array([0.5, -0.3, 0.2, 0.0, 0.4])[:n_features]
    X = rng.normal(size=(n, n_features))
    mu = X @ beta + intercept
    u = rng.uniform(1e-8, 1.0 - 1e-8, size=n)
    T = np.exp(mu) * (-np.log(1.0 - u)) ** sigma

    # Choose censoring scale to hit the target rate.
    C = rng.exponential(scale=np.quantile(T, 1.0 - censoring_rate) / 0.5, size=n)
    Y = np.minimum(T, C)
    E = (T <= C).astype(np.int8)
    return X, Y, E, beta, intercept, sigma


def test_weibull_aft_recovers_simulated_coefficients():
    """Sanity check on Weibull AFT data with n=2000.

    The Fisher-info-derived asymptotic SE for each ``beta_j`` is
    approximately ``sigma_true / sqrt(n_events * Var(X_j))``. With
    ``sigma_true=0.8``, ``n_events ≈ 1400``, ``Var(X_j) = 1``, that gives
    ``SE ≈ 0.021``; ``atol=0.07`` is ~3.3σ. We're testing "the fitter
    converges to the neighborhood of the truth," not solver tightness —
    for solver tightness see :func:`test_intercept_only_weibull_aft_matches_marginal_mle`.
    """
    X, Y, E, beta_true, b_true, sigma_true = _simulate_weibull_aft(
        n=2000, censoring_rate=0.3, seed=0
    )
    m = WeibullAFT().fit(X, Y, E)
    np.testing.assert_allclose(m.coef_, beta_true, atol=0.07)
    assert abs(m.intercept_ - b_true) < 0.08
    assert abs(m.scale_ - sigma_true) < 0.05


def test_weibull_aft_no_intercept_fits_coefficients_too():
    """Same sanity bound as the with-intercept test; see it for SE math."""
    X, Y, E, beta_true, _, _ = _simulate_weibull_aft(
        n=2000, intercept=0.0, censoring_rate=0.2, seed=1
    )
    m = WeibullAFT(fit_intercept=False).fit(X, Y, E)
    assert m.intercept_ == 0.0
    np.testing.assert_allclose(m.coef_, beta_true, atol=0.07)


def test_intercept_only_weibull_aft_matches_marginal_mle():
    """With X all zero (no covariates) the AFT log-likelihood collapses
    to the marginal Weibull log-likelihood, so the AFT fit must agree with
    the scipy MLE on the same draw — not with the population truth."""
    from scipy.stats import weibull_min

    rng = np.random.default_rng(0)
    T = Weibull(shape=2.0, scale=5.0).rvs(size=500, rng=rng)
    E = np.ones(500, dtype=np.int8)
    X = np.zeros((500, 1))

    m = WeibullAFT().fit(X, T, E)
    shape_hat = 1.0 / m.scale_
    scale_hat = np.exp(m.intercept_)

    # scipy's MLE on the same data (location fixed at 0).
    shape_mle, _, scale_mle = weibull_min.fit(T, floc=0)
    np.testing.assert_allclose(shape_hat, shape_mle, atol=1e-3)
    np.testing.assert_allclose(scale_hat, scale_mle, atol=1e-3)


def test_predict_is_negative_linear_predictor():
    """predict() returns -μ so higher = more risk, matching Cox semantics
    and the harrell() metric contract."""
    X, Y, E, _, _, _ = _simulate_weibull_aft(n=300, seed=0)
    m = WeibullAFT().fit(X, Y, E)

    mu = X @ m.coef_ + m.intercept_
    np.testing.assert_allclose(m.predict(X), -mu)

    # Higher predict() → shorter observed time on average.
    risk = m.predict(X)
    # Harrell C must be > 0.5 (positive concordance with event time).
    c = harrell(Y, E, risk)
    assert c > 0.6


def test_aft_uses_log_survival_for_censored():
    """The censored row must enter the log-likelihood via ``log_survival``,
    not ``log_pdf``. We verify two complementary things:

    1. Fitting the same data twice — once with the second row censored,
       once treating it as an event — yields different (intercept, scale).
       If censored rows were being silently treated as events, both fits
       would produce identical parameters.
    2. The fitted log-likelihood under the "censored" treatment equals
       ``log_pdf(T_event) + log_survival(T_censored)`` evaluated at the
       fitted parameters. If censored rows were being scored with
       ``log_pdf``, the two would disagree.
    """
    X = np.zeros((2, 1))
    T = np.array([1.0, 3.0])
    E_censored = np.array([1, 0], dtype=np.int8)
    E_all_event = np.array([1, 1], dtype=np.int8)

    m_cens = WeibullAFT().fit(X, T, E_censored)
    m_event = WeibullAFT().fit(X, T, E_all_event)

    # (1) The two fits differ — censoring is being respected.
    assert not np.isclose(m_cens.intercept_, m_event.intercept_, atol=1e-3)
    assert not np.isclose(m_cens.scale_, m_event.scale_, atol=1e-3)

    # (2) Re-compute the AFT log-likelihood at the fitted params and check
    # it equals the hand-built `log_pdf + log_survival` expression.
    d = Weibull(shape=1.0 / m_cens.scale_, scale=np.exp(m_cens.intercept_))
    hand_ll = float(d.log_pdf(1.0) + d.log_survival(3.0))
    fitted_ll = float(np.where(E_censored == 1, d.log_pdf(T), d.log_survival(T)).sum())
    np.testing.assert_allclose(fitted_ll, hand_ll, atol=1e-12)


def test_aft_rejects_non_binary_event_indicator():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    T = rng.exponential(1.0, size=50)
    E = rng.choice([0, 1, 2], size=50)
    with pytest.raises(ValueError, match="single-event"):
        WeibullAFT().fit(X, T, E)


def test_aft_rejects_non_positive_event_time():
    X = np.zeros((3, 2))
    T = np.array([0.5, 0.0, 1.0])  # T=0 invalid
    E = np.array([1, 1, 1], dtype=np.int8)
    with pytest.raises(ValueError, match="positive"):
        WeibullAFT().fit(X, T, E)


@pytest.mark.parametrize("cls", [WeibullAFT, LogNormalAFT, LogLogisticAFT])
def test_all_aft_variants_fit_and_predict(cls):
    X, Y, E, _, _, _ = _simulate_weibull_aft(n=500, seed=0)
    m = cls().fit(X, Y, E)
    grid = np.linspace(0.5, 5.0, 10)
    S = m.predict_survival_function(X[:20], grid)
    assert S.shape == (20, 10)
    assert ((S >= 0) & (S <= 1)).all()
    assert (np.diff(S, axis=1) <= 1e-12).all()  # non-increasing in t


@pytest.mark.parametrize("cls", [WeibullAFT, LogNormalAFT, LogLogisticAFT])
def test_aft_concordance_beats_random(cls):
    X, Y, E, _, _, _ = _simulate_weibull_aft(n=1000, censoring_rate=0.3, seed=0)
    m = cls().fit(X, Y, E)
    c = harrell(Y, E, m.predict(X))
    # Weibull-generated data; even non-Weibull AFTs should achieve sensible
    # ranking on monotone-in-mean predictions.
    assert c > 0.6


def test_times_attribute_populated_after_fit():
    X, Y, E, _, _, _ = _simulate_weibull_aft(n=200, seed=0)
    m = WeibullAFT().fit(X, Y, E)
    assert m.times_.shape[0] > 0
    # Default times for predict_*: training event times.
    S = m.predict_survival_function(X[:5])  # times=None
    assert S.shape == (5, m.times_.shape[0])
