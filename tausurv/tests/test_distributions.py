"""Tests for the numpy/scipy-backed survival distributions.

Covers:

- Identity relationships: ``cdf + survival = 1``, ``pdf = exp(log_pdf)``,
  ``hazard = pdf / survival``.
- Quantile / CDF round-trip.
- scipy parity for the four distributions scipy implements
  (``weibull_min``, ``lognorm``, ``fisk``, ``expon``).
- Hand-checked closed forms for Gompertz (no scipy parity).
- Sampling: ``rvs`` produces values whose empirical survival matches the
  closed-form.
- Validation: shape/scale/rate must be > 0; sigma > 0.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import expon, fisk, lognorm, weibull_min

from tausurv.distributions import (
    Exponential,
    Gompertz,
    LogLogistic,
    LogNormal,
    Weibull,
)

# Reused grid of evaluation points.
T = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
P = np.array([0.05, 0.25, 0.5, 0.75, 0.95])


def _all_methods_consistent(d, t):
    """Identity checks that hold for every SurvivalDistribution."""
    pdf = d.pdf(t)
    log_pdf = d.log_pdf(t)
    survival = d.survival(t)
    log_survival = d.log_survival(t)
    cdf = d.cdf(t)
    hazard = d.hazard(t)
    log_hazard = d.log_hazard(t)

    np.testing.assert_allclose(np.log(pdf), log_pdf, atol=1e-10)
    np.testing.assert_allclose(np.exp(log_survival), survival, atol=1e-12)
    np.testing.assert_allclose(cdf + survival, 1.0, atol=1e-12)
    np.testing.assert_allclose(hazard, pdf / survival, atol=1e-10)
    np.testing.assert_allclose(log_hazard, log_pdf - log_survival, atol=1e-10)


def test_weibull_matches_scipy():
    d = Weibull(shape=2.5, scale=1.7)
    np.testing.assert_allclose(d.survival(T), weibull_min.sf(T, c=2.5, scale=1.7))
    np.testing.assert_allclose(d.pdf(T), weibull_min.pdf(T, c=2.5, scale=1.7))
    np.testing.assert_allclose(d.cdf(T), weibull_min.cdf(T, c=2.5, scale=1.7))
    np.testing.assert_allclose(d.quantile(P), weibull_min.ppf(P, c=2.5, scale=1.7))
    np.testing.assert_allclose(d.mean(), weibull_min.mean(c=2.5, scale=1.7), rtol=1e-8)


def test_weibull_identities():
    _all_methods_consistent(Weibull(shape=2.5, scale=1.7), T)


def test_weibull_quantile_cdf_roundtrip():
    d = Weibull(shape=1.8, scale=2.3)
    np.testing.assert_allclose(d.cdf(d.quantile(P)), P, atol=1e-12)


def test_weibull_rejects_nonpositive_params():
    with pytest.raises(ValueError, match="shape and scale"):
        Weibull(shape=-1.0, scale=1.0)
    with pytest.raises(ValueError, match="shape and scale"):
        Weibull(shape=1.0, scale=0.0)


def test_exponential_matches_scipy():
    d = Exponential(rate=0.7)
    np.testing.assert_allclose(d.survival(T), expon.sf(T, scale=1 / 0.7))
    np.testing.assert_allclose(d.pdf(T), expon.pdf(T, scale=1 / 0.7))
    np.testing.assert_allclose(d.quantile(P), expon.ppf(P, scale=1 / 0.7))


def test_exponential_identities():
    _all_methods_consistent(Exponential(rate=0.7), T)


def test_exponential_constant_hazard():
    d = Exponential(rate=0.42)
    np.testing.assert_allclose(d.hazard(T), 0.42 * np.ones_like(T))


def test_lognormal_matches_scipy():
    mu, sigma = 0.3, 0.8
    d = LogNormal(mu=mu, sigma=sigma)
    np.testing.assert_allclose(d.survival(T), lognorm.sf(T, s=sigma, scale=np.exp(mu)))
    np.testing.assert_allclose(d.pdf(T), lognorm.pdf(T, s=sigma, scale=np.exp(mu)))
    np.testing.assert_allclose(d.quantile(P), lognorm.ppf(P, s=sigma, scale=np.exp(mu)))


def test_lognormal_identities():
    _all_methods_consistent(LogNormal(mu=0.3, sigma=0.8), T)


def test_lognormal_log_survival_stable_in_upper_tail():
    """Far above the median, scipy.lognorm.sf underflows but log_ndtr stays
    finite — verify our log_survival captures that."""
    d = LogNormal(mu=0.0, sigma=1.0)
    ls = d.log_survival(np.array([1e3, 1e4, 1e5]))
    assert np.all(np.isfinite(ls))
    assert np.all(ls < -10)  # very low survival


def test_loglogistic_matches_scipy():
    d = LogLogistic(shape=2.3, scale=1.8)
    np.testing.assert_allclose(d.survival(T), fisk.sf(T, c=2.3, scale=1.8))
    np.testing.assert_allclose(d.pdf(T), fisk.pdf(T, c=2.3, scale=1.8))
    np.testing.assert_allclose(d.quantile(P), fisk.ppf(P, c=2.3, scale=1.8))


def test_loglogistic_identities():
    _all_methods_consistent(LogLogistic(shape=2.3, scale=1.8), T)


def test_loglogistic_mean_inf_when_shape_le_1():
    assert np.isinf(LogLogistic(shape=0.5, scale=1.0).mean())
    assert np.isinf(LogLogistic(shape=1.0, scale=1.0).mean())
    assert np.isfinite(LogLogistic(shape=2.0, scale=1.0).mean())


def test_gompertz_identities():
    _all_methods_consistent(Gompertz(shape=0.5, rate=0.3), T)


def test_gompertz_survival_closed_form():
    """S(t) = exp(-(b/a)(exp(at) - 1))."""
    a, b = 0.5, 0.3
    d = Gompertz(shape=a, rate=b)
    expected = np.exp(-(b / a) * (np.exp(a * T) - 1))
    np.testing.assert_allclose(d.survival(T), expected, atol=1e-12)


def test_gompertz_hazard_grows_exponentially():
    a, b = 0.4, 0.2
    d = Gompertz(shape=a, rate=b)
    np.testing.assert_allclose(d.hazard(T), b * np.exp(a * T))


def test_gompertz_quantile_cdf_roundtrip():
    d = Gompertz(shape=0.3, rate=0.5)
    np.testing.assert_allclose(d.cdf(d.quantile(P)), P, atol=1e-12)


@pytest.mark.parametrize(
    "d",
    [
        Weibull(shape=2.0, scale=1.0),
        Exponential(rate=1.0),
        LogNormal(mu=0.0, sigma=1.0),
        LogLogistic(shape=2.0, scale=1.0),
        Gompertz(shape=0.3, rate=0.4),
    ],
)
def test_rvs_empirical_survival_matches_closed_form(d):
    rng = np.random.default_rng(0)
    samples = d.rvs(size=20000, rng=rng)
    for q in [0.25, 0.5, 0.75]:
        empirical = float((samples > d.quantile(q)).mean())
        assert abs(empirical - (1.0 - q)) < 0.02, (
            f"survival at q={q} disagrees: empirical={empirical}, closed-form={1 - q}"
        )


@pytest.mark.parametrize(
    "d",
    [
        Weibull(shape=2.0, scale=1.0),
        LogNormal(mu=0.0, sigma=1.0),
        LogLogistic(shape=2.0, scale=1.0),
    ],
)
def test_log_pdf_is_negative_infinity_at_zero(d):
    """For positive-support distributions, log f(0) = -inf."""
    assert np.isneginf(d.log_pdf(0.0))


@pytest.mark.parametrize(
    "d",
    [
        Weibull(shape=2.0, scale=1.0),
        LogNormal(mu=0.0, sigma=1.0),
        LogLogistic(shape=2.0, scale=1.0),
        Exponential(rate=1.0),
        Gompertz(shape=0.5, rate=0.3),
    ],
)
def test_survival_at_zero_is_one(d):
    np.testing.assert_allclose(d.survival(0.0), 1.0, atol=1e-12)
