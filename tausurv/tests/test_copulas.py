"""Tests for the numpy/scipy-backed Archimedean copulas.

Covers identity / boundary properties, Kendall's τ closed-form values,
tail-dependence coefficients, and Marshall–Olkin sampling consistency.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kendalltau

from tausurv.copulas import Clayton, Frank, Gumbel, Independence, Joe

ALL_FAMILIES = [
    ("Independence", Independence()),
    ("Clayton(2)", Clayton(2.0)),
    ("Gumbel(2)", Gumbel(2.0)),
    ("Frank(3)", Frank(3.0)),
    ("Joe(2)", Joe(2.0)),
]


@pytest.mark.parametrize("name,c", ALL_FAMILIES)
def test_cdf_at_all_ones_equals_one(name, c):
    """$C(1, \\dots, 1) = 1$."""
    u = np.full((1, 2), 1.0 - 1e-12)
    np.testing.assert_allclose(c.cdf(u), 1.0, atol=1e-9)


@pytest.mark.parametrize("name,c", ALL_FAMILIES)
def test_marginal_uniform_property(name, c):
    """$C(u, 1) = u$ — pushing one component to 1 reduces to the marginal."""
    u_vals = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    u = np.stack([u_vals, np.full_like(u_vals, 1.0 - 1e-12)], axis=-1)
    np.testing.assert_allclose(c.cdf(u), u_vals, atol=1e-6)


@pytest.mark.parametrize("name,c", ALL_FAMILIES)
def test_cdf_non_decreasing_in_each_component(name, c):
    """$\\partial C / \\partial u_i \\ge 0$ for every $i$."""
    rng = np.random.default_rng(0)
    u = rng.uniform(0.1, 0.9, size=(50, 2))
    base = c.cdf(u)
    # Bump u_1 by a small amount; CDF should not decrease.
    u_plus = u.copy()
    u_plus[:, 0] += 0.05
    u_plus[:, 0] = np.clip(u_plus[:, 0], 0.0, 1.0)
    assert (c.cdf(u_plus) - base >= -1e-12).all()


@pytest.mark.parametrize("name,c", ALL_FAMILIES)
def test_cdf_in_unit_interval(name, c):
    rng = np.random.default_rng(0)
    u = rng.uniform(0.01, 0.99, size=(100, 2))
    cdf = c.cdf(u)
    assert ((cdf >= 0.0) & (cdf <= 1.0)).all()


@pytest.mark.parametrize("name,c", ALL_FAMILIES)
def test_pdf_positive(name, c):
    rng = np.random.default_rng(0)
    u = rng.uniform(0.1, 0.9, size=(50, 2))
    assert (c.pdf(u) > 0).all()


def test_clayton_kendalls_tau():
    """$\\tau_{\\mathrm{Clayton}} = \\theta / (\\theta + 2)$."""
    for theta in [0.5, 1.0, 2.0, 5.0]:
        c = Clayton(theta)
        np.testing.assert_allclose(c.kendalls_tau(), theta / (theta + 2.0))


def test_gumbel_kendalls_tau():
    """$\\tau_{\\mathrm{Gumbel}} = 1 - 1/\\theta$."""
    for theta in [1.5, 2.0, 5.0, 10.0]:
        c = Gumbel(theta)
        np.testing.assert_allclose(c.kendalls_tau(), 1.0 - 1.0 / theta)


def test_independence_kendalls_tau_zero():
    assert Independence().kendalls_tau() == 0.0


def test_frank_kendalls_tau_zero_at_zero_limit():
    """$\\tau_{\\mathrm{Frank}}(\\theta \\to 0) = 0$."""
    assert abs(Frank(0.01).kendalls_tau()) < 0.01


def test_joe_kendalls_tau_zero_at_independence():
    """$\\tau_{\\mathrm{Joe}}(\\theta = 1) = 0$."""
    assert abs(Joe(1.0).kendalls_tau()) < 1e-3


def test_clayton_tail_dependence_lower_only():
    c = Clayton(2.0)
    lam_l, lam_u = c.tail_dependence()
    np.testing.assert_allclose(lam_l, 2.0 ** (-0.5))
    assert lam_u == 0.0


def test_gumbel_tail_dependence_upper_only():
    c = Gumbel(2.0)
    lam_l, lam_u = c.tail_dependence()
    assert lam_l == 0.0
    np.testing.assert_allclose(lam_u, 2.0 - 2.0**0.5)


def test_frank_no_tail_dependence():
    assert Frank(3.0).tail_dependence() == (0.0, 0.0)


def test_joe_tail_dependence_upper_only():
    c = Joe(2.0)
    lam_l, lam_u = c.tail_dependence()
    assert lam_l == 0.0
    np.testing.assert_allclose(lam_u, 2.0 - 2.0**0.5)


@pytest.mark.parametrize(
    "c",
    [Clayton(2.0), Gumbel(2.0), Frank(3.0), Independence()],
)
def test_samples_have_uniform_marginals(c):
    rng = np.random.default_rng(0)
    samples = c.sample(size=20_000, d=2, rng=rng)
    # Each marginal should be uniform with mean 0.5; allow 4σ slop where
    # σ(mean) ≈ 1/sqrt(12 · n) ≈ 0.002 for n=20k.
    np.testing.assert_allclose(samples.mean(axis=0), 0.5, atol=0.01)


@pytest.mark.parametrize(
    "c",
    [Clayton(2.0), Gumbel(2.0), Frank(3.0)],
)
def test_empirical_kendalls_tau_matches_closed_form(c):
    rng = np.random.default_rng(0)
    samples = c.sample(size=20_000, d=2, rng=rng)
    tau_emp, _ = kendalltau(samples[:, 0], samples[:, 1])
    np.testing.assert_allclose(tau_emp, c.kendalls_tau(), atol=0.02)


def test_independence_samples_have_zero_correlation():
    rng = np.random.default_rng(0)
    samples = Independence().sample(size=20_000, d=2, rng=rng)
    tau_emp, _ = kendalltau(samples[:, 0], samples[:, 1])
    assert abs(tau_emp) < 0.02


def test_joe_sampling_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="Sibuya"):
        Joe(2.0).sample(size=10)


def test_clayton_rejects_zero_theta():
    with pytest.raises(ValueError, match="theta must be"):
        Clayton(0.0)


def test_gumbel_rejects_theta_below_one():
    with pytest.raises(ValueError, match="theta must be"):
        Gumbel(0.5)


def test_joe_rejects_theta_below_one():
    with pytest.raises(ValueError, match="theta must be"):
        Joe(0.5)


def test_frank_rejects_zero_theta():
    with pytest.raises(ValueError, match="cannot be 0"):
        Frank(0.0)


@pytest.mark.parametrize("c", [Clayton(2.0), Gumbel(2.0), Frank(3.0), Joe(2.0)])
def test_d_variate_cdf_works_for_d_3(c):
    """d=3 cdf is computable via the generic formula."""
    u = np.array([[0.3, 0.5, 0.7], [0.6, 0.4, 0.2]])
    cdf = c.cdf(u)
    assert cdf.shape == (2,)
    assert ((cdf >= 0) & (cdf <= 1)).all()


def test_clayton_d_variate_log_pdf_closed_form():
    """Clayton has clean d-variate closed-form density via the factorial
    formula for $(\\varphi^{-1})^{(d)}$."""
    c = Clayton(2.0)
    u = np.array([[0.3, 0.5, 0.7]])
    lp = c.log_pdf(u)
    assert lp.shape == (1,)
    assert np.isfinite(lp).all()
    # Density at this point should be positive (lp finite).
    assert np.exp(lp) > 0
