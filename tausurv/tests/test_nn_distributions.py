"""Tests for the differentiable torch survival distributions.

Covers:

- Cross-backend parity with :mod:`tausurv.distributions` on the same
  parameter values.
- Differentiability of ``log_pdf`` and ``log_survival`` w.r.t. distribution
  parameters (essential for DSM and parametric AFT-NN training).
- Reparameterized sampling: ``rsample`` is differentiable w.r.t. params.
- Broadcasting over batched parameters.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tausurv import distributions as np_dist
from tausurv.nn import distributions as t_dist

T_NP = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
T_T = torch.tensor(T_NP, dtype=torch.float64)
P_NP = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
P_T = torch.tensor(P_NP, dtype=torch.float64)

CASES = [
    (
        "Weibull",
        np_dist.Weibull(2.5, 1.7),
        t_dist.Weibull(
            torch.tensor(2.5, dtype=torch.float64),
            torch.tensor(1.7, dtype=torch.float64),
        ),
    ),
    (
        "LogNormal",
        np_dist.LogNormal(0.3, 0.8),
        t_dist.LogNormal(
            torch.tensor(0.3, dtype=torch.float64),
            torch.tensor(0.8, dtype=torch.float64),
        ),
    ),
    (
        "LogLogistic",
        np_dist.LogLogistic(2.3, 1.8),
        t_dist.LogLogistic(
            torch.tensor(2.3, dtype=torch.float64),
            torch.tensor(1.8, dtype=torch.float64),
        ),
    ),
    (
        "Exponential",
        np_dist.Exponential(0.7),
        t_dist.Exponential(torch.tensor(0.7, dtype=torch.float64)),
    ),
    (
        "Gompertz",
        np_dist.Gompertz(0.5, 0.3),
        t_dist.Gompertz(
            torch.tensor(0.5, dtype=torch.float64),
            torch.tensor(0.3, dtype=torch.float64),
        ),
    ),
]


@pytest.mark.parametrize("name,np_d,t_d", CASES)
def test_log_pdf_matches_numpy(name, np_d, t_d):
    np.testing.assert_allclose(t_d.log_pdf(T_T).numpy(), np_d.log_pdf(T_NP), atol=1e-10)


@pytest.mark.parametrize("name,np_d,t_d", CASES)
def test_log_survival_matches_numpy(name, np_d, t_d):
    np.testing.assert_allclose(
        t_d.log_survival(T_T).numpy(), np_d.log_survival(T_NP), atol=1e-10
    )


@pytest.mark.parametrize("name,np_d,t_d", CASES)
def test_survival_matches_numpy(name, np_d, t_d):
    np.testing.assert_allclose(
        t_d.survival(T_T).numpy(), np_d.survival(T_NP), atol=1e-10
    )


@pytest.mark.parametrize("name,np_d,t_d", CASES)
def test_quantile_matches_numpy(name, np_d, t_d):
    np.testing.assert_allclose(
        t_d.quantile(P_T).numpy(), np_d.quantile(P_NP), atol=1e-10
    )


@pytest.mark.parametrize("name,np_d,t_d", [c for c in CASES if c[0] != "Gompertz"])
def test_mean_matches_numpy(name, np_d, t_d):
    """Gompertz mean isn't implemented in torch (no differentiable E_1)."""
    np.testing.assert_allclose(float(t_d.mean().item()), np_d.mean(), rtol=1e-10)


def test_gompertz_torch_mean_raises():
    d = t_dist.Gompertz(torch.tensor(0.5), torch.tensor(0.3))
    with pytest.raises(NotImplementedError, match="E_1"):
        d.mean()


def test_weibull_log_pdf_is_differentiable():
    k = torch.tensor(2.0, requires_grad=True)
    lam = torch.tensor(1.5, requires_grad=True)
    d = t_dist.Weibull(k, lam)
    loss = -d.log_pdf(torch.tensor(1.0)).sum()
    loss.backward()
    assert torch.isfinite(k.grad) and k.grad != 0
    assert torch.isfinite(lam.grad) and lam.grad != 0


def test_lognormal_log_survival_is_differentiable():
    mu = torch.tensor(0.0, requires_grad=True)
    sigma = torch.tensor(1.0, requires_grad=True)
    d = t_dist.LogNormal(mu, sigma)
    loss = -d.log_survival(torch.tensor(2.0)).sum()
    loss.backward()
    assert torch.isfinite(mu.grad) and mu.grad != 0
    assert torch.isfinite(sigma.grad) and sigma.grad != 0


def test_loglogistic_log_pdf_is_differentiable():
    k = torch.tensor(2.0, requires_grad=True)
    lam = torch.tensor(1.5, requires_grad=True)
    d = t_dist.LogLogistic(k, lam)
    loss = -d.log_pdf(torch.tensor(1.0)).sum()
    loss.backward()
    assert torch.isfinite(k.grad) and k.grad != 0
    assert torch.isfinite(lam.grad) and lam.grad != 0


def test_gompertz_log_survival_is_differentiable():
    a = torch.tensor(0.5, requires_grad=True)
    b = torch.tensor(0.3, requires_grad=True)
    d = t_dist.Gompertz(a, b)
    loss = -d.log_survival(torch.tensor(2.0)).sum()
    loss.backward()
    assert torch.isfinite(a.grad) and a.grad != 0
    assert torch.isfinite(b.grad) and b.grad != 0


def test_rsample_is_differentiable_through_params():
    """Reparameterized sampling: gradient flows from sample back to params."""
    k = torch.tensor(2.0, requires_grad=True)
    lam = torch.tensor(1.5, requires_grad=True)
    d = t_dist.Weibull(k, lam)
    torch.manual_seed(0)
    samples = d.rsample(sample_shape=(50,))
    # Sample mean should depend on (k, lam) — backward should flow.
    loss = samples.mean()
    loss.backward()
    assert torch.isfinite(k.grad) and k.grad != 0
    assert torch.isfinite(lam.grad) and lam.grad != 0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: t_dist.Weibull(torch.tensor(2.0), torch.tensor(1.0)),
        lambda: t_dist.LogNormal(torch.tensor(0.0), torch.tensor(1.0)),
        lambda: t_dist.LogLogistic(torch.tensor(2.0), torch.tensor(1.0)),
        lambda: t_dist.Exponential(torch.tensor(1.0)),
        lambda: t_dist.Gompertz(torch.tensor(0.3), torch.tensor(0.4)),
    ],
)
def test_sample_empirical_survival_matches_closed_form(factory):
    d = factory()
    torch.manual_seed(0)
    samples = d.sample(sample_shape=(20000,))
    for q in [0.25, 0.5, 0.75]:
        empirical = float(
            (samples > d.quantile(torch.tensor(q, dtype=samples.dtype))).float().mean()
        )
        assert abs(empirical - (1.0 - q)) < 0.02


def test_weibull_broadcasts_over_batched_params():
    """A 4-element parameter tensor should give a 4-element output."""
    k = torch.tensor([1.0, 2.0, 1.5, 0.8])
    lam = torch.tensor([1.0, 1.5, 2.0, 0.5])
    d = t_dist.Weibull(k, lam)
    out = d.log_pdf(torch.tensor(1.0))
    assert out.shape == (4,)
    # Each entry matches the corresponding numpy distribution.
    for i in range(4):
        np_d = np_dist.Weibull(shape=k[i].item(), scale=lam[i].item())
        np.testing.assert_allclose(out[i].item(), float(np_d.log_pdf(1.0)), atol=1e-6)


def test_torch_dispatch_works_with_python_scalars():
    """Passing a Python scalar for `t` should still produce a valid tensor."""
    d = t_dist.Weibull(torch.tensor(2.0), torch.tensor(1.0))
    out = d.log_pdf(1.0)
    assert torch.is_tensor(out) and out.numel() == 1
