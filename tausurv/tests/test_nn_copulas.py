"""Tests for the differentiable torch copulas.

Cross-backend parity, gradient flow through ``theta``, and the
HACSurv-relevant pattern: joint survival ``cdf(S)`` + per-cause partial
``∂C/∂S_k`` via autograd.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tausurv import copulas as np_cop
from tausurv.nn import copulas as t_cop

U_NP = np.array([[0.3, 0.5], [0.7, 0.2], [0.5, 0.5], [0.1, 0.9]])
U_T = torch.tensor(U_NP, dtype=torch.float64)

CASES = [
    ("Independence", np_cop.Independence(), t_cop.Independence()),
    (
        "Clayton(2)",
        np_cop.Clayton(2.0),
        t_cop.Clayton(torch.tensor(2.0, dtype=torch.float64)),
    ),
    (
        "Gumbel(2)",
        np_cop.Gumbel(2.0),
        t_cop.Gumbel(torch.tensor(2.0, dtype=torch.float64)),
    ),
    (
        "Frank(3)",
        np_cop.Frank(3.0),
        t_cop.Frank(torch.tensor(3.0, dtype=torch.float64)),
    ),
    ("Joe(2)", np_cop.Joe(2.0), t_cop.Joe(torch.tensor(2.0, dtype=torch.float64))),
]


@pytest.mark.parametrize("name,np_c,t_c", CASES)
def test_cdf_matches_numpy(name, np_c, t_c):
    np.testing.assert_allclose(t_c.cdf(U_T).numpy(), np_c.cdf(U_NP), atol=1e-12)


@pytest.mark.parametrize("name,np_c,t_c", CASES)
def test_log_pdf_matches_numpy(name, np_c, t_c):
    np.testing.assert_allclose(t_c.log_pdf(U_T).numpy(), np_c.log_pdf(U_NP), atol=1e-12)


@pytest.mark.parametrize("name,np_c,t_c", CASES)
def test_kendalls_tau_matches_numpy(name, np_c, t_c):
    assert float(t_c.kendalls_tau().item()) == pytest.approx(
        np_c.kendalls_tau(), abs=1e-9
    )


@pytest.mark.parametrize(
    "family,theta",
    [
        (t_cop.Clayton, 2.0),
        (t_cop.Gumbel, 2.0),
        (t_cop.Frank, 3.0),
        (t_cop.Joe, 2.0),
    ],
)
def test_log_pdf_differentiable_through_theta(family, theta):
    theta_t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    c = family(theta_t)
    u = torch.tensor([[0.3, 0.5], [0.7, 0.2]], dtype=torch.float64)
    loss = -c.log_pdf(u).sum()
    loss.backward()
    grad = theta_t.grad
    assert torch.isfinite(grad)
    assert grad != 0, f"{family.__name__}: zero gradient through theta"


@pytest.mark.parametrize(
    "family,theta",
    [
        (t_cop.Clayton, 2.0),
        (t_cop.Gumbel, 2.0),
        (t_cop.Frank, 3.0),
        (t_cop.Joe, 2.0),
    ],
)
def test_cdf_differentiable_through_theta(family, theta):
    theta_t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    c = family(theta_t)
    u = torch.tensor([[0.3, 0.5]], dtype=torch.float64)
    loss = c.cdf(u).sum()
    loss.backward()
    assert torch.isfinite(theta_t.grad)
    assert theta_t.grad != 0


def test_hacsurv_pattern_partial_derivative_through_marginals():
    """The HACSurv likelihood uses ∂C/∂S_k for the realized cause.
    Verify gradients flow from joint survival back to per-cause survival
    inputs."""
    K = 3
    S = torch.tensor(
        [[0.3, 0.5, 0.7], [0.7, 0.4, 0.6]],
        dtype=torch.float64,
        requires_grad=True,
    )
    c = t_cop.Clayton(torch.tensor(2.0, dtype=torch.float64))
    joint = c.cdf(S)
    (grads,) = torch.autograd.grad(joint.sum(), S)
    assert grads.shape == S.shape
    # All partials should be in (0, 1) — the copula is a valid joint
    # distribution function.
    assert (grads > 0).all() and (grads <= 1).all()


def test_hacsurv_pattern_joint_survival_shape_k_causes():
    """For competing risks with K causes, cdf(S) returns one joint
    survival value per subject regardless of K."""
    c = t_cop.Gumbel(torch.tensor(2.0, dtype=torch.float64))
    for K in (2, 3, 5):
        S = torch.rand((10, K), dtype=torch.float64)
        joint = c.cdf(S)
        assert joint.shape == (10,)
        assert ((joint >= 0) & (joint <= 1)).all()


def test_cdf_handles_large_batch_efficiently():
    """Sanity: cdf returns the right shape on a large batch."""
    c = t_cop.Clayton(torch.tensor(2.0, dtype=torch.float64))
    u = torch.rand((5000, 2), dtype=torch.float64)
    cdf = c.cdf(u)
    assert cdf.shape == (5000,)


def test_independence_cdf_equals_product():
    u = torch.tensor([[0.3, 0.5, 0.7], [0.6, 0.4, 0.2]], dtype=torch.float64)
    cdf = t_cop.Independence().cdf(u)
    np.testing.assert_allclose(cdf.numpy(), u.prod(dim=-1).numpy())


def test_independence_log_pdf_is_zero():
    u = torch.tensor([[0.3, 0.5], [0.7, 0.2]], dtype=torch.float64)
    lp = t_cop.Independence().log_pdf(u)
    np.testing.assert_allclose(lp.numpy(), 0.0, atol=1e-12)
