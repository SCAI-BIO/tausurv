"""Tests for IPM functionals used in representation balancing."""

from __future__ import annotations

import pytest
import torch

from causurv.nn.functional.ipm import (
    get_ipm,
    mmd_linear,
    wasserstein_squared,
)


def test_mmd_zero_on_identical_batches():
    torch.manual_seed(0)
    x = torch.randn(20, 8)
    assert float(mmd_linear(x, x)) == pytest.approx(0.0, abs=1e-12)


def test_mmd_zero_when_means_match():
    """Linear MMD is only sensitive to mean shifts."""
    x = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])  # mean = 0
    y = torch.tensor([[0.0, 1.0], [0.0, -1.0]])  # mean = 0
    assert float(mmd_linear(x, y)) == pytest.approx(0.0, abs=1e-12)


def test_mmd_picks_up_mean_shift():
    x = torch.zeros(10, 3)
    y = torch.ones(10, 3) * 2.0
    # ||mean(x) - mean(y)||^2 = ||(0,0,0) - (2,2,2)||^2 = 12
    assert float(mmd_linear(x, y)) == pytest.approx(12.0)


def test_wasserstein_smaller_when_batches_match():
    """Entropic-regularised Wasserstein doesn't hit 0 on identical batches
    (the regulariser smears the transport plan). Test the *direction*
    instead: identical batches transport more cheaply than disjoint ones."""
    torch.manual_seed(0)
    x = torch.randn(30, 5)
    y_close = x + 0.01 * torch.randn_like(x)
    y_far = x + 5.0 * torch.randn_like(x)
    assert float(wasserstein_squared(x, y_close)) < float(
        wasserstein_squared(x, y_far)
    )


def test_wasserstein_positive_on_disjoint_batches():
    x = torch.zeros(20, 4)
    y = torch.ones(20, 4) * 5.0
    assert float(wasserstein_squared(x, y)) > 1.0


def test_ipm_rejects_dim_mismatch():
    x = torch.zeros(5, 4)
    y = torch.zeros(5, 3)
    with pytest.raises(ValueError, match="matching feature dim"):
        mmd_linear(x, y)
    with pytest.raises(ValueError, match="matching feature dim"):
        wasserstein_squared(x, y)


def test_ipm_handles_empty_arm_gracefully():
    """One side empty ⇒ return 0 instead of crashing (the loss skips
    empty arms; the functional should be permissive too)."""
    x = torch.zeros(0, 4)
    y = torch.zeros(5, 4)
    assert float(mmd_linear(x, y)) == 0.0
    assert float(wasserstein_squared(x, y)) == 0.0


def test_ipm_is_differentiable():
    x = torch.randn(10, 4, requires_grad=True)
    y = torch.randn(8, 4)
    loss = wasserstein_squared(x, y) + mmd_linear(x, y)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_get_ipm_returns_correct_function():
    assert get_ipm("wasserstein") is wasserstein_squared
    assert get_ipm("mmd") is mmd_linear


def test_get_ipm_rejects_unknown():
    with pytest.raises(ValueError, match="IPM must be one of"):
        get_ipm("kl")
