"""Tests for SurvITELoss: factual hazard NLL + IPM penalty."""

from __future__ import annotations

import pytest
import torch

from causurv.nn.losses import SurvITELoss


def _make_inputs(*, n=40, n_bins=10, repr_dim=8, n_arms=2, seed=0):
    torch.manual_seed(seed)
    phi = torch.randn(n, repr_dim, requires_grad=True)
    logits = [torch.randn(n, n_bins, requires_grad=True) for _ in range(n_arms)]
    T = torch.randint(1, n_bins, (n,)).float()
    E = torch.randint(0, 2, (n,)).long()
    A = torch.randint(0, n_arms, (n,)).long()
    return phi, logits, T, E, A


def test_returns_decomposed_output():
    phi, logits, T, E, A = _make_inputs()
    loss = SurvITELoss(torch.arange(1, 11, dtype=torch.float32), beta=0.001)
    out = loss(phi, logits, T, E, A)
    # Has the three documented fields.
    assert hasattr(out, "total") and hasattr(out, "hazard_nll") and hasattr(out, "ipm")
    # All scalar.
    assert out.total.ndim == 0 and out.hazard_nll.ndim == 0 and out.ipm.ndim == 0


def test_total_equals_nll_plus_beta_times_ipm():
    phi, logits, T, E, A = _make_inputs()
    beta = 0.5
    loss = SurvITELoss(torch.arange(1, 11, dtype=torch.float32), beta=beta)
    out = loss(phi, logits, T, E, A)
    total = out.total.detach().item()
    nll = out.hazard_nll.detach().item()
    ipm = out.ipm.detach().item()
    assert total == pytest.approx(nll + beta * ipm, rel=1e-5)


def test_zero_beta_disables_ipm_from_gradient_path():
    """With beta=0, the IPM term contributes 0 to the total."""
    phi, logits, T, E, A = _make_inputs()
    loss = SurvITELoss(torch.arange(1, 11, dtype=torch.float32), beta=0.0)
    out = loss(phi, logits, T, E, A)
    assert out.total.detach().item() == pytest.approx(
        out.hazard_nll.detach().item(), rel=1e-6
    )


def test_ipm_is_nonnegative():
    phi, logits, T, E, A = _make_inputs()
    loss = SurvITELoss(torch.arange(1, 11, dtype=torch.float32))
    out = loss(phi, logits, T, E, A)
    assert out.ipm.detach().item() >= 0.0


def test_hazard_nll_is_nonnegative():
    phi, logits, T, E, A = _make_inputs()
    loss = SurvITELoss(torch.arange(1, 11, dtype=torch.float32))
    out = loss(phi, logits, T, E, A)
    assert out.hazard_nll.detach().item() >= 0.0


def test_loss_is_differentiable_through_phi_and_logits():
    phi, logits, T, E, A = _make_inputs()
    loss = SurvITELoss(torch.arange(1, 11, dtype=torch.float32), beta=0.1)
    out = loss(phi, logits, T, E, A)
    out.total.backward()
    assert phi.grad is not None and torch.isfinite(phi.grad).all()
    for z in logits:
        assert z.grad is not None and torch.isfinite(z.grad).all()


def test_factual_only_arm_zero_only_uses_arm_zero_logits():
    """If every subject is in arm 0, arm-1 logits should not affect the
    NLL (their gradient through hazard_nll must be zero)."""
    torch.manual_seed(0)
    phi = torch.randn(20, 8, requires_grad=True)
    logits = [torch.randn(20, 10, requires_grad=True) for _ in range(2)]
    T = torch.randint(1, 10, (20,)).float()
    E = torch.randint(0, 2, (20,)).long()
    A = torch.zeros(20, dtype=torch.long)  # everyone in arm 0
    loss = SurvITELoss(torch.arange(1, 11, dtype=torch.float32), beta=0.0)
    out = loss(phi, logits, T, E, A)
    out.hazard_nll.backward()
    # Arm-1 logits never see any subject ⇒ no gradient flow at all
    # (autograd leaves grad as None rather than a zero tensor).
    assert logits[1].grad is None or (logits[1].grad == 0).all()
    # Arm-0 logits do.
    assert (logits[0].grad != 0).any()


def test_supports_three_arms():
    phi, logits, T, E, A = _make_inputs(n_arms=3, n_bins=8)
    loss = SurvITELoss(torch.arange(1, 9, dtype=torch.float32))
    out = loss(phi, logits, T, E, A)
    assert torch.isfinite(out.total)
