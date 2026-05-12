"""Tests for PositiveLinear and MonotoneMLP.

The architectural guarantee is that ``MonotoneMLP`` is monotone
non-decreasing in *every* input by construction — verify this holds for
random parameter values (i.e., it's a structural property of the
architecture, not something that needs to be learned).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tausurv.nn.modules import MonotoneMLP, PositiveLinear


def test_positive_linear_effective_weights_are_non_negative():
    layer = PositiveLinear(5, 3)
    effective_weights = layer.weight_param**2
    assert (effective_weights >= 0).all()


def test_positive_linear_init_avoids_zero_collapse():
    """After init, the effective weights should be non-trivially positive
    — not collapsed near zero (which would zero out the forward pass and
    starve gradients)."""
    layer = PositiveLinear(64, 64)
    effective = layer.weight_param**2
    # Just check magnitudes are healthy: most weights > 1e-3, mean ~ O(0.01-1).
    assert effective.mean() > 1e-3
    assert (effective > 1e-4).float().mean() > 0.9


def test_positive_linear_forward_shape():
    layer = PositiveLinear(8, 4)
    out = layer(torch.randn(7, 8))
    assert out.shape == (7, 4)


def test_positive_linear_gradients_flow_to_param():
    """A nonzero gradient should flow to the underlying parameter even
    though the effective weights are squared."""
    layer = PositiveLinear(4, 1)
    x = torch.randn(3, 4)
    loss = layer(x).sum()
    loss.backward()
    assert layer.weight_param.grad is not None
    assert (layer.weight_param.grad != 0).any()


@pytest.mark.parametrize("activation", ["tanh", "sigmoid"])
def test_monotone_mlp_output_increases_when_any_input_increases(activation):
    """The defining property: bumping any input upward must not decrease
    the output. Verify on random parameter values across many random
    inputs — this is monotonicity by construction, not by training."""
    torch.manual_seed(0)
    net = MonotoneMLP(
        in_features=4, out_features=1, hidden_features=(16, 16), activation=activation
    )
    x = torch.rand(64, 4)
    base = net(x)

    # For each input dimension, bump by 0.1 and confirm output is >= base.
    for i in range(4):
        x_bumped = x.clone()
        x_bumped[:, i] += 0.1
        bumped = net(x_bumped)
        # Strict inequality is the goal but float slop ~ 1e-7.
        assert (bumped - base >= -1e-6).all(), (
            f"Monotonicity violated in input dim {i}: max decrease "
            f"{(base - bumped).max().item():.6f}"
        )


def test_monotone_mlp_output_strictly_increases_under_large_bump():
    """A large input bump produces a strict (non-tied) output increase
    at random points — guards against accidental constant output."""
    torch.manual_seed(0)
    net = MonotoneMLP(in_features=3, out_features=1, hidden_features=(8, 8))
    x = torch.rand(32, 3)
    out_lo = net(x)
    out_hi = net(x + 1.0)
    assert (out_hi >= out_lo - 1e-6).all()
    # At least some entries should strictly increase.
    assert (out_hi - out_lo > 1e-4).any()


def test_monotone_mlp_forward_shape():
    net = MonotoneMLP(in_features=5, out_features=2, hidden_features=(8, 8))
    out = net(torch.randn(7, 5))
    assert out.shape == (7, 2)


def test_monotone_mlp_rejects_bad_activation():
    with pytest.raises(ValueError, match="tanh.*sigmoid"):
        MonotoneMLP(in_features=3, activation="relu")


def test_monotone_mlp_gradient_flows():
    net = MonotoneMLP(in_features=4, out_features=1, hidden_features=(8, 8))
    x = torch.randn(5, 4, requires_grad=True)
    loss = net(x).sum()
    loss.backward()
    # All PositiveLinear weight_param tensors should have non-trivial grad.
    for m in net.modules():
        if isinstance(m, PositiveLinear):
            assert m.weight_param.grad is not None
            assert (m.weight_param.grad != 0).any()


def test_monotone_mlp_stays_monotone_after_gradient_step():
    """The architecture guarantees monotonicity for *any* parameter values
    (because effective weights are squared and the activation is monotone).
    Verify by running a few SGD steps and re-checking the property — a
    real test that this is structural, not coincidental."""
    torch.manual_seed(0)
    net = MonotoneMLP(in_features=4, out_features=1, hidden_features=(16, 16))
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    target = torch.randn(32, 1)
    x = torch.rand(32, 4)
    for _ in range(20):
        opt.zero_grad()
        loss = ((net(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
    # Monotonicity still holds.
    for i in range(4):
        x_bumped = x.clone()
        x_bumped[:, i] += 0.1
        assert (net(x_bumped) - net(x) >= -1e-6).all()
