"""Tests for HACSurv (math + architecture invariants only).

Per the project's test/sanity-script convention, training-based recovery
checks live in ``scripts/`` (AI workspace) — these tests only verify:

- LearnedGenerator math (Laplace transform values, inversion identity,
  gradient flow).
- HACSurv forward dict shapes and finite values.
- Loss functional/class-wrapper equivalence.
- Save/load round-trip including the time grid.
- predict_cif shapes and bounds.
- Config validation.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tausurv.nn import (
    HACSurv,
    HACSurvConfig,
    HACSurvLoss,
    HACSurvTrainer,
    LearnedGenerator,
)
from tausurv.nn.functional import hacsurv_nll


def _data(n=80, K=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)).astype(np.float32)
    T = np.abs(rng.exponential(1.0, size=n)).astype(np.float32) + 1e-2
    E = rng.choice(list(range(K + 1)), size=n).astype(np.int64)
    return (
        torch.tensor(X),
        torch.tensor(T),
        torch.tensor(E),
    )


def _make_model(family="clayton", n_causes=2, seed=0, **kw):
    torch.manual_seed(seed)
    return HACSurv(
        in_features=5, n_causes=n_causes, t_max=5.0, copula_family=family, **kw
    )


# LearnedGenerator math


def test_learned_generator_phi_inv_at_zero_equals_one():
    """$\\varphi^{-1}(0) = \\mathbb{E}[\\exp(-0 \\cdot M)] = \\mathbb{E}[1] = 1$."""
    torch.manual_seed(0)
    gen = LearnedGenerator(hidden_dim=8, n_samples=64)
    gen.resample()
    value = gen.phi_inv(torch.tensor([0.0, 0.0, 0.0]))
    torch.testing.assert_close(value, torch.ones_like(value), atol=1e-6, rtol=0)


def test_learned_generator_phi_inv_decreasing_in_s():
    """The Laplace transform is monotone non-increasing in s."""
    torch.manual_seed(0)
    gen = LearnedGenerator(hidden_dim=8, n_samples=64)
    gen.resample()
    s = torch.linspace(0.0, 5.0, 30)
    out = gen.phi_inv(s)
    assert (torch.diff(out) <= 1e-6).all()


def test_learned_generator_phi_inv_in_unit_interval():
    torch.manual_seed(0)
    gen = LearnedGenerator(hidden_dim=8, n_samples=64)
    gen.resample()
    s = torch.linspace(0.0, 10.0, 100)
    out = gen.phi_inv(s)
    assert ((out >= 0.0) & (out <= 1.0 + 1e-6)).all()


def test_learned_generator_phi_inverts_phi_inv():
    """``phi(phi_inv(s)) ≈ s`` — Newton inversion is consistent."""
    torch.manual_seed(0)
    gen = LearnedGenerator(hidden_dim=8, n_samples=64)
    gen.resample()
    s = torch.linspace(0.1, 3.0, 20)
    y = gen.phi_inv(s)
    t = gen.phi(y)
    torch.testing.assert_close(t, s, atol=1e-4, rtol=1e-3)


def test_learned_generator_requires_resample():
    gen = LearnedGenerator(hidden_dim=8, n_samples=32)
    with pytest.raises(RuntimeError, match="resample"):
        gen.phi_inv(torch.tensor([0.5]))


def test_learned_generator_gradients_flow_to_weight_net():
    """Calling phi_inv with a loss-like reduction should produce nonzero
    gradients on the weight network parameters."""
    torch.manual_seed(0)
    gen = LearnedGenerator(hidden_dim=8, n_samples=64)
    gen.resample()
    s = torch.tensor([0.5, 1.0, 2.0])
    loss = (gen.phi_inv(s) ** 2).sum()
    loss.backward()
    grads = [p.grad for p in gen._weight_net.parameters() if p.grad is not None]
    assert grads and any((g != 0).any() for g in grads)


# HACSurv forward


def test_forward_returns_dict_with_right_shapes():
    X, T, _ = _data(n=20, K=2)
    model = _make_model(family="clayton")
    out = model(X, T)
    n, K = 20, 2
    assert out["S"].shape == (n, K)
    assert out["f"].shape == (n, K)
    assert out["joint"].shape == (n,)
    assert out["dC_dS"].shape == (n, K)


def test_forward_outputs_finite_and_bounded():
    X, T, _ = _data(n=20, K=2)
    model = _make_model()
    out = model(X, T)
    for k, v in out.items():
        assert torch.isfinite(v).all(), f"non-finite values in {k!r}"
    assert ((out["S"] >= -1e-6) & (out["S"] <= 1 + 1e-6)).all()
    assert ((out["joint"] >= -1e-6) & (out["joint"] <= 1 + 1e-6)).all()
    assert (out["f"] >= -1e-6).all()


def test_forward_supports_3_causes():
    X, T, _ = _data(n=15, K=3)
    model = _make_model(n_causes=3)
    out = model(X, T)
    assert out["S"].shape == (15, 3)
    assert out["dC_dS"].shape == (15, 3)


@pytest.mark.parametrize("family", ["independence", "clayton", "gumbel", "frank", "joe", "learned"])
def test_forward_works_for_every_copula_family(family):
    X, T, _ = _data(n=20, K=2)
    model = _make_model(family=family)
    out = model(X, T)
    assert torch.isfinite(out["joint"]).all()


# Loss


def test_loss_functional_and_class_wrapper_agree():
    X, T, E = _data(n=30, K=2)
    model = _make_model()
    outputs = model(X, T)
    torch.testing.assert_close(
        hacsurv_nll(outputs, E),
        HACSurvLoss()(outputs, E),
    )


def test_loss_reduction_options():
    X, T, E = _data(n=30, K=2)
    model = _make_model()
    out = model(X, T)
    l_mean = hacsurv_nll(out, E, reduction="mean")
    l_sum = hacsurv_nll(out, E, reduction="sum")
    l_none = hacsurv_nll(out, E, reduction="none")
    assert l_none.shape == (30,)
    torch.testing.assert_close(l_sum, l_none.sum())
    torch.testing.assert_close(l_mean, l_none.mean())


def test_loss_gradient_flows_to_all_components():
    X, T, E = _data(n=40, K=2)
    model = _make_model(family="clayton")
    out = model(X, T)
    loss = hacsurv_nll(out, E)
    loss.backward()
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert any((g != 0).any() for g in enc_grads), "encoder got no grad"
    for k in range(model.n_causes):
        net_grads = [
            p.grad for p in model.marginal_nets[k].parameters() if p.grad is not None
        ]
        assert any((g != 0).any() for g in net_grads), f"marginal {k}: no grad"
    assert model.raw_theta.grad is not None and model.raw_theta.grad != 0


def test_loss_gradient_flows_through_learned_generator():
    X, T, E = _data(n=40, K=2)
    model = _make_model(family="learned")
    out = model(X, T)
    loss = hacsurv_nll(out, E)
    loss.backward()
    gen_grads = [
        p.grad
        for p in model.learned_generator._weight_net.parameters()
        if p.grad is not None
    ]
    assert any((g != 0).any() for g in gen_grads), "learned generator got no grad"


# Trainer integration


def test_trainer_runs_one_step_without_error():
    X, T, E = _data(n=30, K=2)
    model = _make_model()
    trainer = HACSurvTrainer(model, loss_fn=hacsurv_nll, lr=1e-3)
    history = trainer.fit((X, T, E), epochs=2, verbose=False)
    assert len(history["train_loss"]) == 2
    assert all(np.isfinite(history["train_loss"]))


def test_trainer_raises_without_loss_fn():
    X, T, E = _data(n=10, K=2)
    model = _make_model()
    with pytest.raises(NotImplementedError, match="loss_fn"):
        HACSurvTrainer(model).fit((X, T, E), epochs=1, verbose=False)


# Predict


def test_predict_cif_shape_and_bounds():
    model = _make_model()
    model.set_time_grid(np.linspace(0.1, 4.0, 10))
    X = torch.randn(5, 5)
    cif = model.predict_cif(X)
    assert cif.shape == (5, 2, 10)
    assert (cif >= -1e-6).all()
    # Marginal CIF (sum over causes) bounded by 1.
    assert cif.sum(axis=1).max() <= 1.0 + 1e-6


def test_predict_cif_per_cause_slice():
    model = _make_model()
    grid = np.linspace(0.1, 4.0, 6)
    model.set_time_grid(grid)
    X = torch.randn(3, 5)
    full = model.predict_cif(X)
    one = model.predict_cif(X, cause=1)
    np.testing.assert_array_equal(one, full[:, 0, :])


def test_predict_cif_requires_time_grid():
    model = _make_model()
    X = torch.randn(3, 5)
    with pytest.raises(RuntimeError, match="times_"):
        model.predict_cif(X)


# Config + save/load


def test_config_rejects_n_causes_below_two():
    with pytest.raises(ValueError, match="n_causes"):
        HACSurv(in_features=5, n_causes=1)


def test_config_rejects_unknown_copula_family():
    with pytest.raises(ValueError, match="copula_family"):
        HACSurv(in_features=5, n_causes=2, copula_family="zzz")


def test_config_kwarg_form_equivalent_to_dataclass():
    cfg = HACSurvConfig(in_features=5, n_causes=2, t_max=3.0)
    m1 = HACSurv(cfg)
    m2 = HACSurv(in_features=5, n_causes=2, t_max=3.0)
    assert m1.config == m2.config


def test_save_pretrained_round_trips_config_weights_and_time_grid(tmp_path):
    model = _make_model(family="clayton")
    grid = np.linspace(0.5, 4.0, 8)
    model.set_time_grid(grid)
    theta_before = float(model.theta.detach())

    model.save_pretrained(tmp_path / "hs")
    restored = HACSurv.from_pretrained(tmp_path / "hs")
    assert restored.config == model.config
    assert abs(float(restored.theta.detach()) - theta_before) < 1e-6
    np.testing.assert_array_equal(restored.times_, grid)


def test_kendalls_tau_is_finite_for_all_families():
    """Sanity: Kendall's τ is computable from a freshly-initialised model
    for every supported copula family."""
    for family in ["clayton", "gumbel", "frank", "joe", "learned"]:
        model = _make_model(family=family)
        tau = model.kendalls_tau()
        assert np.isfinite(tau), f"{family}: tau is {tau}"
        assert -1.0 - 1e-6 <= tau <= 1.0 + 1e-6
