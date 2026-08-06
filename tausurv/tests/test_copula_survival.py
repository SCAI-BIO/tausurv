"""Tests for the CopulaSurv model (Zhang 2023).

Covers: forward returns the expected dict, loss is functional and
swappable (matches the rest of ``tausurv.nn.functional``), training
reduces loss, copula θ stays in the family's valid range, the
``independence`` family recovers an independent-censoring baseline,
save/load round-trips including θ, and the inherited
:class:`SurvivalPredictor` prediction API holds.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tausurv import simulations
from tausurv.nn import (
    CopulaSurv,
    CopulaSurvConfig,
    CopulaSurvLoss,
    CopulaSurvTrainer,
)
from tausurv.nn.functional import copula_survival_nll


def _data(n=300, seed=0):
    X, T, E = simulations.single_risk(n=n, n_features=5, censoring_rate=0.3, seed=seed)
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(T, dtype=torch.float32),
        torch.tensor(E, dtype=torch.long),
    )


def _make_model(family="clayton", seed=0, **kw):
    torch.manual_seed(seed)
    return CopulaSurv(in_features=5, t_max=5.0, copula_family=family, **kw)


def test_forward_returns_dict_with_required_keys():
    X, T, _ = _data(n=20)
    model = _make_model()
    out = model(X, T)
    for k in ("S_T", "S_C", "f_T", "f_C", "joint", "dC_dST", "dC_dSC", "predictions"):
        assert k in out, f"forward output missing key {k!r}"
        assert out[k].shape == (20,), f"{k}: got {out[k].shape}"


def test_forward_outputs_are_in_expected_ranges():
    X, T, _ = _data(n=50)
    model = _make_model()
    out = model(X, T)
    eps = 1e-6
    assert ((out["S_T"] >= -eps) & (out["S_T"] <= 1 + eps)).all()
    assert ((out["S_C"] >= -eps) & (out["S_C"] <= 1 + eps)).all()
    assert (out["f_T"] >= -eps).all()  # densities non-negative (allow float slop)
    assert (out["f_C"] >= -eps).all()
    assert ((out["joint"] >= -eps) & (out["joint"] <= 1 + eps)).all()


def test_loss_is_functional_and_class_wrapper_agree():
    X, T, E = _data(n=30)
    model = _make_model()
    outputs = model(X, T)
    l_fn = copula_survival_nll(outputs, E)
    l_cls = CopulaSurvLoss()(outputs, E)
    torch.testing.assert_close(l_fn, l_cls)


def test_loss_reduction_options():
    X, T, E = _data(n=30)
    model = _make_model()
    outputs = model(X, T)
    l_mean = copula_survival_nll(outputs, E, reduction="mean")
    l_sum = copula_survival_nll(outputs, E, reduction="sum")
    l_none = copula_survival_nll(outputs, E, reduction="none")
    assert l_none.shape == (30,)
    torch.testing.assert_close(l_sum, l_none.sum())
    torch.testing.assert_close(l_mean, l_none.mean())


def test_loss_gradient_flows_to_all_components():
    X, T, E = _data(n=100)
    model = _make_model()
    outputs = model(X, T)
    loss = copula_survival_nll(outputs, E)
    loss.backward()
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert any((g != 0).any() for g in enc_grads), "encoder got no gradient"
    for net_name in ("monotone_T", "monotone_C"):
        net = getattr(model, net_name)
        net_grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert any((g != 0).any() for g in net_grads), f"{net_name}: no gradient"
    assert model.raw_theta.grad is not None and model.raw_theta.grad != 0


def test_trainer_with_functional_loss_reduces_loss():
    X, T, E = _data()
    model = _make_model()
    trainer = CopulaSurvTrainer(model, loss_fn=copula_survival_nll, lr=1e-2)
    history = trainer.fit((X, T, E), epochs=20, verbose=False)
    assert history["train_loss"][-1] < history["train_loss"][0] - 0.5


def test_trainer_with_class_loss_matches_functional():
    """The class wrapper and functional loss must produce identical
    training trajectories — they wrap the same math."""
    X, T, E = _data()

    m1 = _make_model()
    h1 = CopulaSurvTrainer(m1, loss_fn=copula_survival_nll, lr=1e-2).fit(
        (X, T, E), epochs=10, verbose=False
    )

    m2 = _make_model()
    h2 = CopulaSurvTrainer(m2, loss_fn=CopulaSurvLoss(reduction="mean"), lr=1e-2).fit(
        (X, T, E), epochs=10, verbose=False
    )

    np.testing.assert_allclose(h1["train_loss"], h2["train_loss"], atol=1e-5)


def test_trainer_uses_eval_step_with_val_data():
    X, T, E = _data()
    Xv, Tv, Ev = _data(seed=1)
    model = _make_model()
    trainer = CopulaSurvTrainer(model, loss_fn=copula_survival_nll, lr=1e-2)
    history = trainer.fit((X, T, E), (Xv, Tv, Ev), epochs=5, verbose=False)
    assert len(history["val_loss"]) == 5
    assert all(v is not None and np.isfinite(v) for v in history["val_loss"])


def test_trainer_raises_without_loss_fn():
    X, T, E = _data()
    model = _make_model()
    trainer = CopulaSurvTrainer(model, lr=1e-2)
    with pytest.raises(NotImplementedError, match="loss_fn"):
        trainer.fit((X, T, E), epochs=1, verbose=False)


def test_independence_family_has_constant_zero_theta():
    X, T, E = _data()
    model = _make_model(family="independence")
    assert float(model.theta.detach()) == 0.0
    trainer = CopulaSurvTrainer(model, loss_fn=copula_survival_nll, lr=1e-2)
    trainer.fit((X, T, E), epochs=10, verbose=False)
    assert float(model.theta.detach()) == 0.0


@pytest.mark.parametrize(
    "family,theta_lower",
    [
        ("clayton", 0.0),
        ("gumbel", 1.0),
        ("joe", 1.0),
    ],
)
def test_theta_stays_in_valid_range_during_training(family, theta_lower):
    X, T, E = _data(n=200)
    model = _make_model(family=family, copula_theta_init=2.0)
    trainer = CopulaSurvTrainer(model, loss_fn=copula_survival_nll, lr=1e-1)
    trainer.fit((X, T, E), epochs=15, verbose=False)
    assert float(model.theta.detach()) > theta_lower


def test_predict_survival_function_shape_and_monotonicity():
    model = _make_model()
    X = torch.randn(10, 5)
    times = np.linspace(0.1, 4.0, 12)
    S = model.predict_survival_function(X, times)
    assert S.shape == (10, 12)
    assert ((S >= 0) & (S <= 1)).all()
    assert (np.diff(S, axis=1) <= 1e-6).all()


def test_predict_survival_function_uses_default_times_after_set_time_grid():
    model = _make_model()
    model.set_time_grid(np.linspace(0.5, 3.0, 8))
    X = torch.randn(5, 5)
    S = model.predict_survival_function(X)
    assert S.shape == (5, 8)


def test_set_time_grid_round_trips_through_save_load(tmp_path):
    model = _make_model()
    grid = np.linspace(0.5, 3.0, 8)
    model.set_time_grid(grid)
    model.save(tmp_path / "cs")
    restored = CopulaSurv.load(tmp_path / "cs")
    np.testing.assert_array_equal(restored.times_, grid)


def test_predict_requires_times_or_times_attribute():
    model = _make_model()
    X = torch.randn(5, 5)
    with pytest.raises(RuntimeError, match="times_"):
        model.predict_survival_function(X)


def test_save_round_trips_config_and_theta(tmp_path):
    X, T, E = _data()
    model = _make_model(family="clayton", copula_theta_init=2.5)
    trainer = CopulaSurvTrainer(model, loss_fn=copula_survival_nll, lr=1e-2)
    trainer.fit((X, T, E), epochs=5, verbose=False)
    theta_before = float(model.theta.detach())

    model.save(tmp_path / "cs")
    restored = CopulaSurv.load(tmp_path / "cs")
    theta_after = float(restored.theta.detach())
    assert abs(theta_after - theta_before) < 1e-6
    assert restored.config.copula_family == "clayton"

    grid = np.linspace(0.5, 4.0, 6)
    np.testing.assert_allclose(
        model.predict_survival_function(X[:5].numpy(), grid),
        restored.predict_survival_function(X[:5].numpy(), grid),
        atol=1e-6,
    )


def test_config_kwarg_form_equivalent_to_dataclass():
    cfg = CopulaSurvConfig(in_features=5, t_max=3.0)
    m1 = CopulaSurv(cfg)
    m2 = CopulaSurv(in_features=5, t_max=3.0)
    assert m1.config == m2.config


def test_config_rejects_unknown_copula_family():
    with pytest.raises(ValueError, match="copula_family"):
        CopulaSurv(in_features=5, copula_family="zzz")
