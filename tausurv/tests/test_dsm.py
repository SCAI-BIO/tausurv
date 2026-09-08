"""Tests for DSM (math + architecture invariants).

Training-based smoke checks live in scripts/nn_training_smoke.py per
the project convention. These tests verify:

- Loss matches a hand-computed mixture NLL on a toy case.
- Loss functional + class wrapper agree.
- predict_survival_function returns shapes in [0, 1], monotone in t.
- Loss gradient flows to encoder + head.
- Config validation, save/load round-trip (including time grid),
  predictor API conformance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F

from tausurv.nn import DSM, DSMConfig, DSMLoss, fit
from tausurv.nn.distributions import LogNormal, Weibull
from tausurv.nn.functional import dsm_nll


def _make_model(family="weibull", n_components=4, seed=0, **kw):
    torch.manual_seed(seed)
    return DSM(
        in_features=5, n_components=n_components, distribution_family=family, **kw
    )


# Hand-computed mixture NLL


def test_dsm_nll_matches_hand_computed_mixture_weibull():
    """One subject, K=2 Weibull components, known params, event observed.

    With raw params (shape, scale) = ((1.0, 1.0), (0.0, 0.0)) after
    softplus+eps: shape ≈ (1.31, 1.31), scale ≈ (0.69, 0.69). Equal
    weights via raw logits (0, 0) → w = (0.5, 0.5). Event at t=1.0.

    Hand-computed: log f(1.0) = log[0.5 * f_1(1.0) + 0.5 * f_2(1.0)]
    where f_k are Weibull pdfs.
    """
    eps = 1e-4
    shape = F.softplus(torch.tensor(1.0)) + eps   # scalar, applies to both K
    scale = F.softplus(torch.tensor(0.0)) + eps   # both components identical
    w = torch.tensor([0.5, 0.5])

    t = torch.tensor([1.0])
    expected_f = float(Weibull(shape, scale).pdf(t).item())  # both components agree
    expected_log_f = math.log(expected_f)  # log(w_1 f_1 + w_2 f_2) = log(f) since both =

    predictions = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])  # (1, 3*K=6)
    nll = dsm_nll(predictions, t, torch.tensor([1.0]), reduction="none")
    np.testing.assert_allclose(float(nll.item()), -expected_log_f, atol=1e-6)


def test_dsm_nll_censored_uses_survival():
    """For a censored subject the loss must use log S(t), not log f(t)."""
    eps = 1e-4
    predictions = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    t = torch.tensor([1.0])

    nll_censored = dsm_nll(predictions, t, torch.tensor([0.0]), reduction="none")
    nll_event = dsm_nll(predictions, t, torch.tensor([1.0]), reduction="none")

    shape = F.softplus(torch.tensor(1.0)) + eps
    scale = F.softplus(torch.tensor(0.0)) + eps
    d = Weibull(shape, scale)
    expected_log_S = float(d.log_survival(t).item())
    expected_log_f = float(d.log_pdf(t).item())
    np.testing.assert_allclose(float(nll_censored.item()), -expected_log_S, atol=1e-6)
    np.testing.assert_allclose(float(nll_event.item()), -expected_log_f, atol=1e-6)


def test_dsm_nll_matches_hand_computed_mixture_lognormal():
    """LogNormal variant: raw p1 is mu (unconstrained), raw p2 is sigma (softplus)."""
    eps = 1e-4
    predictions = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
    t = torch.tensor([1.0])

    mu = torch.tensor(0.0)
    sigma = F.softplus(torch.tensor(1.0)) + eps
    expected_log_f = float(LogNormal(mu, sigma).log_pdf(t).item())

    nll = dsm_nll(
        predictions, t, torch.tensor([1.0]),
        distribution="lognormal", reduction="none",
    )
    np.testing.assert_allclose(float(nll.item()), -expected_log_f, atol=1e-6)


def test_dsm_nll_logsumexp_handles_imbalanced_weights():
    """When one mixture weight is dominant, the mixture log-likelihood is
    close to that single component's contribution (up to log-sum-exp slop)."""
    # raw weights (10, -10): softmax → ~(1, 0); the first component dominates.
    predictions = torch.tensor([[1.0, 2.0, 0.0, 0.5, 10.0, -10.0]])
    t = torch.tensor([1.0])

    eps = 1e-4
    shape_1 = F.softplus(torch.tensor(1.0)) + eps
    scale_1 = F.softplus(torch.tensor(0.0)) + eps
    log_f_1 = float(Weibull(shape_1, scale_1).log_pdf(t).item())

    nll = dsm_nll(predictions, t, torch.tensor([1.0]), reduction="none")
    np.testing.assert_allclose(float(nll.item()), -log_f_1, atol=1e-3)


# Loss/functional class equivalence


def test_loss_functional_and_class_wrapper_agree():
    torch.manual_seed(0)
    predictions = torch.randn(20, 12)
    t = torch.rand(20) + 0.1
    e = torch.randint(0, 2, (20,)).float()

    for family in ("weibull", "lognormal"):
        torch.testing.assert_close(
            dsm_nll(predictions, t, e, distribution=family),
            DSMLoss(distribution=family)(predictions, t, e),
        )


def test_loss_reduction_options():
    torch.manual_seed(0)
    predictions = torch.randn(20, 12)
    t = torch.rand(20) + 0.1
    e = torch.randint(0, 2, (20,)).float()

    l_mean = dsm_nll(predictions, t, e, reduction="mean")
    l_sum = dsm_nll(predictions, t, e, reduction="sum")
    l_none = dsm_nll(predictions, t, e, reduction="none")
    assert l_none.shape == (20,)
    torch.testing.assert_close(l_sum, l_none.sum())
    torch.testing.assert_close(l_mean, l_none.mean())


def test_loss_rejects_unknown_distribution():
    predictions = torch.randn(5, 12)
    t = torch.rand(5) + 0.1
    e = torch.zeros(5)
    with pytest.raises(ValueError, match="distribution must"):
        dsm_nll(predictions, t, e, distribution="exponential")


def test_loss_rejects_non_3K_predictions():
    """3K parameterization is part of the contract; reject other shapes."""
    predictions = torch.randn(5, 10)  # 10 not divisible by 3
    t = torch.rand(5) + 0.1
    e = torch.zeros(5)
    with pytest.raises(ValueError, match="divisible by 3"):
        dsm_nll(predictions, t, e)


# Forward + predict


def test_forward_shape():
    model = _make_model(n_components=5)
    out = model(torch.randn(8, 5))
    assert out.shape == (8, 15)  # 3 * 5 = 15


@pytest.mark.parametrize("family", ["weibull", "lognormal"])
def test_survival_approaches_one_as_t_goes_to_zero(family):
    """S(t) → 1 as t → 0. Convergence rate depends on the per-component
    shape parameter — for Weibull with shape < 1 the hazard is unbounded
    at t=0, so S approaches 1 slowly; a small slop reflects that.
    """
    model = _make_model(family=family)
    X = torch.randn(5, 5)
    S = model.predict_survival_function(X, np.array([1e-10]))
    assert (S > 0.99).all() and (S <= 1.0).all()


@pytest.mark.parametrize("family", ["weibull", "lognormal"])
def test_survival_monotone_non_increasing_in_t(family):
    model = _make_model(family=family)
    X = torch.randn(8, 5)
    grid = np.linspace(0.01, 20.0, 30)
    S = model.predict_survival_function(X, grid)
    assert (np.diff(S, axis=1) <= 1e-6).all()


@pytest.mark.parametrize("family", ["weibull", "lognormal"])
def test_survival_in_unit_interval(family):
    model = _make_model(family=family)
    X = torch.randn(8, 5)
    grid = np.linspace(0.1, 10.0, 12)
    S = model.predict_survival_function(X, grid)
    assert ((S >= 0.0) & (S <= 1.0)).all()


def test_constrained_params_have_correct_shapes_and_signs():
    """Internal `_constrained_params` returns positive shape/scale (Weibull)
    or positive sigma (LogNormal), and weights summing to 1."""
    model = _make_model(family="weibull", n_components=3)
    raw = torch.randn(10, 9)
    p1, p2, w = model._constrained_params(raw)
    assert p1.shape == (10, 3) and (p1 > 0).all()
    assert p2.shape == (10, 3) and (p2 > 0).all()
    torch.testing.assert_close(w.sum(dim=-1), torch.ones(10))


# Gradient flow


def test_loss_gradient_flows_to_encoder_and_head():
    X = torch.randn(20, 5)
    t = torch.rand(20) + 0.1
    e = torch.randint(0, 2, (20,)).float()
    model = _make_model()
    loss = dsm_nll(model(X), t, e)
    loss.backward()
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    head_grads = [p.grad for p in model.head.parameters() if p.grad is not None]
    assert enc_grads and any((g != 0).any() for g in enc_grads), "encoder grads"
    assert head_grads and any((g != 0).any() for g in head_grads), "head grads"


# Trainer integration (smoke — uses the standard Trainer)


def test_trainer_runs_with_standard_trainer():
    rng = np.random.default_rng(0)
    n, d = 30, 5
    X = torch.tensor(rng.normal(size=(n, d)).astype(np.float32))
    T = torch.tensor((rng.exponential(1.0, size=n) + 0.1).astype(np.float32))
    E = torch.tensor(rng.choice([0, 1], size=n).astype(np.float32))
    model = _make_model()
    history = fit(model, dsm_nll, (X, T, E), epochs=3, lr=1e-3, verbose=False)
    assert len(history["train_loss"]) == 3
    assert all(np.isfinite(history["train_loss"]))


# Config + save/load


def test_config_rejects_unknown_distribution():
    with pytest.raises(ValueError, match="distribution_family"):
        DSM(in_features=5, distribution_family="exponential")


def test_config_rejects_non_positive_n_components():
    with pytest.raises(ValueError, match="n_components"):
        DSM(in_features=5, n_components=0)


def test_config_kwarg_form_equivalent_to_dataclass():
    cfg = DSMConfig(in_features=5, n_components=3)
    m1 = DSM(cfg)
    m2 = DSM(in_features=5, n_components=3)
    assert m1.config == m2.config


def test_save_round_trips_config_weights_and_time_grid(tmp_path):
    model = _make_model(family="lognormal", n_components=3)
    grid = np.linspace(0.5, 5.0, 6)
    model.set_time_grid(grid)
    X = torch.randn(7, 5)
    S_before = model.predict_survival_function(X)

    model.save(tmp_path / "dsm")
    restored = DSM.load(tmp_path / "dsm")
    assert restored.config == model.config
    np.testing.assert_array_equal(restored.times_, grid)

    S_after = restored.predict_survival_function(X)
    np.testing.assert_allclose(S_after, S_before, atol=1e-6)


# Predictor API contract


def test_predict_methods_have_consistent_shapes():
    model = _make_model()
    model.set_time_grid(np.linspace(0.5, 4.0, 8))
    X = torch.randn(6, 5)

    S = model.predict_survival_function(X)
    assert S.shape == (6, 8)

    H = model.predict_cumulative_hazard(X)
    assert H.shape == (6, 8) and (H >= 0).all()

    risk = model.predict(X)
    assert risk.shape == (6,)

    rmst = model.predict_rmst(X, horizon=3.0)
    assert rmst.shape == (6,) and (rmst >= 0).all()


def test_predict_requires_time_grid():
    model = _make_model()
    X = torch.randn(3, 5)
    with pytest.raises(RuntimeError, match="times_"):
        model.predict_survival_function(X)


def test_dsm_fit_sets_grid_to_training_event_times():
    from tausurv import simulations

    X, T, E = simulations.single_risk(n=120, n_features=5, seed=0)
    model = DSM(in_features=5, n_components=2, hidden_features=(8,))
    model.fit(X, T, E, epochs=3)

    assert len(model.history_["train_loss"]) == 3
    np.testing.assert_array_equal(model.times_, np.unique(T[E > 0]))
    S = model.predict_survival_function(X[:4])
    assert S.shape == (4, len(model.times_))
    assert np.all((S >= 0.0) & (S <= 1.0))
