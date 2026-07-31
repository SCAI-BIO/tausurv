"""Cross-model uniform-API tests.

Every fitted survival predictor in tausurv should expose the same shape of
prediction methods: ``predict_survival_function``, ``predict_cumulative_hazard``,
``predict_cif``, ``predict_rmst``, ``predict_risk_at``, ``predict``. These
tests exercise the contract uniformly across CoxPH, SurvivalTree,
RandomSurvivalForest, DeepSurv (after ``fit_baseline``), DeepHit, and
LogisticHazard, plus a competing-risks-specific path for DeepHit.
"""

from __future__ import annotations

import numpy as np
import pytest

from tausurv import simulations
from tausurv.linear import CoxPH, LogLogisticAFT, LogNormalAFT, WeibullAFT
from tausurv.predictor import CompetingRisksPredictor, SurvivalPredictor
from tausurv.trees.random_survival_forest import RandomSurvivalForest
from tausurv.trees.survival_tree import SurvivalTree
from tausurv.trees.survival_boost import SurvivalBoost

torch = pytest.importorskip("torch")

from tausurv.nn import DeepHit, DeepSurv, LogisticHazard, fit, functional


def _xy():
    X, T, E = simulations.single_risk(n=300, n_features=5, seed=0)
    return X, T, E


def _xy_torch():
    X, T, E = _xy()
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(T, dtype=torch.float32),
        torch.tensor(E, dtype=torch.float32),
    )


def _coxph():
    X, T, E = _xy()
    return CoxPH().fit(X, T, E), X


def _weibull_aft():
    X, T, E = _xy()
    return WeibullAFT().fit(X, T, E), X


def _lognormal_aft():
    X, T, E = _xy()
    return LogNormalAFT().fit(X, T, E), X


def _loglogistic_aft():
    X, T, E = _xy()
    return LogLogisticAFT().fit(X, T, E), X


def _survival_tree():
    X, T, E = _xy()
    return SurvivalTree(max_depth=4, min_samples_leaf=10, seed=0).fit(X, T, E), X


def _rsf():
    X, T, E = _xy()
    return RandomSurvivalForest(
        n_estimators=10, max_depth=3, min_samples_leaf=10, seed=0
    ).fit(X, T, E), X


def _survival_boost():
    pytest.importorskip("sklearn")
    X, T, E = _xy()
    return SurvivalBoost(n_iter=20, seed=0).fit(X, T, E), X


def _deepsurv():
    """A predictor-API-conformant DeepSurv. Construct + fit_baseline only —
    skip SGD training (these tests check the predict contract, not learning)."""
    torch.manual_seed(0)
    X_t, T_t, E_t = _xy_torch()
    model = DeepSurv(in_features=5, hidden_dim=16, n_blocks=2, dropout=0.0)
    model.fit_baseline(X_t, T_t.numpy(), E_t.numpy().astype(int))
    return model, X_t


def _deephit_single_event():
    """Construct + set_time_grid only; predict-API tests don't require
    a trained model, just one with the time-grid state set."""
    torch.manual_seed(0)
    X_t, T_t, _ = _xy_torch()
    times = np.quantile(T_t.numpy(), np.linspace(0.1, 0.95, 10))
    model = DeepHit(in_features=5, n_bins=10, hidden_dim=16, dropout=0.0)
    model.set_time_grid(times)
    return model, X_t


def _logistic_hazard():
    torch.manual_seed(0)
    X_t, T_t, _ = _xy_torch()
    times = np.quantile(T_t.numpy(), np.linspace(0.1, 0.95, 10))
    model = LogisticHazard(in_features=5, n_bins=10, hidden_dim=16, dropout=0.0)
    model.set_time_grid(times)
    return model, X_t


ALL_FACTORIES = [
    ("CoxPH", _coxph),
    ("WeibullAFT", _weibull_aft),
    ("LogNormalAFT", _lognormal_aft),
    ("LogLogisticAFT", _loglogistic_aft),
    ("SurvivalTree", _survival_tree),
    ("RandomSurvivalForest", _rsf),
    ("SurvivalBoost", _survival_boost),
    ("DeepSurv", _deepsurv),
    ("DeepHit", _deephit_single_event),
    ("LogisticHazard", _logistic_hazard),
]


@pytest.mark.parametrize("name,factory", ALL_FACTORIES)
def test_predictor_is_a_survival_predictor(name, factory):
    model, _ = factory()
    assert isinstance(model, SurvivalPredictor), (
        f"{name} must inherit from SurvivalPredictor for unified-API guarantees"
    )


@pytest.mark.parametrize("name,factory", ALL_FACTORIES)
def test_survival_function_shape_and_bounds(name, factory):
    model, X = factory()
    n = len(X) if not torch.is_tensor(X) else X.shape[0]
    grid = np.linspace(0.1, 3.0, 12)
    S = model.predict_survival_function(X, grid)
    assert S.shape == (n, 12), f"{name}: got {S.shape}"
    # SurvivalPredictor.predict_survival_function clips to [0, 1] — the
    # bound holds exactly, no float slop.
    assert ((S >= 0.0) & (S <= 1.0)).all(), f"{name}: S out of [0, 1]"


@pytest.mark.parametrize("name,factory", ALL_FACTORIES)
def test_survival_function_non_increasing(name, factory):
    model, X = factory()
    grid = np.linspace(0.1, 3.0, 12)
    S = model.predict_survival_function(X, grid)
    # For NN models the survival comes from a float-32 cumsum / cumprod that
    # can produce ~1e-7 noise across adjacent times even when the math is
    # exactly monotone. For non-NN models the math is float-64 exact.
    slop = 1e-6 if name in {"DeepSurv", "DeepHit", "LogisticHazard"} else 0.0
    assert (np.diff(S, axis=1) <= slop).all(), f"{name}: S not non-increasing"


@pytest.mark.parametrize("name,factory", ALL_FACTORIES)
def test_survival_function_uses_times_when_default(name, factory):
    model, X = factory()
    S = model.predict_survival_function(X)
    assert S.shape[1] == len(model.times_), f"{name}: default times_ shape mismatch"


@pytest.mark.parametrize("name,factory", ALL_FACTORIES)
def test_cumulative_hazard_non_negative_and_monotone(name, factory):
    model, X = factory()
    grid = np.linspace(0.1, 3.0, 12)
    H = model.predict_cumulative_hazard(X, grid)
    # H = -log(clip(S, 1e-12, 1)) ≥ 0 exactly.
    assert (H >= 0.0).all(), f"{name}: H not non-negative"
    # Monotonicity inherits the float slop of the underlying survival
    # function — see test_survival_function_non_increasing.
    slop = 1e-6 if name in {"DeepSurv", "DeepHit", "LogisticHazard"} else 0.0
    assert (np.diff(H, axis=1) >= -slop).all(), f"{name}: H not non-decreasing"


@pytest.mark.parametrize("name,factory", ALL_FACTORIES)
def test_cif_equals_one_minus_survival(name, factory):
    model, X = factory()
    grid = np.linspace(0.1, 3.0, 8)
    S = model.predict_survival_function(X, grid)
    F = model.predict_cif(X, grid)
    # For competing-risks models, predict_cif returns (n, K, T); marginal F is sum across K.
    if F.ndim == 3:
        F = F.sum(axis=1)
    np.testing.assert_allclose(F, 1.0 - S, atol=1e-9)


def test_single_event_cif_rejects_invalid_cause():
    model, X = _coxph()
    with pytest.raises(ValueError, match="single-event"):
        model.predict_cif(X, [1.0], cause=2)


@pytest.mark.parametrize("name,factory", ALL_FACTORIES)
def test_rmst_in_zero_horizon_interval(name, factory):
    model, X = factory()
    horizon = 1.0
    rmst = model.predict_rmst(X, horizon=horizon)
    n = len(X) if not torch.is_tensor(X) else X.shape[0]
    assert rmst.shape == (n,), f"{name}: rmst shape mismatch"
    # 0 <= RMST <= horizon (since 0 <= S <= 1, ∫_0^τ S ≤ τ).
    assert (rmst >= -1e-9).all() and (rmst <= horizon + 1e-6).all()


@pytest.mark.parametrize("name,factory", ALL_FACTORIES)
def test_risk_at_in_unit_interval(name, factory):
    model, X = factory()
    risk = model.predict_risk_at(X, time=1.0)
    n = len(X) if not torch.is_tensor(X) else X.shape[0]
    assert risk.shape == (n,)
    # risk = 1 - clip(S, 0, 1) ∈ [0, 1] exactly.
    assert ((risk >= 0.0) & (risk <= 1.0)).all()


@pytest.mark.parametrize("name,factory", ALL_FACTORIES)
def test_predict_returns_scalar_per_sample(name, factory):
    model, X = factory()
    risk = model.predict(X)
    n = len(X) if not torch.is_tensor(X) else X.shape[0]
    assert risk.shape == (n,), f"{name}: predict() shape mismatch"


def _deephit_competing_risks():
    """Construct + set_time_grid only; predict-API tests check that the
    competing-risks contract holds, not that the model learned anything."""
    torch.manual_seed(0)
    X, T, _ = simulations.competing_risk(n=300, n_features=5, seed=0)
    X_t = torch.tensor(X, dtype=torch.float32)
    times = np.quantile(T, np.linspace(0.1, 0.95, 8))
    model = DeepHit(in_features=5, n_bins=8, n_causes=2, hidden_dim=16, dropout=0.0)
    model.set_time_grid(times)
    return model, X_t


def test_competing_risks_cif_shape_and_bounds():
    model, X = _deephit_competing_risks()
    assert isinstance(model, CompetingRisksPredictor)
    grid = np.linspace(0.1, 2.0, 6)
    cif = model.predict_cif(X, grid)
    assert cif.shape == (X.shape[0], 2, 6)
    # CompetingRisksPredictor.predict_cif clips per-cause and rescales rows
    # whose marginal exceeds 1 — both bounds hold exactly post-clip.
    assert (cif >= 0.0).all()
    assert cif.sum(axis=1).max() <= 1.0
    # Per-cause CIF is non-decreasing in t up to float-32 cumsum noise.
    assert (np.diff(cif, axis=-1) >= -1e-6).all()


def test_competing_risks_cif_per_cause_slice():
    model, X = _deephit_competing_risks()
    grid = np.linspace(0.1, 2.0, 6)
    full = model.predict_cif(X, grid)
    one = model.predict_cif(X, grid, cause=1)
    two = model.predict_cif(X, grid, cause=2)
    np.testing.assert_array_equal(one, full[:, 0, :])
    np.testing.assert_array_equal(two, full[:, 1, :])


def test_competing_risks_cif_rejects_invalid_cause():
    model, X = _deephit_competing_risks()
    with pytest.raises(ValueError, match=r"cause must be in"):
        model.predict_cif(X, [1.0], cause=5)


def test_deepsurv_predict_requires_no_baseline():
    """`predict()` returns log-risk and works immediately after training,
    no baseline needed."""
    torch.manual_seed(0)
    model = DeepSurv(in_features=5, hidden_dim=8, dropout=0.0)
    X_t = torch.randn(10, 5)
    risk = model.predict(X_t)
    assert risk.shape == (10,)


def test_deepsurv_predict_survival_function_requires_baseline():
    torch.manual_seed(0)
    model = DeepSurv(in_features=5, hidden_dim=8, dropout=0.0)
    X_t = torch.randn(10, 5)
    with pytest.raises(RuntimeError, match="fit_baseline"):
        model.predict_survival_function(X_t, [1.0])


def test_deepsurv_baseline_round_trips_through_save_load(tmp_path):
    model, X = _deepsurv()
    grid = np.linspace(0.1, 2.0, 6)
    S_before = model.predict_survival_function(X, grid)

    model.save_pretrained(tmp_path / "ds")
    assert (tmp_path / "ds" / "baseline.npz").exists()

    restored = DeepSurv.from_pretrained(tmp_path / "ds")
    restored.eval()
    S_after = restored.predict_survival_function(X, grid)
    np.testing.assert_allclose(S_after, S_before, atol=1e-6)
