"""Tests for causurv.metrics.pehe — PEHE, integrated PEHE, ATE error."""

from __future__ import annotations

import numpy as np
import pytest

from causurv.metrics.pehe import ate_error, integrated_pehe, pehe
from causurv.predictor import HTEEstimates


def test_pehe_zero_when_identical():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, 7))
    assert np.allclose(pehe(a, a), 0.0)


def test_pehe_is_constant_shift_when_inputs_differ_by_constant():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(100, 5))
    b = a + 0.3  # constant per-subject, per-time shift
    np.testing.assert_allclose(pehe(b, a), 0.3 * np.ones(5))


def test_pehe_pointwise_shape_for_2d_inputs():
    a = np.zeros((20, 6))
    b = np.ones((20, 6))
    out = pehe(a, b)
    assert out.shape == (6,)
    np.testing.assert_allclose(out, np.ones(6))


def test_pehe_scalar_for_1d_inputs():
    """rmst_diff-style: per-subject scalar effects → scalar PEHE."""
    a = np.zeros(50)
    b = np.full(50, 2.0)
    out = pehe(a, b)
    assert np.isscalar(out) or out.ndim == 0
    assert float(out) == pytest.approx(2.0)


def test_pehe_matches_manual_rmse_per_time():
    rng = np.random.default_rng(0)
    hat = rng.normal(size=(80, 4))
    true = rng.normal(size=(80, 4))
    manual = np.sqrt(((hat - true) ** 2).mean(axis=0))
    np.testing.assert_allclose(pehe(hat, true), manual)


def test_pehe_accepts_hte_estimates():
    rng = np.random.default_rng(0)
    vals_hat = rng.normal(size=(30, 5))
    vals_true = vals_hat + 0.1
    times = np.linspace(1, 5, 5)
    h_hat = HTEEstimates(
        values=vals_hat, contrast="survival_diff", treatment=1, reference=0, times=times
    )
    h_true = HTEEstimates(
        values=vals_true, contrast="survival_diff", treatment=1, reference=0, times=times
    )
    np.testing.assert_allclose(pehe(h_hat, h_true), 0.1 * np.ones(5))


def test_pehe_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        pehe(np.zeros((10, 5)), np.zeros((10, 4)))


def test_pehe_rejects_contrast_mismatch_on_hte_estimates():
    a = HTEEstimates(
        values=np.zeros((5, 3)), contrast="survival_diff", treatment=1, reference=0
    )
    b = HTEEstimates(
        values=np.zeros((5, 3)), contrast="cif_diff", treatment=1, reference=0
    )
    with pytest.raises(ValueError, match="contrast mismatch"):
        pehe(a, b)


def test_pehe_rejects_time_grid_mismatch_on_hte_estimates():
    a = HTEEstimates(
        values=np.zeros((5, 3)),
        contrast="survival_diff",
        treatment=1,
        reference=0,
        times=np.array([1.0, 2.0, 3.0]),
    )
    b = HTEEstimates(
        values=np.zeros((5, 3)),
        contrast="survival_diff",
        treatment=1,
        reference=0,
        times=np.array([1.0, 2.5, 3.0]),
    )
    with pytest.raises(ValueError, match="time grids disagree"):
        pehe(a, b)


def test_integrated_pehe_matches_trapezoidal_of_pointwise():
    rng = np.random.default_rng(0)
    hat = rng.normal(size=(40, 6))
    true = rng.normal(size=(40, 6))
    times = np.array([1.0, 2.0, 4.0, 5.0, 7.0, 10.0])
    expected = float(np.trapezoid(pehe(hat, true), x=times))
    assert integrated_pehe(hat, true, times) == pytest.approx(expected)


def test_integrated_pehe_zero_when_identical():
    times = np.linspace(1, 10, 5)
    a = np.zeros((20, 5))
    assert integrated_pehe(a, a, times) == 0.0


def test_integrated_pehe_picks_up_times_from_hte_estimates():
    rng = np.random.default_rng(0)
    times = np.linspace(1, 5, 4)
    vals_hat = rng.normal(size=(20, 4))
    vals_true = rng.normal(size=(20, 4))
    h_hat = HTEEstimates(
        values=vals_hat, contrast="survival_diff", treatment=1, reference=0, times=times
    )
    h_true = HTEEstimates(
        values=vals_true, contrast="survival_diff", treatment=1, reference=0, times=times
    )
    expected = float(np.trapezoid(pehe(vals_hat, vals_true), x=times))
    assert integrated_pehe(h_hat, h_true) == pytest.approx(expected)


def test_integrated_pehe_respects_horizon():
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    hat = np.ones((10, 5))  # PEHE(t) = 1 at every t
    true = np.zeros((10, 5))
    # ∫_0^3 1 dt over the kept grid {1,2,3} = trapezoidal = 2.0
    assert integrated_pehe(hat, true, times, horizon=3.0) == pytest.approx(2.0)
    # Full integral over {1..5} = 4.0
    assert integrated_pehe(hat, true, times) == pytest.approx(4.0)


def test_integrated_pehe_requires_2d():
    with pytest.raises(ValueError, match="pointwise"):
        integrated_pehe(np.zeros(10), np.zeros(10), times=np.array([1.0]))


def test_integrated_pehe_requires_times():
    with pytest.raises(ValueError, match="needs `times`"):
        integrated_pehe(np.zeros((5, 3)), np.zeros((5, 3)))


def test_integrated_pehe_rejects_times_length_mismatch():
    with pytest.raises(ValueError, match="length .* but HTE"):
        integrated_pehe(
            np.zeros((5, 3)), np.zeros((5, 3)), times=np.array([1.0, 2.0])
        )


def test_ate_error_zero_when_identical():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, 4))
    np.testing.assert_allclose(ate_error(a, a), np.zeros(4))


def test_ate_error_picks_up_mean_shift():
    a = np.zeros((100, 3))
    b = np.full((100, 3), 0.5)
    np.testing.assert_allclose(ate_error(a, b), 0.5 * np.ones(3))


def test_ate_error_zero_when_means_match_but_heterogeneity_doesnt():
    """Crucial: ATE error misses heterogeneity. A wildly heterogeneous
    estimator can still ace ATE if the *mean* over X is right."""
    true = np.zeros((100, 1))
    hat = np.concatenate([np.full((50, 1), 1.0), np.full((50, 1), -1.0)])  # mean 0
    np.testing.assert_allclose(ate_error(hat, true), np.array([0.0]))
    # But PEHE catches it:
    assert pehe(hat, true)[0] == pytest.approx(1.0)


def test_ate_error_scalar_for_1d_inputs():
    a = np.zeros(20)
    b = np.full(20, 1.0)
    out = ate_error(a, b)
    assert np.isscalar(out) or out.ndim == 0
    assert float(out) == pytest.approx(1.0)


def test_pehe_against_survite_oracle_with_oracle_as_both_inputs():
    """Sanity check: an oracle benchmarked against itself has PEHE == 0."""
    from causurv.simulations import SurvITE

    sim = SurvITE(scenario="S4")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, sim.n_features))
    times = np.array([5.0, 10.0, 20.0])
    h = sim.predict_hte(X, times)
    np.testing.assert_allclose(pehe(h, h), np.zeros(3), atol=1e-12)
    assert integrated_pehe(h, h) == pytest.approx(0.0)


def test_pehe_matches_catenets_sqrt_pehe():
    """Numerical parity against CATENets' ``sqrt_PEHE``.

    CATENets is the official reference implementation from Alicia Curth
    (SurvITE first author). Its ``sqrt_PEHE(po, hat_te)`` takes
    potential outcomes ``po`` of shape ``(n, 2)`` and predicted effects
    ``hat_te`` of shape ``(n,)``, returning a single scalar:

        sqrt(mean(((po[:, 1] - po[:, 0]) - hat_te) ** 2))

    That is exactly :func:`pehe` for time-collapsed effects, with the
    true effect derived from the potential outcomes. This test catches
    convention drift (e.g., accidentally using squared form instead of
    root form).
    """
    catenets_tester = pytest.importorskip(
        "catenets.experiment_utils.tester",
        reason="CATENets is an optional cross-validation dep",
    )
    rng = np.random.default_rng(0)
    n = 200
    po_0 = rng.normal(size=n)
    po_1 = rng.normal(size=n) + 0.3
    hat_te = rng.normal(size=n)  # noisy estimate
    true_te = po_1 - po_0

    po = np.stack([po_0, po_1], axis=1)
    ref = float(catenets_tester.sqrt_PEHE(po, hat_te))
    ours = float(pehe(hat_te, true_te))
    assert ours == pytest.approx(ref, rel=1e-6)


def test_pehe_matches_catenets_with_signal():
    """Same parity check on a realistic signal (correlated estimates)."""
    catenets_tester = pytest.importorskip("catenets.experiment_utils.tester")
    rng = np.random.default_rng(42)
    n = 500
    po_0 = rng.normal(size=n)
    po_1 = po_0 + 0.5 + 0.2 * rng.normal(size=n)  # heterogeneous treatment effect
    hat_te = (po_1 - po_0) + 0.1 * rng.normal(size=n)

    po = np.stack([po_0, po_1], axis=1)
    ref = float(catenets_tester.sqrt_PEHE(po, hat_te))
    ours = float(pehe(hat_te, po_1 - po_0))
    assert ours == pytest.approx(ref, rel=1e-6)
