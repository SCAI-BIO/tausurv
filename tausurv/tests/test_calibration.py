from __future__ import annotations

import numpy as np

from tausurv.metrics import calibration


def test_d_calibration_perfectly_calibrated_high_p():
    # If Y ~ Exp(1) and S(t) = exp(-t), then S(Y_i) ~ Uniform(0, 1).
    rng = np.random.default_rng(0)
    n = 2000
    Y = rng.exponential(1.0, n)
    delta = np.ones(n, dtype=int)
    time_grid = np.linspace(0.001, 15.0, 500)
    S = np.broadcast_to(np.exp(-time_grid)[None, :], (n, len(time_grid)))

    p = calibration.distributional(Y, delta, S, time_grid)
    # Under H0 the p-value is Uniform(0, 1), so ``p > 0.05`` passes 95% of
    # H0 draws — the standard "no evidence of miscalibration" check. A
    # tighter bound here would *raise* the H0 flake rate, not lower it.
    assert p > 0.05


def test_d_calibration_severely_miscalibrated_low_p():
    # Every prediction = 0.5; bins are concentrated -> chi-square rejects.
    rng = np.random.default_rng(0)
    n = 1000
    Y = rng.exponential(1.0, n)
    delta = np.ones(n, dtype=int)
    time_grid = np.linspace(0.01, 10.0, 200)
    S = np.full((n, len(time_grid)), 0.5)

    p = calibration.distributional(Y, delta, S, time_grid)
    assert p < 1e-3


def test_calibration_curve_matches_for_well_specified_model():
    # Two risk groups; predicted = generative survival per group.
    # Within each bin, predicted mean should match KM estimate.
    rng = np.random.default_rng(0)
    n_per_group = 200
    Y = np.concatenate(
        [rng.exponential(1.0, n_per_group), rng.exponential(5.0, n_per_group)]
    )
    delta = np.ones(2 * n_per_group, dtype=int)
    time_grid = np.linspace(0.01, 12.0, 200)
    S_low = np.exp(-time_grid)
    S_high = np.exp(-time_grid / 5.0)
    survival = np.concatenate(
        [
            np.broadcast_to(S_low[None, :], (n_per_group, len(time_grid))),
            np.broadcast_to(S_high[None, :], (n_per_group, len(time_grid))),
        ]
    )

    pred, obs, sizes = calibration.curve(Y, delta, survival, time_grid, t=3.0, n_bins=2)
    assert len(pred) == 2
    np.testing.assert_allclose(pred, obs, atol=0.07)


def test_calibration_curve_returns_empty_for_constant_predictions():
    rng = np.random.default_rng(0)
    n = 100
    Y = rng.exponential(1.0, n)
    delta = np.ones(n, dtype=int)
    time_grid = np.linspace(0.01, 10.0, 50)
    survival = np.full((n, len(time_grid)), 0.5)

    pred, obs, sizes = calibration.curve(Y, delta, survival, time_grid, t=2.0)
    assert len(pred) == 0
    assert len(obs) == 0
    assert len(sizes) == 0
