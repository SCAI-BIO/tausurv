from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def nll(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    survival: ArrayLike,
    times: ArrayLike,
) -> float:
    r"""Negative log-likelihood under a discrete-time survival model.

    For event subjects, the density at $Y_i$ is approximated by the survival
    drop across the time-grid bin containing $Y_i$:
    $\hat f(Y_i \mid x_i) \approx \hat S(t_{k-1} \mid x_i) - \hat S(t_k \mid x_i)$,
    treating $\hat S(t_{-1}) = 1$ for events below the first grid point.
    For censored subjects, the contribution is $\log \hat S(Y_i \mid x_i)$.

    $$
    \text{NLL} = -\frac{1}{n} \sum_{i=1}^n \left[
        \delta_i \log \hat f(Y_i \mid x_i)
      + (1 - \delta_i) \log \hat S(Y_i \mid x_i)
    \right]
    $$

    Parameters
    ----------
    event_time : (n,) array
    event_indicator : (n,) array
    survival : (n, T) array
    times : (T,) array

    Returns
    -------
    float
        Mean negative log-likelihood. Lower is better; 0 is the minimum
        attainable (impossible in practice).
    """
    Y = np.asarray(event_time, dtype=np.float64)
    delta = np.asarray(event_indicator, dtype=np.int8)
    S = np.asarray(survival, dtype=np.float64)
    t_grid = np.asarray(times, dtype=np.float64)

    t_idx = np.maximum(np.searchsorted(t_grid, Y, side="right") - 1, 0)
    rows = np.arange(len(Y))

    S_at_Y = S[rows, t_idx]
    S_prev = np.where(t_idx > 0, S[rows, np.maximum(t_idx - 1, 0)], 1.0)

    density = np.maximum(S_prev - S_at_Y, 1e-12)
    S_safe = np.maximum(S_at_Y, 1e-12)

    log_lik = np.where(delta == 1, np.log(density), np.log(S_safe))
    return float(-log_lik.mean())


def crps(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    survival: ArrayLike,
    times: ArrayLike,
) -> float:
    r"""Continuous Ranked Probability Score for survival predictions.

    Per-subject squared error of the predicted survival curve against the
    indicator $\mathbb{1}[Y_i > t]$, integrated over the time grid:

    $$
    \text{CRPS}_i = \int_0^{Y_i^*} \big(\hat S(t \mid x_i) - \mathbb{1}[Y_i > t]\big)^2 \, dt
    $$

    where $Y_i^* = Y_i$ for censored subjects (the indicator is unknown
    beyond $Y_i$) and $Y_i^* = t_{\max}$ for event subjects. Discretized via
    trapezoidal integration over ``times``. Returns the mean over
    subjects.

    Differs from :func:`tausurv.metrics.brier.integrated`: CRPS is a
    per-subject proper scoring rule with censoring-aware truncation; IBS is
    an IPCW-weighted population-level Brier integrated over time. Both have
    units of "time"; they answer different questions.

    Parameters
    ----------
    event_time : (n,) array
    event_indicator : (n,) array
    survival : (n, T) array
    times : (T,) array

    Returns
    -------
    float
        Mean per-subject CRPS. Lower is better.

    References
    ----------
    Avati, A. et al. (2020). Countdown Regression: Sharp and Calibrated
    Survival Predictions. Proceedings of UAI 2019, PMLR 115, 145-155.
    """
    Y = np.asarray(event_time, dtype=np.float64)
    delta = np.asarray(event_indicator, dtype=np.int8)
    S = np.asarray(survival, dtype=np.float64)
    t_grid = np.asarray(times, dtype=np.float64)

    eta = (Y[:, None] > t_grid[None, :]).astype(np.float64)
    error_sq = (S - eta) ** 2

    upper = np.where(delta == 1, t_grid[-1], Y)
    upper_idx = np.maximum(np.searchsorted(t_grid, upper, side="right") - 1, 0)

    # Include trapezoid segment (k, k+1) iff k+1 <= upper_idx for that subject.
    segment_mask = (
        np.arange(len(t_grid) - 1)[None, :] + 1 <= upper_idx[:, None]
    ).astype(np.float64)
    dt = np.diff(t_grid)
    segments = 0.5 * (error_sq[:, :-1] + error_sq[:, 1:]) * dt[None, :]
    per_subject = (segments * segment_mask).sum(axis=1)

    return float(per_subject.mean())
