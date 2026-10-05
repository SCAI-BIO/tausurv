from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tausurv.nonparametric import censoring_distribution
from tausurv.step import StepFunction


def score(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    survival: ArrayLike,
    times: ArrayLike,
    *,
    censoring_survival: StepFunction | None = None,
) -> NDArray[np.float64]:
    r"""IPCW Brier score at each point of ``times``.

    Squared error between the true survival status at $t$ — known only for
    subjects who reached the event or were still at risk by $t$ — and the
    predicted survival $\hat S(t \mid x_i)$, reweighted by the censoring
    distribution to account for unobserved status under right-censoring:

    $$
    \widehat{BS}(t) = \frac{1}{n} \sum_{i=1}^n \left[
        \frac{\mathbb{1}[Y_i \le t,\,\delta_i = 1]\,\hat S(t \mid x_i)^2}
             {\hat G(Y_i^-)}
        + \frac{\mathbb{1}[Y_i > t]\,\bigl(1 - \hat S(t \mid x_i)\bigr)^2}
               {\hat G(t)}
    \right]
    $$

    Subjects censored before $t$ contribute zero — their status at $t$ is
    unknown and the IPCW weights for the other two groups account for them.

    Parameters
    ----------
    event_time : (n,) array
        Observed time $Y = \min(T, C)$.
    event_indicator : (n,) array
        $\delta = 1$ if event observed, $0$ if censored.
    survival : (n, T) array
        $\hat S(t \mid x_i)$ at each ``times`` point.
    times : (T,) array
        Time points at which Brier is evaluated.
    censoring_survival : StepFunction, optional
        Pre-computed $\hat G$. If ``None``, computed internally.

    Returns
    -------
    (T,) array
        Brier score at each time. Pointwise; integrate via :func:`integrated`.

    References
    ----------
    Graf, E., Schmoor, C., Sauerbrei, W., Schumacher, M. (1999). Assessment
    and comparison of prognostic classification schemes for survival data.
    Statistics in Medicine, 18(17-18).
    """
    Y = np.asarray(event_time, dtype=np.float64)
    delta = np.asarray(event_indicator, dtype=np.int8)
    S = np.asarray(survival, dtype=np.float64)
    t_grid = np.asarray(times, dtype=np.float64)

    G = (
        censoring_survival
        if censoring_survival is not None
        else censoring_distribution(Y, delta)
    )
    G_at_Y = G(Y, side="left")  # (n,)   — case weight denominator
    G_at_t = G(t_grid, side="right")  # (T,)   — control weight denominator

    case_mask = (Y[:, None] <= t_grid[None, :]) & (delta[:, None] == 1)  # (n, T)
    control_mask = Y[:, None] > t_grid[None, :]  # (n, T)

    case_err = S**2  # (0 - S)^2
    control_err = (1.0 - S) ** 2  # (1 - S)^2

    # 1 / G, zero where G has hit zero: past the censoring distribution's
    # support the status cannot be reweighted, so the contribution is
    # dropped. ``np.divide`` with ``where`` never evaluates the masked-out
    # entries (``np.where`` would, warning on the discarded branch).
    inv_G_at_Y = np.divide(1.0, G_at_Y, out=np.zeros_like(G_at_Y), where=G_at_Y > 0)
    inv_G_at_t = np.divide(1.0, G_at_t, out=np.zeros_like(G_at_t), where=G_at_t > 0)

    contributions = (
        case_mask * inv_G_at_Y[:, None] * case_err
        + control_mask * inv_G_at_t[None, :] * control_err
    )
    return contributions.sum(axis=0) / len(Y)


def score_cause_specific(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    cif: ArrayLike,
    times: ArrayLike,
    *,
    cause: int = 1,
    censoring_survival: StepFunction | None = None,
) -> NDArray[np.float64]:
    r"""IPCW cause-specific Brier score at each point of ``times``.

    For cause $k$ the target is the cause-$k$ indicator
    $N_k(t) = \mathbb{1}[T \le t,\,\delta = k]$ and the prediction is
    $\hat F_k(t \mid x)$. Three populations contribute under right-censoring:

    - Cause-$k$ event by $t$ ($Y_i \le t$, $\delta_i = k$): error
      $(1 - \hat F_k(t \mid x_i))^2$ with weight $1 / \hat G(Y_i^-)$.
    - Competing event by $t$ ($Y_i \le t$, $\delta_i \notin \{0, k\}$):
      true indicator stays at $0$ forever after the competing event, so
      error $\hat F_k(t \mid x_i)^2$ with weight $1 / \hat G(Y_i^-)$.
    - Still at risk past $t$ ($Y_i > t$): error
      $\hat F_k(t \mid x_i)^2$ with weight $1 / \hat G(t)$.

    Subjects censored before $t$ (uninformative for cause-$k$ status)
    contribute zero — the IPCW weights inflate the other contributions to
    compensate. Reduces to :func:`score` (single-event Brier on $1 - S$)
    when only cause $k$ is present.

    Parameters
    ----------
    event_time : (n,) array
        Observed time $Y = \min(T, C)$.
    event_indicator : (n,) array
        $\delta = 0$ censored, $\delta = k \ge 1$ event of cause $k$.
    cif : (n, T) array
        Predicted $\hat F_k(t \mid x_i)$ for the chosen ``cause``, at each
        ``times`` point.
    times : (T,) array
    cause : int, default 1
    censoring_survival : StepFunction, optional
        Pre-computed $\hat G$. If ``None``, computed internally.

    Returns
    -------
    (T,) array

    References
    ----------
    Schoop, R. et al. (2011). Quantifying the predictive accuracy of
    time-to-event models in the presence of competing risks. Biometrics,
    67(2).
    """
    if cause < 1:
        raise ValueError(f"cause must be >= 1, got {cause}")

    Y = np.asarray(event_time, dtype=np.float64)
    delta = np.asarray(event_indicator, dtype=np.int8)
    F = np.asarray(cif, dtype=np.float64)
    t_grid = np.asarray(times, dtype=np.float64)

    G = (
        censoring_survival
        if censoring_survival is not None
        else censoring_distribution(Y, delta)
    )
    G_at_Y = G(Y, side="left")  # (n,)
    G_at_t = G(t_grid, side="right")  # (T,)

    is_cause_k = delta == cause
    is_competing = (delta > 0) & ~is_cause_k

    before_or_at_t = Y[:, None] <= t_grid[None, :]  # (n, T)
    after_t = Y[:, None] > t_grid[None, :]  # (n, T)

    case_mask = before_or_at_t & is_cause_k[:, None]
    competing_mask = before_or_at_t & is_competing[:, None]
    risk_mask = after_t

    err_case = (1.0 - F) ** 2
    err_competing = F**2
    err_risk = F**2

    # Per-subject case-time weights = 1 / G(Y_i^-); per-time control weights
    # = 1 / G(t). Zero where G has hit zero (see :func:`score`).
    w_case_time = np.divide(1.0, G_at_Y, out=np.zeros_like(G_at_Y), where=G_at_Y > 0)[
        :, None
    ]
    w_at_t = np.divide(1.0, G_at_t, out=np.zeros_like(G_at_t), where=G_at_t > 0)[
        None, :
    ]

    contributions = (
        case_mask * w_case_time * err_case
        + competing_mask * w_case_time * err_competing
        + risk_mask * w_at_t * err_risk
    )
    return contributions.sum(axis=0) / len(Y)


def integrated_cause_specific(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    cif: ArrayLike,
    times: ArrayLike,
    *,
    cause: int = 1,
    censoring_survival: StepFunction | None = None,
) -> float:
    r"""Integrated cause-specific Brier score over ``times``.

    Trapezoidal integration of :func:`score_cause_specific` over the grid,
    normalized by the grid's span.

    Parameters
    ----------
    See :func:`score_cause_specific`.

    Returns
    -------
    float
    """
    t_grid = np.asarray(times, dtype=np.float64)
    duration = float(t_grid[-1] - t_grid[0])
    if duration <= 0:
        raise ValueError(
            f"times must span a positive interval; got [{t_grid[0]}, {t_grid[-1]}]"
        )
    bs = score_cause_specific(
        event_time,
        event_indicator,
        cif,
        times,
        cause=cause,
        censoring_survival=censoring_survival,
    )
    return float(np.trapezoid(bs, t_grid) / duration)


def integrated(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    survival: ArrayLike,
    times: ArrayLike,
    *,
    censoring_survival: StepFunction | None = None,
) -> float:
    r"""Integrated Brier score over ``times``.

    Trapezoidal integration of :func:`score` over the grid, normalized by
    the grid's span:

    $$
    \widehat{IBS} = \frac{1}{t_{\max} - t_{\min}}
                    \int_{t_{\min}}^{t_{\max}} \widehat{BS}(t)\,dt
    $$

    The user chooses the integration range by choosing ``times``.

    Parameters
    ----------
    See :func:`score`.

    Returns
    -------
    float
        Time-averaged Brier score.
    """
    t_grid = np.asarray(times, dtype=np.float64)
    duration = float(t_grid[-1] - t_grid[0])
    if duration <= 0:
        raise ValueError(
            f"times must span a positive interval; got [{t_grid[0]}, {t_grid[-1]}]"
        )

    bs = score(
        event_time,
        event_indicator,
        survival,
        times,
        censoring_survival=censoring_survival,
    )
    return float(np.trapezoid(bs, t_grid) / duration)
