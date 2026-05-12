from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tausurv.nonparametric import censoring_distribution, kaplan_meier
from tausurv.step import StepFunction


def uno(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    risk_score: ArrayLike,
    time_grid: ArrayLike,
    *,
    censoring_survival: StepFunction | None = None,
) -> NDArray[np.float64]:
    r"""Uno's IPCW cumulative/dynamic AUC at each point of ``time_grid``.

    At each time $t$, cases are subjects with $Y_i \le t$ and $\delta_i = 1$;
    controls are subjects with $Y_j > t$. The AUC at $t$ is the IPCW-weighted
    probability that a case has higher predicted risk than a control:

    $$
    \widehat{AUC}(t) = \frac{\sum_{i,j} \omega_i(t)\,\mathbb{1}[Y_j > t]\,
                            \big(\mathbb{1}[\hat r_i > \hat r_j] + \tfrac{1}{2}\mathbb{1}[\hat r_i = \hat r_j]\big)}
                           {\big(\sum_i \omega_i(t)\big) \cdot \#\{j : Y_j > t\}}
    $$

    with $\omega_i(t) = \delta_i \mathbb{1}[Y_i \le t] / \hat G(Y_i^-)$. Returns
    NaN at time points with no valid cases or no controls.

    Parameters
    ----------
    event_time : (n,) array
        Observed time $Y = \min(T, C)$.
    event_indicator : (n,) array
        $\delta = 1$ if event observed, $0$ if censored.
    risk_score : (n,) array
        Time-invariant monotone-in-risk scalar; higher = shorter time-to-event.
    time_grid : (T,) array
        Times at which to evaluate AUC.
    censoring_survival : StepFunction, optional
        Pre-computed $\hat G$. If ``None``, computed internally.

    Returns
    -------
    (T,) array

    References
    ----------
    Uno, H. et al. (2007). Evaluating prediction rules for t-year survivors
    with censored regression models. Journal of the American Statistical
    Association, 102(478).
    """
    Y = np.asarray(event_time, dtype=np.float64)
    delta = np.asarray(event_indicator, dtype=np.int8)
    r = np.asarray(risk_score, dtype=np.float64)
    t_grid = np.asarray(time_grid, dtype=np.float64)

    G = (
        censoring_survival
        if censoring_survival is not None
        else censoring_distribution(Y, delta)
    )
    G_at_Y = G(Y, side="left")
    valid_G = G_at_Y > 0

    # Marker is time-invariant: pre-sort by r once. Then "sorted controls at t"
    # is the (already-sorted) subset of subjects with Y > t.
    order_r = np.argsort(r, kind="stable")
    r_in_r_order = r[order_r]
    Y_in_r_order = Y[order_r]

    out = np.full(len(t_grid), np.nan)
    for k in range(len(t_grid)):
        t = t_grid[k]
        case_mask = (Y <= t) & (delta == 1) & valid_G
        if not case_mask.any():
            continue
        control_in_r_order = Y_in_r_order > t
        if not control_in_r_order.any():
            continue

        r_control_sorted = r_in_r_order[control_in_r_order]
        r_case = r[case_mask]
        lt = np.searchsorted(r_control_sorted, r_case, side="left")
        le = np.searchsorted(r_control_sorted, r_case, side="right")
        per_case_score = lt.astype(np.float64) + 0.5 * (le - lt).astype(np.float64)

        omega_case = 1.0 / G_at_Y[case_mask]
        numerator = float((omega_case * per_case_score).sum())
        denominator = float(omega_case.sum() * r_control_sorted.size)

        if denominator > 0:
            out[k] = numerator / denominator

    return out


def blanche(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    marker: ArrayLike,
    time_grid: ArrayLike,
    *,
    censoring_survival: StepFunction | None = None,
) -> NDArray[np.float64]:
    r"""Blanche's IPCW cumulative/dynamic AUC with a time-varying marker.

    Generalizes :func:`uno` to markers that vary with $t$ — e.g., the
    natural choice for survival models is $M_i(t) = 1 - \hat S(t \mid x_i)$.
    For a constant-in-$t$ marker, reduces to :func:`uno` exactly.

    Parameters
    ----------
    event_time : (n,) array
        Observed time $Y = \min(T, C)$.
    event_indicator : (n,) array
        $\delta = 1$ if event observed, $0$ if censored.
    marker : (n, T) array
        $M_i(t)$ — predicted risk at each ``time_grid`` point. Higher value
        means higher predicted risk of event by that time. For survival
        predictions, pass ``1 - survival``.
    time_grid : (T,) array
        Times at which to evaluate AUC.
    censoring_survival : StepFunction, optional
        Pre-computed $\hat G$. If ``None``, computed internally.

    Returns
    -------
    (T,) array

    References
    ----------
    Blanche, P., Dartigues, J.-F., Jacqmin-Gadda, H. (2013). Statistics in
    Medicine, 32(30).
    """
    Y = np.asarray(event_time, dtype=np.float64)
    delta = np.asarray(event_indicator, dtype=np.int8)
    M = np.asarray(marker, dtype=np.float64)
    t_grid = np.asarray(time_grid, dtype=np.float64)

    G = (
        censoring_survival
        if censoring_survival is not None
        else censoring_distribution(Y, delta)
    )
    G_at_Y = G(Y, side="left")

    out = np.full(len(t_grid), np.nan)
    for k in range(len(t_grid)):
        t = t_grid[k]
        case_mask = (Y <= t) & (delta == 1) & (G_at_Y > 0)
        control_mask = Y > t

        if not case_mask.any() or not control_mask.any():
            continue

        m_case = M[case_mask, k]
        m_control = M[control_mask, k]

        # Sort controls; for each case, count controls with marker strictly
        # less than its own (concordant) and tied (half-credit) via two
        # searchsorted lookups.
        m_control_sorted = np.sort(m_control)
        lt = np.searchsorted(m_control_sorted, m_case, side="left")
        le = np.searchsorted(m_control_sorted, m_case, side="right")
        per_case_score = lt.astype(np.float64) + 0.5 * (le - lt).astype(np.float64)

        omega_case = 1.0 / G_at_Y[case_mask]
        numerator = float((omega_case * per_case_score).sum())
        denominator = float(omega_case.sum() * len(m_control))

        if denominator > 0:
            out[k] = numerator / denominator

    return out


def cause_specific(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    cif: ArrayLike,
    time_grid: ArrayLike,
    *,
    cause: int = 1,
    censoring_survival: StepFunction | None = None,
) -> NDArray[np.float64]:
    r"""IPCW cause-specific cumulative/dynamic AUC at each point of ``time_grid``.

    At each time $t$, cases are subjects with $Y_i \le t$ and
    $\delta_i = k$; controls are subjects with $Y_j > t$ (still at risk for
    any cause). The marker is the cause-$k$ predicted CIF
    $\hat F_k(t \mid x_i)$ (time-varying — caller passes the
    ``(n, len(time_grid))`` array). Case weights are IPCW
    $\omega_i = 1 / \hat G(Y_i^-)$. The estimator is

    $$
    \widehat{AUC}_k(t) = \frac{\sum_{i \in \text{cases}} \omega_i
        \big(\mathbb{1}[\hat F_k(t \mid x_i) > \hat F_k(t \mid x_j)] + \tfrac{1}{2}\,\text{ties}\big)}
        {(\sum_i \omega_i) \cdot \#\{j : Y_j > t\}}
    $$

    summed over controls $j$. Reduces to Blanche's single-event AUC when
    only cause $k$ is present. Returns NaN at time points with no eligible
    cases or no controls.

    Parameters
    ----------
    event_time : (n,) array
        Observed time $Y = \min(T, C)$.
    event_indicator : (n,) array
        $\delta = 0$ censored, $\delta = k \ge 1$ event of cause $k$.
    cif : (n, T) array
        Predicted $\hat F_k(t \mid x_i)$ for the chosen ``cause``, at each
        ``time_grid`` point.
    time_grid : (T,) array
    cause : int, default 1
    censoring_survival : StepFunction, optional
        Pre-computed $\hat G$. If ``None``, computed internally.

    Returns
    -------
    (T,) array

    References
    ----------
    Saha, P., Heagerty, P. J. (2010). Time-dependent predictive accuracy in
    the presence of competing risks. Biometrics, 66(4).
    Blanche, P. et al. (2013). Statistics in Medicine, 32(30).
    """
    if cause < 1:
        raise ValueError(f"cause must be >= 1, got {cause}")

    Y = np.asarray(event_time, dtype=np.float64)
    delta = np.asarray(event_indicator, dtype=np.int8)
    F = np.asarray(cif, dtype=np.float64)
    t_grid = np.asarray(time_grid, dtype=np.float64)

    G = (
        censoring_survival
        if censoring_survival is not None
        else censoring_distribution(Y, delta)
    )
    G_at_Y = G(Y, side="left")
    valid_G = G_at_Y > 0

    out = np.full(len(t_grid), np.nan)
    for k_idx in range(len(t_grid)):
        t = t_grid[k_idx]
        case_mask = (Y <= t) & (delta == cause) & valid_G
        control_mask = Y > t
        if not case_mask.any() or not control_mask.any():
            continue

        m_case = F[case_mask, k_idx]
        m_control = F[control_mask, k_idx]

        m_control_sorted = np.sort(m_control)
        lt = np.searchsorted(m_control_sorted, m_case, side="left")
        le = np.searchsorted(m_control_sorted, m_case, side="right")
        per_case = lt.astype(np.float64) + 0.5 * (le - lt).astype(np.float64)

        omega_case = 1.0 / G_at_Y[case_mask]
        numerator = float((omega_case * per_case).sum())
        denominator = float(omega_case.sum() * m_control_sorted.size)

        if denominator > 0:
            out[k_idx] = numerator / denominator

    return out


def integrated(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    auc_per_time: ArrayLike,
    time_grid: ArrayLike,
) -> float:
    r"""Survival-weighted integrated AUC (Heagerty-Zheng).

    Weights per-time AUC by the marginal event density estimated by KM, so
    times with more events count more:

    $$
    \overline{AUC} =
      \frac{\int_{t_{\min}}^{t_{\max}} \widehat{AUC}(t)\,d\hat F(t)}
           {\hat F(t_{\max}) - \hat F(t_{\min})}
    $$

    with $\hat F(t) = 1 - \hat S(t)$ from the Kaplan-Meier estimate of the
    marginal event distribution. Discretized as a trapezoidal-style sum
    against the KM drops between consecutive ``time_grid`` points.

    Generalizes Harrell's C over a time range — the simple time-average
    $\int \widehat{AUC}(t)\,dt / (t_{\max} - t_{\min})$ is *not* used because
    it gives equal weight to times with few events, which is rarely what
    survival analysis cares about.

    Parameters
    ----------
    event_time : (n,) array
        Observed time $Y = \min(T, C)$. Used to estimate $\hat S$ for the weighting.
    event_indicator : (n,) array
        $\delta = 1$ if event observed, $0$ if censored.
    auc_per_time : (T,) array
        Per-time AUC values, as returned by :func:`uno` or :func:`blanche`.
    time_grid : (T,) array
        Time points corresponding to ``auc_per_time``.

    Returns
    -------
    float

    References
    ----------
    Heagerty, P. J., Zheng, Y. (2005). Survival model predictive accuracy and
    ROC curves. Biometrics, 61(1).
    """
    Y = np.asarray(event_time, dtype=np.float64)
    delta = np.asarray(event_indicator, dtype=np.int8)
    a = np.asarray(auc_per_time, dtype=np.float64)
    t_grid = np.asarray(time_grid, dtype=np.float64)

    if len(t_grid) < 2:
        raise ValueError("integrated AUC requires at least 2 time_grid points")

    S = kaplan_meier(Y, delta)
    S_at_t = S(t_grid, side="right")

    total_drop = float(S_at_t[0] - S_at_t[-1])
    if total_drop <= 0:
        raise ValueError(
            "no marginal events within time_grid range; integrated AUC is undefined"
        )

    drops = -np.diff(S_at_t)
    auc_midpoints = 0.5 * (a[:-1] + a[1:])
    return float(np.sum(drops * auc_midpoints) / total_drop)
