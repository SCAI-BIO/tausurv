from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tausurv.nonparametric import censoring_distribution
from tausurv.step import StepFunction


def harrell(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    risk_score: ArrayLike,
) -> float:
    r"""Harrell's C-index.

    The fraction of comparable pairs $(i, j)$ for which the predicted risk
    ordering agrees with the observed event ordering:

    $$
    C = \frac{\sum_{i,j} \mathbb{1}[Y_i < Y_j,\ \delta_i = 1]\,
              w_{ij}}
             {\sum_{i,j} \mathbb{1}[Y_i < Y_j,\ \delta_i = 1]}
    $$

    where $w_{ij} = 1$ if $\hat r_i > \hat r_j$, $\tfrac{1}{2}$ if
    $\hat r_i = \hat r_j$, $0$ otherwise. A pair is comparable when the
    event order is known: $i$ has an event and $j$ has a later time, or the
    same time and is censored (still event-free when $i$ failed). Two events
    at the same time are not comparable. These are Harrell's rules, and the
    ones scikit-survival, lifelines and R's survival package use; pycox's
    default additionally scores tied-event pairs. Biased toward $0.5$ under
    heavy censoring; use :func:`uno` for an IPCW-corrected estimate.

    Parameters
    ----------
    event_time : (n,) array
        Observed time $Y = \min(T, C)$.
    event_indicator : (n,) array
        $\delta = 1$ if event observed, $0$ if censored.
    risk_score : (n,) array
        Monotone-in-risk scalar; higher values predict shorter time-to-event.

    Returns
    -------
    float
        Concordance in $[0, 1]$.

    References
    ----------
    Harrell, F. E. et al. (1996). Multivariable prognostic models. Statistics
    in Medicine, 15(4).
    """
    Y = np.asarray(event_time, dtype=np.float64)
    delta = np.asarray(event_indicator, dtype=np.int8)
    r = np.asarray(risk_score, dtype=np.float64)

    # Sort by Y ascending (stable); for each case, partners are the suffix
    # of r past the case's Y-tie block.
    order = np.argsort(Y, kind="stable")
    Y_s = Y[order]
    delta_s = delta[order]
    r_s = r[order]
    comparable = _comparable_partners(Y_s, delta_s)

    n_concordant = 0
    n_tied = 0
    n_comparable = 0
    for p in np.flatnonzero(delta_s == 1):
        partners = r_s[comparable(int(p))]
        r_p = float(r_s[p])
        n_comparable += partners.size
        n_concordant += int((partners < r_p).sum())
        n_tied += int((partners == r_p).sum())

    if n_comparable == 0:
        raise ValueError(
            "no comparable pairs: need at least one event followed by a "
            "later observation"
        )

    return (n_concordant + 0.5 * n_tied) / n_comparable


def uno(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    risk_score: ArrayLike,
    *,
    horizon: float,
    censoring_survival: StepFunction | None = None,
) -> float:
    r"""Uno's IPCW concordance index.

    Each comparable pair $(i, j)$ — with $Y_i < Y_j$, $Y_i < \tau$, and
    $\delta_i = 1$, ties in time as in :func:`harrell` — is weighted by
    $1 / \hat G(Y_i^-)^2$, where $\hat G$ is the Kaplan-Meier estimator of
    the censoring distribution $G(t) = P(C > t)$:

    $$
    \hat C_\tau = \frac{\sum_{i,j} w_{ij}\,\mathbb{1}[Y_i < Y_j,\ Y_i < \tau,\ \delta_i = 1]\,
                       \big(\mathbb{1}[\hat r_i > \hat r_j] + \tfrac{1}{2}\mathbb{1}[\hat r_i = \hat r_j]\big)}
                      {\sum_{i,j} w_{ij}\,\mathbb{1}[Y_i < Y_j,\ Y_i < \tau,\ \delta_i = 1]}
    $$

    with $w_{ij} = 1 / \hat G(Y_i^-)^2$. Asymptotically unbiased under random
    censoring; correct choice when censoring is non-trivial.

    Parameters
    ----------
    event_time : (n,) array
        Observed time $Y = \min(T, C)$.
    event_indicator : (n,) array
        $\delta = 1$ if event observed, $0$ if censored.
    risk_score : (n,) array
        Monotone-in-risk scalar; higher values predict shorter time-to-event.
    horizon : float
        Truncation time $\tau$. Pairs with $Y_i \ge \tau$ are dropped. Choose
        within the censoring distribution's support; weights blow up where
        $\hat G(t) \to 0$.
    censoring_survival : StepFunction, optional
        Pre-computed $\hat G$ as returned by
        :func:`tausurv.nonparametric.kaplan_meier` on the censoring indicator.
        If ``None``, computed internally.

    Returns
    -------
    float
        Concordance in $[0, 1]$.

    References
    ----------
    Uno, H. et al. (2011). On the C-statistics for evaluating overall adequacy
    of risk prediction procedures with censored survival data. Statistics in
    Medicine, 30(10).
    """
    Y = np.asarray(event_time, dtype=np.float64)
    delta = np.asarray(event_indicator, dtype=np.int8)
    r = np.asarray(risk_score, dtype=np.float64)

    G = (
        censoring_survival
        if censoring_survival is not None
        else censoring_distribution(Y, delta)
    )
    G_at_Y = G(Y, side="left")

    order = np.argsort(Y, kind="stable")
    Y_s = Y[order]
    delta_s = delta[order]
    r_s = r[order]
    G_s = G_at_Y[order]
    comparable = _comparable_partners(Y_s, delta_s)

    numerator = 0.0
    denominator = 0.0
    eligible = (delta_s == 1) & (Y_s < horizon) & (G_s > 0)
    for p in np.flatnonzero(eligible):
        partners = r_s[comparable(int(p))]
        r_p = float(r_s[p])
        w = 1.0 / float(G_s[p]) ** 2
        n_concordant = int((partners < r_p).sum())
        n_tied = int((partners == r_p).sum())
        numerator += w * (n_concordant + 0.5 * n_tied)
        denominator += w * partners.size

    if denominator == 0:
        raise ValueError(
            "no comparable pairs before the horizon with positive censoring "
            "survival; choose a smaller horizon or check for excessive censoring"
        )

    return numerator / denominator


def antolini(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    survival: ArrayLike,
    times: ArrayLike,
) -> float:
    r"""Antolini's time-dependent C-index.

    A pair $(i, j)$ with $Y_i < Y_j$ and $\delta_i = 1$, ties in time as in
    :func:`harrell`, is concordant if
    $\hat S(Y_i \mid x_i) < \hat S(Y_i \mid x_j)$ — the model predicts lower
    survival probability for the case at its own event time than for the
    later partner at that same time. Ties in $\hat S$ contribute
    $\tfrac{1}{2}$.

    Differs from Harrell's C by evaluating the full survival curve at each
    event time rather than collapsing predictions to a scalar risk score. The
    natural choice for models with time-varying predictions (DeepHit,
    DeepSurv-PMF, parametric AFT).

    Parameters
    ----------
    event_time : (n,) array
        Observed time $Y = \min(T, C)$.
    event_indicator : (n,) array
        $\delta = 1$ if event observed, $0$ if censored.
    survival : (n, T) array
        $\hat S(t \mid x_i)$ for each sample at each ``times`` point.
    times : (T,) array
        Time points at which ``survival`` is evaluated. Right-continuous
        step is used when stepping from grid to event times.

    Returns
    -------
    float
        Concordance in $[0, 1]$.

    References
    ----------
    Antolini, L., Boracchi, P., Biganzoli, E. (2005). A time-dependent
    discrimination index for survival data. Statistics in Medicine, 24(24).
    """
    Y = np.asarray(event_time, dtype=np.float64)
    delta = np.asarray(event_indicator, dtype=np.int8)
    S = np.asarray(survival, dtype=np.float64)
    t_grid = np.asarray(times, dtype=np.float64)

    order = np.argsort(Y, kind="stable")
    Y_s = Y[order]
    delta_s = delta[order]
    S_s = S[order]
    comparable = _comparable_partners(Y_s, delta_s)

    n_concordant = 0
    n_tied = 0
    n_comparable = 0
    for p in np.flatnonzero(delta_s == 1):
        t_idx = int(max(np.searchsorted(t_grid, Y_s[p], side="right") - 1, 0))
        S_case = float(S_s[p, t_idx])
        partner_S = S_s[:, t_idx][comparable(int(p))]
        n_comparable += partner_S.size
        n_concordant += int((partner_S > S_case).sum())
        n_tied += int((partner_S == S_case).sum())

    if n_comparable == 0:
        raise ValueError(
            "no comparable pairs: need at least one event followed by a "
            "later observation"
        )

    return (n_concordant + 0.5 * n_tied) / n_comparable


def _comparable_partners(
    Y_s: NDArray[np.float64], delta_s: NDArray[np.int8]
) -> Callable[[int], NDArray[np.intp]]:
    """Partner indices of an event at sorted position ``p``, per Harrell.

    Subjects with a later time, plus those censored at the same time: the
    event order is known for both. Events tied in time are left out.
    """
    block_start = np.searchsorted(Y_s, Y_s, side="left")
    block_end = np.searchsorted(Y_s, Y_s, side="right")
    n = len(Y_s)

    def partners(p: int) -> NDArray[np.intp]:
        start, end = int(block_start[p]), int(block_end[p])
        tied_censored = start + np.flatnonzero(delta_s[start:end] == 0)
        return np.concatenate([tied_censored, np.arange(end, n)])

    return partners


def harrell_cause_specific(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    risk_score: ArrayLike,
    *,
    cause: int = 1,
) -> float:
    r"""Cause-specific Harrell C-index for competing risks (Wolbers et al., 2009).

    For a fixed cause $k$, a pair $(i, j)$ is comparable iff $\delta_i = k$
    and one of:

    - $Y_j > Y_i$ (subject $j$ is still at risk for cause $k$ past $Y_i$), or
    - $Y_j < Y_i$ and $\delta_j \notin \{0, k\}$ (subject $j$ already had a
      competing event — known not to have cause $k$).

    Censored partners with $Y_j \le Y_i$ are excluded — their cause-$k$
    status is unknown. Concordant pairs have $\hat r_i > \hat r_j$; ties
    contribute $\tfrac{1}{2}$. Reduces to :func:`harrell` when the data are
    single-event (the competing-event partner set is empty).

    Parameters
    ----------
    event_time : (n,) array
        Observed time $Y = \min(T, C)$.
    event_indicator : (n,) array
        $\delta = 0$ censored, $\delta = k \ge 1$ event of cause $k$.
    risk_score : (n,) array
        Monotone-in-cause-$k$-risk scalar — typically
        $\hat F_k(t^* \mid x)$ at a fixed horizon $t^*$.
    cause : int, default 1
        Which cause to evaluate.

    Returns
    -------
    float
        Concordance in $[0, 1]$.

    References
    ----------
    Wolbers, M., Koller, M. T., Witteman, J. C. M., Steyerberg, E. W.
    (2009). Prognostic models with competing risks: methods and application
    to coronary risk prediction. Epidemiology, 20(4), 555-561.
    Wolbers, M. et al. (2014). Concordance for prognostic models with
    competing risks. Biostatistics, 15(3), 526-539. Describes the
    IPCW-weighted version, which this function does not implement.
    """
    if cause < 1:
        raise ValueError(f"cause must be >= 1, got {cause}")

    Y = np.asarray(event_time, dtype=np.float64)
    delta = np.asarray(event_indicator, dtype=np.int8)
    r = np.asarray(risk_score, dtype=np.float64)

    order = np.argsort(Y, kind="stable")
    Y_s = Y[order]
    delta_s = delta[order]
    r_s = r[order]

    # Index slices of the Y-tied block around each subject.
    block_start = np.searchsorted(Y_s, Y_s, side="left")
    partner_start = np.searchsorted(Y_s, Y_s, side="right")

    is_competing = (delta_s > 0) & (delta_s != cause)

    n_concordant = 0
    n_tied = 0
    n_comparable = 0
    for p in np.flatnonzero(delta_s == cause):
        r_p = float(r_s[p])

        # Type-1: subjects with Y > Y_p (strictly later in sorted order).
        s = int(partner_start[p])
        type1 = r_s[s:]

        # Type-2: subjects with Y < Y_p and a competing event observed.
        prefix_end = int(block_start[p])
        prefix_mask = is_competing[:prefix_end]
        type2 = r_s[:prefix_end][prefix_mask]

        partners = np.concatenate([type1, type2]) if type2.size else type1
        n_comparable += partners.size
        n_concordant += int((partners < r_p).sum())
        n_tied += int((partners == r_p).sum())

    if n_comparable == 0:
        raise ValueError(
            f"no comparable pairs for cause={cause}: need at least one "
            f"cause-{cause} event with a valid partner"
        )

    return (n_concordant + 0.5 * n_tied) / n_comparable


def blanche(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    risk_score: ArrayLike,
    *,
    horizon: float,
    censoring_survival: StepFunction | None = None,
) -> float:
    r"""Blanche's cumulative-case / dynamic-control concordance at horizon $\tau$.

    Cases are subjects with an observed event by $\tau$ ($Y_i \le \tau$,
    $\delta_i = 1$); controls are subjects still at risk past $\tau$
    ($Y_j > \tau$). A case/control pair is concordant if $\hat r_i > \hat r_j$.
    Weights are IPCW: each case carries $1 / \hat G(Y_i^-)$ and each control
    carries $1 / \hat G(\tau^-)$, so

    $$
    \hat C_\tau =
      \frac{\sum_{i \in \text{cases}} \tfrac{1}{\hat G(Y_i^-)}
            \sum_{j \in \text{controls}}
            \big(\mathbb{1}[\hat r_i > \hat r_j] + \tfrac{1}{2}\mathbb{1}[\hat r_i = \hat r_j]\big)}
           {n_{\text{controls}} \cdot \sum_{i \in \text{cases}} \tfrac{1}{\hat G(Y_i^-)}}
    $$

    (the constant control weight cancels in num/den). Differs from Uno's C
    in pair structure — Blanche partitions on $\tau$ rather than on within-pair
    ordering, which makes it the natural target when interest is in
    discrimination at a specific horizon.

    Parameters
    ----------
    event_time : (n,) array
        Observed time $Y = \min(T, C)$.
    event_indicator : (n,) array
        $\delta = 1$ if event observed, $0$ if censored.
    risk_score : (n,) array
        Monotone-in-risk scalar; for predictions from a survival model,
        $1 - \hat S(\tau \mid x)$ is the natural choice.
    horizon : float
        The horizon $\tau$.
    censoring_survival : StepFunction, optional
        Pre-computed $\hat G$. If ``None``, computed internally.

    Returns
    -------
    float
        Concordance in $[0, 1]$.

    References
    ----------
    Blanche, P., Dartigues, J.-F., Jacqmin-Gadda, H. (2013). Estimating and
    comparing time-dependent areas under receiver operating characteristic
    curves for censored event times with competing risks. Statistics in
    Medicine, 32(30).
    """
    Y = np.asarray(event_time, dtype=np.float64)
    delta = np.asarray(event_indicator, dtype=np.int8)
    r = np.asarray(risk_score, dtype=np.float64)

    G = (
        censoring_survival
        if censoring_survival is not None
        else censoring_distribution(Y, delta)
    )

    case_mask = (delta == 1) & (Y <= horizon)
    control_mask = Y > horizon
    n_cases = int(case_mask.sum())
    n_controls = int(control_mask.sum())

    if n_cases == 0 or n_controls == 0:
        raise ValueError(
            "blanche needs at least one case (event by the horizon) and one "
            "control (survivor past the horizon)"
        )

    if float(G(horizon, side="left")) <= 0:
        raise ValueError(
            "censoring survival is zero at the horizon; choose a smaller horizon"
        )

    G_at_cases = G(Y[case_mask], side="left")
    if np.any(G_at_cases <= 0):
        raise ValueError(
            "censoring survival is zero at one or more case times; choose a smaller horizon"
        )

    w_case = 1.0 / G_at_cases
    r_case = r[case_mask]
    r_control = r[control_mask]

    r_control_sorted = np.sort(r_control)
    lt = np.searchsorted(r_control_sorted, r_case, side="left")
    le = np.searchsorted(r_control_sorted, r_case, side="right")
    per_case_score = lt.astype(np.float64) + 0.5 * (le - lt).astype(np.float64)

    numerator = float((w_case * per_case_score).sum())
    denominator = float(w_case.sum() * n_controls)

    return numerator / denominator
