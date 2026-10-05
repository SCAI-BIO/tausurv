from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2

from tausurv.nonparametric import aalen_johansen, kaplan_meier


def distributional(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    survival: ArrayLike,
    times: ArrayLike,
    *,
    n_bins: int = 10,
) -> float:
    r"""D-calibration test (Haider et al., 2020).

    Under a well-calibrated model, the values $\hat S(Y_i \mid x_i)$ — the
    predicted survival probability at each subject's observed time — are
    uniformly distributed on $[0, 1]$. The test partitions $[0, 1]$ into
    ``n_bins`` equal-width bins and applies a chi-square goodness-of-fit
    test to the bin counts.

    Event subjects contribute 1 to the bin containing $\hat S(Y_i \mid x_i)$.
    Censored subjects spread their contribution uniformly across bins below
    $\hat S(Y_i \mid x_i)$: the true event time is in $(Y_i, \infty)$, so its
    probability lies in $[0, \hat S(Y_i))$.

    Parameters
    ----------
    event_time : (n,) array
    event_indicator : (n,) array
    survival : (n, T) array
        $\hat S(t \mid x_i)$ at each ``times`` point.
    times : (T,) array
    n_bins : int, default 10

    Returns
    -------
    float
        p-value of the chi-square test. High (e.g. > 0.05) = no evidence of
        miscalibration; low (e.g. < 0.05) = miscalibrated.

    References
    ----------
    Haider, H., Hoehn, B., Davis, S., Greiner, R. (2020). Effective ways to
    build and evaluate individual survival distributions. JMLR 21(85).
    """
    Y = np.asarray(event_time, dtype=np.float64)
    delta = np.asarray(event_indicator, dtype=np.int8)
    S = np.asarray(survival, dtype=np.float64)
    t_grid = np.asarray(times, dtype=np.float64)

    # S(Y_i | x_i) for each subject, right-continuous step from times to Y_i.
    t_idx = np.maximum(np.searchsorted(t_grid, Y, side="right") - 1, 0)
    p = S[np.arange(len(Y)), t_idx]

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    a_edges, b_edges = bin_edges[:-1], bin_edges[1:]

    observed = np.zeros(n_bins)

    event_mask = delta == 1
    if event_mask.any():
        idx = np.clip(
            np.searchsorted(bin_edges, p[event_mask], side="right") - 1,
            0,
            n_bins - 1,
        )
        np.add.at(observed, idx, 1.0)

    cens_mask = ~event_mask
    if cens_mask.any():
        # Spread mass uniformly over [0, p_i] for each censored subject.
        p_cens = np.maximum(p[cens_mask], 1e-12)
        overlap = np.maximum(
            0.0, np.minimum(b_edges[None, :], p_cens[:, None]) - a_edges[None, :]
        )
        observed += (overlap / p_cens[:, None]).sum(axis=0)

    expected = len(Y) / n_bins
    d_squared = float(np.sum((observed - expected) ** 2 / expected))
    return float(chi2.sf(d_squared, df=n_bins - 1))


def curve(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    survival: ArrayLike,
    times: ArrayLike,
    t: float,
    *,
    n_bins: int = 10,
    min_bin_size: int = 5,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    r"""Survival calibration curve at horizon $t$.

    Subjects are split into ``n_bins`` quantile-based bins of predicted
    survival $\hat S(t \mid x_i)$. Within each bin, the observed survival at
    $t$ is estimated by Kaplan-Meier on the bin's subjects — so censoring is
    handled nonparametrically. A well-calibrated model has bin-mean
    predictions close to KM estimates.

    Parameters
    ----------
    event_time : (n,) array
    event_indicator : (n,) array
    survival : (n, T) array
    times : (T,) array
    t : float
        Horizon at which calibration is assessed.
    n_bins : int, default 10
    min_bin_size : int, default 5
        Drop bins with fewer subjects than this.

    Returns
    -------
    predicted : (K,) array — per-bin mean predicted survival.
    observed  : (K,) array — per-bin KM survival at $t$.
    bin_sizes : (K,) int array — per-bin subject count.
    """
    Y = np.asarray(event_time, dtype=np.float64)
    delta = np.asarray(event_indicator, dtype=np.int8)
    S = np.asarray(survival, dtype=np.float64)
    pred = S[:, _horizon_index(times, t)]

    def observed(mask: NDArray[np.bool_]) -> float:
        return float(kaplan_meier(Y[mask], delta[mask])(t))

    return _binned(pred, observed, n_bins, min_bin_size)


def curve_cause_specific(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    cif: ArrayLike,
    times: ArrayLike,
    t: float,
    *,
    cause: int = 1,
    n_bins: int = 10,
    min_bin_size: int = 5,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    r"""Cumulative-incidence calibration curve for one cause at horizon $t$.

    Subjects are split into ``n_bins`` quantile-based bins of predicted
    incidence $\hat F_k(t \mid x_i)$. Within each bin the observed incidence
    at $t$ is the Aalen-Johansen estimate on the bin's subjects, so competing
    events count as competing events rather than as censoring.

    Parameters
    ----------
    event_time : (n,) array
    event_indicator : (n,) array
        Integer-valued: $0$ for censored, $k \ge 1$ for an event of cause $k$.
    cif : (n, T) array
        Predicted $\hat F_k(t \mid x_i)$ for ``cause`` on ``times``.
    times : (T,) array
    t : float
        Horizon at which calibration is assessed.
    cause : int, default 1
    n_bins : int, default 10
    min_bin_size : int, default 5
        Drop bins with fewer subjects than this.

    Returns
    -------
    predicted : (K,) array — per-bin mean predicted incidence.
    observed  : (K,) array — per-bin Aalen-Johansen incidence at $t$.
    bin_sizes : (K,) int array — per-bin subject count.
    """
    Y = np.asarray(event_time, dtype=np.float64)
    E = np.asarray(event_indicator, dtype=np.int64)
    F = np.asarray(cif, dtype=np.float64)
    pred = F[:, _horizon_index(times, t)]

    def observed(mask: NDArray[np.bool_]) -> float:
        return float(aalen_johansen(Y[mask], E[mask], cause)(t))

    return _binned(pred, observed, n_bins, min_bin_size)


def _horizon_index(times: ArrayLike, t: float) -> int:
    t_grid = np.asarray(times, dtype=np.float64)
    return int(np.maximum(np.searchsorted(t_grid, t, side="right") - 1, 0))


def _binned(
    pred: NDArray[np.float64],
    observed: Callable[[NDArray[np.bool_]], float],
    n_bins: int,
    min_bin_size: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    bin_edges = np.unique(np.quantile(pred, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(bin_edges) < 2:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
        )

    bin_idx = np.digitize(pred, bin_edges[1:-1])

    pred_means: list[float] = []
    obs_means: list[float] = []
    sizes: list[int] = []
    for k in range(len(bin_edges) - 1):
        mask = bin_idx == k
        if int(mask.sum()) < min_bin_size:
            continue
        pred_means.append(float(pred[mask].mean()))
        obs_means.append(observed(mask))
        sizes.append(int(mask.sum()))

    return (
        np.asarray(pred_means, dtype=np.float64),
        np.asarray(obs_means, dtype=np.float64),
        np.asarray(sizes, dtype=np.int64),
    )
