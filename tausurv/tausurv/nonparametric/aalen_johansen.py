from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tausurv.step import StepFunction


def aalen_johansen(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    cause: int,
) -> StepFunction:
    r"""Aalen-Johansen estimate of the cumulative incidence for a given cause.

    Competing-risks generalization of Kaplan-Meier:

    $$
    \hat F_k(t) = \sum_{t_i \le t} \hat S(t_i^-)\,\frac{d_{ik}}{n_i}
    $$

    where $\hat S$ is the any-cause KM survival, $d_{ik}$ counts cause-$k$
    events at $t_i$, and $n_i$ is the number at risk just before $t_i$. For
    $K = 1$ this reduces to $1 - \hat S_{KM}$.

    Event coding: $\delta = 0$ is censored, $\delta = k \ge 1$ is an event of
    cause $k$.

    Parameters
    ----------
    event_time : (n,) array
    event_indicator : (n,) array
        Integer-valued: $0$ for censored, $k \ge 1$ for an event of cause $k$.
    cause : int
        Positive integer denoting the cause of interest.

    Returns
    -------
    StepFunction
        $\hat F_k$ with ``baseline=0.0``; right-continuous.

    References
    ----------
    Aalen, O., Johansen, S. (1978). Scandinavian Journal of Statistics, 5(3).
    """
    time, _, _, _, _, cif = _tabulate(event_time, event_indicator, cause)
    return StepFunction(time=time, value=cif, side="right", baseline=0.0)


def aalen_johansen_variance(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    cause: int,
) -> StepFunction:
    r"""Pointwise variance of the Aalen-Johansen estimate.

    Greenwood-type estimator of Marubini and Valsecchi (1995), the form SAS
    ``PROC LIFETEST`` uses with ``ERROR=DELTA``:

    $$
    \widehat{\mathrm{Var}}\,\hat F_k(t) =
        \sum_{t_i \le t} \bigl(\hat F_k(t) - \hat F_k(t_i)\bigr)^2
            \frac{d_i}{n_i (n_i - d_i)}
        + \sum_{t_i \le t} \hat S(t_i^-)^2 \frac{d_{ik} (n_i - d_{ik})}{n_i^3}
        - 2 \sum_{t_i \le t} \bigl(\hat F_k(t) - \hat F_k(t_i)\bigr)
            \hat S(t_i^-) \frac{d_{ik}}{n_i^2}
    $$

    with $d_i$ the any-cause events at $t_i$ and $d_{ik}$ those of cause $k$.
    With a single cause this is Greenwood's formula for $1 - \hat S$.

    Parameters
    ----------
    event_time : (n,) array
    event_indicator : (n,) array
        Integer-valued: $0$ for censored, $k \ge 1$ for an event of cause $k$.
    cause : int
        Positive integer denoting the cause of interest.

    Returns
    -------
    StepFunction
        Variance on the same grid as :func:`aalen_johansen`; ``baseline=0.0``.

    References
    ----------
    Marubini, E., Valsecchi, M. G. (1995). Analysing Survival Data from
    Clinical Trials and Observational Studies. Wiley. Section 10.3.
    """
    time, n_at_risk, d_any, d_k, S_left, cif = _tabulate(
        event_time, event_indicator, cause
    )

    survivors = n_at_risk - d_any
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(survivors > 0, d_any / (n_at_risk * survivors), 0.0)
    b = S_left**2 * d_k * (n_at_risk - d_k) / n_at_risk**3
    c = S_left * d_k / n_at_risk**2

    # Each term is a sum over t_i <= t of a polynomial in F(t); expand the
    # squares so every sum is a prefix sum over the grid.
    a0, a1, a2 = np.cumsum(a), np.cumsum(a * cif), np.cumsum(a * cif**2)
    c0, c1 = np.cumsum(c), np.cumsum(c * cif)
    variance = cif**2 * a0 - 2.0 * cif * a1 + a2 + np.cumsum(b) - 2.0 * (cif * c0 - c1)
    variance = np.maximum(variance, 0.0)
    return StepFunction(time=time, value=variance, side="right", baseline=0.0)


def _tabulate(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    cause: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Per-unique-time counts behind the estimator and its variance.

    Returns ``(time, n_at_risk, d_any, d_cause, S_left, cif)`` where
    ``S_left`` is the any-cause KM survival just before each time.
    """
    if cause <= 0:
        raise ValueError(f"cause must be a positive integer, got {cause}")

    Y = np.asarray(event_time, dtype=np.float64)
    E = np.asarray(event_indicator, dtype=np.int64)

    order = np.argsort(Y, kind="stable")
    Y_sorted = Y[order]
    E_sorted = E[order]

    time, inverse = np.unique(Y_sorted, return_inverse=True)
    n_at_each = np.bincount(inverse)
    d_any = np.bincount(inverse, weights=(E_sorted > 0).astype(np.float64))
    d_k = np.bincount(inverse, weights=(E_sorted == cause).astype(np.float64))

    n_at_risk = (len(Y) - np.concatenate([[0], np.cumsum(n_at_each[:-1])])).astype(
        np.float64
    )
    S = np.cumprod(1.0 - d_any / n_at_risk)
    S_left = np.concatenate([[1.0], S[:-1]]).astype(np.float64)

    cif = np.cumsum(S_left * d_k / n_at_risk)
    return (
        time,
        n_at_risk,
        d_any.astype(np.float64),
        d_k.astype(np.float64),
        S_left,
        cif,
    )
