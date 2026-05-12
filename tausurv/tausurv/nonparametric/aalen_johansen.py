from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

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

    n_at_risk = len(Y) - np.concatenate([[0], np.cumsum(n_at_each[:-1])])
    S = np.cumprod(1.0 - d_any / n_at_risk)
    S_left = np.concatenate([[1.0], S[:-1]])

    cif = np.cumsum(S_left * d_k / n_at_risk)
    return StepFunction(time=time, value=cif, side="right", baseline=0.0)
