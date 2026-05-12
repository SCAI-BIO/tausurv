from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from tausurv.step import StepFunction


def kaplan_meier(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
) -> StepFunction:
    r"""Kaplan-Meier estimate of the survival function.

    $$
    \hat S(t) = \prod_{t_i \le t} \left(1 - \frac{d_i}{n_i}\right)
    $$

    where $t_i$ are the unique observed times, $d_i$ the number of events at
    $t_i$, and $n_i$ the number at risk just before $t_i$. Reports $\hat S$ at
    every unique observation time (events and censorings combined), so callers
    can step-evaluate at any subject's $Y_i$.

    Parameters
    ----------
    event_time : (n,) array
        Observed time $Y = \min(T, C)$.
    event_indicator : (n,) array
        $\delta = 1$ if event observed, $0$ if censored.

    Returns
    -------
    StepFunction
        Right-continuous $\hat S$ over the unique observation times.
        Evaluate as ``km(times)``; pass ``side="left"`` for $\hat S(t^-)$.

    References
    ----------
    Kaplan & Meier (1958), JASA 53(282).
    """
    Y = np.asarray(event_time, dtype=np.float64)
    delta = np.asarray(event_indicator, dtype=np.int8)

    order = np.argsort(Y, kind="stable")
    Y_sorted = Y[order]
    delta_sorted = delta[order]

    time, inverse = np.unique(Y_sorted, return_inverse=True)
    n_at_each = np.bincount(inverse)
    d_at_each = np.bincount(inverse, weights=delta_sorted).astype(np.float64)

    n_at_risk = len(Y) - np.concatenate([[0], np.cumsum(n_at_each[:-1])])
    survival = np.cumprod(1.0 - d_at_each / n_at_risk)

    return StepFunction(time=time, value=survival, side="right", baseline=1.0)
