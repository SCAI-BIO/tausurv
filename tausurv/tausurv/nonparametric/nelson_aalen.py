from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from tausurv.step import StepFunction


def nelson_aalen(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
) -> StepFunction:
    r"""Nelson-Aalen estimate of the cumulative hazard.

    $$
    \hat\Lambda(t) = \sum_{t_i \le t} \frac{d_i}{n_i}
    $$

    with $d_i$ the events at $t_i$ and $n_i$ the number at risk just before
    $t_i$. Asymptotically equivalent to $-\log \hat S_{KM}$ but with different
    finite-sample behavior — Nelson-Aalen is the natural object when the
    cumulative hazard itself is of interest (e.g., bootstrap influence
    functions, Cox baseline hazard).

    Parameters
    ----------
    event_time : (n,) array
        Observed time $Y = \min(T, C)$.
    event_indicator : (n,) array
        $\delta = 1$ if event observed, $0$ if censored.

    Returns
    -------
    StepFunction
        $\hat\Lambda$ with ``baseline=0.0``; right-continuous.

    References
    ----------
    Nelson, W. (1969). Hazard plotting for incomplete failure data. Journal
    of Quality Technology, 1(1). Aalen, O. (1978). Nonparametric inference
    for a family of counting processes. Annals of Statistics, 6(4).
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
    cum_hazard = np.cumsum(d_at_each / n_at_risk)

    return StepFunction(time=time, value=cum_hazard, side="right", baseline=0.0)
