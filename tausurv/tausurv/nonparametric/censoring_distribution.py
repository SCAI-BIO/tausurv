from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from tausurv.nonparametric.kaplan_meier import kaplan_meier
from tausurv.step import StepFunction


def censoring_distribution(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
) -> StepFunction:
    r"""Kaplan-Meier estimate of the censoring distribution $\hat G(t) = P(C > t)$.

    Reverse-KM: the censoring indicator becomes the event, so subjects with
    $\delta = 0$ drive the drops in $\hat G$. The natural input to IPCW
    estimators (Uno's C, Blanche's C, IPCW Brier, time-dependent AUC).

    Parameters
    ----------
    event_time : (n,) array
        Observed time $Y = \min(T, C)$.
    event_indicator : (n,) array
        $\delta = 1$ if event observed, $0$ if censored.

    Returns
    -------
    StepFunction
        $\hat G$; use ``G(times, side="left")`` for the left limit
        $\hat G(t^-)$ that IPCW weight formulas typically require.
    """
    return kaplan_meier(event_time, 1 - np.asarray(event_indicator, dtype=np.int8))
