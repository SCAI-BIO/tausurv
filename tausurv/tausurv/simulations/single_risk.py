from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tausurv.simulations._coefficients import _default_coefficients


def single_risk(
    n: int,
    *,
    n_features: int = 5,
    censoring_rate: float = 0.3,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int8]]:
    r"""Single-risk survival data from a Cox proportional-hazards model.

    Covariates $X \sim \mathcal{N}(0, I_d)$. Linear predictor
    $\eta_i = \beta^\top X_i$ with fixed alternating-sign coefficients.
    Event times $T_i \sim \mathrm{Exp}(\exp(\eta_i))$; censoring times
    $C_i \sim \mathrm{Exp}(c)$ with $c$ tuned to give an approximate
    target censoring rate (exact rate depends on $X$).

    Parameters
    ----------
    n : int
        Number of subjects.
    n_features : int, default 5
    censoring_rate : float, default 0.3
        Approximate marginal censoring fraction.
    seed : int, optional

    Returns
    -------
    X : (n, n_features) array
    event_time : (n,) array, $Y = \min(T, C)$.
    event_indicator : (n,) int8 array; $1$ if event observed, $0$ if censored.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    beta = _default_coefficients(n_features)
    rate_T = np.exp(X @ beta)
    T = rng.exponential(scale=1.0 / rate_T)

    rate_C = censoring_rate / (1.0 - censoring_rate)
    C = rng.exponential(scale=1.0 / rate_C, size=n)

    event_time = np.minimum(T, C)
    event_indicator = (T <= C).astype(np.int8)
    return X, event_time, event_indicator
