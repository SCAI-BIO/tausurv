from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tausurv.simulations._coefficients import _default_coefficients


def competing_risks(
    n: int,
    *,
    n_features: int = 5,
    n_causes: int = 2,
    censoring_rate: float = 0.3,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int8]]:
    r"""Competing-risks survival data with per-cause Cox-style hazards.

    For each cause $k \in \{1, \dots, K\}$, a latent failure time
    $T_{k,i} \sim \mathrm{Exp}(\exp(\beta_k^\top X_i))$. The observed event
    time is $\min_k T_{k,i}$ and the cause is the corresponding $k$.
    Coefficients scale by $(1 - 0.2 k)$ so later causes carry weaker signal.

    Event coding follows the package convention: $\delta = 0$ censored,
    $\delta = k \ge 1$ event of cause $k$.

    Parameters
    ----------
    n : int
    n_features : int, default 5
    n_causes : int, default 2
    censoring_rate : float, default 0.3
    seed : int, optional

    Returns
    -------
    X : (n, n_features) array
    event_time : (n,) array
    event_indicator : (n,) int8 array with values in $\{0, 1, \dots, K\}$.
    """
    if n_causes < 1:
        raise ValueError(f"n_causes must be >= 1, got {n_causes}")

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))

    latent_times = np.empty((n_causes, n))
    for k in range(n_causes):
        beta_k = _default_coefficients(n_features) * (1.0 - 0.2 * k)
        rate_k = np.exp(X @ beta_k)
        latent_times[k] = rng.exponential(scale=1.0 / rate_k)

    T = latent_times.min(axis=0)
    cause = latent_times.argmin(axis=0) + 1

    rate_C = censoring_rate / (1.0 - censoring_rate)
    C = rng.exponential(scale=1.0 / rate_C, size=n)

    event_time = np.minimum(T, C)
    is_event = T <= C
    event_indicator = np.where(is_event, cause, 0).astype(np.int8)

    return X, event_time, event_indicator
