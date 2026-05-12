from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tausurv.distributions.base import SurvivalDistribution


class Exponential(SurvivalDistribution):
    r"""Exponential distribution with rate $\lambda$.

    $$
    f(t \mid \lambda) = \lambda \exp(-\lambda t), \qquad t > 0
    $$

    Memoryless; constant hazard $\lambda$. Special case of Weibull with
    $k = 1$, scale $= 1/\lambda$.

    Parameters
    ----------
    rate : array_like
        $\lambda > 0$.
    """

    def __init__(self, rate: ArrayLike) -> None:
        rate = np.asarray(rate, dtype=np.float64)
        if np.any(rate <= 0):
            raise ValueError("Exponential rate must be > 0")
        self.rate = rate

    def log_pdf(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t, dtype=np.float64)
        return np.where(t >= 0, np.log(self.rate) - self.rate * t, -np.inf)

    def log_survival(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t, dtype=np.float64)
        return np.where(t >= 0, -self.rate * t, 0.0)

    def quantile(self, p: ArrayLike) -> NDArray[np.float64]:
        p = np.asarray(p, dtype=np.float64)
        return -np.log1p(-p) / self.rate

    def mean(self) -> float:
        return float(1.0 / self.rate)
