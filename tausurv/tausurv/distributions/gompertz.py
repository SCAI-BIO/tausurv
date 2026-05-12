from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tausurv.distributions.base import SurvivalDistribution


class Gompertz(SurvivalDistribution):
    r"""Gompertz distribution with shape $a$ and rate $b$.

    Exponentially increasing hazard
    $\lambda(t) = b \, e^{a t}$, integrating to

    $$
    S(t \mid a, b) = \exp\!\left(-\frac{b}{a}\,(e^{a t} - 1)\right),
    \qquad t > 0,\;\; a, b > 0.
    $$

    Classical actuarial / demographic model for adult mortality.

    Parameters
    ----------
    shape : array_like
        $a > 0$ — rate at which the hazard grows.
    rate : array_like
        $b > 0$ — baseline hazard at $t = 0$.
    """

    def __init__(self, shape: ArrayLike, rate: ArrayLike) -> None:
        shape = np.asarray(shape, dtype=np.float64)
        rate = np.asarray(rate, dtype=np.float64)
        if np.any(shape <= 0) or np.any(rate <= 0):
            raise ValueError("Gompertz shape and rate must be > 0")
        self.shape = shape
        self.rate = rate

    def log_pdf(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t, dtype=np.float64)
        a, b = self.shape, self.rate
        # log f(t) = log b + a*t - (b/a) * (exp(a*t) - 1)
        log_p = np.log(b) + a * t - (b / a) * np.expm1(a * t)
        return np.where(t >= 0, log_p, -np.inf)

    def log_survival(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t, dtype=np.float64)
        a, b = self.shape, self.rate
        return np.where(t >= 0, -(b / a) * np.expm1(a * t), 0.0)

    def quantile(self, p: ArrayLike) -> NDArray[np.float64]:
        p = np.asarray(p, dtype=np.float64)
        a, b = self.shape, self.rate
        # Invert S(t) = exp(-(b/a)(exp(a t) - 1)) = 1 - p
        return np.log1p(-(a / b) * np.log1p(-p)) / a

    def mean(self) -> float:
        # E[T] = (1/a) * exp(b/a) * E1(b/a) where E1 is the exponential integral.
        from scipy.special import exp1

        a, b = self.shape, self.rate
        return float(np.exp(b / a) * exp1(b / a) / a)
