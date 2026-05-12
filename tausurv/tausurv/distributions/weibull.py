from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln

from tausurv.distributions.base import SurvivalDistribution


class Weibull(SurvivalDistribution):
    r"""Weibull distribution with shape $k$ and scale $\lambda$.

    $$
    f(t \mid k, \lambda) = \frac{k}{\lambda}
        \left(\frac{t}{\lambda}\right)^{k - 1}
        \exp\!\left(-\left(\frac{t}{\lambda}\right)^k\right),
    \qquad t > 0
    $$

    with survival $S(t) = \exp(-(t/\lambda)^k)$ and constant hazard
    when $k = 1$ (Exponential), increasing when $k > 1$, decreasing
    when $k < 1$.

    Parameters
    ----------
    shape : array_like
        $k > 0$.
    scale : array_like
        $\lambda > 0$.
    """

    def __init__(self, shape: ArrayLike, scale: ArrayLike) -> None:
        shape = np.asarray(shape, dtype=np.float64)
        scale = np.asarray(scale, dtype=np.float64)
        if np.any(shape <= 0) or np.any(scale <= 0):
            raise ValueError("Weibull shape and scale must be > 0")
        self.shape = shape
        self.scale = scale

    def log_pdf(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t, dtype=np.float64)
        k, lam = self.shape, self.scale
        with np.errstate(divide="ignore", invalid="ignore"):
            log_p = (
                np.log(k)
                - np.log(lam)
                + (k - 1.0) * (np.log(np.where(t > 0, t, 1.0)) - np.log(lam))
                - (t / lam) ** k
            )
        return np.where(t > 0, log_p, -np.inf)

    def log_survival(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t, dtype=np.float64)
        return np.where(t > 0, -((t / self.scale) ** self.shape), 0.0)

    def quantile(self, p: ArrayLike) -> NDArray[np.float64]:
        p = np.asarray(p, dtype=np.float64)
        return self.scale * (-np.log1p(-p)) ** (1.0 / self.shape)

    def mean(self) -> float:
        # E[T] = lambda * Gamma(1 + 1/k); use lgamma for numerical safety.
        return float(self.scale * np.exp(gammaln(1.0 + 1.0 / self.shape)))
