from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tausurv.distributions.base import SurvivalDistribution


class LogLogistic(SurvivalDistribution):
    r"""Log-logistic (Fisk) distribution with shape $k$ and scale $\lambda$.

    $$
    S(t \mid k, \lambda) = \frac{1}{1 + (t / \lambda)^k}, \qquad t > 0
    $$

    Closed-form CDF/survival makes inference straightforward; unimodal
    hazard when $k > 1$ (rises then falls), monotone decreasing when
    $k \le 1$.

    Mean exists only for $k > 1$:
    $E[T] = \lambda \, \pi / (k \sin(\pi/k))$. Returns ``inf`` when
    $k \le 1$.

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
            raise ValueError("LogLogistic shape and scale must be > 0")
        self.shape = shape
        self.scale = scale

    def log_pdf(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t, dtype=np.float64)
        k, lam = self.shape, self.scale
        log_t = np.log(np.where(t > 0, t, 1.0))
        z_log = log_t - np.log(lam)  # log(t/lam)
        # log f(t) = log k - log lam + (k-1) * log(t/lam) - 2 * log1p((t/lam)^k)
        log_p = (
            np.log(k)
            - np.log(lam)
            + (k - 1.0) * z_log
            - 2.0 * np.log1p(np.exp(k * z_log))
        )
        return np.where(t > 0, log_p, -np.inf)

    def log_survival(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t, dtype=np.float64)
        k, lam = self.shape, self.scale
        log_t = np.log(np.where(t > 0, t, 1.0))
        return np.where(t > 0, -np.log1p(np.exp(k * (log_t - np.log(lam)))), 0.0)

    def quantile(self, p: ArrayLike) -> NDArray[np.float64]:
        p = np.asarray(p, dtype=np.float64)
        return self.scale * (p / (1.0 - p)) ** (1.0 / self.shape)

    def mean(self) -> float:
        k, lam = self.shape, self.scale
        if np.any(k <= 1):
            return float(np.inf)
        return float(lam * np.pi / (k * np.sin(np.pi / k)))
