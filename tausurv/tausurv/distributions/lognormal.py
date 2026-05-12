from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import log_ndtr, ndtri

from tausurv.distributions.base import SurvivalDistribution


class LogNormal(SurvivalDistribution):
    r"""Log-normal distribution with $\log T \sim \mathcal{N}(\mu, \sigma^2)$.

    $$
    f(t \mid \mu, \sigma) = \frac{1}{t \sigma \sqrt{2\pi}}
        \exp\!\left(-\frac{(\log t - \mu)^2}{2 \sigma^2}\right), \qquad t > 0
    $$

    Hump-shaped hazard (rises then falls). The log-survival uses
    :func:`scipy.special.log_ndtr` for numerical stability in the upper
    tail.

    Parameters
    ----------
    mu : array_like
        Location of $\log T$.
    sigma : array_like
        Standard deviation of $\log T$; $\sigma > 0$.
    """

    def __init__(self, mu: ArrayLike, sigma: ArrayLike) -> None:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        if np.any(sigma <= 0):
            raise ValueError("LogNormal sigma must be > 0")
        self.mu = mu
        self.sigma = sigma

    def log_pdf(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t, dtype=np.float64)
        log_t = np.log(np.where(t > 0, t, 1.0))
        z = (log_t - self.mu) / self.sigma
        log_p = -log_t - np.log(self.sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * z**2
        return np.where(t > 0, log_p, -np.inf)

    def log_survival(self, t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t, dtype=np.float64)
        log_t = np.log(np.where(t > 0, t, 1.0))
        z = (log_t - self.mu) / self.sigma
        # log P(Z > z) = log_ndtr(-z) — stable in the tails.
        return np.where(t > 0, log_ndtr(-z), 0.0)

    def quantile(self, p: ArrayLike) -> NDArray[np.float64]:
        p = np.asarray(p, dtype=np.float64)
        return np.exp(self.mu + self.sigma * ndtri(p))

    def mean(self) -> float:
        return float(np.exp(self.mu + 0.5 * self.sigma**2))
