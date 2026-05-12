from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tausurv.copulas.base import ArchimedeanCopula


class Clayton(ArchimedeanCopula):
    r"""Clayton copula with parameter $\theta > 0$.

    Generator $\varphi(t) = (t^{-\theta} - 1) / \theta$,
    pseudo-inverse $\varphi^{-1}(s) = (1 + \theta s)^{-1/\theta}$. The
    bivariate distribution is

    $$
    C(u, v) = (u^{-\theta} + v^{-\theta} - 1)^{-1/\theta}.
    $$

    Captures **lower-tail dependence** — extreme low-marginal events
    are co-incident more often than under independence. Kendall's
    $\tau = \theta / (\theta + 2)$. Tail coefficients
    $(\lambda_L, \lambda_U) = (2^{-1/\theta}, 0)$.

    Independence limit at $\theta \to 0$.
    """

    theta_range = (1e-8, 100.0)

    def phi(self, t):
        t = np.asarray(t, dtype=np.float64)
        return (np.power(t, -self.theta) - 1.0) / self.theta

    def phi_inv(self, s):
        s = np.asarray(s, dtype=np.float64)
        return np.power(1.0 + self.theta * s, -1.0 / self.theta)

    def log_phi_deriv_abs(self, t, k=1):
        # phi^(k)(t) = (-1)^k * theta^(k-1) * (1)(theta+1)...(theta+k-2) * t^(-theta-k+1)... wait
        # phi(t) = (t^(-theta) - 1)/theta -> phi'(t) = -t^(-theta-1)
        # phi''(t) = (theta+1) t^(-theta-2). General: phi^(k)(t) = (-1)^k * (theta+1)(theta+2)...(theta+k-1) * t^(-theta-k)
        # (for k=1: -t^(-theta-1); for k=2: (theta+1) t^(-theta-2); etc.)
        # log |phi^(k)(t)| = sum_{j=1}^{k-1} log(theta+j) - (theta+k) log(t)
        t = np.asarray(t, dtype=np.float64)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        log_factor = float(np.sum(np.log(self.theta + np.arange(1, k))))
        return log_factor - (self.theta + k) * np.log(t)

    def log_phi_inv_deriv_abs(self, s, k=1):
        # phi_inv(s) = (1 + theta s)^(-1/theta)
        # (phi_inv)^(k)(s) = (-1)^k * theta^k * (1/theta)(1/theta + 1)...(1/theta + k - 1) * (1 + theta s)^(-1/theta - k)
        # = (-1)^k * (1)(1 + theta)...(1 + (k-1) theta) * (1 + theta s)^(-1/theta - k)
        s = np.asarray(s, dtype=np.float64)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        log_factor = float(np.sum(np.log1p(self.theta * np.arange(k))))
        return log_factor - (1.0 / self.theta + k) * np.log1p(self.theta * s)

    def kendalls_tau(self) -> float:
        return self.theta / (self.theta + 2.0)

    def tail_dependence(self) -> tuple[float, float]:
        return (2.0 ** (-1.0 / self.theta), 0.0)

    def _sample_frailty(self, size, rng):
        # M ~ Gamma(shape=1/theta, rate=1/theta) so E[exp(-sM)] = (1+theta s)^(-1/theta).
        return rng.gamma(shape=1.0 / self.theta, scale=self.theta, size=size)
