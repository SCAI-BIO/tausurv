from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tausurv.copulas.base import ArchimedeanCopula


class Independence(ArchimedeanCopula):
    r"""Independence copula $C(u_1, \dots, u_d) = \prod_i u_i$.

    Generator $\varphi(t) = -\log t$. Useful as a baseline (no
    dependence) and as the limit of every Archimedean family at its
    independence boundary (Clayton $\theta \to 0$, Gumbel/Joe $\theta = 1$,
    Frank $\theta \to 0$).
    """

    theta_range = (0.0, 0.0)

    def __init__(self) -> None:
        self.theta = 0.0

    def phi(self, t):
        t = np.asarray(t, dtype=np.float64)
        return -np.log(t)

    def phi_inv(self, s):
        s = np.asarray(s, dtype=np.float64)
        return np.exp(-s)

    def log_phi_deriv_abs(self, t, k=1):
        # phi(t) = -log(t), phi^(k)(t) = -(k-1)! * t^(-k)  (sign (-1)^k)
        from scipy.special import gammaln

        t = np.asarray(t, dtype=np.float64)
        return gammaln(k) - k * np.log(t)  # = log((k-1)!) - k*log(t)

    def log_phi_inv_deriv_abs(self, s, k=1):
        # phi_inv(s) = exp(-s), |(phi_inv)^(k)(s)| = exp(-s)
        s = np.asarray(s, dtype=np.float64)
        return -s

    def cdf(self, u):
        u = np.asarray(u, dtype=np.float64)
        return np.prod(u, axis=-1)

    def log_pdf(self, u):
        u = np.asarray(u, dtype=np.float64)
        return np.zeros(u.shape[:-1])

    def kendalls_tau(self) -> float:
        return 0.0

    def tail_dependence(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def sample(self, size, d=2, rng=None) -> NDArray[np.float64]:
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(size=(size, d))
