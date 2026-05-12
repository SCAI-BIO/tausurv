from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.stats import logser

from tausurv.copulas.base import ArchimedeanCopula


class Frank(ArchimedeanCopula):
    r"""Frank copula with parameter $\theta \in \mathbb{R} \setminus \{0\}$.

    Generator
    $\varphi(t) = -\log\!\big[(e^{-\theta t} - 1) / (e^{-\theta} - 1)\big]$;
    pseudo-inverse
    $\varphi^{-1}(s) = -\theta^{-1}\log\!\big[1 + e^{-s}(e^{-\theta} - 1)\big]$.

    The only Archimedean family that is **radially symmetric** (no tail
    dependence) and admits **negative dependence** ($\theta < 0$).
    Good default for mid-range, symmetric dependence with no tail
    co-occurrence. Independence at $\theta \to 0$.

    Kendall's $\tau = 1 - 4/\theta \cdot (1 - D_1(\theta))$ where $D_1$
    is the first Debye function (numerical integration is used). Tail
    coefficients $(\lambda_L, \lambda_U) = (0, 0)$.

    Sampling supports $\theta > 0$ in v0.1 (logarithmic-series frailty).
    """

    theta_range = (-50.0, 50.0)

    def __init__(self, theta: float) -> None:
        if theta == 0:
            raise ValueError("Frank theta cannot be 0 (use Independence instead)")
        super().__init__(theta)

    def phi(self, t):
        t = np.asarray(t, dtype=np.float64)
        # phi(t) = -log( (e^{-theta t} - 1) / (e^{-theta} - 1) )
        # Use expm1 for stability.
        return -np.log(np.expm1(-self.theta * t) / np.expm1(-self.theta))

    def phi_inv(self, s):
        s = np.asarray(s, dtype=np.float64)
        # phi_inv(s) = -log(1 + e^{-s} * (e^{-theta} - 1)) / theta
        return -np.log1p(np.exp(-s) * np.expm1(-self.theta)) / self.theta

    def log_phi_deriv_abs(self, t, k=1):
        # phi'(t) = theta * e^{-theta t} / (e^{-theta t} - 1)
        # |phi'(t)| = theta / (1 - e^{theta t})   when theta > 0
        # Generic: log|phi'(t)| = log|theta| - theta t - log|1 - e^{-theta t}|
        #   = log|theta| - theta t - log(-expm1(-theta t))   (for theta t > 0)
        # For theta < 0, theta t can be negative; use abs.
        if k != 1:
            raise NotImplementedError(
                "Frank.log_phi_deriv_abs is implemented for k=1 only."
            )
        t = np.asarray(t, dtype=np.float64)
        # |phi'(t)| = |theta| * |e^{-theta t} / (e^{-theta t} - 1)|
        #           = |theta| * 1 / |1 - e^{theta t}|
        # log|phi'(t)| = log|theta| - log|expm1(theta t)|
        return np.log(np.abs(self.theta)) - np.log(np.abs(np.expm1(self.theta * t)))

    def log_phi_inv_deriv_abs(self, s, k=1):
        # Let q = 1 - e^{-theta}, so phi_inv(s) = -log(1 - q e^{-s}) / theta. (Note q = -expm1(-theta).)
        # Let r = q e^{-s}. Then phi_inv(s) = -log(1 - r) / theta.
        # (phi_inv)'(s) = (-1/theta) * (dr/ds) / (1 - r) = (-1/theta) * (-r) / (1 - r) = r / (theta (1 - r))
        # |phi_inv'(s)| = |r / (theta (1 - r))|
        # log|phi_inv'(s)| = log r - log|theta| - log|1 - r|  (assuming r > 0 and 1 - r > 0)
        s = np.asarray(s, dtype=np.float64)
        q = -np.expm1(-self.theta)  # equals 1 - e^{-theta}
        log_abs_q = np.log(np.abs(q))
        # r = q * exp(-s); log|r| = log|q| - s
        if k == 1:
            log_r = log_abs_q - s
            log_one_minus_r = np.log1p(-q * np.exp(-s))
            return log_r - np.log(np.abs(self.theta)) - log_one_minus_r
        if k == 2:
            # (phi_inv)''(s) = -r (1 - 2r ... ) hmm let me redo.
            # Let f(s) = -log(1 - r) / theta with r = q e^{-s}.
            # f'(s) = r / (theta (1 - r))
            # f''(s) = ?
            #   numerator: dr/ds = -r.
            #   derivative of r/(theta(1-r)) w.r.t. s:
            #   = [(-r) (1 - r) - r * (-)(-r)] / (theta (1-r)^2)
            #     wait: derivative of r/(1-r) = (dr/ds (1 - r) - r * (-dr/ds)) / (1-r)^2
            #         = dr/ds * (1 - r + r) / (1 - r)^2 = dr/ds / (1 - r)^2
            #         = -r / (1 - r)^2
            #   So f''(s) = -r / (theta (1 - r)^2)
            # |f''(s)| = r / (|theta| (1 - r)^2)   (since theta, (1-r)^2 > 0 in domain)
            # log|f''(s)| = log r - log|theta| - 2 log|1 - r|
            log_r = log_abs_q - s
            log_one_minus_r = np.log1p(-q * np.exp(-s))
            return log_r - np.log(np.abs(self.theta)) - 2.0 * log_one_minus_r
        raise NotImplementedError(
            f"Frank.log_phi_inv_deriv_abs supports k in {{1, 2}}; got k={k}."
        )

    def kendalls_tau(self) -> float:
        # tau = 1 - 4/theta * (1 - D_1(theta))
        # D_1(theta) = (1/theta) int_0^theta x / (e^x - 1) dx
        if self.theta == 0:
            return 0.0

        def integrand(x):
            return x / np.expm1(x) if x != 0 else 1.0

        debye_1, _ = quad(integrand, 0.0, abs(self.theta), limit=200)
        debye_1 /= abs(self.theta)
        return 1.0 - 4.0 / self.theta * (1.0 - debye_1)

    def tail_dependence(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def _sample_frailty(self, size, rng):
        r"""Logarithmic-series frailty with parameter $p = 1 - e^{-\theta}$ for $\theta > 0$.

        For $\theta < 0$, no positive-stable frailty exists (Frank with
        negative $\theta$ is still a valid copula via direct
        construction, but Marshall–Olkin sampling does not apply). Raise.
        """
        if self.theta <= 0:
            raise NotImplementedError(
                "Marshall-Olkin sampling for Frank is only implemented for "
                f"theta > 0 in v0.1, got theta={self.theta}. Use the "
                "conditional-inversion algorithm for negative dependence."
            )
        p = -np.expm1(-self.theta)
        # logser is implemented in scipy.stats and supports rvs with random_state.
        return logser.rvs(p=p, size=size, random_state=rng).astype(np.float64)
