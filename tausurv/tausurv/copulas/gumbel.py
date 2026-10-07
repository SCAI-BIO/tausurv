from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tausurv.copulas.base import ArchimedeanCopula


class Gumbel(ArchimedeanCopula):
    r"""Gumbel copula with parameter $\theta \ge 1$.

    Generator $\varphi(t) = (-\log t)^\theta$,
    pseudo-inverse $\varphi^{-1}(s) = \exp(-s^{1/\theta})$. The
    bivariate distribution is

    $$
    C(u, v) = \exp\!\left(-\left[(-\log u)^\theta + (-\log v)^\theta\right]^{1/\theta}\right).
    $$

    Captures **upper-tail dependence** — extreme high-marginal events
    are co-incident more often than under independence. Kendall's
    $\tau = 1 - 1/\theta$. Tail coefficients
    $(\lambda_L, \lambda_U) = (0, 2 - 2^{1/\theta})$. Independence at
    $\theta = 1$.

    For competing risks, this is the natural choice when failure modes
    co-occur in the long-time tail (e.g., shared frailty for
    cardiovascular and cerebrovascular endpoints in old age).
    """

    theta_range = (1.0, 100.0)

    def phi(self, t):
        t = np.asarray(t, dtype=np.float64)
        return np.power(-np.log(t), self.theta)

    def phi_inv(self, s):
        s = np.asarray(s, dtype=np.float64)
        return np.exp(-np.power(s, 1.0 / self.theta))

    def log_phi_deriv_abs(self, t, k=1):
        # phi(t) = (-log t)^theta -> phi'(t) = theta * (-log t)^(theta-1) * (-1/t)
        # |phi'(t)| = theta * (-log t)^(theta-1) / t
        # log|phi'(t)| = log(theta) + (theta-1) log(-log t) - log t
        if k != 1:
            raise NotImplementedError(
                "Gumbel.log_phi_deriv_abs is implemented for k=1 only "
                "(closed-form bivariate density / d-variate cdf uses k=1)."
            )
        t = np.asarray(t, dtype=np.float64)
        return np.log(self.theta) + (self.theta - 1.0) * np.log(-np.log(t)) - np.log(t)

    def log_phi_inv_deriv_abs(self, s, k=1):
        # phi_inv(s) = exp(-s^(1/theta))
        # (phi_inv)'(s) = -(1/theta) * s^(1/theta - 1) * exp(-s^(1/theta))
        # (phi_inv)''(s) = exp(-s^(1/theta)) * (1/theta^2) * s^(2/theta - 2) * (s^(1/theta) + theta - 1)
        #               wait let me re-derive.
        # let w = s^(1/theta). phi_inv = exp(-w). dw/ds = (1/theta) s^(1/theta - 1).
        # (phi_inv)' = -exp(-w) * (1/theta) s^(1/theta - 1)
        # (phi_inv)'' = (phi_inv)' * (-dw/ds) + (-exp(-w)) * (1/theta)(1/theta - 1) s^(1/theta - 2)
        #             = exp(-w) * (1/theta)^2 s^(2/theta - 2) - exp(-w) (1/theta)(1/theta - 1) s^(1/theta - 2)
        #             = exp(-w) * s^(1/theta - 2) * [ (1/theta)^2 * s^(1/theta) - (1/theta)(1/theta - 1) ]
        #             = exp(-w) * s^(1/theta - 2) / theta^2 * [ s^(1/theta) - (1 - theta) ]  ... since 1/theta - 1 = (1-theta)/theta
        # Hmm let me redo: (1/theta)(1/theta - 1) = (1 - theta)/theta^2. So:
        # (phi_inv)''(s) = exp(-w) * (1/theta)^2 * s^(2/theta - 2)  -  exp(-w) * (1 - theta)/theta^2 * s^(1/theta - 2)
        #                = exp(-w) * s^(1/theta - 2) / theta^2 * (s^(1/theta) + theta - 1)
        # (for theta >= 1, theta - 1 >= 0; and s^(1/theta) > 0, so the bracket is positive).
        s = np.asarray(s, dtype=np.float64)
        if k == 1:
            log_s = np.log(s)
            w = np.power(s, 1.0 / self.theta)
            return -w - np.log(self.theta) + (1.0 / self.theta - 1.0) * log_s
        if k == 2:
            log_s = np.log(s)
            w = np.power(s, 1.0 / self.theta)
            # log|phi_inv''| = -w + (1/theta - 2) log s - 2 log theta + log(w + theta - 1)
            return (
                -w
                + (1.0 / self.theta - 2.0) * log_s
                - 2.0 * np.log(self.theta)
                + np.log(w + self.theta - 1.0)
            )
        raise NotImplementedError(
            f"Gumbel.log_phi_inv_deriv_abs is implemented for k in {{1, 2}}; "
            f"d-variate log_pdf for d > 2 is not in v0.1."
        )

    def kendalls_tau(self) -> float:
        return 1.0 - 1.0 / self.theta

    def tail_dependence(self) -> tuple[float, float]:
        return (0.0, 2.0 - 2.0 ** (1.0 / self.theta))

    def _sample_frailty(self, size, rng):
        r"""$M$ is positive stable with $\alpha = 1/\theta$.

        Implements the Chambers–Mallows–Stuck algorithm specialized to
        one-sided ($\beta = 1$) positive stable laws — gives $M > 0$
        with the right Laplace transform $\exp(-s^{1/\theta})$.
        """
        alpha = 1.0 / self.theta
        if alpha == 1.0:  # Gumbel theta=1 is independence; M is a point mass at 1.
            return np.ones(size, dtype=np.float64)
        # CMS for positive stable, beta=1, sigma chosen to match the standard form.
        # M = sin(alpha * (V + pi/2)) / cos(V)^(1/alpha)
        #   * (cos(V - alpha * (V + pi/2)) / W) ** ((1 - alpha) / alpha)
        # Chambers, Mallows & Stuck (1976), JASA 71(354), 340-344.
        V = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size=size)
        W = rng.exponential(scale=1.0, size=size)
        phase = alpha * (V + np.pi / 2.0)
        return (
            np.sin(phase)
            / np.power(np.cos(V), 1.0 / alpha)
            * np.power(np.cos(V - phase) / W, (1.0 - alpha) / alpha)
        )
