from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad

from tausurv.copulas.base import ArchimedeanCopula


class Joe(ArchimedeanCopula):
    r"""Joe copula with parameter $\theta \ge 1$.

    Generator $\varphi(t) = -\log\!\big[1 - (1 - t)^\theta\big]$;
    pseudo-inverse $\varphi^{-1}(s) = 1 - (1 - e^{-s})^{1/\theta}$.

    Captures **strong upper-tail dependence** — more concentrated in
    the extreme upper tail than Gumbel. Kendall's $\tau$ has no
    closed form (numerical integration is used). Tail coefficients
    $(\lambda_L, \lambda_U) = (0, 2 - 2^{1/\theta})$. Independence at
    $\theta = 1$.

    Use when failure modes co-occur very near the upper boundary of the
    marginal distributions (e.g., late-life death from any cause).
    """

    theta_range = (1.0, 50.0)

    def phi(self, t):
        t = np.asarray(t, dtype=np.float64)
        # phi(t) = -log(1 - (1 - t)^theta)
        # Use log1p(-x) for stability when (1-t)^theta is small.
        log_1m_t = np.log1p(-t)  # log(1 - t)
        return -np.log1p(-np.exp(self.theta * log_1m_t))

    def phi_inv(self, s):
        s = np.asarray(s, dtype=np.float64)
        # phi_inv(s) = 1 - (1 - e^{-s})^(1/theta), with phi_inv(0) = 1.
        # Use ``-expm1(-s)`` for an accurate small-``s`` 1-e^{-s}, and
        # clamp away from zero so the subsequent log is finite (gives
        # the correct ``1.0`` limit at s = 0).
        s_safe = np.maximum(s, 1e-300)
        return 1.0 - np.exp(np.log(-np.expm1(-s_safe)) / self.theta)

    def log_phi_deriv_abs(self, t, k=1):
        # phi'(t) = -theta * (1 - t)^{theta - 1} / (1 - (1 - t)^theta)
        # |phi'(t)| = theta * (1 - t)^{theta - 1} / (1 - (1 - t)^theta)
        # log|phi'(t)| = log(theta) + (theta - 1) log(1 - t) - log(1 - (1 - t)^theta)
        if k != 1:
            raise NotImplementedError(
                "Joe.log_phi_deriv_abs is implemented for k=1 only."
            )
        t = np.asarray(t, dtype=np.float64)
        log_1m_t = np.log1p(-t)
        return (
            np.log(self.theta)
            + (self.theta - 1.0) * log_1m_t
            - np.log1p(-np.exp(self.theta * log_1m_t))
        )

    def log_phi_inv_deriv_abs(self, s, k=1):
        # Let q = 1 - e^{-s}. phi_inv(s) = 1 - q^{1/theta}.
        # dq/ds = e^{-s} = 1 - q.
        # phi_inv'(s) = -(1/theta) q^{1/theta - 1} * (1 - q)
        # |phi_inv'(s)| = q^{1/theta - 1} (1 - q) / theta
        # log|phi_inv'(s)| = (1/theta - 1) log q + log(1 - q) - log(theta)
        s = np.asarray(s, dtype=np.float64)
        # q = 1 - e^{-s}; log q = log1p(-exp(-s)); log(1 - q) = -s.
        log_q = np.log1p(-np.exp(-s))
        log_1mq = -s
        if k == 1:
            return (1.0 / self.theta - 1.0) * log_q + log_1mq - np.log(self.theta)
        if k == 2:
            # phi_inv''(s) = d/ds [-(1/theta) q^{1/theta - 1} (1 - q)]
            #   = -(1/theta) * [(1/theta - 1) q^{1/theta - 2} (1-q) (1-q) + q^{1/theta - 1} * (-(1-q))]
            #   = -(1/theta) q^{1/theta - 2} (1 - q) [ (1/theta - 1)(1 - q) - q ]
            #   = -(1/theta) q^{1/theta - 2} (1 - q) [ (1/theta - 1) - (1/theta - 1) q - q ]
            #   = -(1/theta) q^{1/theta - 2} (1 - q) [ (1/theta - 1) - (1/theta) q ]
            #   = -(1/theta) q^{1/theta - 2} (1 - q) (1/theta) [ (1 - theta) - q ]   ... wait
            # let me redo carefully.
            # Let a = 1/theta. phi_inv = 1 - q^a. phi_inv' = -a q^{a-1} q' where q' = 1 - q.
            # phi_inv'(s) = -a (1 - q) q^{a - 1}.
            # phi_inv''(s) = d/ds[-a (1 - q) q^{a-1}]
            #   = -a [ -q'(s) * q^{a-1} + (1 - q) (a-1) q^{a-2} q'(s) ]
            #   = -a q'(s) [ -q^{a-1} + (1 - q)(a-1) q^{a-2} ]
            #   = -a (1 - q) q^{a-2} [ -q + (1 - q)(a - 1) ]
            #   = -a (1 - q) q^{a-2} [ (a - 1) - q(a - 1) - q ]
            #   = -a (1 - q) q^{a-2} [ (a - 1) - q*a ]
            #   = -a (1 - q) q^{a-2} (a - 1 - a*q)
            # = a (1 - q) q^{a-2} (1 - a + a*q)  ... factor out -1
            # = a (1 - q) q^{a-2} (1 - a(1 - q))
            # Sign analysis: for theta >= 1, a = 1/theta in (0, 1]. (1 - q) > 0, q^{a-2} > 0. (1 - a(1-q)) > 0 when a(1-q) < 1; since a <= 1 and (1 - q) < 1, yes always positive.
            # So phi_inv''(s) > 0 — good (convex).
            # log|phi_inv''(s)| = log a + log(1 - q) + (a - 2) log q + log(1 - a(1 - q))
            #   = -log(theta) + log(1 - q) + (1/theta - 2) log q + log(1 - (1 - q)/theta)
            a = 1.0 / self.theta
            return (
                -np.log(self.theta)
                + log_1mq
                + (a - 2.0) * log_q
                + np.log1p(-a * (1.0 - np.exp(log_q)))
            )
        raise NotImplementedError(
            f"Joe.log_phi_inv_deriv_abs supports k in {{1, 2}}; got k={k}."
        )

    def kendalls_tau(self) -> float:
        r"""Numerical: $\tau = 1 + 4 \int_0^1 \varphi(t) / \varphi'(t) \, dt$."""
        if self.theta == 1.0:
            return 0.0
        # tau = 1 + 4 * int_0^1 phi(t)/phi'(t) dt
        # phi(t) = -log(1 - (1-t)^theta); phi'(t) = -theta (1-t)^{theta-1} / (1 - (1-t)^theta)
        # Use the integrand 1 - 4 sum form: tau_Joe = 1 - 4 * sum_{k=1}^inf 1 / (k(theta k + 2)(theta (k - 1) + 2))
        # Numerical sum is faster and simpler.
        total = 0.0
        for k in range(1, 1000):
            term = 1.0 / (k * (self.theta * k + 2.0) * (self.theta * (k - 1) + 2.0))
            total += term
            if term < 1e-15:
                break
        return 1.0 - 4.0 * total

    def tail_dependence(self) -> tuple[float, float]:
        return (0.0, 2.0 - 2.0 ** (1.0 / self.theta))

    def _sample_frailty(self, size, rng):
        r"""Sibuya frailty: not implemented in v0.1.

        The Sibuya distribution has a heavy power-law tail
        ($P(X > k) \sim k^{-1/\theta}$), so a naive truncated inverse-CDF
        sampler loses meaningful mass at large $\theta$. The proper
        algorithm requires Mittag-Leffler or exponentially-tilted stable
        sampling (Hofert 2011, Devroye 2009); we'll add it when there's a
        real consumer. For now, use a different family for sampling, or
        sample Joe copulas via conditional inversion / Rosenblatt
        externally.
        """
        raise NotImplementedError(
            "Joe._sample_frailty (Sibuya) is not implemented in v0.1. "
            "Use Clayton/Gumbel/Frank for sampling, or implement "
            "conditional-inversion sampling for Joe externally."
        )
