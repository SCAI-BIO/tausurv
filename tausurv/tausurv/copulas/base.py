r"""Archimedean copula base class (Nelsen convention, numpy/scipy-backed).

A $d$-variate Archimedean copula has the form

$$
C(u_1, \dots, u_d) = \varphi^{-1}\!\left(\sum_{i=1}^d \varphi(u_i)\right)
$$

where the **generator** $\varphi : [0, 1] \to [0, \infty]$ is decreasing,
convex, with $\varphi(1) = 0$, and $\varphi^{-1}$ is its pseudo-inverse
(equivalently, the Laplace transform of a positive frailty distribution
under the Marshall–Olkin representation).

The same construction defines a **survival copula** if the inputs are
interpreted as marginal survival values rather than CDF values — the
math is identical, the inputs differ in meaning.

Concrete subclasses implement:

- ``phi``, ``phi_inv``
- ``log_phi_deriv_abs(t, k)``: $\log |\varphi^{(k)}(t)|$ for $k \in \{1, 2\}$
- ``log_phi_inv_deriv_abs(s, k)``: $\log |(\varphi^{-1})^{(k)}(s)|$
- ``kendalls_tau``, ``tail_dependence``, ``_sample_frailty``
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ArchimedeanCopula:
    """Base class for Archimedean copulas (Nelsen convention)."""

    #: Allowed range for the dependence parameter ``theta``.
    theta_range: tuple[float, float] = (-np.inf, np.inf)

    #: Dependence parameter.
    theta: float

    def __init__(self, theta: float) -> None:
        lo, hi = self.theta_range
        if not (lo <= theta <= hi):
            raise ValueError(
                f"{type(self).__name__}: theta must be in [{lo}, {hi}], got {theta}"
            )
        self.theta = float(theta)

    def phi(self, t: ArrayLike) -> NDArray[np.float64]:
        r"""Generator $\varphi(t)$: $[0, 1] \to [0, \infty]$, decreasing convex."""
        raise NotImplementedError

    def phi_inv(self, s: ArrayLike) -> NDArray[np.float64]:
        r"""Pseudo-inverse $\varphi^{-1}(s)$: $[0, \infty] \to [0, 1]$."""
        raise NotImplementedError

    def log_phi_deriv_abs(self, t: ArrayLike, k: int = 1) -> NDArray[np.float64]:
        r"""$\log |\varphi^{(k)}(t)|$. Sign is $(-1)^k$."""
        raise NotImplementedError

    def log_phi_inv_deriv_abs(self, s: ArrayLike, k: int = 1) -> NDArray[np.float64]:
        r"""$\log |(\varphi^{-1})^{(k)}(s)|$. Sign is $(-1)^k$."""
        raise NotImplementedError

    def kendalls_tau(self) -> float:
        raise NotImplementedError

    def tail_dependence(self) -> tuple[float, float]:
        r"""Lower and upper tail dependence coefficients
        $(\lambda_L, \lambda_U)$."""
        raise NotImplementedError

    def _sample_frailty(
        self, size: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        raise NotImplementedError

    def cdf(self, u: ArrayLike) -> NDArray[np.float64]:
        r"""$C(u_1, \dots, u_d) = \varphi^{-1}(\sum_i \varphi(u_i))$.

        Parameters
        ----------
        u : array_like, shape ``(..., d)``
            The last axis indexes the $d$ components. Each value in
            $[0, 1]$.
        """
        u = np.asarray(u, dtype=np.float64)
        return self.phi_inv(self.phi(u).sum(axis=-1))

    def log_pdf(self, u: ArrayLike) -> NDArray[np.float64]:
        r"""$\log c(u_1, \dots, u_d)$ via

        $$\log c(u) = \log |(\varphi^{-1})^{(d)}(s)|
            + \sum_i \log |\varphi'(u_i)|, \qquad s = \sum_i \varphi(u_i).$$

        Both terms under the logs are positive — the sign cancellation
        from $(-1)^d$ on each piece is handled algebraically. Bivariate
        ($d = 2$) is supported uniformly; $d > 2$ depends on whether the
        subclass has implemented $\log |(\varphi^{-1})^{(d)}|$.
        """
        u = np.asarray(u, dtype=np.float64)
        d = u.shape[-1]
        s = self.phi(u).sum(axis=-1)
        return self.log_phi_inv_deriv_abs(s, k=d) + self.log_phi_deriv_abs(u, k=1).sum(
            axis=-1
        )

    def pdf(self, u: ArrayLike) -> NDArray[np.float64]:
        return np.exp(self.log_pdf(u))

    def sample(
        self,
        size: int,
        d: int = 2,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        r"""Marshall–Olkin frailty sampling.

        Draws $M$ from the frailty distribution whose Laplace transform is
        $\varphi^{-1}$, then sets

        $$U_i = \varphi^{-1}\!\left(-\frac{\log V_i}{M}\right),
            \qquad V_i \stackrel{\text{iid}}{\sim} U(0, 1).$$

        Returns an ``(size, d)`` array with each marginal $U_i \sim U(0, 1)$
        and the dependence determined by the copula.
        """
        if d < 2:
            raise ValueError(f"d must be >= 2, got {d}")
        if rng is None:
            rng = np.random.default_rng()
        M = self._sample_frailty(size, rng)  # (size,)
        V = rng.uniform(size=(size, d))
        return self.phi_inv(-np.log(V) / M[:, None])
