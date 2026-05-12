r"""Survival-flavored distribution base class (numpy/scipy-backed).

All concrete distributions implement a small required set:

- :meth:`log_pdf` — log density, the canonical likelihood term for events.
- :meth:`log_survival` — log of $S(t) = 1 - F(t)$, the censored-contribution term.
- :meth:`quantile` — inverse CDF, enables :meth:`rvs` and :meth:`median`.
- :meth:`mean` — closed-form expectation (or ``np.nan`` if no first moment).

Everything else (:meth:`pdf`, :meth:`survival`, :meth:`cdf`, :meth:`hazard`,
:meth:`log_hazard`, :meth:`median`, :meth:`rvs`) is derived. Subclasses
override the derived methods only when they have a numerically better path.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class SurvivalDistribution:
    """Base class for parametric survival distributions."""

    def log_pdf(self, t: ArrayLike) -> NDArray[np.float64]:
        raise NotImplementedError

    def log_survival(self, t: ArrayLike) -> NDArray[np.float64]:
        raise NotImplementedError

    def quantile(self, p: ArrayLike) -> NDArray[np.float64]:
        raise NotImplementedError

    def mean(self) -> float:
        raise NotImplementedError

    def pdf(self, t: ArrayLike) -> NDArray[np.float64]:
        return np.exp(self.log_pdf(t))

    def survival(self, t: ArrayLike) -> NDArray[np.float64]:
        return np.exp(self.log_survival(t))

    def cdf(self, t: ArrayLike) -> NDArray[np.float64]:
        return -np.expm1(self.log_survival(t))

    def hazard(self, t: ArrayLike) -> NDArray[np.float64]:
        return np.exp(self.log_hazard(t))

    def log_hazard(self, t: ArrayLike) -> NDArray[np.float64]:
        return self.log_pdf(t) - self.log_survival(t)

    def median(self) -> float:
        return float(self.quantile(0.5))

    def rvs(
        self,
        size: int | tuple[int, ...],
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Inverse-CDF sampling: ``T = quantile(U)`` with ``U ~ Uniform(0, 1)``."""
        if rng is None:
            rng = np.random.default_rng()
        u = rng.uniform(size=size)
        return np.asarray(self.quantile(u), dtype=np.float64)
