r"""Unified survival prediction API.

Two mixins ``SurvivalPredictor`` and ``CompetingRisksPredictor`` define the
contract that every fitted survival model in tausurv exposes:

- :meth:`predict_survival_function` — $\hat S(t \mid x)$ on a time grid.
- :meth:`predict_cumulative_hazard` — $\hat \Lambda(t \mid x) = -\log \hat S$.
- :meth:`predict_cif` — $1 - \hat S$ (single-event) or
  $\hat F_k(t \mid x)$ (competing risks).
- :meth:`predict_rmst` — restricted mean survival time up to a horizon.
- :meth:`predict_risk_at` — $1 - \hat S(t \mid x)$ at a single time.
- :meth:`predict` — Cox-style scalar ranking, larger = higher risk.

Each model implements one hook — ``_survival_function`` (single-event) or
``_cif`` (competing risks) — and inherits the rest. NumPy arrays in, NumPy
arrays out; neural models do the torch ↔ numpy conversion internally.

The :attr:`times_` attribute is the model's natural time grid (fitted event
times for KM/Cox/RSF, bin upper-bounds for discrete-time NN models, Breslow
support for DeepSurv after :meth:`fit_baseline`). Predict methods default
to it when ``times=None``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class SurvivalPredictor:
    r"""Mixin: unified survival prediction API for single-event models.

    Subclasses must:

    - implement :meth:`_survival_function(X, times) -> ndarray (n, len(times))`
      returning $\hat S(t \mid x) \in [0, 1]$.
    - set ``self.times_`` (a sorted 1-D ``np.ndarray``) at fit-time or
      construction-time so callers can default ``times=None``.

    All other prediction methods are derived from
    :meth:`predict_survival_function` and do not need overriding.
    """

    times_: NDArray[np.float64]

    def predict_survival_function(
        self,
        X: ArrayLike,
        times: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        r"""$\hat S(t \mid x)$ on ``times`` (defaults to :attr:`times_`).

        Returns an ``(n, len(times))`` array with values in $[0, 1]$. The
        result is clipped to that interval to absorb float-32 softmax noise
        from neural models.
        """
        S = self._survival_function(X, self._resolve_times(times))
        return np.clip(S, 0.0, 1.0)

    def predict_cumulative_hazard(
        self,
        X: ArrayLike,
        times: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        r"""$\hat \Lambda(t \mid x) = -\log \hat S(t \mid x)$."""
        S = self.predict_survival_function(X, times)
        return -np.log(np.clip(S, 1e-12, 1.0))

    def predict_cif(
        self,
        X: ArrayLike,
        times: ArrayLike | None = None,
        *,
        cause: int | None = None,
    ) -> NDArray[np.float64]:
        r"""$1 - \hat S(t \mid x)$.

        ``cause`` is accepted for API uniformity with competing-risks models
        and must be ``None`` or ``1`` for a single-event predictor.
        """
        if cause is not None and cause != 1:
            raise ValueError(
                f"{type(self).__name__} is single-event; cause must be None or 1, "
                f"got {cause}"
            )
        return 1.0 - self.predict_survival_function(X, times)

    def predict_rmst(
        self,
        X: ArrayLike,
        horizon: float,
        *,
        times: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        r"""Restricted mean survival time up to ``horizon``.

        $$
        \text{RMST}(\tau \mid x) = \mathbb{E}[\min(T, \tau) \mid x]
            = \int_0^\tau \hat S(t \mid x)\, dt.
        $$

        Integrates the survival function with the trapezoidal rule on a grid
        anchored at ``0`` and ``horizon``. If ``times`` is given, it is
        merged into that grid (clipped to ``[0, horizon]``); otherwise
        :attr:`times_` is used.
        """
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")
        grid = self._resolve_times(times)
        grid = grid[(grid > 0) & (grid <= horizon)]
        grid = np.unique(np.concatenate([[0.0], grid, [float(horizon)]]))
        S = self.predict_survival_function(X, grid)
        return np.trapezoid(S, x=grid, axis=1)

    def predict_risk_at(self, X: ArrayLike, time: float) -> NDArray[np.float64]:
        r"""$1 - \hat S(\text{time} \mid x)$ — risk by ``time``.

        Useful as a ranking score for time-dependent metrics like AUC(t)."""
        S = self.predict_survival_function(X, np.atleast_1d(float(time)))
        return 1.0 - S[:, 0]

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        r"""Cox-style scalar ranking; larger value = higher risk.

        Default: $1 - \hat S(t^* \mid x)$ where $t^*$ is the median of
        :attr:`times_`. Subclasses with a natural risk score (CoxPH's
        linear predictor, DeepSurv's log-risk) should override.
        """
        if not hasattr(self, "times_"):
            raise NotImplementedError(
                f"{type(self).__name__}.predict requires either a fitted "
                f"times_ attribute or an override."
            )
        return self.predict_risk_at(X, float(np.median(self.times_)))

    def _survival_function(
        self,
        X: ArrayLike,
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _survival_function(X, times)."
        )

    def _resolve_times(self, times: ArrayLike | None) -> NDArray[np.float64]:
        if times is not None:
            return np.asarray(times, dtype=np.float64)
        if not hasattr(self, "times_"):
            raise RuntimeError(
                f"{type(self).__name__} has no times_; pass `times` explicitly "
                f"or call fit() / fit_baseline() first."
            )
        return np.asarray(self.times_, dtype=np.float64)


class CompetingRisksPredictor(SurvivalPredictor):
    r"""Mixin: unified prediction API for competing-risks models.

    Subclasses must:

    - implement :meth:`_cif(X, times) -> ndarray (n, n_causes, len(times))`
      returning $\hat F_k(t \mid x)$ for each cause $k = 1, \dots, K$.
    - set ``self.n_causes`` and ``self.times_``.

    The marginal survival function $\hat S(t \mid x) = 1 - \sum_k \hat F_k$
    is derived from the CIFs, so ``_survival_function`` does not need
    overriding.
    """

    n_causes: int

    def predict_cif(
        self,
        X: ArrayLike,
        times: ArrayLike | None = None,
        *,
        cause: int | None = None,
    ) -> NDArray[np.float64]:
        r"""$\hat F_k(t \mid x)$ — cause-specific cumulative incidence.

        Returns ``(n, n_causes, len(times))`` if ``cause`` is ``None``, or
        ``(n, len(times))`` for the chosen cause. Causes are 1-indexed.
        """
        cif = self._cif(X, self._resolve_times(times))
        cif = np.clip(cif, 0.0, None)
        # Guarantee the marginal CDF ``sum_k F_k(t | x)`` is bounded by 1 by
        # rescaling float-32 softmax noise (typically ~1e-7 above 1).
        marginal = cif.sum(axis=1, keepdims=True)
        scale = np.minimum(1.0, 1.0 / np.maximum(marginal, 1e-12))
        cif = cif * scale
        if cause is None:
            return cif
        if not (1 <= cause <= self.n_causes):
            raise ValueError(f"cause must be in [1, {self.n_causes}], got {cause}")
        return cif[:, cause - 1, :]

    def _survival_function(
        self,
        X: ArrayLike,
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        cif = self._cif(X, times)
        return 1.0 - cif.sum(axis=1)

    def _cif(
        self,
        X: ArrayLike,
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _cif(X, times)."
        )
