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

:class:`CauseSpecificPredictor` composes per-cause single-event models into
a competing-risks predictor via the standard cause-specific-hazard CIF
(Prentice et al., 1978; https://pubmed.ncbi.nlm.nih.gov/373811/).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

# How far predicted cumulative incidences may leave [0, 1], or their sum
# exceed 1, before it counts as a model bug rather than rounding.
_CIF_TOLERANCE = 1e-9


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
    overriding. The CIFs are checked on every prediction: a model whose
    $\hat F_k$ leave $[0, 1]$ or sum to more than 1 beyond rounding raises
    instead of being corrected silently.
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
        cif = self._checked_cif(X, self._resolve_times(times))
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
        cif = self._checked_cif(X, times)
        return 1.0 - cif.sum(axis=1)

    def _checked_cif(
        self,
        X: ArrayLike,
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """:meth:`_cif`, verified to be cumulative incidences.

        Violations within ``_CIF_TOLERANCE`` are rounding and are removed:
        negatives are clipped, and a subject whose marginal exceeds 1 has its
        curves scaled down as a whole, which keeps them monotone. Anything
        larger raises. Survival and CIF predictions both go through here, so
        ``S = 1 - sum_k F_k`` holds exactly.
        """
        cif = np.asarray(self._cif(X, times), dtype=np.float64)
        marginal = cif.sum(axis=1).max(axis=-1, keepdims=True)[..., None]
        violation = max(
            float(marginal.max(initial=0.0)) - 1.0, -float(cif.min(initial=0.0))
        )
        if violation > _CIF_TOLERANCE:
            raise ValueError(
                f"{type(self).__name__} predicted cumulative incidences outside "
                f"[0, 1] or summing to more than 1 (off by {violation:.2e}); "
                f"this is a bug in the model, not in the input"
            )
        bounded = np.clip(cif, 0.0, None) / np.maximum(marginal, 1.0)
        return np.asarray(bounded, dtype=np.float64)

    def _cif(
        self,
        X: ArrayLike,
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _cif(X, times)."
        )


class CauseSpecificPredictor(CompetingRisksPredictor):
    r"""Composite competing-risks predictor from per-cause hazard models.

    Fits one single-event :class:`SurvivalPredictor` per competing cause
    $k = 1, \dots, K$, treating events from other causes as censoring. The
    per-cause cumulative hazards $\hat\Lambda_k(t \mid x)$ give the overall
    discrete survival

    $$
    \hat S(t \mid x) = \prod_{t_i \le t}\bigl(1 - \Delta\hat\Lambda(t_i)\bigr),
    \qquad \Delta\hat\Lambda = \sum_k \Delta\hat\Lambda_k,
    $$

    and the cause-$k$ cumulative incidence function

    $$
    \hat F_k(t \mid x) = \sum_{t_i \le t}
        \hat S(t_i^- \mid x)\, \Delta\hat\Lambda_k(t_i).
    $$

    Because $\hat S(t_i^-)\, \Delta\hat\Lambda(t_i) = \hat S(t_i^-) - \hat S(t_i)$
    holds for this discrete product, $\sum_k \hat F_k = 1 - \hat S$ exactly;
    the marginal survival returned by :meth:`predict_survival_function`
    is $1 - \sum_k \hat F_k$, consistent with it.

    This is the cause-specific-hazard alternative to the subdistribution
    approach of :class:`~tausurv.linear.FineGray`; compare Fine & Gray (1999) and
    cause-specific Cox (Prentice et al., 1978).

    The class follows the sklearn-style composition pattern: construction
    takes a *factory* (any zero-argument callable returning a fresh
    :class:`SurvivalPredictor`) plus ``n_causes``; :meth:`fit` clones and
    fits one model per cause, so each instance receives its own independent
    fit. Any :class:`SurvivalPredictor` subclass works — :class:`~tausurv.linear.CoxPH`,
    :class:`~tausurv.linear.AFT`, :class:`~tausurv.nn.DeepSurv`,
    :class:`~tausurv.trees.RandomSurvivalForest`, etc.

    Parameters
    ----------
    model_factory : callable
        Zero-argument callable returning a fresh, unfitted
        :class:`SurvivalPredictor`. A bare class (``CoxPH``) works; use
        ``functools.partial`` or a lambda for constructor kwargs.
    n_causes : int
        Number of competing causes ($K$). Must be $\ge 1$.

    Attributes
    ----------
    models_ : list of :class:`SurvivalPredictor`
        Fitted per-cause models, indexed ``[0]`` through ``[n_causes - 1]``.
    times_ : (m,) array
        Sorted unique event times across all causes — the grid on which the
        CIFs are computed.

    Notes
    -----
    :attr:`n_causes` and :attr:`times_` are set at :meth:`fit` time, following
    the package's fitted-state convention. No architecturally interesting
    configuration lives here beyond the factory and the cause count; there is
    no ``<Model>Config`` dataclass (mirrors the composition predictors in
    :mod:`tausurv.model_selection`, not the nn architectures that carry
    large nested config objects).
    """

    def __init__(
        self,
        model_factory: Callable[[], SurvivalPredictor],
        n_causes: int,
    ) -> None:
        if n_causes < 1:
            raise ValueError(f"n_causes must be >= 1, got {n_causes}")
        self.model_factory = model_factory
        self.n_causes = n_causes

    def fit(
        self,
        X: ArrayLike,
        event_time: ArrayLike,
        event_indicator: ArrayLike,
    ) -> "CauseSpecificPredictor":
        r"""Fit one wrapped model per competing cause.

        For cause $k$, the event indicator is collapsed to
        $\delta_k = \mathbb{1}[\varepsilon = k]$ (0 for censored *and* for
        competing events). Each wrapped model is a freshly constructed
        instance from ``model_factory``.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(event_time, dtype=np.float64)
        eps = np.asarray(event_indicator)

        unique_causes = set(np.unique(eps).tolist())
        if not unique_causes <= set(range(self.n_causes + 1)):
            raise ValueError(
                f"event_indicator must be in [0, {self.n_causes}], "
                f"got values {sorted(unique_causes)}"
            )

        self.models_: list[SurvivalPredictor] = []
        for k in range(1, self.n_causes + 1):
            delta_k = (eps == k).astype(np.int8)
            model_k = self.model_factory()
            model_k.fit(X, Y, delta_k)
            self.models_.append(model_k)

        event_times = Y[eps > 0]
        self.times_ = np.unique(event_times) if event_times.size else np.unique(Y)
        self.times_ = self.times_.astype(np.float64)
        return self

    def _cif(
        self,
        X: ArrayLike,
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""Cause-specific CIFs on ``times``.

        Returns ``(n, n_causes, len(times))``.
        """
        # X: (n, d), times: (T,)
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]

        # Truncate the internal grid to the prefix needed for the requested
        # times: the cumulative product / sum must run through every event
        # up to max(times), but nothing beyond it.
        grid = self.times_
        last_needed = float(times.max())
        cut = int(np.searchsorted(grid, last_needed, side="right"))
        grid = grid[: cut + 1]
        m = grid.shape[0]

        # Per-cause survival on the (truncated) internal grid.
        # S_k_all: (n, n_causes, m)
        S_k_all = np.empty((n, self.n_causes, m), dtype=np.float64)
        for k, model_k in enumerate(self.models_):
            S_k_all[:, k, :] = model_k.predict_survival_function(X, grid)

        # Cause-specific cumulative hazards and their increments.
        # dLam_k: (n, n_causes, m), with zero in the first column.
        Lam_k = -np.log(np.clip(S_k_all, 1e-12, 1.0))
        dLam_k = np.diff(Lam_k, axis=-1, prepend=0.0)

        # Overall hazard increment; clip at 1 (rare late-time jumps from
        # small risk sets) and rescale per-cause increments to match so
        # sum_k dLam_k equals the clipped total.
        dLam_raw = dLam_k.sum(axis=1)  # (n, m)
        scale = np.divide(
            1.0, dLam_raw, out=np.ones_like(dLam_raw), where=(dLam_raw > 1.0)
        )  # (n, m)
        dLam_k = dLam_k * scale[:, None, :]
        dLam = np.clip(dLam_raw, 0.0, 1.0)  # (n, m)

        # S(t_i) = prod_{j<=i} (1 - dLam(t_j)),  so S(t_i^-) * dLam(t_i) = S(t_i^-) - S(t_i).
        S = np.cumprod(1.0 - dLam, axis=-1)  # (n, m)

        # S(t_i^-) = [1, S(t_0), ..., S(t_{m-2})]  -> (n, m)
        S_before = np.empty((n, m), dtype=np.float64)
        S_before[:, 0] = 1.0
        S_before[:, 1:] = S[:, :-1]

        # Standard cause-specific CIF:
        # F_k(t|x) = sum_{t_i <= t} S(t_i^-) * dLam_k(t_i)
        # cif_full: (n, n_causes, m)
        cif_full = np.cumsum(S_before[:, None, :] * dLam_k, axis=-1)

        # Evaluate at requested times.
        idx = np.searchsorted(grid, times, side="right") - 1
        below = idx < 0
        idx_clipped = np.clip(idx, 0, m - 1)

        # cif_at: (n, n_causes, T)
        cif_at = cif_full[..., idx_clipped]
        cif_at[..., below] = 0.0
        return cif_at
