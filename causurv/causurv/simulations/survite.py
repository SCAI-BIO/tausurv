r"""SurvITE benchmark simulation (Curth et al., 2021).

Reproduces the data-generating process from

    Curth, A., Lee, C., & van der Schaar, M. (2021).
    *SurvITE: Learning Heterogeneous Treatment Effects from
    Time-to-Event Data.* NeurIPS 34.

The hazard is discrete-time on the integer grid $\{1, \dots, t_{\max}\}$.
The full setting (``"S4"``) combines covariate-dependent treatment
assignment with informative censoring; ``"S1"``..``"S3"`` are ablations
that isolate one source of difficulty at a time.

The generator is bundled with a closed-form **oracle** for the survival
function and CATE — needed to benchmark a learner's HTE against ground
truth on the same simulation parameters.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from causurv.predictor import HTEEstimates, HTEPredictor

Scenario = Literal["S1", "S2", "S3", "S4"]
_SCENARIOS: dict[str, tuple[bool, bool]] = {
    "S1": (False, False),
    "S2": (False, True),
    "S3": (True, False),
    "S4": (True, True),
}


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-z))


class SurvITE(HTEPredictor):
    r"""SurvITE single-event benchmark simulator with oracle HTE.

    Parameters
    ----------
    scenario : ``{"S1", "S2", "S3", "S4"}``, default ``"S4"``
        Selects which sources of difficulty are active:

        ===========  =========  ==========
        scenario     treatment  censoring
        ===========  =========  ==========
        ``"S1"``     no         no
        ``"S2"``     no         informative
        ``"S3"``     yes        no
        ``"S4"``     yes        informative
        ===========  =========  ==========

        Under no-treatment scenarios every subject is in arm $0$.
    n_features : int, default ``10``
        Dimension of $X$. Must be $\ge 4$ — the hazard reads
        ``X[:, 0..3]``.
    eta : float, default ``1.0``
        Treatment-assignment strength.
        $\pi(x) = \eta \cdot \sigma\big(\sum_{j \in P} x_j\big)$, where
        $\sigma$ is the logistic sigmoid and $P$ is selected by
        ``overlap_treat``.
    t_max : int, default ``30``
        Administrative censoring horizon; time is discrete on
        $\{1, \dots, t_{\max}\}$.
    rho : float, default ``0.2``
        Off-diagonal of the equicorrelated covariate covariance
        $\Sigma = (1 - \rho) I + \rho \mathbf{1}\mathbf{1}^\top$.
    overlap_treat : bool, default ``True``
        If ``True`` the treatment-assignment features are ``[0, 1]``
        (overlap with the hazard, mimicking confounding); if ``False``
        they are ``[n_features - 2, n_features - 1]`` (no overlap with
        the hazard, easier setting).

    Attributes
    ----------
    times_ : ``(t_max,)`` array
        Natural integer time grid $\{1, \dots, t_{\max}\}$.

    Examples
    --------
    Generate the full S4 setting and benchmark a T-learner against the
    oracle CATE::

        from causurv.simulations import SurvITE
        from causurv.learners import TLearner
        from tausurv.linear import CoxPH

        sim = SurvITE(scenario="S4")
        X, T, E, A = sim.generate(n=2000, seed=0)

        tl = TLearner(lambda: CoxPH()).fit(X, T, E, A)
        hte_hat  = tl.predict_hte(X, times=[5, 10, 20])
        hte_true = sim.predict_hte(X, times=[5, 10, 20])
        mse = np.mean((hte_hat.values - hte_true.values) ** 2)
    """

    def __init__(
        self,
        scenario: Scenario = "S4",
        *,
        n_features: int = 10,
        eta: float = 1.0,
        t_max: int = 30,
        rho: float = 0.2,
        overlap_treat: bool = True,
    ) -> None:
        if scenario not in _SCENARIOS:
            raise ValueError(
                f"scenario must be one of {sorted(_SCENARIOS)}; got {scenario!r}"
            )
        if n_features < 4:
            raise ValueError(
                f"n_features must be >= 4 (hazard reads X[:, 0..3]); got {n_features}"
            )
        if t_max < 2:
            raise ValueError(f"t_max must be >= 2; got {t_max}")
        if not (0.0 <= rho < 1.0):
            raise ValueError(f"rho must be in [0, 1); got {rho}")

        self.scenario = scenario
        self.treated, self.censored = _SCENARIOS[scenario]
        self.n_features = n_features
        self.eta = float(eta)
        self.t_max = int(t_max)
        self.rho = float(rho)
        self.overlap_treat = overlap_treat

        if overlap_treat:
            self._P = (0, 1)
        else:
            self._P = (n_features - 2, n_features - 1)

        self.times_ = np.arange(1, self.t_max + 1, dtype=np.float64)
        self._last_X: NDArray[np.float64] | None = None

    def generate(
        self, n: int, *, seed: int | None = None
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.int8],
        NDArray[np.int8],
    ]:
        r"""Sample $n$ subjects from the configured scenario.

        Returns
        -------
        X : ``(n, n_features)`` array
            Covariates, $X_i \sim \mathcal{N}(0, \Sigma)$.
        event_time : ``(n,)`` array
            $T_i = \min(T_i^{\text{event}}, T_i^{\text{cens}})$, integer-valued.
        event_indicator : ``(n,)`` int8 array
            $1$ if the event was observed before censoring, else $0$.
        treatment : ``(n,)`` int8 array
            $A_i \in \{0, 1\}$; identically $0$ under ``"S1"``/``"S2"``.
        """
        rng = np.random.default_rng(seed)
        d = self.n_features
        Sigma = (1.0 - self.rho) * np.eye(d) + self.rho * np.ones((d, d))
        X = rng.multivariate_normal(np.zeros(d), Sigma, size=n)

        if self.treated:
            p = self.eta * _sigmoid(np.sum(X[:, self._P], axis=1))
            p = np.clip(p, 0.0, 1.0)
            A = (rng.uniform(size=n) < p).astype(np.int8)
        else:
            A = np.zeros(n, dtype=np.int8)

        event_h = self._hazard_grid(X, A, event=True)
        hit = rng.uniform(size=event_h.shape) < event_h
        event_occurs = hit.any(axis=1)
        event_idx = hit.argmax(axis=1)
        T_event = np.where(event_occurs, event_idx + 1, self.t_max).astype(
            np.float64
        )

        if self.censored:
            cens_h = self._hazard_grid(X, A, event=False)
            hit_c = rng.uniform(size=cens_h.shape) < cens_h
            cens_idx = hit_c.argmax(axis=1)
            # Last column of cens_h is 1 (administrative), so hit_c is
            # guaranteed to have at least one True per row.
            T_cens = (cens_idx + 1).astype(np.float64)
            T = np.minimum(T_event, T_cens)
            E = (T_event <= T_cens).astype(np.int8)
            E[T >= self.t_max] = 0
        else:
            T = T_event
            E = event_occurs.astype(np.int8)

        self._last_X = X
        return X, T, E, A

    def predict_potential_outcomes(
        self,
        X: ArrayLike,
        times: ArrayLike | None = None,
        *,
        cause: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""Oracle potential-outcome survival functions.

        Returns ``(S_0, S_1)``, each shaped ``(n, T)`` — the true
        $S_a(t \mid x)$ derived from the discrete-time hazard. ``times``
        may contain any non-negative values; survival is step-evaluated
        on the integer grid.
        """
        if cause is not None:
            raise ValueError(
                "SurvITE is single-event; pass cause=None (got "
                f"cause={cause})"
            )
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n, d); got shape {X.shape}")
        times_arr = self._resolve_times(times)

        out = []
        for a in (0, 1):
            haz = self._hazard_grid(X, a, event=True)
            # Pad column 0 with S(t)=1 for t < 1, then cumprod of (1-h).
            S_full = np.empty((X.shape[0], self.t_max + 1), dtype=np.float64)
            S_full[:, 0] = 1.0
            S_full[:, 1:] = np.cumprod(1.0 - haz, axis=1)
            idx = np.clip(
                np.floor(times_arr).astype(int), 0, self.t_max
            )
            out.append(S_full[:, idx])
        return out[0], out[1]

    def hazard(
        self, X: ArrayLike, treatment: ArrayLike | int = 0, *, event: bool = True
    ) -> NDArray[np.float64]:
        r"""Discrete-time hazard $h(\tau \mid x, a)$ on the natural grid.

        Returns ``(n, t_max)``. With ``event=False`` returns the
        censoring hazard; the last column is $1$ (administrative
        censoring at $t_{\max}$).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n, d); got shape {X.shape}")
        if np.isscalar(treatment):
            A = np.full(X.shape[0], int(treatment), dtype=np.int8)
        else:
            A = np.asarray(treatment, dtype=np.int8)
            if A.shape != (X.shape[0],):
                raise ValueError(
                    f"treatment must broadcast to ({X.shape[0]},); got {A.shape}"
                )
        return self._hazard_grid(X, A, event=event)

    def _hazard_grid(
        self,
        X: NDArray[np.float64],
        A: NDArray[np.int8] | int,
        *,
        event: bool,
    ) -> NDArray[np.float64]:
        n = X.shape[0]
        out = np.zeros((n, self.t_max), dtype=np.float64)
        for tau in range(1, self.t_max + 1):
            if event:
                out[:, tau - 1] = self._event_hazard(tau, X, A)
            else:
                out[:, tau - 1] = self._censoring_hazard(tau, X)
        return out

    def _event_hazard(
        self,
        tau: int,
        X: NDArray[np.float64],
        A: NDArray[np.int8] | int,
    ) -> NDArray[np.float64]:
        treat_shift = np.asarray(A, dtype=np.float64) * (
            (X[:, 2] >= 0).astype(np.float64) + 0.5
        )
        if tau <= 10:
            z = -5.0 * X[:, 0] ** 2 - treat_shift
        else:
            z = 10.0 * X[:, 1] - treat_shift
        return 0.1 * _sigmoid(z)

    def _censoring_hazard(
        self, tau: int, X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if tau >= self.t_max:
            return np.ones(X.shape[0], dtype=np.float64)
        return 0.01 * _sigmoid(10.0 * X[:, 3] ** 2)

    def _stored_X(self) -> NDArray[np.float64]:
        # Override: the oracle has no fit(), so cache the last-generated X
        # instead of the fitted X used by the HTEPredictor default.
        if self._last_X is None:
            raise RuntimeError(
                "SurvITE.predict_ate without `X` requires a prior "
                "generate() call; either pass `X` or call generate first."
            )
        return self._last_X

    def fit(self, *args, **kwargs):  # noqa: D401, ARG002
        """No-op: SurvITE is parameterized at construction.

        Present only to satisfy the :class:`HTEPredictor` interface;
        the oracle has no learnable parameters.
        """
        raise NotImplementedError(
            "SurvITE is a simulator/oracle, not a learner — it has nothing "
            "to fit. Call generate() to sample data and use predict_* for "
            "the oracle."
        )

    def predict_hte(
        self,
        X: ArrayLike,
        *,
        estimand,
        **kwargs,
    ) -> HTEEstimates:
        from causurv.estimands import resolve as _resolve_estimand

        est = _resolve_estimand(estimand, **kwargs)
        if (est.treatment, est.reference) not in {(1, 0), (0, 1)}:
            raise ValueError(
                "SurvITE is binary-treatment; treatment/reference must be "
                f"a permutation of (0, 1); got ({est.treatment}, {est.reference})"
            )
        return self._predict_hte_impl(X, est)
