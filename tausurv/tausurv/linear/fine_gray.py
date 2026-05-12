r"""Fine-Gray subdistribution hazard model (Fine & Gray, 1999).

Models the **subdistribution hazard** for a single cause of interest $k$
in a competing-risks setting:

$$
\lambda_k(t \mid x) = \lambda_{k,0}(t) \, \exp(\beta^\top x).
$$

The subdistribution hazard differs from the cause-specific hazard in
how subjects who experienced a competing event are handled — they stay
in the risk set with IPCW-decaying weight rather than being removed.
Predictions are on the **cumulative incidence function** scale:

$$
F_k(t \mid x) = 1 - \exp\!\big(-\hat H_{k,0}(t) \, \exp(\hat\beta^\top x)\big).
$$

Fit one model per cause. Use cause-specific Cox via :class:`CoxPH` for
the alternative cause-specific approach; use this when the question is
about cumulative incidence directly (which is what most clinical
papers want).

The likelihood uses IPCW weights against the censoring distribution
estimated via reverse-KM. The implementation evaluates the partial
likelihood in $O(n)$ per call via prefix/suffix cumulative sums (rather
than the naive $O(n^2)$ over risk-set membership).

References
----------
Fine, J. P., Gray, R. J. (1999). A proportional hazards model for the
subdistribution of a competing risk. JASA, 94(446).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from tausurv.nonparametric import kaplan_meier
from tausurv.predictor import SurvivalPredictor
from tausurv.step import StepFunction


class FineGray(SurvivalPredictor):
    r"""Fine-Gray proportional subdistribution-hazards model.

    One model per cause of interest. Predictions are on the cumulative
    incidence function scale: :meth:`predict_cif` returns
    $\hat F_k(t \mid x)$ (one cause only — pass ``cause=1`` or omit it).

    The inherited :meth:`predict_survival_function` returns
    $1 - \hat F_k(t \mid x)$, the **subdistribution survival** —
    interpret as "no cause-$k$ event by $t$" rather than "no event of
    any kind." Likewise :meth:`predict_cumulative_hazard` returns the
    cumulative subdistribution hazard.

    Parameters
    ----------
    cause : int, default 1
        Cause of interest. Subjects with ``event_indicator == cause`` are
        treated as cases; subjects with ``event_indicator > 0`` but
        $\ne$ ``cause`` are treated as competing events (kept in the
        risk set with IPCW weight).
    max_iter : int, default 200
    tol : float, default 1e-7
        L-BFGS-B convergence tolerance on the gradient.

    Attributes
    ----------
    coef_ : (d,) array
        Estimated $\hat \beta$.
    baseline_subdist_cumhazard_ : StepFunction
        Right-continuous $\hat H_{k,0}$, baseline 0.
    times_ : (m,) array
        Unique cause-$k$ event times — the model's natural time grid.
    """

    coef_: NDArray[np.float64]
    baseline_subdist_cumhazard_: StepFunction

    def __init__(
        self,
        cause: int = 1,
        *,
        max_iter: int = 200,
        tol: float = 1e-7,
    ) -> None:
        if cause < 1:
            raise ValueError(f"cause must be >= 1, got {cause}")
        self.cause = cause
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        X: ArrayLike,
        event_time: ArrayLike,
        event_indicator: ArrayLike,
    ) -> "FineGray":
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(event_time, dtype=np.float64)
        delta = np.asarray(event_indicator, dtype=np.int8)
        n, d = X.shape
        k = self.cause

        if not (delta == k).any():
            raise ValueError(
                f"no observations with event_indicator == cause={k}; "
                f"nothing to fit"
            )

        # Sort by Y ascending so prefix/suffix cumsums correspond to
        # "before time τ" / "at or after time τ".
        order = np.argsort(Y, kind="stable")
        X_s = X[order]
        Y_s = Y[order]
        delta_s = delta[order]
        is_cause = delta_s == k
        is_competitor = (delta_s > 0) & (delta_s != k)

        # KM of the censoring distribution: censorings are the "events" in G.
        is_censored_for_G = (delta_s == 0).astype(np.int8)
        G_step = kaplan_meier(Y_s, is_censored_for_G)
        G_at_Y = G_step(Y_s, side="left")

        # For competitors only, weight contributes via 1/G(Y_j).
        # Non-competitors get weight contribution 0 from this term — they
        # either contribute via the at-risk part (weight 1) or aren't in
        # the risk set at all.
        eps = 1e-12
        inv_G = np.where(
            is_competitor & (G_at_Y > eps), 1.0 / np.maximum(G_at_Y, eps), 0.0
        )

        # Pre-locate cause events and their τ.
        cause_positions = np.flatnonzero(is_cause)
        tau_per_event = Y_s[cause_positions]
        # Index of the first sorted subject with Y >= τ — same as cause_positions
        # for non-tied times. Use searchsorted for correctness under ties.
        start_per_event = np.searchsorted(Y_s, tau_per_event, side="left")

        def _S0_S1(beta: NDArray[np.float64]) -> tuple[
            NDArray[np.float64], NDArray[np.float64]
        ]:
            r"""Evaluate $S_0(\tau_i, \beta)$ and $S_1(\tau_i, \beta)$
            at every cause-$k$ event time in one pass."""
            lp = X_s @ beta
            exp_lp = np.exp(lp)
            X_exp = X_s * exp_lp[:, None]

            # At-risk part: subjects with Y_j >= τ, weight 1.
            # Suffix sums from position p give Σ_{q >= p} exp(β^T x_q).
            at_risk_S0_suffix = np.empty(n + 1)
            at_risk_S0_suffix[:-1] = np.cumsum(exp_lp[::-1])[::-1]
            at_risk_S0_suffix[-1] = 0.0
            at_risk_S1_suffix = np.empty((n + 1, d))
            at_risk_S1_suffix[:-1] = np.cumsum(X_exp[::-1], axis=0)[::-1]
            at_risk_S1_suffix[-1] = 0.0

            # Competitor part: subjects with Y_j < τ and competitor; weight
            # at τ is G(τ) / G(Y_j) = G(τ) * inv_G_j.
            comp_S0_prefix = np.concatenate([[0.0], np.cumsum(exp_lp * inv_G)])
            comp_S1_prefix = np.vstack(
                [np.zeros((1, d)), np.cumsum(X_exp * inv_G[:, None], axis=0)]
            )

            # G(τ) per cause-event time.
            G_at_tau = G_step(tau_per_event, side="left")

            S0 = (
                at_risk_S0_suffix[start_per_event]
                + G_at_tau * comp_S0_prefix[start_per_event]
            )
            S1 = (
                at_risk_S1_suffix[start_per_event]
                + G_at_tau[:, None] * comp_S1_prefix[start_per_event]
            )
            return S0, S1

        def neg_log_lik_and_grad(beta: NDArray[np.float64]) -> tuple[
            float, NDArray[np.float64]
        ]:
            S0, S1 = _S0_S1(beta)
            lp_events = X_s[cause_positions] @ beta
            log_lik = float(np.sum(lp_events - np.log(S0)))
            grad = -np.sum(X_s[cause_positions] - S1 / S0[:, None], axis=0)
            return -log_lik, grad

        result = minimize(
            fun=neg_log_lik_and_grad,
            x0=np.zeros(d),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter, "gtol": self.tol},
        )
        self.coef_ = result.x

        # Baseline subdistribution cumulative hazard (Breslow-style):
        # H_{k,0}(t) = Σ_{i: δ_i=k, Y_i ≤ t} 1 / S_0(Y_i; β̂)
        S0_hat, _ = _S0_S1(self.coef_)
        # Aggregate over tied event times.
        cumulative = np.cumsum(1.0 / S0_hat)
        unique_t, _ = np.unique(tau_per_event, return_index=True)
        last_idx = np.searchsorted(tau_per_event, unique_t, side="right") - 1
        h_at_unique = cumulative[last_idx]

        self.baseline_subdist_cumhazard_ = StepFunction(
            time=unique_t,
            value=h_at_unique,
            side="right",
            baseline=0.0,
        )
        self.times_ = unique_t
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        r"""Linear subdistribution risk $\beta^\top x$ (Cox-style scalar)."""
        return np.asarray(X, dtype=np.float64) @ self.coef_

    def _survival_function(
        self,
        X: ArrayLike,
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""$1 - F_k(t \mid x) = \exp(-\hat H_{k,0}(t) \, e^{\hat\beta^\top x})$
        — the **subdistribution survival**, *not* the marginal survival
        of $T$."""
        X = np.asarray(X, dtype=np.float64)
        exp_lp = np.exp(X @ self.coef_)
        H = self.baseline_subdist_cumhazard_(times)
        return np.exp(-H[None, :] * exp_lp[:, None])
