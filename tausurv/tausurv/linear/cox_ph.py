from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from tausurv.linear._checkpoint import _check_model_name
from tausurv.predictor import SurvivalPredictor
from tausurv.step import StepFunction


class CoxPH(SurvivalPredictor):
    r"""Cox proportional-hazards model with Breslow tie handling.

    Fits $\lambda(t \mid x) = \lambda_0(t) \exp(\beta^\top x)$ by maximizing
    the partial log-likelihood

    $$
    \ell(\beta) = \sum_{i: \delta_i = 1}
        \left[\beta^\top x_i - \log \sum_{j \in R(Y_i)} \exp(\beta^\top x_j)\right]
    $$

    with L-BFGS-B, then estimates the baseline cumulative hazard
    $\hat \Lambda_0(t)$ by Breslow. Inherits the unified survival prediction
    API from :class:`SurvivalPredictor`.

    Parameters
    ----------
    max_iter : int, default 100
    tol : float, default 1e-6
        Convergence tolerance on the gradient.

    Attributes
    ----------
    coef_ : (d,) array
        Estimated $\hat\beta$.
    baseline_cumulative_hazard_ : StepFunction
        Right-continuous $\hat \Lambda_0$; baseline 0.
    times_ : (k,) array
        Unique observed event times — the model's natural time grid.

    References
    ----------
    Cox, D. R. (1972). Regression models and life-tables. JRSS-B, 34(2).
    Breslow, N. (1974). Covariance analysis of censored survival data.
    Biometrics, 30(1).
    """

    coef_: NDArray[np.float64]
    baseline_cumulative_hazard_: StepFunction

    def __init__(self, *, max_iter: int = 100, tol: float = 1e-6) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        X: ArrayLike,
        event_time: ArrayLike,
        event_indicator: ArrayLike,
    ) -> "CoxPH":
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(event_time, dtype=np.float64)
        delta = np.asarray(event_indicator, dtype=np.int8)

        unique_deltas = np.unique(delta).tolist()
        if not set(unique_deltas) <= {0, 1}:
            raise ValueError(
                "CoxPH supports single-event data; event_indicator must be 0/1, "
                f"got values {unique_deltas}"
            )

        _, d = X.shape

        # Sort by event_time descending (stable). In descending order, the risk
        # set R(Y_i) is exactly the prefix up to the end of Y_i's tie block.
        order_desc = np.argsort(-Y, kind="stable")
        X_s = X[order_desc]
        Y_s = Y[order_desc]
        delta_s = delta[order_desc].astype(np.float64)

        # End of each subject's tie block in descending order.
        neg_Y_s = -Y_s
        end_tie = np.searchsorted(neg_Y_s, neg_Y_s, side="right") - 1

        def neg_log_lik_and_grad(
            beta: NDArray[np.float64],
        ) -> tuple[float, NDArray[np.float64]]:
            lp_s = X_s @ beta
            exp_lp_s = np.exp(lp_s)
            S0 = np.cumsum(exp_lp_s)
            S1 = np.cumsum(X_s * exp_lp_s[:, None], axis=0)
            S0_at_event = S0[end_tie]
            S1_at_event = S1[end_tie]
            log_lik = float(np.sum(delta_s * (lp_s - np.log(S0_at_event))))
            grad = -np.sum(
                delta_s[:, None] * (X_s - S1_at_event / S0_at_event[:, None]),
                axis=0,
            )
            return -log_lik, grad

        result = minimize(
            fun=neg_log_lik_and_grad,
            x0=np.zeros(d),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "gtol": self.tol},
        )
        self.coef_ = result.x
        self.baseline_cumulative_hazard_ = self._breslow_baseline(X, Y, delta)
        self.times_ = np.asarray(
            self.baseline_cumulative_hazard_.time, dtype=np.float64
        )
        return self

    def _breslow_baseline(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        delta: NDArray[np.int8],
    ) -> StepFunction:
        exp_lp = np.exp(X @ self.coef_)
        order_asc = np.argsort(Y, kind="stable")
        Y_s = Y[order_asc]
        delta_s = delta[order_asc]
        exp_lp_s = exp_lp[order_asc]

        # Sum of exp_lp over the risk set at each position (ascending order).
        S0_from_right = np.cumsum(exp_lp_s[::-1])[::-1]

        event_times = Y_s[delta_s == 1]
        unique_times, counts = np.unique(event_times, return_counts=True)
        first_pos = np.searchsorted(Y_s, unique_times, side="left")
        S0_at_unique = S0_from_right[first_pos]

        cum_hazard = np.cumsum(counts / S0_at_unique)
        return StepFunction(
            time=unique_times,
            value=cum_hazard,
            side="right",
            baseline=0.0,
        )

    def save(self, path: str | Path) -> None:
        """Write ``config.json`` and ``state.npz`` to ``path``."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("CoxPH is not fitted; call fit() before save()")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "model": type(self).__name__,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }
        (path / "config.json").write_text(json.dumps(config, indent=2))
        np.savez_compressed(
            path / "state.npz",
            coef=self.coef_,
            times=self.times_,
            baseline_time=self.baseline_cumulative_hazard_.time,
            baseline_value=self.baseline_cumulative_hazard_.value,
        )

    @classmethod
    def load(cls, path: str | Path) -> CoxPH:
        """Reconstruct a model saved by `save`."""
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        _check_model_name(config.pop("model"), cls, path)
        model = cls(**config)
        with np.load(path / "state.npz") as state:
            model.coef_ = state["coef"]
            model.times_ = state["times"]
            model.baseline_cumulative_hazard_ = StepFunction(
                time=state["baseline_time"],
                value=state["baseline_value"],
                side="right",
                baseline=0.0,
            )
        return model

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        r"""Linear predictor $\beta^\top x$ — Cox's natural risk score."""
        return np.asarray(X, dtype=np.float64) @ self.coef_

    def _survival_function(
        self,
        X: ArrayLike,
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""$\hat S(t \mid x) = \exp(-\hat\Lambda_0(t)\,e^{\hat\beta^\top x})$."""
        X = np.asarray(X, dtype=np.float64)
        exp_lp = np.exp(X @ self.coef_)
        H = self.baseline_cumulative_hazard_(times)
        return np.exp(-H[None, :] * exp_lp[:, None])
