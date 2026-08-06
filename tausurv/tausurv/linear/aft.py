r"""Accelerated Failure Time models.

Three parametric AFT models built on top of :mod:`tausurv.distributions`:
:class:`WeibullAFT`, :class:`LogNormalAFT`, :class:`LogLogisticAFT`. All
share the same regression structure

$$
\log T = \beta^\top x + \sigma \, \varepsilon
$$

with $\varepsilon$ drawn from a fixed parameterless distribution that gives
each variant its name (extreme-value → Weibull $T$, normal → log-normal $T$,
logistic → log-logistic $T$). Coefficients are fitted by maximum
likelihood (L-BFGS-B on $(\beta, \log \sigma)$), reusing the closed-form
``log_pdf`` and ``log_survival`` of the underlying distribution.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from tausurv.distributions import LogLogistic, LogNormal, SurvivalDistribution, Weibull
from tausurv.linear._checkpoint import _check_model_name
from tausurv.predictor import SurvivalPredictor


class AFT(SurvivalPredictor):
    r"""Base class for Accelerated Failure Time models.

    Subclasses set :attr:`distribution_class` and implement
    :meth:`_make_dist` — the rest of the fit/predict pipeline is shared.
    The likelihood at $(\beta, \sigma)$ is

    $$
    \ell(\beta, \sigma) = \sum_{i: \delta_i = 1} \log f(T_i; \mu_i, \sigma)
        + \sum_{i: \delta_i = 0} \log S(T_i; \mu_i, \sigma)
    $$

    with $\mu_i = \beta^\top x_i + b$ (intercept $b$ included by default).
    Optimization is performed over the unconstrained
    $(\beta, \log \sigma)$ to handle $\sigma > 0$ without bounds.

    Parameters
    ----------
    fit_intercept : bool, default True
        If True, prepend a column of 1s to ``X`` and learn a separate
        intercept :attr:`intercept_`.
    max_iter : int, default 200
    tol : float, default 1e-6
        Convergence tolerance on the gradient.

    Attributes
    ----------
    coef_ : (d,) array
        Estimated $\hat \beta$.
    intercept_ : float
        Estimated intercept; ``0.0`` if ``fit_intercept=False``.
    scale_ : float
        Estimated $\hat \sigma$.
    times_ : (k,) array
        Unique training event times — the model's natural time grid.
    """

    distribution_class: type[SurvivalDistribution]

    coef_: NDArray[np.float64]
    intercept_: float
    scale_: float

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        max_iter: int = 200,
        tol: float = 1e-6,
    ) -> None:
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        X: ArrayLike,
        event_time: ArrayLike,
        event_indicator: ArrayLike,
    ) -> "AFT":
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(event_time, dtype=np.float64)
        delta = np.asarray(event_indicator, dtype=np.int8)

        if not set(np.unique(delta).tolist()) <= {0, 1}:
            raise ValueError(
                f"{type(self).__name__} supports single-event data; "
                f"event_indicator must be 0/1"
            )
        if np.any(Y <= 0):
            raise ValueError("event_time must be strictly positive")

        n, d = X.shape
        if self.fit_intercept:
            X_design = np.column_stack([np.ones(n), X])
            n_beta = d + 1
        else:
            X_design = X
            n_beta = d
        is_event = delta == 1

        def neg_log_lik(params: NDArray[np.float64]) -> float:
            beta = params[:n_beta]
            sigma = float(np.exp(params[n_beta]))
            mu = X_design @ beta
            dist = self._make_dist(mu, sigma)
            log_pdf = dist.log_pdf(Y)
            log_surv = dist.log_survival(Y)
            ll = float(np.where(is_event, log_pdf, log_surv).sum())
            return -ll

        x0 = np.zeros(n_beta + 1)
        result = minimize(
            fun=neg_log_lik,
            x0=x0,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "gtol": self.tol},
        )

        params = result.x
        if self.fit_intercept:
            self.intercept_ = float(params[0])
            self.coef_ = params[1:n_beta].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = params[:n_beta].copy()
        self.scale_ = float(np.exp(params[n_beta]))
        self.times_ = np.unique(Y[is_event])
        return self

    def save(self, path: str | Path) -> None:
        """Write ``config.json`` and ``state.npz`` to ``path``."""
        if not hasattr(self, "coef_"):
            raise RuntimeError(
                f"{type(self).__name__} is not fitted; call fit() before save()"
            )
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "model": type(self).__name__,
            "fit_intercept": self.fit_intercept,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }
        (path / "config.json").write_text(json.dumps(config, indent=2))
        np.savez_compressed(
            path / "state.npz",
            coef=self.coef_,
            intercept=self.intercept_,
            scale=self.scale_,
            times=self.times_,
        )

    @classmethod
    def load(cls, path: str | Path) -> AFT:
        """Reconstruct a model saved by `save`; call on the concrete class."""
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        _check_model_name(config.pop("model"), cls, path)
        model = cls(**config)
        with np.load(path / "state.npz") as state:
            model.coef_ = state["coef"]
            model.intercept_ = float(state["intercept"])
            model.scale_ = float(state["scale"])
            model.times_ = state["times"]
        return model

    def _make_dist(
        self,
        mu: ArrayLike,
        sigma: float | NDArray[np.float64],
    ) -> SurvivalDistribution:
        r"""Map regression parameters $(\mu, \sigma)$ to the underlying
        distribution's native parameters. Override per subclass."""
        raise NotImplementedError

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        r"""Negated linear predictor $-\mu(x)$ — Cox-style ranking score.

        For AFT, larger $\mu = \beta^\top x + b$ means a *longer* survival
        time (covariates "decelerate" failure), so a risk score where
        "higher = more risk" requires negation. Plug into
        :func:`tausurv.metrics.concordance.harrell` directly.
        """
        X = np.asarray(X, dtype=np.float64)
        return -(X @ self.coef_ + self.intercept_)

    def _survival_function(
        self,
        X: ArrayLike,
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        X = np.asarray(X, dtype=np.float64)
        mu = X @ self.coef_ + self.intercept_  # (n,)
        # Broadcast: mu -> (n, 1), times -> (1, T) → distribution survival is (n, T).
        dist = self._make_dist(mu[:, None], self.scale_)
        return dist.survival(times[None, :])


class WeibullAFT(AFT):
    r"""Weibull AFT model.

    $$
    T \mid x \sim \mathrm{Weibull}\big(\text{shape}=1/\sigma,\;
        \text{scale}=\exp(\mu)\big), \qquad \mu = \beta^\top x + b.
    $$

    Equivalent to a Cox PH model with Weibull baseline hazard, but more
    interpretable: $e^{-\beta_j}$ is the multiplicative *time* effect of a
    one-unit increase in $x_j$ — larger $\beta_j$ means *longer* survival.
    """

    distribution_class = Weibull

    def _make_dist(self, mu, sigma):
        return Weibull(shape=1.0 / sigma, scale=np.exp(mu))


class LogNormalAFT(AFT):
    r"""Log-normal AFT model.

    $$
    \log T \mid x \sim \mathcal{N}(\mu, \sigma^2),
    \qquad \mu = \beta^\top x + b.
    $$

    Useful when the hazard is non-monotone (rises then falls) — for
    example, post-surgical recovery curves.
    """

    distribution_class = LogNormal

    def _make_dist(self, mu, sigma):
        return LogNormal(mu=mu, sigma=sigma)


class LogLogisticAFT(AFT):
    r"""Log-logistic (Fisk) AFT model.

    $$
    T \mid x \sim \mathrm{LogLogistic}\big(\text{shape}=1/\sigma,\;
        \text{scale}=\exp(\mu)\big), \qquad \mu = \beta^\top x + b.
    $$

    Has closed-form survival and a unimodal hazard for $\sigma < 1$
    (shape $> 1$) — popular in clinical AFT modelling for that reason.
    """

    distribution_class = LogLogistic

    def _make_dist(self, mu, sigma):
        return LogLogistic(shape=1.0 / sigma, scale=np.exp(mu))
