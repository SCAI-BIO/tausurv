from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn

from tausurv.nn._utils import as_model_tensor
from tausurv.nn.modules.mlp import MLP
from tausurv.nn.pretrained import PretrainedMixin
from tausurv.predictor import SurvivalPredictor


@dataclass
class DeepSurvConfig:
    """Configuration for :class:`DeepSurv`."""

    in_features: int
    hidden_dim: int = 64
    n_blocks: int = 2
    activation: Literal["gelu", "relu"] = "gelu"
    norm: Literal["layer", "batch", "none"] = "layer"
    dropout: float = 0.1
    residual: bool = True


class DeepSurv(SurvivalPredictor, PretrainedMixin, nn.Module):
    r"""Feed-forward Cox-style risk model (Katzman et al., 2018).

    Predicts a scalar log-risk $\theta(x) = f_\theta(x)$ per subject. Pair
    with :func:`tausurv.nn.functional.cox_nll` for training; the model itself
    has no notion of survival times or censoring — that lives in the loss.

    Producing a survival function $\hat S(t \mid x)$ requires a baseline
    cumulative hazard. After training, call :meth:`fit_baseline` with the
    training data to estimate $\hat \Lambda_0$ by Breslow; afterwards
    :meth:`predict_survival_function`, :meth:`predict_cumulative_hazard`,
    :meth:`predict_cif`, :meth:`predict_rmst`, and :meth:`predict_risk_at`
    are all available.

    Construction is HF-style — pass a :class:`DeepSurvConfig` or kwargs:

    >>> model = DeepSurv(in_features=5, hidden_dim=32)
    >>> # or equivalently
    >>> cfg = DeepSurvConfig(in_features=5, hidden_dim=32)
    >>> model = DeepSurv(cfg)

    Parameters
    ----------
    config : DeepSurvConfig, optional
        Typed config. Pass either this OR kwargs, not both.
    **kwargs
        Quick form — forwarded to :class:`DeepSurvConfig`.

    References
    ----------
    Katzman, J. L. et al. (2018). DeepSurv: personalized treatment
    recommender using a Cox proportional hazards deep neural network. BMC
    Medical Research Methodology, 18(1).
    """

    config_class = DeepSurvConfig
    config: DeepSurvConfig

    def __init__(
        self,
        config: DeepSurvConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is None:
            config = DeepSurvConfig(**kwargs)
        elif kwargs:
            raise TypeError(f"{type(self).__name__}: pass `config` OR kwargs, not both")
        self.config = config
        self.backbone = MLP(
            config.in_features,
            1,
            hidden_dim=config.hidden_dim,
            n_blocks=config.n_blocks,
            activation=config.activation,
            norm=config.norm,
            dropout=config.dropout,
            residual=config.residual,
        )
        self._times: NDArray[np.float64] | None = None
        self._baseline_cumhazard: NDArray[np.float64] | None = None

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x).squeeze(-1)

    def fit_baseline(
        self,
        X: ArrayLike,
        event_time: ArrayLike,
        event_indicator: ArrayLike,
    ) -> "DeepSurv":
        r"""Breslow estimate of $\hat \Lambda_0$ on the training set.

        Stores :attr:`times_` (unique event times) and
        :attr:`baseline_cumhazard_` (right-continuous step function values),
        enabling all survival-function-based prediction methods.
        """
        Y = np.asarray(event_time, dtype=np.float64)
        delta = np.asarray(event_indicator, dtype=np.int8)
        X_t = as_model_tensor(X, self)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            log_risk = self.forward(X_t).cpu().numpy().astype(np.float64)
        if was_training:
            self.train()

        exp_lp = np.exp(log_risk)
        order_asc = np.argsort(Y, kind="stable")
        Y_s = Y[order_asc]
        delta_s = delta[order_asc]
        exp_lp_s = exp_lp[order_asc]
        S0_from_right = np.cumsum(exp_lp_s[::-1])[::-1]

        event_times = Y_s[delta_s == 1]
        unique_times, counts = np.unique(event_times, return_counts=True)
        first_pos = np.searchsorted(Y_s, unique_times, side="left")
        S0_at_unique = S0_from_right[first_pos]
        cum_hazard = np.cumsum(counts / S0_at_unique)

        self._times = unique_times
        self._baseline_cumhazard = cum_hazard
        return self

    @property
    def times_(self) -> NDArray[np.float64]:  # type: ignore[override]
        if self._times is None:
            raise RuntimeError(
                f"{type(self).__name__}.times_ is not set; call fit_baseline() first"
            )
        return self._times

    @property
    def baseline_cumhazard_(self) -> NDArray[np.float64]:
        if self._baseline_cumhazard is None:
            raise RuntimeError(
                f"{type(self).__name__}.baseline_cumhazard_ is not set; "
                f"call fit_baseline() first"
            )
        return self._baseline_cumhazard

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        r"""Log-risk $\theta(x)$ — the natural Cox-style ranking score.

        Available immediately after training; does not require a fitted
        baseline.
        """
        X_t = as_model_tensor(X, self)
        self.eval()
        with torch.no_grad():
            return self.forward(X_t).cpu().numpy().astype(np.float64)

    def _survival_function(
        self,
        X: ArrayLike,
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self._times is None or self._baseline_cumhazard is None:
            raise RuntimeError(
                f"{type(self).__name__}.predict_survival_function requires a "
                f"fitted baseline; call fit_baseline() first"
            )
        exp_lp = np.exp(self.predict(X))
        # Right-continuous step lookup; H0(t) = 0 for t before the first event.
        idx = np.searchsorted(self._times, times, side="right") - 1
        H0_at = np.where(
            idx >= 0,
            self._baseline_cumhazard[np.clip(idx, 0, None)],
            0.0,
        )
        return np.exp(-H0_at[None, :] * exp_lp[:, None])

    def save_pretrained(self, path: str | Path) -> None:
        """Persist config + weights, plus the fitted baseline (if any)."""
        super().save_pretrained(path)
        if self._times is not None and self._baseline_cumhazard is not None:
            np.savez(
                Path(path) / "baseline.npz",
                times=self._times,
                cumhazard=self._baseline_cumhazard,
            )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "DeepSurv":
        model: DeepSurv = super().from_pretrained(path)  # type: ignore[assignment]
        baseline_path = Path(path) / "baseline.npz"
        if baseline_path.exists():
            data = np.load(baseline_path)
            model._times = data["times"]
            model._baseline_cumhazard = data["cumhazard"]
        return model
