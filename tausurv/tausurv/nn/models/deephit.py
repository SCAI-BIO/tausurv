from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn

from tausurv.nn._utils import (
    as_model_tensor,
    as_training_tensors,
    fit_time_grid,
    pmf_probabilities,
    reset_parameters,
    step_lookup,
)
from tausurv.nn.checkpoint import CheckpointMixin
from tausurv.nn.losses.deephit import DeepHitLoss
from tausurv.nn.modules.mlp import MLP
from tausurv.nn.training.fit import fit
from tausurv.predictor import CompetingRisksPredictor


@dataclass
class DeepHitConfig:
    """Architectural configuration for :class:`DeepHit`.

    Only architectural choices live here. The real-time interpretation
    of the bins is data-derived state and is set post-construction via
    :meth:`DeepHit.set_time_grid`.
    """

    in_features: int
    n_bins: int
    n_causes: int = 1
    hidden_dim: int = 64
    n_blocks: int = 2
    activation: Literal["gelu", "relu"] = "gelu"
    norm: Literal["layer", "batch", "none"] = "layer"
    dropout: float = 0.1
    residual: bool = True


class DeepHit(CompetingRisksPredictor, CheckpointMixin, nn.Module):
    r"""Discrete-time PMF survival model (Lee et al., 2018).

    Predicts logits over $K$ time bins for each of $K_{\text{causes}}$ event
    causes. The output shape is always ``(n, n_causes, n_bins)`` — default
    ``n_causes=1`` gives single-event survival; ``n_causes >= 2`` gives the
    competing-risks head from the original DeepHit-CR paper.

    A joint softmax over the flattened ``(n_causes * n_bins)`` dimension
    converts logits to a PMF over (cause, bin) pairs that sums to 1 per
    subject. Cause-specific CIFs are recovered as a cumulative sum along
    the bin axis; survival is the probability mass left after each bin.

    The model's output is bin-indexed (1..``n_bins``). :meth:`fit` derives
    the real-time interpretation of those bins from the training data with
    :func:`tausurv.discretization.time_grid`; when training through
    :class:`~tausurv.nn.Trainer` instead, set it with :meth:`set_time_grid`.
    If unset, prediction methods raise. The grid round-trips through
    :meth:`save` / :meth:`load`.

    Pair with :func:`tausurv.nn.functional.pmf_nll` and optionally
    :func:`tausurv.nn.functional.deephit_ranking` for training.

    References
    ----------
    Lee, C., Zame, W. R., Yoon, J., van der Schaar, M. (2018). DeepHit: A
    deep learning approach to survival analysis with competing risks. AAAI.
    """

    config_class = DeepHitConfig
    config: DeepHitConfig
    n_causes: int
    history_: dict[str, Any]

    def __init__(
        self,
        config: DeepHitConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is None:
            config = DeepHitConfig(**kwargs)
        elif kwargs:
            raise TypeError(f"{type(self).__name__}: pass `config` OR kwargs, not both")
        if config.n_causes < 1:
            raise ValueError(f"n_causes must be >= 1, got {config.n_causes}")

        self.config = config
        self.n_bins = config.n_bins
        self.n_causes = config.n_causes
        self.backbone = MLP(
            config.in_features,
            config.n_causes * config.n_bins,
            hidden_dim=config.hidden_dim,
            n_blocks=config.n_blocks,
            activation=config.activation,
            norm=config.norm,
            dropout=config.dropout,
            residual=config.residual,
        )

    def forward(self, x: Tensor) -> Tensor:
        logits_flat = self.backbone(x)
        return logits_flat.reshape(-1, self.n_causes, self.n_bins)

    def fit(
        self,
        X: ArrayLike,
        event_time: ArrayLike,
        event_indicator: ArrayLike,
        *,
        val_data: tuple[ArrayLike, ArrayLike, ArrayLike] | None = None,
        loss: Callable[..., Tensor] | None = None,
        epochs: int = 100,
        batch_size: int | None = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        seed: int = 0,
        device: str | torch.device | None = None,
        verbose: bool = False,
    ) -> "DeepHit":
        r"""Train on the DeepHit likelihood-plus-ranking loss.

        A thin wrapper around :func:`tausurv.nn.fit` with
        :class:`~tausurv.nn.losses.DeepHitLoss` as the default loss. The time
        grid comes from :func:`tausurv.discretization.time_grid` on the
        training data unless :meth:`set_time_grid` was called first. Use
        :class:`~tausurv.nn.Trainer` directly for schedules, early stopping,
        checkpointing or a custom training step.

        Parameters
        ----------
        X : (n, d) array
        event_time : (n,) array
        event_indicator : (n,) array
        val_data : (X, event_time, event_indicator), optional
            Held-out data; its loss is recorded in ``history_``.
        loss : callable, optional
            Replaces the default loss. Called as
            ``loss(logits, event_time, event_indicator, time_bins)``.
        epochs, batch_size, lr, weight_decay, seed, device, verbose
            Forwarded to :func:`tausurv.nn.fit`.

        Weights are re-initialised under ``seed`` first, so a second call
        retrains from scratch rather than continuing.

        Returns
        -------
        self
            With ``history_`` and the time grid set.
        """
        reset_parameters(self, seed)
        train = as_training_tensors(self, X, event_time, event_indicator)
        val = as_training_tensors(self, *val_data) if val_data is not None else None
        fit_time_grid(self, event_time, event_indicator)
        time_bins = as_model_tensor(self.times_, self)
        criterion = DeepHitLoss() if loss is None else loss

        def loss_fn(logits: Tensor, time: Tensor, indicator: Tensor) -> Tensor:
            return criterion(logits, time, indicator, time_bins)

        self.history_ = fit(
            self,
            loss_fn,
            train,
            val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
            device=device,
            verbose=verbose,
        )
        return self

    def set_time_grid(self, times: ArrayLike) -> "DeepHit":
        r"""Set the real-time interpretation of the model's bins.

        ``times`` must have length ``n_bins`` and gives the real-time
        upper-bound of each discrete bin — the same array passed as
        ``time_bins`` to the training loss, typically built with
        :func:`tausurv.discretization.time_grid`. Stored as ``self.times_`` and
        used by :class:`SurvivalPredictor` predict methods. Round-trips
        through :meth:`save` / :meth:`load`.
        """
        times = np.asarray(times, dtype=np.float64)
        if times.shape != (self.n_bins,):
            raise ValueError(
                f"times must have shape ({self.n_bins},), got {times.shape}"
            )
        self._times = times
        return self

    @property
    def times_(self) -> NDArray[np.float64]:  # type: ignore[override]
        if not hasattr(self, "_times"):
            raise RuntimeError(
                f"{type(self).__name__}.times_ is not set; call "
                f"`model.set_time_grid(times)` first or pass `times=` "
                f"explicitly to predict methods."
            )
        return self._times

    def _cif(
        self,
        X: ArrayLike,
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        cif, _ = self._bin_probabilities(X)
        return step_lookup(self._times, cif, times, before=0.0)

    def _survival_function(
        self,
        X: ArrayLike,
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        _, survival = self._bin_probabilities(X)
        return step_lookup(self._times, survival, times, before=1.0)

    def _bin_probabilities(
        self, X: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if not hasattr(self, "_times"):
            raise RuntimeError(
                f"{type(self).__name__} predictions require a time grid; "
                f"call `model.set_time_grid(times)` first."
            )
        X_t = as_model_tensor(X, self)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            logits = self.forward(X_t)
        if was_training:
            self.train()
        return pmf_probabilities(logits)

    def save(self, path: str | Path) -> None:
        """Persist config + weights, plus the time grid set by
        :meth:`set_time_grid` (if any)."""
        super().save(path)
        if hasattr(self, "_times"):
            np.savez(Path(path) / "time_grid.npz", times=self._times)

    @classmethod
    def load(cls, path: str | Path) -> "DeepHit":
        model: DeepHit = super().load(path)  # type: ignore[assignment]
        time_grid_path = Path(path) / "time_grid.npz"
        if time_grid_path.exists():
            model._times = np.load(time_grid_path)["times"]
        return model
