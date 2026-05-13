r"""SurvITE — direct CATE learner (Curth et al., 2021).

A neural-network direct estimator for survival CATE. Architecturally a
shared encoder feeds K treatment-specific hazard heads (one MLP per
arm); training combines a discrete-hazard NLL on each arm's factual
data with an IPM regulariser on the encoder output to balance
representations across arms (Shalit et al., 2017).

Despite the name, SurvITE — like every observational HTE method —
estimates **CATE**, not the Rubin ITE. See
:mod:`causurv.predictor` for the estimand glossary.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from causurv.nn.losses.survite import SurvITELoss
from causurv.nn.models.survite import SurvITEConfig, SurvITEModule, SurvITETrainer
from causurv.predictor import HTEPredictor, _validate_fit_inputs
from tausurv.nn._utils import as_model_tensor


class SurvITE(HTEPredictor):
    r"""Direct CATE learner via balanced representations + per-arm hazard heads.

    Parameters
    ----------
    n_bins : int, optional
        Number of discrete-time bins. If ``None`` (default), the bin
        right-edges are chosen from the training event-time quantiles
        (a la :class:`tausurv.nn.LogisticHazard`'s recipe).
    repr_dim : int, default ``100``
        Encoder output (representation) dimension.
    encoder_hidden, encoder_blocks : int, defaults ``100, 2``
        Encoder MLP geometry.
    heads_hidden, heads_blocks : int, defaults ``100, 1``
        Per-arm head MLP geometry.
    activation : ``{"gelu", "relu"}``, default ``"relu"``
    norm : ``{"layer", "batch", "none"}``, default ``"layer"``
    dropout : float, default ``0.1``
    ipm_beta : float, default ``1e-3``
        Weight on the representation-balance IPM penalty. ``0`` disables
        balancing entirely (the model becomes a representation-shared
        T-learner).
    ipm_distance : ``{"wasserstein", "mmd"}``, default ``"wasserstein"``
    lr : float, default ``1e-3``
    weight_decay : float, default ``0.0``
    epochs : int, default ``100``
    batch_size : int, optional
        Mini-batch size. ``None`` uses full-batch.
    grad_clip : float, default ``0.0``
    device : str or torch.device, optional
        Defaults to CUDA if available else CPU.
    verbose : bool, default ``False``
    seed : int, default ``0``

    Attributes
    ----------
    times_ : ``(n_bins,)`` array
        Bin right-edges fitted to the training data.

    Examples
    --------
    Train on the SurvITE benchmark, evaluate against its oracle::

        from causurv.simulations import SurvITE as SurvITESim
        from causurv.learners import SurvITE as SurvITELearner
        from causurv.metrics.pehe import integrated_pehe

        sim = SurvITESim(scenario="S4")
        X, T, E, A = sim.generate(n=2000, seed=0)

        m = SurvITELearner(epochs=100, batch_size=256, seed=0).fit(X, T, E, A)
        hte_hat  = m.predict_hte(X, times=[5, 10, 20])
        hte_true = sim.predict_hte(X, times=[5, 10, 20])
        score = integrated_pehe(hte_hat, hte_true)
    """

    def __init__(
        self,
        *,
        n_bins: int | None = None,
        repr_dim: int = 100,
        encoder_hidden: int = 100,
        encoder_blocks: int = 2,
        heads_hidden: int = 100,
        heads_blocks: int = 1,
        activation: Literal["gelu", "relu"] = "relu",
        norm: Literal["layer", "batch", "none"] = "layer",
        dropout: float = 0.1,
        ipm_beta: float = 1e-3,
        ipm_distance: Literal["wasserstein", "mmd"] = "wasserstein",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 100,
        batch_size: int | None = None,
        grad_clip: float = 0.0,
        device: "str | torch.device | None" = None,
        verbose: bool = False,
        seed: int = 0,
    ) -> None:
        self.n_bins = n_bins
        self.repr_dim = repr_dim
        self.encoder_hidden = encoder_hidden
        self.encoder_blocks = encoder_blocks
        self.heads_hidden = heads_hidden
        self.heads_blocks = heads_blocks
        self.activation = activation
        self.norm = norm
        self.dropout = dropout
        self.ipm_beta = ipm_beta
        self.ipm_distance = ipm_distance
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.verbose = verbose
        self.seed = seed

        self._module: SurvITEModule | None = None
        self._loss_fn: SurvITELoss | None = None

    def fit(
        self,
        X: ArrayLike,
        event_time: ArrayLike,
        event_indicator: ArrayLike,
        treatment: ArrayLike,
    ) -> "SurvITE":
        X, T, E, A = _validate_fit_inputs(X, event_time, event_indicator, treatment)

        n_arms = int(A.max()) + 1
        time_bins = self._choose_time_bins(T, E)
        self.times_ = time_bins

        # Seed before module construction so weight init is reproducible
        # (Trainer's own seeding happens after the model is built).
        torch.manual_seed(self.seed)
        cfg = SurvITEConfig(
            in_features=X.shape[1],
            n_bins=len(time_bins),
            n_arms=n_arms,
            repr_dim=self.repr_dim,
            encoder_hidden=self.encoder_hidden,
            encoder_blocks=self.encoder_blocks,
            heads_hidden=self.heads_hidden,
            heads_blocks=self.heads_blocks,
            activation=self.activation,
            norm=self.norm,
            dropout=self.dropout,
        )
        module = SurvITEModule(cfg)
        time_bins_t = torch.as_tensor(time_bins, dtype=torch.float32)
        loss_fn = SurvITELoss(
            time_bins_t, beta=self.ipm_beta, ipm=self.ipm_distance
        )

        trainer = SurvITETrainer(
            module,
            loss_fn=loss_fn,
            optimizer="adamw",
            lr=self.lr,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            device=self.device,
            seed=self.seed,
        )
        train_data = (
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(T, dtype=torch.float32),
            torch.as_tensor(E, dtype=torch.float32),
            torch.as_tensor(A, dtype=torch.long),
        )
        trainer.fit(
            train_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            save_best=False,
        )

        self._module = module
        self._loss_fn = loss_fn
        self._fit_X = X
        return self

    def predict_potential_outcomes(
        self,
        X: ArrayLike,
        times: ArrayLike | None = None,
        *,
        cause: int | None = None,
    ) -> tuple[NDArray[np.float64], ...]:
        if cause is not None:
            raise ValueError(
                f"SurvITE is single-event; cause must be None (got cause={cause})"
            )
        if self._module is None:
            raise RuntimeError(f"{type(self).__name__}: call fit() first")
        times_arr = self._resolve_times(times)
        X_t = as_model_tensor(X, self._module)

        self._module.eval()
        with torch.no_grad():
            _, logits_list = self._module(X_t)
            hazards = [torch.sigmoid(z) for z in logits_list]
            # S(t_k) = prod_{j <= k} (1 - h_j) on the natural bin grid.
            S_grid = [
                torch.exp(
                    torch.log1p(-h.clamp(max=1.0 - 1e-7)).cumsum(dim=1)
                ).cpu().numpy().astype(np.float64)
                for h in hazards
            ]

        # Step-evaluate onto the requested grid using the bin right-edges
        # as breakpoints (S(t) = S at the last bin whose edge is <= t).
        bin_edges = np.asarray(self.times_, dtype=np.float64)
        idx = np.searchsorted(bin_edges, times_arr, side="right") - 1
        before = idx < 0
        idx_clipped = np.clip(idx, 0, len(bin_edges) - 1)

        out = []
        for S in S_grid:
            S_at = S[:, idx_clipped]
            if before.any():
                S_at[:, before] = 1.0
            out.append(S_at)
        return tuple(out)

    def _choose_time_bins(
        self,
        event_time: NDArray[np.float64],
        event_indicator: NDArray,
    ) -> NDArray[np.float64]:
        """Pick bin right-edges from event times.

        - If ``self.n_bins`` is set, use that many quantile-spaced bins
          over uncensored event times.
        - Otherwise, default to the integer grid implied by the data
          (one bin per integer up to ``ceil(max T)``), which is what
          the SurvITE simulator natively produces.
        """
        T = np.asarray(event_time, dtype=np.float64)
        E = np.asarray(event_indicator)
        if self.n_bins is None:
            t_max = int(np.ceil(T.max()))
            return np.arange(1, t_max + 1, dtype=np.float64)
        events_T = T[E.astype(bool)]
        if events_T.size < self.n_bins:
            return np.unique(np.linspace(T.min(), T.max(), self.n_bins))
        qs = np.linspace(0.0, 1.0, self.n_bins + 1)[1:]
        edges = np.quantile(events_T, qs)
        return np.unique(edges)
