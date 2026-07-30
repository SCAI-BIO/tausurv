from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tausurv.predictor import CompetingRisksPredictor

if TYPE_CHECKING:
    from hazardous import SurvivalBoost as _HazardousSurvivalBoost


class SurvivalBoost(CompetingRisksPredictor):
    r"""Gradient-boosted competing-risks model.

    Estimates the cause-specific cumulative incidence $\hat F_j(t \mid x)$
    directly by boosting a classifier over randomly sampled time horizons,
    with inverse-probability-of-censoring weights correcting for censoring
    bias.

    This class is an adapter. The estimator itself lives in the external
    `hazardous <https://soda-inria.github.io/hazardous/>`_ package, an
    optional dependency: install with ``pip install tausurv[boost]``. The
    import is deferred to :meth:`fit`, so importing ``tausurv`` does not
    require it.

    Parameters
    ----------
    hard_zero_fraction : float, default 0.1
        Fraction of sampled horizons pinned at $t = 0$, anchoring the
        model on $F_j(0) = 0$.
    n_iter : int, default 100
        Boosting iterations.
    learning_rate : float, default 0.05
        Shrinkage applied to each boosting iteration.
    max_leaf_nodes : int, default 31
        Leaves per base tree.
    max_depth : int, optional
        Depth cap per base tree. Unbounded by default; ``max_leaf_nodes``
        does the limiting.
    min_samples_leaf : int, default 50
        Minimum observations per leaf.
    n_time_grid_steps : int, default 100
        Size of the internal time grid used for horizon sampling and
        censoring weights. Becomes :attr:`times_` after fitting.
    time_horizon : float, optional
        Default horizon for the underlying estimator's own scoring
        helpers. Unused by the tausurv predict methods, which take an
        explicit ``times`` grid.
    ipcw_strategy : {"alternating", "complete"}, default "alternating"
        How the censoring model is estimated. ``"alternating"`` refits it
        periodically against the current incidence model;
        ``"complete"`` fits it once up front from the marginal censoring
        distribution.
    n_iter_before_feedback : int, default 20
        Boosting iterations between censoring-model refits. Ignored when
        ``ipcw_strategy="complete"``.
    seed : int, optional
        Seed for horizon sampling and base-tree randomness.
    n_horizons_per_observation : int, default 3
        Horizons drawn per observation per iteration. Higher values
        reduce gradient variance at proportional cost.

    Attributes
    ----------
    times_ : (k,) array
        The fitted estimator's internal time grid.
    n_causes : int
        Number of competing causes, counted from the causes **present in
        the training data**. A training split missing a rare cause yields
        a model with fewer causes, and :meth:`predict_cif` will reject
        that cause's index. Check it after fitting when working with
        cross-validation folds or heavily imbalanced causes.

    Notes
    -----
    Parameters are forwarded verbatim to ``hazardous.SurvivalBoost``; see
    its documentation for the full detail behind each one.

    References
    ----------
    hazardous: https://soda-inria.github.io/hazardous/
    """

    _impl: _HazardousSurvivalBoost

    def __init__(
        self,
        *,
        hard_zero_fraction: float = 0.1,
        n_iter: int = 100,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
        max_depth: int | None = None,
        min_samples_leaf: int = 50,
        n_time_grid_steps: int = 100,
        time_horizon: float | None = None,
        ipcw_strategy: Literal["alternating", "complete"] = "alternating",
        n_iter_before_feedback: int = 20,
        seed: int | None = None,
        n_horizons_per_observation: int = 3,
    ) -> None:
        self.hard_zero_fraction = hard_zero_fraction
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_time_grid_steps = n_time_grid_steps
        self.time_horizon = time_horizon
        self.ipcw_strategy = ipcw_strategy
        self.n_iter_before_feedback = n_iter_before_feedback
        self.seed = seed
        self.n_horizons_per_observation = n_horizons_per_observation

    def fit(
        self,
        X: ArrayLike,
        event_time: ArrayLike,
        event_indicator: ArrayLike,
    ) -> "SurvivalBoost":
        self._impl = self._build()
        self._impl.fit(X, {"event": event_indicator, "duration": event_time})
        self.times_ = np.asarray(self._impl.time_grid_, dtype=np.float64)
        self.n_causes = sum(1 for cause in self._impl.event_ids_ if cause != 0)
        return self

    def _build(self) -> _HazardousSurvivalBoost:
        try:
            from hazardous import SurvivalBoost as HazardousSurvivalBoost
        except ImportError as exc:
            raise ImportError(
                "SurvivalBoost requires the optional 'hazardous' dependency; "
                "install it with `pip install tausurv[boost]`"
            ) from exc
        return HazardousSurvivalBoost(
            hard_zero_fraction=self.hard_zero_fraction,
            n_iter=self.n_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            show_progressbar=False,
            n_time_grid_steps=self.n_time_grid_steps,
            time_horizon=self.time_horizon,
            ipcw_strategy=self.ipcw_strategy,
            n_iter_before_feedback=self.n_iter_before_feedback,
            random_state=self.seed,
            n_horizons_per_observation=self.n_horizons_per_observation,
        )

    def _cif(
        self,
        X: ArrayLike,
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        cif = self._impl.predict_cumulative_incidence(X, times)
        # hazardous returns the event-free probability as cause 0; the
        # predictor contract covers causes 1..J only.
        cif = cif[:, 1:, :]
        # Horizon-sampled boosting carries no monotonicity constraint, so
        # the raw estimate can dip on fine grids.
        return np.maximum.accumulate(cif, axis=-1)
