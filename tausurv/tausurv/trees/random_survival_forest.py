from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tausurv._tausurv_core import LogRankSurvivalForest, fit_log_rank_forest
from tausurv.predictor import SurvivalPredictor
from tausurv.trees.survival_tree import _coerce_inputs


class RandomSurvivalForest(SurvivalPredictor):
    r"""Random Survival Forest (Ishwaran et al., 2008).

    Bootstrap-aggregates log-rank survival trees with per-node random
    feature subsampling. Predictions average cumulative-hazard estimates
    across trees. The fit and predict paths are implemented in compiled
    Rust (:func:`tausurv.core.fit_log_rank_forest`); this Python class
    is a thin wrapper exposing the :class:`SurvivalPredictor` contract.

    Parameters
    ----------
    n_estimators : int, default 100
    max_depth : int, optional
    min_samples_leaf : int, default 15
    max_features : int | ``"sqrt"`` | ``"all"`` | None, default ``"sqrt"``
        Features sampled per split. ``"sqrt"`` is $\lfloor \sqrt d \rfloor$;
        ``"all"`` / ``None`` uses every feature.
    bootstrap : bool, default True
        Resample with replacement for each tree.
    seed : int, optional

    Attributes
    ----------
    times_ : (k,) array
        Unique training event times — the model's natural time grid.

    References
    ----------
    Ishwaran, H., Kogalur, U. B., Blackstone, E. H., Lauer, M. S. (2008).
    Random survival forests. Annals of Applied Statistics, 2(3).
    """

    _handle: LogRankSurvivalForest

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int = 15,
        max_features: int | str | None = "sqrt",
        bootstrap: bool = True,
        seed: int | None = None,
    ) -> None:
        if max_features is not None and not isinstance(max_features, int):
            if max_features not in ("sqrt", "all"):
                raise ValueError(
                    f"invalid max_features: {max_features!r}; "
                    f"expected None, 'sqrt', 'all', or a positive integer"
                )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.seed = seed

    def fit(
        self,
        X: ArrayLike,
        event_time: ArrayLike,
        event_indicator: ArrayLike,
    ) -> "RandomSurvivalForest":
        X_f, T_c, E_u = _coerce_inputs(X, event_time, event_indicator)
        self.times_ = np.unique(T_c[E_u == 1])
        self._handle = fit_log_rank_forest(
            X_f, T_c, E_u,
            n_trees=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            seed=self.seed,
        )
        return self

    def predict_cumulative_hazard(
        self,
        X: ArrayLike,
        times: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        r"""Tree-averaged $\hat \Lambda(t \mid x)$.

        Overrides the default :class:`SurvivalPredictor` derivation
        ``-log S`` so very large hazards survive the log/exp clip.
        """
        times_arr = self._resolve_times(times)
        X_f = np.asfortranarray(np.asarray(X, dtype=np.float64))
        return np.asarray(
            self._handle.predict_cumulative_hazard(X_f, times_arr),
            dtype=np.float64,
        )

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        r"""Ishwaran mortality, averaged across trees."""
        H = self.predict_cumulative_hazard(X, self.times_)
        return H.sum(axis=1)

    def _survival_function(
        self,
        X: ArrayLike,
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return np.exp(-self.predict_cumulative_hazard(X, times))
