from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tausurv.predictor import SurvivalPredictor
from tausurv.trees.survival_tree import SurvivalTree


class RandomSurvivalForest(SurvivalPredictor):
    r"""Random Survival Forest (Ishwaran et al., 2008).

    Bootstrap-aggregates :class:`SurvivalTree` instances with per-node random
    feature subsampling. Predictions average cumulative-hazard estimates
    across trees. Inherits the unified survival prediction API from
    :class:`SurvivalPredictor`.

    Parameters
    ----------
    n_estimators : int, default 100
    max_depth : int, optional
    min_samples_leaf : int, default 15
    max_features : int | "sqrt" | None, default "sqrt"
        Features sampled per split; "sqrt" is $\lfloor \sqrt d \rfloor$.
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

    _trees: list[SurvivalTree]

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
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(event_time, dtype=np.float64)
        delta = np.asarray(event_indicator, dtype=np.int8)

        n = len(Y)
        rng = np.random.default_rng(self.seed)
        self.times_ = np.unique(Y[delta == 1])

        self._trees = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n) if self.bootstrap else np.arange(n)
            tree_seed = int(rng.integers(0, 2**31))
            tree = SurvivalTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                seed=tree_seed,
            )
            tree.fit(X[idx], Y[idx], delta[idx])
            self._trees.append(tree)
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
        total = np.zeros((np.asarray(X).shape[0], len(times_arr)))
        for tree in self._trees:
            total += tree.predict_cumulative_hazard(X, times_arr)
        return total / len(self._trees)

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
