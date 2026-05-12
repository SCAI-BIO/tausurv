from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tausurv.nonparametric.nelson_aalen import nelson_aalen
from tausurv.predictor import SurvivalPredictor
from tausurv.step import StepFunction


@dataclass(slots=True)
class _Node:
    is_leaf: bool = False
    split_feature: int = -1
    split_threshold: float = 0.0
    left: Union["_Node", None] = None
    right: Union["_Node", None] = None
    leaf_hazard: StepFunction | None = None


def _log_rank_statistic(
    Y: NDArray[np.float64],
    delta: NDArray[np.int8],
    left_mask: NDArray[np.bool_],
) -> float:
    r"""Log-rank chi-square statistic for the binary split.

    $$
    Z = \frac{\big(\sum_i (O_{L,i} - E_{L,i})\big)^2}{\sum_i V_{L,i}}
    $$

    Higher is better; favors splits that separate the two groups' hazards.
    """
    if not (delta == 1).any():
        return 0.0

    order = np.argsort(Y, kind="stable")
    Y_s = Y[order]
    delta_s = delta[order].astype(np.int64)
    left_s = left_mask[order].astype(np.int64)

    event_times = np.unique(Y_s[delta_s == 1])
    first_idx = np.searchsorted(Y_s, event_times, side="left")
    end_idx = np.searchsorted(Y_s, event_times, side="right")

    cum_left = np.concatenate([[0], np.cumsum(left_s)])
    cum_delta = np.concatenate([[0], np.cumsum(delta_s)])
    cum_delta_left = np.concatenate([[0], np.cumsum(delta_s * left_s)])

    n_total = len(Y)
    at_risk_total = n_total - first_idx
    at_risk_left = cum_left[-1] - cum_left[first_idx]
    d_total = cum_delta[end_idx] - cum_delta[first_idx]
    d_left = cum_delta_left[end_idx] - cum_delta_left[first_idx]

    p = at_risk_left / at_risk_total
    e_left = d_total * p

    valid = at_risk_total > 1
    v_left = np.zeros_like(p)
    if valid.any():
        d_v = d_total[valid]
        p_v = p[valid]
        n_v = at_risk_total[valid]
        v_left[valid] = d_v * p_v * (1 - p_v) * (n_v - d_v) / (n_v - 1)

    obs_minus_exp = float((d_left - e_left).sum())
    var_sum = float(v_left.sum())
    if var_sum <= 0:
        return 0.0
    return obs_minus_exp**2 / var_sum


class SurvivalTree(SurvivalPredictor):
    r"""Single survival regression tree with log-rank splitting.

    Recursively partitions the covariate space; each leaf stores a
    Nelson-Aalen cumulative hazard estimated on the leaf's samples.

    Parameters
    ----------
    max_depth : int, optional
        Maximum tree depth.
    min_samples_leaf : int, default 15
        Smallest allowed leaf size.
    max_features : int | "sqrt" | None, default None
        Number of features considered per split.
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

    _root: _Node
    _n_features: int

    def __init__(
        self,
        *,
        max_depth: int | None = None,
        min_samples_leaf: int = 15,
        max_features: int | str | None = None,
        seed: int | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed

    def fit(
        self,
        X: ArrayLike,
        event_time: ArrayLike,
        event_indicator: ArrayLike,
    ) -> "SurvivalTree":
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(event_time, dtype=np.float64)
        delta = np.asarray(event_indicator, dtype=np.int8)

        self._n_features = X.shape[1]
        self._rng = np.random.default_rng(self.seed)
        self._n_features_per_split = self._resolve_max_features()
        self.times_ = np.unique(Y[delta == 1])

        self._root = self._build(X, Y, delta, depth=0)
        return self

    def _resolve_max_features(self) -> int:
        if self.max_features is None:
            return self._n_features
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(self._n_features)))
        if isinstance(self.max_features, int):
            return min(max(1, self.max_features), self._n_features)
        raise ValueError(f"invalid max_features: {self.max_features!r}")

    def _build(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        delta: NDArray[np.int8],
        depth: int,
    ) -> _Node:
        node = _Node()

        stop = (
            len(Y) < 2 * self.min_samples_leaf
            or (self.max_depth is not None and depth >= self.max_depth)
            or not (delta == 1).any()
        )
        if stop:
            node.is_leaf = True
            node.leaf_hazard = nelson_aalen(Y, delta)
            return node

        split = self._find_best_split(X, Y, delta)
        if split is None:
            node.is_leaf = True
            node.leaf_hazard = nelson_aalen(Y, delta)
            return node

        feature, threshold = split
        left_mask = X[:, feature] <= threshold
        node.split_feature = feature
        node.split_threshold = threshold
        node.left = self._build(X[left_mask], Y[left_mask], delta[left_mask], depth + 1)
        node.right = self._build(
            X[~left_mask], Y[~left_mask], delta[~left_mask], depth + 1
        )
        return node

    def _find_best_split(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        delta: NDArray[np.int8],
    ) -> tuple[int, float] | None:
        candidate_features = self._rng.choice(
            self._n_features,
            size=self._n_features_per_split,
            replace=False,
        )
        best_score = -np.inf
        best_split: tuple[int, float] | None = None

        for feature in candidate_features:
            vals = X[:, feature]
            unique_vals = np.unique(vals)
            if len(unique_vals) < 2:
                continue
            thresholds = 0.5 * (unique_vals[:-1] + unique_vals[1:])
            for threshold in thresholds:
                left_mask = vals <= threshold
                n_left = int(left_mask.sum())
                if n_left < self.min_samples_leaf:
                    continue
                if len(Y) - n_left < self.min_samples_leaf:
                    continue
                score = _log_rank_statistic(Y, delta, left_mask)
                if score > best_score:
                    best_score = score
                    best_split = (int(feature), float(threshold))
        return best_split

    def predict_cumulative_hazard(
        self,
        X: ArrayLike,
        times: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        r"""$\hat \Lambda(t \mid x)$ — the leaf's Nelson-Aalen estimate.

        Overrides the default :class:`SurvivalPredictor` derivation
        ``-log S`` so very large hazards are not lost to the log/exp clip.
        """
        times_arr = self._resolve_times(times)
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        out = np.empty((n, len(times_arr)))
        for i in range(n):
            leaf = self._leaf_for_sample(X[i])
            out[i] = leaf.leaf_hazard(times_arr)
        return out

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        r"""Ishwaran mortality: $M(x) = \sum_i \hat\Lambda(t_i \mid x)$ summed
        over training event times in :attr:`times_`."""
        H = self.predict_cumulative_hazard(X, self.times_)
        return H.sum(axis=1)

    def _survival_function(
        self,
        X: ArrayLike,
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return np.exp(-self.predict_cumulative_hazard(X, times))

    def _leaf_for_sample(self, x: NDArray[np.float64]) -> _Node:
        node = self._root
        while not node.is_leaf:
            if x[node.split_feature] <= node.split_threshold:
                node = node.left
            else:
                node = node.right
        return node
