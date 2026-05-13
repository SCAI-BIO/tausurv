from typing import Literal

import numpy as np
from numpy.typing import NDArray

def version() -> str: ...

class LogRankSurvivalTree:
    @property
    def n_features(self) -> int: ...
    @property
    def n_nodes(self) -> int: ...
    @property
    def n_leaves(self) -> int: ...
    @property
    def depth(self) -> int: ...
    def predict_cumulative_hazard(
        self,
        X: NDArray[np.float64],
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...

def fit_log_rank_tree(
    X: NDArray[np.float64],
    event_time: NDArray[np.float64],
    event_indicator: NDArray[np.uint8],
    *,
    min_samples_leaf: int = ...,
    max_depth: int | None = ...,
    max_features: int | Literal["sqrt", "all"] | None = ...,
    honesty: bool = ...,
    honesty_fraction: float = ...,
    seed: int | None = ...,
) -> LogRankSurvivalTree: ...

class LogRankSurvivalForest:
    @property
    def n_trees(self) -> int: ...
    @property
    def n_features(self) -> int: ...
    @property
    def n_train_samples(self) -> int: ...
    def predict_cumulative_hazard(
        self,
        X: NDArray[np.float64],
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...
    def forest_weights(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def forest_weights_one(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

def fit_log_rank_forest(
    X: NDArray[np.float64],
    event_time: NDArray[np.float64],
    event_indicator: NDArray[np.uint8],
    *,
    n_trees: int = ...,
    min_samples_leaf: int = ...,
    max_depth: int | None = ...,
    max_features: int | Literal["sqrt", "all"] | None = ...,
    bootstrap: bool = ...,
    subsample_fraction: float = ...,
    honesty: bool = ...,
    honesty_fraction: float = ...,
    seed: int | None = ...,
) -> LogRankSurvivalForest: ...

class GradientForest:
    @property
    def n_trees(self) -> int: ...
    @property
    def n_features(self) -> int: ...
    @property
    def n_train_samples(self) -> int: ...
    def forest_weights(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def forest_weights_one(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

def fit_gradient_forest(
    X: NDArray[np.float64],
    pseudo_outcome: NDArray[np.float64],
    *,
    n_trees: int = ...,
    min_samples_leaf: int = ...,
    max_depth: int | None = ...,
    max_features: int | Literal["sqrt", "all"] | None = ...,
    bootstrap: bool = ...,
    subsample_fraction: float = ...,
    honesty: bool = ...,
    honesty_fraction: float = ...,
    seed: int | None = ...,
) -> GradientForest: ...
