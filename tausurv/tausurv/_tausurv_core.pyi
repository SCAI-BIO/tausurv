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
    seed: int | None = ...,
) -> LogRankSurvivalTree: ...
