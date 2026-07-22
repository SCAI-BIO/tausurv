from hazardous import SurvivalBoost as SB
import numpy as np
from numpy.typing import ArrayLike, NDArray

from tausurv.predictor import CompetingRisksPredictor


class SurvivalBoost(CompetingRisksPredictor):
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
        ipcw_strategy: str = "alternating",
        n_iter_before_feedback: int = 20,
        seed: int | None = None,
        n_horizons_per_observation: int = 3,
    ) -> None:

        self._impl = SB(
            hard_zero_fraction=hard_zero_fraction,
            n_iter=n_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            show_progressbar=False,
            n_time_grid_steps=n_time_grid_steps,
            time_horizon=time_horizon,
            ipcw_strategy=ipcw_strategy,
            n_iter_before_feedback=n_iter_before_feedback,
            random_state=seed,
            n_horizons_per_observation=n_horizons_per_observation,
        )

    def fit(
        self,
        X: ArrayLike,
        event_time: ArrayLike,
        event_indicator: ArrayLike,
    ) -> "SurvivalBoost":
        self._impl.fit(X, {"event": event_indicator, "duration": event_time})
        self.times_ = self._impl.time_grid_
        self.n_causes = len(
            [event_id for event_id in self._impl.event_ids_ if event_id != 0]
        )
        return self

    def _cif(
        self,
        X: ArrayLike,
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        cif = self._impl.predict_cumulative_incidence(X, times)
        cif = cif[:, 1:, :]  # remove probabilities for staying event-free in general
        return np.maximum.accumulate(cif, axis=-1)  # ensure monotonicity
