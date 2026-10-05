from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tausurv._tausurv_core import LogRankSurvivalTree, fit_log_rank_tree
from tausurv.predictor import SurvivalPredictor


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

    Reference NumPy implementation kept for transparency, tests, and
    use by callers outside the tree machinery. The compiled tree fitter
    uses an equivalent Rust implementation internally.
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
    The fit and predict paths are implemented in compiled Rust
    (:func:`tausurv.core.fit_log_rank_tree`); this Python class is a
    thin wrapper that exposes the :class:`SurvivalPredictor` contract.

    Parameters
    ----------
    max_depth : int, optional
        Maximum tree depth.
    min_samples_leaf : int, default 15
        Smallest allowed leaf size.
    max_features : int | ``"sqrt"`` | None, default None
        Features considered per split. ``None`` uses all features;
        ``"sqrt"`` uses $\lfloor \sqrt d \rfloor$; an integer uses that
        many.
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

    _handle: LogRankSurvivalTree

    def __init__(
        self,
        *,
        max_depth: int | None = None,
        min_samples_leaf: int = 15,
        max_features: int | str | None = None,
        seed: int | None = None,
    ) -> None:
        if max_features is not None and not isinstance(max_features, int):
            if max_features not in ("sqrt", "all"):
                raise ValueError(
                    f"invalid max_features: {max_features!r}; "
                    f"expected None, 'sqrt', 'all', or a positive integer"
                )
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
        X_f, T_c, E_u = _coerce_inputs(X, event_time, event_indicator)
        self.times_ = np.unique(T_c[E_u == 1])
        self._handle = fit_log_rank_tree(
            X_f,
            T_c,
            E_u,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            max_features=self.max_features,
            seed=self.seed,
        )
        return self

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
        X_f = np.asfortranarray(np.asarray(X, dtype=np.float64))
        return np.asarray(
            self._handle.predict_cumulative_hazard(X_f, times_arr),
            dtype=np.float64,
        )

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


def _coerce_inputs(
    X: ArrayLike,
    event_time: ArrayLike,
    event_indicator: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.uint8]]:
    """Cast inputs to the layout the Rust core expects.

    The fitter requires column-major `float64` X, contiguous `float64`
    event times, and contiguous `uint8` event indicators. ``np.asfortranarray``
    and ``np.ascontiguousarray`` are no-ops when the caller already
    provides the right layout.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2D (n, d); got shape {X_arr.shape}")
    X_f = np.asfortranarray(X_arr)
    T_c = np.ascontiguousarray(np.asarray(event_time, dtype=np.float64))
    E_u = np.ascontiguousarray(np.asarray(event_indicator, dtype=np.uint8))
    return X_f, T_c, E_u
