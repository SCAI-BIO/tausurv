from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def time_grid(
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    n_bins: int = 20,
) -> NDArray[np.float64]:
    r"""Choose right bin edges for discrete-time models from training data.

    Interior edges sit at quantiles of the uncensored event times, so each
    bin captures roughly the same number of events — the spacing that keeps
    per-bin hazard estimates stable. The last edge is the largest observed
    time, event or censored, so the grid spans the training data and no
    observation falls beyond the final bin.

    Edge $t_k$ closes the interval $(t_{k-1}, t_k]$; :func:`bin_index`
    places observed times under this convention. Pass the returned grid
    both as ``time_bins`` to the discrete-time losses and to
    ``model.set_time_grid`` so training and prediction agree.

    Parameters
    ----------
    event_time : (n,) array
        Observed time $Y = \min(T, C)$.
    event_indicator : (n,) array
        $\delta = 0$ if censored; any positive value counts as an event,
        so competing-risks cause labels can be passed as-is.
    n_bins : int, default 20
        Number of edges requested. Tied quantiles are collapsed, so the
        grid can come back shorter — size the model to the grid
        (``n_bins=len(grid)``), not the other way around.

    Returns
    -------
    (K,) float array
        Strictly increasing right bin edges, ``K <= n_bins``, ending at
        ``max(event_time)``.

    References
    ----------
    Kvamme, H., Borgan, Ø. (2021). Continuous and discrete-time survival
    prediction with neural networks. Lifetime Data Analysis, 27(4).
    """
    Y = np.asarray(event_time, dtype=np.float64)
    delta = np.asarray(event_indicator)
    if Y.ndim != 1 or delta.shape != Y.shape:
        raise ValueError(
            f"event_time and event_indicator must be 1-d with matching "
            f"shapes, got {Y.shape} and {delta.shape}"
        )
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    events = Y[delta > 0]
    if events.size == 0:
        raise ValueError("event_indicator marks no events; a grid needs at least one")
    edges = np.quantile(events, np.linspace(0.0, 1.0, n_bins + 1)[1:])
    edges[-1] = Y.max()
    return np.unique(edges)


def bin_index(time_grid: ArrayLike, times: ArrayLike) -> NDArray[np.intp]:
    r"""Map times onto a grid of right bin edges.

    Bin $k$ covers $(t_{k-1}, t_k]$: a time equal to an edge belongs to
    the bin that edge closes, and anything at or below the first edge
    lands in bin 0. Times beyond the last edge return $K$ — one past the
    final bin — leaving the caller to fold them into the last bin
    (``.clip(max=K - 1)``, what the discrete-time losses do) or treat
    them as beyond the model's horizon.

    Parameters
    ----------
    time_grid : (K,) array
        Right bin edges, sorted ascending, e.g. from :func:`time_grid`.
    times : array
        Times to place, any shape.

    Returns
    -------
    integer array
        Bin indices in ``0..K``, same shape as ``times``.
    """
    edges = np.asarray(time_grid, dtype=np.float64)
    return np.searchsorted(edges, np.asarray(times, dtype=np.float64), side="left")
