r"""Survival-aware splitting and cross-validation utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def train_test_split(
    X: ArrayLike,
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    *,
    test_size: float = 0.25,
    stratify: bool = True,
    seed: int | None = None,
) -> tuple[
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
]:
    r"""Survival-aware train/test split.

    Returns ``(X_train, X_test, T_train, T_test, E_train, E_test)`` —
    flat 6-tuple, sklearn-compatible to unpack.

    With ``stratify=True`` (the default), each unique value of
    ``event_indicator`` is split independently so the proportions of
    each event class match in both train and test. This handles
    single-event (``δ ∈ {0, 1}``) and competing risks (``δ ∈ {0, 1,
    ..., K}``) the same way — blind random splits on rare-event or
    competing-risks data can produce all-censored test sets by chance.

    Parameters
    ----------
    X : (n, d) array_like
    event_time : (n,) array_like
    event_indicator : (n,) array_like
        Event class. By package convention $\delta = 0$ censored and
        $\delta = k \ge 1$ event of cause $k$.
    test_size : float, default 0.25
        Proportion of samples to allocate to the test split. Must be in
        $(0, 1)$.
    stratify : bool, default True
        Stratify the split by ``event_indicator``. Set ``False`` for a
        plain random split.
    seed : int, optional

    Returns
    -------
    X_train, X_test : (n_train, d), (n_test, d) arrays
    T_train, T_test : (n_train,), (n_test,) arrays
    E_train, E_test : (n_train,), (n_test,) arrays
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    X = np.asarray(X)
    T = np.asarray(event_time)
    E = np.asarray(event_indicator)
    n = T.shape[0]
    if X.shape[0] != n or E.shape[0] != n:
        raise ValueError(
            f"X, event_time, event_indicator must share their first axis; "
            f"got shapes {X.shape}, {T.shape}, {E.shape}"
        )

    rng = np.random.default_rng(seed)

    if stratify:
        train_parts: list[NDArray] = []
        test_parts: list[NDArray] = []
        for cls in np.unique(E):
            idx = np.flatnonzero(E == cls)
            rng.shuffle(idx)
            n_test_cls = int(round(idx.size * test_size))
            test_parts.append(idx[:n_test_cls])
            train_parts.append(idx[n_test_cls:])
        train_idx = np.concatenate(train_parts)
        test_idx = np.concatenate(test_parts)
    else:
        perm = rng.permutation(n)
        n_test = int(round(n * test_size))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]

    # Shuffle the final indices so order isn't grouped by class.
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return (
        X[train_idx],
        X[test_idx],
        T[train_idx],
        T[test_idx],
        E[train_idx],
        E[test_idx],
    )
