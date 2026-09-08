r"""Survival-aware splits: one hold-out split and stratified folds."""

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


def stratified_folds(
    event_indicator: ArrayLike,
    n_splits: int = 5,
    *,
    seed: int | None = None,
) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    r"""K folds with every event class spread evenly across them.

    Each unique value of ``event_indicator`` is shuffled and dealt into the
    folds separately, so censored subjects and each event cause appear in
    every test fold in the same proportion as in the data. A plain random
    K-fold on rare-event or competing-risks data can leave a fold without a
    single event of some cause.

    Parameters
    ----------
    event_indicator : (n,) array_like
        Event class, $\delta = 0$ censored and $\delta = k \ge 1$ event of
        cause $k$.
    n_splits : int, default 5
    seed : int, optional

    Returns
    -------
    list of (train_idx, test_idx)
        ``n_splits`` pairs of index arrays; every subject is in exactly one
        test fold.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")
    E = np.asarray(event_indicator)
    rng = np.random.default_rng(seed)

    test_parts: list[list[NDArray[np.intp]]] = [[] for _ in range(n_splits)]
    for cls in np.unique(E):
        idx = np.flatnonzero(E == cls)
        rng.shuffle(idx)
        for k, chunk in enumerate(np.array_split(idx, n_splits)):
            test_parts[k].append(chunk)

    folds = []
    for parts in test_parts:
        test_idx = np.concatenate(parts)
        rng.shuffle(test_idx)
        train_idx = np.setdiff1d(np.arange(E.shape[0]), test_idx)
        rng.shuffle(train_idx)
        folds.append((train_idx, test_idx))
    return folds
