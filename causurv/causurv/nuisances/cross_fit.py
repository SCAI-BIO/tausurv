r"""Nuisance fitting and K-fold cross-fitting.

Cross-fitting is the substrate orthogonal/DR learners need: the
Chernozhukov et al. (2018) double/debiased-ML theory only holds when
the final-stage estimator consumes nuisance predictions computed on
data the nuisance models never saw. Without it, nuisance-estimation
bias contaminates the final stage and the doubly-robust guarantee
disappears.

Two public helpers:

- :func:`fit_nuisances` — fit each nuisance on the full dataset. Use
  when you have pre-fit models or want to skip cross-fitting (S/T/X
  learners, which are not orthogonal, accept either).
- :func:`cross_fit` — K-fold cross-fitting producing both OOF
  predictions *and* full-data refits, returned together in a
  :class:`CrossFitNuisances`.

Cross-fits are stratified by treatment arm by default to prevent
degenerate folds with zero subjects in some arm.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from causurv.nuisances.types import CrossFitNuisances, Nuisances


def fit_nuisances(
    X: ArrayLike,
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    treatment: ArrayLike,
    *,
    outcome_factory: Callable[[], Any],
    propensity_factory: Callable[[], Any],
    censoring_factory: Callable[[], Any] | None = None,
) -> Nuisances:
    r"""Fit a nuisance bundle on the full dataset (no cross-fitting).

    Suitable for learners that don't require Neyman-orthogonality
    (S-learner, T-learner, X-learner). Orthogonal learners (R, DR,
    AIPW) should use :func:`cross_fit` instead.

    Parameters
    ----------
    X : ``(n, d)`` array
    event_time, event_indicator : ``(n,)`` arrays
    treatment : ``(n,)`` integer array
        Arm labels in ``{0, 1, ..., n_arms - 1}``.
    outcome_factory : callable
        Zero-argument factory returning a fresh tausurv
        :class:`SurvivalPredictor`. Called once per arm.
    propensity_factory : callable
        Zero-argument factory returning a fresh sklearn-style
        classifier exposing ``fit(X, y)`` and ``predict_proba(X)``.
    censoring_factory : callable, optional
        Factory for the censoring distribution. If ``None``, the
        returned :class:`Nuisances` has ``censoring=None``.

    Returns
    -------
    :class:`Nuisances`
    """
    X_arr, T_arr, E_arr, A_arr = _check_inputs(X, event_time, event_indicator, treatment)
    n_arms = int(A_arr.max()) + 1

    outcome_models: dict[int, Any] = {}
    for a in range(n_arms):
        mask = A_arr == a
        if not mask.any():
            continue
        outcome_models[a] = outcome_factory().fit(
            X_arr[mask], T_arr[mask], E_arr[mask]
        )

    propensity_model = propensity_factory().fit(X_arr, A_arr)

    censoring_model = None
    if censoring_factory is not None:
        # Censoring indicator: 1 - event_indicator (subjects who were censored
        # have the "censoring event" observed; events are censored from the
        # censoring perspective).
        censoring_model = censoring_factory().fit(X_arr, T_arr, 1 - E_arr)

    return Nuisances(
        outcome=outcome_models,
        propensity=propensity_model,
        censoring=censoring_model,
    )


def cross_fit(
    X: ArrayLike,
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    treatment: ArrayLike,
    *,
    outcome_factory: Callable[[], Any],
    propensity_factory: Callable[[], Any],
    censoring_factory: Callable[[], Any] | None = None,
    times: ArrayLike | None = None,
    n_folds: int = 5,
    stratify_by_arm: bool = True,
    seed: int = 0,
) -> CrossFitNuisances:
    r"""K-fold cross-fit nuisances; return OOF predictions + full-data refits.

    For each fold, fit the three nuisance functions on the other
    ``n_folds - 1`` folds and predict on the held-out fold. After all
    folds, every subject has an OOF prediction from a model that never
    saw them. Models are then refit on the full dataset and returned
    as :class:`Nuisances` for inference on new $X$.

    All three nuisances share the *same* fold assignment so that any
    joint orthogonal score downstream uses consistent OOF estimates.

    Parameters
    ----------
    X : ``(n, d)`` array
    event_time, event_indicator : ``(n,)`` arrays
    treatment : ``(n,)`` integer array
    outcome_factory, propensity_factory : callable
    censoring_factory : callable, optional
        Pass to enable IPCW-style adjustments downstream.
    times : ``(T,)`` array, optional
        Time grid for OOF outcome (and censoring) predictions. If
        ``None``, derived from the empirical event-time quantiles of
        the training data.
    n_folds : int, default ``5``
        Must be $\ge 2$.
    stratify_by_arm : bool, default ``True``
        Ensure each fold contains subjects from every arm. Off by
        default would risk fitting an outcome model with zero training
        subjects in some arm.
    seed : int, default ``0``

    Returns
    -------
    :class:`CrossFitNuisances`
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2; got {n_folds}")

    X_arr, T_arr, E_arr, A_arr = _check_inputs(X, event_time, event_indicator, treatment)
    n, _ = X_arr.shape
    n_arms = int(A_arr.max()) + 1

    times_arr = _resolve_times(times, T_arr, E_arr)
    T_grid = len(times_arr)

    rng = np.random.default_rng(seed)
    fold_assignment = _kfold_assignment(
        A_arr, n_folds=n_folds, stratify_by_arm=stratify_by_arm, rng=rng
    )

    oof_outcome = {a: np.full((n, T_grid), np.nan) for a in range(n_arms)}
    oof_propensity = np.full((n, n_arms), np.nan)
    oof_censoring = (
        np.full((n, T_grid), np.nan) if censoring_factory is not None else None
    )

    for k in range(n_folds):
        train_mask = fold_assignment != k
        test_mask = ~train_mask
        if not test_mask.any():
            continue

        # Outcome models — one per arm, fit on the arm's subset of the train fold.
        for a in range(n_arms):
            arm_train_mask = train_mask & (A_arr == a)
            if not arm_train_mask.any():
                raise ValueError(
                    f"fold {k} has no training subjects in arm {a}; "
                    f"try stratify_by_arm=True or more folds"
                )
            model_a = outcome_factory().fit(
                X_arr[arm_train_mask],
                T_arr[arm_train_mask],
                E_arr[arm_train_mask],
            )
            oof_outcome[a][test_mask] = np.asarray(
                model_a.predict_survival_function(X_arr[test_mask], times_arr),
                dtype=np.float64,
            )

        # Propensity — fit on the full train fold.
        prop = propensity_factory().fit(X_arr[train_mask], A_arr[train_mask])
        proba = np.asarray(prop.predict_proba(X_arr[test_mask]), dtype=np.float64)
        # Align to our arm-index convention (sklearn classes_ may differ).
        classes = np.asarray(getattr(prop, "classes_", np.arange(n_arms)))
        for col, arm in enumerate(classes):
            arm_idx = int(arm)
            if 0 <= arm_idx < n_arms:
                oof_propensity[test_mask, arm_idx] = proba[:, col]
        # Fill any missing arm columns with 0 (a class not seen in this fold).
        missing = np.isnan(oof_propensity[test_mask])
        if missing.any():
            slice_ = oof_propensity[test_mask]
            slice_[missing] = 0.0
            oof_propensity[test_mask] = slice_

        # Censoring — fit on the full train fold with the censoring indicator.
        if censoring_factory is not None:
            cens = censoring_factory().fit(
                X_arr[train_mask], T_arr[train_mask], 1 - E_arr[train_mask]
            )
            oof_censoring[test_mask] = np.asarray(
                cens.predict_survival_function(X_arr[test_mask], times_arr),
                dtype=np.float64,
            )

    # Sanity: every entry must have been filled.
    for a in range(n_arms):
        if np.isnan(oof_outcome[a]).any():
            raise RuntimeError(
                f"internal: arm {a} has unfilled OOF entries after cross-fit"
            )
    if np.isnan(oof_propensity).any():
        raise RuntimeError("internal: propensity OOF has unfilled entries")

    # Full-data refits for the inference-time Nuisances.
    full = fit_nuisances(
        X_arr,
        T_arr,
        E_arr,
        A_arr,
        outcome_factory=outcome_factory,
        propensity_factory=propensity_factory,
        censoring_factory=censoring_factory,
    )

    return CrossFitNuisances(
        outcome=full.outcome,
        propensity=full.propensity,
        censoring=full.censoring,
        oof_outcome=oof_outcome,
        oof_propensity=oof_propensity,
        oof_censoring=oof_censoring,
        times=times_arr,
        fold_assignment=fold_assignment,
        n_folds=n_folds,
    )


def _check_inputs(
    X: ArrayLike,
    event_time: ArrayLike,
    event_indicator: ArrayLike,
    treatment: ArrayLike,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int8],
    NDArray[np.int_],
]:
    X_arr = np.asarray(X, dtype=np.float64)
    T_arr = np.asarray(event_time, dtype=np.float64)
    E_arr = np.asarray(event_indicator, dtype=np.int8)
    A_arr = np.asarray(treatment, dtype=np.int_)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2D (n, d); got shape {X_arr.shape}")
    n = X_arr.shape[0]
    if not (T_arr.shape == (n,) and E_arr.shape == (n,) and A_arr.shape == (n,)):
        raise ValueError(
            "X, event_time, event_indicator, and treatment must share first axis; "
            f"got {X_arr.shape}, {T_arr.shape}, {E_arr.shape}, {A_arr.shape}"
        )
    if A_arr.min() < 0:
        raise ValueError(
            f"treatment values must be non-negative integers; min was {A_arr.min()}"
        )
    return X_arr, T_arr, E_arr, A_arr


def _resolve_times(
    times: ArrayLike | None,
    T: NDArray[np.float64],
    E: NDArray[np.int8],
) -> NDArray[np.float64]:
    if times is not None:
        return np.asarray(times, dtype=np.float64)
    # Default: 20 quantile-spaced bins over observed event times.
    events_T = T[E.astype(bool)]
    if events_T.size == 0:
        # No events at all — fall back to the full time range.
        return np.linspace(T.min(), T.max(), 20)
    qs = np.linspace(0.05, 0.95, 20)
    return np.unique(np.quantile(events_T, qs))


def _kfold_assignment(
    A: NDArray[np.int_],
    *,
    n_folds: int,
    stratify_by_arm: bool,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    r"""Assign each subject to a fold in ``range(n_folds)``.

    With ``stratify_by_arm=True``, the arm proportions inside each fold
    match the population proportions (up to rounding). This prevents
    folds with zero subjects in some arm — which would make the
    arm-specific outcome model un-fittable for that fold.
    """
    n = A.shape[0]
    fold = np.empty(n, dtype=np.int_)
    if not stratify_by_arm:
        # Round-robin on a shuffled index.
        perm = rng.permutation(n)
        fold[perm] = np.arange(n) % n_folds
        return fold

    # Stratified: per-arm round-robin on a shuffled per-arm index.
    for a in np.unique(A):
        idx = np.flatnonzero(A == a)
        perm = rng.permutation(idx)
        fold[perm] = np.arange(len(perm)) % n_folds
    return fold
