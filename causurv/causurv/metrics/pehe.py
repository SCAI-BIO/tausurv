r"""HTE evaluation metrics: PEHE and ATE error.

The headline metric for heterogeneous-effect models is **PEHE**
(Precision in Estimation of Heterogeneous Effect, Hill 2011):

$$
\text{PEHE}(t) = \sqrt{\mathbb{E}\big[(\hat\tau(X, t) - \tau(X, t))^2\big]}.
$$

For survival/CIF contrasts the effect is a function of time, so PEHE is
naturally pointwise; Curth et al. (2021) report **integrated PEHE**
$\int_0^\tau \text{PEHE}(t)\,dt$ as a scalar summary.

ATE error tracks bias of the *population mean* and is included for
completeness — a learner with low PEHE always has low ATE error, but
the reverse is not true (a learner that misses the *heterogeneity*
while getting the mean right looks fine on ATE alone).

All functions accept either raw arrays or :class:`HTEEstimates` —
when both inputs are :class:`HTEEstimates`, the contrast and time
grids are checked for consistency.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from causurv.predictor import HTEEstimates

HTELike = Union[HTEEstimates, ArrayLike]


def pehe(hte_hat: HTELike, hte_true: HTELike) -> NDArray[np.float64] | float:
    r"""Root-PEHE across subjects.

    Parameters
    ----------
    hte_hat, hte_true : :class:`HTEEstimates` or array
        Per-subject treatment-effect estimates and ground truth. Shapes
        must match exactly. Typical shapes:

        - ``(n, T)`` for pointwise-in-time contrasts (``survival_diff``,
          ``cif_diff``, …) — returns a ``(T,)`` array.
        - ``(n,)`` for time-collapsed contrasts (``rmst_diff``) — returns
          a scalar.

    Returns
    -------
    PEHE on the same time grid as the inputs.

    Notes
    -----
    When both inputs are :class:`HTEEstimates`, their ``contrast`` and
    ``times`` are checked for consistency. Mismatched contrasts almost
    always mean a benchmark bug — PEHE between e.g. a ``survival_diff``
    and a ``cif_diff`` would be meaningless.
    """
    hat, true = _coerce_pair(hte_hat, hte_true)
    return np.sqrt(np.mean((hat - true) ** 2, axis=0))


def integrated_pehe(
    hte_hat: HTELike,
    hte_true: HTELike,
    times: ArrayLike | None = None,
    *,
    horizon: float | None = None,
) -> float:
    r"""Time-integrated PEHE, $\int_0^\tau \mathrm{PEHE}(t)\,dt$.

    Trapezoidal integration over the supplied time grid (or the grid
    carried on the :class:`HTEEstimates` inputs). For survival-scale
    contrasts this gives a single scalar summary of an HTE learner's
    accuracy across the follow-up window.

    Parameters
    ----------
    hte_hat, hte_true : :class:`HTEEstimates` or ``(n, T)`` array
        Must be 2D (pointwise contrasts only).
    times : ``(T,)`` array, optional
        Required if neither input carries ``times``.
    horizon : float, optional
        Integrate only up to ``t <= horizon``. Defaults to the full grid.

    Returns
    -------
    Scalar integrated PEHE.
    """
    hat, true = _coerce_pair(hte_hat, hte_true)
    if hat.ndim != 2:
        raise ValueError(
            f"integrated_pehe requires pointwise (n, T) inputs; got shape {hat.shape}"
        )
    times_arr = _resolve_times(times, hte_hat, hte_true)
    if times_arr.shape != (hat.shape[1],):
        raise ValueError(
            f"times has length {times_arr.shape[0]} but HTE has T={hat.shape[1]}"
        )
    pointwise = pehe(hat, true)
    if horizon is not None:
        if horizon < times_arr[0]:
            raise ValueError(
                f"horizon={horizon} is below the start of times ({times_arr[0]})"
            )
        keep = times_arr <= horizon
        pointwise = pointwise[keep]
        times_arr = times_arr[keep]
    return float(np.trapezoid(pointwise, x=times_arr))


def ate_error(
    hte_hat: HTELike, hte_true: HTELike
) -> NDArray[np.float64] | float:
    r"""Absolute error on the *population-mean* treatment effect.

    $$\big|\,\overline{\hat\tau}(t) - \overline\tau(t)\,\big|,$$
    where the bar denotes the empirical mean over the sample. Returns
    a ``(T,)`` array for pointwise contrasts, scalar for time-collapsed.
    """
    hat, true = _coerce_pair(hte_hat, hte_true)
    return np.abs(hat.mean(axis=0) - true.mean(axis=0))


def _as_array(x: HTELike) -> NDArray[np.float64]:
    if isinstance(x, HTEEstimates):
        return np.asarray(x.values, dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def _coerce_pair(
    a: HTELike, b: HTELike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    arr_a = _as_array(a)
    arr_b = _as_array(b)
    if arr_a.shape != arr_b.shape:
        raise ValueError(
            f"hte_hat and hte_true must have the same shape; "
            f"got {arr_a.shape} vs {arr_b.shape}"
        )
    if isinstance(a, HTEEstimates) and isinstance(b, HTEEstimates):
        if a.contrast != b.contrast:
            raise ValueError(
                f"contrast mismatch: hte_hat is {a.contrast!r} but "
                f"hte_true is {b.contrast!r} — PEHE across different "
                f"contrasts is not meaningful"
            )
        if a.times is not None and b.times is not None and not np.array_equal(
            np.asarray(a.times), np.asarray(b.times)
        ):
            raise ValueError(
                "time grids disagree between hte_hat and hte_true"
            )
    return arr_a, arr_b


def _resolve_times(
    times: ArrayLike | None, *hte_inputs: HTELike
) -> NDArray[np.float64]:
    if times is not None:
        return np.asarray(times, dtype=np.float64)
    for h in hte_inputs:
        if isinstance(h, HTEEstimates) and h.times is not None:
            return np.asarray(h.times, dtype=np.float64)
    raise ValueError(
        "integrated_pehe needs `times`, or HTEEstimates inputs carrying `.times`"
    )
