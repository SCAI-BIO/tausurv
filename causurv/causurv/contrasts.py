r"""Treatment-effect contrasts.

A "contrast" turns a pair (or tuple) of arm-specific potential outcomes
into a single per-subject effect. The same arithmetic works on survival
functions ($S_a(t \mid x)$) or cumulative incidence functions
($F_{a, k}(t \mid x)$) — the caller passes whichever scale matches the
contrast name.

Supported contrast names (locked in early so future learners don't
introduce a fragmented vocabulary):

- ``"survival_diff"`` — $S_a(t \mid x) - S_b(t \mid x)$. Positive ⇒ arm
  $a$ improves survival vs reference $b$.
- ``"survival_ratio"`` — $S_a / S_b$. $> 1$ ⇒ arm $a$ improves survival.
- ``"cif_diff"`` — $F_{a,k}(t \mid x) - F_{b,k}(t \mid x)$. Positive ⇒
  arm $a$ raises incidence of cause $k$. Requires ``cause`` to be set
  on the predictor.
- ``"cif_ratio"`` — $F_{a,k} / F_{b,k}$.
- ``"rmst_diff"`` — $\int_0^\tau S_a(t \mid x) - S_b(t \mid x) \, dt$ —
  difference in restricted mean survival time up to horizon $\tau$.
  Requires both ``times`` and ``horizon`` (the integration bound).

When we add hazard-ratio or other forms later, add them here — keep the
vocabulary in one place.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

ContrastName = Literal[
    "survival_diff",
    "survival_ratio",
    "cif_diff",
    "cif_ratio",
    "rmst_diff",
]

_VALID = ("survival_diff", "survival_ratio", "cif_diff", "cif_ratio", "rmst_diff")
_SURVIVAL_SCALE = {"survival_diff", "survival_ratio", "rmst_diff"}
_CIF_SCALE = {"cif_diff", "cif_ratio"}


def apply_contrast(
    arm_reference: NDArray[np.float64],
    arm_treatment: NDArray[np.float64],
    contrast: str,
    *,
    times: NDArray[np.float64] | None = None,
    horizon: float | None = None,
) -> NDArray[np.float64]:
    r"""Combine two arm-specific potential outcomes into a per-subject effect.

    Parameters
    ----------
    arm_reference, arm_treatment : ``(n, T)`` arrays
        Potential outcomes under the reference and treatment arms. For
        survival-scale contrasts (``survival_diff``, ``survival_ratio``,
        ``rmst_diff``) these are $S(t \mid x)$ values; for CIF-scale
        contrasts they are $F_k(t \mid x)$ values.
    contrast : one of :data:`_VALID`
    times : ``(T,)`` array, required for ``rmst_diff``
        The time grid both potential-outcome arrays were evaluated at.
    horizon : float, required for ``rmst_diff``
        Upper integration bound $\tau$ for the restricted mean survival
        time. Must be $\le t_{\max}$ of ``times``.

    Returns
    -------
    ``(n, T)`` for pointwise contrasts (survival/cif), ``(n,)`` for
    ``rmst_diff``.
    """
    if contrast not in _VALID:
        raise ValueError(
            f"contrast must be one of {_VALID}, got {contrast!r}"
        )
    if contrast == "survival_diff" or contrast == "cif_diff":
        return arm_treatment - arm_reference
    if contrast == "survival_ratio" or contrast == "cif_ratio":
        return arm_treatment / np.clip(arm_reference, 1e-12, None)
    if contrast == "rmst_diff":
        if times is None or horizon is None:
            raise ValueError("rmst_diff requires both `times` and `horizon`")
        times = np.asarray(times, dtype=np.float64)
        if horizon > float(times[-1]) + 1e-9:
            raise ValueError(
                f"horizon={horizon} exceeds max time {times[-1]} in `times`"
            )
        keep = times <= horizon
        diff = arm_treatment[:, keep] - arm_reference[:, keep]
        # Trapezoidal integration over the truncated grid (and the horizon endpoint).
        grid = np.concatenate([times[keep], [horizon]])
        # If the last kept point already equals horizon, don't double-count.
        if grid[-2] == grid[-1]:
            grid = grid[:-1]
            return np.trapezoid(diff, x=grid, axis=1)
        # Extend diff with the value at horizon (right-continuous step from
        # the last kept time — the survival functions are step-evaluated).
        diff_horizon = diff[:, -1:]
        return np.trapezoid(
            np.concatenate([diff, diff_horizon], axis=1), x=grid, axis=1
        )
    raise AssertionError("unreachable")  # pragma: no cover


def is_survival_scale(contrast: str) -> bool:
    """``True`` if the contrast operates on survival values $S(t|x)$
    (not on CIF values)."""
    return contrast in _SURVIVAL_SCALE


def is_cif_scale(contrast: str) -> bool:
    """``True`` if the contrast operates on CIF values $F_k(t|x)$."""
    return contrast in _CIF_SCALE
