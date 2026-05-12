from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Side = Literal["right", "left"]


@dataclass(frozen=True, slots=True)
class StepFunction:
    r"""Right- or left-continuous step function over a sorted time grid.

    Canonical I/O type for nonparametric estimators (Kaplan-Meier survival,
    Nelson-Aalen cumulative hazard, Aalen-Johansen CIF) and anything that
    evaluates them at arbitrary times. The default continuity matches the
    quantity ($\hat S$ and $\hat G$ are right-continuous; $\hat F$ from
    Aalen-Johansen is too), and the caller can request the left limit on a
    per-call basis — IPCW estimators commonly need $\hat G(t^-)$.

    Below the first grid time, returns ``baseline`` — ``1.0`` for survival,
    ``0.0`` for cumulative hazard and CIF.

    Parameters
    ----------
    time : (k,) float array
        Sorted, strictly increasing grid points.
    value : (k,) float array
        Step value at each grid point, after any drop or jump occurring there.
    side : "right" | "left", default "right"
        Default continuity used when called without an override.
    baseline : float, default 1.0
        Value returned for queries below ``time[0]``.
    """

    time: NDArray[np.float64]
    value: NDArray[np.float64]
    side: Side = "right"
    baseline: float = 1.0

    def __call__(
        self, query: ArrayLike, *, side: Side | None = None
    ) -> NDArray[np.float64]:
        s = side if side is not None else self.side
        if s not in ("right", "left"):
            raise ValueError(f"side must be 'right' or 'left', got {s!r}")
        q = np.asarray(query, dtype=np.float64)
        idx = np.searchsorted(self.time, q, side=s) - 1
        return np.where(idx >= 0, self.value[np.maximum(idx, 0)], self.baseline)
