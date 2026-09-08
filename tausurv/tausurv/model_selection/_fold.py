from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray


class Fold(NamedTuple):
    """One side of a split, handed to scorers as the training fold."""

    X: NDArray[Any]
    event_time: NDArray[np.float64]
    event_indicator: NDArray[Any]
