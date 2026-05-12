from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _default_coefficients(n_features: int) -> NDArray[np.float64]:
    """Alternating-sign coefficients with decreasing magnitude."""
    return np.array([(-1) ** i * (0.5 - 0.05 * i) for i in range(n_features)])
