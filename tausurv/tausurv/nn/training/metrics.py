from __future__ import annotations

from collections import deque


class Metrics:
    """Rolling-window statistics tracker.

    Holds the last ``window`` values per metric; ``snapshot()`` returns the
    current mean of each.
    """

    def __init__(self, window: int = 128) -> None:
        self.window = window
        self._data: dict[str, deque[float]] = {}

    def push(self, **values: float) -> None:
        for k, v in values.items():
            if k not in self._data:
                self._data[k] = deque(maxlen=self.window)
            self._data[k].append(float(v))

    def snapshot(self) -> dict[str, float]:
        return {k: sum(vals) / len(vals) for k, vals in self._data.items() if vals}
