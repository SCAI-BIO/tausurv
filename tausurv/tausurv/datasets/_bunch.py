"""SurvivalBunch — the container returned by every tausurv.datasets loader."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SurvivalBunch:
    r"""Loaded survival dataset.

    Returned by :func:`tausurv.datasets.load_dataset` and the per-dataset
    ``load_<name>`` shortcuts. Holds the design matrix as a strictly typed
    :class:`polars.DataFrame` (categoricals and nulls preserved), the
    observed time and event indicator as NumPy arrays, and the dataset's
    metadata.

    Single-event and competing-risks data share this container. For
    competing risks, ``cause`` carries the multi-class outcome and
    ``event_indicator`` is the derived binary ``(cause > 0)`` — so
    single-event-style operations (Kaplan-Meier, "any-cause" Cox) still
    work on competing-risks data via the standard ``(X, event_time,
    event_indicator)`` triple. Cause-aware estimators (Aalen-Johansen,
    Fine-Gray) read ``ds.cause`` directly.

    Parameters
    ----------
    X : polars.DataFrame, shape (n, d)
        Covariates with original dtypes preserved. Encoding of categoricals
        (``X.to_dummies(...)``, ordinal, target) is left to the caller.
    event_time : (n,) float64 array
        Observed time $Y_i = \min(T_i, C_i)$.
    event_indicator : (n,) int8 array
        $\delta_i = 1$ if any event was observed, $0$ if censored. For
        competing-risks datasets this is derived as ``(cause > 0)``.
    feature_names : tuple of str
        Column names of ``X``.
    name, description, citation, license, url : str
        Dataset metadata.
    cause : (n,) int8 array, optional
        Cause of event for competing-risks data: $0$ for censored,
        $1, \dots, K$ for each cause. ``None`` for single-event datasets.
    n_causes : int, default 1
        Number of competing causes. ``1`` for single-event.
    cause_labels : tuple of str, optional
        Human-readable label per cause, in the order indexed by ``cause``
        (label ``[0]`` is for cause ``1``, label ``[1]`` for cause ``2``,
        ...). ``None`` for single-event or unlabelled datasets.
    time_unit : str, optional
        Units of ``event_time`` (``"days"``, ``"hours"``, ``"months"``,
        ``"weeks"``, ``"cycles"``, ...). Used for axis labels in plots
        and human-readable display. ``None`` if not declared.
    tags : tuple of str
        Categorical labels for discovery: ``"clinical"``, ``"reliability"``,
        ``"competing-risks"``, ``"recidivism"``, ... A dataset can carry
        several (e.g. ``("clinical", "competing-risks")``).
    """

    X: pl.DataFrame
    event_time: NDArray[np.float64]
    event_indicator: NDArray[np.int8]
    feature_names: tuple[str, ...]
    name: str
    description: str
    citation: str
    license: str
    url: str
    cause: NDArray[np.int8] | None = None
    n_causes: int = 1
    cause_labels: tuple[str, ...] | None = None
    time_unit: str | None = None
    tags: tuple[str, ...] = ()

    @property
    def n(self) -> int:
        """Number of subjects (length of ``event_time``)."""
        return int(self.event_time.size)

    @property
    def d(self) -> int:
        """Number of covariate columns (zero is valid: e.g. ``genfan``)."""
        return self.X.width

    @property
    def is_competing_risks(self) -> bool:
        """True iff this dataset carries cause-of-event information."""
        return self.cause is not None

    def __iter__(self) -> Iterator[Any]:
        """Tuple-unpack as ``(X, event_time, event_indicator)``.

        Strict for both single-event and competing-risks data. Callers
        doing cause-aware analysis read ``ds.cause`` explicitly.
        """
        yield self.X
        yield self.event_time
        yield self.event_indicator
