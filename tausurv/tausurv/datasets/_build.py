"""Shared constructors every parser ends with.

Parsers differ only in how they find the endpoint; they agree on how to hand
it back. Keeping the two constructors here means metadata always travels from
the :class:`~tausurv.datasets._spec.DatasetSpec` and never gets retyped into a
parser, where it could drift.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from tausurv.datasets._bunch import SurvivalBunch
from tausurv.datasets._spec import DatasetSpec


def bunch_from(
    spec: DatasetSpec,
    X: pl.DataFrame,
    event_time: np.ndarray,
    event_indicator: np.ndarray,
) -> SurvivalBunch:
    """Single-event bunch."""
    return SurvivalBunch(
        X=X,
        event_time=np.asarray(event_time, dtype=np.float64),
        event_indicator=np.asarray(event_indicator, dtype=np.int8),
        feature_names=tuple(X.columns),
        name=spec.name,
        description=spec.description,
        citation=spec.citation,
        license=spec.license,
        url=spec.url or "",
        time_unit=spec.time_unit,
        tags=spec.tags,
    )


def bunch_from_competing(
    spec: DatasetSpec,
    X: pl.DataFrame,
    event_time: np.ndarray,
    cause: np.ndarray,
    *,
    n_causes: int,
    cause_labels: tuple[str, ...] | None,
) -> SurvivalBunch:
    """Competing-risks bunch; ``event_indicator`` is derived as ``cause > 0``."""
    cause = np.asarray(cause, dtype=np.int8)
    return SurvivalBunch(
        X=X,
        event_time=np.asarray(event_time, dtype=np.float64),
        event_indicator=(cause > 0).astype(np.int8),
        feature_names=tuple(X.columns),
        name=spec.name,
        description=spec.description,
        citation=spec.citation,
        license=spec.license,
        url=spec.url or "",
        cause=cause,
        n_causes=n_causes,
        cause_labels=cause_labels,
        time_unit=spec.time_unit,
        tags=spec.tags,
    )


def drop_present(df: pl.DataFrame, *names: str) -> pl.DataFrame:
    """Drop the named columns, ignoring any that are absent.

    Rdatasets exports carry a ``rownames`` index column for some packages and
    not others; this keeps the per-dataset parsers from having to care.
    """
    return df.drop([n for n in names if n in df.columns])
