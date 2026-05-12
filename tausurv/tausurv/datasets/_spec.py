"""Registry types: Access enum, DatasetSpec, DatasetInfo."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from tausurv.datasets._bunch import SurvivalBunch


class Access(StrEnum):
    """How a dataset is delivered to the user.

    OPEN
        Public URL, auto-fetched on first use.
    CREDENTIALED
        Requires user-supplied credentials (Kaggle API, PhysioNet, ...). The
        loader raises with setup instructions; a retry succeeds once
        credentials are in place.
    USER_PROVIDED
        Redistribution is forbidden by the upstream license (MIMIC, full
        METABRIC, ...). The loader never attempts a download; the user
        supplies an extracted directory via ``prepare_<name>(path=...)``.
    """

    OPEN = "open"
    CREDENTIALED = "credentialed"
    USER_PROVIDED = "user_provided"


Parser = Callable[[Path, "DatasetSpec"], SurvivalBunch]


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Registry entry for one dataset."""

    name: str
    access: Access
    license: str
    citation: str
    description: str
    parser: Parser
    url: str | None = None
    sha256: str | None = None
    access_help: str | None = None
    tags: tuple[str, ...] = ()
    time_unit: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    """Metadata-only view of a :class:`DatasetSpec` — no parser, no fetch."""

    name: str
    access: Access
    license: str
    citation: str
    description: str
    url: str | None
    sha256: str | None
    tags: tuple[str, ...] = ()
    time_unit: str | None = None
