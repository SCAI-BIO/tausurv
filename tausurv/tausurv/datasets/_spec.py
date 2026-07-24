"""Registry types: Access enum, DatasetSpec, DatasetInfo.

A dataset name is ``base`` or ``base:variant``. The base name identifies the
*study*; the variant identifies one processing of it. Variants exist because
the survival literature does not agree on a single canonical table for most
cohorts: PBC is analysed both as all 418 patients and as the 312 randomised
ones, FLCHAIN both whole and complete-case, GBSG both as the 686-patient
German trial and as the 2232-row Rotterdam combination. Each of those is a
legitimate published table, and reporting a number against the wrong one is
the single easiest way to draw a false comparison.

So every variant is its own :class:`DatasetSpec` with its own parser, and the
base name always resolves to the reading a survival textbook would call
default. Variants of a study share ``url``/``sha256`` with their base, and the
cache is content-addressed, so they cost one download between them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from tausurv.datasets._bunch import SurvivalBunch

#: Separator between a base dataset name and its variant.
VARIANT_SEP = ":"


class Access(StrEnum):
    """How a dataset is delivered to the user.

    OPEN
        Public URL, auto-fetched on first use.
    CREDENTIALED
        Requires user-supplied credentials (PhysioNet, BioLINCC, cBioPortal,
        ...). The loader raises with setup instructions; a retry succeeds
        once credentials are in place.
    USER_PROVIDED
        Redistribution is forbidden by the upstream license or data-use
        agreement (SEER, UNOS/OPTN, MIMIC, full METABRIC). The loader never
        attempts a download; the user passes their own extracted directory
        as ``load_dataset(name, path=...)``.
    """

    OPEN = "open"
    CREDENTIALED = "credentialed"
    USER_PROVIDED = "user_provided"


Parser = Callable[[Path, "DatasetSpec"], SurvivalBunch]


def split_name(name: str) -> tuple[str, str | None]:
    """``"pbc:randomised"`` -> ``("pbc", "randomised")``; ``"pbc"`` -> ``("pbc", None)``."""
    base, sep, variant = name.partition(VARIANT_SEP)
    return base, (variant if sep else None)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Registry entry for one dataset, or for one variant of one.

    Attributes:
        name: registry key, ``base`` or ``base:variant``.
        access: how the data is obtained; see :class:`Access`.
        license: license of the *data*, not of tausurv.
        citation: the work to cite when reporting on this dataset.
        description: what the cohort is, what the endpoint is, and any
            recoding the parser applies. Recoding must be stated here --
            it is the difference between two published numbers.
        parser: cached path (a file, or a directory for ``USER_PROVIDED``)
            to :class:`~tausurv.datasets._bunch.SurvivalBunch`.
        url: source URL. ``None`` for non-OPEN datasets.
        sha256: expected digest of the fetched bytes. ``None`` for non-OPEN.
        access_help: setup instructions, shown in the error raised for
            CREDENTIALED and USER_PROVIDED datasets. Required for those.
        tags: discovery labels (``"clinical"``, ``"reliability"``,
            ``"competing-risks"``, ``"benchmark"``, ...).
        time_unit: units of ``event_time``.
        requires: import name of an optional dependency the parser needs
            (``"h5py"``), or ``None``.
    """

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
    requires: str | None = None

    @property
    def base(self) -> str:
        """The study name, without the variant suffix."""
        return split_name(self.name)[0]

    @property
    def variant(self) -> str | None:
        """The variant suffix, or ``None`` for a canonical entry."""
        return split_name(self.name)[1]

    @property
    def is_variant(self) -> bool:
        """True iff this entry is a named processing of another study."""
        return self.variant is not None


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
    requires: str | None = None

    @property
    def base(self) -> str:
        """The study name, without the variant suffix."""
        return split_name(self.name)[0]

    @property
    def variant(self) -> str | None:
        """The variant suffix, or ``None`` for a canonical entry."""
        return split_name(self.name)[1]
