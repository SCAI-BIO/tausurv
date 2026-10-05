"""Minimal ARFF reader for the scikit-survival data bundle.

ARFF appears here for one reason: the Hosmer & Lemeshow cohorts (WHAS500,
ACTG320) are distributed in it and in no other machine-readable form that is
both public and stable. The format is small enough that reading it directly
costs less than a dependency -- a header of ``@attribute`` declarations, then
comma-separated rows, with ``?`` for missing.

Nominal attributes whose levels are all numeric (``{0,1}``, ``{70,80,90,100}``)
are returned as numbers rather than strings: they are numeric codes that ARFF
happens to declare as an enumeration, and reading them as text would push
every event indicator through a needless cast.
"""

from __future__ import annotations

import csv
from pathlib import Path

import polars as pl

_MISSING = "?"


def _is_numeric(values: list[str]) -> bool:
    for v in values:
        try:
            float(v)
        except ValueError:
            return False
    return bool(values)


def _parse_attribute(line: str) -> tuple[str, str]:
    """``@attribute cd4 numeric`` -> ``("cd4", "numeric")``; nominal keeps braces."""
    rest = line.split(None, 1)[1].strip()
    if rest.startswith(("'", '"')):
        quote = rest[0]
        end = rest.index(quote, 1)
        return rest[1:end], rest[end + 1 :].strip()
    parts = rest.split(None, 1)
    return parts[0], (parts[1].strip() if len(parts) > 1 else "")


def read_arff(path: Path) -> pl.DataFrame:
    """Read an ARFF file into a :class:`polars.DataFrame`.

    Numeric and numeric-nominal attributes become ``Float64``; other nominal
    and string attributes become ``String``. ``?`` becomes null in every
    column. Only the dense format is supported -- the sparse ``{index value}``
    form is not used by any registered dataset.
    """
    names: list[str] = []
    kinds: list[str] = []
    rows: list[list[str]] = []
    in_data = False

    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            if in_data:
                rows.extend(csv.reader([line]))
                continue
            lowered = line.lower()
            if lowered.startswith("@attribute"):
                name, kind = _parse_attribute(line)
                names.append(name)
                kinds.append(kind)
            elif lowered.startswith("@data"):
                in_data = True

    if not names:
        raise ValueError(f"{path}: no @attribute declarations found")

    columns: dict[str, pl.Series] = {}
    for i, (name, kind) in enumerate(zip(names, kinds, strict=True)):
        values = [(row[i].strip() if i < len(row) else _MISSING) for row in rows]
        cleaned = [None if v in (_MISSING, "") else v for v in values]
        present = [v for v in cleaned if v is not None]
        numeric = kind.lower().startswith(("numeric", "real", "integer")) or (
            kind.startswith("{") and _is_numeric(present)
        )
        series = pl.Series(name, cleaned, dtype=pl.String)
        columns[name] = series.cast(pl.Float64, strict=False) if numeric else series

    return pl.DataFrame(columns)
