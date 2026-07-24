"""Errors raised by tausurv.datasets loaders."""

from __future__ import annotations

import difflib


class UnknownDatasetError(ValueError):
    """Requested dataset name is not in the registry.

    Carries near-miss suggestions, which matter more here than in most
    registries: several studies are known by more than one name in the
    literature (``gbsg2`` for ``gbsg``, ``actg320`` for ``aids``), and the
    variant suffix is easy to misremember.
    """

    def __init__(self, name: str, available: list[str]) -> None:
        close = difflib.get_close_matches(name, available, n=4, cutoff=0.6)
        base = name.partition(":")[0]
        siblings = sorted(
            n for n in available if n == base or n.startswith(f"{base}:")
        )
        hint = ""
        if siblings and name not in siblings:
            hint = f"\n{base!r} exists with variants: {siblings}"
        elif close:
            hint = f"\ndid you mean: {close}?"
        super().__init__(
            f"unknown dataset {name!r} ({len(available)} registered)."
            f"{hint}\nlist_datasets() enumerates them all."
        )
        self.name = name
        self.available = available
        self.suggestions = tuple(close)


class DatasetIntegrityError(RuntimeError):
    """Downloaded bytes did not match the expected SHA256."""

    def __init__(self, name: str, expected: str, actual: str) -> None:
        super().__init__(
            f"sha256 mismatch for {name!r}: expected {expected}, got {actual}"
        )
        self.name = name
        self.expected = expected
        self.actual = actual


class OfflineModeError(RuntimeError):
    """TAUSURV_OFFLINE is set but the dataset is not in the cache."""

    def __init__(self, name: str, would_fetch: str) -> None:
        super().__init__(
            f"TAUSURV_OFFLINE is set and {name!r} is not cached; "
            f"would have fetched {would_fetch}"
        )
        self.name = name
        self.would_fetch = would_fetch


class CredentialedDatasetError(RuntimeError):
    """Dataset requires credentials the caller has not provided."""

    def __init__(self, name: str, instructions: str) -> None:
        super().__init__(f"{name!r} requires credentials.\n\n{instructions}")
        self.name = name
        self.instructions = instructions


class UserProvidedDatasetError(RuntimeError):
    """Dataset is redistribution-restricted; caller must supply a local path."""

    def __init__(self, name: str, instructions: str) -> None:
        super().__init__(
            f"{name!r} cannot be auto-fetched, because its data-use agreement "
            f"forbids redistribution. Obtain it yourself, then pass the "
            f"extracted directory:\n\n"
            f"    load_dataset({name!r}, path=...)\n\n{instructions}"
        )
        self.name = name
        self.instructions = instructions


class MissingDependencyError(ImportError):
    """Parser needs an optional dependency that is not installed."""

    def __init__(self, name: str, module: str, extra: str) -> None:
        super().__init__(
            f"{name!r} needs the optional dependency {module!r}.\n\n"
            f"    pip install 'tausurv[{extra}]'      # or: pip install {module}"
        )
        self.name = name
        self.module = module
        self.extra = extra
