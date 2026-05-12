"""Errors raised by tausurv.datasets loaders."""

from __future__ import annotations


class UnknownDatasetError(ValueError):
    """Requested dataset name is not in the registry."""

    def __init__(self, name: str, available: list[str]) -> None:
        super().__init__(f"unknown dataset {name!r}; available: {sorted(available)}")
        self.name = name
        self.available = available


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
            f"{name!r} cannot be auto-fetched. "
            f"Use prepare_{name}(path=...).\n\n{instructions}"
        )
        self.name = name
        self.instructions = instructions
