"""Guards shared by every model checkpoint."""

from __future__ import annotations

from pathlib import Path


def check_model_name(saved: str, cls: type, path: Path) -> None:
    """Reject loading a checkpoint written by a different model class.

    Guards against silently wrong math: ``WeibullAFT.load`` on a
    ``LogNormalAFT`` checkpoint would fit the state but change the
    distribution.
    """
    if saved != cls.__name__:
        raise ValueError(
            f"checkpoint at {path} was written by {saved}; "
            f"load it with {saved}.load, not {cls.__name__}.load"
        )


def check_format_version(saved: object, supported: int, path: Path) -> None:
    """Reject a checkpoint written in a format this version cannot read."""
    if saved != supported:
        raise ValueError(
            f"checkpoint at {path} has format version {saved!r}; "
            f"this tausurv reads version {supported}"
        )
