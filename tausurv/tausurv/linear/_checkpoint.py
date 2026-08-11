from __future__ import annotations

from pathlib import Path


def _check_model_name(saved: str, cls: type, path: Path) -> None:
    """Reject loading a checkpoint written by a different model class.

    Guards against silently wrong math — e.g. ``WeibullAFT.load`` on a
    ``LogNormalAFT`` checkpoint would fit the state but change the
    distribution.
    """
    if saved != cls.__name__:
        raise ValueError(
            f"checkpoint at {path} was written by {saved}; "
            f"load it with {saved}.load, not {cls.__name__}.load"
        )
