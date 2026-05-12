from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_state(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save model + optimizer state + epoch counter to ``path/state.pt``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    if extra:
        state["extra"] = extra
    torch.save(state, path / "state.pt")


def load_state(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, dict[str, Any]]:
    """Load state from ``path/state.pt``. Returns ``(epoch, extra)``."""
    path = Path(path)
    state = torch.load(path / "state.pt", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return int(state["epoch"]), state.get("extra", {})


def prune_checkpoints(out_dir: str | Path, keep_last: int) -> None:
    """Keep only the last ``keep_last`` step-NNNNNN checkpoint directories.

    Never deletes ``best/``, ``final/``, ``interrupt/``.
    """
    out_dir = Path(out_dir)
    step_dirs = sorted(
        (
            p
            for p in out_dir.iterdir()
            if p.is_dir() and re.fullmatch(r"step-\d+", p.name)
        ),
        key=lambda p: int(p.name.split("-", 1)[1]),
    )
    for stale in step_dirs[:-keep_last]:
        for f in stale.iterdir():
            f.unlink()
        stale.rmdir()
