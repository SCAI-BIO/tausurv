"""The checkpoint format of the tree models on the Rust core.

A checkpoint is a directory with ``config.json``, the constructor
parameters plus the model name and a format version, and ``state.npz``,
the fitted forest or tree in the flat array form the core exports
(``to_arrays``) together with ``times_``. ``load`` rebuilds the core
handle through ``from_arrays``, which validates every array, so a
corrupt or foreign file fails loudly instead of predicting nonsense.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tausurv._checkpoint import check_format_version, check_model_name

FORMAT_VERSION = 1


def save(model: Any, path: str | Path, params: dict[str, Any]) -> None:
    """Write ``config.json`` and ``state.npz`` for a fitted tree model."""
    if not hasattr(model, "_handle"):
        raise RuntimeError(
            f"{type(model).__name__} is not fitted; call fit() before save()"
        )
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    config = {
        "model": type(model).__name__,
        "format_version": FORMAT_VERSION,
        **params,
    }
    (target / "config.json").write_text(json.dumps(config, indent=2))
    np.savez_compressed(
        target / "state.npz", times=model.times_, **model._handle.to_arrays()
    )


def load(
    cls: type[Any],
    path: str | Path,
    from_arrays: Any,
    param_names: tuple[str, ...],
) -> Any:
    """Rebuild a tree model from a directory written by `save`."""
    source = Path(path)
    config = json.loads((source / "config.json").read_text())
    check_model_name(config["model"], cls, source)
    check_format_version(config.get("format_version"), FORMAT_VERSION, source)
    model = cls(**{name: config[name] for name in param_names})
    with np.load(source / "state.npz") as state:
        model.times_ = np.asarray(state["times"], dtype=np.float64)
        arrays = {
            key: int(state[key]) if state[key].ndim == 0 else state[key]
            for key in state.files
            if key != "times"
        }
    model._handle = from_arrays(arrays)
    return model
