from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch


class PretrainedMixin:
    r"""Save/load a model's typed config + weights together.

    Subclasses declare a dataclass :attr:`config_class` and store an instance
    of it on ``self.config``. :meth:`save_pretrained` writes
    ``config.json`` + ``model.pt`` to a directory and
    :meth:`from_pretrained` reconstructs the model by deserializing the
    config and loading the state dict.

    The dataclass is the single source of truth for what defines a model —
    constructor arguments, save/load schema, and IDE/type checking all
    derive from it.

    Example
    -------
    >>> model = DeepSurv(in_features=5, hidden_dim=64)
    >>> # ... train ...
    >>> model.save_pretrained("runs/deepsurv-v1")
    >>> restored = DeepSurv.from_pretrained("runs/deepsurv-v1")

    The ``Trainer`` automatically calls :meth:`save_pretrained` alongside its
    own ``state.pt`` checkpoint when ``out_dir`` is set, so every ``best/``,
    ``final/``, and ``step-NNNNNN/`` directory is both resume-compatible and
    ``from_pretrained``-compatible.
    """

    #: Dataclass type used to validate / round-trip the config. Set by subclass.
    config_class: ClassVar[type]

    #: The model's typed config instance.
    config: Any

    def save_pretrained(self, path: str | Path) -> None:
        """Write ``config.json`` and ``model.pt`` to ``path``."""
        if not is_dataclass(self.config):
            raise TypeError(
                f"{type(self).__name__}.config must be a dataclass instance "
                f"(got {type(self.config).__name__})"
            )
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps(asdict(self.config), indent=2))
        torch.save(self.state_dict(), path / "model.pt")

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "PretrainedMixin":
        """Reconstruct a model from a directory with ``config.json`` + ``model.pt``."""
        if not hasattr(cls, "config_class"):
            raise TypeError(
                f"{cls.__name__} must declare a `config_class` to use from_pretrained"
            )
        path = Path(path)
        config_dict = json.loads((path / "config.json").read_text())
        config = cls.config_class(**config_dict)
        model = cls(config=config)
        state = torch.load(path / "model.pt", weights_only=True)
        model.load_state_dict(state)
        return model
