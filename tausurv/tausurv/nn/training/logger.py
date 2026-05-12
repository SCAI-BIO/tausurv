from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import IO, Any


class Logger:
    r"""Live training logger with an in-place progress bar.

    Three sinks, all flushed on every write:

    - ``out_dir/train.jsonl`` — one JSON record per event (``train``, ``val``,
      ``checkpoint``, ``message``).
    - ``out_dir/train.log`` — one flat human-readable line per event; safe to
      ``tail -f``.
    - Stdout — when attached to a tty, a single line is overwritten in place
      with an ASCII bar showing ``epoch/total_epochs`` plus current train
      metrics. Validation, checkpoints and messages clear the bar line,
      print on their own line, and the next ``log_train`` redraws the bar.
      When stdout is not a tty (piped to a file, CI logs) or when
      ``use_progress_bar=False``, this falls back to one flat line per event.

    Parameters
    ----------
    out_dir : str or Path or None
        Files are only written when set.
    verbose : bool, default True
        Suppresses all stdout output when False.
    total_epochs : int, default 0
        Used to render the bar's fill fraction. ``0`` disables the fill.
    use_progress_bar : bool or None, default None
        ``None`` auto-detects from ``sys.stdout.isatty()``. Set ``False`` to
        force flat per-line output; ``True`` to force the bar (useful in tests).
    """

    def __init__(
        self,
        out_dir: str | Path | None = None,
        *,
        verbose: bool = True,
        total_epochs: int = 0,
        use_progress_bar: bool | None = None,
    ) -> None:
        self.verbose = verbose
        self.total_epochs = total_epochs
        self._t0 = time.time()
        self._jsonl: IO | None = None
        self._log: IO | None = None
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            self._jsonl = open(out_dir / "train.jsonl", "a")
            self._log = open(out_dir / "train.log", "a")

        if use_progress_bar is None:
            use_progress_bar = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._use_bar: bool = verbose and use_progress_bar
        self._last_bar_len = 0

    def log_train(self, *, epoch: int, **fields: Any) -> None:
        record = {"event": "train", "epoch": epoch, **fields}
        self._write_jsonl(record)
        line = self._format_train(epoch, fields)
        self._write_log(line)
        if self._use_bar:
            self._draw_bar(epoch, fields)
        elif self.verbose:
            self._print_line(line)

    def log_val(self, *, epoch: int, **fields: Any) -> None:
        record = {"event": "val", "epoch": epoch, **fields}
        self._write_jsonl(record)
        line = self._format_val(epoch, fields)
        self._write_log(line)
        self._emit_above_bar(line)

    def log_checkpoint(
        self, *, epoch: int, path: str | Path, kind: str = "step"
    ) -> None:
        path = str(path)
        record = {"event": "checkpoint", "epoch": epoch, "path": path, "kind": kind}
        self._write_jsonl(record)
        line = f"  saved [{kind}] -> {path}"
        self._write_log(line)
        self._emit_above_bar(line)

    def message(self, msg: str) -> None:
        self._write_jsonl({"event": "message", "msg": msg})
        self._write_log(msg)
        self._emit_above_bar(msg)

    def close(self) -> None:
        if self._use_bar and self._last_bar_len > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._last_bar_len = 0
        if self._jsonl is not None:
            self._jsonl.close()
            self._jsonl = None
        if self._log is not None:
            self._log.close()
            self._log = None

    def _write_jsonl(self, record: dict[str, Any]) -> None:
        if self._jsonl is None:
            return
        record.setdefault("elapsed", round(time.time() - self._t0, 2))
        self._jsonl.write(json.dumps(record) + "\n")
        self._jsonl.flush()

    def _write_log(self, line: str) -> None:
        if self._log is None:
            return
        self._log.write(line + "\n")
        self._log.flush()

    def _print_line(self, line: str) -> None:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def _emit_above_bar(self, line: str) -> None:
        """Write ``line`` on its own row, leaving the bar slot free underneath."""
        if not self.verbose:
            return
        if self._use_bar:
            self._clear_bar()
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
            self._last_bar_len = 0
        else:
            self._print_line(line)

    def _draw_bar(self, epoch: int, fields: dict[str, Any]) -> None:
        cur = epoch + 1
        total = max(self.total_epochs, 1)
        bar_len = 24
        filled = min(bar_len, max(0, int(round(bar_len * cur / total))))
        bar = (
            "=" * max(0, filled - 1)
            + (">" if filled < bar_len else "=")
            + " " * (bar_len - filled)
        )
        if self.total_epochs > 0:
            head = f"[{bar}] {cur}/{self.total_epochs}"
        else:
            head = f"[epoch {cur}]"
        parts = [head] + [f"{k}={_fmt(v)}" for k, v in fields.items()]
        line = "  ".join(parts)
        pad = max(0, self._last_bar_len - len(line))
        sys.stdout.write("\r" + line + (" " * pad))
        sys.stdout.flush()
        self._last_bar_len = len(line)

    def _clear_bar(self) -> None:
        if self._last_bar_len > 0:
            sys.stdout.write("\r" + " " * self._last_bar_len + "\r")
            sys.stdout.flush()

    @staticmethod
    def _format_train(epoch: int, fields: dict[str, Any]) -> str:
        parts = [f"[epoch {epoch + 1}]"] + [f"{k}={_fmt(v)}" for k, v in fields.items()]
        return "  ".join(parts)

    @staticmethod
    def _format_val(epoch: int, fields: dict[str, Any]) -> str:
        parts = [f"[epoch {epoch + 1}] val"] + [
            f"{k}={_fmt(v)}" for k, v in fields.items()
        ]
        return "  ".join(parts)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if v != 0 and abs(v) < 1e-3:
            return f"{v:.2e}"
        return f"{v:.4g}"
    return str(v)
