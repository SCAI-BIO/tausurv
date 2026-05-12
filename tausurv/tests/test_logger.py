from __future__ import annotations

import json

from tausurv.nn.training.logger import Logger


def test_progress_bar_overwrites_in_place(capsys):
    """Successive log_train calls write a \\r-prefixed line, not a newline."""
    logger = Logger(out_dir=None, verbose=True, total_epochs=4, use_progress_bar=True)
    logger.log_train(epoch=0, loss=1.0, lr=1e-2)
    logger.log_train(epoch=1, loss=0.8, lr=1e-2)
    logger.close()

    out = capsys.readouterr().out
    # Both updates land on the same row via carriage return.
    assert out.count("\r") >= 2
    # No newline between the two bar updates (only the final one on close).
    assert out.count("\n") == 1
    # Bar markers and fraction present.
    assert "1/4" in out
    assert "2/4" in out


def test_val_clears_bar_and_prints_on_new_line(capsys):
    logger = Logger(out_dir=None, verbose=True, total_epochs=3, use_progress_bar=True)
    logger.log_train(epoch=0, loss=1.0, lr=1e-2)
    logger.log_val(epoch=0, loss=0.9, harrell=0.72)
    logger.close()

    out = capsys.readouterr().out
    # Val output landed as a full standalone line.
    assert "[epoch 1] val" in out
    assert "harrell=0.72" in out
    # The val print is followed by a newline (so the next bar redraw goes
    # cleanly on its own row).
    assert "val  loss=0.9  harrell=0.72\n" in out


def test_checkpoint_and_message_clear_bar(capsys):
    logger = Logger(out_dir=None, verbose=True, total_epochs=2, use_progress_bar=True)
    logger.log_train(epoch=0, loss=1.0, lr=1e-2)
    logger.log_checkpoint(epoch=0, path="/tmp/best", kind="best")
    logger.message("custom note")
    logger.close()

    out = capsys.readouterr().out
    assert "saved [best] -> /tmp/best" in out
    assert "custom note" in out


def test_progress_bar_disabled_falls_back_to_flat_lines(capsys):
    """When the bar is off (e.g., piped output), each event is its own line."""
    logger = Logger(out_dir=None, verbose=True, total_epochs=3, use_progress_bar=False)
    logger.log_train(epoch=0, loss=1.0, lr=1e-2)
    logger.log_train(epoch=1, loss=0.8, lr=1e-2)
    logger.close()

    out = capsys.readouterr().out
    # No \r overwriting in flat mode.
    assert "\r" not in out
    # Two flat lines.
    assert out.count("\n") == 2


def test_verbose_false_silences_stdout_but_files_still_written(tmp_path, capsys):
    logger = Logger(out_dir=tmp_path, verbose=False, total_epochs=2)
    logger.log_train(epoch=0, loss=1.0, lr=1e-2)
    logger.log_val(epoch=0, loss=0.9)
    logger.close()

    assert capsys.readouterr().out == ""
    # JSONL has structured records.
    records = [
        json.loads(line)
        for line in (tmp_path / "train.jsonl").read_text().strip().splitlines()
    ]
    events = [r["event"] for r in records]
    assert events == ["train", "val"]
    # Each JSONL record carries elapsed time.
    assert all("elapsed" in r for r in records)
    # train.log has flat lines (no \r).
    log_text = (tmp_path / "train.log").read_text()
    assert "\r" not in log_text
    assert "[epoch 1]" in log_text
    assert "[epoch 1] val" in log_text


def test_log_files_never_contain_bar_control_chars(tmp_path):
    """Even with the bar enabled, file output stays flat."""
    logger = Logger(
        out_dir=tmp_path, verbose=True, total_epochs=3, use_progress_bar=True
    )
    logger.log_train(epoch=0, loss=1.0, lr=1e-2)
    logger.log_train(epoch=1, loss=0.8, lr=1e-2)
    logger.log_val(epoch=1, loss=0.9)
    logger.close()

    log_text = (tmp_path / "train.log").read_text()
    assert "\r" not in log_text
    # Two train rows, one val row.
    assert log_text.count("[epoch ") == 3
