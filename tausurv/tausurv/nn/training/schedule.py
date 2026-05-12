from __future__ import annotations

import math


def get_lr(schedule: str, step: int, total_steps: int, base_lr: float) -> float:
    """Learning-rate schedule.

    Parameters
    ----------
    schedule : {"constant", "cosine", "linear_warmup_cosine"}
        "constant" — base_lr throughout.
        "cosine" — half-period cosine decay from base_lr to 0.
        "linear_warmup_cosine" — 5% warmup, then cosine decay.
    step : int — current step (0-indexed).
    total_steps : int
    base_lr : float
    """
    if total_steps <= 1:
        return base_lr
    progress = step / total_steps
    if schedule == "constant":
        return base_lr
    if schedule == "cosine":
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    if schedule == "linear_warmup_cosine":
        warmup_frac = 0.05
        warmup_steps = max(1, int(total_steps * warmup_frac))
        if step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps
        decay_progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return base_lr * 0.5 * (1 + math.cos(math.pi * decay_progress))
    raise ValueError(
        f"schedule must be 'constant', 'cosine', or 'linear_warmup_cosine', got {schedule!r}"
    )
