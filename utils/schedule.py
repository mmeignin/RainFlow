import math


def cosine_warmup(step: int, warmup: int, total: int, min_lr_ratio: float = 0.1) -> float:
    """Linear warmup to 1.0 over `warmup` steps, then cosine decay to `min_lr_ratio` at `total` (held flat past `total`)."""
    if step < warmup:
        return step / max(warmup, 1)
    progress = min((step - warmup) / max(total - warmup, 1), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def linear_warmup(step: int, warmup: int, total: int) -> float:
    """Linear warmup to 1.0 over `warmup` steps, then flat. Matches torchcfm's own schedule."""
    del total
    if warmup <= 0:
        return 1.0
    return min(step, warmup) / warmup


def constant(step: int, warmup: int, total: int) -> float:
    """No warmup, no decay — LR stays at its configured value throughout."""
    del step, warmup, total
    return 1.0


SCHEDULES = {
    "cosine_warmup": cosine_warmup,
    "linear_warmup": linear_warmup,
    "constant": constant,
}


def build_schedule_fn(name: str):
    """Look up a schedule function by config name."""
    if name not in SCHEDULES:
        raise ValueError(f"unknown scheduler {name!r}; expected one of {sorted(SCHEDULES)}.")
    return SCHEDULES[name]
