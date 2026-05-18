"""Task-level concurrency limit construction and stream buffer sizing."""

from typing import Literal

import anyio

from sieval.core.utils.concurrency import CompositeLimiter

from .consts import TaskAction


def compute_stream_buffer_capacity(
    record_each_stage: bool, global_limit: int | None, per_stage: dict[str, int]
) -> int:
    """Calculate memory stream buffer size from concurrency config.

    Heuristic: base * msg_multiplier * feedback_amp * safety + 16, floored at 32.
    """
    base = global_limit or (max(per_stage.values()) if per_stage else 32)
    msg_per_completion = 2 if record_each_stage else 1
    feedback_amp = 1.25 if "feedback" in per_stage else 1.1
    safety = 2 if base < 64 else 1.5
    cap = int(base * msg_per_completion * feedback_amp * safety) + 16
    return max(cap, 32)


def _validate_limit(name: str, v: int):
    if v < 1:
        raise ValueError(f"{name} must be >=1")


def prepare_limiters(
    global_limit: int | None, stage_limits: dict[str, int]
) -> tuple[anyio.CapacityLimiter | None, dict[str, anyio.CapacityLimiter]]:
    """Build a global limiter and per-stage limiters from concurrency config.

    Stage limiters are capped to *global_limit* when present.
    """
    if global_limit is not None:
        _validate_limit("concurrency_limit", global_limit)
    for k, v in stage_limits.items():
        _validate_limit(f"stage concurrency '{k}'", v)
    global_limiter = anyio.CapacityLimiter(global_limit) if global_limit else None
    stage_limiters: dict[str, anyio.CapacityLimiter] = {}
    for action in (
        TaskAction.PREPROCESS,
        TaskAction.INFER,
        TaskAction.POSTPROCESS,
        TaskAction.FEEDBACK,
    ):
        lim = stage_limits.get(action.value)
        if lim:
            eff = lim if global_limit is None else min(lim, global_limit)
            stage_limiters[action.value] = anyio.CapacityLimiter(eff)
    return global_limiter, stage_limiters


def get_limiter_for(
    action: (
        TaskAction | Literal["preprocess", "infer", "postprocess", "feedback"] | None
    ),
    global_limiter: anyio.CapacityLimiter | None,
    stage_limiters: dict[str, anyio.CapacityLimiter],
) -> CompositeLimiter:
    """Return a CompositeLimiter combining global + stage limits for *action*.

    Returns an unconstrained limiter when *action* is ``None``.
    """
    if not action:
        return CompositeLimiter()
    key = action.value if isinstance(action, TaskAction) else action
    # Enforce BOTH the global limit AND the specific stage limit simultaneously
    return CompositeLimiter(global_limiter, stage_limiters.get(key))


def min_limit(a: int | None, b: int | None) -> int | None:
    """Return the stricter (smaller) of two optional concurrency limits.

    ``None`` is treated as unbounded, so it never wins over a finite value.
    If both are ``None``, ``None`` is returned.
    """
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)
