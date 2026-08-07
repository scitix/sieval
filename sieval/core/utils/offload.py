"""Run CPU-bound stage work off the event loop, in a worker process.

**Reach for ``anyio.to_thread.run_sync`` first** — that is the house pattern,
called directly at the site (``core/tasks/loader.py``, ``infer/deployer.py``,
``cli/leaderboard/session.py``, and the two DeepSeek-Math graders). This module
is for the one case where a thread provably changes the answer: grading through
``math-verify``.

Why it matters: every runner in a session shares one event loop —
:meth:`MultiTaskRunner.arun` starts each :class:`TaskRunner` with
``tg.start_soon`` inside a single ``anyio.run`` — so a stage that computes
synchronously stalls *every* other task, not just its own samples. Measured, a
co-running benchmark dropped to 0.4% of its solo throughput.

Why a process and not a thread: ``math-verify`` bounds its ``parse`` / ``verify``
with ``signal.SIGALRM``, which only arms on the main thread, and off it raises
rather than degrading::

    ValueError: Math-Verify 'parse' function doesn't support threaded environment

Callers wrap that in a broad ``except``, so in a thread the whole math-verify
strategy vanishes silently and verdicts flip (``\\frac{1}{2}`` against ``0.5``
goes True -> False). Disabling its timeout makes it thread-safe but then, in
math-verify's own words, the caller "must provide the logic for timeout
interuption yourself" — which a thread cannot, being uninterruptible. A worker
process is the main thread of its own process, so timeouts work and verdicts are
unchanged. A hang is contained too: it occupies one worker instead of the loop.

Degrades rather than fails — if the pool cannot start, work runs inline: the
behaviour from before this module existed, slow but correct.
``SIEVAL_OFFLOAD_WORKERS=0`` forces that path.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import atexit
import os
import threading
from collections.abc import Callable
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from functools import partial

import anyio
import anyio.to_thread
from loguru import logger

#: Worker count. Grading is CPU-bound, so this tracks cores rather than the
#: sample concurrency — queueing beyond the cores available buys nothing.
#: ``SIEVAL_OFFLOAD_WORKERS=0`` disables offloading entirely, which is the
#: escape hatch for an environment where spawning is not allowed.
_ENV_WORKERS = "SIEVAL_OFFLOAD_WORKERS"

#: Default ceiling for grading one rollout. Generous against the tens of
#: milliseconds a symbolic comparison normally costs and the 5 s ``math-verify``
#: allows itself per parse/verify, so reaching it means an input that got past
#: the caller's own guards — worth surfacing rather than a silent slow sample.
GRADE_TIMEOUT = 30.0

#: Extra admissions beyond the worker count (see :data:`_limiter`). Enough that
#: a worker never idles waiting for the next caller to be let in, small enough
#: that the queue a caller can sit behind stays a fixed multiple of the pool.
_QUEUE_SLACK = 2

_pool: ProcessPoolExecutor | None = None
_pool_failed = False
_lock = threading.Lock()

#: Admission control, sized to the pool rather than to the sample concurrency.
#:
#: Two things ride on this, and neither is served by the stage limiters upstream
#: of here — those bound *samples in flight*, a different quantity from *pool
#: capacity*.
#:
#: 1. ``timeout`` measures the grade, not the backlog. ``future.result(timeout)``
#:    starts counting when the caller begins waiting, and it waits out the queue
#:    ahead of it as well as its own call. Capping how many callers may be
#:    waiting at once caps that queue at ``_QUEUE_SLACK`` items, so the worst
#:    case is a small multiple of one grade instead of a function of how many
#:    samples the session happens to run. Unbounded, an ordinary 1 s grade
#:    "times out" purely from queueing.
#: 2. It keeps grading off anyio's shared thread tokens. Passing ``limiter=``
#:    to ``run_sync`` substitutes this one for the default 40, which the loader,
#:    the deployer and scicode's reads are also drawing on.
_limiter: anyio.CapacityLimiter | None = None


def _worker_count() -> int:
    configured = os.environ.get(_ENV_WORKERS)
    if configured is not None:
        try:
            return max(0, int(configured))
        except ValueError:
            logger.warning(
                "{} is not an integer ({!r}); using the default worker count.",
                _ENV_WORKERS,
                configured,
            )
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def _get_pool() -> ProcessPoolExecutor | None:
    """The shared pool, created on first use. ``None`` means "run inline".

    Call from the event loop: the admission limiter is an anyio primitive and
    needs a running async context to be constructed.
    """
    global _pool, _pool_failed, _limiter
    if _pool is not None or _pool_failed:
        return _pool
    with _lock:
        if _pool is not None or _pool_failed:
            return _pool
        workers = _worker_count()
        if workers == 0:
            _pool_failed = True
            return None
        try:
            import multiprocessing

            # `spawn`, not `fork`: the parent is an async process with live
            # worker threads, and forking one of those risks inheriting a held
            # lock and deadlocking the child.
            _pool = ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            )
            _limiter = anyio.CapacityLimiter(workers + _QUEUE_SLACK)
        except Exception as exc:
            _pool_failed = True
            logger.warning(
                "Could not start the offload pool ({}); CPU-bound stage work "
                "will run on the event loop, which slows every task sharing it.",
                exc,
            )
        return _pool


def shutdown() -> None:
    """Tear the pool down. Registered with :mod:`atexit`; safe to call twice."""
    global _pool, _limiter
    with _lock:
        pool, _pool = _pool, None
        _limiter = None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


# Registered once, at import. Registering alongside each pool would add another
# handler every time a pool is rebuilt after `shutdown()`.
atexit.register(shutdown)


async def run_cpu_bound[T](
    func: Callable[..., T], *args, timeout: float | None = None
) -> T:
    """Run *func(\\*args)* in a worker process, leaving the event loop free.

    *func* must be importable by name (a module-level function, not a lambda or
    a closure) and its arguments and return value must pickle — the grading
    entry points take strings and return bools, which is the shape this is for.

    Raises :exc:`TimeoutError` when *timeout* elapses. The worker is left to
    finish on its own: a process pool cannot interrupt a running call, and the
    alternative — tearing down the pool — would punish every other in-flight
    sample for one bad input. The slot returns when the worker does.

    *timeout* bounds the call itself, not the backlog in front of it — see
    :data:`_limiter`. The one place it does not apply is the inline fallback
    below: running on the event loop, there is nothing left to interrupt it
    with. That path is bounded only by whatever guards the caller applies to
    its own input.

    Falls back to running inline when no pool is available, so behaviour is
    degraded (slow) rather than broken.
    """
    pool = _get_pool()
    if pool is None:
        return func(*args)
    try:
        future = pool.submit(func, *args)
    except Exception as exc:
        # Pool died (a worker segfaulted), is shutting down, or could not start
        # a worker at all — ENOMEM and a blocked `clone` surface here as
        # OSError and PermissionError, not as BrokenExecutor. Every failure to
        # submit means the same thing: do not retry into it, take the slow path
        # for the rest of the run.
        _mark_unusable(exc)
        return func(*args)
    try:
        # `_limiter` gates how many callers may be *waiting*, which is what puts
        # a ceiling on `timeout`. Submissions and token handoffs are both FIFO,
        # so a caller holding a token is within `_QUEUE_SLACK` of the front of
        # the pool queue however long the backlog behind it grows.
        return await anyio.to_thread.run_sync(
            partial(future.result, timeout), limiter=_limiter
        )
    except TimeoutError:
        future.cancel()
        raise
    except BrokenExecutor as exc:
        _mark_unusable(exc)
        return func(*args)


def _mark_unusable(exc: Exception) -> None:
    global _pool_failed
    with _lock:
        if not _pool_failed:
            _pool_failed = True
            logger.warning(
                "Offload pool became unusable ({}); CPU-bound stage work falls "
                "back to the event loop for the rest of this run.",
                exc,
            )
