"""Run CPU-bound stage work off the event loop, in a worker process.

**The house pattern for not blocking the loop is ``anyio.to_thread.run_sync``,
called directly at the site** — ``core/tasks/loader.py``, ``infer/deployer.py``
and ``cli/leaderboard/session.py`` all do exactly that, and so do the two
DeepSeek-Math graders, whose pure-sympy verdicts are unchanged in a thread.
Reach for a thread first. This module exists for the one case where a thread
provably changes the answer, and it is the *only* such case today.

Every runner in a session shares one event loop:
:meth:`MultiTaskRunner.arun` starts each :class:`TaskRunner` with
``tg.start_soon`` inside a single ``anyio.run``. A stage that computes
synchronously therefore stalls *every* other task in the session, not only its
own samples — a benchmark grading with sympy measured a co-running benchmark
down to 0.4% of its solo throughput.

**Why a process and not a thread.** ``math-verify`` bounds its own
``parse`` / ``verify`` with ``signal.SIGALRM``, which only arms on the main
thread, and it refuses outright rather than degrading::

    ValueError: Math-Verify 'parse' function doesn't support threaded environment

Callers wrap that in a broad ``except``, so in a worker thread the whole
math-verify strategy would vanish silently and verdicts would flip
(``\\frac{1}{2}`` against ``0.5`` goes True -> False). Disabling its timeout
does make it thread-safe, but math-verify then warns that the caller "must
provide the logic for timeout interruption yourself" — which a thread cannot
do, since it cannot be interrupted. A worker process is the main thread of its
own process, so the timeouts keep working and verdicts are unchanged.

A hang is also contained rather than fatal: it occupies one worker while the
others keep grading, where the same hang on the event loop stops the session.

One caller today (``ugmathbench_0shot_gen_fixed``), but the contract it encodes
is shared: *every* site grading through ``math-verify`` must avoid a thread or
its verdicts flip, and twelve more math tasks grade that way. They are left on
the loop deliberately — their whole run costs 0.2-0.7 s of it, against
ugmathbench's ~190 s — so this is where that rule lives when one of them needs
it, not a helper waiting for a second user.

Degrades rather than fails. If the pool cannot start (a restricted sandbox, a
platform without ``spawn``), work runs inline — the same behaviour as before
this module existed, which is slow but correct.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import atexit
import os
import threading
from collections.abc import Callable
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from functools import partial

import anyio.to_thread
from loguru import logger

#: Worker count. Grading is CPU-bound, so this tracks cores rather than the
#: sample concurrency — queueing beyond the cores available buys nothing.
#: ``SIEVAL_OFFLOAD_WORKERS=0`` disables offloading entirely, which is the
#: escape hatch for an environment where spawning is not allowed.
_ENV_WORKERS = "SIEVAL_OFFLOAD_WORKERS"

_pool: ProcessPoolExecutor | None = None
_pool_failed = False
_lock = threading.Lock()


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
    """The shared pool, created on first use. ``None`` means "run inline"."""
    global _pool, _pool_failed
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
            atexit.register(shutdown)
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
    global _pool
    with _lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


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

    Falls back to running inline when no pool is available, so behaviour is
    degraded (slow) rather than broken.
    """
    pool = _get_pool()
    if pool is None:
        return func(*args)
    try:
        future = pool.submit(func, *args)
    except (BrokenExecutor, RuntimeError) as exc:
        # Pool died (a worker segfaulted) or is shutting down. Do not retry into
        # a broken pool; take the slow path for the rest of the run.
        _mark_unusable(exc)
        return func(*args)
    try:
        return await anyio.to_thread.run_sync(partial(future.result, timeout))
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
