"""Unit tests for the CPU-bound offload pool.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import os
import time

import pytest

from sieval.core.utils import offload


@pytest.fixture
def pool_spy(monkeypatch):
    """Capture the kwargs the pool is constructed with.

    Asserting on the construction rather than on the executor's private
    attributes: `_mp_context` / `_max_workers` are CPython internals, and a test
    that reads them is one stdlib refactor away from breaking.
    """
    captured: dict = {}

    class _Spy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def shutdown(self, **kwargs):
            pass

    monkeypatch.setattr(offload, "ProcessPoolExecutor", _Spy)
    return captured


@pytest.fixture
def warnings_sink():
    """Collect loguru WARNINGs — this module logs through loguru, not stdlib."""
    from loguru import logger

    collected: list[str] = []
    sink_id = logger.add(collected.append, format="{message}", level="WARNING")
    try:
        yield collected
    finally:
        logger.remove(sink_id)


@pytest.fixture(autouse=True)
def _reset_pool():
    """Each test starts from a cold module, and leaves no pool behind."""
    offload.shutdown()
    offload._pool_failed = False
    yield
    offload.shutdown()
    offload._pool_failed = False


# Module-level so the worker can import it by name; a closure would not pickle.
def _double(value: int) -> int:
    return value * 2


def _boom() -> None:
    raise ValueError("worker raised")


def _sleep_forever() -> None:
    time.sleep(30)


@pytest.mark.anyio
async def test_runs_the_function_and_returns_its_value():
    assert await offload.run_cpu_bound(_double, 21) == 42


@pytest.mark.anyio
async def test_worker_exceptions_propagate_to_the_caller():
    # A grader relies on its own try/except; swallowing here would turn a real
    # error into a silent wrong verdict.
    with pytest.raises(ValueError, match="worker raised"):
        await offload.run_cpu_bound(_boom)


#: mutmut instruments the module under test, and the trampoline it injects
#: imports `mutmut.__main__` on its first hit. In a *spawned* worker that module
#: is not yet in `sys.modules`, so the import re-executes it, and its top-level
#: `set_start_method('fork')` raises `RuntimeError: context has already been set`
#: — spawn fixes the start method before user code runs. The worker dies, the
#: pool reports itself broken, and `run_cpu_bound` correctly falls back to
#: running inline. Every other test still passes through that fallback (inline
#: returns the same answer), which is why only the timeout assertion notices.
_UNDER_MUTMUT = "MUTANT_UNDER_TEST" in os.environ


@pytest.mark.anyio
@pytest.mark.skipif(
    _UNDER_MUTMUT,
    reason="mutmut cannot instrument a spawned worker; the pool degrades to "
    "inline, which by design cannot time out",
)
async def test_timeout_raises_rather_than_returning_a_wrong_answer():
    with pytest.raises(TimeoutError):
        await offload.run_cpu_bound(_sleep_forever, timeout=0.5)


@pytest.mark.anyio
async def test_runs_inline_when_the_pool_is_disabled(monkeypatch):
    # The documented escape hatch for a sandbox that cannot spawn. Behaviour
    # must stay correct, only slower.
    monkeypatch.setenv(offload._ENV_WORKERS, "0")
    offload.shutdown()
    offload._pool_failed = False
    assert await offload.run_cpu_bound(_double, 5) == 10
    assert offload._get_pool() is None


@pytest.mark.anyio
async def test_falls_back_inline_when_the_pool_cannot_start(monkeypatch):
    def _explode(*_args, **_kwargs):
        raise OSError("no processes here")

    monkeypatch.setattr(offload, "ProcessPoolExecutor", _explode)
    assert await offload.run_cpu_bound(_double, 8) == 16


@pytest.mark.anyio
async def test_a_broken_pool_degrades_instead_of_failing_the_run(monkeypatch):
    from concurrent.futures import BrokenExecutor

    class _BrokenPool:
        def submit(self, *_args, **_kwargs):
            raise BrokenExecutor("worker died")

        def shutdown(self, **_kwargs):
            pass

    monkeypatch.setattr(offload, "_pool", _BrokenPool())
    assert await offload.run_cpu_bound(_double, 3) == 6
    # And it does not keep retrying into the broken pool.
    assert offload._pool_failed is True


def test_the_pool_is_spawned_never_forked(pool_spy):
    # The load-bearing choice in this module: the parent is an async process with
    # live worker threads, and forking one can inherit a held lock and deadlock
    # the child. A mutation to "fork" fails no behavioural test — it just
    # occasionally hangs a run — so it is asserted at construction.
    offload._get_pool()
    assert pool_spy["mp_context"].get_start_method() == "spawn"


def test_the_pool_is_built_once_and_reused(monkeypatch):
    # Rebuilding per call would pay spawn + sympy import on every grade, which is
    # the cost this module exists to avoid.
    monkeypatch.setenv(offload._ENV_WORKERS, "2")
    first = offload._get_pool()
    assert first is not None
    assert offload._get_pool() is first


def test_the_pool_is_sized_by_the_worker_count(monkeypatch, pool_spy):
    monkeypatch.setenv(offload._ENV_WORKERS, "3")
    offload._get_pool()
    assert pool_spy["max_workers"] == 3


def test_a_disabled_pool_is_not_reconsidered(monkeypatch):
    # Once disabled it must stay disabled for the run, or every sample pays the
    # decision again.
    monkeypatch.setenv(offload._ENV_WORKERS, "0")
    assert offload._get_pool() is None
    assert offload._pool_failed is True
    # Even with the env var flipped back, the run does not silently change mode.
    monkeypatch.setenv(offload._ENV_WORKERS, "4")
    assert offload._get_pool() is None


def test_a_failed_pool_start_is_not_retried_per_sample(monkeypatch):
    starts = {"n": 0}

    def _explode(*_args, **_kwargs):
        starts["n"] += 1
        raise OSError("no processes here")

    monkeypatch.setattr(offload, "ProcessPoolExecutor", _explode)
    assert offload._get_pool() is None
    assert offload._get_pool() is None
    assert starts["n"] == 1, "a failed start must not be attempted again"


def test_worker_count_honours_the_env_var(monkeypatch):
    monkeypatch.setenv(offload._ENV_WORKERS, "3")
    assert offload._worker_count() == 3


def test_worker_count_ignores_a_non_integer(monkeypatch, warnings_sink):
    monkeypatch.setenv(offload._ENV_WORKERS, "many")
    assert offload._worker_count() >= 1
    # Silently falling back would hide a typo'd override for the whole run, so
    # the warning has to quote what was actually set.
    text = " ".join(warnings_sink)
    assert offload._ENV_WORKERS in text
    assert "many" in text


def test_worker_count_clamps_a_negative_request_to_disabled(monkeypatch):
    # Negative is nonsense, and 0 is the documented "run inline" value, so it
    # floors there rather than becoming a huge pool via a sign error.
    monkeypatch.setenv(offload._ENV_WORKERS, "-5")
    assert offload._worker_count() == 0


def test_worker_count_caps_the_default_on_a_big_machine(monkeypatch):
    # Grading is CPU-bound but the pool is shared across a whole session; more
    # than 8 workers buys nothing and costs one interpreter each.
    monkeypatch.delenv(offload._ENV_WORKERS, raising=False)
    monkeypatch.setattr(offload.os, "cpu_count", lambda: 64)
    assert offload._worker_count() == 8


def test_worker_count_keeps_one_worker_on_a_single_core_box(monkeypatch):
    # `cpu_count - 1` is 0 here; a pool of 0 workers is not constructible.
    monkeypatch.delenv(offload._ENV_WORKERS, raising=False)
    monkeypatch.setattr(offload.os, "cpu_count", lambda: 1)
    assert offload._worker_count() == 1


def test_worker_count_survives_an_unknown_cpu_count(monkeypatch):
    # os.cpu_count() may return None; the fallback must still be constructible.
    monkeypatch.delenv(offload._ENV_WORKERS, raising=False)
    monkeypatch.setattr(offload.os, "cpu_count", lambda: None)
    assert offload._worker_count() == 1


def test_worker_count_leaves_room_for_the_event_loop(monkeypatch):
    # One core is deliberately left to the loop that is dispatching the work.
    monkeypatch.delenv(offload._ENV_WORKERS, raising=False)
    monkeypatch.setattr(offload.os, "cpu_count", lambda: 4)
    assert offload._worker_count() == 3


def test_shutdown_is_idempotent():
    offload.shutdown()
    offload.shutdown()


def test_shutdown_releases_the_pool_without_waiting(monkeypatch):
    # Waiting would hang a run on exactly the sample that already misbehaved,
    # and leaving `_pool` set would hand out a shut-down executor afterwards.
    calls = {}

    class _Pool:
        def shutdown(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(offload, "_pool", _Pool())
    offload.shutdown()

    assert calls == {"wait": False, "cancel_futures": True}
    assert offload._pool is None


def test_a_broken_pool_is_reported_once_not_per_sample(warnings_sink):
    # `_mark_unusable` fires on a path taken by every subsequent sample; warning
    # each time would bury the run's real output.
    offload._pool_failed = False
    offload._mark_unusable(RuntimeError("first"))
    offload._mark_unusable(RuntimeError("second"))

    warnings = [m for m in warnings_sink if "unusable" in m]
    assert len(warnings) == 1
    assert offload._pool_failed is True
    # The message is the operator's only signal that grading silently went back
    # on the event loop, so it has to name the cause and the consequence.
    assert "first" in warnings[0], "the warning must name the exception"
    assert "event loop" in warnings[0]
    assert "second" not in warnings[0]


@pytest.mark.anyio
async def test_math_verify_still_works_in_the_worker():
    """The whole reason this is a process and not a thread.

    In a worker *thread* math-verify raises outright, callers swallow it, and
    these verdicts silently flip to False.
    """
    from sieval.community.ugmathbench import judge_answers

    for pred, gold, kind in [
        (r"\frac{1}{2}", "0.5", "NV"),
        (r"\frac{\pi}{4}", "pi/4", "EX"),
    ]:
        got = await offload.run_cpu_bound(judge_answers, [pred], [gold], [kind])
        assert got == [True], f"{pred} vs {gold} should still grade correct"
