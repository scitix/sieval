"""Unit tests for the CPU-bound offload pool.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import time

import pytest

from sieval.core.utils import offload


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


@pytest.mark.anyio
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


def test_worker_count_honours_the_env_var(monkeypatch):
    monkeypatch.setenv(offload._ENV_WORKERS, "3")
    assert offload._worker_count() == 3


def test_worker_count_ignores_a_non_integer(monkeypatch):
    monkeypatch.setenv(offload._ENV_WORKERS, "many")
    assert offload._worker_count() >= 1


def test_shutdown_is_idempotent():
    offload.shutdown()
    offload.shutdown()


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
