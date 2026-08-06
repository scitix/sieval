"""Per-case test-case budget in the vendored code evaluator.

Official LiveCodeBench budgets each test case, not the suite: ``lcb_runner`` re-arms
``signal.alarm(timeout)`` inside the case loop of ``grade_call_based`` / ``grade_stdio``
(``lcb_runner/evaluation/testing_util.py``), with ``codegen_metrics(..., timeout=6)``
supplying the default, and ``check_correctness`` joining the worker at
``(timeout + 1) * n + 5`` as a backstop. The evaluator previously had only that
backstop, which is a different rule: a 43-case suite where one case takes 200 s and
the rest take 1 s fits inside a 258 s whole-suite wall and fails a 6 s-per-case one.

Tests that spawn a subprocess and wait on real wall-clock timeouts are marked
``stress`` and so are excluded from CI; the guard's own semantics -- including the
``BaseException`` choice that the rest of the module depends on -- are covered
in-process, in milliseconds, and do run there.

The task half of the same feature is covered by
``tests/unit/tasks/test_livecodebench_code_generation_0shot_gen.py``.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import asyncio
import importlib
import pathlib
import signal
import sys
import time

import pytest

_EVALUATOR = pathlib.Path(__file__).resolve().parents[4] / "vendor/code-evaluator"


@pytest.fixture(scope="module")
def evaluator():
    """Import the vendored evaluator's ``app.exec_py_test``.

    By putting the evaluator root on ``sys.path`` and importing normally, rather than
    stitching a synthetic package together from a file path. ``execute_test`` runs the
    submission in a **spawned** subprocess, and spawn hands the child ``sys.path`` but
    not the parent's ``sys.modules``: the child unpickles ``_subprocess_target`` by its
    dotted name, so ``app.exec_py_test`` has to be importable from scratch over there.
    A synthetic module satisfies the parent and leaves the child with
    "no result from subprocess" -- which reads exactly like the code under test
    failing.

    Undone afterwards so ``app`` -- a name generic enough to collide -- does not stay
    importable for the rest of the session.
    """
    added = str(_EVALUATOR) not in sys.path
    if added:
        sys.path.insert(0, str(_EVALUATOR))
    try:
        yield importlib.import_module("app.exec_py_test")
    finally:
        if added:
            sys.path.remove(str(_EVALUATOR))
            for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
                del sys.modules[name]


def _spin(seconds: float = 5.0) -> None:
    """Burn CPU in Python bytecode, bounded so a broken guard fails instead of hangs."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        pass


# --------------------------------------------------------------------------- #
# The guard itself -- in-process, sub-second, runs in CI
# --------------------------------------------------------------------------- #
def test_guard_fires_and_reports_the_budget(evaluator):
    started = time.monotonic()
    with pytest.raises(evaluator.CaseTimeout):
        with evaluator._case_time_limit(0.05):
            _spin()
    # Fired on its own budget, not on `_spin`'s bound.
    assert time.monotonic() - started < 1.0


def test_guard_is_not_swallowed_by_the_submissions_except_exception(evaluator):
    # Why CaseTimeout derives from BaseException: a submission wrapping its work in
    # `except Exception` must not be able to turn its own timeout into a wrong answer.
    with pytest.raises(evaluator.CaseTimeout):
        with evaluator._case_time_limit(0.05):
            try:
                _spin()
            except Exception:  # noqa: BLE001 - the point of the test
                pytest.fail("CaseTimeout was swallowed by `except Exception`")


@pytest.mark.parametrize("budget", [None, 0, 0.0])
def test_guard_is_a_noop_without_a_budget(evaluator, budget):
    previous = signal.getsignal(signal.SIGALRM)
    with evaluator._case_time_limit(budget):
        pass
    # No handler installed, nothing armed.
    assert signal.getsignal(signal.SIGALRM) is previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_guard_disarms_and_restores_on_the_way_out(evaluator):
    previous = signal.getsignal(signal.SIGALRM)
    with evaluator._case_time_limit(30.0):
        pass
    assert signal.getsignal(signal.SIGALRM) is previous
    # Left armed, the next case would inherit this one's remaining budget.
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_worker_catch_covers_the_base_exception(evaluator):
    # `_subprocess_target` catches Exception, which does not cover CaseTimeout.
    # Cancelling the timer cannot un-deliver a signal the kernel already sent, so a
    # late CaseTimeout can surface outside `_unsafe_execute`'s own except; uncaught
    # it would leave the queue empty and strand the parent until the suite wall.
    class _Queue:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    q = _Queue()
    module = evaluator

    def _raise_late(*args, **kwargs):
        _ = (args, kwargs)
        raise module.CaseTimeout

    original = module._unsafe_execute
    module._unsafe_execute = _raise_late
    try:
        module._subprocess_target(q, "code", [], [], None, None, 6.0)
    finally:
        module._unsafe_execute = original

    assert len(q.items) == 1, "worker died without reporting -- parent would hang"
    ok, msg, n_passed = q.items[0]
    assert ok is False
    assert "CaseTimeout" in msg
    assert n_passed == 0


# --------------------------------------------------------------------------- #
# End to end through a spawned worker -- real wall clock, excluded from CI
# --------------------------------------------------------------------------- #
# Case 2 of 3 spins forever; cases 1 and 3 are correct and instant.
_SLOW_ON_CASE_2 = "n = int(input())\nif n == 2:\n    while True: pass\nprint(n * 2)\n"
_CLEAN = "n = int(input())\nprint(n * 2)\n"
_INPUTS = ["1\n", "2\n", "3\n"]
_OUTPUTS = ["2", "4", "6"]

# Hangs at module level, before any test case runs -- upstream budgets this on the
# per-case clock too (its `compile_code` arms the alarm around the `exec`).
_HANGS_WHILE_COMPILING = "while True:\n    pass\n\n\ndef solve():\n    return 1\n"


@pytest.mark.stress
def test_whole_suite_wall_is_unchanged_and_still_loses_the_count(evaluator):
    ok, msg, _stats, n_passed = asyncio.run(
        evaluator.execute_test(_SLOW_ON_CASE_2, _INPUTS, _OUTPUTS, None, timeout=8.0)
    )
    assert not ok
    assert "subprocess timeout" in msg
    # The worker is killed, so how far it got was never reported. `None` is
    # "unknown", never "zero" -- the reason a per-case budget is worth having.
    assert n_passed is None


@pytest.mark.stress
def test_per_case_budget_fires_and_keeps_the_count(evaluator):
    ok, msg, _stats, n_passed = asyncio.run(
        evaluator.execute_test(
            _SLOW_ON_CASE_2,
            _INPUTS,
            _OUTPUTS,
            None,
            timeout=60.0,
            timeout_per_case=2.0,
        )
    )
    assert not ok
    assert "case timeout" in msg, msg
    # The worker returned normally, so the cases already passed are known.
    assert n_passed == 1


@pytest.mark.stress
@pytest.mark.parametrize("per_case", [None, 2.0])
def test_a_correct_submission_is_unaffected_either_way(evaluator, per_case):
    ok, _msg, _stats, n_passed = asyncio.run(
        evaluator.execute_test(
            _CLEAN, _INPUTS, _OUTPUTS, None, timeout=60.0, timeout_per_case=per_case
        )
    )
    assert ok
    assert n_passed == 3


@pytest.mark.stress
def test_a_hang_during_compilation_is_on_the_per_case_clock(evaluator):
    # A hang at module level is inside no test case, so an execution-only budget
    # would let it through to the whole-suite wall and lose the count with it.
    ok, msg, _stats, n_passed = asyncio.run(
        evaluator.execute_test(
            _HANGS_WHILE_COMPILING,
            ["[]"],
            ["1"],
            "solve",
            timeout=60.0,
            timeout_per_case=2.0,
        )
    )
    assert not ok
    assert "compile timeout" in msg, msg
    assert n_passed == 0


@pytest.mark.stress
def test_without_a_per_case_budget_the_same_hang_only_hits_the_suite_wall(evaluator):
    ok, msg, _stats, n_passed = asyncio.run(
        evaluator.execute_test(
            _HANGS_WHILE_COMPILING, ["[]"], ["1"], "solve", timeout=5.0
        )
    )
    assert not ok
    assert "subprocess timeout" in msg
    assert n_passed is None
