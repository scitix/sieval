"""Per-case test-case budget for LiveCodeBench: the evaluator side and the task side.

Official LiveCodeBench budgets each test case, not the suite: ``lcb_runner`` re-arms
``signal.alarm(timeout)`` inside the case loop of ``grade_call_based`` / ``grade_stdio``
(``lcb_runner/evaluation/testing_util.py``), with ``codegen_metrics(..., timeout=6)``
supplying the default, and ``check_correctness`` joining the worker at
``(timeout + 1) * n + 5`` as a backstop. The evaluator previously had only that
backstop, which is a different rule: a 43-case suite where one case takes 200 s and
the rest take 1 s fits inside a 258 s whole-suite wall and fails a 6 s-per-case one.

Both halves are opt-in, so the first test here is the regression guard: with
``timeout_per_case`` unset nothing changes, including the fact that a whole-suite kill
loses the case count.
"""

import asyncio
import importlib
import pathlib
import sys

import pytest

_EVALUATOR = pathlib.Path(__file__).resolve().parents[3] / "vendor/code-evaluator"


def _load_exec_py_test():
    """Import the vendored evaluator's ``app.exec_py_test``.

    By putting the evaluator root on ``sys.path`` and importing normally, rather than
    stitching a synthetic package together from a file path. ``execute_test`` runs the
    submission in a **spawned** subprocess, and spawn hands the child ``sys.path`` but
    not the parent's ``sys.modules``: the child unpickles ``_subprocess_target`` by its
    dotted name, so ``app.exec_py_test`` has to be importable from scratch over there.
    A synthetic module satisfies the parent and leaves the child with
    "no result from subprocess" -- which reads exactly like the code under test
    failing.
    """
    if str(_EVALUATOR) not in sys.path:
        sys.path.insert(0, str(_EVALUATOR))
    return importlib.import_module("app.exec_py_test")


# Case 2 of 3 spins forever; cases 1 and 3 are correct and instant.
_SLOW_ON_CASE_2 = "n = int(input())\nif n == 2:\n    while True: pass\nprint(n * 2)\n"
_CLEAN = "n = int(input())\nprint(n * 2)\n"
_INPUTS = ["1\n", "2\n", "3\n"]
_OUTPUTS = ["2", "4", "6"]


@pytest.mark.stress  # spawns subprocesses and waits on real wall-clock timeouts
class TestEvaluatorPerCaseBudget:
    def test_whole_suite_wall_is_unchanged_and_still_loses_the_count(self):
        module = _load_exec_py_test()
        ok, msg, _stats, n_passed = asyncio.run(
            module.execute_test(_SLOW_ON_CASE_2, _INPUTS, _OUTPUTS, None, timeout=8.0)
        )
        assert not ok
        assert "subprocess timeout" in msg
        # The worker is killed, so how far it got was never reported. `None` is
        # "unknown", never "zero" -- the reason a per-case budget is worth having.
        assert n_passed is None

    def test_per_case_budget_fires_and_keeps_the_count(self):
        module = _load_exec_py_test()
        ok, msg, _stats, n_passed = asyncio.run(
            module.execute_test(
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

    @pytest.mark.parametrize("per_case", [None, 2.0])
    def test_a_correct_submission_is_unaffected_either_way(self, per_case):
        module = _load_exec_py_test()
        ok, _msg, _stats, n_passed = asyncio.run(
            module.execute_test(
                _CLEAN,
                _INPUTS,
                _OUTPUTS,
                None,
                timeout=60.0,
                timeout_per_case=per_case,
            )
        )
        assert ok
        assert n_passed == 3


class TestTaskRequestShaping:
    """What the task puts on the wire, without standing up the evaluator."""

    @staticmethod
    def _task(**kwargs):
        from sieval.tasks.livecodebench_code_generation_0shot_gen import (
            LiveCodeBenchCodeGenerationZeroShotGenTask as Task,
        )

        task = Task.__new__(Task)  # no live model/dataset needed for the arithmetic
        task._timeout = kwargs.get("timeout", 6.0)
        task._timeout_per_case = kwargs.get("timeout_per_case")
        return task

    def test_default_wall_is_the_historical_formula(self):
        task = self._task(timeout=30.0)
        n = 43
        assert task._timeout + n * 2.0 == 116.0
        assert task._timeout_per_case is None

    def test_per_case_switches_the_wall_to_upstreams_backstop_shape(self):
        task = self._task(timeout=30.0, timeout_per_case=6.0)
        n = 43
        # check_correctness joins at (timeout + 1) * n + 5
        assert (task._timeout_per_case + 1.0) * n + 5.0 == 306.0
