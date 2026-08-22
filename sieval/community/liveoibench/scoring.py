# adapted from https://github.com/LiveOIBench/LiveOIBench-Evaluation/blob/7759e3b8672307cfbdc8ab8e679bd87cc1dd4c12/src/judges/base_judge.py
# (BaseJudge.interprete_task_result)
"""Subtask scoring for LiveOIBench.

Divergence from upstream: none in the arithmetic. Upstream is a method on the
judge that has just executed the tests; here the per-test verdicts arrive from
the evaluator service, so the same function takes them as an argument. The
misspelled name is upstream's and is kept so the two can be diffed.

The ``min-score`` branch is reachable only with a custom checker, which supplies
a fractional per-test ``score``. No checker ships in the published dataset (see
``process_dataset.py``, which materializes no ``checkers/`` directory), so on
that data every test scores 1.0 or 0.0 and the branch collapses onto the
all-or-nothing rule below it. It is ported anyway, because dropping it would
make the port wrong for any release that does ship checkers.
"""

from typing import Any, Mapping, Sequence


def interprete_task_result(
    results: Sequence[Mapping[str, Any]],
    subtasks: Mapping[str, Mapping[str, Any]],
) -> dict:
    """Score one submission from its per-test verdicts.

    *results* is one mapping per executed test with ``test_case`` (the test's
    name, no extension), ``correct`` (bool) and optionally ``score`` (float,
    checker-supplied). *subtasks* is the problem's ``subtasks`` payload: an id
    to ``{"score": int, "testcases": [name, ...]}`` mapping, optionally with
    ``"grading": "min-score"``.

    Returns ``{"subtasks": {id: {...}}, "score": int, "tests_passed": float,
    "ace": bool}``.
    """
    score: dict[str, Any] = {"subtasks": {}}
    results_dict = {result["test_case"]: result for result in results}
    test_cases_passed = [result["test_case"] for result in results if result["correct"]]

    for i, subtask in subtasks.items():
        subtask_score = 0
        subtask_passed = 0

        for test_case in subtask["testcases"]:
            if results_dict[test_case]["correct"]:
                subtask_passed += 1

        # Handle min-score grading
        if "grading" in subtask and subtask["grading"] == "min-score":
            test_scores = [
                results_dict[test_case].get("score", 0) for test_case in subtask["testcases"]
            ]
            subtask_score = subtask["score"] * min(test_scores)

        # Full score if all tests passed
        if subtask_passed == len(subtask["testcases"]):
            subtask_score = subtask["score"]

        score["subtasks"][i] = {
            "testcases": len(subtask["testcases"]),
            "score": subtask_score,
            "passed": subtask_passed / len(subtask["testcases"]),
        }

    score["score"] = round(sum([subtask["score"] for subtask in score["subtasks"].values()]))
    score["tests_passed"] = len(test_cases_passed) / len(results) if results else 0
    score["ace"] = score["tests_passed"] == 1

    return score


def total_points(subtasks: Mapping[str, Mapping[str, Any]]) -> int:
    """The problem's maximum score — upstream ``Problem.get_total_points``."""
    return sum(int(subtask["score"]) for subtask in subtasks.values())
