"""Unit tests for LiveOIBench subtask scoring.

The arithmetic is the benchmark: a subtask is all-or-nothing, so a submission
that passes most tests of every subtask can score zero while another that passes
fewer tests scores points. Pinning that here keeps a "reasonable" partial-credit
rewrite from quietly becoming a different benchmark.

AI-Generated Code - Claude Opus 4.5 (Anthropic)
"""

import pytest

from sieval.community.liveoibench.scoring import interprete_task_result, total_points

SUBTASKS = {
    "1": {"task": "Subtask 1", "score": 0, "testcases": ["s0"]},
    "2": {"task": "Subtask 2", "score": 30, "testcases": ["a1", "a2"]},
    "3": {"task": "Subtask 3", "score": 70, "testcases": ["b1", "b2"]},
}


def _results(**passed: bool) -> list[dict]:
    return [{"test_case": name, "correct": ok} for name, ok in passed.items()]


def test_a_subtask_pays_only_when_every_test_in_it_passes():
    scored = interprete_task_result(
        _results(s0=True, a1=True, a2=True, b1=True, b2=False), SUBTASKS
    )
    # Subtask 3 lost one test of two, so its 70 points are gone entirely.
    assert scored["score"] == 30
    assert scored["subtasks"]["3"]["score"] == 0
    assert scored["subtasks"]["3"]["passed"] == 0.5


def test_more_tests_passed_can_still_score_less():
    # 4 of 5 tests, spread so that both paying subtasks are broken -> 0 points.
    spread = interprete_task_result(
        _results(s0=True, a1=True, a2=False, b1=True, b2=True), SUBTASKS
    )
    # 3 of 5 tests, but one whole subtask intact -> 30 points.
    focused = interprete_task_result(
        _results(s0=False, a1=True, a2=True, b1=True, b2=False), SUBTASKS
    )
    assert spread["tests_passed"] > focused["tests_passed"]
    assert spread["score"] == 70  # subtask 3 whole
    assert focused["score"] == 30


def test_ace_and_tests_passed_span_every_result():
    scored = interprete_task_result(
        _results(s0=True, a1=True, a2=True, b1=True, b2=True), SUBTASKS
    )
    assert scored["score"] == 100
    assert scored["tests_passed"] == 1
    assert scored["ace"] is True


def test_a_zero_point_sample_subtask_does_not_change_the_score():
    without = interprete_task_result(
        _results(s0=False, a1=True, a2=True, b1=True, b2=True), SUBTASKS
    )
    assert without["score"] == 100
    assert without["ace"] is False, "a failed sample test still breaks the ace flag"


def test_total_is_rounded_not_truncated():
    # min-score grading with a checker score of 0.5 on a 45-point subtask -> 22.5.
    subtasks = {"1": {"score": 45, "grading": "min-score", "testcases": ["a", "b"]}}
    results = [
        {"test_case": "a", "correct": False, "score": 0.5},
        {"test_case": "b", "correct": True, "score": 1.0},
    ]
    assert interprete_task_result(results, subtasks)["score"] == 22


def test_min_score_collapses_to_all_or_nothing_without_a_checker():
    # The published dataset ships no checkers, so no result carries a `score` and
    # upstream's `.get("score", 0)` reads 0 -- the same verdict as the default
    # rule. Pinned because a port that invented per-test fractions here would
    # silently pay partial credit upstream does not.
    subtasks = {"1": {"score": 45, "grading": "min-score", "testcases": ["a", "b"]}}
    partial = interprete_task_result(
        [{"test_case": "a", "correct": True}, {"test_case": "b", "correct": False}],
        subtasks,
    )
    assert partial["score"] == 0
    whole = interprete_task_result(
        [{"test_case": "a", "correct": True}, {"test_case": "b", "correct": True}],
        subtasks,
    )
    assert whole["score"] == 45


def test_an_empty_result_set_scores_zero_rather_than_dividing_by_zero():
    scored = interprete_task_result([], {})
    assert scored == {"subtasks": {}, "score": 0, "tests_passed": 0, "ace": False}


def test_a_subtask_naming_an_unknown_test_raises():
    # The two parquets are out of step; scoring it would silently understate.
    with pytest.raises(KeyError):
        interprete_task_result([{"test_case": "a", "correct": True}], SUBTASKS)


def test_total_points_sums_the_rubric():
    assert total_points(SUBTASKS) == 100
    assert total_points({}) == 0
