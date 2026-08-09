"""Tests for the shared rollout estimators.

The properties worth pinning are the ones a copy-pasted helper drifts away from:
pass@1 is c/n and not "the first sample", pass@k is "solved at least once", and maj@k
votes on ANSWERS so it can disagree with a verdict tally in both directions.
"""

import pytest

from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
from sieval.core.tasks.metrics import (
    aggregate,
    avg_at_k,
    budget_metrics,
    count_short,
    majority_at_k,
    pass_at_k,
    rollout_metrics,
    rollout_view,
    zero_metrics,
)


@pytest.mark.parametrize(
    "n,c,k,want",
    [
        (4, 0, 1, 0.0),
        (4, 4, 1, 1.0),
        (4, 1, 1, 0.25),  # pass@1 IS c/n
        (4, 2, 1, 0.5),
        (4, 1, 4, 1.0),  # one correct anywhere -> pass@4 is certain
        (4, 2, 2, 1 - (2 / 4) * (1 / 3)),
        (1, 1, 1, 1.0),
        (2, 1, 4, 0.0),  # k > n is unreachable by config; must not raise
    ],
)
def test_pass_at_k(n, c, k, want):
    assert pass_at_k(n, c, k) == pytest.approx(want)


def test_pass_at_1_equals_avg():
    for c in range(5):
        verdicts = [True] * c + [False] * (4 - c)
        assert pass_at_k(4, c, 1) == pytest.approx(avg_at_k(verdicts))


def test_pass_at_k_monotone_in_k():
    for k in range(1, 4):
        assert pass_at_k(4, 2, k) <= pass_at_k(4, 2, k + 1)


def test_majority_needs_answers_not_verdicts():
    # 2 right agreeing vs 2 wrong disagreeing: a verdict tally sees 2/4, the modal
    # ANSWER is correct, so self-consistency wins.
    assert majority_at_k([True, True, False, False], ["a", "a", "b", "c"]) == 1.0
    # the mirror: one right, three wrong but agreeing -> the majority is wrong
    assert majority_at_k([True, False, False, False], ["a", "b", "b", "b"]) == 0.0


def test_majority_ignores_empty_answers():
    assert majority_at_k([True, True], [None, None]) == 0.0
    assert majority_at_k([True, False], ["", "x"]) == 0.0


def test_majority_uses_the_normalizer():
    # unnormalised the two correct answers are distinct, so no answer has a plurality;
    # normalising merges them into a clear majority.
    args = ([True, True, False, False], ["1/2", "0.5", "9", "8"])
    assert majority_at_k(*args) == 0.0
    assert majority_at_k(*args, normalize=lambda s: {"1/2": "0.5"}.get(s, s)) == 1.0


def test_majority_trims_after_normalising():
    """`"18"` and `"18 "` are one candidate, not two splitting a real majority."""
    assert majority_at_k([True, True, False], ["18", "18 ", "20"]) == 1.0


def test_majority_is_order_independent():
    """A tie is not a majority -- otherwise the metric depends on sample order."""
    assert majority_at_k([True, False], ["a", "b"]) == 0.0
    assert majority_at_k([False, True], ["b", "a"]) == 0.0
    # and a real plurality is found wherever it sits
    for i in range(3):
        ans = ["b", "b"]
        ok = [False, False]
        ans.insert(i, "a")
        ok.insert(i, True)
        assert majority_at_k(ok, ans) == 0.0  # 2 vs 1 -> the WRONG answer wins


def test_majority_rejects_mismatched_lengths():
    assert majority_at_k([True, True], ["a"]) == 0.0


def test_keys_are_literal_not_interpolated():
    """The budget lives in the n/k report fields, never in a column name."""
    keys = set(rollout_metrics([True, False, False, False], ["a", "b", "c", "d"], k=4))
    assert keys == {"pass@1", "avg@k", "pass@k", "maj@k"}
    # nothing spells the value of k
    assert not any("4" in key for key in keys)


def test_rollout_metrics_omits_what_it_cannot_compute():
    # no answers -> no maj@k, rather than a 0.0 indistinguishable from a real 0.0
    assert set(rollout_metrics([True, False], None, k=2)) == {
        "pass@1",
        "avg@k",
        "pass@k",
    }
    # k == 1 -> no pass@k duplicate of pass@1
    assert set(rollout_metrics([True], None, k=1)) == {"pass@1", "avg@k"}
    assert set(rollout_metrics([True, False], ["a", "b"], k=2)) == {
        "pass@1",
        "avg@k",
        "pass@k",
        "maj@k",
    }


def test_majority_only_when_k_equals_n():
    """Sub-sampling k < n would need an estimator or a seed (RFC #74 D.2)."""
    four = ([True, True, False, False], ["a", "a", "b", "c"])
    assert "maj@k" in rollout_metrics(*four, k=4)
    assert "maj@k" not in rollout_metrics(*four, k=2)
    assert "maj@k" not in rollout_metrics(*four, k=1)


def test_avg_and_pass_at_1_both_reported_though_equal():
    """Equal arithmetic, different questions -- neither key subsumes the other."""
    metrics = rollout_metrics([True, False, False, False], k=2)
    assert metrics["pass@1"] == metrics["avg@k"]
    assert "pass@1" in metrics and "avg@k" in metrics


def test_empty_input_is_zero_not_an_error():
    assert avg_at_k([]) == 0.0
    assert majority_at_k([], []) == 0.0
    assert rollout_metrics([], None, k=1)["pass@1"] == 0.0


def test_count_short():
    assert count_short([4, 4, 4], 4) == 0
    assert count_short([4, 1, 3], 4) == 2
    assert count_short([], 4) == 0


def test_aggregate_uses_the_given_denominator():
    """Failures count as wrong, so the denominator is the task's, not len()."""
    per_problem = [{"pass@1": 1.0}, {"pass@1": 1.0}]
    assert aggregate(per_problem, 4) == {"pass@1": 50.0}
    assert aggregate(per_problem, 2) == {"pass@1": 100.0}


def test_aggregate_drops_partial_keys_rather_than_averaging_them():
    """An omitted per-problem key must not reappear as a 0.0 contribution."""
    per_problem = [
        {"pass@1": 1.0, "maj@k": 1.0},
        {"pass@1": 1.0},  # maj@k could not be computed here
    ]
    out = aggregate(per_problem, 2)
    assert "maj@k" not in out
    assert out["pass@1"] == 100.0


def test_aggregate_degenerate_inputs():
    assert aggregate([], 4) == {}
    assert aggregate([{"pass@1": 1.0}], 0) == {}


# --------------------------------------------------------------------------- #
# zero_metrics -- the key set of a run that scored nothing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "n,k,want",
    [
        # k == n, so maj@k is defined even for a single rollout. Tasks gate the
        # whole sampling block on n > 1 and never read it there.
        (1, 1, {"pass@1", "avg@k", "maj@k"}),
        (4, 1, {"pass@1", "avg@k"}),  # k == 1 < n: pass@k would restate pass@1
        (4, 2, {"pass@1", "avg@k", "pass@k"}),  # k < n: majority is undefined
        (4, 4, {"pass@1", "avg@k", "pass@k", "maj@k"}),
    ],
)
def test_zero_metrics_matches_what_a_clean_run_would_report(n, k, want):
    """The empty path must not invent or drop a column relative to a scored run."""
    zeros = zero_metrics(n=n, k=k)
    assert set(zeros) == want
    assert set(zeros.values()) == {0.0}
    # Derived from rollout_metrics rather than listed, so the two cannot drift:
    # a clean all-correct draw at the same budget carries the same keys.
    assert set(zero_metrics(n=n, k=k)) == set(
        rollout_metrics([True] * n, ["42"] * n, k=k)
    )


@pytest.mark.parametrize("n,k", [(1, 1), (4, 4)])
def test_zero_metrics_without_votes_matches_a_task_that_never_votes(n, k):
    """The code family passes no answers, so it must not grow a maj@k column
    on the empty path that its scored path never reports."""
    zeros = zero_metrics(n=n, k=k, votes=False)
    assert "maj@k" not in zeros
    assert set(zeros) == set(rollout_metrics([True] * n, None, k=k))


# --------------------------------------------------------------------------- #
# rollout_view -- verdicts and answers of one judged sample
# --------------------------------------------------------------------------- #


def _ctx(judgement, postprocess) -> TaskContext:
    ctx = TaskContext(sample_id=0, raw_sample={})
    ctx = ctx.to_preprocessed({"prompt": "p"})
    ctx = ctx.to_inferred("inf")
    ctx = ctx.to_postprocessed(postprocess)
    return ctx.to_feedback(judgement).to_final()


def test_rollout_view_pairs_verdicts_with_answers():
    correct, answers = rollout_view(
        _ctx(
            build_judgement_record(
                "42", [build_rollout_judgement(i, i == 0) for i in range(2)]
            ),
            build_prediction_record(["42", "7"]),
        )
    )
    assert correct == [True, False]
    assert answers == ["42", "7"]


def test_rollout_view_keeps_an_unextracted_answer_as_none():
    """`None` is "could not extract", and it must not vote as an empty string."""
    _, answers = rollout_view(
        _ctx(
            build_judgement_record("42", [build_rollout_judgement(0, False)]),
            build_prediction_record([None]),
        )
    )
    assert answers == [None]


def test_rollout_view_drops_answers_it_cannot_align():
    """A resumed run without the prediction stage must omit maj@k, not guess it.

    Returning the short list instead would silently vote a 4-rollout draw on
    whichever rollouts happened to survive.
    """
    judgement = build_judgement_record(
        "42", [build_rollout_judgement(i, True) for i in range(4)]
    )
    _, answers = rollout_view(_ctx(judgement, build_prediction_record(["42"])))
    assert answers is None
    assert "maj@k" not in rollout_metrics([True] * 4, answers, k=4)


def test_rollout_view_survives_a_missing_stage_result():
    # Nothing judged and nothing extracted still align, so this reports an empty
    # draw rather than raising on the `None` stage results.
    assert rollout_view(TaskContext(sample_id=0, raw_sample={})) == ([], [])


# --------------------------------------------------------------------------- #
# budget_metrics -- the sampling budget as report keys
# --------------------------------------------------------------------------- #


def test_budget_metrics_reports_the_budget_and_the_short_draws():
    assert budget_metrics([4, 4, 2], n=4, k=2) == {"n": 4.0, "k": 2.0, "n_short": 1.0}


def test_budget_metrics_reports_zero_short_rather_than_omitting_it():
    """A missing `n_short` and a zero one read the same to a consumer; only one
    of them means "checked, and nothing was short"."""
    assert budget_metrics([4, 4], n=4, k=4) == {"n": 4.0, "k": 4.0, "n_short": 0.0}
    assert budget_metrics([], n=4, k=4)["n_short"] == 0.0
