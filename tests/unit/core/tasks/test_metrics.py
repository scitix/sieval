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
    count_unextracted,
    first_rollout_correct,
    majority_at_k,
    pass_at_k,
    pass_pow_k,
    rollout_metrics,
    rollout_view,
    sampling_report,
    warn_unscored_rollouts,
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


@pytest.mark.parametrize(
    "n,c,k,want",
    [
        (4, 4, 4, 1.0),  # every draw correct -> certain
        (4, 3, 4, 0.0),  # one wrong anywhere -> impossible
        (4, 2, 2, (2 / 4) * (1 / 3)),
        (4, 2, 1, 0.5),  # k=1 collapses onto pass@1
        (4, 0, 1, 0.0),
        (2, 2, 4, 0.0),  # k > n is unreachable by config; must not raise
    ],
)
def test_pass_pow_k(n, c, k, want):
    assert pass_pow_k(n, c, k) == pytest.approx(want)


def test_pass_pow_k_is_the_opposite_direction_from_pass_at_k():
    """The pair is the point: `pass@k` rises with sampling variance and this
    falls, so a model that got less reliable cannot flatter both at once."""
    tight = pass_at_k(4, 4, 2), pass_pow_k(4, 4, 2)  # unanimous
    loose = pass_at_k(4, 2, 2), pass_pow_k(4, 2, 2)  # same mean, wider spread
    assert loose[0] < tight[0]
    assert loose[1] < tight[1]
    # And within one draw, the optimistic bound is never below the pessimistic.
    for c in range(5):
        assert pass_pow_k(4, c, 2) <= pass_at_k(4, c, 2)


def test_pass_pow_k_monotone_down_in_k():
    """More rollouts required, never easier to satisfy."""
    for k in range(1, 4):
        assert pass_pow_k(4, 3, k) >= pass_pow_k(4, 3, k + 1)


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
    assert keys == {"pass@1", "avg@k", "pass@k", "pass^k", "maj@k"}
    # nothing spells the value of k
    assert not any("4" in key for key in keys)


def test_rollout_metrics_omits_what_it_cannot_compute():
    # no answers -> no maj@k, rather than a 0.0 indistinguishable from a real 0.0
    assert set(rollout_metrics([True, False], None, k=2)) == {
        "pass@1",
        "avg@k",
        "pass@k",
        "pass^k",
    }
    # k == 1 -> no pass@k duplicate of pass@1
    assert set(rollout_metrics([True], None, k=1)) == {"pass@1", "avg@k"}
    assert set(rollout_metrics([True, False], ["a", "b"], k=2)) == {
        "pass@1",
        "avg@k",
        "pass@k",
        "pass^k",
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
        # k < n: majority is undefined, but both directions of the
        # k-sample estimator are.
        (4, 2, {"pass@1", "avg@k", "pass@k", "pass^k"}),
        (4, 4, {"pass@1", "avg@k", "pass@k", "pass^k", "maj@k"}),
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


# --------------------------------------------------------------------------- #
# first_rollout_correct -- the upstream-comparable count
# --------------------------------------------------------------------------- #


def test_first_rollout_correct_ignores_the_rest_of_the_draw():
    """The published number was one greedy draw, so this must NOT become c/n."""
    finals = [
        _ctx(
            build_judgement_record(
                "42", [build_rollout_judgement(i, i > 0) for i in range(4)]
            ),
            build_prediction_record(["7", "42", "42", "42"]),
        ),
        _ctx(
            build_judgement_record(
                "42", [build_rollout_judgement(i, True) for i in range(4)]
            ),
            build_prediction_record(["42"] * 4),
        ),
    ]
    # 3 of 4 right in the first sample, but its first rollout was wrong.
    assert first_rollout_correct(finals) == 1


def test_first_rollout_correct_survives_a_sample_with_no_judgement():
    assert first_rollout_correct([TaskContext(sample_id=0, raw_sample={})]) == 0


# --------------------------------------------------------------------------- #
# sampling_report -- the whole n>1 block
# --------------------------------------------------------------------------- #


def test_sampling_report_covers_the_whole_block():
    finals = [
        _ctx(
            build_judgement_record(
                "42", [build_rollout_judgement(i, i < 2) for i in range(4)]
            ),
            build_prediction_record(["42", "42", "7", "8"]),
        )
    ]
    out = sampling_report(finals, n=4, k=4, denominator=1)
    assert out["pass@1"] == pytest.approx(50.0)
    assert out["avg@k"] == pytest.approx(50.0)
    assert out["pass@k"] == pytest.approx(100.0)
    # Two votes for the correct answer against one each for two wrong ones.
    assert out["maj@k"] == pytest.approx(100.0)
    assert (out["n"], out["k"], out["n_short"]) == (4.0, 4.0, 0.0)


def test_sampling_report_denominator_is_the_callers_not_len_finals():
    """A task counting failed samples as wrong passes the wider population."""
    finals = [
        _ctx(
            build_judgement_record("42", [build_rollout_judgement(0, True)]),
            build_prediction_record(["42"]),
        )
    ]
    assert sampling_report(finals, n=1, k=1, denominator=1)["pass@1"] == 100.0
    assert sampling_report(finals, n=1, k=1, denominator=2)["pass@1"] == 50.0


def test_sampling_report_always_carries_pass_at_1():
    """A task whose headline IS pass@1 reads it back out at any n, including
    from a run where every sample failed and there is nothing to aggregate."""
    for n, k in [(1, 1), (4, 4)]:
        assert sampling_report([], n=n, k=k, denominator=3)["pass@1"] == 0.0


def test_sampling_report_without_votes_never_grows_a_maj_at_k():
    """The code family, on both the scored and the empty path."""
    finals = [
        _ctx(
            build_judgement_record(
                "x", [build_rollout_judgement(i, True) for i in range(4)]
            ),
            build_prediction_record(["prog"] * 4),
        )
    ]
    assert "maj@k" not in sampling_report(finals, n=4, k=4, denominator=1, votes=False)
    assert "maj@k" not in sampling_report([], n=4, k=4, denominator=1, votes=False)


def test_count_unextracted_separates_parser_error_from_model_error():
    # Two samples, one bad draw each: the count is over ROLLOUTS, because one
    # miss in four is a different fact from four, and a run whose extractor
    # stopped matching looks identical in every other key to one whose model
    # got worse.
    finals = [
        _ctx(
            build_judgement_record(
                "42", [build_rollout_judgement(i, False) for i in range(2)]
            ),
            build_prediction_record(["42", None]),
        ),
        _ctx(
            build_judgement_record(
                "42", [build_rollout_judgement(i, False) for i in range(2)]
            ),
            build_prediction_record([None, None]),
        ),
    ]
    assert count_unextracted(finals) == 3
    assert sampling_report(finals, n=2, k=2, denominator=2)["n_unextracted"] == 3.0


def test_count_unextracted_is_zero_when_everything_parsed():
    finals = [
        _ctx(
            build_judgement_record("42", [build_rollout_judgement(0, True)]),
            build_prediction_record(["42"]),
        )
    ]
    assert count_unextracted(finals) == 0
    # Reported as 0.0 rather than omitted: absent and zero read the same to a
    # consumer, and only one of them means "checked".
    assert sampling_report(finals, n=1, k=1, denominator=1)["n_unextracted"] == 0.0


def test_warn_unscored_rollouts_counts_the_draws_the_headline_ignores():
    # A task with no `n` knob still receives one from the model config. The
    # extra draws are graded and stored; the headline scores the first alone.
    finals = [
        _ctx(
            build_judgement_record(
                "A", [build_rollout_judgement(i, True) for i in range(4)]
            ),
            build_prediction_record(["A"] * 4),
        ),
        _ctx(
            build_judgement_record("A", [build_rollout_judgement(0, True)]),
            build_prediction_record(["A"]),
        ),
    ]
    assert warn_unscored_rollouts(finals, knob="tasks.x.args") == 3
    assert warn_unscored_rollouts(finals[1:], knob="tasks.x.args") == 0
    assert warn_unscored_rollouts([], knob="tasks.x.args") == 0


def test_sampling_report_counts_a_short_draw_and_drops_the_majority():
    # Two of the four requested rollouts came back. `n_short` says so, and maj@k
    # is dropped rather than voted on the half that arrived -- a truncated draw
    # is whatever finished first, not a random subsample.
    finals = [
        _ctx(
            build_judgement_record(
                "42", [build_rollout_judgement(i, True) for i in range(2)]
            ),
            build_prediction_record(["42", "42"]),
        )
    ]
    out = sampling_report(finals, n=4, k=4, denominator=1)
    assert out["n_short"] == 1.0
    assert "maj@k" not in out
    # pass@k needs k of them too, so a short draw scores 0 there -- which is
    # exactly why n_short has to be in the report and not only in a log line.
    assert out["pass@k"] == 0.0
    assert out["pass@1"] == pytest.approx(100.0)
