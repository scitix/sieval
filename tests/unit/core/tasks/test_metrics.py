"""Tests for the shared rollout estimators.

The properties worth pinning are the ones a copy-pasted helper drifts away from:
pass@1 is c/n and not "the first sample", pass@k is "solved at least once", and maj@k
votes on ANSWERS so it can disagree with a verdict tally in both directions.
"""

import math

import pytest
from loguru import logger

from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
from sieval.core.tasks.metrics import (
    CI_SUFFIX,
    CI_UNITS_FIELD,
    PROBLEM_COUNT_FIELD,
    SCORE_CI_FIELD,
    ProblemGrouping,
    aggregate,
    avg_at_n,
    budget_metrics,
    ci_field,
    count_short,
    count_unextracted,
    first_rollout_correct,
    health_metrics,
    interval_metrics,
    majority_at_k,
    merge_metrics,
    metric_interval,
    pass_at_k,
    pass_pow_k,
    problem_population,
    rollout_metrics,
    rollout_view,
    sampling_report,
    self_consistency,
    warn_unscored_rollouts,
    wilson_interval,
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
        assert pass_at_k(4, c, 1) == pytest.approx(avg_at_n(verdicts))


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


def test_rollout_metrics_wires_pass_pow_k_not_a_second_pass_at_k():
    """The KEY existing proves nothing -- `pass^k` has to carry the opposite
    direction at the call site, or the pair the docs tell readers to compare is
    two names for one number and nothing fails."""
    # 2 of 4 correct: the draw where the two estimators are furthest apart.
    out = rollout_metrics([True, True, False, False], k=2)
    assert out["pass@k"] == pytest.approx(pass_at_k(4, 2, 2))
    assert out["pass^k"] == pytest.approx(pass_pow_k(4, 2, 2))
    assert out["pass^k"] < out["pass@k"]
    # Unanimity is the one place they meet; a mutant aliasing them survives a
    # test that only ever looks here.
    tight = rollout_metrics([True] * 4, k=2)
    assert tight["pass^k"] == tight["pass@k"] == pytest.approx(1.0)


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


def test_self_consistency_is_continuous_where_majority_is_thresholded():
    # The whole reason it is a separate key: both draws have a correct modal
    # answer, so maj@k cannot tell them apart -- and the second model is
    # measurably less stable.
    unanimous = ["a", "a", "a", "a"]
    split = ["a", "a", "a", "b"]
    assert majority_at_k([True] * 4, unanimous) == 1.0
    assert majority_at_k([True, True, True, False], split) == 1.0
    assert self_consistency(unanimous) == 1.0
    assert self_consistency(split) == 0.75


def test_self_consistency_is_correctness_blind():
    """A consistently WRONG model scores 1.0 -- read it beside a correctness
    key, never instead of one."""
    assert self_consistency(["b", "b", "b", "b"]) == 1.0
    assert majority_at_k([False] * 4, ["b", "b", "b", "b"]) == 0.0


def test_self_consistency_counts_unextracted_against_the_draw():
    # Denominator is the whole draw, not the answers that voted: two agreeing
    # answers out of four rollouts is not the same evidence of stability as
    # four agreeing ones, whatever happened to the other two.
    assert self_consistency(["a", "a", None, None]) == 0.5
    assert self_consistency([None, None, None, None]) == 0.0
    assert self_consistency(["", "", "a", "a"]) == 0.5
    assert self_consistency([]) == 0.0


def test_self_consistency_uses_the_normalizer():
    # Same clustering as maj@k, so the two cannot disagree about what one
    # answer is.
    args = ["1/2", "0.5", "9", "8"]
    assert self_consistency(args) == 0.25
    assert self_consistency(args, normalize=lambda s: {"1/2": "0.5"}.get(s, s)) == 0.5


def test_self_consistency_needs_no_k_equals_n_gate():
    """It describes the draw that arrived, not the budget a majority is over --
    so unlike maj@k it survives k < n."""
    four = ([True, True, False, False], ["a", "a", "b", "c"])
    assert "maj@k" not in rollout_metrics(*four, k=2)
    assert rollout_metrics(*four, k=2)["self_consistency"] == 0.5


def test_keys_are_literal_not_interpolated():
    """The budget lives in the n/k report fields, never in a column name."""
    keys = set(rollout_metrics([True, False, False, False], ["a", "b", "c", "d"], k=4))
    assert keys == {
        "pass@1",
        "avg@n",
        "pass@k",
        "pass^k",
        "maj@k",
        "self_consistency",
    }
    # nothing spells the value of k
    assert not any("4" in key for key in keys)


def test_rollout_metrics_omits_what_it_cannot_compute():
    # no answers -> no maj@k, rather than a 0.0 indistinguishable from a real 0.0
    assert set(rollout_metrics([True, False], None, k=2)) == {
        "pass@1",
        "avg@n",
        "pass@k",
        "pass^k",
    }
    # k == 1 -> no pass@k duplicate of pass@1
    assert set(rollout_metrics([True], None, k=1)) == {"pass@1", "avg@n"}
    assert set(rollout_metrics([True, False], ["a", "b"], k=2)) == {
        "pass@1",
        "avg@n",
        "pass@k",
        "pass^k",
        "maj@k",
        "self_consistency",
    }


def test_majority_only_when_k_equals_n():
    """Sub-sampling k < n would need an estimator or a seed (RFC #74 D.2)."""
    four = ([True, True, False, False], ["a", "a", "b", "c"])
    assert "maj@k" in rollout_metrics(*four, k=4)
    assert "maj@k" not in rollout_metrics(*four, k=2)
    assert "maj@k" not in rollout_metrics(*four, k=1)


def test_majority_rejects_a_sub_sample_of_the_budget_not_a_short_draw():
    """The gate is on the BUDGET (`k == requested`), never on what arrived.

    `k < n` has no definition -- "which 2 of the 4" needs a seed -- so it is
    refused whatever the draw did. A draw that came back SHORT still votes: the
    arrived count is run health, reported as `n_short`, and withholding a column
    for it is a policy no other key here applies.
    """
    short = ([True, True], ["a", "a"])
    full = ([True] * 4, ["a"] * 4)
    # k=2 does not cover the n=4 budget -- refused either way, which is what
    # keeps a draw truncated to exactly k from looking like a complete one.
    assert "maj@k" not in rollout_metrics(*short, k=2, n_requested=4)
    assert "maj@k" not in rollout_metrics(*full, k=2, n_requested=4)
    # k == n: the ordinary case, and the short draw now votes with it.
    assert "maj@k" in rollout_metrics(*full, k=4, n_requested=4)
    assert "maj@k" in rollout_metrics(*short, k=4, n_requested=4)


def test_majority_and_self_consistency_agree_on_what_is_votable():
    """Same answers, same normalizer -- so they cannot disagree on whether a
    draw is fit to cluster. Gating one on arrival and not the other is how a
    report ends up with two answers to one question."""
    for observed in (1, 2, 3, 4):
        out = rollout_metrics([True] * observed, ["a"] * observed, k=4, n_requested=4)
        assert ("maj@k" in out) == ("self_consistency" in out)


def test_majority_falls_back_to_the_observed_count():
    """Without `n_requested` the draw is assumed complete -- which is what
    `zero_metrics` needs, since it synthesizes a full one."""
    four = ([True, True, False, False], ["a", "a", "b", "c"])
    assert "maj@k" in rollout_metrics(*four, k=4)
    assert "maj@k" in rollout_metrics(*four, k=4, n_requested=4)


def test_sampling_report_does_not_vote_on_a_truncated_draw():
    # The same edge through the production entry point: every sample came back
    # two rollouts short of the requested four.
    finals = [
        _ctx(
            build_judgement_record(
                "42", [build_rollout_judgement(i, True) for i in range(2)]
            ),
            build_prediction_record(["42", "42"]),
        )
    ]
    out = sampling_report(finals, n=4, k=2, denominator=1)
    assert out["n_short"] == 1.0
    assert "maj@k" not in out


def test_avg_and_pass_at_1_both_reported_though_equal():
    """Equal arithmetic, different questions -- neither key subsumes the other."""
    metrics = rollout_metrics([True, False, False, False], k=2)
    assert metrics["pass@1"] == metrics["avg@n"]
    assert "pass@1" in metrics and "avg@n" in metrics


def test_empty_input_is_zero_not_an_error():
    assert avg_at_n([]) == 0.0
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
        (1, 1, {"pass@1", "avg@n", "maj@k", "self_consistency"}),
        # k == 1 < n: pass@k would restate pass@1. `self_consistency`
        # is not gated on k at all -- it describes the draw.
        (4, 1, {"pass@1", "avg@n", "self_consistency"}),
        # k < n: majority is undefined, but both directions of the
        # k-sample estimator are.
        (
            4,
            2,
            {"pass@1", "avg@n", "pass@k", "pass^k", "self_consistency"},
        ),
        (
            4,
            4,
            {
                "pass@1",
                "avg@n",
                "pass@k",
                "pass^k",
                "maj@k",
                "self_consistency",
            },
        ),
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


@pytest.fixture
def _finals_factory():
    """Judged finals from per-sample lists of per-rollout verdicts.

    One judged final per inner list, one rollout per verdict in it -- the
    shape `sampling_report`'s interval tests need without each spelling out a
    judgement/prediction pair by hand.
    """

    def make(samples) -> list[TaskContext]:
        finals = []
        for verdicts in samples:
            judgement = build_judgement_record(
                "42",
                [build_rollout_judgement(i, ok) for i, ok in enumerate(verdicts)],
            )
            prediction = build_prediction_record(
                ["42" if ok else "0" for ok in verdicts]
            )
            finals.append(_ctx(judgement, prediction))
        return finals

    return make


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
    assert out["avg@n"] == pytest.approx(50.0)
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
    assert health_metrics(finals)["n_unextracted"] == 3.0
    # NOT part of the sampling block: it measures the parser, not the draw.
    assert "n_unextracted" not in sampling_report(finals, n=2, k=2, denominator=2)


def test_count_unextracted_is_zero_when_everything_parsed():
    finals = [
        _ctx(
            build_judgement_record("42", [build_rollout_judgement(0, True)]),
            build_prediction_record(["42"]),
        )
    ]
    assert count_unextracted(finals) == 0
    # Reported as 0.0 rather than omitted: absent and zero read the same to a
    # consumer, and only one of them means "checked". At n=1 too -- the default
    # budget is where a silently-stopped extractor survives longest.
    assert health_metrics(finals)["n_unextracted"] == 0.0


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
    assert warn_unscored_rollouts(finals, task="x") == 3
    assert warn_unscored_rollouts(finals[1:], task="x") == 0
    assert warn_unscored_rollouts([], task="x") == 0


def test_warn_unscored_rollouts_actually_warns_and_names_the_model_config():
    """The return value has no production consumer -- every caller discards it,
    so the log line IS the behaviour. It must also point somewhere real: these
    tasks take no `n`, so telling the reader to set one per task is advice that
    raises TypeError."""
    finals = [
        _ctx(
            build_judgement_record(
                "A", [build_rollout_judgement(i, True) for i in range(4)]
            ),
            build_prediction_record(["A"] * 4),
        )
    ]
    emitted: list[str] = []
    handle = logger.add(lambda message: emitted.append(message), level="WARNING")
    try:
        warn_unscored_rollouts(finals, task="mmlu_0shot_gen")
        assert emitted, "no warning was emitted"
        text = "".join(emitted)
        assert "mmlu_0shot_gen" in text
        assert "MODEL config" in text
        # Never route the reader to a task arg that does not exist.
        assert "per task" not in text
        emitted.clear()
        warn_unscored_rollouts(finals[:0], task="mmlu_0shot_gen")
        assert not emitted, "warned with nothing to warn about"
    finally:
        logger.remove(handle)


def test_sampling_report_counts_a_short_draw_without_dropping_the_majority():
    # Two of the four requested rollouts came back. `n_short` says so, and the
    # two that arrived still vote -- one truncated sample must not remove a
    # column from every other sample in the run.
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
    assert out["maj@k"] == pytest.approx(100.0)
    # pass@k needs k of them, so a short draw scores 0 there -- which is exactly
    # why n_short has to be in the report and not only in a log line.
    assert out["pass@k"] == 0.0
    assert out["pass@1"] == pytest.approx(100.0)


def test_one_short_draw_does_not_cost_the_run_its_majority_column():
    """The regression this gate split exists to prevent: 500 clean samples plus
    a single truncated one used to report NO `maj@k` at all, while a run where
    every sample failed still reported one."""
    clean = [
        _ctx(
            build_judgement_record(
                "42", [build_rollout_judgement(i, True) for i in range(4)]
            ),
            build_prediction_record(["42"] * 4),
        )
        for _ in range(500)
    ]
    short = _ctx(
        build_judgement_record(
            "42", [build_rollout_judgement(i, True) for i in range(3)]
        ),
        build_prediction_record(["42"] * 3),
    )
    out = sampling_report([*clean, short], n=4, k=4, denominator=501)
    assert out["n_short"] == 1.0
    assert out["maj@k"] == pytest.approx(100.0)


def test_sampling_report_adds_no_interval_without_a_score_key(_finals_factory):
    got = sampling_report(_finals_factory([[True], [False]]), n=1, k=1, denominator=2)
    assert "score_ci95" not in got
    assert "n_problems" not in got


def test_sampling_report_intervals_the_named_key(_finals_factory):
    finals = _finals_factory([[True], [False], [True], [False], [True]])
    got = sampling_report(finals, n=1, k=1, denominator=5, score_key="pass@1")
    assert got["n_problems"] == 5.0
    ci = got["score_ci95"]
    assert isinstance(ci, list)
    lo, hi = ci
    pass_at_1 = got["pass@1"]
    assert isinstance(pass_at_1, float)
    assert lo < pass_at_1 < hi


def test_sampling_report_refuses_a_score_key_it_did_not_compute(_finals_factory):
    # `accuracy`, deliberately: it is never in this block's key set. `maj@k` would
    # be the wrong probe -- it IS computed whenever k == n_requested, so the test
    # would pass or fail on a fixture detail instead of on the guard.
    with pytest.raises(ValueError, match="does not compute"):
        sampling_report(
            _finals_factory([[True], [False]]),
            n=1,
            k=1,
            denominator=2,
            score_key="accuracy",
        )


def test_sampling_report_interval_is_reported_at_n_equals_one(_finals_factory):
    # Not gated on n > 1: the interval is WIDEST at n=1, which is where a reader
    # most needs it -- the same argument health_metrics already makes.
    finals = _finals_factory([[True], [False], [True], [False]])
    got = sampling_report(finals, n=1, k=1, denominator=4, score_key="pass@1")
    assert "score_ci95" in got


def test_sampling_report_wires_grouping_into_the_interval(_finals_factory):
    # 4 repeat copies of a 50-problem 50/50 split -- the same shape as
    # `test_interval_metrics_widens_by_root_times_on_a_pure_repeat`, but driven
    # through the production entry point (a ProblemGrouping) instead of calling
    # interval_metrics with raw values.
    pattern = [True, False] * 25
    verdicts = pattern * 4
    finals = _finals_factory([[v] for v in verdicts])
    keys = [g for _ in range(4) for g in range(50)]
    grouping = ProblemGrouping(keys=keys, n_problems=50)

    ungrouped = sampling_report(finals, n=1, k=1, denominator=200, score_key="pass@1")
    grouped = sampling_report(
        finals, n=1, k=1, denominator=200, score_key="pass@1", grouping=grouping
    )

    # The GROUPING's count, not len(finals) -- the tell if `keys` and
    # `n_problems` were swapped at the call site: swapped, this line either
    # reads back 200 (n_problems silently defaulted past a broken group_keys)
    # or the call raises before returning at all (`len()` on an int).
    assert grouped["n_problems"] == 50.0

    ungrouped_ci, grouped_ci = ungrouped["score_ci95"], grouped["score_ci95"]
    assert isinstance(ungrouped_ci, list)
    assert isinstance(grouped_ci, list)
    # Collapsing 200 samples into the 50 problems they are repeats of must
    # WIDEN the interval -- the whole point of passing a grouping through.
    assert (grouped_ci[1] - grouped_ci[0]) > (ungrouped_ci[1] - ungrouped_ci[0])

    # Not just "wider than ungrouped": the SAME numbers interval_metrics would
    # compute if handed these keys directly -- pinning the field wiring, not
    # only its direction.
    values = [1.0 if v else 0.0 for v in verdicts]
    want = interval_metrics(values, denominator=200, group_keys=keys, n_problems=50)
    assert grouped_ci == pytest.approx(want["score_ci95"])


# --------------------------------------------------------------------------- #
# wilson_interval -- a clustered 95% interval over problems
# --------------------------------------------------------------------------- #


def _plain_wilson(k: int, m: int, z: float = 1.96) -> tuple[float, float]:
    """Textbook Wilson score interval, as the reduction target."""
    p = k / m
    centre = (p + z * z / (2 * m)) / (1 + z * z / m)
    half = z / (1 + z * z / m) * math.sqrt(p * (1 - p) / m + z * z / (4 * m * m))
    return 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


@pytest.mark.parametrize(
    "correct,m", [(1, 30), (5, 30), (15, 30), (29, 30), (391, 500), (101, 198), (3, 7)]
)
def test_wilson_interval_reduces_to_plain_wilson_on_booleans(correct, m):
    # The population SD divisor is what makes this EXACT rather than off by the
    # m/(m-1) factor -- a sample divisor puts m_eff at m-1 and misses by 0.34pp
    # at 1/30.
    values = [1.0] * correct + [0.0] * (m - correct)
    got = wilson_interval(values, m)
    assert got is not None
    want = _plain_wilson(correct, m)
    assert got[0] == pytest.approx(want[0], abs=1e-9)
    assert got[1] == pytest.approx(want[1], abs=1e-9)


def test_wilson_interval_stays_inside_the_unit_range_at_the_extremes():
    # A Wald half-width would put the lower bound at -3.09 for 1/30. The bound
    # is exactly where saturated and very hard sets live, so it must hold.
    got = wilson_interval([1.0] + [0.0] * 29, 30)
    assert got is not None
    lo, hi = got
    assert lo > 0.0
    assert hi < 100.0


def test_wilson_interval_uses_clopper_pearson_when_nothing_was_correct():
    # p=0 leaves no dispersion to estimate, and m_eff is undefined -- but a 0.0
    # headline is exactly when a reader needs the upper bound.
    got = wilson_interval([0.0] * 30, 30)
    assert got is not None
    lo, hi = got
    assert lo == 0.0
    assert hi == pytest.approx(100 * (1 - 0.025 ** (1 / 30)), abs=1e-9)


def test_wilson_interval_uses_clopper_pearson_when_everything_was_correct():
    got = wilson_interval([1.0] * 30, 30)
    assert got is not None
    lo, hi = got
    assert lo == pytest.approx(100 * 0.025 ** (1 / 30), abs=1e-9)
    assert hi == 100.0


def test_wilson_interval_narrows_when_failures_pad_the_denominator():
    # Failed samples are FIXED ZEROS carrying no variance, so the estimator's
    # variance is m*s^2/D^2, not s^2/m. Using s^2/m would overstate the width
    # by 67% at D=50, m=30.
    values = [1.0] * 15 + [0.0] * 15
    tight = wilson_interval(values, 50)
    loose = wilson_interval(values, 30)
    assert tight is not None
    assert loose is not None
    assert tight[1] - tight[0] < loose[1] - loose[0]


def test_wilson_interval_omitted_below_two_problems():
    assert wilson_interval([1.0], 1) is None
    assert wilson_interval([], 0) is None


def test_wilson_interval_omitted_when_every_problem_scored_alike():
    # Zero observed dispersion at m >= 2 is a real signal, but not a variance
    # estimate; a zero-width interval would claim certainty the run lacks.
    assert wilson_interval([0.5] * 8, 8) is None


def test_wilson_interval_is_order_independent():
    a = wilson_interval([1.0, 0.0, 0.5, 0.25], 4)
    b = wilson_interval([0.25, 0.5, 0.0, 1.0], 4)
    assert a == b


# --------------------------------------------------------------------------- #
# interval_metrics -- collapse repeat copies, then estimate
# --------------------------------------------------------------------------- #


def test_interval_metrics_reports_the_problem_count_and_a_pair():
    got = interval_metrics([1.0] * 10 + [0.0] * 10, denominator=20)
    assert got["n_problems"] == 20.0
    assert isinstance(got["score_ci95"], list)
    assert len(got["score_ci95"]) == 2


def test_interval_metrics_collapsing_does_not_move_the_mean():
    # The whole change is additive: collapsing must leave `score` alone. Each
    # group's summed value becomes its per-problem SHARE, so
    # `sum(units) / n_problems` is still `sum(values) / denominator` -- here
    # 1.5/3 == 3/6. Stated as an equality against the UNGROUPED branch fed those
    # shares directly: that branch applies no scale of its own, so it cannot
    # agree by accident.
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    keys = [0, 1, 2, 0, 1, 2]
    grouped = interval_metrics(values, denominator=6, group_keys=keys, n_problems=3)
    assert grouped == interval_metrics([1.0, 0.0, 0.5], denominator=3)
    assert grouped["n_problems"] == 3.0
    grouped_ci = grouped["score_ci95"]
    assert isinstance(grouped_ci, list)
    # And the mean itself, read straight off the interval: at `p` exactly 0.5
    # Wilson is symmetric, so the midpoint IS the headline. A mis-scaled unit
    # moves `p` and takes the whole interval with it.
    assert (grouped_ci[0] + grouped_ci[1]) / 2 == pytest.approx(50.0)
    flat = interval_metrics(values, denominator=6)
    flat_ci = flat["score_ci95"]
    assert isinstance(flat_ci, list)
    # Same headline, wider interval -- three problems, not six.
    assert (grouped_ci[1] - grouped_ci[0]) > (flat_ci[1] - flat_ci[0])


def test_interval_metrics_widens_by_root_times_on_a_pure_repeat():
    # A 4x repeat of the same 50/50 split: the honest interval is ~2x wider.
    per_problem = [1.0, 0.0] * 25
    flat = interval_metrics(per_problem * 4, denominator=200)
    keys = [g for _ in range(4) for g in range(50)]
    grouped = interval_metrics(
        per_problem * 4, denominator=200, group_keys=keys, n_problems=50
    )
    flat_ci, grouped_ci = flat["score_ci95"], grouped["score_ci95"]
    assert isinstance(flat_ci, list)
    assert isinstance(grouped_ci, list)
    ratio = (grouped_ci[1] - grouped_ci[0]) / (flat_ci[1] - flat_ci[0])
    assert 1.8 < ratio < 2.2


def test_interval_metrics_omits_both_keys_together_when_it_cannot_estimate():
    assert interval_metrics([0.5] * 4, denominator=4) == {}
    assert interval_metrics([1.0], denominator=1) == {}


def test_interval_metrics_omits_the_pair_on_an_empty_denominator_when_grouped():
    # The two paths must not disagree about an impossible input. Ungrouped,
    # `wilson_interval`'s own `denominator <= 0` guard refuses it -- but the
    # grouped path hands that function `n_problems`, not the denominator, so the
    # guard never sees this. Scaling the units to zero would then read as
    # `p == 0` and draw a Clopper-Pearson bound over a mean of nothing.
    values = [1.0, 0.0, 1.0, 0.0]
    assert interval_metrics(values, denominator=0) == {}
    assert (
        interval_metrics(values, denominator=0, group_keys=[0, 1, 2, 3], n_problems=4)
        == {}
    )


def test_interval_metrics_keeps_the_slot_of_a_wholly_failed_problem():
    # 4 problems x 2 copies under `requested`, and both copies of problem 3
    # failed: 6 finals, denominator still 8. The DECLARED population is what the
    # interval is quoted over, so `n_problems` stays 4 and the width is the
    # 4-problem width -- not the 3-problem one the observed groups would give.
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    keys = [0, 0, 1, 1, 2, 2]
    got = interval_metrics(values, denominator=8, group_keys=keys, n_problems=4)
    assert got["n_problems"] == 4.0
    # scale is 4/8, so the shares are exact: 1.0, 2.0 and 0.0 halved.
    assert got == interval_metrics([0.5, 1.0, 0.0], denominator=4)


def test_interval_metrics_rejects_a_grouping_that_does_not_align():
    with pytest.raises(ValueError, match="one key per value"):
        interval_metrics([1.0, 0.0], denominator=2, group_keys=[0], n_problems=1)


def test_interval_metrics_needs_the_problem_count_with_the_keys():
    with pytest.raises(ValueError, match="n_problems"):
        interval_metrics([1.0, 0.0], denominator=2, group_keys=[0, 0])


# --------------------------------------------------------------------------- #
# the per-metric contract -- an interval, its population, and its unit
# --------------------------------------------------------------------------- #


def test_the_interval_field_names_are_the_suffix_rule_applied():
    """The spelled-out headline constant must BE what `ci_field` derives.

    It is a literal because a static reader has to name it; that is exactly what
    lets it drift from the rule everything else follows.
    """
    assert ci_field("score") == SCORE_CI_FIELD
    assert ci_field("pass@k") == f"pass@k{CI_SUFFIX}"


@pytest.mark.parametrize(
    "values,denominator,keys,problems",
    [
        ([1.0, 0.0, 1.0, 1.0, 0.0], 5, None, None),
        ([1.0, 0.0, 1.0, 1.0, 0.0], 8, None, None),
        ([0.25, 0.5, 0.75, 0.0], 4, None, None),
        ([1.0, 0.0] * 4, 8, [0, 1, 2, 3] * 2, 4),
        ([0.5] * 4, 4, None, None),  # nothing to estimate -> both empty
    ],
)
def test_interval_metrics_is_metric_interval_pinned_to_the_headline(
    values, denominator, keys, problems
):
    """The duplicated three-key dict cannot drift: it is pinned by equality.

    `interval_metrics` spells `score_ci95` / `n_problems` out as literals so the
    preflight can name them, which means the same fragment is now built in two
    places. Byte-equality with the general function is what keeps the second
    spelling honest.
    """
    assert interval_metrics(
        values, denominator=denominator, group_keys=keys, n_problems=problems
    ) == metric_interval(
        "score", values, denominator=denominator, group_keys=keys, n_problems=problems
    )


def test_interval_metrics_declares_the_unit_its_interval_is_clustered_on():
    got = interval_metrics([1.0] * 6 + [0.0] * 6, denominator=12)
    assert got[CI_UNITS_FIELD] == {"score": PROBLEM_COUNT_FIELD}
    # The declaration names a key the same fragment writes: an entry pointing at
    # a population nothing published is unreadable.
    units = got[CI_UNITS_FIELD]
    assert isinstance(units, dict)
    assert all(unit in got for unit in units.values())


def test_metric_interval_publishes_a_rate_on_its_own_population_key():
    got = metric_interval(
        "aacc", [1.0, 0.0, 1.0, 1.0], denominator=4, unit="n_versions"
    )
    assert set(got) == {"aacc_ci95", "n_versions", CI_UNITS_FIELD}
    assert got[CI_UNITS_FIELD] == {"aacc": "n_versions"}
    # The problem count is NOT written: a version-level rate borrowing it would
    # quote a per-version interval over a population of problems.
    assert PROBLEM_COUNT_FIELD not in got


def test_metric_interval_omits_the_declaration_with_the_interval():
    # No interval, no population, no unit entry -- a declaration for a key that
    # is not there describes nothing.
    assert metric_interval("pass@k", [0.5] * 4, denominator=4) == {}
    assert metric_interval("pass@k", [1.0], denominator=1) == {}


def test_merge_metrics_unions_declarations_a_plain_merge_would_drop():
    headline = interval_metrics([1.0, 0.0, 1.0, 1.0], denominator=4)
    versions = metric_interval(
        "aacc", [1.0, 0.0, 0.0, 1.0], denominator=4, unit="n_versions"
    )
    # The failure this function exists to prevent: `|` keeps only the LAST
    # declaration, so `score_ci95` survives with no unit recorded anywhere.
    plain = headline | versions
    plain_units = plain[CI_UNITS_FIELD]
    assert isinstance(plain_units, dict)
    assert plain_units == {"aacc": "n_versions"}
    assert "score" not in plain_units

    merged = merge_metrics(headline, versions)
    assert merged[CI_UNITS_FIELD] == {
        "score": PROBLEM_COUNT_FIELD,
        "aacc": "n_versions",
    }
    # Everything else is a plain merge: both intervals and both populations.
    assert merged[SCORE_CI_FIELD] == headline[SCORE_CI_FIELD]
    assert merged["aacc_ci95"] == versions["aacc_ci95"]
    assert merged["n_versions"] == versions["n_versions"]


def test_merge_metrics_lets_a_later_fragment_win_every_other_key():
    assert merge_metrics({"a": 1.0, "b": 2.0}, {"b": 3.0}) == {"a": 1.0, "b": 3.0}


def test_merge_metrics_refuses_one_metric_on_two_populations():
    with pytest.raises(ValueError, match="one population"):
        merge_metrics(
            {CI_UNITS_FIELD: {"eacc": PROBLEM_COUNT_FIELD}},
            {CI_UNITS_FIELD: {"eacc": "n_versions"}},
        )
    # Re-declaring the SAME unit is not a conflict -- every fragment of one
    # block declares the population it shares.
    assert merge_metrics(
        {CI_UNITS_FIELD: {"eacc": PROBLEM_COUNT_FIELD}},
        {CI_UNITS_FIELD: {"eacc": PROBLEM_COUNT_FIELD}},
    ) == {CI_UNITS_FIELD: {"eacc": PROBLEM_COUNT_FIELD}}


def test_merge_metrics_refuses_a_declaration_that_is_not_a_map():
    with pytest.raises(ValueError, match="map of metric"):
        merge_metrics({CI_UNITS_FIELD: 1.0})


class _Final:
    def __init__(self, sample_id):
        self.sample_id = sample_id


def test_problem_population_passes_a_real_grouping_through():
    """A repeated split already counts problems, so it is returned untouched."""
    grouping = ProblemGrouping([7, 7, 8], 2)
    out = problem_population(grouping, [], n_problems=999)
    assert out is grouping


def test_problem_population_gives_one_group_per_sample_when_unrepeated():
    out = problem_population(None, [_Final(4), _Final(9)], n_problems=3)
    assert out.n_problems == 3
    # One key per sample, and DISTINCT -- "each sample is its own problem".
    assert len(out.keys) == 2
    assert len(set(out.keys)) == 2


def test_problem_population_does_not_collapse_samples_sharing_a_sample_id():
    """Positional keys, not `sample_id`.

    Keying on `sample_id` would merge these two into one group, which drops the
    interval entirely at two samples -- and a task whose contexts happen to
    repeat an id is a fixture away, not a hypothetical.
    """
    out = problem_population(None, [_Final(0), _Final(0)], n_problems=2)
    assert len(set(out.keys)) == 2


def test_problem_population_leaves_the_interval_untouched():
    """The whole point: it corrects what `n_problems` REPORTS, nothing else.

    4 problems at n=4 with one wholly-failed problem. `n_problems` moves from the
    rollout count to the problem count; `score_ci95` must not move at all,
    because the factor cancels out of both `p` and the variance.
    """
    values = [4.0, 0.0, 2.0]
    denominator = 4 * 4

    ungrouped = interval_metrics(values, denominator=denominator)
    grouping = problem_population(None, [_Final(i) for i in range(3)], n_problems=4)
    grouped = interval_metrics(
        values,
        denominator=denominator,
        group_keys=grouping.keys,
        n_problems=grouping.n_problems,
    )

    assert ungrouped[PROBLEM_COUNT_FIELD] == 16.0  # rollouts — the old, wrong unit
    assert grouped[PROBLEM_COUNT_FIELD] == 4.0  # problems
    assert grouped[SCORE_CI_FIELD] == ungrouped[SCORE_CI_FIELD]
