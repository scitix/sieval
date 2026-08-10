"""Tests for the shared rollout estimators.

The properties worth pinning are the ones a copy-pasted helper drifts away from:
pass@1 is c/n and not "the first sample", pass@k is "solved at least once", and maj@k
votes on ANSWERS so it can disagree with a verdict tally in both directions.
"""

import pytest
from loguru import logger

from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
from sieval.core.tasks.metrics import (
    aggregate,
    avg_at_n,
    budget_metrics,
    count_short,
    count_unextracted,
    first_rollout_correct,
    health_metrics,
    majority_at_k,
    pass_at_k,
    pass_pow_k,
    rollout_metrics,
    rollout_view,
    sampling_report,
    self_consistency,
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
