"""Unit tests for the ComplexConstraints rubric-grading assets.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import pytest

from sieval.community.complex_constraints import (
    aggregate_metrics,
    build_grader_prompt,
    format_criteria,
    parse_verdicts,
)

# --- prompt assembly ---


def test_format_criteria_numbers_from_one():
    assert format_criteria(["alpha", "beta"]) == "1. alpha\n2. beta"


def test_build_grader_prompt_carries_prompt_response_and_count():
    prompt = build_grader_prompt("do the thing", "here you go", ["a", "b", "c"])
    # The original prompt must reach the judge: most criteria are written
    # against it and are uncheckable from the response alone.
    assert "do the thing" in prompt
    assert "here you go" in prompt
    assert "1. a\n2. b\n3. c" in prompt
    # The count is interpolated in both places the template asks for it, so a
    # judge told to grade "3 criteria" cannot be handed a different rubric size.
    assert "Grade all 3 criteria" in prompt
    assert "3: <verdict>" in prompt


# --- verdict parsing ---


def test_parse_verdicts_reads_indexed_block():
    reply = "1: PASS\n2: FAIL\n3: PASS"
    assert parse_verdicts(reply, 3) == [True, False, True]


@pytest.mark.parametrize(
    "line",
    [
        "1: PASS",
        "1. PASS",
        "1) PASS",
        "1 - PASS",
        "- 1: PASS",
        "* **1: PASS**",
        "**1:** PASS",
        "Criterion 1: pass",
    ],
)
def test_parse_verdicts_tolerates_judge_formatting(line: str):
    # Judges wrap verdicts in list bullets and markdown emphasis; a parser that
    # only accepts the bare form silently reports every criterion unparsed.
    assert parse_verdicts(line, 1) == [True]


def test_parse_verdicts_last_verdict_per_index_wins():
    # A reasoning judge revises: the block it is asked to end with is the answer.
    reply = "Working through it: 1: PASS, but on reflection no.\n\n1: FAIL\n2: PASS"
    assert parse_verdicts(reply, 2) == [False, True]


def test_parse_verdicts_missing_index_is_none_not_false():
    # None and False must stay distinct -- the caller scores both as
    # not-satisfied but counts None separately as judge format drift.
    assert parse_verdicts("1: PASS\n3: PASS", 3) == [True, None, True]


def test_parse_verdicts_empty_reply_is_all_none():
    assert parse_verdicts("", 2) == [None, None]


def test_parse_verdicts_ignores_out_of_range_indices():
    # A hallucinated "3: PASS" is not evidence about a 2-criterion rubric, and
    # must not be clamped onto criterion 2.
    assert parse_verdicts("1: PASS\n2: FAIL\n3: PASS\n0: PASS", 2) == [True, False]


def test_parse_verdicts_ignores_mid_sentence_prose():
    # Anchored to line starts: a sentence mentioning a number and the word PASS
    # is not a verdict line.
    assert parse_verdicts("The response would need 2 more sections to PASS.", 2) == [
        None,
        None,
    ]


# --- verdict parsing: the echoed-instruction hazard ---
#
# The one misparse that would inflate a score rather than depress it, and inflate
# it invisibly: a parsed verdict is by definition not counted in `n_grader_unparsed`, so
# the drift counter reads 0 while every criterion scores a pass. Pinned here are
# the parser rejecting any line that names both verdict words, and the template
# not containing the "<PASS|FAIL>" spelling for a judge to echo. The template
# guard reaches no further than that spelling -- a template has to name both
# words somewhere -- so the parser is what covers the rest.


@pytest.mark.parametrize(
    "line",
    [
        "1: <PASS|FAIL>",
        "1: PASS|FAIL",
        "1: **<PASS|FAIL>**",
        "1: <FAIL|PASS>",
        # The separator is incidental: a judge restating the format writes it
        # whichever way it likes, and every spelling inflates identically.
        "1: PASS or FAIL",
        "1: PASS/FAIL",
        "1: PASS, FAIL",
        "1: <PASS> or <FAIL>",
        "1: **PASS** or **FAIL**",
    ],
)
def test_parse_verdicts_rejects_a_line_naming_both_verdicts(line: str):
    # A judge restating its instructions has reported nothing. Scoring this as a
    # pass is worse than scoring it unparsed, because unparsed is counted.
    assert parse_verdicts(line, 1) == [None]


def test_parse_verdicts_still_reads_a_bracketed_single_verdict():
    # The guard must reject naming BOTH words, not the angle brackets -- a judge
    # that fills the placeholder in as "<PASS>" has answered.
    assert parse_verdicts("1: <PASS>\n2: <FAIL>", 2) == [True, False]


def test_parse_verdicts_still_reads_a_verdict_with_a_trailing_rationale():
    # Rejecting the pair must not reject a verdict that merely carries a reason:
    # only a line naming the OTHER verdict word too is a restated format.
    assert parse_verdicts("1: PASS - lists all four sections\n2: FAIL.", 2) == [
        True,
        False,
    ]


def test_parse_verdicts_drops_a_verdict_whose_rationale_names_the_other_word():
    # The cost of the pair rule, pinned rather than left to be discovered: a
    # rationale mentioning the other word makes its own verdict unreadable. That
    # is one lost verdict, counted in `n_grader_unparsed` and scored not-satisfied --
    # the direction that cannot flatter a model.
    assert parse_verdicts("1: FAIL (would PASS if it listed the risks)", 1) == [None]


def test_template_does_not_self_parse():
    # The likeliest thing a confused or length-truncated judge emits verbatim is
    # the format it was just asked for. So no substring of the prompt may read as
    # a verdict: if it did, an echoed instruction -- or a reply cut off right
    # after restating the format -- would score a full rubric.
    prompt = build_grader_prompt("do the thing", "here you go", ["a", "b", "c"])
    assert parse_verdicts(prompt, 3) == [None, None, None]


def test_parse_verdicts_truncated_block_leaves_the_tail_unparsed():
    # A grader cut off mid-block is the safe direction: the tail is counted and
    # scored not-satisfied, biasing DOWN. Pinned so the guard above is not
    # mistaken for covering truncation generally.
    assert parse_verdicts("1: PASS\n2: PASS\n3:", 3) == [True, True, None]


# --- aggregation ---


def test_aggregate_metrics_empty_is_zero():
    m = aggregate_metrics([])
    assert m["task_pass_rate"] == 0.0
    assert m["criterion_pass_rate_macro"] == 0.0
    assert m["criterion_pass_rate_micro"] == 0.0


def test_aggregate_metrics_task_pass_requires_every_criterion():
    # 10/10 passes the task; 9/10 does not, despite a 0.9 criterion rate.
    m = aggregate_metrics([(10, 10), (9, 10)])
    assert m["task_pass_rate"] == pytest.approx(0.5)
    assert m["criterion_pass_rate_macro"] == pytest.approx(0.95)


def test_aggregate_metrics_macro_and_micro_diverge_on_uneven_rubrics():
    # 5/10 and 30/40: macro averages the two rates (0.5, 0.75) => 0.625;
    # micro pools 35/50 => 0.70. Reporting one number for both would be wrong
    # for whichever reading the reader assumed.
    m = aggregate_metrics([(5, 10), (30, 40)])
    assert m["criterion_pass_rate_macro"] == pytest.approx(0.625)
    assert m["criterion_pass_rate_micro"] == pytest.approx(0.70)


def test_aggregate_metrics_zero_criteria_unit_is_a_failure_not_a_pass():
    # A failure whose rubric size could not be recovered enters as (0, 0). It
    # must dilute the rates, never satisfy "every criterion" vacuously.
    m = aggregate_metrics([(10, 10), (0, 0)])
    assert m["task_pass_rate"] == pytest.approx(0.5)
    assert m["criterion_pass_rate_macro"] == pytest.approx(0.5)
    # It adds nothing to the pooled denominator, so micro stays 10/10.
    assert m["criterion_pass_rate_micro"] == pytest.approx(1.0)
