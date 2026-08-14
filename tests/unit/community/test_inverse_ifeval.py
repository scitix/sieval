"""Unit tests for the Inverse IFEval judge-reply parser and breakdowns.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import pytest

from sieval.community.inverse_ifeval import (
    INSTRUCTION_TYPES,
    LANGUAGES,
    PAPER_COLUMNS,
    breakdown_metrics,
    parse_answer_score,
    type_key,
)

# The shipped judge output contract: a rationale, a score line, then a fenced
# JSON block. Reproduced here so a parser change has to keep reading the real
# shape, not a simplified one.
_SHIPPED_REPLY = """【评分依据】：学生答案中“跳绳”一词后没有加emoji。
【评分】：0分
【JSON】：
```
{"answer_score": 0}
```"""


# --- the shipped format parses, both verdicts ---


@pytest.mark.parametrize("score", [0, 1])
def test_parses_the_shipped_reply_format(score: int):
    reply = _SHIPPED_REPLY.replace('"answer_score": 0', f'"answer_score": {score}')
    reply = reply.replace("【评分】：0分", f"【评分】：{score}分")
    assert parse_answer_score(reply) == (score, str(score))


@pytest.mark.parametrize(
    "reply",
    [
        '```json\n{"answer_score": 1}\n```',
        '{"answer_score": 1}',
        "{'answer_score': 1}",
        '{"answer_score" : 1}',
        '{"answer_score"：1}',  # full-width colon
        '{"answer_score": 1.0}',  # a float still means the `1` tier
        "answer_score: 1",
    ],
)
def test_tolerates_judge_formatting_drift(reply: str):
    score, _raw = parse_answer_score(reply)
    assert score == 1


def test_falls_back_to_the_score_line_when_no_json_block():
    # Same prompts mandate `【评分】：X分`; a judge that skips the JSON fence has
    # still stated a verdict, and dropping it would count a real 1 as a failure.
    assert parse_answer_score("【评分依据】：符合要求。\n【评分】：1分") == (1, "1")


def test_fenced_block_wins_over_an_inconsistent_score_line():
    # The shipped prompts make the JSON block the authoritative output, so a
    # judge that contradicts itself is read the way it was instructed to be.
    reply = '【评分】：1分\n【JSON】：\n```\n{"answer_score": 0}\n```'
    assert parse_answer_score(reply) == (0, "0")


def test_last_verdict_wins_when_the_judge_restates_examples():
    # A judge that echoes a worked example (score 1) before its own verdict (0)
    # must be read as 0 — the real verdict comes last.
    reply = (
        '【示例3】：\n【JSON】：\n```\n{"answer_score": 1}\n```\n'
        '以下是正式评分：\n【JSON】：\n```\n{"answer_score": 0}\n```'
    )
    assert parse_answer_score(reply) == (0, "0")


# --- the cases that must NOT silently become a pass ---


@pytest.mark.parametrize(
    "reply", ["", "   \n\t ", "抱歉，我无法完成。", "【评分依据】："]
)
def test_no_verdict_is_none_never_a_pass(reply: str):
    # The one misread direction that inflates a score invisibly. An absent
    # verdict must stay absent rather than defaulting to either tier.
    assert parse_answer_score(reply) == (None, None)


@pytest.mark.parametrize("raw", ["100", "2", "-1", "0.5"])
def test_off_rubric_scores_are_not_verdicts_but_stay_inspectable(raw: str):
    # The rubric is two-tier; a judge scoring 0-100 has violated it, so there is
    # no verdict — but the token is returned so the violation is diagnosable
    # instead of collapsing into "unparseable".
    score, matched = parse_answer_score(f'{{"answer_score": {raw}}}')
    assert score is None
    assert matched == raw


def test_truncated_fence_still_yields_the_verdict():
    # A reply cut off by a token budget has no closing fence; the bare-regex
    # fallback is what keeps it from being scored as a judge failure.
    assert parse_answer_score('【JSON】：\n```\n{"answer_score": 1}') == (1, "1")


# --- no fallback may reach back into an echoed example ---
#
# One defect in three disguises: the echo is FENCED (as every shipped example is)
# and scores 1, while the judge's own verdict is not fenced, not closed, or not
# parseable. Preferring a fence over its position reads all three as a PASS.
# Asserted separately because each takes a different branch out of the anchor.


def test_an_echoed_example_does_not_outrank_an_unfenced_verdict():
    # The judge restates a worked example inside a fence, then answers WITHOUT
    # one. Its verdict is 0 and sits after its own `【JSON】`, so that is the
    # only region a verdict may be read from.
    reply = (
        '我先回顾示例：\n【JSON】：\n```\n{"answer_score": 1}\n```\n'
        "现在评判本题。\n【评分依据】：未遵守要求。\n【评分】：0分\n"
        '【JSON】：\n{"answer_score": 0}'
    )
    assert parse_answer_score(reply) == (0, "0")


def test_a_truncated_verdict_block_falls_back_to_the_score_line_not_the_echo():
    # Same echo, but the real block is cut off mid-JSON by a token budget. The
    # `【评分】` line the judge did finish states 0; the echo states 1. Reading
    # the echo here would turn a budget overrun into a free pass.
    reply = (
        '示例：\n```\n{"answer_score": 1}\n```\n'
        "【评分依据】：不符合要求，因此\n【评分】：0分\n"
        '【JSON】：\n```\n{"answer_sco'
    )
    assert parse_answer_score(reply) == (0, "0")


def test_a_malformed_verdict_block_is_unparsed_rather_than_an_earlier_echo():
    # The authoritative block was located but carries no number. That is judge
    # format failure (scored 0, counted in `n_grader_unparsed`) — not licence to
    # reach for the echoed 1 above it.
    reply = '```\n{"answer_score": 1}\n```\n本题：\n```\n{"answer_score": null}\n```'
    assert parse_answer_score(reply) == (None, None)


# --- the self-parse hazard, from the real shipped prompts ---


def test_every_shipped_judge_system_prompt_would_read_as_a_pass():
    """Regression guard for the one hazard that inflates a score invisibly.

    Every judge system prompt the dataset ships ends with a worked example
    scoring 1, so feeding one to the parser returns PASS. This test asserts that
    property holds — it is *why* the task passes only the grader's reply, and why
    the empty case above must stay ``None`` rather than defaulting. If upstream
    ever stops ending on a passing example this test fails loudly, at which point
    the reasoning in ``parse_answer_score``'s docstring needs revisiting rather
    than silently becoming stale.
    """
    prompt_tail = """【示例3】：
<标准答案>：学生答案需要询问用户的需求。
<学生答案>：你能告诉我你遇到了什么问题吗？
【评分依据】：学生答案询问了用户的需求。
【评分】：1分
【JSON】：
```
{"answer_score": 1}
```
以下是正式的问题："""
    assert parse_answer_score(prompt_tail) == (1, "1")


# --- report key slugs ---


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Code without Comments", "code_without_comments"),
        ("Counter-Conventional Formatting", "counter_conventional_formatting"),
        ("Mid-turn Instruction Modification", "mid_turn_instruction_modification"),
        ("english", "english"),
    ],
)
def test_type_key_slugs(label: str, expected: str):
    assert type_key(label) == expected


def test_type_keys_are_distinct_across_the_declared_vocabularies():
    # A collision would have two groups overwrite each other's report key.
    keys = [type_key(name) for name in (*INSTRUCTION_TYPES, *LANGUAGES)]
    assert len(set(keys)) == len(keys)


def test_paper_columns_cover_the_eight_types_bijectively():
    # Guards the CC/CCF correction: the mapping must stay a bijection onto the
    # dataset's own type labels, or a per-type comparison lines up wrong.
    assert len(PAPER_COLUMNS) == len(INSTRUCTION_TYPES)
    assert set(PAPER_COLUMNS.values()) == set(INSTRUCTION_TYPES)
    assert PAPER_COLUMNS["CC"] == "Counter-Conventional Formatting"
    assert PAPER_COLUMNS["CCF"] == "Code without Comments"


# --- breakdowns ---


def test_breakdowns_are_per_language_and_per_type_with_counts():
    graded = [
        ("english", "Question Correction", True),
        ("english", "Question Correction", False),
        ("chinese", "Code without Comments", True),
    ]
    out = breakdown_metrics(graded)
    assert out["score_english"] == 50.0
    assert out["n_english"] == 2.0
    assert out["score_chinese"] == 100.0
    assert out["score_question_correction"] == 50.0
    assert out["n_question_correction"] == 2.0
    assert out["score_code_without_comments"] == 100.0


def test_breakdowns_omit_absent_groups_rather_than_reporting_zero():
    # A 0.0 for lack of input reads identically to a real 0.0.
    out = breakdown_metrics([("english", "Question Correction", True)])
    assert "score_chinese" not in out
    assert "score_code_without_comments" not in out


def test_breakdowns_emit_no_overall_score():
    # The headline belongs to `sampling_report`; a second one here could drift.
    out = breakdown_metrics([("english", "Question Correction", True)])
    assert "score" not in out
    assert "pass@1" not in out


def test_breakdown_of_nothing_is_empty():
    assert breakdown_metrics([]) == {}


def test_declared_vocabularies_order_the_keys_before_unknown_groups():
    # Stable key order across runs, with upstream drift still surfaced.
    graded = [
        ("klingon", "Question Correction", True),
        ("chinese", "Question Correction", True),
        ("english", "Question Correction", False),
    ]
    languages = [
        key.removeprefix("score_")
        for key in breakdown_metrics(graded)
        if key.startswith("score_") and "question" not in key
    ]
    assert languages == ["english", "chinese", "klingon"]


def test_pooled_mean_is_the_metric_not_the_macro_average():
    """The paper's Overall is a pooled mean; unequal group sizes prove which.

    Two types with very different counts and rates: pooled and macro disagree, so
    this pins the reading recovered from the published tables. The pooled number
    is the headline the task reports via ``pass@1``; per-type cells here are what
    the paper's eight columns are.
    """
    graded = [("english", "Question Correction", True)] * 9
    graded += [("english", "Question Correction", False)]
    graded += [("english", "Code without Comments", False)] * 2
    out = breakdown_metrics(graded)
    assert out["score_question_correction"] == 90.0
    assert out["score_code_without_comments"] == 0.0
    # Pooled over all 12 rollouts, which is what `score_english` is.
    assert out["score_english"] == pytest.approx(75.0)
    # The macro-average over the two types would be 45.0 — a different number.
    macro = (out["score_question_correction"] + out["score_code_without_comments"]) / 2
    assert macro == pytest.approx(45.0)
    assert out["score_english"] != pytest.approx(macro)
