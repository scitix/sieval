"""Unit tests for the AA-LCR community prompt + scoring kernels.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import pytest

from sieval.community.aa_lcr import (
    aggregate_metrics,
    build_prompt,
    parse_grade,
)


def test_build_prompt_wraps_and_orders_documents():
    prompt = build_prompt(["first doc", "second doc"], "What is the trend?")
    # Per-document markers are 1-based and in the given order.
    assert "BEGIN DOCUMENT 1:\nfirst doc\nEND DOCUMENT 1" in prompt
    assert "BEGIN DOCUMENT 2:\nsecond doc\nEND DOCUMENT 2" in prompt
    assert prompt.index("first doc") < prompt.index("second doc")
    # Outer framing + question.
    assert prompt.startswith("BEGIN INPUT DOCUMENTS")
    assert "END INPUT DOCUMENTS" in prompt
    assert "START QUESTION\n\nWhat is the trend?\n\nEND QUESTION" in prompt


def test_build_prompt_single_document_no_join_separator():
    prompt = build_prompt(["only doc"], "q")
    assert "BEGIN DOCUMENT 1:\nonly doc\nEND DOCUMENT 1" in prompt
    assert "BEGIN DOCUMENT 2" not in prompt


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("CORRECT", "CORRECT"),
        ("correct", "CORRECT"),
        ("The answer is CORRECT.", "CORRECT"),
        ("INCORRECT", "INCORRECT"),
        # "INCORRECT" contains "correct" as a substring — must not misread.
        ("incorrect", "INCORRECT"),
        # Verbose replies mentioning both: the last verdict wins.
        ("This is not correct, so INCORRECT", "INCORRECT"),
        ("INCORRECT — the candidate contradicts the official answer", "INCORRECT"),
        # Negated phrasing must NOT fall through to the bare CORRECT token.
        ("The candidate answer is not correct.", "INCORRECT"),
        ("Not correct", "INCORRECT"),
        # Reasoning-then-verdict: the final verdict wins over earlier mentions.
        ("<think>maybe INCORRECT?</think>\nCORRECT", "CORRECT"),
        # A word starting with "CORRECT" must not match the CORRECT token.
        ("CORRECTNESS: INCORRECT", "INCORRECT"),
        ("correctness: incorrect", "INCORRECT"),
        # Hedged verdicts are INCORRECT: the checker is binary and requires the
        # candidate be *consistent with* the official answer, so a partial match
        # is not CORRECT. Reading these as CORRECT would inflate accuracy.
        ("Partially correct.", "INCORRECT"),
        ("PARTLY CORRECT", "INCORRECT"),
        ("SEMI-CORRECT", "INCORRECT"),
        ("Mostly correct", "INCORRECT"),
        ("somewhat correct", "INCORRECT"),
        ("almost correct", "INCORRECT"),
        ("partially incorrect", "INCORRECT"),
        # A hedge that does not qualify the verdict must not suppress it.
        ("The candidate is almost identical to the official. CORRECT", "CORRECT"),
        # Ordinary lead-ins still read as a bare verdict.
        ("Verdict: CORRECT", "CORRECT"),
        ("", "INCORRECT"),
        ("gibberish", "INCORRECT"),
    ],
)
def test_parse_grade(reply: str, expected: str):
    assert parse_grade(reply) == expected


def test_aggregate_metrics_accuracy():
    m = aggregate_metrics(["CORRECT", "CORRECT", "INCORRECT", "INCORRECT"])
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["is_correct"] == pytest.approx(0.5)
    assert m["is_incorrect"] == pytest.approx(0.5)


def test_aggregate_metrics_empty_is_zero():
    m = aggregate_metrics([])
    assert m["accuracy"] == 0.0
    assert m["is_correct"] == 0.0
    assert m["is_incorrect"] == 0.0
