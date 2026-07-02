"""Unit tests for the MathArena-aligned prompt + answer extraction helpers.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from sieval.community.matharena import (
    AIME_INSTRUCTION,
    HMMT_INSTRUCTION,
    build_prompt,
    extract_answer,
    extract_boxed_answer,
    extract_last_integer,
)


def test_build_prompt_appends_problem():
    assert (
        build_prompt(HMMT_INSTRUCTION, "P?")
        == "Put your final answer within \\boxed{}.\n\nP?"
    )
    assert AIME_INSTRUCTION.endswith("between 0 and 999 inclusive.")
    assert "\\boxed{}" in AIME_INSTRUCTION


def test_extract_boxed_takes_last_box():
    assert extract_boxed_answer("first \\boxed{42}, then \\boxed{277}") == "277"


def test_extract_boxed_balanced_braces():
    assert extract_boxed_answer("\\boxed{\\frac{7}{2}}") == "\\frac{7}{2}"


def test_extract_boxed_strips_inner_box():
    assert extract_boxed_answer("\\boxed{\\boxed{8\\sqrt{6}}}") == "8\\sqrt{6}"


def test_extract_boxed_fbox_supported():
    assert extract_boxed_answer("\\fbox{1230}") == "1230"


def test_extract_boxed_absent_returns_none():
    assert extract_boxed_answer("no box, just 503 here") is None


def test_extract_last_integer():
    assert extract_last_integer("a 12 b 503 done") == "503"
    assert extract_last_integer("none") is None


def test_extract_answer_prefers_box_over_integer():
    # 99 appears after the box but the boxed value wins.
    assert extract_answer("\\boxed{277} ... 99", strict_parsing=False) == "277"


def test_extract_answer_nonstrict_falls_back_to_integer():
    assert extract_answer("final answer is 156", strict_parsing=False) == "156"


def test_extract_answer_strict_no_box_returns_none():
    assert extract_answer("final answer is 156", strict_parsing=True) is None


def test_extract_walks_back_over_approximation_box():
    # The last box is a bare "≈ …" approximation → prefer the earlier exact box
    # (matharena parser.py walk-back; the old hand-rolled parser returned the
    # trailing approximation).
    text = "Exact: \\boxed{\\sqrt{2}}, numerically \\boxed{\\approx 1.41421}"
    assert extract_answer(text, strict_parsing=False) == "\\sqrt{2}"


def test_extract_walks_back_over_decimal_approximation():
    # The last box is a pure decimal → prefer the earlier exact (\frac) box.
    text = "value is \\boxed{\\frac{1}{3}} \\approx \\boxed{0.333}"
    assert extract_answer(text, strict_parsing=False) == "\\frac{1}{3}"


def test_extract_boxed_recursive_braces():
    assert extract_boxed_answer("\\boxed{\\frac{a}{b}^{2}}") == "\\frac{a}{b}^{2}"


def test_extract_last_integer_is_unsigned_word_bounded():
    # matharena uses \b\d+\b: the leading sign is not part of the integer.
    assert extract_last_integer("the result is -5") == "5"
