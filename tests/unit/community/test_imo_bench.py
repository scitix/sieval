"""Unit tests for the vendored IMO-Bench answer verification.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from sieval.community.imo_bench import (
    parse_answer,
    verify_answer_gen,
    verify_math_answer,
)


def test_integer_match():
    assert verify_math_answer("3", "3") is True
    assert verify_math_answer("3", "4") is False


def test_latex_equivalence_via_math_verify():
    # math-verify treats these as equal even though the strings differ.
    assert verify_math_answer("\\frac{1}{2}", "0.5") is True


def test_unparseable_falls_back_to_normalized_string():
    # Non-mathy answers can't be parsed -> case/space-insensitive string compare.
    assert verify_math_answer("Yes", "yes") is True
    assert verify_math_answer("red", "blue") is False


def test_parse_answer_handles_bare_latex():
    # Bare LaTeX without $...$ is retried wrapped in a math environment.
    assert parse_answer("\\frac{1}{2}") != []
    assert parse_answer("") == []


def test_gen_recovers_formatting_only_differences():
    # $-wrapping / whitespace / \left\right / trailing newline: clearly equal.
    assert verify_answer_gen("$2^{u-2}$", "2^{u-2}") is True
    assert verify_answer_gen("(0, 0)", "(0,0)") is True
    assert verify_answer_gen("$2^n$\n", "2^n") is True
    assert verify_answer_gen("$\\frac{3}{2}(XZ-XY)$", "\\frac{3}{2}(XZ - XY)") is True


def test_gen_recovers_multi_answer_lists():
    # comma-separated set + "and"/"or" list separators (agent would submit clean).
    assert verify_answer_gen("3,7", "3 \\text{ and } 7") is True
    assert (
        verify_answer_gen(
            "P(x)=-1, P(x)=x+1",
            "P(x) = -1 \\quad\\text{or}\\quad P(x) = x+1",
        )
        is True
    )


def test_gen_is_conservative_no_false_positive():
    # Different parameterization / prose-vs-formula must stay wrong (no over-count).
    assert (
        verify_answer_gen(
            "$X(y)=1+(u-1)\\bar{y}$",
            "X(z)=c\\overline{z}+1 \\text{ for some } c \\text{ with } |c|=1",
        )
        is False
    )
    assert (
        verify_answer_gen(
            "$n=2k, n=3k$",
            "\\text{All } n \\ge 2 \\text{ divisible by } 2 \\text{ or } 3",
        )
        is False
    )
    assert verify_answer_gen("5", "7") is False
    assert verify_answer_gen("5", None) is False


def test_gen_empty_after_normalize_is_not_a_match():
    # Both reduce to "" under _normalize (\text{...} stripped) — must NOT grade
    # equal; upstream verify_math_answer returns False here.
    assert verify_answer_gen("\\text{No}", "\\text{Yes}") is False
    # Identical text answers are still matched by the verify_math_answer fast path.
    assert verify_answer_gen("\\text{Yes}", "\\text{Yes}") is True
