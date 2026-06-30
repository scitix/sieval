"""Unit tests for the vendored IMO-Bench answer verification.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from sieval.community.imo_bench import parse_answer, verify_math_answer


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
