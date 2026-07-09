"""Unit tests for IMO-Bench answer grading + gen-mode normalization.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from sieval.community.imo_bench import (
    normalize_answer,
    parse_answer,
    verify_math_answer,
)


def _grade(gold: str, pred: str | None) -> bool:
    """Mirror the task's feedback: normalize both, then the verbatim upstream
    grader with symmetric $-wrapping (HMMT-aligned)."""
    g, p = normalize_answer(gold), normalize_answer(pred)
    if g is None or p is None:
        return False
    return bool(verify_math_answer(f"${g}$", f"${p}$"))


# --- vendored upstream verify_math_answer / parse_answer (unchanged) --------


def test_integer_match():
    assert verify_math_answer("3", "3") is True
    assert verify_math_answer("3", "4") is False


def test_latex_equivalence_via_math_verify():
    assert verify_math_answer("\\frac{1}{2}", "0.5") is True


def test_unparseable_falls_back_to_normalized_string():
    assert verify_math_answer("Yes", "yes") is True
    assert verify_math_answer("red", "blue") is False


def test_parse_answer_handles_bare_latex():
    assert parse_answer("\\frac{1}{2}") != []
    assert parse_answer("") == []


# --- gen-mode normalize_answer (parsing layer) ------------------------------


def test_normalize_strips_wrappers_and_prefix():
    assert normalize_answer("$2^{u-2}$") == "2^{u-2}"
    assert normalize_answer("A(x)=x+1") == "x+1"
    assert normalize_answer("180^\\circ") == "180"
    assert normalize_answer(None) is None
    assert normalize_answer("$$") is None


def test_normalize_strips_trailing_qualifier_only():
    # a trailing "for/where/such/with" qualifier is dropped ...
    assert normalize_answer("2x^3 \\text{ for any real constant}") == "2x^3"
    # ... but a meaningful inline \text (piecewise condition) is kept.
    assert "if" in (normalize_answer("x \\text{ if } x \\ge 2") or "")


# --- effective grading: normalize + verbatim math_verify --------------------


def test_grading_recovers_math_equivalence():
    assert _grade("$\\sqrt{2}+1$", "1+\\sqrt{2}") is True  # commutativity
    assert _grade("$12^{10}$", "2^{20} \\cdot 3^{10}") is True  # factoring
    assert _grade("847288609444", "3^{25} + 1") is True  # evaluation
    assert _grade("1/2,1,2", "\\left\\{\\frac{1}{2}, 1, 2\\right\\}") is True  # set
    assert _grade("$180 - 2\\alpha$", "180^\\circ - 2\\alpha") is True  # degree
    assert (
        _grade("$A(x)=\\frac{1}{2}(x^2-x-4)$", "\\frac{1}{2}x^2 - \\frac{1}{2}x - 2")
        is True
    )


def test_grading_no_false_positive():
    assert _grade("5", "7") is False
    assert _grade("f(x)=2x^3+c", "f(x)=2x^3") is False  # missing the +c
    assert _grade("\\{1,2,3\\}", "\\{1,2,4\\}") is False  # distinct sets
    assert _grade("5", None) is False


def test_grading_prose_is_out_of_scope():
    # Prose answers can't be math-verified — left wrong (needs the agentic answer
    # channel or an LLM judge, per reference_impl.notes), never a false positive.
    assert (
        _grade("1 and odd prime numbers", "1 \\text{ and all odd prime numbers}")
        is False
    )
