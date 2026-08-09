"""The case-preserving LaTeX reading, and the three mechanisms it closes.

Each accepting case below was a real wrong verdict on recorded model output, and each
has a wrong-answer twin, so a "fix" that merely loosened the grader would fail here.
"""

import pytest
import sympy

from sieval.community.ugmathbench import _parse_latex_strict, judge_answers


@pytest.mark.parametrize("gold,pred,want", [
    # 1. math_verify lower-cases symbols; the sympy-source gold does not.
    (["13*pi*R^2"], [r"13 \pi R^{2}"], True),
    (["s*W-3*sin(6*x)"], [r"W s - 3 \sin{\left(6 x \right)}"], True),
    # 2. the sympy reader keeps case but cannot read LaTeX; `ln` would split into l*n.
    (["0.5*ln(7+x^4)+C"], [r"\frac{1}{2} \ln(7 + x^4) + C"], True),
    # 3. parse_latex reads \pi as Symbol('pi'), not the constant.
    (["(6+7)*pi*4"], [r"52 \pi"], True),
    # a model that simplifies correctly must not be punished for it
    (["8*[2*cos(t)]^2"], [r"32 \cos^2 t"], True),
    # lower-case control: already passed before, must keep passing
    (["13*pi*r^2"], [r"13 \pi r^{2}"], True),
    # wrong answers stay wrong
    (["13*pi*R^2"], [r"14 \pi R^{2}"], False),
    (["13*pi*R^2"], [r"13 \pi R^{3}"], False),
    (["0.5*ln(7+x^4)+C"], [r"\frac{1}{3} \ln(7 + x^4) + C"], False),
    (["x+1"], [r"x + 2"], False),
])
def test_symbolic_equivalence(gold, pred, want):
    assert judge_answers(pred, gold, ["EX"]) == [want]


def test_pi_binds_to_the_constant():
    (expr,) = _parse_latex_strict(r"\pi r^{2}")
    assert expr.has(sympy.pi)
    assert sympy.Symbol("pi") not in expr.free_symbols


def test_other_single_letters_stay_symbols():
    """`e` and `i` are ordinary variable names often enough that binding them to
    sympy's constants would manufacture agreements rather than find them."""
    (expr,) = _parse_latex_strict(r"e + i")
    assert {s.name for s in expr.free_symbols} == {"e", "i"}


def test_empty_input_yields_nothing():
    assert _parse_latex_strict("") == []
    assert _parse_latex_strict("   ") == []


def test_prose_reads_as_a_product_of_symbols_and_that_is_safe():
    """`parse_latex` does not reject non-LaTeX; it reads bare letters as symbols, the
    same behaviour that turns `ln` into l*n.

    That is why this is added as one CANDIDATE reading rather than a replacement: a
    nonsense reading of one side cannot match a sane reading of the other, because
    `_same_function` requires the two sides to involve the same set of symbol NAMES and
    demands several probe points agree. The empirical check is stronger than the
    argument -- 11,424 recorded rows regraded across two runs, 0 verdicts moved
    right-to-wrong.
    """
    (expr,) = _parse_latex_strict("dog")
    assert expr.free_symbols                      # letters became symbols, not an error
    # how it splits them is a sympy detail and not worth pinning; what must hold is that
    # the nonsense reading never matches an unrelated gold
    assert judge_answers(["dog"], ["13*pi*R^2"], ["EX"]) == [False]
    assert judge_answers([r"\pi"], ["dog"], ["EX"]) == [False]
