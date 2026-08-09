"""The case-preserving LaTeX reading, and the three mechanisms it closes.

Each accepting case below was a real wrong verdict on recorded model output, and each
has a wrong-answer twin, so a "fix" that merely loosened the grader would fail here.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import pytest
import sympy

from sieval.community import ugmathbench
from sieval.community.ugmathbench import (
    _parse_latex_strict,
    _sympy_candidates,
    judge_answers,
)


@pytest.mark.parametrize(
    "gold,pred,want",
    [
        # 1. math_verify lower-cases symbols; the sympy-source gold does not.
        (["13*pi*R^2"], [r"13 \pi R^{2}"], True),
        (["s*W-3*sin(6*x)"], [r"W s - 3 \sin{\left(6 x \right)}"], True),
        # 2. the sympy reader keeps case but cannot read LaTeX; `ln` splits into l*n.
        (["0.5*ln(7+x^4)+C"], [r"\frac{1}{2} \ln(7 + x^4) + C"], True),
        # 3. parse_latex reads \pi as Symbol('pi'), not the constant. Wrong without
        #    the binding even though the case-preserving read itself succeeds.
        (["4*pi*R^3/3"], [r"\frac{4}{3} \pi R^{3}"], True),
        # a model that simplifies correctly must not be punished for it
        (["8*[2*cos(t)]^2"], [r"32 \cos^2 t"], True),
        # lower-case control: already passed before, must keep passing
        (["13*pi*r^2"], [r"13 \pi r^{2}"], True),
        # wrong answers stay wrong
        (["13*pi*R^2"], [r"14 \pi R^{2}"], False),
        (["13*pi*R^2"], [r"13 \pi R^{3}"], False),
        (["0.5*ln(7+x^4)+C"], [r"\frac{1}{3} \ln(7 + x^4) + C"], False),
        (["x+1"], [r"x + 2"], False),
    ],
)
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
    assert expr.free_symbols  # letters became symbols, not an error
    # how it splits them is a sympy detail and not worth pinning; what must hold is that
    # the nonsense reading never matches an unrelated gold
    assert judge_answers(["dog"], ["13*pi*R^2"], ["EX"]) == [False]
    assert judge_answers([r"\pi"], ["dog"], ["EX"]) == [False]


def test_a_post_parse_failure_yields_no_candidate_rather_than_raising(monkeypatch):
    """`parse_latex` succeeding is not the end of the reading, so neither is the guard.

    It builds unflattened `evaluate=False` trees, so `free_symbols` and `subs` recurse
    over them and raise RecursionError on a deep enough prediction -- below the parse,
    where a guard wrapping only the parse would not catch it.
    """
    import sympy.parsing.latex

    class Exploding(sympy.Basic):
        @property
        def free_symbols(self):
            raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(sympy.parsing.latex, "parse_latex", lambda _text: Exploding())
    assert _parse_latex_strict(r"\pi r^{2}") == []


def test_one_readings_failure_does_not_discard_the_others(monkeypatch):
    """A parser that blows up must cost only its own candidates.

    Collecting all three readings under a single guard let a late failure throw away
    readings that had already succeeded. That direction matters: the substitution pass
    runs only after the others have said "not equal", so it can only move a verdict
    wrong-to-right -- and a lost candidate can therefore only move one right-to-wrong,
    which is the class this change exists to shrink.
    """

    def explode(_text):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(ugmathbench, "_parse_latex_strict", explode)
    # the sympy-source reading of the gold is untouched by the LaTeX reader's failure
    assert _sympy_candidates("0.5*ln(7+x^4)+C")
    # and a verdict that never needed the LaTeX reading still stands
    assert judge_answers([r"13 \pi r^{2}"], ["13*pi*r^2"], ["EX"]) == [True]


def test_an_uncomparable_candidate_does_not_evict_the_kept_ones(monkeypatch):
    """Dedup compares with `==`, which sorts factors and recurses over those same
    unflattened trees. A candidate that cannot be compared is dropped on its own."""

    class Uncomparable(sympy.Basic):
        def __eq__(self, other):
            raise RecursionError("maximum recursion depth exceeded")

        __hash__ = sympy.Basic.__hash__

    monkeypatch.setattr(
        ugmathbench, "_parse_latex_strict", lambda _text: [Uncomparable()]
    )
    kept = _sympy_candidates("0.5*ln(7+x^4)+C")
    assert kept and all(not isinstance(item, Uncomparable) for item in kept)
