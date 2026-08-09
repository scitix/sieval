"""Tests for the answer helpers shared by the math-competition tasks.

Only ``normalize_vote`` is covered here. ``verify_answer`` is a two-line delegate
to ``math_verify`` and is exercised through the per-task grading tests, which
have the optional ``math`` dependency group they need.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import pytest

from sieval.tasks._math_verify import normalize_vote


@pytest.mark.parametrize(
    "a,b",
    [
        (r"\dfrac{1}{2}", r"\frac{1}{2}"),
        (r"\tfrac{1}{2}", r"\frac{1}{2}"),
        (r"\left(3\right)", "(3)"),
        ("1/2", r"\frac{1}{2}"),
        (r"\sqrt3", r"\sqrt{3}"),
    ],
)
def test_equivalent_spellings_share_one_vote_key(a, b):
    # Each pair is one answer written two ways. Splitting them costs a majority:
    # three rollouts agreeing 2-1 in substance read as a three-way tie, and a tie
    # is not a majority -- so maj@k would report 0.0 for a self-consistent model.
    assert normalize_vote(a) == normalize_vote(b)


def test_genuinely_different_answers_keep_different_keys():
    # The guard against over-normalizing: a canonicalizer that collapsed these
    # would manufacture a majority rather than miss one, which is the direction
    # that cannot be detected downstream.
    assert normalize_vote("42") != normalize_vote("43")
    assert normalize_vote(r"\frac{1}{2}") != normalize_vote(r"\frac{1}{3}")


@pytest.mark.parametrize("answer", ["\\frac", "\\sqrt", "x\\frac", "2\\sqrt"])
def test_an_answer_that_breaks_the_canonicalizer_falls_back_to_itself(answer):
    # `strip_string` indexes into the string it is repairing, so a trailing
    # `\frac` / `\sqrt` raises IndexError. These are raw model answers reaching
    # report() at the end of a scored run: an unclustered vote costs one cluster,
    # an exception costs the whole report.
    from sieval.community.math import strip_string

    with pytest.raises(IndexError):
        strip_string(answer)
    assert normalize_vote(answer) == answer


def test_it_does_not_bridge_string_and_symbolic_equality():
    # Documented limitation, asserted so it stays a known lower bound rather
    # than becoming a surprise: maj@k clusters on strings where the grader
    # compares symbolically, so these are two votes and not one.
    assert normalize_vote("0.5") != normalize_vote(r"\frac{1}{2}")
