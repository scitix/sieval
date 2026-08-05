"""
Pinning tests for the vendored PlatinumBench parse/score logic.

``sieval/community/platinum_bench.py`` is a byte-faithful port, so these tests
are not "does this look right" tests — they pin the upstream behaviours the task
layer is written against, including the two that upstream expresses as exceptions
its runner swallows. If a future re-port changes any of them, the task's
verdicts change with it.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import pytest

from sieval.community.platinum_bench import check_prediction, get_parse_fn

_parse_math = get_parse_fn("math")


# ---------------------------------------------------------------------------
# get_parse_fn("math") — the strategy all five shipped subsets declare
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        # The prompt asks for a trailing "Answer: <n>" line, so this is the
        # designed path.
        ("Answer: 42", "42"),
        ("Reasoning about apples.\nAnswer: 42", "42"),
        # Markdown bold/heading noise is stripped before the split.
        ("**Answer: 42**", "42"),
        ("#### Answer: 42", "42"),
        # Thousands separators are removed, not treated as a decimal point.
        ("Answer: 1,000", "1000"),
        # A trailing ".0" is stripped so exact float equality still holds for
        # models that answer "18.0" to an integer question.
        ("Answer: 18.0", "18"),
        ("Answer: 18.000", "18"),
        # Negatives keep their sign.
        ("Answer: -7", "-7"),
        # No "Answer:" line but a \boxed{} — recovered via the tex pattern.
        ("So the total is $\\boxed{1000.0}$.", "1000"),
        # Neither marker: the last line is used as the answer section.
        ("Step one.\nThe result is 30 apples", "30"),
        # Trailing units after the number are dropped by the regex.
        ("Answer: 5 apples", "5"),
    ],
)
def test_parse_math_extracts_number(output, expected):
    assert _parse_math(output) == expected


def test_parse_math_raises_attribute_error_when_no_digit():
    # Upstream's `re.search(...).group()` on None. run_benchmark.py catches this
    # with a bare `except` and stores 'parsing error'; the task layer catches it
    # explicitly and records prediction=None. Either way the row scores wrong —
    # but the exception type is the contract, so pin it.
    with pytest.raises(AttributeError):
        _parse_math("Answer: I cannot determine the value.")


def test_get_parse_fn_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Invalid parsing strategy"):
        get_parse_fn("not_a_strategy")


def test_get_parse_fn_ships_all_five_strategies():
    # Kept whole on purpose: a later non-math subset needs no re-port.
    for strategy in ("math", "multiple_choice", "bbh_multiple_choice", "text", "squad"):
        assert callable(get_parse_fn(strategy))


# ---------------------------------------------------------------------------
# check_prediction — float equality for math subsets, membership otherwise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subset", ["gsm8k", "svamp", "multiarith", "singleop", "singleq"]
)
def test_math_subsets_take_the_float_comparison_branch(subset):
    # The branch is selected by a hardcoded dataset-name list. All five shipped
    # subsets must be in it, or scoring silently degrades to string membership.
    assert check_prediction("42", ["42"], "prompt", subset) is True
    # String-unequal but float-equal still counts, which is the point of the
    # branch.
    assert check_prediction("42.0", ["42"], "prompt", subset) is True


def test_math_comparison_is_exact_not_tolerant():
    # No isclose(): 42.0001 is simply wrong. This is why the parse function has
    # to strip a trailing ".0" itself.
    assert check_prediction("42.0001", ["42"], "prompt", "gsm8k") is False


def test_math_comparison_reads_only_the_first_target():
    # `float(platinum_target[0])` — a second gold answer is unreachable on the
    # math branch. All 953 shipped math rows have exactly one.
    assert check_prediction("7", ["42", "7"], "prompt", "gsm8k") is False


def test_non_math_dataset_falls_back_to_membership():
    assert check_prediction("paris", ["paris", "france"], "prompt", "squad") is True
    assert check_prediction("42", ["42.0"], "prompt", "squad") is False


def test_parsing_error_guard_is_dead_code():
    # Upstream guards on the capitalized 'Parsing error' while its runner stores
    # the lowercase 'parsing error', so the guard never fires and float() raises.
    # Documented in the vendor's docstring; pinned here because the task layer
    # relies on it NOT protecting anything (it filters None itself).
    with pytest.raises(ValueError):
        check_prediction("parsing error", ["42"], "prompt", "gsm8k")
    # The one value the guard does catch takes the membership branch instead.
    assert check_prediction("Parsing error", ["42"], "prompt", "gsm8k") is False


def test_unparseable_numeric_string_raises_value_error():
    # The parse regex accepts "1.2.3" (any run of digits and dots), which float()
    # then rejects. The task layer catches this; pin that it is a ValueError.
    with pytest.raises(ValueError):
        check_prediction("1.2.3", ["42"], "prompt", "gsm8k")
