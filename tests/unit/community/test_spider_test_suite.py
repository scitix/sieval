"""Pinning tests for the vendored Spider test-suite comparison.

``sieval/community/spider_test_suite/`` is upstream byte-for-byte, and one of
those bytes is an **unseeded** ``random.choice``. Every other vendored grader in
this tree is deterministic, so a reader is right to stop at that line: an
unseeded draw inside a scoring path normally means the score is not reproducible.

`sieval/tasks/_spider_test_suite.py` states the argument that it cannot move a
verdict, and leaves upstream unpatched on the strength of it. This file is the
measurement that argument is worth leaving unpatched for, in the three parts it
breaks into:

* the RNG path is **reached** at all (a test that quietly took the ``num_cols <=
  3`` bypass would prove nothing);
* the draw **does** change the work done, so the property is not vacuous;
* the verdict does not change, and the mechanism behind that -- the true column
  permutation surviving every draw -- holds directly.

Seeding upstream instead would be one line. It is not taken because it buys a
divergence from the published metric in exchange for a guarantee already held,
and because a seed would hide this measurement rather than make it unnecessary.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import random

import pytest

from sieval.community.spider_test_suite.exec_eval import (
    get_constraint_permutation,
    permute_tuple,
    result_eq,
)

#: Four columns with *partially* overlapping value domains. Both halves matter:
#: four is above upstream's ``num_cols <= 3`` bypass, and the overlap is what
#: gives the pruning real work -- with disjoint domains one row would fix the
#: permutation, and with identical domains nothing would ever be pruned.
DOMAINS = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]

#: The column permutation applied to build the second result set.
PERM = (2, 0, 3, 1)

#: How many different draws each invariance test samples. Upstream draws 20 rows
#: per call, so a handful of seeds would not separate the draws.
DRAWS = 300


def make_pair(seed=0, rows=40):
    """A result set and a column-permuted copy of it: denotationally equal."""
    rng = random.Random(seed)
    result1 = [tuple(rng.choice(domain) for domain in DOMAINS) for _ in range(rows)]
    return result1, [permute_tuple(row, PERM) for row in result1]


def pruned_space(result1, result2):
    """The permutations upstream will actually enumerate for the current draw."""
    num_cols = len(result1[0])
    sets_by_column = [{row[i] for row in result1} for i in range(num_cols)]
    return list(get_constraint_permutation(sets_by_column, result2))


# --- the RNG path is reached -------------------------------------------------


@pytest.mark.parametrize(
    ("num_cols", "expected_draws"),
    [
        # Upstream returns the full product without sampling at 3 columns or
        # fewer. Pinned so the tests below cannot pass by taking that bypass.
        (3, 0),
        # 20 draws, upstream's fixed sample size.
        (4, 20),
    ],
)
def test_the_sampling_path_is_taken_only_above_three_columns(
    monkeypatch, num_cols, expected_draws
):
    calls = []
    real_choice = random.choice
    monkeypatch.setattr(
        random, "choice", lambda seq: (calls.append(1), real_choice(seq))[1]
    )

    rows = [tuple(range(num_cols)), tuple(range(num_cols))]
    pruned_space(rows, rows)
    assert len(calls) == expected_draws


# --- the draw changes the work ----------------------------------------------


def test_different_draws_prune_to_different_search_spaces():
    """Otherwise 'only the runtime varies' is half a claim.

    A row whose values are outside every column's domain empties the constraints
    when it is sampled and is invisible when it is not, so the space upstream
    enumerates is either empty or not depending on the draw. Both outcomes reach
    the same verdict; the test below is what says so.
    """
    result1, result2 = make_pair()
    result2 = [(6, 1, 6, 1), *result2[1:]]  # not a permutation of any result1 row

    sizes = set()
    for seed in range(DRAWS):
        random.seed(seed)
        sizes.add(len(pruned_space(result1, result2)))
    assert len(sizes) > 1, sizes


# --- the verdict does not ----------------------------------------------------


def test_the_true_permutation_survives_every_draw():
    """The mechanism, asserted directly rather than through the verdict.

    Every permutation upstream discards is discarded on a *necessary* condition:
    a value present in one result's column and absent from the other's rules
    that pairing out whatever else was sampled. So an unlucky draw can only fail
    to prune -- it cannot prune the answer away. Fails if a future re-port makes
    the condition merely sufficient (say, by testing the sampled row against a
    subset of the column rather than the whole of it).
    """
    result1, result2 = make_pair()
    # `permute_tuple(row, perm)` builds `[row[i] for i in perm]`, so the
    # permutation that maps result2 back onto result1 is PERM's inverse.
    inverse = tuple(PERM.index(column) for column in range(len(PERM)))

    for seed in range(DRAWS):
        random.seed(seed)
        assert inverse in pruned_space(result1, result2), seed


@pytest.mark.parametrize("equivalent", [True, False])
def test_the_verdict_is_invariant_across_draws(equivalent):
    """The property the task's docstring rests on, as an outcome.

    Run under `-p no:randomly` or with a fixed global seed this would pass
    trivially, which is why each iteration seeds a *different* draw explicitly
    rather than relying on the suite's RNG state.
    """
    result1, result2 = make_pair()
    if not equivalent:
        result2 = [(6, 1, 6, 1), *result2[1:]]

    verdicts = set()
    for seed in range(DRAWS):
        random.seed(seed)
        verdicts.add(result_eq(result1, result2, order_matters=False))
    assert verdicts == {equivalent}


def test_an_unpruned_search_still_reaches_the_same_verdict():
    """The no-pruning extreme, which the partial-overlap fixture never hits.

    With one shared domain across all four columns no draw can prune anything,
    so upstream enumerates all 256 permutations every time. Included because it
    is the case where the RNG provably cannot matter, and it must agree with the
    cases where it fires.
    """
    rng = random.Random(0)
    result1 = [tuple(rng.choice([1, 2, 3]) for _ in range(4)) for _ in range(40)]
    result2 = [permute_tuple(row, PERM) for row in result1]

    random.seed(0)
    assert len(pruned_space(result1, result2)) == 256
    assert result_eq(result1, result2, order_matters=False) is True
