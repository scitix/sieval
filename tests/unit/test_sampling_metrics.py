"""Tests for the shared sampling estimators.

The properties worth pinning are the ones a copy-pasted helper drifts away from:
pass@1 is c/n and not "the first sample", pass@n is "solved at least once", and maj@k
votes on ANSWERS so it can disagree with a verdict tally in both directions.
"""

import pytest

from sieval.core.tasks.sampling_metrics import (
    avg_at_k,
    majority_at_k,
    pass_at_k,
    sampling_metrics,
)


@pytest.mark.parametrize("n,c,k,want", [
    (4, 0, 1, 0.0),
    (4, 4, 1, 1.0),
    (4, 1, 1, 0.25),          # pass@1 IS c/n
    (4, 2, 1, 0.5),
    (4, 1, 4, 1.0),           # one correct anywhere -> pass@4 is certain
    (4, 2, 2, 1 - (2 / 4) * (1 / 3)),
    (1, 1, 1, 1.0),
    (2, 1, 4, 0.0),           # k > n is unreachable by config; must not raise
])
def test_pass_at_k(n, c, k, want):
    assert pass_at_k(n, c, k) == pytest.approx(want)


def test_pass_at_1_equals_avg():
    for c in range(5):
        verdicts = [True] * c + [False] * (4 - c)
        assert pass_at_k(4, c, 1) == pytest.approx(avg_at_k(verdicts))


def test_pass_at_k_monotone_in_k():
    for k in range(1, 4):
        assert pass_at_k(4, 2, k) <= pass_at_k(4, 2, k + 1)


def test_majority_needs_answers_not_verdicts():
    # 2 right agreeing vs 2 wrong disagreeing: a verdict tally sees 2/4, the modal
    # ANSWER is correct, so self-consistency wins.
    assert majority_at_k([True, True, False, False], ["a", "a", "b", "c"]) == 1.0
    # the mirror: one right, three wrong but agreeing -> the majority is wrong
    assert majority_at_k([True, False, False, False], ["a", "b", "b", "b"]) == 0.0


def test_majority_ignores_empty_answers():
    assert majority_at_k([True, True], [None, None]) == 0.0
    assert majority_at_k([True, False], ["", "x"]) == 0.0


def test_majority_uses_the_normalizer():
    # unnormalised the two correct answers are distinct, so no answer has a plurality;
    # normalising merges them into a clear majority.
    args = ([True, True, False, False], ["1/2", "0.5", "9", "8"])
    assert majority_at_k(*args) == 0.0
    assert majority_at_k(*args, normalize=lambda s: {"1/2": "0.5"}.get(s, s)) == 1.0


def test_majority_is_order_independent():
    """A tie is not a majority -- otherwise the metric depends on sample order."""
    assert majority_at_k([True, False], ["a", "b"]) == 0.0
    assert majority_at_k([False, True], ["b", "a"]) == 0.0
    # and a real plurality is found wherever it sits
    for i in range(3):
        ans = ["b", "b"]; ok = [False, False]
        ans.insert(i, "a"); ok.insert(i, True)
        assert majority_at_k(ok, ans) == 0.0        # 2 vs 1 -> the WRONG answer wins


def test_majority_rejects_mismatched_lengths():
    assert majority_at_k([True, True], ["a"]) == 0.0


def test_sampling_metrics_omits_what_it_cannot_compute():
    # no answers -> no maj@k, rather than a 0.0 indistinguishable from a real 0.0
    assert set(sampling_metrics([True, False], None, k=2)) == {"pass@1", "avg@k", "pass@2"}
    # k == 1 -> no pass@1 duplicate under a second name
    assert set(sampling_metrics([True], None, k=1)) == {"pass@1", "avg@k"}
    assert set(sampling_metrics([True, False], ["a", "b"], k=2)) == {
        "pass@1", "avg@k", "pass@2", "maj@2"}


def test_empty_input_is_zero_not_an_error():
    assert avg_at_k([]) == 0.0
    assert majority_at_k([], []) == 0.0
    assert sampling_metrics([], None, k=1)["pass@1"] == 0.0
