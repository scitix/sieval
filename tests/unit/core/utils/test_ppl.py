"""
Tests for sieval.core.utils.ppl.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import pytest

from sieval.core.utils.ppl import (
    choice_scores_from_top_logprobs,
    extract_option_logprob,
    total_logprob,
)

CHOICES = ("A", "B", "C", "D")


class TestChoiceScoresFromTopLogprobs:
    def test_all_present(self):
        top = [{"A": -1.0, "B": -0.1, "C": -2.0, "D": -3.0}]
        scores, all_present = choice_scores_from_top_logprobs(top, CHOICES)
        assert all_present is True
        assert scores == {"A": -1.0, "B": -0.1, "C": -2.0, "D": -3.0}
        assert max(scores, key=lambda k: scores[k]) == "B"

    def test_space_prefixed_tokens_map_to_labels(self):
        top = [{" A": -0.2, " B": -1.0, " C": -1.5, " D": -2.0}]
        scores, all_present = choice_scores_from_top_logprobs(top, CHOICES)
        assert all_present is True
        assert scores["A"] == -0.2

    def test_missing_choice_reports_not_all_present(self):
        top = [{"A": -0.2, "B": -1.0, "C": -1.5}]  # no D
        scores, all_present = choice_scores_from_top_logprobs(top, CHOICES)
        assert all_present is False
        assert scores["D"] == float("-inf")

    def test_duplicate_token_keeps_max(self):
        top = [{"A": -2.0, " A": -0.5, "B": -1.0, "C": -1.0, "D": -1.0}]
        scores, all_present = choice_scores_from_top_logprobs(top, CHOICES)
        assert all_present is True
        assert scores["A"] == -0.5  # max over the two "A" spellings

    def test_non_option_tokens_are_ignored(self):
        # A real top-k is mostly non-option tokens; only A/B/C/D must be kept,
        # and distractors must not leak into scores or inflate all_present.
        top = [{"A": -1.0, "B": -0.1, "C": -2.0, "D": -3.0, "的": -0.5, " the": -4.0}]
        scores, all_present = choice_scores_from_top_logprobs(top, CHOICES)
        assert all_present is True
        assert set(scores) == set(CHOICES)
        assert max(scores, key=lambda k: scores[k]) == "B"

    def test_empty_or_none(self):
        expected = dict.fromkeys(CHOICES, float("-inf"))
        assert choice_scores_from_top_logprobs(None, CHOICES) == (expected, False)
        assert choice_scores_from_top_logprobs([], CHOICES) == (expected, False)


class TestExtractOptionLogprob:
    def test_exact_match(self):
        tokens = ["The", " answer", " is", " A"]
        logprobs = [-1.0, -0.5, -0.3, -0.1]
        assert extract_option_logprob(tokens, logprobs, "A") == -0.1

    def test_prefix_space_match(self):
        tokens = ["The", " answer", " is", " B"]
        logprobs = [-1.0, -0.5, -0.3, -0.2]
        assert extract_option_logprob(tokens, logprobs, "B") == -0.2

    def test_exact_prefixed_token_branch(self):
        tokens = ["x", "  A"]
        logprobs = [-1.0, -0.25]
        # Keep leading space to hit token == f" {option_label}" branch.
        assert extract_option_logprob(tokens, logprobs, " A") == -0.25

    def test_short_token_inclusion(self):
        # Token is "A." (len <= 2 and contains "A")
        tokens = ["The", " answer", " is", "A."]
        logprobs = [-1.0, -0.5, -0.3, -0.15]
        assert extract_option_logprob(tokens, logprobs, "A") == -0.15

    def test_not_found(self):
        tokens = ["The", " answer", " is", " unknown"]
        logprobs = [-1.0, -0.5, -0.3, -0.1]
        assert extract_option_logprob(tokens, logprobs, "A") is None

    def test_empty_tokens(self):
        assert extract_option_logprob([], [], "A") is None

    def test_none_logprob_skipped(self):
        tokens = ["A", "B"]
        logprobs = [None, -0.5]
        # "B" at index 1 matches
        assert extract_option_logprob(tokens, logprobs, "B") == -0.5

    def test_all_none_logprobs(self):
        tokens = ["A", "B"]
        logprobs = [None, None]
        assert extract_option_logprob(tokens, logprobs, "A") is None

    def test_max_search_limits_scan(self):
        # Put the match far back, beyond max_search
        tokens = ["A"] + ["x"] * 20
        logprobs = [-0.1] + [-1.0] * 20
        # max_search=5 means only last 5 tokens are scanned
        assert extract_option_logprob(tokens, logprobs, "A", max_search=5) is None

    def test_mismatched_lengths(self):
        tokens = ["A", "B", "C"]
        logprobs = [-0.1, -0.2]
        # Should use min(len(tokens), len(logprobs)) = 2
        assert extract_option_logprob(tokens, logprobs, "B") == -0.2

    def test_searches_backwards(self):
        # Two matches — should find the last one (searches from end)
        tokens = ["A", "B", "A"]
        logprobs = [-0.1, -0.5, -0.9]
        assert extract_option_logprob(tokens, logprobs, "A") == -0.9


class TestTotalLogprob:
    def test_basic(self):
        tokens = ["The", " cat", " sat"]
        logprobs = [-1.0, -0.5, -0.3]
        total, count = total_logprob(tokens, logprobs)
        # skip_first=True by default, so skip index 0
        assert total == pytest.approx(-0.8)
        assert count == 2

    def test_no_skip_first(self):
        tokens = ["The", " cat", " sat"]
        logprobs = [-1.0, -0.5, -0.3]
        total, count = total_logprob(tokens, logprobs, skip_first=False)
        assert total == pytest.approx(-1.8)
        assert count == 3

    def test_empty(self):
        total, count = total_logprob([], [])
        assert total == 0.0
        assert count == 0

    def test_none_logprobs_skipped(self):
        tokens = ["a", "b", "c"]
        logprobs = [None, -0.5, None]
        total, count = total_logprob(tokens, logprobs)
        assert total == pytest.approx(-0.5)
        assert count == 1

    def test_single_token(self):
        tokens = ["a"]
        logprobs = [-1.0]
        # skip_first=True, so nothing to sum
        total, count = total_logprob(tokens, logprobs)
        assert total == 0.0
        assert count == 0

    def test_single_token_no_skip(self):
        tokens = ["a"]
        logprobs = [-1.0]
        total, count = total_logprob(tokens, logprobs, skip_first=False)
        assert total == pytest.approx(-1.0)
        assert count == 1

    def test_mismatched_lengths(self):
        tokens = ["a", "b", "c", "d"]
        logprobs = [-1.0, -0.5]
        total, count = total_logprob(tokens, logprobs)
        # min length = 2, skip first, so only index 1
        assert total == pytest.approx(-0.5)
        assert count == 1
