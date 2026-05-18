"""Tests for engine-param key normalization helpers.

AI-Generated Code - Claude Opus 4.7 (Anthropic)
"""

import pytest

from sieval.infer.params import merge_params, normalize_param_key


class TestNormalizeParamKey:
    def test_dash_form_becomes_underscore(self):
        assert normalize_param_key("tensor-parallel-size") == "tensor_parallel_size"

    def test_underscore_form_passes_through(self):
        assert normalize_param_key("tensor_parallel_size") == "tensor_parallel_size"

    def test_mixed_form(self):
        assert normalize_param_key("foo-bar_baz") == "foo_bar_baz"

    def test_empty_string(self):
        assert normalize_param_key("") == ""


class TestMergeParams:
    def test_empty_input_returns_empty_dict(self):
        assert merge_params() == {}

    def test_single_source_passes_through(self):
        assert merge_params({"foo_bar": 1}) == {"foo_bar": 1}

    def test_single_source_normalizes_dash_form(self):
        assert merge_params({"foo-bar": 1}) == {"foo_bar": 1}

    def test_cross_source_later_wins_same_form(self):
        assert merge_params({"a": 1}, {"a": 2}) == {"a": 2}

    def test_cross_source_later_wins_across_forms(self):
        # recipe preset in underscore, user override in dash — user wins
        assert merge_params({"foo_bar": 0}, {"foo-bar": 42}) == {"foo_bar": 42}

    def test_cross_source_later_wins_reverse_forms(self):
        # recipe preset in dash, user override in underscore — user wins
        assert merge_params({"foo-bar": 0}, {"foo_bar": 42}) == {"foo_bar": 42}

    def test_cross_source_non_colliding_keys_both_preserved(self):
        assert merge_params({"a": 1}, {"b-c": 2}) == {"a": 1, "b_c": 2}

    def test_intra_source_dash_underscore_collision_raises(self):
        with pytest.raises(ValueError, match="foo-bar.*foo_bar|foo_bar.*foo-bar"):
            merge_params({"foo-bar": 1, "foo_bar": 2})

    def test_three_sources_chain_last_wins(self):
        result = merge_params(
            {"foo_bar": 0},
            {"foo-bar": 1},
            {"foo_bar": 2},
        )
        assert result == {"foo_bar": 2}

    def test_returns_dict_not_mapping(self):
        # Callers mutate the result (e.g. params["dtype"] = ...); ensure mutable dict
        result = merge_params({"a": 1})
        result["b"] = 2
        assert result == {"a": 1, "b": 2}
