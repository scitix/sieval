"""Only reported token counts enter canonical usage; zero remains measured.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

import pytest
from sieval_decisions.providers._wire import normalize_usage


@pytest.mark.parametrize("value", [-1, True, 1.5, "7", None])
@pytest.mark.parametrize("key", ["input_tokens", "output_tokens", "total_tokens"])
def test_invalid_usage_is_not_coerced(value, key):
    with pytest.raises(ValueError, match="non-negative integer"):
        normalize_usage({key: value})


def test_missing_usage_is_distinct_from_reported_zero():
    assert normalize_usage(None) is None
    assert normalize_usage({"input_tokens": 0, "output_tokens": 0}) == {
        "input_tokens": 0,
        "output_tokens": 0,
    }
    assert normalize_usage({"input_tokens": 7, "output_tokens": 0}) == {
        "input_tokens": 7,
        "output_tokens": 0,
    }


def test_optional_usage_breakdowns_keep_reported_zero_without_double_counting():
    usage = normalize_usage(
        {
            "input_tokens": 10,
            "output_tokens": 0,
            "total_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 5},
            "output_tokens_details": {"reasoning_tokens": 0},
        }
    )
    assert usage == {
        "input_tokens": 10,
        "output_tokens": 0,
        "total_tokens": 10,
        "cached_tokens": 0,
        "cache_write_tokens": 5,
        "reasoning_tokens": 0,
    }


@pytest.mark.parametrize(
    "container, key",
    [
        ("input_tokens_details", "cached_tokens"),
        ("input_tokens_details", "cache_write_tokens"),
        ("output_tokens_details", "reasoning_tokens"),
    ],
)
def test_invalid_breakdown_is_rejected(container, key):
    with pytest.raises(ValueError):
        normalize_usage({container: {key: -1}})
