"""Tests for the shared RULER loader helpers.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import pytest

from sieval.datasets.ruler import thinking_prefill

_QWEN3_TAGS = "<think>\n\n</think>\n\n"


@pytest.mark.parametrize(
    ("model_name", "enable_thinking", "expected"),
    [
        # Qwen3 with thinking off → empty think block is prefilled.
        ("Qwen/Qwen3-8B", False, _QWEN3_TAGS),
        ("qwen3-8b-instruct", False, _QWEN3_TAGS),  # case-insensitive match
        # Qwen3 with thinking on → the model emits its own block, nothing to add.
        ("Qwen/Qwen3-8B", True, ""),
        # Non-reasoning models never get the placeholder, regardless of the flag.
        ("meta-llama/Llama-3-8B", False, ""),
        ("gpt-4o", False, ""),
        ("cl100k_base", True, ""),
    ],
)
def test_thinking_prefill(model_name, enable_thinking, expected):
    assert thinking_prefill(model_name, enable_thinking) == expected
