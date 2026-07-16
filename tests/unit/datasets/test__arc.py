"""
Unit tests for the shared AI2 ARC dataset normalization.

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

import pytest

from sieval.datasets._arc import process_arc_sample


def test_process_arc_sample_maps_letter_answer_key_to_index():
    sample = {
        "question": "Which is hottest?",
        "choices": {"label": ["A", "B", "C"], "text": ["ice", "fire", "snow"]},
        "answerKey": "B",
    }

    assert process_arc_sample(sample) == {
        "question": "Which is hottest?",
        "choices": ["ice", "fire", "snow"],
        "answer": 1,
    }


def test_process_arc_sample_maps_numeric_label_to_index():
    # Some ARC rows label choices "1".."4" instead of "A".."D"; the stored
    # answer is the position of the label, not the digit value.
    sample = {
        "question": "Pick one.",
        "choices": {"label": ["1", "2", "3", "4"], "text": ["w", "x", "y", "z"]},
        "answerKey": "1",
    }

    assert process_arc_sample(sample)["answer"] == 0


def test_process_arc_sample_raises_on_unknown_answer_key():
    sample = {
        "question": "Pick one.",
        "choices": {"label": ["A", "B"], "text": ["x", "y"]},
        "answerKey": "Z",
    }

    with pytest.raises(ValueError, match="answerKey"):
        process_arc_sample(sample)
