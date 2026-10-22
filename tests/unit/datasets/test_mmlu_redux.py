"""
Unit tests for the MMLU-Redux dataset loader.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from unittest.mock import patch

import pytest
from datasets import Dataset as HFDataset

import sieval.datasets.mmlu_redux as mmlu_redux_module
from sieval.core.datasets.meta import get_dataset_meta
from sieval.datasets.mmlu_redux import (
    MMLU_REDUX_REVISION,
    MMLU_REDUX_SUBJECTS,
    MMLUReduxDataset,
)


def _rows(subject: str) -> HFDataset:
    return HFDataset.from_list(
        [
            {
                "question": f"{subject} q",
                "choices": ["a", "b", "c", "d"],
                "answer": 3,
                "error_type": "wrong_groundtruth",
                "source": "https://example.com",
                "correct_answer": "1",
                "potential_reason": None,
            }
        ]
    )


def test_source_revision_is_pinned():
    assert MMLU_REDUX_REVISION == "3720db6aeb3d019de48bf37916c1a54074ff4997"
    assert get_dataset_meta(MMLUReduxDataset).source == (
        f"hf:edinburgh-dawg/mmlu-redux@{MMLU_REDUX_REVISION}",
    )


def test_subjects_are_30_and_sorted():
    assert len(set(MMLU_REDUX_SUBJECTS)) == 30
    assert list(MMLU_REDUX_SUBJECTS) == sorted(MMLU_REDUX_SUBJECTS)


def test_load_merges_subjects_and_keeps_annotation_columns():
    with patch.object(
        mmlu_redux_module,
        "load_dataset",
        side_effect=lambda path, subject, split, **kw: _rows(subject),
    ) as mock:
        loaded = MMLUReduxDataset("/staged/edinburgh-dawg/mmlu-redux").dataset_dict

    assert [c.args[1] for c in mock.call_args_list] == list(MMLU_REDUX_SUBJECTS)
    assert all(c.kwargs["split"] == "test" for c in mock.call_args_list)
    test = loaded["test"]
    assert test.num_rows == 30
    assert test["subject"] == list(MMLU_REDUX_SUBJECTS)
    # Upstream row shape preserved: no error_type filtering, no answer re-keying.
    assert test[0] == {
        "question": "anatomy q",
        "choices": ["a", "b", "c", "d"],
        "answer": 3,
        "error_type": "wrong_groundtruth",
        "source": "https://example.com",
        "correct_answer": "1",
        "potential_reason": None,
        "subject": "anatomy",
    }


def test_empty_subject_is_refused():
    def load(_path, subject, **_kwargs):
        rows = _rows(subject)
        return rows.select([]) if subject == "virology" else rows

    with (
        patch.object(mmlu_redux_module, "load_dataset", side_effect=load),
        pytest.raises(ValueError, match="'virology' is empty"),
    ):
        MMLUReduxDataset("/staged/edinburgh-dawg/mmlu-redux")
