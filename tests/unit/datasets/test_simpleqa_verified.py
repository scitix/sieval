"""Unit tests for the SimpleQA Verified dataset wrapper.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from unittest.mock import patch

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

import sieval.datasets.simpleqa_verified as sqav_module
from sieval.core.datasets.meta import get_dataset_meta
from sieval.datasets.simpleqa_verified import (
    SIMPLEQA_VERIFIED_REVISION,
    SimpleQAVerifiedDataset,
)


def _hf_dict(rows: int = 1) -> HFDatasetDict:
    row = {
        "original_index": 0,
        "problem": "Who wrote Hamlet?",
        "answer": "William Shakespeare",
        "topic": "Art",
        "answer_type": "Person",
        "multi_step": False,
        "requires_reasoning": False,
        "urls": "['https://en.wikipedia.org/wiki/Hamlet']",
    }
    ds = HFDataset.from_list([row] * rows)
    return HFDatasetDict({"test": ds})


def test_source_pins_hf_revision():
    meta_source = get_dataset_meta(SimpleQAVerifiedDataset).source
    assert meta_source == (f"hf:google/simpleqa-verified@{SIMPLEQA_VERIFIED_REVISION}",)


def test_load_reads_csv_into_test_split():
    hf_dict = _hf_dict()
    dataset = SimpleQAVerifiedDataset(_hf_dict=hf_dict)
    with (
        patch.object(sqav_module, "load_dataset", return_value=hf_dict) as mock_load,
        patch("os.path.isdir", return_value=True),
    ):
        loaded = dataset.load("/staged/simpleqa_verified")

    assert mock_load.call_args.args[0] == "csv"
    data_files = mock_load.call_args.kwargs["data_files"]
    assert data_files["test"].endswith("/simpleqa_verified/simpleqa_verified.csv")
    cols = loaded["test"].column_names
    assert "problem" in cols and "answer" in cols


def test_empty_test_split_raises():
    empty = HFDatasetDict({"test": HFDataset.from_list([])})
    dataset = SimpleQAVerifiedDataset(_hf_dict=_hf_dict())
    with (
        patch.object(sqav_module, "load_dataset", return_value=empty),
        patch("os.path.isdir", return_value=False),
        pytest.raises(ValueError, match="empty 'test' split"),
    ):
        dataset.load("/staged/simpleqa_verified/simpleqa_verified.csv")
