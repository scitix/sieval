"""
Unit tests for the MMLU dataset loader.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from unittest.mock import patch

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

import sieval.datasets.mmlu as mmlu_module
from sieval.datasets.mmlu import MMLU_REVISION, MMLUDataset


def _hf_dict() -> HFDatasetDict:
    row = {
        "question": "What?",
        "subject": "anatomy",
        "choices": ["a", "b", "c", "d"],
        "answer": 1,
    }
    return HFDatasetDict({"test": HFDataset.from_list([row])})


def test_source_revision_is_pinned():
    # Pinning guard: the cais/mmlu commit the source/downloader resolve against.
    assert MMLU_REVISION == "c30699e8356da336a370243923dbaf21066bb9fe"


def test_load_forwards_path_and_config_without_mutating_schema():
    hf_dict = _hf_dict()
    dataset = MMLUDataset(_hf_dict=hf_dict)

    with patch.object(mmlu_module, "load_dataset", return_value=hf_dict) as mock_load:
        loaded = dataset.load("cais/mmlu")

    # Path + default config forwarded positionally; the revision pin is applied
    # at download time, not re-forwarded to the loader (the staged path is
    # already at the pinned commit).
    assert mock_load.call_args.args == ("cais/mmlu", "all")
    assert "revision" not in mock_load.call_args.kwargs
    # Native cais/mmlu schema preserved as-is — no rename/coercion in the loader.
    assert set(loaded["test"].column_names) == {
        "question",
        "subject",
        "choices",
        "answer",
    }


def test_load_remaps_eval_split_to_test():
    hf_dict = HFDatasetDict(
        {
            "validation": HFDataset.from_list([dict(_hf_dict()["test"][0])]),
        }
    )
    dataset = MMLUDataset(_hf_dict=hf_dict)

    with patch.object(mmlu_module, "load_dataset", return_value=hf_dict):
        loaded = dataset.load("cais/mmlu", eval_split="validation")

    assert "test" in loaded and loaded["test"].num_rows == 1
