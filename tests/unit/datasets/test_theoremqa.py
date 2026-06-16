"""
Unit tests for the TheoremQA dataset wrapper.

AI-Generated Code - GPT-5 (OpenAI)
"""

from unittest.mock import patch

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

import sieval.datasets.theoremqa as theoremqa_module
from sieval.datasets.theoremqa import TheoremQADataset


def test_load_does_not_forward_revision_pin_to_local_dataset_loader():
    sample = {
        "Question": "What is 2+2?",
        "Answer": "4",
        "Answer_type": "integer",
        "Picture": "unused",
    }
    hf_dict = HFDatasetDict({"test": HFDataset.from_list([sample])})
    dataset = TheoremQADataset(_hf_dict=hf_dict)

    with patch.object(
        theoremqa_module, "load_dataset", return_value=hf_dict
    ) as mock_load:
        loaded = dataset.load("/tmp/TIGER-Lab/TheoremQA")

    assert mock_load.call_args.args == ("/tmp/TIGER-Lab/TheoremQA",)
    assert "revision" not in mock_load.call_args.kwargs
    assert "Picture" not in loaded["test"].column_names
