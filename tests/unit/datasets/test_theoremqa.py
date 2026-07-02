"""
Unit tests for the TheoremQA dataset wrapper.

AI-Generated Code - GPT-5 (OpenAI)
"""

from unittest.mock import patch

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import Features, Value
from datasets import Image as HFImage

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
    assert "Picture" in loaded["test"].column_names


def test_load_disables_picture_decode_to_avoid_pillow_dependency():
    # Upstream Picture is an HF Image feature (decode=True by default); decoding
    # it on materialization would require Pillow, which is outside sieval[math].
    # load() must mirror the column but disable decode so a clean install does
    # not crash. Empty rows keep the fixture Pillow-free (no bytes to encode).
    features = Features(
        {
            "Question": Value("string"),
            "Answer": Value("string"),
            "Answer_type": Value("string"),
            "Picture": HFImage(),
        }
    )
    empty = HFDataset.from_dict(
        {"Question": [], "Answer": [], "Answer_type": [], "Picture": []},
        features=features,
    )
    hf_dict = HFDatasetDict({"test": empty})
    dataset = TheoremQADataset(_hf_dict=hf_dict)

    with patch.object(theoremqa_module, "load_dataset", return_value=hf_dict):
        loaded = dataset.load("/tmp/TIGER-Lab/TheoremQA")

    picture = loaded["test"].features["Picture"]
    assert isinstance(picture, HFImage)
    assert picture.decode is False
    assert "Picture" in loaded["test"].column_names
