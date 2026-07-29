"""Unit tests for the HLE dataset wrapper.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from unittest.mock import patch

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import Features, Image, Value

import sieval.datasets.hle as hle_module
from sieval.core.datasets.meta import get_dataset_meta
from sieval.datasets.hle import HLE_REVISION, HLEDataset


def _row(image: str = "") -> dict:
    # Mirrors the pinned revision's schema, including the auxiliary Image
    # columns — `load` asserts they exist before disabling their decoding.
    return {
        "id": "q1",
        "question": "What is 2 + 2?",
        "image": image,
        "image_preview": None,
        "answer": "4",
        "answer_type": "exactMatch",
        "author_name": "author",
        "rationale": "",
        "rationale_image": None,
        "raw_subject": "Math",
        "category": "Math",
    }


def _hf_dict(rows: int = 1) -> HFDatasetDict:
    return HFDatasetDict({"test": HFDataset.from_list([_row()] * rows)})


def test_source_pins_hf_revision():
    meta_source = get_dataset_meta(HLEDataset).source
    assert meta_source == (f"hf:cais/hle@{HLE_REVISION}",)


def test_load_forwards_path_and_preserves_image_column():
    hf_dict = _hf_dict()
    dataset = HLEDataset(_hf_dict=hf_dict)
    with patch.object(hle_module, "load_dataset", return_value=hf_dict) as mock_load:
        loaded = dataset.load("cais/hle")

    # Minimal loader: pass name_or_path straight through, keep the "test" split.
    assert mock_load.call_args.args[0] == "cais/hle"
    assert "test" in loaded
    # The multimodal `image` column must be preserved (not welded to text-only).
    assert "image" in loaded["test"].column_names


def test_load_disables_auxiliary_image_decoding():
    # image_preview / rationale_image are HF Image features (decode=True upstream);
    # load() must disable decoding so a row fetch never requires Pillow.
    features = Features(
        {
            "id": Value("string"),
            "question": Value("string"),
            "image": Value("string"),
            "image_preview": Image(decode=True),
            "answer": Value("string"),
            "answer_type": Value("string"),
            "author_name": Value("string"),
            "rationale": Value("string"),
            "rationale_image": Image(decode=True),
            "raw_subject": Value("string"),
            "category": Value("string"),
        }
    )
    row = {**_row(), "image_preview": None, "rationale_image": None}
    hf = HFDatasetDict(
        {
            "test": HFDataset.from_dict(
                {k: [row[k]] for k in features}, features=features
            )
        }
    )
    dataset = HLEDataset(_hf_dict=hf)
    with patch.object(hle_module, "load_dataset", return_value=hf):
        loaded = dataset.load("cais/hle")

    assert loaded["test"].features["image_preview"].decode is False
    assert loaded["test"].features["rationale_image"].decode is False


def test_missing_auxiliary_image_column_raises():
    # `cast_column` fails open — on an absent column it silently adds an empty
    # one instead of raising, which would drop the Pillow guard with no signal.
    # Schema drift must therefore be caught before the cast.
    drifted = {k: v for k, v in _row().items() if k != "rationale_image"}
    hf = HFDatasetDict({"test": HFDataset.from_list([drifted])})
    with (
        patch.object(hle_module, "load_dataset", return_value=hf),
        pytest.raises(ValueError, match="rationale_image"),
    ):
        HLEDataset("cais/hle")


# --- subset selection: sieval addition, applied at load time (before operations:) ---


def _build(rows: list[dict], *, text_only: bool = True) -> HLEDataset:
    hf = HFDatasetDict({"test": HFDataset.from_list(rows)})
    with patch.object(hle_module, "load_dataset", return_value=hf):
        return HLEDataset("cais/hle", text_only=text_only)


def test_text_only_default_drops_image_questions():
    dataset = _build([_row(), _row(image="data:image/png;base64,AAAA")])
    assert dataset.test_set is not None
    assert len(dataset.test_set) == 1
    assert dataset.test_set[0]["image"] == ""
    assert dataset.text_only is True


def test_full_set_keeps_image_questions():
    dataset = _build(
        [_row(), _row(image="data:image/png;base64,AAAA")], text_only=False
    )
    assert dataset.test_set is not None
    assert len(dataset.test_set) == 2
    assert dataset.text_only is False


def test_text_only_all_multimodal_raises():
    with pytest.raises(ValueError, match="empty 'test' split"):
        _build([_row(image="data:image/png;base64,AAAA")])


def test_text_only_never_leaks_into_load_dataset():
    # `text_only` is captured by the signature, so it must not reach the Hub call.
    hf = HFDatasetDict({"test": HFDataset.from_list([_row()])})
    with patch.object(hle_module, "load_dataset", return_value=hf) as mock_load:
        HLEDataset("cais/hle", text_only=True, revision=HLE_REVISION)
    assert "text_only" not in mock_load.call_args.kwargs
    assert mock_load.call_args.kwargs["revision"] == HLE_REVISION
