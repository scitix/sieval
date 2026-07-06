"""Unit tests for the HellaSwag dataset loader.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.datasets import hellaswag as hs
from sieval.datasets.hellaswag import HellaSwagDataset

_COLUMNS = {
    "ind",
    "activity_label",
    "ctx_a",
    "ctx_b",
    "ctx",
    "endings",
    "source_id",
    "split",
    "split_type",
    "label",
}


def _row(label: str) -> dict:
    return {
        "ind": 0,
        "activity_label": "Removing ice from car",
        "ctx_a": "A woman is outside.",
        "ctx_b": "she",
        "ctx": "A woman is outside. she",
        "endings": ["a", "b", "c", "d"],
        "source_id": "activitynet~v_x",
        "split": "val",
        "split_type": "indomain",
        "label": label,
    }


def _fake_dict() -> HFDatasetDict:
    return HFDatasetDict(
        {
            "train": HFDataset.from_list([_row("0")]),
            "validation": HFDataset.from_list([_row("2")]),
            "test": HFDataset.from_list([_row("")]),  # HellaSwag withholds test labels
        }
    )


def test_load_defaults_eval_split_to_validation(monkeypatch):
    monkeypatch.setattr(hs, "load_dataset", lambda *a, **k: _fake_dict())
    ds = HellaSwagDataset(name_or_path="Rowan/hellaswag")
    test_split = ds.dataset_dict["test"]
    # the eval split ("test") now carries the LABELED validation rows
    assert test_split[0]["label"] == "2"
    # mirror: native schema preserved exactly (no columns added/removed)
    assert set(test_split.column_names) == _COLUMNS


def test_load_respects_explicit_eval_split_test(monkeypatch):
    monkeypatch.setattr(hs, "load_dataset", lambda *a, **k: _fake_dict())
    ds = HellaSwagDataset(name_or_path="Rowan/hellaswag", eval_split="test")
    # native (label-less) test split kept unchanged — no remap when eval_split == "test"
    assert ds.dataset_dict["test"][0]["label"] == ""


def test_load_mirrors_source_without_preprocessing(monkeypatch):
    monkeypatch.setattr(hs, "load_dataset", lambda *a, **k: _fake_dict())
    ds = HellaSwagDataset(name_or_path="Rowan/hellaswag")
    # ctx_b stays raw ("she"), NOT capitalized — query construction is task-side
    assert ds.dataset_dict["test"][0]["ctx_b"] == "she"
    assert ds.dataset_dict["train"][0]["label"] == "0"
