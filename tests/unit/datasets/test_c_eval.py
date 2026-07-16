"""Unit tests for the C-Eval dataset loader.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.datasets.c_eval import CEvalDataset


def _fake_load_dataset(_path, _subject=None, **_kwargs) -> HFDatasetDict:
    """Mimic ``load_dataset(repo, <config>)``; raw rows carry no subject."""
    base = {"id": 0, "question": "q", "A": "a", "B": "b", "C": "c", "D": "d"}
    return HFDatasetDict(
        {
            "dev": HFDataset.from_list([{**base, "answer": "A", "explanation": "e"}]),
            "val": HFDataset.from_list([{**base, "answer": "B"}]),
            "test": HFDataset.from_list([{**base, "answer": "C"}]),
        }
    )


def _loader() -> CEvalDataset:
    """A constructed instance whose ``load()`` we then call directly."""
    row = {
        "question": "x",
        "A": "a",
        "B": "b",
        "C": "c",
        "D": "d",
        "answer": "A",
        "subject": "x",
    }
    placeholder = HFDataset.from_list([row])
    return CEvalDataset(_hf_dict=HFDatasetDict({"test": placeholder}))


def test_load_injects_subject_and_concatenates(monkeypatch):
    monkeypatch.setattr("sieval.datasets.c_eval.load_dataset", _fake_load_dataset)
    out = _loader().load("ignored/path", subjects=["computer_network", "law"])

    assert set(out.keys()) == {"dev", "val", "test"}
    # One row per subject, concatenated; subject injected from the config name.
    assert len(out["test"]) == 2
    assert set(out["test"]["subject"]) == {"computer_network", "law"}
    # Default eval target is the released `test` split (answer "C" in the mock).
    assert set(out["test"]["answer"]) == {"C"}


def test_load_eval_split_val_remaps_to_test(monkeypatch):
    monkeypatch.setattr("sieval.datasets.c_eval.load_dataset", _fake_load_dataset)
    out = _loader().load("ignored/path", subjects=["law"], eval_split="val")

    # `test` now serves val rows (answer "B"), while dev is untouched for few-shot.
    assert out["test"]["answer"] == ["B"]
    assert "dev" in out


def test_load_empty_eval_split_raises(monkeypatch):
    monkeypatch.setattr(
        "sieval.datasets.c_eval.load_dataset",
        lambda *_a, **_k: HFDatasetDict({}),
    )
    with pytest.raises(ValueError, match="no eval"):
        _loader().load("ignored/path", subjects=["law"])
