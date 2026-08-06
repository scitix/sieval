"""Unit tests for the GSM-Plus dataset loader.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.datasets import gsm_plus as gp
from sieval.datasets.gsm_plus import GSMPlusDataset

_COLUMNS = {
    "question",
    "solution",
    "answer",
    "perturbation_type",
    "seed_question",
    "seed_solution",
    "seed_answer",
}


def _row(perturbation_type: str, answer: str = "27") -> dict:
    return {
        "question": "Janet's ducks lay 20 eggs per day. ...",
        "solution": f"Work.\n#### {answer}",
        "answer": answer,
        "perturbation_type": perturbation_type,
        "seed_question": "Janet's ducks lay 16 eggs per day. ...",
        "seed_solution": "Work.\n#### 18",
        "seed_answer": "18",
    }


def _fake_dict() -> HFDatasetDict:
    return HFDatasetDict(
        {
            "test": HFDataset.from_list(
                [_row("numerical substitution"), _row("critical thinking", "None")]
            ),
            "testmini": HFDataset.from_list([_row("digit expansion", "42")]),
        }
    )


def test_load_defaults_to_full_test_split(monkeypatch):
    monkeypatch.setattr(gp, "load_dataset", lambda *a, **k: _fake_dict())
    ds = GSMPlusDataset(name_or_path="qintongli/GSM-Plus")
    assert ds.test_set is not None
    assert len(ds.test_set) == 2
    # mirror: native schema preserved exactly (no columns added/removed)
    assert set(ds.test_set.column_names) == _COLUMNS


def test_load_remaps_testmini_to_test(monkeypatch):
    monkeypatch.setattr(gp, "load_dataset", lambda *a, **k: _fake_dict())
    ds = GSMPlusDataset(name_or_path="qintongli/GSM-Plus", eval_split="testmini")
    assert ds.test_set is not None
    assert len(ds.test_set) == 1
    assert ds.test_set[0]["perturbation_type"] == "digit expansion"


def test_load_rejects_unknown_split(monkeypatch):
    # A silent no-op here would fall through to the 10552-row test split, so a
    # misspelled "testmini" must fail loudly rather than run 4x the samples.
    monkeypatch.setattr(gp, "load_dataset", lambda *a, **k: _fake_dict())
    with pytest.raises(ValueError, match="no split 'testmni'"):
        GSMPlusDataset(name_or_path="qintongli/GSM-Plus", eval_split="testmni")


def test_load_rejects_empty_eval_split(monkeypatch):
    empty = HFDatasetDict({"test": HFDataset.from_list([])})
    monkeypatch.setattr(gp, "load_dataset", lambda *a, **k: empty)
    with pytest.raises(ValueError, match="loaded 0 samples"):
        GSMPlusDataset(name_or_path="qintongli/GSM-Plus")


def test_load_preserves_seed_columns_and_none_answer(monkeypatch):
    monkeypatch.setattr(gp, "load_dataset", lambda *a, **k: _fake_dict())
    ds = GSMPlusDataset(name_or_path="qintongli/GSM-Plus")
    assert ds.test_set is not None
    # seed_* is what makes paired GSM8K-vs-GSM-Plus comparison possible
    assert ds.test_set[0]["seed_answer"] == "18"
    # "None" is a string answer ("unanswerable"), not a null
    assert ds.test_set[1]["answer"] == "None"
