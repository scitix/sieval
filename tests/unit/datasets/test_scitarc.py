"""Unit tests for the SciTaRC dataset wrapper.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from unittest.mock import patch

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

import sieval.datasets.scitarc as scitarc_module
from sieval.core.datasets.meta import get_dataset_meta
from sieval.datasets.scitarc import SCITARC_REVISION, SciTaRCDataset


def _row(plan: str | None = "SELECT all models\nRETURN best") -> dict:
    return {
        "paper": "2401.06769",
        "relevant_tables": [["\\begin{table}\n", "row\n", "\\end{table}\n"]],
        "tables": [["\\begin{table}\n", "other\n", "\\end{table}\n"]],
        "fulltext": "\\documentclass{article}",
        "question": "Which model is best?",
        "answer": "NLLB-200-1.3B. 64.71",
        "plan": plan,
    }


def _hf_dict(rows: list[dict] | None = None) -> HFDatasetDict:
    return HFDatasetDict({"test": HFDataset.from_list(rows or [_row()])})


def test_source_pins_hf_revision():
    assert get_dataset_meta(SciTaRCDataset).source == (
        f"hf:jhu-clsp/SciTaRC@{SCITARC_REVISION}",
    )


def test_license_is_the_data_license_not_the_harness_one():
    """CC-BY-NC-4.0 is the dataset's; upstream's harness code is MIT.

    The repo carries both badges, and the code license is the one that governs
    ``sieval.community.scitarc`` — not this.
    """
    assert get_dataset_meta(SciTaRCDataset).license == "CC-BY-NC-4.0"


def test_load_mirrors_the_test_split_as_is():
    hf_dict = _hf_dict()
    dataset = SciTaRCDataset(_hf_dict=hf_dict)
    with patch.object(
        scitarc_module, "load_dataset", return_value=hf_dict
    ) as mock_load:
        loaded = dataset.load("jhu-clsp/SciTaRC", revision=SCITARC_REVISION)

    assert mock_load.call_args.args[0] == "jhu-clsp/SciTaRC"
    assert mock_load.call_args.kwargs["revision"] == SCITARC_REVISION
    assert set(loaded) == {"test"}
    assert loaded["test"].column_names == [
        "paper",
        "relevant_tables",
        "tables",
        "fulltext",
        "question",
        "answer",
        "plan",
    ]


def test_nested_table_lists_survive_the_round_trip():
    """``relevant_tables`` is ``list[list[str]]`` and must stay that shape.

    Flattening it one level would make ``get_table_text`` join characters
    instead of lines, which produces a prompt that still renders and is wrong.
    """
    hf_dict = _hf_dict()
    dataset = SciTaRCDataset(_hf_dict=hf_dict)
    with patch.object(scitarc_module, "load_dataset", return_value=hf_dict):
        loaded = dataset.load("jhu-clsp/SciTaRC")

    tables = loaded["test"][0]["relevant_tables"]
    assert tables == [["\\begin{table}\n", "row\n", "\\end{table}\n"]]


def test_null_plan_survives():
    """Exactly one row of the pinned revision has no plan; it must load."""
    hf_dict = _hf_dict([_row(plan=None)])
    dataset = SciTaRCDataset(_hf_dict=hf_dict)
    with patch.object(scitarc_module, "load_dataset", return_value=hf_dict):
        loaded = dataset.load("jhu-clsp/SciTaRC")

    assert loaded["test"][0]["plan"] is None


def test_empty_test_split_raises():
    empty = HFDatasetDict({"test": HFDataset.from_list([])})
    dataset = SciTaRCDataset(_hf_dict=_hf_dict())
    with (
        patch.object(scitarc_module, "load_dataset", return_value=empty),
        pytest.raises(ValueError, match="empty 'test' split"),
    ):
        dataset.load("jhu-clsp/SciTaRC")
