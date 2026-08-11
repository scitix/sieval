"""Unit tests for the ComplexConstraints dataset wrapper.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from unittest.mock import patch

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

import sieval.datasets.complex_constraints as cc_module
from sieval.core.datasets.meta import get_dataset_meta
from sieval.datasets.complex_constraints import (
    COMPLEX_CONSTRAINTS_REVISION,
    CSV_FILENAME,
    MAX_CRITERIA,
    ComplexConstraintsDataset,
)


def _row(criteria: list[str], benchmark_id: str = "CIF-001") -> dict:
    """One wide-format CSV row: 5 item columns + 40 sparse criterion columns."""
    padded = list(criteria) + [None] * (MAX_CRITERIA - len(criteria))
    return {
        "benchmark_id": benchmark_id,
        "prompt": "Write a rota.",
        "use_case": "Logistics, Scheduling & Event Planning",
        "instruction_type": "Negative",
        "prompt_style": "Context prompting",
        **{f"criterion_{i + 1}": value for i, value in enumerate(padded)},
    }


def _hf_dict(rows: list[dict] | None = None) -> HFDatasetDict:
    rows = rows if rows is not None else [_row(["alpha", "beta"])]
    return HFDatasetDict({"test": HFDataset.from_list(rows)})


def _load(hf_dict: HFDatasetDict, path: str = "/staged/complex_constraints"):
    dataset = ComplexConstraintsDataset(_hf_dict=_hf_dict())
    with (
        patch.object(cc_module, "load_dataset", return_value=hf_dict) as mock_load,
        patch("os.path.isdir", return_value=True),
    ):
        return dataset.load(path), mock_load


def test_source_pins_hf_revision():
    meta_source = get_dataset_meta(ComplexConstraintsDataset).source
    assert meta_source == (
        f"hf:surgeai/ComplexConstraints@{COMPLEX_CONSTRAINTS_REVISION}",
    )


def test_load_reads_the_repos_real_csv_filename():
    loaded, mock_load = _load(_hf_dict())
    assert mock_load.call_args.args[0] == "csv"
    data_files = mock_load.call_args.kwargs["data_files"]
    # The dataset card's configs entry spells this file with different casing
    # and does not exist; loading that name would fail outright.
    assert data_files["test"].endswith(f"/complex_constraints/{CSV_FILENAME}")
    assert CSV_FILENAME == "ComplexConstraints_benchmark_set.csv"
    assert len(loaded["test"]) == 1


def test_load_collapses_criterion_columns_into_a_list():
    loaded, _ = _load(_hf_dict())
    split = loaded["test"]
    assert split[0]["criteria"] == ["alpha", "beta"]
    # The 40 sparse source columns are gone -- leaving them would make the
    # sample type a 40-key mostly-absent dict.
    assert not [c for c in split.column_names if c.startswith("criterion_")]
    # The five item columns keep their upstream names.
    assert split[0]["benchmark_id"] == "CIF-001"
    assert split[0]["prompt_style"] == "Context prompting"


def test_load_keeps_every_non_empty_criterion_across_a_gap():
    # Upstream's filled cells are a contiguous prefix, but stopping at the first
    # empty cell would silently drop criteria if that ever changed -- and a
    # dropped criterion inflates the score. Every non-empty cell is kept.
    row = _row(["alpha", "beta"])
    row["criterion_2"] = None
    row["criterion_5"] = "epsilon"
    loaded, _ = _load(_hf_dict([row]))
    assert loaded["test"][0]["criteria"] == ["alpha", "epsilon"]


def test_load_drops_whitespace_only_cells():
    row = _row(["alpha", "   ", "gamma"])
    loaded, _ = _load(_hf_dict([row]))
    assert loaded["test"][0]["criteria"] == ["alpha", "gamma"]


def test_empty_test_split_raises():
    empty = HFDatasetDict({"test": HFDataset.from_list([])})
    dataset = ComplexConstraintsDataset(_hf_dict=_hf_dict())
    with (
        patch.object(cc_module, "load_dataset", return_value=empty),
        patch("os.path.isdir", return_value=False),
        pytest.raises(ValueError, match="empty 'test' split"),
    ):
        dataset.load(f"/staged/complex_constraints/{CSV_FILENAME}")


def test_missing_criterion_columns_raise():
    # A shape change upstream must fail loudly, not yield empty rubrics that
    # would score every response as a vacuous task pass.
    narrow = _row(["alpha"])
    del narrow[f"criterion_{MAX_CRITERIA}"]
    with pytest.raises(ValueError, match=r"missing criterion column\(s\)"):
        _load(_hf_dict([narrow]))


def test_extra_criterion_columns_raise():
    # The other direction of the same shape change, and the quieter one: a wider
    # rubric fails nowhere downstream, it just drops the criteria past
    # MAX_CRITERIA -- fewer criteria to satisfy, so the task pass rate goes UP.
    wide = _row(["alpha"])
    wide[f"criterion_{MAX_CRITERIA + 1}"] = "omega"
    with pytest.raises(ValueError, match=r"unexpected criterion column\(s\)"):
        _load(_hf_dict([wide]))
