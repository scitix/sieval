"""
Unit tests for the BBH dataset loader.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from unittest.mock import patch

import pytest
from datasets import Dataset as HFDataset

import sieval.datasets.bbh as bbh_module
from sieval.core.datasets.meta import get_dataset_meta
from sieval.datasets.bbh import BBH_REVISION, BBH_SUBSETS, BBHDataset


def _fake_load(_path, subset, split, **_kwargs):
    assert split == "test"
    return HFDataset.from_list(
        [
            {"input": f"{subset} q0", "target": "A"},
            {"input": f"{subset} q1", "target": "B"},
        ]
    )


def test_source_revision_is_pinned():
    assert BBH_REVISION == "982bb89fd79532a8ac676a61fc42eb1aeec63f99"
    assert get_dataset_meta(BBHDataset).source == (f"hf:lukaemon/bbh@{BBH_REVISION}",)


def test_subsets_are_the_27_upstream_configs():
    assert len(BBH_SUBSETS) == 27
    assert len(set(BBH_SUBSETS)) == 27


def test_load_merges_every_subset_in_fixed_order_and_stamps_subset():
    with patch.object(bbh_module, "load_dataset", side_effect=_fake_load) as mock:
        loaded = BBHDataset("/staged/lukaemon/bbh").dataset_dict

    assert [c.args[1] for c in mock.call_args_list] == list(BBH_SUBSETS)
    assert list(loaded) == ["test"]
    test = loaded["test"]
    assert test.num_rows == 2 * len(BBH_SUBSETS)
    # Upstream column names kept; `subset` added.
    assert test.column_names == ["input", "target", "subset"]
    assert test["subset"][:3] == [BBH_SUBSETS[0], BBH_SUBSETS[0], BBH_SUBSETS[1]]
    assert test[2]["input"] == f"{BBH_SUBSETS[1]} q0"
    assert test["subset"][-1] == BBH_SUBSETS[-1]


def test_empty_subset_is_refused():
    def load(path, subset, split, **_kwargs):
        rows = _fake_load(path, subset, split)
        return rows.select([]) if subset == "navigate" else rows

    with (
        patch.object(bbh_module, "load_dataset", side_effect=load),
        pytest.raises(ValueError, match="'navigate' is empty"),
    ):
        BBHDataset("/staged/lukaemon/bbh")
