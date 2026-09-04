"""Unit tests for the NL2SH-ALFA dataset wrapper.

The loader's job is a positional join between two upstreams that disagree, so
the tests are mostly about what happens when the join cannot be trusted: a count
mismatch has to raise rather than pair the shorter side, because nothing but row
order relates the Hub's rows to the harness's gold table.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from unittest.mock import patch

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

import sieval.datasets.nl2sh_alfa as nl2sh_module
from sieval.community.intercode_alfa import IMAGE_SPLITS, gold_table
from sieval.core.datasets.meta import get_dataset_meta
from sieval.datasets.nl2sh_alfa import (
    EXPECTED_TEST_ROWS,
    NL2SH_ALFA_REVISION,
    NL2SHAlfaDataset,
)


def _hub_rows(count: int = EXPECTED_TEST_ROWS) -> HFDataset:
    """A stand-in for the Hub's `test` config, in its four published columns."""
    return HFDataset.from_list(
        [
            {
                "nl": f"instruction {i}",
                "bash": f"cmd-{i}",
                "bash2": f"alt-{i}",
                "difficulty": i % 3,
            }
            for i in range(count)
        ]
    )


def _load(hub: HFDataset):
    dataset = NL2SHAlfaDataset(_hf_dict=HFDatasetDict({"test": hub}))
    with patch.object(nl2sh_module, "load_dataset", return_value=hub) as mock_load:
        loaded = dataset.load("westenfelder/NL2SH-ALFA")
    return loaded, mock_load


def test_source_pins_the_hub_revision():
    assert get_dataset_meta(NL2SHAlfaDataset).source == (
        f"hf:westenfelder/NL2SH-ALFA@{NL2SH_ALFA_REVISION}",
    )


def test_expected_rows_is_the_harness_split_table():
    # Not an independent constant: if the two ever disagree the join is wrong,
    # so the dataset derives its count from `index_to_img`'s own table.
    assert EXPECTED_TEST_ROWS == sum(IMAGE_SPLITS) == 300


def test_load_reads_the_test_config_under_its_train_split_name():
    _, mock_load = _load(_hub_rows())
    # Both Hub configs name their single split `train`; picking the config by
    # position and the split by keyword is what makes the 300 verified rows the
    # ones that get evaluated.
    assert mock_load.call_args.args == ("westenfelder/NL2SH-ALFA", "test")
    assert mock_load.call_args.kwargs["split"] == "train"


def test_load_exposes_both_upstreams_and_stamps_the_filesystem():
    loaded, _ = _load(_hub_rows())
    assert list(loaded.keys()) == ["test"]
    rows = loaded["test"]
    assert len(rows) == EXPECTED_TEST_ROWS
    assert set(rows.column_names) == {
        "nl",
        "bash",
        "bash2",
        "difficulty",
        "query",
        "gold",
        "gold2",
        "fs_id",
    }
    # `fs_id` counts must be the split table itself, in order.
    counts = [sum(1 for f in rows["fs_id"] if f == i) for i in range(1, 6)]
    assert counts == list(IMAGE_SPLITS)


def test_the_graded_gold_comes_from_the_harness_not_the_hub():
    loaded, _ = _load(_hub_rows())
    rows = loaded["test"]
    graded = gold_table()
    # The Hub column is carried verbatim and is NOT what grading reads. This is
    # the whole reason both are exposed: the two differ on two rows, and a
    # reader who assumed `bash` was scored would mis-attribute the difference.
    assert rows[0]["bash"] == "cmd-0"
    assert rows[0]["gold"] == graded[0]["gold"]
    assert rows[38]["gold"] == "echo 'hello' | base64"
    assert rows[100]["gold"] == "awk 'length < 40' setup_nl2b_fs_1.sh"


@pytest.mark.parametrize("count", [299, 301])
def test_a_row_count_mismatch_refuses_to_pair(count: int):
    # Truncating to the shorter side would pair instruction i with gold i for a
    # while and then silently stop -- every later sample graded against the
    # wrong command. Refuse instead.
    with pytest.raises(ValueError, match="positional join"):
        _load(_hub_rows(count))
