import json

import pytest

from sieval.datasets.mbpp import MBPPDataset, _resolve_data_file


def test_mbpp_loads_jsonl_and_builds_official_splits(tmp_path):
    rows = [
        {
            "task_id": 1,
            "text": "prompt split",
            "code": "def a(): pass",
            "test_list": ["assert True"],
        },
        {
            "task_id": 11,
            "text": "test split",
            "code": "def b(): pass",
            "test_list": ["assert True"],
            "test_setup_code": "import math",
            "challenge_test_list": ["assert True"],
        },
        {
            "task_id": 511,
            "prompt": "validation split",
            "code": "def c(): pass",
            "test_list": ["assert True"],
        },
        {
            "task_id": 601,
            "text": "train split",
            "code": "def d(): pass",
            "test_list": ["assert True"],
        },
    ]
    data_file = tmp_path / "mbpp.jsonl"
    data_file.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    dataset = MBPPDataset(str(tmp_path))

    assert {split: len(ds) for split, ds in dataset.dataset_dict.items()} == {
        "prompt": 1,
        "test": 1,
        "validation": 1,
        "train": 1,
    }
    assert dataset.dataset_dict["validation"][0]["text"] == "validation split"
    assert dataset.dataset_dict["test"][0]["test_setup_code"] == "import math"
    assert dataset.dataset_dict["prompt"][0]["challenge_test_list"] == []


def test_mbpp_empty_split_raises(tmp_path):
    # All rows land in the train range, leaving prompt/test/validation empty.
    rows = [
        {
            "task_id": 601,
            "text": "t",
            "code": "def d(): pass",
            "test_list": ["assert True"],
        },
    ]
    data_file = tmp_path / "mbpp.jsonl"
    data_file.write_text(json.dumps(rows[0]), encoding="utf-8")

    with pytest.raises(ValueError, match="empty split"):
        MBPPDataset(str(tmp_path))


def test_resolve_data_file_passes_through_url():
    url = "https://example.com/mbpp.jsonl"
    assert _resolve_data_file(url) == url


def test_resolve_data_file_appends_filename_for_dir(tmp_path):
    (tmp_path / "mbpp.jsonl").write_text("{}", encoding="utf-8")
    assert _resolve_data_file(str(tmp_path)) == str(tmp_path / "mbpp.jsonl")


def test_resolve_data_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="sieval dataset download mbpp"):
        _resolve_data_file(str(tmp_path / "nope.jsonl"))
