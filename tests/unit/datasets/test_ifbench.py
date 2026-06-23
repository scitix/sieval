"""Unit tests for the IFBench dataset loader.

AI-Generated Code - GPT-5 (OpenAI)
"""

import json

from sieval.datasets.ifbench import IFBenchDataset


def test_load_local_jsonl_file_as_train_and_test(tmp_path):
    jsonl_path = tmp_path / "IFBench_test.jsonl"
    sample = {
        "key": "ifbench-1",
        "prompt": "Write a short answer.",
        "instruction_id_list": ["format:no_whitespace"],
        "kwargs": [{}],
    }
    jsonl_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    dataset = IFBenchDataset(str(jsonl_path))

    assert set(dataset.dataset_dict) == {"train", "test"}
    train_set = dataset.train_set
    test_set = dataset.test_set
    assert train_set is not None
    assert test_set is not None
    assert len(train_set) == 1
    assert len(test_set) == 1
    assert test_set[0]["prompt"] == sample["prompt"]


def test_load_directory_prefers_ifbench_test_jsonl(tmp_path):
    jsonl_path = tmp_path / "IFBench_test.jsonl"
    sample = {
        "key": "ifbench-2",
        "prompt": "Answer with one word.",
        "instruction_id_list": ["format:no_whitespace"],
        "kwargs": [{}],
    }
    jsonl_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    dataset = IFBenchDataset(str(tmp_path))

    test_set = dataset.test_set
    assert test_set is not None
    assert len(test_set) == 1
    assert test_set[0]["key"] == sample["key"]
