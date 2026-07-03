"""Unit tests for the IFBench dataset loader.

AI-Generated Code - GPT-5 (OpenAI)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.datasets.ifbench import IFBenchDataset

_SAMPLE = {
    "key": "ifbench-1",
    "prompt": "Write a short answer.",
    "instruction_id_list": ["format:no_whitespace"],
    "kwargs": [{}],
}


def test_load_mirrors_single_train_split_to_test(monkeypatch: pytest.MonkeyPatch):
    # The pinned source ships only a "train" split; load() must expose it as "test".
    def fake_load_dataset(_name_or_path: str, **_kwargs):
        return HFDatasetDict({"train": HFDataset.from_list([_SAMPLE])})

    monkeypatch.setattr("sieval.datasets.ifbench.load_dataset", fake_load_dataset)

    dataset = IFBenchDataset("allenai/IFBench_test")

    assert set(dataset.dataset_dict) == {"train", "test"}
    assert dataset.test_set is not None
    assert len(dataset.test_set) == 1
    assert dataset.test_set[0]["key"] == "ifbench-1"


def test_load_forwards_kwargs_without_forcing_split(monkeypatch: pytest.MonkeyPatch):
    # Loads the whole DatasetDict (no split slicing) so the pinned staged parquet
    # is the only artifact read; apply_eval_split handles the split mapping.
    captured_kwargs: dict[str, object] = {}
    captured_name: list[str] = []

    def fake_load_dataset(name_or_path: str, **kwargs):
        captured_name.append(name_or_path)
        captured_kwargs.update(kwargs)
        return HFDatasetDict({"train": HFDataset.from_list([_SAMPLE])})

    monkeypatch.setattr("sieval.datasets.ifbench.load_dataset", fake_load_dataset)

    IFBenchDataset("allenai/IFBench_test", trust_remote_code=False)

    assert "split" not in captured_kwargs
    assert captured_kwargs == {"trust_remote_code": False}
    assert captured_name[0].endswith("allenai/IFBench_test")
