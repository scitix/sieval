from unittest.mock import patch

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.datasets.mbpp import MBPPDataset

# Upstream split sizes per config of google-research-datasets/mbpp. The `full`
# config is the one lm-eval (and the Qwen2.5-72B-Base pass@1=76.6 run) scores
# on; `sanitized` is a smaller, differently-sized subset. Unit tests don't hit
# the network, so the loader is exercised against a stub that reproduces these
# shapes — keyed by config so the count assertion below also proves `load`
# requested `full` (test=500) rather than `sanitized` (test=257).
_CONFIG_SPLIT_SIZES = {
    "full": {"prompt": 10, "test": 500, "validation": 90, "train": 374},
    "sanitized": {"prompt": 7, "test": 257, "validation": 43, "train": 120},
}


def _stub_split(n: int) -> HFDataset:
    return HFDataset.from_list(
        [
            {
                "task_id": i,
                "text": "t",
                "code": "def f(): pass",
                "test_list": ["assert True"],
                "test_setup_code": "",
                "challenge_test_list": [],
            }
            for i in range(n)
        ]
    )


def _fake_load_dataset(name_or_path, config=None, **kwargs):
    _ = (name_or_path, kwargs)  # stub branches only on config
    sizes = _CONFIG_SPLIT_SIZES.get(config)
    if sizes is None:
        raise AssertionError(f"unexpected MBPP config requested: {config!r}")
    return HFDatasetDict({split: _stub_split(n) for split, n in sizes.items()})


def test_load_uses_full_config_and_preserves_official_splits():
    with patch("sieval.datasets.mbpp.load_dataset", _fake_load_dataset):
        dataset = MBPPDataset("google-research-datasets/mbpp")

    # `full` config → the four official splits at their published counts. A
    # regression to `sanitized` (or `None`) would surface here as wrong counts
    # (or the AssertionError in the stub), i.e. a different evaluated test set.
    assert {split: len(ds) for split, ds in dataset.dataset_dict.items()} == {
        "prompt": 10,
        "test": 500,
        "validation": 90,
        "train": 374,
    }


def test_load_passes_explicit_config_override():
    with patch("sieval.datasets.mbpp.load_dataset", _fake_load_dataset):
        dataset = MBPPDataset("google-research-datasets/mbpp", config="sanitized")

    assert len(dataset.dataset_dict["test"]) == 257


def test_load_rejects_non_dataset_dict():
    # ensure_dataset_dict must reject a bare Dataset (e.g. a single-split load).
    with (
        patch(
            "sieval.datasets.mbpp.load_dataset",
            lambda *a, **k: _stub_split(1),
        ),
        pytest.raises(TypeError, match="Expected DatasetDict"),
    ):
        MBPPDataset("google-research-datasets/mbpp")
