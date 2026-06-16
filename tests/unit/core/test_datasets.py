"""
Unit tests for sieval/core/datasets.py.

Covers: Dataset.repeat, select, shuffle, retrieve_samples (random/fixed/lazy),
_clone_with_new_dict, property accessors.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import pytest

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from sieval.core.datasets import Dataset


# ===================================================================
# Minimal concrete implementation
# ===================================================================
class _ListDataset(Dataset):
    """Dataset backed by a plain list of dicts."""

    def __init__(self, samples, train_samples=None):
        self._raw_samples = samples
        self._raw_train = train_samples
        super().__init__("dummy")

    def load(self, name_or_path, **kwargs) -> HFDatasetDict:
        d = {"test": HFDataset.from_list(self._raw_samples)}
        if self._raw_train:
            d["train"] = HFDataset.from_list(self._raw_train)
        return HFDatasetDict(d)


class _BypassLoadDataset(Dataset):
    def load(self, name_or_path, **kwargs) -> HFDatasetDict:
        raise AssertionError("load() should not be called when _hf_dict is provided")


def _make(n=5, with_train=False):
    samples = [{"id": i, "val": f"v{i}"} for i in range(n)]
    train = [{"id": i, "val": f"t{i}"} for i in range(3)] if with_train else None
    return _ListDataset(samples, train)


# ===================================================================
# Properties
# ===================================================================
class TestDatasetInit:
    def test_init_with_hf_dict_bypasses_load(self):
        dataset_dict = HFDatasetDict({"test": HFDataset.from_list([{"id": 1}])})
        ds = _BypassLoadDataset(_hf_dict=dataset_dict)
        assert ds.dataset_dict is dataset_dict

    def test_init_without_name_or_hf_dict_raises(self):
        with pytest.raises(ValueError, match="Either name_or_path or _hf_dict"):
            _BypassLoadDataset()


class TestDatasetProperties:
    def test_test_set_and_dataset_dict(self):
        ds = _make(3)
        assert ds.test_set is not None
        assert len(ds.test_set) == 3
        assert isinstance(ds.dataset_dict, HFDatasetDict)
        # Verify actual content, not just existence
        ids = [ds.test_set[i]["id"] for i in range(3)]
        assert ids == [0, 1, 2]
        vals = [ds.test_set[i]["val"] for i in range(3)]
        assert vals == ["v0", "v1", "v2"]

    def test_train_set_presence(self):
        ds = _make()
        assert ds.train_set is None

        ds = _make(with_train=True)
        assert ds.train_set is not None
        assert len(ds.train_set) == 3
        # Verify train content
        ids = [ds.train_set[i]["id"] for i in range(3)]
        assert ids == [0, 1, 2]


# ===================================================================
# select
# ===================================================================
class TestSelect:
    def test_select_size_behavior_and_type(self):
        ds = _make(10)
        result = ds.select(4)
        assert len(result.test_set) == 4
        assert type(result) is type(ds)

        ds = _make(3)
        result = ds.select(100)
        assert len(result.test_set) == 3

    def test_select_no_test_set_returns_self(self):
        class _NoTestDataset(_ListDataset):
            def load(self, name_or_path, **kwargs):
                return HFDatasetDict({"train": HFDataset.from_list([{"id": 0}])})

        ds = _NoTestDataset([], None)
        result = ds.select(3)
        assert result is ds


# ===================================================================
# repeat
# ===================================================================
class TestRepeat:
    def test_repeat_multiplies_size_and_preserves_type(self):
        ds = _make(3)
        result = ds.repeat(2)
        assert len(result.test_set) == 6
        assert type(result) is type(ds)

    def test_repeat_no_test_set_returns_self(self):
        class _NoTestDataset(_ListDataset):
            def load(self, name_or_path, **kwargs):
                return HFDatasetDict({})

        ds = _NoTestDataset([], None)
        assert ds.repeat(3) is ds


# ===================================================================
# shuffle
# ===================================================================
class TestShuffle:
    def test_shuffle_preserves_size_and_type(self):
        ds = _make(5)
        result = ds.shuffle(seed=42)
        assert len(result.test_set) == 5
        assert type(result) is type(ds)

    def test_shuffle_different_seeds_produce_different_orderings(self):
        ds = _make(10)
        ids_seed0 = [r["id"] for r in ds.shuffle(seed=0).test_set]
        ids_seed99 = [r["id"] for r in ds.shuffle(seed=99).test_set]
        # Verify both are deterministic (same seed → same result)
        assert ids_seed0 == [r["id"] for r in ds.shuffle(seed=0).test_set]
        assert ids_seed99 == [r["id"] for r in ds.shuffle(seed=99).test_set]
        # Different seeds should produce different orderings
        assert ids_seed0 != ids_seed99

    def test_shuffle_no_test_set_returns_self(self):
        class _NoTestDataset(_ListDataset):
            def load(self, name_or_path, **kwargs):
                return HFDatasetDict({"train": HFDataset.from_list([{"id": 0}])})

        ds = _NoTestDataset([], None)
        assert ds.shuffle(seed=123) is ds


# ===================================================================
# retrieve_samples
# ===================================================================
class TestRetrieveSamples:
    def test_random_mode_list_and_clipping(self):
        ds = _make(with_train=True)
        samples = ds.retrieve_samples(2, split="train", mode="random")
        assert isinstance(samples, list)
        assert len(samples) == 2

        samples = ds.retrieve_samples(100, split="train", mode="random")
        assert len(samples) == 3  # train has 3 items

    def test_fixed_mode_variants(self):
        ds = _make(with_train=True)
        samples = ds.retrieve_samples(2, split="train", mode="fixed")
        assert len(samples) == 2

        samples = ds.retrieve_samples(2, split="train", mode="fixed", indices=[0, 2])
        assert len(samples) == 2

        samples = ds.retrieve_samples(
            5, split="train", mode="fixed", indices=[0, 1, 999]
        )
        # 999 is out-of-range, only 0 and 1 survive
        assert len(samples) == 2

        # Upper bound is exclusive: index == len(ds) must be filtered out.
        samples = ds.retrieve_samples(5, split="train", mode="fixed", indices=[0, 3])
        assert len(samples) == 1

    def test_lazy_modes_return_iterators(self):
        ds = _make(with_train=True)
        random_result = ds.retrieve_samples(2, split="train", mode="random", lazy=True)
        fixed_result = ds.retrieve_samples(2, split="train", mode="fixed", lazy=True)
        from collections.abc import Iterator

        assert isinstance(random_result, Iterator)
        assert isinstance(fixed_result, Iterator)
        items = list(random_result)
        assert len(items) == 2

    def test_missing_split_returns_empty(self):
        ds = _make()
        eager_result = ds.retrieve_samples(3, split="train", mode="random")
        lazy_result = ds.retrieve_samples(3, split="train", mode="random", lazy=True)
        assert eager_result == []
        assert list(lazy_result) == []

    def test_unknown_mode_raises(self):
        ds = _make(with_train=True)
        with pytest.raises(ValueError, match="Unknown mode"):
            ds.retrieve_samples(2, split="train", mode="unknown")

    def test_random_seed_reproducible(self):
        ds = _make(5, with_train=True)
        # Use test split for both
        s1 = ds.retrieve_samples(3, split="test", mode="random", seed=7)
        s2 = ds.retrieve_samples(3, split="test", mode="random", seed=7)
        assert [r["id"] for r in s1] == [r["id"] for r in s2]
