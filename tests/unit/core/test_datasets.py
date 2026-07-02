"""
Unit tests for sieval/core/datasets.py.

Covers: Dataset.repeat, slice, shuffle, retrieve_samples (random/fixed/lazy),
_clone_with_new_dict, property accessors.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import io

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from loguru import logger

from sieval.core.datasets import Dataset


def _capture_logs(fn) -> str:
    sink = io.StringIO()
    logger_id = logger.add(sink, format="{message}")
    try:
        fn()
    finally:
        logger.remove(logger_id)
    return sink.getvalue()


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
# slice
# ===================================================================
class TestSlice:
    def test_slice_size_behavior_and_type(self):
        ds = _make(10)
        result = ds.slice(4)
        assert len(result.test_set) == 4
        assert type(result) is type(ds)

        ds = _make(3)
        result = ds.slice(100)
        assert len(result.test_set) == 3

    def test_slice_no_test_set_returns_self(self):
        class _NoTestDataset(_ListDataset):
            def load(self, name_or_path, **kwargs):
                return HFDatasetDict({"train": HFDataset.from_list([{"id": 0}])})

        ds = _NoTestDataset([], None)
        result = ds.slice(3)
        assert result is ds

    def test_slice_acts_on_explicit_non_test_split(self):
        ds = _BypassLoadDataset(
            _hf_dict=HFDatasetDict(
                {"validation": HFDataset.from_list([{"id": i} for i in range(10)])}
            )
        )
        result = ds.slice(3, split="validation")
        assert len(result.dataset_dict["validation"]) == 3

    def test_slice_missing_split_returns_self(self):
        ds = _BypassLoadDataset(
            _hf_dict=HFDatasetDict({"test": HFDataset.from_list([{"id": 0}])})
        )
        assert ds.slice(1, split="train") is ds


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

    def test_repeat_acts_on_explicit_non_test_split(self):
        ds = _BypassLoadDataset(
            _hf_dict=HFDatasetDict(
                {"validation": HFDataset.from_list([{"id": i} for i in range(3)])}
            )
        )
        result = ds.repeat(2, split="validation")
        assert len(result.dataset_dict["validation"]) == 6

    def test_repeat_missing_split_returns_self(self):
        ds = _BypassLoadDataset(
            _hf_dict=HFDatasetDict({"test": HFDataset.from_list([{"id": 0}])})
        )
        assert ds.repeat(2, split="train") is ds


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

    def test_shuffle_acts_on_explicit_non_test_split(self):
        ds = _BypassLoadDataset(
            _hf_dict=HFDatasetDict(
                {"validation": HFDataset.from_list([{"id": i} for i in range(10)])}
            )
        )
        result = ds.shuffle(seed=1, split="validation")
        assert result is not ds
        assert len(result.dataset_dict["validation"]) == 10

    def test_shuffle_missing_split_returns_self(self):
        ds = _BypassLoadDataset(
            _hf_dict=HFDatasetDict({"test": HFDataset.from_list([{"id": 0}])})
        )
        assert ds.shuffle(seed=1, split="train") is ds


# ===================================================================
# stratified_sample
# ===================================================================
def _make_grouped(group_sizes):
    """Build a _ListDataset with a 'subject' column per {group: size} mapping."""
    samples = []
    idx = 0
    for group, n in group_sizes.items():
        for _ in range(n):
            samples.append({"id": idx, "subject": group})
            idx += 1
    return _ListDataset(samples)


def _make_grouped2(cell_sizes):
    """Build a _ListDataset with 'locale'+'subject' columns per {(locale, subject): n}.

    cell_sizes maps (locale, subject) tuples to row counts.
    """
    samples = []
    idx = 0
    for (locale, subject), n in cell_sizes.items():
        for _ in range(n):
            samples.append({"id": idx, "locale": locale, "subject": subject})
            idx += 1
    return _ListDataset(samples)


def _cell_counts(ds):
    counts: dict = {}
    for row in ds.test_set:
        key = (row["locale"], row["subject"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def _subject_counts(ds):
    counts: dict = {}
    for row in ds.test_set:
        counts[row["subject"]] = counts.get(row["subject"], 0) + 1
    return counts


class TestStratifiedSample:
    def test_proportional_allocation_with_zero_floor(self):
        ds = _make_grouped({"a": 100, "b": 50, "c": 50})
        result = ds.stratified_sample(num=40, by="subject", min_per_group=0, seed=0)
        assert _subject_counts(result) == {"a": 20, "b": 10, "c": 10}
        assert type(result) is type(ds)

    def test_floor_guarantees_small_groups_capped_by_size(self):
        ds = _make_grouped({"a": 100, "b": 2, "c": 2})
        result = ds.stratified_sample(num=12, by="subject", min_per_group=3, seed=0)
        # small groups capped at their full size (< floor); big group takes the rest
        assert _subject_counts(result) == {"a": 8, "b": 2, "c": 2}

    def test_floor_sum_exceeding_num_raises_total_to_floor(self):
        ds = _make_grouped({"a": 5, "b": 5, "c": 5})
        result = ds.stratified_sample(num=2, by="subject", min_per_group=2, seed=0)
        # 3 groups x floor 2 = 6 > num 2 → total raised to 6 to honour the floor
        assert _subject_counts(result) == {"a": 2, "b": 2, "c": 2}

    def test_num_exceeding_total_returns_all(self):
        ds = _make_grouped({"a": 3, "b": 2})
        result = ds.stratified_sample(num=999, by="subject", min_per_group=1, seed=0)
        assert len(result.test_set) == 5

    def test_same_seed_is_deterministic(self):
        ds = _make_grouped({"a": 100, "b": 50, "c": 50})
        ids1 = sorted(
            r["id"] for r in ds.stratified_sample(num=40, by="subject", seed=7).test_set
        )
        ids2 = sorted(
            r["id"] for r in ds.stratified_sample(num=40, by="subject", seed=7).test_set
        )
        assert ids1 == ids2

    def test_different_seed_changes_rows_not_counts(self):
        ds = _make_grouped({"a": 100, "b": 50, "c": 50})
        r0 = ds.stratified_sample(num=40, by="subject", min_per_group=0, seed=0)
        r1 = ds.stratified_sample(num=40, by="subject", min_per_group=0, seed=1)
        assert _subject_counts(r0) == _subject_counts(r1)
        ids0 = sorted(x["id"] for x in r0.test_set)
        ids1 = sorted(x["id"] for x in r1.test_set)
        assert ids0 != ids1

    def test_missing_by_column_raises(self):
        ds = _make_grouped({"a": 3})
        with pytest.raises(ValueError, match="nonexistent"):
            ds.stratified_sample(num=2, by="nonexistent", seed=0)

    def test_no_test_set_returns_self(self):
        class _NoTestDataset(_ListDataset):
            def load(self, name_or_path, **kwargs):
                return HFDatasetDict(
                    {"train": HFDataset.from_list([{"id": 0, "subject": "a"}])}
                )

        ds = _NoTestDataset([], None)
        assert ds.stratified_sample(num=2, by="subject", seed=0) is ds

    def test_empty_split_returns_self(self):
        # An empty split is schema-less; the guard must short-circuit before the
        # column check so it doesn't misreport 'by' as a missing column.
        ds = _ListDataset([])
        assert ds.stratified_sample(num=2, by="subject", seed=0) is ds

    def test_floor_overshoot_logs_warning(self):
        ds = _make_grouped({"a": 5, "b": 5, "c": 5})
        log = _capture_logs(
            lambda: ds.stratified_sample(num=2, by="subject", min_per_group=2, seed=0)
        )
        assert "exceeding the requested num=2" in log

    def test_default_floor_overshoot_logs_warning(self):
        ds = _make_grouped({"a": 1, "b": 1, "c": 1})
        log = _capture_logs(lambda: ds.stratified_sample(num=2, by="subject", seed=0))
        assert "min_per_group=1" in log
        assert "exceeding the requested num=2" in log

    def test_proportional_target_does_not_warn(self):
        ds = _make_grouped({"a": 100, "b": 50, "c": 50})
        log = _capture_logs(
            lambda: ds.stratified_sample(num=40, by="subject", min_per_group=0, seed=0)
        )
        assert log == ""

    def test_single_field_proportional_is_byte_identical(self):
        # Reproducibility lock: golden ids captured from the pre-change
        # implementation. a=ids 0..99, b=100..149, c=150..199.
        ds = _make_grouped({"a": 100, "b": 50, "c": 50})
        result = ds.stratified_sample(num=40, by="subject", min_per_group=0, seed=0)
        ids = sorted(r["id"] for r in result.test_set)
        assert ids == [
            5,
            11,
            15,
            17,
            20,
            27,
            28,
            34,
            41,
            54,
            59,
            68,
            75,
            76,
            77,
            83,
            88,
            93,
            97,
            98,
            106,
            111,
            125,
            130,
            133,
            135,
            136,
            137,
            140,
            145,
            150,
            152,
            155,
            160,
            161,
            166,
            178,
            183,
            187,
            194,
        ]

    def test_equal_allocation_per_group_single_field(self):
        ds = _make_grouped({"a": 100, "b": 50, "c": 50})
        result = ds.stratified_sample(by="subject", per_group=20, seed=42)
        assert _subject_counts(result) == {"a": 20, "b": 20, "c": 20}

    def test_equal_allocation_composite_key(self):
        ds = _make_grouped2(
            {("en", "math"): 5, ("en", "bio"): 5, ("fr", "math"): 5, ("fr", "bio"): 5}
        )
        result = ds.stratified_sample(by=["locale", "subject"], per_group=2, seed=42)
        assert _cell_counts(result) == {
            ("en", "math"): 2,
            ("en", "bio"): 2,
            ("fr", "math"): 2,
            ("fr", "bio"): 2,
        }

    def test_equal_allocation_caps_short_stratum_and_warns(self):
        ds = _make_grouped({"a": 2, "b": 5})
        result = {}

        def run():
            result["ds"] = ds.stratified_sample(by="subject", per_group=3, seed=0)

        log = _capture_logs(run)
        assert _subject_counts(result["ds"]) == {"a": 2, "b": 3}
        assert "per_group=3 unmet for 1 of 2 strata" in log
        assert "short 1 rows total" in log

    def test_composite_key_same_seed_deterministic(self):
        ds = _make_grouped2({("en", "math"): 10, ("fr", "math"): 10})
        ids1 = sorted(
            r["id"]
            for r in ds.stratified_sample(
                by=["locale", "subject"], per_group=3, seed=7
            ).test_set
        )
        ids2 = sorted(
            r["id"]
            for r in ds.stratified_sample(
                by=["locale", "subject"], per_group=3, seed=7
            ).test_set
        )
        assert ids1 == ids2

    def test_requires_exactly_one_budget(self):
        ds = _make_grouped({"a": 5})
        with pytest.raises(ValueError, match="exactly one of 'num' or 'per_group'"):
            ds.stratified_sample(by="subject", seed=0)
        with pytest.raises(ValueError, match="exactly one of 'num' or 'per_group'"):
            ds.stratified_sample(by="subject", num=2, per_group=2, seed=0)

    def test_min_per_group_excludes_per_group(self):
        ds = _make_grouped({"a": 5})
        with pytest.raises(ValueError, match="cannot be combined with 'per_group'"):
            ds.stratified_sample(by="subject", per_group=2, min_per_group=1, seed=0)

    def test_empty_by_list_raises(self):
        ds = _make_grouped({"a": 5})
        with pytest.raises(ValueError, match="at least one column"):
            ds.stratified_sample(by=[], per_group=2, seed=0)

    def test_missing_composite_column_raises(self):
        ds = _make_grouped2({("en", "math"): 5})
        with pytest.raises(ValueError, match="nonexistent"):
            ds.stratified_sample(by=["locale", "nonexistent"], per_group=2, seed=0)


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
