"""
Unit tests for sieval/core/utils/hf.py.

Covers: apply_eval_split, ensure_dataset_dict, ensure_dataset,
ensure_dataset_list — all four public functions and their error branches.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import IterableDataset as HFIterableDataset
from datasets import IterableDatasetDict as HFIterableDatasetDict

from sieval.core.utils.hf import (
    apply_eval_split,
    ensure_dataset,
    ensure_dataset_dict,
    ensure_dataset_list,
    maybe_resolve_hf_path,
)


# ===================================================================
# Helpers
# ===================================================================
def _make_dict(*splits: str) -> HFDatasetDict:
    return HFDatasetDict(
        {split: HFDataset.from_list([{"x": i} for i in range(3)]) for split in splits}
    )


def _make_iterable_dataset() -> HFIterableDataset:
    # Avoid constructing real IterableDataset from generator.
    # That path can require shared memory / file locks in sandboxed CI.
    return object.__new__(HFIterableDataset)


def _make_iterable_dict() -> HFIterableDatasetDict:
    # Empty construction is enough for isinstance checks in ensure_dataset_dict.
    return HFIterableDatasetDict()


# ===================================================================
# apply_eval_split
# ===================================================================
class TestApplyEvalSplit:
    def test_no_eval_split_returns_unchanged(self):
        """eval_split=None should return the dataset unchanged."""
        d = _make_dict("train", "test")
        result = apply_eval_split(d, eval_split=None)
        assert set(result.keys()) == {"train", "test"}

    def test_eval_split_equal_to_test_is_no_op(self):
        """eval_split='test' must not modify the dict."""
        d = _make_dict("train", "test")
        result = apply_eval_split(d, eval_split="test")
        assert set(result.keys()) == {"train", "test"}

    def test_eval_split_copies_split_to_test(self):
        """A valid non-test eval_split should be mapped to 'test'."""
        d = _make_dict("train", "validation")
        result = apply_eval_split(d, eval_split="validation")
        assert "test" in result
        # The mapped test split should contain the same rows as validation
        assert len(result["test"]) == len(d["validation"])

    def test_eval_split_missing_key_is_no_op(self):
        """eval_split pointing to a non-existent split is a no-op."""
        d = _make_dict("train")
        result = apply_eval_split(d, eval_split="validation")
        assert "test" not in result


# ===================================================================
# ensure_dataset_dict
# ===================================================================
class TestEnsureDatasetDict:
    def test_returns_dataset_dict_as_is(self):
        d = _make_dict("train", "test")
        result = ensure_dataset_dict(d)
        assert result is d

    def test_raises_for_iterable_dataset_dict(self):
        with pytest.raises(TypeError, match="IterableDatasetDict"):
            ensure_dataset_dict(_make_iterable_dict())

    def test_iterable_dataset_dict_hits_specific_unsupported_message(self):
        with pytest.raises(
            TypeError, match="is not supported by current dataset interfaces"
        ):
            ensure_dataset_dict(_make_iterable_dict())

    def test_raises_for_plain_dataset(self):
        ds = HFDataset.from_list([{"x": 0}])
        with pytest.raises(TypeError, match="DatasetDict"):
            ensure_dataset_dict(ds)

    def test_raises_for_iterable_dataset(self):
        with pytest.raises(TypeError, match="DatasetDict"):
            ensure_dataset_dict(_make_iterable_dataset())


# ===================================================================
# ensure_dataset
# ===================================================================
class TestEnsureDataset:
    def test_returns_dataset_as_is(self):
        ds = HFDataset.from_list([{"x": 0}])
        result = ensure_dataset(ds)
        assert result is ds

    def test_raises_for_iterable_dataset(self):
        with pytest.raises(TypeError, match="IterableDataset"):
            ensure_dataset(_make_iterable_dataset())

    def test_raises_for_dataset_dict(self):
        d = _make_dict("test")
        with pytest.raises(TypeError, match="Dataset"):
            ensure_dataset(d)

    def test_raises_for_iterable_dataset_dict(self):
        with pytest.raises(TypeError, match="Dataset"):
            ensure_dataset(_make_iterable_dict())


# ===================================================================
# ensure_dataset_list
# ===================================================================
class TestEnsureDatasetList:
    def test_returns_list_of_datasets(self):
        ds1 = HFDataset.from_list([{"x": 0}])
        ds2 = HFDataset.from_list([{"x": 1}])
        result = ensure_dataset_list([ds1, ds2])
        assert result == [ds1, ds2]

    def test_empty_list_returns_empty(self):
        assert ensure_dataset_list([]) == []

    def test_raises_for_iterable_dataset_in_list(self):
        ds = HFDataset.from_list([{"x": 0}])
        with pytest.raises(TypeError, match="IterableDataset"):
            ensure_dataset_list([ds, _make_iterable_dataset()])

    def test_raises_for_dataset_dict_in_list(self):
        with pytest.raises(TypeError, match="Dataset"):
            ensure_dataset_list([_make_dict("test")])


# ===================================================================
# maybe_resolve_hf_path
# ===================================================================
class TestMaybeResolveHfPath:
    def test_bare_repo_id_resolves_to_data_dir(self, monkeypatch, tmp_path):
        """`org/name` patterns resolve to `{SIEVAL_DATA_DIR}/org/name`."""
        monkeypatch.setenv("SIEVAL_DATA_DIR", str(tmp_path))
        result = maybe_resolve_hf_path("HuggingFaceH4/aime_2024")
        assert result == str(tmp_path / "HuggingFaceH4" / "aime_2024")

    def test_absolute_path_with_env_expansion_passes_through(self):
        """`/data4/sieval/data/drop` after env expansion has multiple slashes
        and a leading slash; the regex must reject these."""
        path = "/data4/sieval/data/drop"
        assert maybe_resolve_hf_path(path) == path

    def test_unexpanded_var_passes_through(self):
        """`${SIEVAL_DATA_DIR}/mmlu` (caller didn't expand) should pass through.

        The shape after env expansion is the absolute path which the regex
        rejects; the unexpanded form is rejected for the same reason
        (contains `$`, `{`, `}` which `\\w` doesn't match)."""
        path = "${SIEVAL_DATA_DIR}/mmlu"
        assert maybe_resolve_hf_path(path) == path

    def test_csv_path_passes_through(self):
        """`./data/foo.csv` is a relative file path, not a repo_id."""
        assert maybe_resolve_hf_path("./data/foo.csv") == "./data/foo.csv"

    def test_three_segment_path_passes_through(self):
        """`org/sub/name` is not a valid HF repo_id (HF is exactly `org/name`)."""
        assert maybe_resolve_hf_path("org/sub/name") == "org/sub/name"

    def test_leading_dot_segment_passes_through(self):
        """`../foo`, `./.`, `..a/.b` all have leading-dot segments — the regex
        requires the first char to be alphanumeric/underscore, so these are
        rejected and don't get silently rewritten to `{data_dir}/../foo`."""
        for path in ("../foo", "./.", "..a/.b", ".hidden/repo"):
            assert maybe_resolve_hf_path(path) == path, f"should pass through: {path!r}"

    def test_missing_staging_dir_still_returns_resolved_path(
        self, monkeypatch, tmp_path
    ):
        """When the repo_id matches but the staging dir doesn't exist yet,
        the helper returns the resolved path anyway. The downstream
        `load_dataset` call then raises FileNotFoundError, which
        `session._setup_datasets` catches to emit the dataset-download hint."""
        monkeypatch.setenv("SIEVAL_DATA_DIR", str(tmp_path))
        result = maybe_resolve_hf_path("nonexistent_org/nonexistent_repo")
        assert result == str(tmp_path / "nonexistent_org" / "nonexistent_repo")
