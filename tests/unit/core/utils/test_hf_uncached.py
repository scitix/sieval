"""Tests for sieval/core/utils/hf_uncached.py.

Each case loads one local source twice, through the Arrow cache and inside
``uncached_reads``, and compares what comes back: rows, features, split names,
fingerprints, and which files land under ``HF_DATASETS_CACHE``.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset as HFDataset
from datasets import (
    DatasetDict,
    DatasetInfo,
    Features,
    Value,
    config,
    load_dataset,
    load_dataset_builder,
)
from datasets.builder import DatasetBuilder
from datasets.exceptions import DatasetGenerationError
from datasets.utils.info_utils import NonMatchingSplitsSizesError

import sieval.core.utils.hf_uncached as hf_uncached
from sieval.core.utils.hf import ensure_dataset
from sieval.core.utils.hf_uncached import uncached_reads


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh ``HF_DATASETS_CACHE``, as a job's first dry-run sees it."""
    root = tmp_path / "hf-cache"
    monkeypatch.setattr(config, "HF_DATASETS_CACHE", root)
    return root


def _arrow_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.arrow")) if root.exists() else []


def _parquet_repo(root: Path) -> Path:
    """A staged repo of two splits, the test split in two shards."""
    repo = root / "org" / "bench"
    (repo / "main").mkdir(parents=True)
    rows = [{"question": f"q{i}", "answer": i} for i in range(7)]
    pq.write_table(pa.Table.from_pylist(rows[:4]), repo / "main" / "test-0.parquet")
    pq.write_table(pa.Table.from_pylist(rows[4:]), repo / "main" / "test-1.parquet")
    pq.write_table(
        pa.Table.from_pylist([{"question": "dev", "answer": -1}]),
        repo / "main" / "dev-0.parquet",
    )
    return repo


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _load_parquet(repo: Path) -> DatasetDict:
    files = {
        "test": [str(p) for p in sorted((repo / "main").glob("test-*.parquet"))],
        "dev": [str(repo / "main" / "dev-0.parquet")],
    }
    loaded = load_dataset("parquet", data_files=files)
    assert isinstance(loaded, DatasetDict)
    return loaded


class TestSameRows:
    def test_parquet_splits_match_the_cached_load(
        self, tmp_path: Path, cache: Path
    ) -> None:
        repo = _parquet_repo(tmp_path)
        with uncached_reads() as log:
            light = _load_parquet(repo)
        assert _arrow_files(cache) == []
        full = _load_parquet(repo)
        assert _arrow_files(cache), "the cached load writes Arrow files"

        assert list(light) == list(full) == ["test", "dev"]
        for split in full:
            assert light[split].to_list() == full[split].to_list()
            assert light[split].features == full[split].features
            assert str(light[split].split) == str(full[split].split)
            # The same fingerprint, so a later transform hashes alike.
            assert light[split]._fingerprint == full[split]._fingerprint
        assert [r.path for r in log.reads] == ["memory"]

    def test_one_named_split(self, tmp_path: Path, cache: Path) -> None:
        rows = [{"x": i} for i in range(5)]
        path = _jsonl(tmp_path / "rows.jsonl", rows)
        with uncached_reads() as log:
            light = ensure_dataset(
                load_dataset("json", data_files=str(path), split="train")
            )
        assert light.to_list() == rows
        assert _arrow_files(cache) == []
        assert [r.path for r in log.reads] == ["memory"]

    def test_transforms_after_the_read_write_no_cache(
        self, tmp_path: Path, cache: Path
    ) -> None:
        # What a loader such as C-Eval does to each subject's rows.
        path = _jsonl(tmp_path / "rows.jsonl", [{"x": i} for i in range(6)])
        with uncached_reads():
            rows = ensure_dataset(
                load_dataset("json", data_files=str(path), split="train")
            )
            tagged = rows.map(lambda row: {"tag": "s"}).filter(lambda r: r["x"] % 2)
        assert tagged.to_list() == [{"x": x, "tag": "s"} for x in (1, 3, 5)]
        assert _arrow_files(cache) == []


class TestCachedFallback:
    def test_a_sliced_split_reads_through_the_cache(
        self, tmp_path: Path, cache: Path
    ) -> None:
        path = _jsonl(tmp_path / "rows.jsonl", [{"x": i} for i in range(5)])
        with uncached_reads() as log:
            head = ensure_dataset(
                load_dataset("json", data_files=str(path), split="train[:2]")
            )
        assert head.to_list() == [{"x": 0}, {"x": 1}]
        assert _arrow_files(cache), "a slice is the cached path's to read"
        (read,) = log.reads
        assert (read.path, read.reason) == (
            "cache",
            "split train[:2] is not a split name",
        )

    def test_a_split_over_the_size_bound_reads_through_the_cache(
        self, tmp_path: Path, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _jsonl(tmp_path / "rows.jsonl", [{"x": i} for i in range(5)])
        monkeypatch.setattr(hf_uncached, "MAX_IN_MEMORY_BYTES", 1)
        with uncached_reads() as log:
            rows = ensure_dataset(
                load_dataset("json", data_files=str(path), split="train")
            )
        assert len(rows) == 5
        (read,) = log.reads
        assert read.path == "cache"
        assert read.reason.startswith("split train is ")

    def test_batches_of_different_schemas_read_through_the_cache(
        self, tmp_path: Path, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A column null in the first batch and a string in the next: the cache
        # writer casts each batch to one schema, which a concatenation cannot.
        monkeypatch.setattr(hf_uncached, "_BATCH_ROWS", 1)
        rows = [{"x": 1, "note": None}, {"x": 2, "note": "n"}]
        path = _jsonl(tmp_path / "rows.jsonl", rows)

        def concat_fails(*_: object, **__: object) -> pa.Table:
            raise pa.ArrowInvalid("Schema at index 1 was different")

        monkeypatch.setattr(hf_uncached.pa, "concat_tables", concat_fails)
        with uncached_reads() as log:
            loaded = ensure_dataset(
                load_dataset("json", data_files=str(path), split="train")
            )
        monkeypatch.undo()
        assert loaded.to_list() == rows
        (read,) = log.reads
        assert read.path == "cache"
        assert read.reason.startswith("split did not stream")

    def test_an_empty_split_fails_as_the_cached_load_does(
        self, tmp_path: Path, cache: Path
    ) -> None:
        path = tmp_path / "empty.csv"
        path.write_text("x,y\n")
        features = Features({"x": Value("int64"), "y": Value("string")})
        # Light first: the cached load leaves a complete cache, read from then.
        with (
            pytest.raises(ValueError, match="corresponds to no data") as light,
            uncached_reads() as log,
        ):
            load_dataset("csv", data_files=str(path), split="train", features=features)
        with pytest.raises(ValueError, match="corresponds to no data") as full:
            load_dataset("csv", data_files=str(path), split="train", features=features)
        assert str(light.value) == str(full.value)
        assert [(r.path, r.reason) for r in log.reads] == [
            ("cache", "split train has no rows")
        ]

    def test_a_loading_script_reads_through_the_cache(
        self, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The generator builder's cached path also checks for duplicate keys.
        from datasets import GeneratorBasedBuilder

        builder = object.__new__(GeneratorBasedBuilder)
        assert hf_uncached._memory_splits(builder, "test") == (None, "loading script")

    def test_a_split_off_local_disk_reads_through_the_cache(
        self, tmp_path: Path, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _jsonl(tmp_path / "rows.jsonl", [{"x": 1}])
        size = hf_uncached._local_size
        monkeypatch.setattr(
            hf_uncached, "_local_size", lambda paths: size(["hf://org/r/f.jsonl"])
        )
        with uncached_reads() as log:
            rows = ensure_dataset(
                load_dataset("json", data_files=str(path), split="train")
            )
        assert rows.to_list() == [{"x": 1}]
        assert [(r.path, r.reason) for r in log.reads] == [
            ("cache", "split train is not on local disk")
        ]


class TestChecksKept:
    def test_a_split_size_the_card_contradicts_is_refused(
        self, tmp_path: Path, cache: Path
    ) -> None:
        # The card's `num_examples` is what the cached path verifies.
        repo = tmp_path / "card"
        repo.mkdir()
        _jsonl(repo / "test.jsonl", [{"x": i} for i in range(3)])
        (repo / "README.md").write_text(
            "---\n"
            "dataset_info:\n"
            "  features:\n"
            "  - name: x\n"
            "    dtype: int64\n"
            "  splits:\n"
            "  - name: test\n"
            "    num_examples: 4\n"
            "configs:\n"
            "- config_name: default\n"
            "  data_files:\n"
            "  - split: test\n"
            "    path: test.jsonl\n"
            "---\n"
        )
        with pytest.raises(NonMatchingSplitsSizesError):
            load_dataset(str(repo))
        with pytest.raises(NonMatchingSplitsSizesError), uncached_reads():
            load_dataset(str(repo))

    def test_an_unrequested_split_that_does_not_parse_fails_as_the_cached_load_does(
        self, tmp_path: Path, cache: Path
    ) -> None:
        # The cached path prepares every split, not only the one asked for.
        pq.write_table(pa.table({"x": [1, 2, 3]}), tmp_path / "test.parquet")
        (tmp_path / "train.parquet").write_bytes(b"not parquet")
        files = {
            "test": str(tmp_path / "test.parquet"),
            "train": str(tmp_path / "train.parquet"),
        }
        with pytest.raises(DatasetGenerationError):
            load_dataset("parquet", data_files=files, split="test")
        with pytest.raises(DatasetGenerationError), uncached_reads() as log:
            load_dataset("parquet", data_files=files, split="test")
        (read,) = log.reads
        assert read.path == "cache"
        assert read.reason.startswith("split did not stream")

    def test_an_unrequested_split_size_the_card_contradicts_is_refused(
        self, tmp_path: Path, cache: Path
    ) -> None:
        repo = _card_repo(tmp_path, test_rows=3, dev_rows=2, dev_card=5)
        with pytest.raises(NonMatchingSplitsSizesError):
            load_dataset(str(repo), split="test")
        with pytest.raises(NonMatchingSplitsSizesError), uncached_reads():
            load_dataset(str(repo), split="test")

    def test_no_checks_skips_the_split_size_check_as_the_cached_load_does(
        self, tmp_path: Path, cache: Path
    ) -> None:
        # `datasets` 5 passes the mode to `download_and_prepare` only.
        # The in-memory read goes first: a cached one records the true counts.
        repo = _card_repo(tmp_path, test_rows=3, dev_rows=2, dev_card=5)
        with uncached_reads() as log:
            rows = ensure_dataset(
                load_dataset(str(repo), split="test", verification_mode="no_checks")
            )
        assert len(rows) == 3
        assert [r.path for r in log.reads] == ["memory"]
        load_dataset(str(repo), split="test", verification_mode="no_checks")

    def test_an_unrequested_split_is_counted_not_held(
        self, tmp_path: Path, cache: Path
    ) -> None:
        repo = _card_repo(tmp_path, test_rows=3, dev_rows=2, dev_card=2)
        with uncached_reads() as log:
            rows = ensure_dataset(load_dataset(str(repo), split="test"))
        assert rows.to_list() == [{"x": i} for i in range(3)]
        assert [r.path for r in log.reads] == ["memory"]
        assert _arrow_files(cache) == []


def _card_repo(root: Path, *, test_rows: int, dev_rows: int, dev_card: int) -> Path:
    """A repo of a test and a dev split whose card claims *dev_card* dev rows."""
    repo = root / "card2"
    repo.mkdir()
    _jsonl(repo / "test.jsonl", [{"x": i} for i in range(test_rows)])
    _jsonl(repo / "dev.jsonl", [{"x": -i} for i in range(dev_rows)])
    (repo / "README.md").write_text(
        "---\n"
        "dataset_info:\n"
        "  features:\n"
        "  - name: x\n"
        "    dtype: int64\n"
        "  splits:\n"
        f"  - name: test\n    num_examples: {test_rows}\n"
        f"  - name: dev\n    num_examples: {dev_card}\n"
        "configs:\n"
        "- config_name: default\n"
        "  data_files:\n"
        "  - split: test\n    path: test.jsonl\n"
        "  - split: dev\n    path: dev.jsonl\n"
        "---\n"
    )
    return repo


class TestPatching:
    def test_the_builder_is_restored_after_an_error(self, tmp_path: Path) -> None:
        before = (DatasetBuilder.download_and_prepare, DatasetBuilder.as_dataset)
        with pytest.raises(FileNotFoundError), uncached_reads():
            load_dataset("json", data_files=str(tmp_path / "absent.jsonl"))
        after = (DatasetBuilder.download_and_prepare, DatasetBuilder.as_dataset)
        assert after == before

    @pytest.mark.parametrize(
        "forwarded",
        [
            {"in_memory": False},
            {"verification_mode": "basic_checks", "in_memory": False},
            {"run_post_process": True, "verification_mode": None, "in_memory": False},
        ],
        ids=["datasets-5", "datasets-4", "datasets-4-full"],
    )
    def test_the_cached_path_passes_on_only_what_the_caller_gave(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forwarded: dict
    ) -> None:
        # `datasets` 5's `as_dataset` raises TypeError on a 4-only keyword.
        seen: list[dict] = []

        def real(_builder: DatasetBuilder, **kwargs: Any) -> str:
            kwargs.pop("split")
            seen.append(kwargs)
            return "cached"

        monkeypatch.setattr(DatasetBuilder, "as_dataset", real)
        monkeypatch.setattr(
            DatasetBuilder, "download_and_prepare", lambda *a, **k: None
        )
        path = _jsonl(tmp_path / "rows.jsonl", [{"x": 0}, {"x": 1}])
        builder = load_dataset_builder("json", data_files=str(path))
        with uncached_reads():
            # A slice always reads through the cache.
            read = DatasetBuilder.as_dataset(builder, split="train[:1]", **forwarded)
        assert read == "cached"
        assert seen == [forwarded]

    def test_a_nested_block_is_refused(self) -> None:
        with (
            uncached_reads(),
            pytest.raises(RuntimeError, match="already open"),
            uncached_reads(),
        ):
            pass
        # The outer block released it.
        with uncached_reads():
            pass

    def test_a_card_is_parsed_once_per_block(
        self, tmp_path: Path, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # C-Eval loads one config per subject from the same repo.
        repo = _parquet_repo(tmp_path)
        calls: list[str] = []
        factory = hf_uncached.hf_load.dataset_module_factory

        def counting(path: str, *args: Any, **kwargs: Any) -> Any:
            calls.append(path)
            return factory(path, *args, **kwargs)

        monkeypatch.setattr(hf_uncached.hf_load, "dataset_module_factory", counting)
        with uncached_reads():
            first = load_dataset(str(repo), data_dir="main", split="test")
            again = load_dataset(str(repo), data_dir="main", split="test")
        assert ensure_dataset(first).to_list() == ensure_dataset(again).to_list()
        assert calls == [str(repo)]


class TestAlreadyCached:
    def test_a_complete_cache_is_read_from_and_not_rebuilt(
        self, tmp_path: Path, cache: Path
    ) -> None:
        repo = _parquet_repo(tmp_path)
        full = _load_parquet(repo)
        before = _arrow_files(cache)
        with uncached_reads() as log:
            light = _load_parquet(repo)
        assert _arrow_files(cache) == before
        assert [(r.path, r.reason) for r in log.reads] == [("cache", "already cached")]
        for split in full:
            assert light[split].to_list() == full[split].to_list()
            assert light[split].cache_files == full[split].cache_files

    def test_a_forced_redownload_still_reads_in_memory(
        self, tmp_path: Path, cache: Path
    ) -> None:
        repo = _parquet_repo(tmp_path)
        _load_parquet(repo)
        with uncached_reads() as log:
            load_dataset(
                str(repo / "main"),
                data_files={"test": "test-*.parquet"},
                download_mode="force_redownload",
            )
        assert [r.path for r in log.reads] == ["memory"]


def _pid(_row: dict) -> dict:
    return {"pid": os.getpid()}


def _even(row: dict) -> bool:
    return row["x"] % 2 == 0


class TestCostsDropped:
    def test_map_on_rows_held_in_memory_runs_in_this_process(
        self, tmp_path: Path, cache: Path
    ) -> None:
        path = _jsonl(tmp_path / "rows.jsonl", [{"x": i} for i in range(8)])
        with uncached_reads():
            rows = ensure_dataset(
                load_dataset("json", data_files=str(path), split="train")
            )
            mapped = rows.map(_pid, num_proc=4)
            kept = rows.filter(_even, num_proc=4)
        assert set(mapped["pid"]) == {os.getpid()}
        assert kept["x"] == [0, 2, 4, 6]
        assert mapped._fingerprint == rows.map(_pid)._fingerprint

    def test_map_on_a_cached_dataset_keeps_num_proc(
        self, tmp_path: Path, cache: Path
    ) -> None:
        path = _jsonl(tmp_path / "rows.jsonl", [{"x": 0}, {"x": 1}])
        load_dataset("json", data_files=str(path), split="train")
        with uncached_reads() as log:
            rows = ensure_dataset(
                load_dataset("json", data_files=str(path), split="train")
            )
            mapped = rows.map(_pid, num_proc=2)
        assert [r.reason for r in log.reads] == ["already cached"]
        assert len(set(mapped["pid"])) == 2

    def test_map_keeps_num_proc_after_the_block(
        self, tmp_path: Path, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Any] = []
        real = HFDataset.map

        def spy(self: HFDataset, *args: Any, **kwargs: Any) -> Any:
            seen.append(kwargs.get("num_proc"))
            return real(self, *args, **kwargs)

        monkeypatch.setattr(HFDataset, "map", spy)
        with uncached_reads():
            pass
        HFDataset.from_list([{"x": 1}]).map(lambda row: row, num_proc=1)
        assert seen == [1]

    def test_no_download_count_is_sent(
        self, tmp_path: Path, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent: list[str] = []

        class Session:
            def head(self, url: str, **_: Any) -> None:
                sent.append(url)

        monkeypatch.setattr(config, "HF_HUB_OFFLINE", False)
        monkeypatch.setattr(config, "HF_UPDATE_DOWNLOAD_COUNTS", True)
        monkeypatch.setattr(hf_uncached.hf_load, "get_session", lambda: Session())
        path = _jsonl(tmp_path / "rows.jsonl", [{"x": 1}])
        with uncached_reads():
            load_dataset("json", data_files=str(path), split="train")
        assert sent == []
        load_dataset("json", data_files=str(path), split="train")
        assert len(sent) == 1

    def test_each_load_gets_its_own_config_info(self, tmp_path: Path) -> None:
        repo = _card_repo(tmp_path, test_rows=3, dev_rows=2, dev_card=2)
        with uncached_reads():
            first = load_dataset_builder(str(repo))
            first.info.description = "changed"
            again = load_dataset_builder(str(repo))
        assert again.info.description != "changed"

    def test_an_info_read_by_key_is_a_copy(self) -> None:
        infos = hf_uncached._InfosOnRead({"default": DatasetInfo(description="a")})
        infos["default"].description = "changed"
        assert infos["default"].description == "a"
