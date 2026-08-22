"""Unit tests for the Spider 1.0 dataset wrapper.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest
from datasets import DatasetDict as HFDatasetDict

from sieval.core.datasets.meta import get_dataset_meta
from sieval.datasets.spider import (
    SPIDER_ARCHIVE_REVISION,
    SPIDER_ARCHIVE_SHA256,
    SpiderDataset,
)


def _row(db_id: str = "concert_singer", question: str = "How many singers do we have?"):
    return {
        "db_id": db_id,
        "query": "SELECT count(*) FROM singer",
        "query_toks": ["SELECT", "count", "(", "*", ")", "FROM", "singer"],
        "query_toks_no_value": ["select", "count", "(", "*", ")", "from", "singer"],
        "question": question,
        "question_toks": ["How", "many", "singers", "do", "we", "have", "?"],
        # The column with no stable Arrow schema; the loader must drop it.
        "sql": {
            "select": [False, []],
            "from": {"conds": [], "table_units": []},
            "where": [],
            "groupBy": [],
            "having": [],
            "orderBy": [],
            "limit": None,
            "intersect": None,
            "union": None,
            "except": None,
        },
    }


def _make_archive(tmp_path: Path) -> Path:
    """A miniature spider_data.zip with upstream's member layout."""
    staged = tmp_path / "spider"
    staged.mkdir()
    db_file = tmp_path / "concert_singer.sqlite"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE singer (id int, name text)")
    conn.execute("INSERT INTO singer VALUES (1, 'Joe')")
    conn.commit()
    conn.close()
    with zipfile.ZipFile(staged / "spider_data.zip", "w") as archive:
        archive.writestr("spider_data/dev.json", json.dumps([_row()]))
        archive.writestr(
            "spider_data/train_spider.json", json.dumps([_row(question="Train q?")])
        )
        archive.writestr(
            "spider_data/tables.json", json.dumps([{"db_id": "concert_singer"}])
        )
        archive.write(
            db_file, "spider_data/database/concert_singer/concert_singer.sqlite"
        )
    return staged


def test_source_and_checksum_are_pinned():
    meta = get_dataset_meta(SpiderDataset)
    assert meta.source == (
        "url:https://huggingface.co/datasets/HAL-9001/spider-databases/resolve/"
        f"{SPIDER_ARCHIVE_REVISION}/spider_data.zip",
    )
    assert dict(meta.checksums)["spider_data.zip"] == f"sha256:{SPIDER_ARCHIVE_SHA256}"


def test_license_is_the_data_license():
    """CC-BY-SA-4.0 governs the rows and databases; the grader code is Apache-2.0."""
    assert get_dataset_meta(SpiderDataset).license == "CC-BY-SA-4.0"


def test_load_drops_the_unrepresentable_sql_column(tmp_path):
    """`sql` is a recursive parse tree; Arrow cannot infer a schema for it.

    Upstream's own grader re-derives it from `query`, so dropping it loses
    nothing. Without the drop, `load_dataset` raises ArrowInvalid outright.
    """
    dataset = SpiderDataset(str(_make_archive(tmp_path)))
    assert dataset.dataset_dict["test"].column_names == [
        "db_id",
        "query",
        "query_toks",
        "query_toks_no_value",
        "question",
        "question_toks",
    ]


def test_dev_is_exposed_as_the_test_split(tmp_path):
    """The runner evaluates `Dataset.test_set`, which reads only `test`.

    Spider's reported split is its dev set, so dev.json lands under `test`.
    Any other name evaluates zero samples, silently.
    """
    dataset = SpiderDataset(str(_make_archive(tmp_path)))
    assert dataset.test_set is not None
    assert len(dataset.test_set) == 1


def test_load_exposes_both_splits(tmp_path):
    dataset = SpiderDataset(str(_make_archive(tmp_path)))
    assert set(dataset.dataset_dict.keys()) == {"train", "test"}
    assert dataset.dataset_dict["test"][0]["question"] == (
        "How many singers do we have?"
    )
    assert dataset.dataset_dict["train"][0]["question"] == "Train q?"


def test_db_dir_and_tables_json_point_at_extracted_files(tmp_path):
    dataset = SpiderDataset(str(_make_archive(tmp_path)))
    db_dir = dataset.db_dir
    tables_json_path = dataset.tables_json_path
    assert db_dir is not None and tables_json_path is not None
    assert Path(db_dir, "concert_singer", "concert_singer.sqlite").is_file()
    assert Path(tables_json_path).is_file()


def test_extraction_is_idempotent(tmp_path):
    """A second load must not re-extract; the marker guards a 206 MB unzip."""
    staged = _make_archive(tmp_path)
    SpiderDataset(str(staged))
    marker = staged / "spider_data" / ".sieval-extracted"
    stamp = marker.stat().st_mtime_ns
    SpiderDataset(str(staged))
    assert marker.stat().st_mtime_ns == stamp


def test_db_dir_is_none_without_load():
    """Preloaded-dict construction (tests, slices) has nothing staged."""
    dataset = SpiderDataset(_hf_dict=HFDatasetDict({}))
    assert dataset.db_dir is None
    assert dataset.tables_json_path is None


def test_missing_archive_names_the_download_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="sieval dataset download spider"):
        SpiderDataset(str(tmp_path))
