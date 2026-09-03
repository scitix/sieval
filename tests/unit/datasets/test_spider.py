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
    ARCHIVE_BASENAME,
    SPIDER_ARCHIVE_REVISION,
    SPIDER_ARCHIVE_SHA256,
    TEST_SUITE_BASENAME,
    TEST_SUITE_SHA256,
    TEST_SUITE_URL,
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


def _make_db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE singer (id int, name text)")
    conn.execute("INSERT INTO singer VALUES (1, 'Joe')")
    conn.commit()
    conn.close()
    return path


def _make_archive(tmp_path: Path, with_test_suite: bool = True) -> Path:
    """Miniatures of both staged archives, each in upstream's own layout.

    They are shaped differently on purpose, which is the thing most likely to
    break: `spider_data.zip` carries its own top-level directory and the
    test-suite archive does not.
    """
    staged = tmp_path / "spider"
    staged.mkdir()
    db_file = _make_db(tmp_path / "concert_singer.sqlite")
    with zipfile.ZipFile(staged / ARCHIVE_BASENAME, "w") as archive:
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
    if with_test_suite:
        with zipfile.ZipFile(staged / TEST_SUITE_BASENAME, "w") as archive:
            # No top-level directory of its own -- `database/` is the root.
            archive.write(db_file, "database/concert_singer/concert_singer.sqlite")
            archive.write(db_file, "database/concert_singer/concert_singer_1.sqlite")
            # AppleDouble metadata from however upstream zipped this. The real
            # archive ships ~48 of these mirroring `database/`.
            archive.writestr("__MACOSX/database/._concert_singer", "junk")
    return staged


def test_source_and_checksums_are_pinned():
    """Both archives, both checksums.

    The test-suite archive has no mirror and comes from Drive, so the checksum
    is the whole provenance story: a link that starts serving something else has
    to fail the download rather than quietly change a published score.
    """
    meta = get_dataset_meta(SpiderDataset)
    assert meta.source == (
        "url:https://huggingface.co/datasets/HAL-9001/spider-databases/resolve/"
        f"{SPIDER_ARCHIVE_REVISION}/{ARCHIVE_BASENAME}",
        f"url:{TEST_SUITE_URL}",
    )
    checksums = dict(meta.checksums)
    assert checksums[ARCHIVE_BASENAME] == f"sha256:{SPIDER_ARCHIVE_SHA256}"
    assert checksums[TEST_SUITE_BASENAME] == f"sha256:{TEST_SUITE_SHA256}"


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


def test_the_test_suite_archive_is_extracted_beside_the_data(tmp_path):
    """The distilled databases back the headline metric, so they must be staged.

    Extracted under its own root rather than into `spider_data/`, because the
    archive has no top-level directory to keep them apart.
    """
    dataset = SpiderDataset(str(_make_archive(tmp_path)))
    test_suite_db_dir = dataset.test_suite_db_dir
    assert test_suite_db_dir is not None
    suite = Path(test_suite_db_dir, "concert_singer")
    assert sorted(p.name for p in suite.iterdir()) == [
        "concert_singer.sqlite",
        "concert_singer_1.sqlite",
    ]
    # Not folded into the dataset's own tree, whose `database/` holds the one
    # shipped copy the pre-2020 metrics read.
    assert Path(test_suite_db_dir) != Path(str(dataset.db_dir))


def test_apple_double_metadata_is_not_extracted(tmp_path):
    """Cosmetic, not correctness -- the grader only globs `database/<db_id>/`.

    Asserted anyway because a `__MACOSX/` tree next to the real one is the kind
    of thing a later reader takes for a second copy of the suite.
    """
    dataset = SpiderDataset(str(_make_archive(tmp_path)))
    root = Path(str(dataset.test_suite_db_dir)).parent
    assert sorted(p.name for p in root.iterdir()) == [
        ".sieval-extracted",
        "database",
    ]


@pytest.mark.parametrize(
    ("marker", "label"),
    [
        ("spider_data/.sieval-extracted", "206 MB"),
        ("spider_test_suite/.sieval-extracted", "1.3 GB"),
    ],
)
def test_extraction_is_idempotent(tmp_path, marker, label):
    """A second load must not re-extract; each marker guards its own unzip."""
    staged = _make_archive(tmp_path)
    SpiderDataset(str(staged))
    stamp = (staged / marker).stat().st_mtime_ns
    SpiderDataset(str(staged))
    assert (staged / marker).stat().st_mtime_ns == stamp, label


def test_db_dir_is_none_without_load():
    """Preloaded-dict construction (tests, slices) has nothing staged."""
    dataset = SpiderDataset(_hf_dict=HFDatasetDict({}))
    assert dataset.db_dir is None
    assert dataset.tables_json_path is None


def test_missing_archive_names_the_download_command(tmp_path):
    # Matched on the archive-specific wording, not just the command: both
    # archives name the same command, so the looser pattern would let this test
    # and the one below pass on each other's error.
    with pytest.raises(FileNotFoundError, match="Spider archive not found"):
        SpiderDataset(str(tmp_path))


def test_missing_test_suite_archive_names_the_download_command(tmp_path):
    """The failure a user staged before the metric moved will actually hit.

    It must not read as an internal error, and it must not suggest the metric
    can be run without the archive -- it cannot.
    """
    staged = _make_archive(tmp_path, with_test_suite=False)
    with pytest.raises(FileNotFoundError, match="test-suite databases not found"):
        SpiderDataset(str(staged))
