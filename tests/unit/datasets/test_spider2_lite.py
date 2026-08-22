"""Unit tests for the Spider 2.0-lite dataset wrapper.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest
from datasets import DatasetDict as HFDatasetDict

from sieval.core.datasets.meta import get_dataset_meta
from sieval.datasets.spider2_lite import (
    ARCHIVE_BASENAME,
    LOCALDB_SHA256,
    SPIDER2_ARCHIVE_SHA256,
    Spider2LiteDataset,
    backend_for,
)

_SUBTREE = f"Spider2-{ARCHIVE_BASENAME[:-4]}/spider2-lite/"

_ROWS = [
    {"instance_id": "local001", "db": "tiny", "question": "How many?"},
    {
        "instance_id": "bq001",
        "db": "warehouse",
        "question": "Revenue?",
        "external_knowledge": "notes.md",
    },
    # Only some rows carry `temporal`; Arrow cannot infer across that.
    {"instance_id": "sf_bq001", "db": "snow", "question": "When?", "temporal": "Yes"},
]


def _make_archives(tmp_path: Path) -> Path:
    staged = tmp_path / "spider2_lite"
    staged.mkdir()

    with zipfile.ZipFile(staged / ARCHIVE_BASENAME, "w") as archive:
        archive.writestr(
            _SUBTREE + "spider2-lite.jsonl",
            "\n".join(json.dumps(row) for row in _ROWS),
        )
        archive.writestr(
            _SUBTREE + "evaluation_suite/gold/spider2lite_eval.jsonl",
            json.dumps(
                {"instance_id": "local001", "condition_cols": [0], "ignore_order": True}
            ),
        )
        archive.writestr(
            _SUBTREE + "evaluation_suite/gold/exec_result/local001.csv", "x\n1\n"
        )
        archive.writestr(_SUBTREE + "resource/documents/notes.md", "some knowledge")
        archive.writestr(
            _SUBTREE + "resource/databases/bigquery/warehouse/s/DDL.csv",
            "table_name,ddl\nt,CREATE TABLE t (a INT64)\n",
        )
        # Outside the subtree: must not be extracted.
        archive.writestr(
            f"Spider2-{ARCHIVE_BASENAME[:-4]}/spider2-snow/README.md", "nope"
        )

    db_file = tmp_path / "tiny.sqlite"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE t (a int)")
    conn.commit()
    conn.close()
    with zipfile.ZipFile(staged / "sqlite.zip", "w") as archive:
        archive.write(db_file, "tiny.sqlite")
        # The AppleDouble stub that makes an unfiltered extraction wrong.
        archive.writestr("__MACOSX/._tiny.sqlite", b"\x00\x05\x16\x07stub")
    return staged


def test_both_sources_are_pinned_and_checksummed():
    meta = get_dataset_meta(Spider2LiteDataset)
    assert len(meta.source) == 2
    assert all(s.startswith("url:") for s in meta.source)
    checksums = dict(meta.checksums)
    assert checksums[ARCHIVE_BASENAME] == f"sha256:{SPIDER2_ARCHIVE_SHA256}"
    assert checksums["sqlite.zip"] == f"sha256:{LOCALDB_SHA256}"
    assert meta.license == "MIT"


@pytest.mark.parametrize(
    ("instance_id", "expected"),
    [
        ("local001", "sqlite"),
        ("bq001", "bigquery"),
        ("ga010", "bigquery"),
        # `sf_bq` must not be swallowed by the `bq` rule, nor `sf` by `sf_bq`.
        ("sf_bq189", "snowflake"),
        ("sf018", "snowflake"),
    ],
)
def test_backend_routing_is_by_longest_prefix(instance_id, expected):
    assert backend_for(instance_id) == expected


def test_unknown_prefix_is_rejected():
    with pytest.raises(ValueError, match="Unrecognised"):
        backend_for("mystery001")


def test_load_reads_every_row(tmp_path):
    dataset = Spider2LiteDataset(str(_make_archives(tmp_path)))
    assert len(dataset.dataset_dict["test"]) == 3


def test_temporal_is_normalised_across_rows(tmp_path):
    """37 of 547 real rows carry `temporal`; Arrow needs one schema."""
    dataset = Spider2LiteDataset(str(_make_archives(tmp_path)))
    rows = dataset.dataset_dict["test"]
    assert rows.column_names == [
        "instance_id",
        "db",
        "question",
        "external_knowledge",
        "temporal",
    ]
    assert [r["temporal"] for r in rows] == [None, None, "Yes"]
    assert [r["external_knowledge"] for r in rows] == [None, "notes.md", None]


def test_only_the_lite_subtree_is_extracted(tmp_path):
    """The archive holds spider2-snow and spider2-dbt too — 1 GB we do not need."""
    staged = _make_archives(tmp_path)
    Spider2LiteDataset(str(staged))
    assert not (staged / "spider2-lite" / "README.md").exists()
    assert not list(staged.glob("**/spider2-snow"))
    assert (staged / "spider2-lite" / "spider2-lite.jsonl").is_file()


def test_appledouble_stubs_are_filtered(tmp_path):
    """Unfiltered, sqlite.zip yields ~40 files ending .sqlite that are 163-byte
    AppleDouble stubs, not databases."""
    staged = _make_archives(tmp_path)
    dataset = Spider2LiteDataset(str(staged))
    localdb_dir = dataset.localdb_dir
    assert localdb_dir is not None
    names = sorted(Path(localdb_dir).iterdir())
    assert [p.name for p in names] == ["tiny.sqlite"]
    # And the survivor is a real database.
    conn = sqlite3.connect(names[0])
    assert conn.execute("SELECT count(*) FROM t").fetchone() == (0,)
    conn.close()


def test_staged_paths_resolve(tmp_path):
    dataset = Spider2LiteDataset(str(_make_archives(tmp_path)))
    for attribute in (
        "localdb_dir",
        "gold_dir",
        "eval_config_path",
        "documents_dir",
        "db_schema_dir",
    ):
        assert Path(getattr(dataset, attribute)).exists(), attribute


def test_paths_are_none_without_load():
    dataset = Spider2LiteDataset(_hf_dict=HFDatasetDict({}))
    assert dataset.localdb_dir is None
    assert dataset.gold_dir is None


def test_extraction_is_idempotent(tmp_path):
    staged = _make_archives(tmp_path)
    Spider2LiteDataset(str(staged))
    marker = staged / "spider2-lite" / ".sieval-extracted"
    stamp = marker.stat().st_mtime_ns
    Spider2LiteDataset(str(staged))
    assert marker.stat().st_mtime_ns == stamp


def test_missing_archive_names_the_download_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="sieval dataset download spider2_lite"):
        Spider2LiteDataset(str(tmp_path))
