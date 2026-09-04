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
    ALL_ENGINES,
    ARCHIVE_BASENAME,
    LOCALDB_BASENAME,
    LOCALDB_SHA256,
    LOCALDB_URL,
    SPIDER2_ARCHIVE_SHA256,
    Spider2LiteDataset,
    backend_for,
    normalise_engines,
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
    with zipfile.ZipFile(staged / LOCALDB_BASENAME, "w") as archive:
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
    assert checksums[LOCALDB_BASENAME] == f"sha256:{LOCALDB_SHA256}"
    assert meta.license == "MIT"


def test_local_databases_come_from_upstreams_own_archive():
    """Not the Hub's ``sqlite.zip``, which is a different corpus.

    It ships 40 databases covering only 23 of the 30 the 135 ``local``
    questions name, so seven databases and 48 questions have no file to open.
    Pinning is what makes that choice legible, so assert the source rather than
    leaving a comment: the Drive id upstream's README links, carried with
    ``confirm=t`` because Drive interstitials anything this large.
    """
    meta = get_dataset_meta(Spider2LiteDataset)
    localdb = next(s for s in meta.source if "github.com" not in s)
    assert localdb == f"url:{LOCALDB_URL}"
    assert "1coEVsCZq-Xvj9p2TnhBFoFTsY-UoYGmG" in localdb
    assert "confirm=t" in localdb
    assert "spider2-localdb" not in localdb


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
    dataset = Spider2LiteDataset(str(_make_archives(tmp_path)), engines=ALL_ENGINES)
    assert len(dataset.dataset_dict["test"]) == 3


# --- engine selection -------------------------------------------------------


def test_the_default_selection_is_sqlite_only(tmp_path):
    """The subset that runs with no credentials, and the reason is spend.

    `missing_credentials` cannot be asked before a sample exists, so a cloud
    question is inferred — at the largest prompt sizes in the benchmark — and
    only then scored unreachable.
    """
    dataset = Spider2LiteDataset(str(_make_archives(tmp_path)))
    rows = list(dataset.dataset_dict["test"])
    assert [r["instance_id"] for r in rows] == ["local001"]
    assert dataset.engines == ("sqlite",)


def test_engines_selects_and_is_reported_in_all_engines_order(tmp_path):
    dataset = Spider2LiteDataset(
        str(_make_archives(tmp_path)), engines=["snowflake", "sqlite"]
    )
    # Requested out of order; reported and applied in ALL_ENGINES order so two
    # configs naming the same engines produce the same sample ids.
    assert dataset.engines == ("sqlite", "snowflake")
    assert [r["instance_id"] for r in dataset.dataset_dict["test"]] == [
        "local001",
        "sf_bq001",
    ]


def test_a_single_engine_may_be_named_as_a_bare_string(tmp_path):
    dataset = Spider2LiteDataset(str(_make_archives(tmp_path)), engines="bigquery")
    assert dataset.engines == ("bigquery",)
    assert [r["instance_id"] for r in dataset.dataset_dict["test"]] == ["bq001"]


@pytest.mark.parametrize("bad", ["postgres", ["sqlite", "duckdb"], []])
def test_an_unusable_engine_selection_is_a_loud_stop(bad):
    """Not a quietly empty split, which scores zero questions and looks fine."""
    with pytest.raises(ValueError, match="engine"):
        normalise_engines(bad)


def test_a_bigquery_only_run_is_not_stopped_by_an_absent_local_database(tmp_path):
    """The completeness guard covers the engines asked for, and no others."""
    staged = _make_archives(tmp_path)
    Spider2LiteDataset(str(staged))
    Path(
        staged, "spider2-lite", "resource/databases/spider2-localdb/tiny.sqlite"
    ).unlink()
    dataset = Spider2LiteDataset(str(staged), engines="bigquery")
    assert [r["instance_id"] for r in dataset.dataset_dict["test"]] == ["bq001"]


def test_temporal_is_normalised_across_rows(tmp_path):
    """37 of 547 real rows carry `temporal`; Arrow needs one schema."""
    dataset = Spider2LiteDataset(str(_make_archives(tmp_path)), engines=ALL_ENGINES)
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


def test_a_local_question_without_its_database_fails_at_load(tmp_path):
    """The defect the Hub archive shipped: 48 questions, no file to open.

    Left to the task, each one raises ``unable to open database file`` from
    inside ``preprocess`` — naming no database, once per affected sample. The
    loader knows the whole set, so it says which are missing, once.
    """
    staged = _make_archives(tmp_path)
    # Stage everything, then remove the one local database the rows name.
    Spider2LiteDataset(str(staged))
    (
        Path(staged, "spider2-lite", "resource/databases/spider2-localdb/tiny.sqlite")
    ).unlink()
    with pytest.raises(ValueError, match=r"missing 1 of 1 local databases"):
        Spider2LiteDataset(str(staged))


def test_load_accepts_a_complete_local_corpus(tmp_path):
    """The guard must not fire on a staging that is actually complete."""
    staged = _make_archives(tmp_path)
    dataset = Spider2LiteDataset(str(staged))
    # The default selection is SQLite only, so this is the one local row.
    assert [r["instance_id"] for r in dataset.dataset_dict["test"]] == ["local001"]


def test_appledouble_stubs_are_filtered(tmp_path):
    """The pinned archive carries ``__MACOSX/._chinook.sqlite``, which ends in
    ``.sqlite``, is a few hundred bytes, and is not a database."""
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
