"""WikiSQL loader: archive reading, denormalisation, dtype and null handling.

Builds a miniature ``data.tar.bz2`` with upstream's member layout rather than
touching the real 26 MB download, so the shape contract is pinned offline.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import io
import json
import tarfile

import pytest

from sieval.core.datasets.meta import get_dataset_meta
from sieval.datasets.wikisql import (
    WIKISQL_COMMIT,
    WIKISQL_URL,
    WikiSQLDataset,
)

_TABLE = {
    "id": "1-234-5",
    "header": ["Player", "Points"],
    "types": ["text", "real"],
    "rows": [["Terrence Ross", 12], ["Chris Bosh", 22]],
    "page_title": "Roster",
    "section_title": "R",
    "caption": "R",
    "name": "table_1_234_5",
    "page_id": 99,
}

#: Upstream omits these keys entirely on a measured fraction of tables rather
#: than nulling them, so the loader must read them with `.get()`.
_SPARSE_TABLE = {
    "id": "9-999-9",
    "header": ["A"],
    "types": ["text"],
    "rows": [["x"]],
}

_QUESTION = {
    "phase": 1,
    "table_id": "1-234-5",
    "question": "How many points did Terrence Ross score?",
    "sql": {"sel": 1, "agg": 0, "conds": [[0, 0, "Terrence Ross"]]},
}

_SPARSE_QUESTION = {
    "phase": 2,
    "table_id": "9-999-9",
    "question": "What is A?",
    "sql": {"sel": 0, "agg": 0, "conds": []},
}


def _jsonl(rows) -> bytes:
    return "".join(json.dumps(r) + "\n" for r in rows).encode()


def _archive(tmp_path, members: dict[str, bytes]):
    path = tmp_path / "data.tar.bz2"
    with tarfile.open(path, "w:bz2") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return path


def _full_members() -> dict[str, bytes]:
    return {
        "data/test.jsonl": _jsonl([_QUESTION, _SPARSE_QUESTION]),
        "data/test.tables.jsonl": _jsonl([_TABLE, _SPARSE_TABLE]),
        "data/dev.jsonl": _jsonl([_QUESTION]),
        "data/dev.tables.jsonl": _jsonl([_TABLE]),
    }


def test_source_is_commit_pinned_with_a_checksum():
    """A branch URL would let the bytes move under the recorded checksum."""
    meta = get_dataset_meta(WikiSQLDataset)
    assert WIKISQL_COMMIT in WIKISQL_URL
    assert "master" not in WIKISQL_URL
    assert meta.source == (f"url:{WIKISQL_URL}",)
    assert dict(meta.checksums)["data.tar.bz2"].startswith("sha256:")


def test_loads_both_published_splits(tmp_path):
    path = _archive(tmp_path, _full_members())
    dd = WikiSQLDataset(str(path)).dataset_dict
    # Upstream's `dev` is exposed as `validation`; `train` is not materialised.
    assert set(dd) == {"test", "validation"}
    assert len(dd["test"]) == 2
    assert len(dd["validation"]) == 1


def test_accepts_a_directory_as_well_as_the_archive(tmp_path):
    """Works from `sieval dataset download` and from a hand-placed copy."""
    _archive(tmp_path, _full_members())
    dd = WikiSQLDataset(str(tmp_path)).dataset_dict
    assert len(dd["test"]) == 2


def test_table_is_denormalised_onto_the_question_row(tmp_path):
    path = _archive(tmp_path, _full_members())
    row = WikiSQLDataset(str(path)).dataset_dict["test"][0]
    assert row["question"] == _QUESTION["question"]
    assert row["table_id"] == "1-234-5"
    assert row["header"] == ["Player", "Points"]
    assert row["types"] == ["text", "real"]
    assert row["page_title"] == "Roster"
    assert row["page_id"] == 99


def test_sql_and_rows_round_trip_through_json(tmp_path):
    """Both are JSON strings because their dtypes cannot be typed Arrow columns.

    `conds` values are str/int/float across rows and table cells are str/int
    even within one `real` column, so the serialisation is forced, not chosen —
    and it must round-trip exactly, since the scorer parses it back.
    """
    path = _archive(tmp_path, _full_members())
    row = WikiSQLDataset(str(path)).dataset_dict["test"][0]
    assert json.loads(row["sql_json"]) == _QUESTION["sql"]
    assert json.loads(row["rows_json"]) == _TABLE["rows"]
    # The mixed dtype survives: a str cell and an int cell in the same table.
    cells = json.loads(row["rows_json"])[0]
    assert isinstance(cells[0], str)
    assert isinstance(cells[1], int)


def test_absent_table_metadata_becomes_none_not_empty_string(tmp_path):
    """329 of 5,230 test tables omit page_title/section_title/caption."""
    path = _archive(tmp_path, _full_members())
    row = WikiSQLDataset(str(path)).dataset_dict["test"][1]
    assert row["table_id"] == "9-999-9"
    for key in ("page_title", "section_title", "caption", "name", "page_id"):
        assert row[key] is None, key


def test_missing_archive_member_is_reported_with_what_was_found(tmp_path):
    members = _full_members()
    del members["data/test.tables.jsonl"]
    path = _archive(tmp_path, members)
    with pytest.raises(ValueError, match="test.tables.jsonl"):
        WikiSQLDataset(str(path))


def test_empty_split_is_rejected(tmp_path):
    """A silently empty split is how a schema change or bad download shows up."""
    members = _full_members()
    members["data/test.jsonl"] = b""
    path = _archive(tmp_path, members)
    with pytest.raises(ValueError, match="empty 'test' split"):
        WikiSQLDataset(str(path))


def test_question_referencing_an_unknown_table_is_not_silently_dropped(tmp_path):
    """Every question must find its table, or the row count would lie."""
    members = _full_members()
    members["data/test.tables.jsonl"] = _jsonl([_SPARSE_TABLE])
    path = _archive(tmp_path, members)
    with pytest.raises(KeyError):
        WikiSQLDataset(str(path))
