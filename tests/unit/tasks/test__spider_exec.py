"""Safety and correctness tests for the Spider 1.0 grading worker.

Each guard is proved by deletion: the assertion must fail if the guard is
removed, or it has tested nothing.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import json
import sqlite3

import pytest

from sieval.community.spider import get_schema
from sieval.tasks._spider_exec import (
    _run_bounded,
    grade_one,
    open_readonly,
    read_schema_readonly,
)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "concert_singer" / "concert_singer.sqlite"
    path.parent.mkdir()
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE singer (id int, name text)")
    conn.executemany("INSERT INTO singer VALUES (?, ?)", [(1, "Joe"), (2, "Ann")])
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def tables_json(tmp_path):
    path = tmp_path / "tables.json"
    path.write_text(
        json.dumps(
            [
                {
                    "db_id": "concert_singer",
                    "table_names_original": ["singer"],
                    "table_names": ["singer"],
                    "column_names_original": [[-1, "*"], [0, "id"], [0, "name"]],
                    "column_names": [[-1, "*"], [0, "id"], [0, "name"]],
                    "column_types": ["text", "number", "text"],
                    "foreign_keys": [],
                    "primary_keys": [1],
                }
            ]
        )
    )
    return path


# --- guard 1: read-only -----------------------------------------------------


def test_write_is_rejected_by_the_connection(db):
    """A model can emit DDL/DML. Read-only must stop it at the driver."""
    conn = open_readonly(str(db))
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("DROP TABLE singer")
    conn.close()


# --- guard 2: no ATTACH -----------------------------------------------------


def test_attach_is_rejected(db, tmp_path):
    """`mode=ro` alone does not stop ATTACH — that is the escape hatch."""
    conn = open_readonly(str(db))
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        conn.execute(f"ATTACH DATABASE '{tmp_path / 'evil.db'}' AS evil")
    conn.close()


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO singer VALUES (3, 'Eve')",
        "UPDATE singer SET name = 'x'",
        "DELETE FROM singer",
        "CREATE TABLE evil (a int)",
    ],
)
def test_every_write_vector_is_rejected(db, statement):
    """Not just DROP — DML and DDL alike must die at the driver."""
    conn = open_readonly(str(db))
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute(statement)
    conn.close()


def test_reads_still_work(db):
    """The guards must not have made the database unusable."""
    conn = open_readonly(str(db))
    assert conn.execute("SELECT count(*) FROM singer").fetchall() == [(2,)]
    conn.close()


def test_non_utf8_text_is_readable(tmp_path):
    """Spider's `wta_1.players.last_name` holds bytes that are not valid UTF-8.

    sqlite3's default text factory raises on them, which is why upstream dies on
    two dev examples rather than scoring them. Fails without the surrogateescape
    factory in `open_readonly`.
    """
    path = tmp_path / "latin.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (name text)")
    # x'41FF42' is 'A', an invalid continuation byte, then 'B'.
    conn.execute("INSERT INTO t VALUES (CAST(x'41FF42' AS TEXT))")
    conn.commit()
    conn.close()

    reader = open_readonly(str(path))
    try:
        (value,) = reader.execute("SELECT name FROM t").fetchone()
    finally:
        reader.close()
    # Round-trips losslessly, so two different bad byte sequences stay different.
    assert value.encode("utf-8", "surrogateescape") == b"\x41\xff\x42"


def test_default_bounds_do_not_bind_on_a_realistic_result():
    """The cap must sit above real gold results, not truncate them.

    Measured on the pinned dev set: largest gold is 20,662 rows, slowest is
    0.486 s. This pins the constants so a later 'tightening' has to argue with
    a failing test rather than silently rescoring the benchmark.
    """
    from sieval.tasks._spider_exec import DEFAULT_DEADLINE_S, DEFAULT_MAX_ROWS

    assert DEFAULT_MAX_ROWS > 20_662
    assert DEFAULT_DEADLINE_S > 0.486


# --- guard 3: deadline and row cap ------------------------------------------


def test_runaway_query_is_aborted_by_the_deadline(db, tables_json):
    """An unbounded recursive CTE must abort inside the worker, not hang it."""
    runaway = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) "
        "SELECT count(*) FROM c"
    )
    out = grade_one(
        str(db),
        str(tables_json),
        "concert_singer",
        runaway,
        "SELECT count(*) FROM singer",
        deadline_s=0.5,
    )
    assert out["execution"] is False
    assert out["error"] is not None
    assert "interrupted" in out["error"]


def test_row_cap_bounds_a_large_result(db):
    """Tested against the executor directly.

    It cannot go through `grade_one`: a result big enough to trip the cap needs
    a recursive CTE, and upstream's parser rejects those, so such a query can
    never be a Spider gold.
    """
    big = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c WHERE x<100000) "
        "SELECT x FROM c"
    )
    conn = open_readonly(str(db))
    try:
        with pytest.raises(RuntimeError, match="exceeded 10 rows"):
            _run_bounded(conn, big, deadline_s=10.0, max_rows=10)
    finally:
        conn.close()


# --- schema reading equivalence ---------------------------------------------


def test_read_schema_readonly_matches_upstream_exactly(db):
    """Ours must equal upstream's dict, or parsing behaviour has changed.

    Upstream's `get_schema` opens read-write. It only runs fixed introspection,
    so it cannot be driven to write — but it can still drop -wal/-shm sidecars
    into a shared data directory. We reproduce it read-only; this test is what
    makes that substitution safe rather than merely plausible.
    """
    assert read_schema_readonly(str(db)) == get_schema(str(db))


# --- grading correctness ----------------------------------------------------


def test_identical_query_scores_both_metrics(db, tables_json):
    out = grade_one(
        str(db),
        str(tables_json),
        "concert_singer",
        "SELECT count(*) FROM singer",
        "SELECT count(*) FROM singer",
    )
    assert out["exact_match"] is True
    assert out["execution"] is True
    assert out["error"] is None


def test_wrong_query_scores_neither(db, tables_json):
    out = grade_one(
        str(db),
        str(tables_json),
        "concert_singer",
        "SELECT name FROM singer",
        "SELECT count(*) FROM singer",
    )
    assert out["exact_match"] is False
    assert out["execution"] is False


def test_unparseable_prediction_scores_zero_rather_than_raising(db, tables_json):
    """Upstream substitutes an empty parse and still scores it. Mirror that."""
    out = grade_one(
        str(db),
        str(tables_json),
        "concert_singer",
        "not sql at all",
        "SELECT count(*) FROM singer",
    )
    assert out["exact_match"] is False
    assert out["execution"] is False


def test_empty_prediction_is_scored_not_skipped(db, tables_json):
    out = grade_one(
        str(db), str(tables_json), "concert_singer", "", "SELECT count(*) FROM singer"
    )
    assert out["exact_match"] is False
    assert out["execution"] is False


def test_hardness_is_reported(db, tables_json):
    out = grade_one(
        str(db),
        str(tables_json),
        "concert_singer",
        "SELECT count(*) FROM singer",
        "SELECT count(*) FROM singer",
    )
    assert out["hardness"] in {"easy", "medium", "hard", "extra"}


def test_broken_gold_raises_rather_than_scoring_zero(db, tables_json):
    """A gold that will not parse is our bug, not the model's failure."""
    with pytest.raises(ValueError, match="gold SQL failed to parse"):
        grade_one(
            str(db), str(tables_json), "concert_singer", "SELECT 1", "!!! not sql !!!"
        )
