"""Tests for Spider 2.0-lite's execution backends.

The local engine's guards are proved by deletion, as Spider 1.0's are. The
remote engines are covered only for the paths reachable without an account: a
missing credential must raise something the task can distinguish from a wrong
answer.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import sqlite3

import pandas as pd
import pytest

from sieval.tasks._spider2_backends import (
    BIGQUERY_CREDENTIAL_ENV,
    SNOWFLAKE_ENV,
    MissingCredentials,
    execute,
    open_readonly,
)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "tiny.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (a int, b text)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, "x"), (2, "y")])
    conn.commit()
    conn.close()
    return str(path)


# --- local engine: the guards -----------------------------------------------


def test_write_is_rejected(db):
    conn = open_readonly(db)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("DROP TABLE t")
    conn.close()


def test_attach_is_rejected(db, tmp_path):
    conn = open_readonly(db)
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        conn.execute(f"ATTACH DATABASE '{tmp_path / 'evil.db'}' AS evil")
    conn.close()


def test_runaway_query_is_aborted(db):
    runaway = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) "
        "SELECT count(*) FROM c"
    )
    with pytest.raises(sqlite3.OperationalError, match="interrupted"):
        execute("sqlite", runaway, db_path=db, deadline_s=0.5)


def test_row_cap_is_enforced(db):
    big = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c WHERE x<10000) "
        "SELECT x FROM c"
    )
    with pytest.raises(RuntimeError, match="exceeded 10 rows"):
        execute("sqlite", big, db_path=db, deadline_s=30.0, max_rows=10)


def test_non_utf8_text_is_readable(tmp_path):
    """Warehouse exports carry bytes the default text factory raises on."""
    path = tmp_path / "latin.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (name text)")
    conn.execute("INSERT INTO t VALUES (CAST(x'41FF42' AS TEXT))")
    conn.commit()
    conn.close()
    frame = execute("sqlite", "SELECT name FROM t", db_path=str(path))
    assert frame["name"][0].encode("utf-8", "surrogateescape") == b"\x41\xff\x42"


# --- local engine: results --------------------------------------------------


def test_result_is_a_frame_with_column_names(db):
    frame = execute("sqlite", "SELECT a, b FROM t ORDER BY a", db_path=db)
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["a", "b"]
    assert frame["a"].tolist() == [1, 2]


def test_empty_result_is_an_empty_frame_not_an_error(db):
    frame = execute("sqlite", "SELECT a FROM t WHERE a > 99", db_path=db)
    assert frame.empty
    assert list(frame.columns) == ["a"]


def test_sqlite_without_db_path_is_a_programming_error():
    with pytest.raises(ValueError, match="db_path"):
        execute("sqlite", "SELECT 1")


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        execute("oracle", "SELECT 1")


# --- remote engines: credentials --------------------------------------------


def test_bigquery_without_credentials_raises_missing_credentials(monkeypatch):
    """Distinguishable from a wrong answer, and it names the variable."""
    monkeypatch.delenv(BIGQUERY_CREDENTIAL_ENV, raising=False)
    with pytest.raises(MissingCredentials, match=BIGQUERY_CREDENTIAL_ENV):
        execute("bigquery", "SELECT 1")


def test_snowflake_without_credentials_names_every_missing_variable(monkeypatch):
    for name in SNOWFLAKE_ENV:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(MissingCredentials) as excinfo:
        execute("snowflake", "SELECT 1")
    for name in SNOWFLAKE_ENV:
        assert name in str(excinfo.value)


def test_snowflake_reports_only_the_variables_actually_missing(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct")
    monkeypatch.setenv("SNOWFLAKE_USER", "user")
    monkeypatch.delenv("SNOWFLAKE_PASSWORD", raising=False)
    with pytest.raises(MissingCredentials) as excinfo:
        execute("snowflake", "SELECT 1")
    assert "SNOWFLAKE_PASSWORD" in str(excinfo.value)
    assert "SNOWFLAKE_ACCOUNT" not in str(excinfo.value)


def test_missing_credentials_is_not_a_generic_error():
    """The task branches on this type to count it separately."""
    assert issubclass(MissingCredentials, RuntimeError)
