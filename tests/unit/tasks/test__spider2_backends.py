"""Tests for Spider 2.0-lite's execution backends.

The guards themselves are proved by deletion once, in
`tests/unit/tasks/test__sqlite_exec.py`, where the shared module lives. What
this file owns is that the local engine is **wired** to them — every guard is
exercised through `execute`, which is the only entry point the task uses, so
re-implementing the connection here would fail these rather than pass them.

The remote engines are covered only for the paths reachable without an account:
a missing credential must be nameable *before* a query is attempted, and the
caller's budget must outlive the bound the engine enforces.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import sqlite3

import pandas as pd
import pytest

from sieval.core.utils.offload import GRADE_TIMEOUT
from sieval.tasks._spider2_backends import (
    BIGQUERY_CREDENTIAL_ENV,
    DEFAULT_DEADLINE_S,
    DEFAULT_REMOTE_TIMEOUT_S,
    SNOWFLAKE_ENV,
    MissingCredentials,
    caller_timeout,
    execute,
    missing_credentials,
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


# --- local engine: the guards are reached through `execute` ------------------


def test_write_is_rejected(db):
    """Asked through `execute`, so this fails if the engine stops using the
    shared connection — which is the only thing this file can add to the
    deletion tests in `test__sqlite_exec.py`."""
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        execute("sqlite", "DROP TABLE t", db_path=db)


def test_attach_is_rejected(db, tmp_path):
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        execute(
            "sqlite", f"ATTACH DATABASE '{tmp_path / 'evil.db'}' AS evil", db_path=db
        )


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


@pytest.fixture
def no_bigquery_credentials(monkeypatch):
    """No environment key *and* no application default credentials.

    Both halves are needed since `missing_credentials` learned to accept ADC:
    deleting the variable alone leaves the answer dependent on whether the host
    running the suite happens to have `gcloud auth application-default login`
    in its state — green in CI, red on a developer box that uses GCP.
    """
    monkeypatch.delenv(BIGQUERY_CREDENTIAL_ENV, raising=False)
    try:
        import google.auth
    except ImportError:
        # The `spider2` group is absent, so the probe's own ImportError branch
        # already reports "unreachable" — nothing to patch.
        return

    def _nothing_configured(*_args, **_kwargs):
        raise RuntimeError("no application default credentials")

    monkeypatch.setattr(google.auth, "default", _nothing_configured)


@pytest.mark.usefixtures("no_bigquery_credentials")
def test_bigquery_without_credentials_raises_missing_credentials():
    """Distinguishable from a wrong answer, and it names the variable."""
    with pytest.raises(MissingCredentials, match=BIGQUERY_CREDENTIAL_ENV):
        execute("bigquery", "SELECT 1")


def test_application_default_credentials_count_as_configured(monkeypatch):
    """`gcloud auth application-default login` is a configured host.

    Upstream builds a bare `bigquery.Client()`, so Google's whole default chain
    is in scope; treating only the environment variable as configured would
    report all 205 BigQuery questions as `missing_credentials` on a host that
    can reach BigQuery perfectly well.
    """
    google_auth = pytest.importorskip(
        "google.auth", reason="requires the `spider2` extra"
    )
    monkeypatch.delenv(BIGQUERY_CREDENTIAL_ENV, raising=False)
    monkeypatch.setattr(google_auth, "default", lambda *a, **k: (object(), "a-project"))
    assert missing_credentials("bigquery") is None


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
    """A direct caller can still tell it apart from an engine error.

    The *task* does not rely on this — it asks `missing_credentials` before it
    offloads anything, because an exception crossing a process boundary can only
    be classified by catching broadly at the grading call site. This is the
    backstop for a credential that is present and rejected, and for a caller
    using `execute` in-process.
    """
    assert issubclass(MissingCredentials, RuntimeError)


# --- credentials, asked before anything runs --------------------------------


@pytest.mark.usefixtures("no_bigquery_credentials")
def test_missing_credentials_names_the_variable_without_executing():
    """The question the task actually asks. No client, no query, no network."""
    reason = missing_credentials("bigquery")
    assert reason is not None
    assert BIGQUERY_CREDENTIAL_ENV in reason


def test_snowflake_reason_lists_only_what_is_absent(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct")
    monkeypatch.setenv("SNOWFLAKE_USER", "user")
    monkeypatch.delenv("SNOWFLAKE_PASSWORD", raising=False)
    reason = missing_credentials("snowflake")
    assert reason is not None
    assert "SNOWFLAKE_PASSWORD" in reason
    assert "SNOWFLAKE_ACCOUNT" not in reason


def test_a_configured_remote_engine_has_no_reason(monkeypatch):
    """Otherwise every cloud sample would be reported unreachable on a host
    that is in fact configured — the failure this function must not have."""
    monkeypatch.setenv(BIGQUERY_CREDENTIAL_ENV, "/tmp/key.json")
    for name in SNOWFLAKE_ENV:
        monkeypatch.setenv(name, "set")
    assert missing_credentials("bigquery") is None
    assert missing_credentials("snowflake") is None


def test_sqlite_needs_no_credentials():
    """The 135 local questions are the subset a credential-less run scores."""
    assert missing_credentials("sqlite") is None


# --- the caller's budget must outlive the engine's bound ---------------------


@pytest.mark.parametrize(
    ("backend", "engine_bound"),
    [
        ("sqlite", DEFAULT_DEADLINE_S),
        ("bigquery", DEFAULT_REMOTE_TIMEOUT_S),
        ("snowflake", DEFAULT_REMOTE_TIMEOUT_S),
    ],
)
def test_caller_timeout_is_longer_than_the_bound_it_covers(backend, engine_bound):
    """A process pool cannot interrupt a running call.

    Give the caller less than the engine's own deadline and that deadline never
    fires: the caller gives up first, the query keeps running in a worker
    nothing can reclaim, and the sample is reported as a timeout rather than as
    a bad query. Fails if either number is edited without the other.
    """
    assert caller_timeout(backend) > engine_bound


def test_the_shared_default_would_invert_every_bound():
    """Why `caller_timeout` exists at all, stated as a number.

    `run_cpu_bound`'s own default is 30 s, which is *below* this benchmark's
    60 s local deadline and 90 s remote one — so a grading call site that simply
    omitted `timeout=` would make all three decorative. This is the inequality
    that makes passing it load-bearing rather than tidy.
    """
    assert GRADE_TIMEOUT < DEFAULT_DEADLINE_S
    assert GRADE_TIMEOUT < DEFAULT_REMOTE_TIMEOUT_S
    for backend in ("sqlite", "bigquery", "snowflake"):
        assert caller_timeout(backend) > GRADE_TIMEOUT
