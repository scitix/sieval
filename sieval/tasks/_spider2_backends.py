"""Query execution for Spider 2.0-lite's three engines.

One entry point, :func:`execute`, routes by engine and returns a pandas frame.
Everything above it — prompt, comparison, reporting — is engine-agnostic.

**SQLite is hardened, for the same reason Spider 1.0's is.** Upstream copies the
whole database into memory (``conn.backup``) and runs the prediction against the
copy, which stops writes reaching disk but does nothing about an unbounded
query, and costs a full copy of a database that can run to hundreds of megabytes
— per call, while sieval grades concurrently. A read-only connection buys the
same write protection without the copy, and the deadline is what upstream has no
answer for at all. Guards match ``_spider_exec``: ``mode=ro&immutable=1``,
``ATTACH``/``DETACH`` denied, a progress-handler deadline that aborts inside
SQLite, and a row cap.

**BigQuery and Snowflake are not hardened, because there is nothing here to
harden.** The query runs on someone else's server under someone else's
permissions; what this module can control is the *bound*, so both carry a
timeout and a row cap, and BigQuery additionally runs read-only by construction
(the client is only ever asked to run a query job).

**Snowflake is first-party, and that is a divergence worth knowing about.**
Upstream's current lite evaluator routes ``bq``/``ga`` and ``local`` and sends
everything else to "Unsupported instance id prefix" — 207 of the 547 instances,
even though gold results ship for all 547. An older ``evaluate_utils.py`` in the
same repo still has ``get_snowflake_sql_result``, but it is the stale module
whose comparison logic upstream has already superseded, so this backend is
written here rather than vendored from it. A Snowflake number therefore has no
upstream *lite* counterpart; the comparable published setting is Spider 2.0-Snow,
which asks the same questions against the same warehouse.

**Credentials are read from the environment and their absence is loud.** A
missing credential raises :class:`MissingCredentials` naming the variable, which
becomes a per-sample error rather than a silent zero — the alternative is a run
that reports 24% and looks like a bad model instead of an unconfigured host.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import os
import sqlite3
import time
from pathlib import Path

import pandas as pd

#: Per-query wall-clock budget for the local engine.
DEFAULT_DEADLINE_S = 60.0
#: Remote engines are slower and billed; upstream uses 90s for BigQuery.
DEFAULT_REMOTE_TIMEOUT_S = 90.0
#: Result rows kept. Spider 2.0 answers are aggregates, so this is far above
#: any gold; it exists so a runaway prediction cannot exhaust memory.
DEFAULT_MAX_ROWS = 100_000
_PROGRESS_INTERVAL = 1_000

#: Environment variables each remote engine needs.
BIGQUERY_CREDENTIAL_ENV = "GOOGLE_APPLICATION_CREDENTIALS"
SNOWFLAKE_ENV = ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD")

_DENIED = frozenset({sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH})


class MissingCredentials(RuntimeError):
    """A remote engine was asked for without the credentials to reach it."""


def _authorizer(action: int, *_args) -> int:
    return sqlite3.SQLITE_DENY if action in _DENIED else sqlite3.SQLITE_OK


def open_readonly(db_path: str) -> sqlite3.Connection:
    """Open *db_path* read-only and immutable, with ATTACH/DETACH denied."""
    uri = f"{Path(db_path).absolute().as_uri()}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.set_authorizer(_authorizer)
    # Warehouse exports routinely carry bytes that are not valid UTF-8; the
    # default text factory would raise on them mid-fetch. Round-trip-lossless,
    # and applied to predictions and gold alike.
    conn.text_factory = lambda raw: raw.decode("utf-8", "surrogateescape")
    return conn


def _run_sqlite(
    db_path: str, sql: str, deadline_s: float, max_rows: int
) -> pd.DataFrame:
    conn = open_readonly(db_path)
    end = time.monotonic() + deadline_s
    conn.set_progress_handler(
        lambda: 1 if time.monotonic() > end else 0, _PROGRESS_INTERVAL
    )
    try:
        cursor = conn.execute(sql)
        columns = [description[0] for description in cursor.description or []]
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            raise RuntimeError(f"result exceeded {max_rows} rows")
        return pd.DataFrame(rows, columns=pd.Index(columns))
    finally:
        conn.set_progress_handler(None, 0)
        conn.close()


def _run_bigquery(sql: str, timeout_s: float, max_rows: int) -> pd.DataFrame:
    credential = os.getenv(BIGQUERY_CREDENTIAL_ENV)
    if not credential:
        raise MissingCredentials(
            f"BigQuery instance needs {BIGQUERY_CREDENTIAL_ENV} pointing at a "
            "service-account JSON key. See upstream's Bigquery_Guideline."
        )
    from google.cloud import bigquery

    client = bigquery.Client.from_service_account_json(credential)
    job = client.query(sql)
    rows = job.result(timeout=timeout_s, max_results=max_rows + 1)
    frame = rows.to_dataframe()
    if len(frame) > max_rows:
        raise RuntimeError(f"result exceeded {max_rows} rows")
    return frame


def _run_snowflake(sql: str, timeout_s: float, max_rows: int) -> pd.DataFrame:
    missing = [name for name in SNOWFLAKE_ENV if not os.getenv(name)]
    if missing:
        raise MissingCredentials(
            f"Snowflake instance needs {', '.join(missing)}. Access is granted "
            "by upstream's 'Spider2 Snowflake Access' form; note upstream "
            "recorded an evaluation-account suspension on 2026-08-12."
        )
    import snowflake.connector

    conn = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        login_timeout=timeout_s,
        network_timeout=timeout_s,
    )
    try:
        cursor = conn.cursor()
        try:
            cursor.execute(sql, timeout=int(timeout_s))
            columns = [description[0] for description in cursor.description or []]
            rows = cursor.fetchmany(max_rows + 1)
            if len(rows) > max_rows:
                raise RuntimeError(f"result exceeded {max_rows} rows")
            return pd.DataFrame(rows, columns=pd.Index(columns))
        finally:
            cursor.close()
    finally:
        conn.close()


def execute(
    backend: str,
    sql: str,
    *,
    db_path: str | None = None,
    deadline_s: float = DEFAULT_DEADLINE_S,
    remote_timeout_s: float = DEFAULT_REMOTE_TIMEOUT_S,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> pd.DataFrame:
    """Run *sql* on *backend* and return the result frame.

    Raises :class:`MissingCredentials` when a remote engine is unconfigured, and
    lets any engine-level error propagate — the caller turns both into a scored
    zero with the reason recorded, which is what separates "wrong answer" from
    "could not ask".
    """
    if backend == "sqlite":
        if not db_path:
            raise ValueError("sqlite backend needs db_path")
        return _run_sqlite(db_path, sql, deadline_s, max_rows)
    if backend == "bigquery":
        return _run_bigquery(sql, remote_timeout_s, max_rows)
    if backend == "snowflake":
        return _run_snowflake(sql, remote_timeout_s, max_rows)
    raise ValueError(f"Unknown Spider 2.0-lite backend: {backend!r}")
