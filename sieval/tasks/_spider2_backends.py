"""Query execution for Spider 2.0-lite's three engines.

One entry point, :func:`execute`, routes by engine and returns a pandas frame.
Everything above it — prompt, comparison, reporting — is engine-agnostic.

**SQLite is hardened, for the same reason Spider 1.0's is.** Upstream copies the
whole database into memory (``conn.backup``) and runs the prediction against the
copy, which stops writes reaching disk but does nothing about an unbounded
query, and costs a full copy of a database that can run to hundreds of megabytes
— per call, while sieval grades concurrently. A read-only connection buys the
same write protection without the copy, and the deadline is what upstream has no
answer for at all. The guards are not re-implemented here: they are
``sieval.tasks._sqlite_exec``, shared with Spider 1.0's graders and prompt
builder, so the two benchmarks cannot drift on what "hardened" means. Only the
*bounds* are this benchmark's own.

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

**Credentials are discovered the way each vendor's client does, and their
absence is loud.** BigQuery accepts anything Google's default chain accepts —
``GOOGLE_APPLICATION_CREDENTIALS``, or ``gcloud auth application-default
login``, or workload identity — because upstream builds a bare
``bigquery.Client()`` and narrowing that to one environment variable would
report a working host as unconfigured. Snowflake reads its three variables.
Either way a missing credential raises :class:`MissingCredentials` naming what
is absent, which becomes a per-sample error rather than a silent zero — the
alternative is a run that reports 24% and looks like a bad model instead of an
unconfigured host.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import os
from typing import TYPE_CHECKING

from ._sqlite_exec import open_readonly, run_bounded

if TYPE_CHECKING:
    # pandas is behind the `spider2` group. Importing it here would put it at
    # module scope for anything that merely *registers* the task, so the runtime
    # imports are inside the two functions that build a frame and every
    # annotation naming it is a string.
    import pandas as pd

#: Per-query wall-clock budget for the local engine, enforced inside SQLite.
#: **Not measured** — unlike Spider 1.0's, whose 5 s is checked against every
#: gold, this one cannot be until someone runs the task on staged data. It is
#: set from what the questions look like: enterprise schemas, multi-CTE
#: aggregates, and a local corpus whose largest database is 372 MB.
DEFAULT_DEADLINE_S = 60.0
#: Remote engines are slower and billed; upstream uses 90s for BigQuery.
DEFAULT_REMOTE_TIMEOUT_S = 90.0
#: Result rows kept. Spider 2.0 answers are aggregates, so this is far above
#: any gold; it exists so a runaway prediction cannot exhaust memory.
DEFAULT_MAX_ROWS = 100_000

#: Slack between an engine's own bound and the caller's, covering what happens
#: after the query returns: handing a frame back across the process boundary,
#: then loading gold and comparing.
_CALLER_HEADROOM_S = 30.0

#: Environment variables each remote engine needs.
BIGQUERY_CREDENTIAL_ENV = "GOOGLE_APPLICATION_CREDENTIALS"
SNOWFLAKE_ENV = ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD")


class MissingCredentials(RuntimeError):
    """A remote engine was asked for without the credentials to reach it."""


def caller_timeout(backend: str) -> float:
    """The ``run_cpu_bound`` budget a caller must allow *backend*.

    Derived here rather than left to the caller because the two numbers are one
    decision: a bound the engine enforces is only real if the caller waits
    longer than it. Give the caller less — ``GRADE_TIMEOUT``'s 30 s against this
    module's 60 s and 90 s, say — and every engine bound becomes decorative,
    since the caller gives up first and the query keeps running in a worker the
    pool cannot interrupt. The deadline that was supposed to stop it never
    fires, and the failure reads as a timeout rather than as a bad query.
    """
    engine = DEFAULT_DEADLINE_S if backend == "sqlite" else DEFAULT_REMOTE_TIMEOUT_S
    return engine + _CALLER_HEADROOM_S


def _bigquery_reachable() -> bool:
    """Whether *any* credential this host offers would satisfy BigQuery.

    ``GOOGLE_APPLICATION_CREDENTIALS`` is checked first because it is a plain
    environment read and costs nothing. Falling back to ``google.auth.default``
    is what makes ``gcloud auth application-default login`` — and workload
    identity, and an external-account JSON — count: upstream builds a bare
    ``bigquery.Client()``, so the whole default chain is in scope there, and
    treating only the environment variable as "configured" would report 205
    questions as ``missing_credentials`` on a host that can in fact reach
    BigQuery. The import is inside the branch, so a run over the local engine
    never pays for it.
    """
    if os.getenv(BIGQUERY_CREDENTIAL_ENV):
        return True
    try:
        import google.auth

        google.auth.default()
    except Exception:
        # `DefaultCredentialsError` when nothing is configured, `ImportError`
        # when the `spider2` group is absent. Either way there is no credential
        # to reach BigQuery with, which is what the caller asked.
        return False
    return True


def missing_credentials(backend: str) -> str | None:
    """Why *backend* cannot be reached from this environment, or ``None``.

    Asked before any query is attempted, so an unconfigured host is separated
    from a bad prediction *without* an exception having to travel back through
    a worker process — see the note on ``execute`` below.
    """
    if backend == "bigquery" and not _bigquery_reachable():
        return (
            f"BigQuery instance needs credentials: either "
            f"{BIGQUERY_CREDENTIAL_ENV} pointing at a service-account JSON key, "
            "or application default credentials from 'gcloud auth "
            "application-default login'. See upstream's Bigquery_Guideline."
        )
    if backend == "snowflake":
        missing = [name for name in SNOWFLAKE_ENV if not os.getenv(name)]
        if missing:
            return (
                f"Snowflake instance needs {', '.join(missing)}. Access is "
                "granted by upstream's 'Spider2 Snowflake Access' form; note "
                "upstream recorded an evaluation-account suspension on "
                "2026-08-12."
            )
    return None


def _run_sqlite(
    db_path: str, sql: str, deadline_s: float, max_rows: int
) -> "pd.DataFrame":
    import pandas as pd

    conn = open_readonly(db_path)
    try:
        columns, rows = run_bounded(conn, sql, deadline_s, max_rows)
    finally:
        conn.close()
    return pd.DataFrame(rows, columns=pd.Index(columns))


def _run_bigquery(sql: str, timeout_s: float, max_rows: int) -> "pd.DataFrame":
    reason = missing_credentials("bigquery")
    if reason:
        raise MissingCredentials(reason)
    from google.cloud import bigquery

    # Upstream's own construction. `from_service_account_json` would accept
    # only one of the credential kinds `_bigquery_reachable` reports on.
    client = bigquery.Client()
    job = client.query(sql)
    rows = job.result(timeout=timeout_s, max_results=max_rows + 1)
    frame = rows.to_dataframe()
    if len(frame) > max_rows:
        raise RuntimeError(f"result exceeded {max_rows} rows")
    return frame


def _run_snowflake(sql: str, timeout_s: float, max_rows: int) -> "pd.DataFrame":
    reason = missing_credentials("snowflake")
    if reason:
        raise MissingCredentials(reason)
    import pandas as pd
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
) -> "pd.DataFrame":
    """Run *sql* on *backend* and return the result frame.

    Every engine-level error propagates: the caller turns it into a scored zero
    with the reason recorded, which is what separates "wrong answer" from "could
    not ask".

    :class:`MissingCredentials` is raised here too, but a caller running this in
    a worker process should not be *relying* on it — ask
    :func:`missing_credentials` first. An exception crossing a process boundary
    is re-raised as whatever unpickled, and telling classes apart on the far
    side means catching broadly at a grading call site, which is exactly what
    must not happen there (``.claude/rules/tasks.md``). The raise stays for a
    direct caller, and as the backstop for a credential that is present but
    rejected by the host.

    A caller must give this at least :func:`caller_timeout` for *backend*, or
    the bounds below never get to fire.
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
