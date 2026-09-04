"""How a text-to-SQL task opens a SQLite database and bounds a statement.

One safety contract, shared by every task that executes model-generated SQL
against a local file: Spider 1.0's two graders and its prompt builder, and
Spider 2.0-lite's local engine. They disagree about what to *do* with a result —
row tuples for one, a frame for the other — and must never disagree about how to
run one, which is the coupling that puts this here rather than in either
benchmark's own tree.

The **bounds** are not shared. A deadline and a row cap are measured against a
particular corpus, and a number carried across from another benchmark is a guess
wearing a measurement's clothes, so each caller passes its own; this module has
no defaults to fall back on.

Both upstreams open a read-write connection, run model SQL with no timeout behind
a bare ``except:``, and fetch the whole result. That cannot be bounded, so the
hardened reading is what ships — the one divergence that does **not** earn a
``_fixed`` variant, because a variant exists so two readings can be compared and
the unsafe reading is not one we will run. Three guards:

* **Read-only** — a ``mode=ro&immutable=1`` URI, so writes fail in the driver
  rather than being pattern-matched out of the SQL. ``immutable=1`` also keeps
  ``-wal``/``-shm`` sidecars out of a data directory shared across samples.
* **No ATTACH** — read-only does not stop ``ATTACH DATABASE`` reaching a writable
  file. An authorizer denies it (and ``DETACH``) outright.
* **Deadline** — a progress handler aborts the statement from inside SQLite. This
  is the guard that matters: ``run_cpu_bound`` bounds how long the *caller*
  waits, but a process pool cannot interrupt a running call, so without this a
  runaway query would hold a worker for the rest of the run. It follows that a
  caller's own timeout must be strictly **longer** than the deadline it passes,
  or the deadline never fires and the guard is decorative.

Kept out of the ``spider`` subpackage, and importing nothing but the standard
library, so that a prompt path can open a database without paying for a grading
dependency behind an optional group.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import sqlite3
import time
from pathlib import Path

#: How often SQLite calls the progress handler, in VM instructions.
_PROGRESS_INTERVAL = 1_000

_DENIED = frozenset({sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH})


def _authorizer(action: int, *_args) -> int:
    return sqlite3.SQLITE_DENY if action in _DENIED else sqlite3.SQLITE_OK


def open_readonly(db_path: str) -> sqlite3.Connection:
    """Open *db_path* read-only and immutable, with ATTACH/DETACH denied."""
    uri = f"{Path(db_path).absolute().as_uri()}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.set_authorizer(_authorizer)
    # Real corpora carry bytes that are not valid UTF-8 — Spider 1.0's
    # `wta_1.players` holds one `first_name` and one `last_name` that are
    # truncated 3-byte sequences, and warehouse exports do it routinely — and
    # sqlite3's default text factory raises on them. The two Spider 1.0
    # upstreams differ: `taoyds/spider` sets no factory and fetches the gold
    # OUTSIDE its bare `except:`, so it *dies* on those two dev examples, while
    # `taoyds/test-suite-sql-eval` sets a lossy `b.decode(errors="ignore")`.
    #
    # `surrogateescape` is chosen over both deliberately: it is
    # round-trip-lossless, so two different invalid byte sequences stay
    # different. Both sides of a comparison are decoded by the same factory, so
    # equality is preserved exactly — this turns a crash into a verdict without
    # being able to change one.
    #
    # Against `ignore` the argument runs the other way and needs measuring,
    # because a *lossy* decode is the one that can move a verdict: it can fold
    # two distinct stored values onto one string. **Measured on Spider 1.0,
    # 2026-09-03: it never does on the pinned data.** Over all 715 databases the
    # graded dev set reaches, exactly two columns hold invalid bytes, and
    # `ignore` is injective over every distinct value in them (0 collisions in
    # 41,324), so the two factories induce the *same* equality relation. A
    # prediction computing new bytes with `substr`/`upper` is covered by the
    # general guarantee above: ours cannot merge what upstream separates, only
    # the reverse.
    #
    # That covers GRADING, where the surrogates never leave the process. A
    # PROMPT path is the other consumer, and its output goes into an HTTP JSON
    # body, where a lone surrogate raises `UnicodeEncodeError` under an
    # `ensure_ascii=False` encoder. It does not fire on either benchmark's
    # pinned data — no sampled row carries the bad bytes — which holds by data
    # rather than by construction, so a mirror with different rows would need
    # the prompt path to sanitise instead.
    conn.text_factory = lambda raw: raw.decode("utf-8", "surrogateescape")
    return conn


def run_bounded(
    conn: sqlite3.Connection, sql: str, deadline_s: float, max_rows: int
) -> tuple[list[str], list[tuple]]:
    """Execute *sql*, aborting past *deadline_s* or beyond *max_rows* rows.

    Returns the cursor's column names alongside the rows. A caller comparing
    tuples ignores the names; one building a frame needs them, and taking them
    from the same cursor is what keeps the two readings of one statement from
    drifting apart.
    """
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
        return columns, rows
    finally:
        conn.set_progress_handler(None, 0)
