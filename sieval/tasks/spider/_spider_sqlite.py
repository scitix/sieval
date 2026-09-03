"""How Spider 1.0's databases are opened and how a statement is bounded.

Three consumers, one contract: the prompt builder reads a schema and sample rows,
and the two graders each execute model-generated SQL. They disagree about what to
*do* with a result — that is the difference between the two metrics — and must
never disagree about how to run one.

Kept here rather than in ``_spider_exec`` to keep the optional ``spider`` group
optional: ``_spider_exec`` imports the vendored parser, which reaches nltk at
module scope, so a prompt path importing *it* for a connection would make
registering the task pay for a grader it may never run.

Both upstreams open a read-write connection, run model SQL with no timeout behind
a bare ``except:``, and fetch the whole result. That cannot be bounded, so this
task carries the hardened reading instead — the one divergence that does **not**
earn a ``_fixed`` variant, because a variant exists so two readings can be
compared and the unsafe reading is not one we will run. Three guards:

* **Read-only** — a ``mode=ro&immutable=1`` URI, so writes fail in the driver
  rather than being pattern-matched out of the SQL. ``immutable=1`` also keeps
  ``-wal``/``-shm`` sidecars out of a data directory shared across samples.
* **No ATTACH** — read-only does not stop ``ATTACH DATABASE`` reaching a writable
  file. An authorizer denies it (and ``DETACH``) outright.
* **Deadline** — a progress handler aborts the statement from inside SQLite. This
  is the guard that matters: ``run_cpu_bound`` bounds how long the *caller*
  waits, but a process pool cannot interrupt a running call, so without this a
  runaway query would hold a worker for the rest of the run.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import sqlite3
import time
from pathlib import Path

#: Per-statement wall-clock budget. Measured against every dev gold on the
#: pinned data: the slowest completes in 0.486 s on the shipped databases and
#: 0.359 s across the distilled ones, so this is ~10x headroom and does not
#: bind. Short enough that a pathological prediction cannot occupy a pool worker.
DEFAULT_DEADLINE_S = 5.0
#: Row cap. A cap that truncates a real comparison is a scoring change wearing a
#: safety label, so it is measured rather than guessed, against every gold on
#: both paths: the largest result is 20,662 rows on the shipped databases and
#: 92,450 across the distilled suite, and only four dev golds exceed 10,000 rows
#: at all. 500,000 is ~5.4x the worst of those. Time is bounded separately by the
#: deadline above, so this is a memory bound and not a second time one.
#:
#: Raised from 100,000, where a real gold sat 7.5% under the cap — a bound that
#: close is not evidence that no bound binds, and on the gold side binding means
#: a *raise*, failing a sample rather than mis-scoring it. The raise cannot move
#: a verdict either metric previously reported: both compare row-for-row, so a
#: prediction returning between the old cap and the new one differs in length
#: from every gold here and was already wrong.
#:
#: **What the raise costs, measured** (2026-09-03). The cap is also the ceiling:
#: ``run_bounded`` fetches ``max_rows + 1`` in one call, so a capped result is
#: fully materialised before the length check rejects it. A three-way cross join
#: on ``car_1`` moves peak RSS by ~450 MB for one statement (~90 MB at the old
#: cap), and with ``result_eq`` copying both sides across up to 8 pool workers
#: the worst case is a multi-GB transient. Kept anyway: the exposure is not
#: reachable by the data — no dev gold comes within 5x, and no prediction in
#: either full dev pass hit it — while the old cap's failure mode was a *failed
#: sample*. Should it ever need lowering, the fix is a chunked fetch that rejects
#: at the threshold rather than a smaller number, which decouples the ceiling
#: from the cap.
DEFAULT_MAX_ROWS = 500_000
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
    # Spider's `wta_1.players` holds bytes that are not valid UTF-8 — one
    # `first_name` and one `last_name`, both truncated 3-byte sequences — and
    # sqlite3's default text factory raises on them. The two upstreams differ:
    # `taoyds/spider` sets no factory and fetches the gold OUTSIDE its bare
    # `except:`, so it *dies* on those two dev examples, while
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
    # two distinct stored values onto one string. **Measured, 2026-09-03: it
    # never does on the pinned data.** Over all 715 databases the graded dev set
    # reaches, exactly two columns hold invalid bytes, and `ignore` is injective
    # over every distinct value in them (0 collisions in 41,324), so the two
    # factories induce the *same* equality relation. A prediction computing new
    # bytes with `substr`/`upper` is covered by the general guarantee above:
    # ours cannot merge what upstream separates, only the reverse.
    #
    # That covers GRADING, where the surrogates never leave the process.
    # `_spider_schema.build_prompt` is the other consumer, and its output goes
    # into an HTTP JSON body, where a lone surrogate raises `UnicodeEncodeError`
    # under an `ensure_ascii=False` encoder. It does not fire on the pinned data
    # — no sampled row carries the bad bytes — which holds by data rather than by
    # construction, so a mirror with different rows would need the prompt path to
    # sanitise instead.
    conn.text_factory = lambda raw: raw.decode("utf-8", "surrogateescape")
    return conn


def run_bounded(
    conn: sqlite3.Connection, sql: str, deadline_s: float, max_rows: int
) -> list[tuple]:
    """Execute *sql*, aborting past *deadline_s* or beyond *max_rows* rows."""
    end = time.monotonic() + deadline_s
    conn.set_progress_handler(
        lambda: 1 if time.monotonic() > end else 0, _PROGRESS_INTERVAL
    )
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            raise RuntimeError(f"result exceeded {max_rows} rows")
        return rows
    finally:
        conn.set_progress_handler(None, 0)
