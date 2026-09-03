"""How Spider 1.0's databases are opened and how a statement is bounded.

Three consumers, one contract: the prompt builder reads a schema and sample rows,
and the two graders each execute model-generated SQL. They disagree about what to
*do* with a result — deliberately, since that is the difference between the two
metrics — and must never disagree about how to run one. Hence one implementation
of the guards and one pair of bounds rather than three that drift.

Keeping them here rather than in ``_spider_exec`` is also what keeps the optional
``spider`` dependency group optional: ``_spider_exec`` imports the vendored SQL
parser, which reaches nltk at module scope, so a prompt path importing *it* for a
connection would make registering the task pay for a grader it may never run.

Upstream's ``eval_exec_match`` opens a read-write connection, runs
model-generated SQL with no timeout behind a bare ``except:``, and fetches the
whole result. That path executes model output and cannot be bounded, so this task
carries the hardened reading instead — the one divergence that does **not** earn
a ``_fixed`` variant, because a variant exists so two readings can be compared
and the unsafe reading is not one we will run.

Three guards:

* **Read-only** — the database is opened through a ``mode=ro&immutable=1`` URI,
  so writes fail in the driver rather than being pattern-matched out of the SQL.
  ``immutable=1`` additionally keeps ``-wal``/``-shm`` sidecars out of a data
  directory shared across concurrent samples.
* **No ATTACH** — read-only does not stop ``ATTACH DATABASE``, which would reach
  a writable file. An authorizer denies it (and ``DETACH``) outright.
* **Deadline** — a progress handler aborts the statement from inside SQLite.
  This is the guard that matters: ``run_cpu_bound`` bounds how long the *caller*
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
#: Row cap. A cap that truncates a real comparison is a scoring change wearing
#: a safety label, so the number is measured rather than guessed, against every
#: gold on both paths:
#:
#: * one shipped database per sample (execution accuracy) — largest 20,662 rows,
#:   ``SELECT first_name, last_name FROM players ORDER BY birth_date`` on
#:   ``wta_1``;
#: * ~39 distilled databases per sample (test-suite accuracy, 40,167 executions)
#:   — largest 92,450 rows, on a ``dog_kennels`` variant. Only four dev golds
#:   return over 10,000 rows at all, so the distribution is two outliers and a
#:   long flat tail.
#:
#: 500,000 is ~5.4x the worst of those, the same ratio the single-database
#: measurement first set this at. It was raised from 100,000 when the test-suite
#: path put a real gold within 7.5% of the cap: a bound that close is not
#: evidence that no bound binds, and on the gold side binding means a *raise*,
#: failing a sample rather than mis-scoring it. Raising it cannot move a verdict
#: either metric previously reported — both compare row-for-row, so a
#: prediction returning between the old cap and the new one differs in length
#: from every gold here and was already wrong. Time is bounded separately by
#: the deadline above, which the progress handler enforces during the fetch as
#: well as the execute, so this is a memory bound and not a second time one.
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
    # Spider's `wta_1.players` holds bytes that are not valid UTF-8 — one value
    # in `first_name` and one in `last_name`, both truncated 3-byte sequences
    # (`b"Selin G\xe3\xbcLseren"`, `b"Treyes Albarrac\xe3\x8dN"`) — and
    # sqlite3's default text factory raises on them.
    #
    # The two upstreams differ here, so this factory diverges from each of them
    # differently:
    #
    # * `taoyds/spider` (the pre-2020 path) sets no factory at all and fetches
    #   the gold OUTSIDE its bare `except:`, so on the pinned data it does not
    #   score those two dev examples — it dies on them.
    # * `taoyds/test-suite-sql-eval` (the headline path) sets
    #   `b.decode(errors="ignore")`, which drops the bad bytes rather than
    #   raising.
    #
    # `surrogateescape` is chosen over both `replace` and upstream's `ignore`
    # deliberately: it is round-trip-lossless, so two different invalid byte
    # sequences stay different. Both sides of a comparison are decoded by the
    # same factory, so equality is preserved exactly — this turns a crash into a
    # verdict without being able to change one.
    #
    # Against `ignore` that argument runs the other way and needs measuring,
    # because a *lossy* decode is the one that can move a verdict: it can fold
    # two distinct stored values onto one string and make result sets compare
    # equal that ours separates. **Measured, 2026-09-03: it never does on the
    # pinned data.** Over all 715 databases the graded dev set reaches (695
    # distilled plus the 20 shipped), exactly two columns hold invalid bytes —
    # the two above, in the shipped `wta_1` and in the 1 of `wta_1`'s 33
    # distilled variants that kept the original rows — and `ignore` is
    # injective over every distinct value present in them, 0 collisions out of
    # 41,324 name values. An injective decode is a bijection onto what it
    # produces, so the two factories induce the *same* equality relation and
    # neither can outvote the other. The measurement covers stored values; a
    # prediction that runs a string function over those two rows
    # (`substr`, `upper`, concatenation) computes bytes no column holds, and
    # there the guarantee is the general one above — ours cannot merge what
    # upstream separates, only the reverse.
    #
    # That argument covers GRADING, where the surrogates never leave the
    # process. `_spider_schema.build_prompt` is the other consumer, and its
    # output goes out in an HTTP JSON body, where a lone surrogate raises
    # `UnicodeEncodeError` under an `ensure_ascii=False` encoder. It does not
    # fire on the pinned data — no sampled row of any dev database carries the
    # bad bytes, and the archive is checksum-pinned, so that holds by data
    # rather than by construction. A mirror carrying different rows would need
    # the prompt path to sanitise instead.
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
