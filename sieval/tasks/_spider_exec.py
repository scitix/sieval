"""Hardened SQLite execution behind Spider 1.0's execution accuracy.

Upstream's ``eval_exec_match`` opens a read-write connection, runs
model-generated SQL with no timeout behind a bare ``except:``, and fetches the
whole result. That path executes model output and cannot be bounded, so this
task carries the hardened reading instead — the one divergence that does **not**
earn a ``_fixed`` variant, because a variant exists so two readings can be
compared and the unsafe reading is not one we will run.

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

Everything upstream does that safety does not object to is preserved, including
where upstream is wrong: the comparison is still ``eval_exec_match``'s
column-keyed ``res_map`` equality rather than a plain result-set compare, and an
unparseable prediction is still scored against an empty parse rather than
skipped.

One further substitution, for the same reason: upstream's ``get_schema`` opens
its own **read-write** connection. It runs only fixed introspection, so it cannot
be driven to write, but it can still create sidecars in a shared directory.
:func:`read_schema_readonly` reproduces its dict through the hardened connection
instead — mirroring its queries exactly, lowercasing and unfiltered
``sqlite_master`` rows included, so the parse it feeds is unchanged. The
equivalence is asserted against upstream's own function in the tests.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import sqlite3
import time
from functools import cache
from pathlib import Path

from sieval.community.spider import (
    EMPTY_SQL,
    Evaluator,
    Schema,
    build_foreign_key_map_from_json,
    build_valid_col_units,
    get_sql,
    rebuild_sql_col,
    rebuild_sql_val,
)

#: Per-statement wall-clock budget. Measured against every dev gold on the
#: pinned data: the slowest completes in 0.486 s, so this is ~10x headroom and
#: does not bind. Short enough that a pathological prediction cannot occupy a
#: pool worker.
DEFAULT_DEADLINE_S = 5.0
#: Row cap. Measured against every dev gold: the largest returns 20,662 rows
#: (``SELECT first_name, last_name FROM players ORDER BY birth_date`` on
#: ``wta_1``), so this is ~5x headroom and does not bind. An earlier 10,000 did
#: bind on exactly that query — a cap that truncates a real comparison is a
#: scoring change wearing a safety label, which is why the number is measured
#: rather than guessed.
DEFAULT_MAX_ROWS = 100_000
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
    # Spider's `wta_1.players.last_name` holds bytes that are not valid UTF-8,
    # and sqlite3's default text factory raises on them. Upstream hits the same
    # rows and fetches the gold OUTSIDE its bare `except:`, so on the pinned
    # data it does not score those two dev examples — it dies on them.
    #
    # `surrogateescape` is chosen over `replace` deliberately: it is
    # round-trip-lossless, so two different invalid byte sequences stay
    # different. Both sides of a comparison are decoded by the same factory, so
    # equality is preserved exactly — this turns a crash into a verdict without
    # being able to change one.
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


def read_schema_readonly(db_path: str) -> dict:
    """Upstream ``get_schema``'s dict, read through a hardened connection.

    Mirrors upstream query for query: table names straight from
    ``sqlite_master`` with no ``sqlite_%`` filter, every name lowercased, columns
    from ``PRAGMA table_info``. Any quirk upstream has, this has too — which is
    the point, since the result feeds the parser.
    """
    conn = open_readonly(db_path)
    try:
        tables = [
            str(row[0].lower())
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        ]
        return {
            table: [
                str(col[1].lower())
                for col in conn.execute(f"PRAGMA table_info({table})").fetchall()
            ]
            for table in tables
        }
    finally:
        conn.close()


def _run_bounded(
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


@cache
def _kmaps(tables_json_path: str) -> dict:
    """Foreign-key maps for every database, parsed once per worker process."""
    return build_foreign_key_map_from_json(tables_json_path)


def _res_map(rows: list[tuple], parsed: dict) -> dict:
    """Upstream ``eval_exec_match``'s column-keyed projection, lifted verbatim.

    Kept identical on purpose: a plain result-set comparison would be a scoring
    change wearing a safety label.
    """
    val_units = [unit[1] for unit in parsed["select"][1]]
    rmap = {}
    for idx, val_unit in enumerate(val_units):
        key = (
            tuple(val_unit[1])
            if not val_unit[2]
            else (val_unit[0], tuple(val_unit[1]), tuple(val_unit[2]))
        )
        rmap[key] = [r[idx] for r in rows]
    return rmap


def grade_one(
    db_path: str,
    tables_json_path: str,
    db_id: str,
    pred_sql: str,
    gold_sql: str,
    deadline_s: float = DEFAULT_DEADLINE_S,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> dict:
    """Score one prediction, mirroring upstream's per-example sequence.

    Module-level and picklable by name so ``run_cpu_bound`` can ship it to a
    worker process. Returns ``exact_match``, ``execution``, ``hardness`` and
    ``error`` (the reason the prediction would not run, when it would not).
    """
    error: str | None = None
    schema = Schema(read_schema_readonly(db_path))
    try:
        g_sql = get_sql(schema, gold_sql)
    except Exception as exc:  # a broken gold is our bug, not a model failure
        raise ValueError(f"Spider gold SQL failed to parse for {db_id!r}") from exc

    evaluator = Evaluator()
    hardness = evaluator.eval_hardness(g_sql)

    try:
        p_sql = get_sql(schema, pred_sql)
    except Exception:
        # Upstream's behaviour: score the empty parse rather than skip the row.
        # Annotated because the literal's inferred value type would otherwise
        # narrow `p_sql["from"]` to a union including `list`.
        p_sql: dict = dict(EMPTY_SQL)

    kmap = _kmaps(tables_json_path)[db_id]
    g_valid = build_valid_col_units(g_sql["from"]["table_units"], schema)
    g_sql = rebuild_sql_col(g_valid, rebuild_sql_val(g_sql), kmap)
    p_valid = build_valid_col_units(p_sql["from"]["table_units"], schema)
    p_sql = rebuild_sql_col(p_valid, rebuild_sql_val(p_sql), kmap)

    # Execution BEFORE exact match, which is upstream's order in `evaluate()`
    # and is load-bearing: `eval_exact_match` mutates both parse trees in place
    # (it sorts and rewrites their clauses), so scoring it first would hand
    # `_res_map` corrupted select units and silently zero the execution column.
    #
    # Within execution, gold runs FIRST, where upstream runs the prediction
    # first and returns False without ever touching the gold. The order is
    # visible only when BOTH are unrunnable: upstream scores that a wrong
    # answer, this raises and the sample lands in `fails`. Deliberate, and
    # unreachable on the pinned data — all 1,034 dev golds execute — but it is
    # the reading that keeps a gold we cannot run our bug rather than the
    # model's, which is the same rule the parse above follows.
    execution = False
    conn = open_readonly(db_path)
    try:
        gold_rows = _run_bounded(conn, gold_sql, deadline_s, max_rows)
        try:
            pred_rows = _run_bounded(conn, pred_sql, deadline_s, max_rows)
        except Exception as exc:
            # Upstream returns False for any prediction that will not run.
            error = f"{type(exc).__name__}: {exc}"
        else:
            execution = _res_map(pred_rows, p_sql) == _res_map(gold_rows, g_sql)
    finally:
        conn.close()

    exact_match = bool(evaluator.eval_exact_match(p_sql, g_sql))

    return {
        "exact_match": exact_match,
        "execution": execution,
        "hardness": hardness,
        "error": error,
    }
