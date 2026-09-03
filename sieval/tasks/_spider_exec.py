"""Spider 1.0's pre-2020 execution accuracy and exact set match.

Both metrics run through the vendored parser: exact set match compares the two
parse trees, and execution accuracy compares results keyed by the *parsed*
select clause. That is what makes this the pre-2020 reading —
``_spider_test_suite`` carries the current one, which compares results directly
and needs no parse.

Everything upstream does that safety does not object to is preserved, including
where upstream is wrong: the comparison is still ``eval_exec_match``'s
column-keyed ``res_map`` equality rather than a plain result-set compare, and an
unparseable prediction is still scored against an empty parse rather than
skipped. The guards safety *does* object to, and the bounds they apply, are in
``_spider_sqlite``.

One further substitution, for the same reason: upstream's ``get_schema`` opens
its own **read-write** connection. It runs only fixed introspection, so it cannot
be driven to write, but it can still create sidecars in a shared directory.
:func:`read_schema_readonly` reproduces its dict through the hardened connection
instead — mirroring its queries exactly, lowercasing and unfiltered
``sqlite_master`` rows included, so the parse it feeds is unchanged. The
equivalence is asserted against upstream's own function in the tests.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from functools import cache

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

from ._spider_sqlite import (
    DEFAULT_DEADLINE_S,
    DEFAULT_MAX_ROWS,
    open_readonly,
    run_bounded,
)


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
    worker process. Returns ``exact_match``, ``execution``, ``hardness``,
    ``error`` (the reason the prediction would not run, when it would not) and
    ``parsed`` (whether the parser accepted the prediction at all).
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
        # Reported, not acted on. Both metrics below are computed from `p_sql`,
        # so a prediction that lands here scores 0 on each of them whatever
        # SQLite would have returned for it — which makes them uninterpretable
        # without a count of how often it happens. The flag is the only place
        # that count can come from: `error` stays `None` on this path, because
        # the SQL itself may well run.
        parsed = False
    else:
        parsed = True

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
        gold_rows = run_bounded(conn, gold_sql, deadline_s, max_rows)
        try:
            pred_rows = run_bounded(conn, pred_sql, deadline_s, max_rows)
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
        "parsed": parsed,
    }
