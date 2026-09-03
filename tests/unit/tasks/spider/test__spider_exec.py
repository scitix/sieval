"""Correctness tests for Spider 1.0's pre-2020 grading worker.

Scope is the two parse-based metrics and the read-only schema substitution the
parse depends on. The connection guards and the execution bounds they share with
the prompt builder and the test-suite grader are tested in
`test__spider_sqlite.py`, next to the module that owns them.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import json
import sqlite3

import pytest

from sieval.community.spider import (
    EMPTY_SQL,
    Schema,
    build_foreign_key_map_from_json,
    build_valid_col_units,
    eval_exec_match,
    get_schema,
    get_sql,
    rebuild_sql_col,
    rebuild_sql_val,
)
from sieval.tasks.spider._spider_exec import grade_one, read_schema_readonly
from sieval.tasks.spider._spider_sqlite import open_readonly, run_bounded


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


# --- the deadline reaches the grader ----------------------------------------


def test_runaway_prediction_is_scored_wrong_with_a_reason(db, tables_json):
    """The bound is plumbed through to a verdict, not just available.

    `test__spider_sqlite.py` proves the abort happens; this proves `grade_one`
    turns it into `execution=False` plus a named `error`, which is what
    `n_execution_errors` counts.
    """
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
    assert out["parsed"] is True


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
    # The gate is REPORTED, so its rate is readable. Here SQLite rejects the
    # string too, so `error` is also set; the case where it is NOT is below,
    # and that is the one the flag exists for.
    assert out["parsed"] is False


@pytest.mark.parametrize(
    "prediction",
    [
        "SELECT name AS singer_name FROM singer",
        'SELECT "name" FROM singer',
        "SELECT name FROM (SELECT name FROM singer)",
        "SELECT name FROM singer WHERE name IS NOT NULL",
    ],
)
def test_a_correct_prediction_the_parser_rejects_scores_zero_silently(
    db, tables_json, prediction
):
    """The parse gate's whole cost, in the cases that make it invisible.

    Each of these runs in SQLite and returns EXACTLY the gold's rows, so both
    pre-2020 metrics are reporting 0 for a right answer -- with `error` None,
    because nothing failed to run. That is why `parsed` is recorded: it is the
    only place the rate of this can come from, and without it a low execution
    accuracy is uninterpretable. `_spider_test_suite` is the metric that scores
    these correctly.

    Ordinary SQL, not exotica: a column alias, a quoted identifier, a derived
    table, `IS NOT NULL`. Fails if `parsed` is dropped, and fails loudly if a
    construct here starts parsing -- in which case move it out rather than
    weakening the assertion.
    """
    gold = "SELECT name FROM singer"
    conn = open_readonly(str(db))
    try:
        assert run_bounded(conn, prediction, 5.0, 1000) == run_bounded(
            conn, gold, 5.0, 1000
        )
    finally:
        conn.close()

    out = grade_one(str(db), str(tables_json), "concert_singer", prediction, gold)
    assert out["parsed"] is False
    assert out["error"] is None
    assert out["execution"] is False
    assert out["exact_match"] is False


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


# --- anchored against upstream ----------------------------------------------
#
# The hardened executor's whole claim is that it changes HOW a query runs and
# never WHAT the comparison concludes. Naming each guard and arguing it is
# verdict-preserving is not the same as having run both, so this pins ours
# against upstream's own `eval_exec_match` -- called with its own module
# globals, so `DISABLE_VALUE` and `res_map` are upstream's rather than a
# reimplementation that could agree with us for the wrong reason.


def _upstream_execution(db_path, db_id, tables_json_path, pred_sql, gold_sql):
    """Upstream's `evaluate()` sequence, up to its execution verdict.

    Fresh parses each call: `eval_exact_match` rewrites both trees in place, so
    a shared parse would hand the second caller corrupted select units.
    """
    schema = Schema(get_schema(str(db_path)))
    gold = get_sql(schema, gold_sql)
    try:
        pred = get_sql(schema, pred_sql)
    except Exception:
        pred = dict(EMPTY_SQL)
    kmap = build_foreign_key_map_from_json(str(tables_json_path))[db_id]
    gold = rebuild_sql_col(
        build_valid_col_units(gold["from"]["table_units"], schema),
        rebuild_sql_val(gold),
        kmap,
    )
    pred = rebuild_sql_col(
        build_valid_col_units(pred["from"]["table_units"], schema),
        rebuild_sql_val(pred),
        kmap,
    )
    return bool(eval_exec_match(str(db_path), pred_sql, gold_sql, pred, gold))


@pytest.mark.parametrize(
    "pred",
    [
        "SELECT count(*) FROM singer",  # identical to the gold
        "SELECT COUNT(*) FROM singer",  # same query, different case
        "SELECT count(*) FROM singer WHERE id > 0",  # same answer, other route
        "SELECT count(*) FROM singer WHERE id > 1",  # a different answer
        "SELECT name FROM singer",  # a different projection
        "SELECT nope FROM singer",  # will not run at all
        "SELECT c FROM singer s",  # runs nowhere; parser rejects it too
    ],
)
def test_execution_verdict_matches_upstream(db, tables_json, pred):
    gold = "SELECT count(*) FROM singer"
    ours = grade_one(str(db), str(tables_json), "concert_singer", pred, gold)
    theirs = _upstream_execution(db, "concert_singer", tables_json, pred, gold)
    assert ours["execution"] is theirs
