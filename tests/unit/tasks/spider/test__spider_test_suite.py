"""Spider's post-2020 test-suite accuracy, the metric the task reports.

Two properties carry the whole difference from the pre-2020 reading, and each
gets a test that fails if it is lost:

* **No parse.** A prediction upstream's parser rejects is still scored on what
  SQLite returns for it. Those are ordinary constructs -- a column alias, a
  quoted identifier -- so this is most of the gap.
* **Every database.** A prediction must agree with the gold on *all* the
  distilled databases, not the one shipped with the dataset. Agreeing on one is
  what the metric exists to stop counting as equivalence.

The fixtures build a miniature test suite rather than reading the 1.3 GB archive:
two databases, same schema, rows chosen so a query that is wrong but coincidental
agrees on the first and disagrees on the second. That is the archive's own
construction in the small, and it keeps the test hermetic.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import json
import os
import sqlite3

import pytest

from sieval.community.spider_test_suite import eval_exec_match
from sieval.tasks.spider._spider_test_suite import (
    KEEP_DISTINCT,
    PLUG_VALUE,
    grade_one,
)
from sieval.tasks.spider._spider_test_suite import (
    # Aliased, not renamed at the source: pytest collects any module-level
    # `test_*` callable, so importing these under their own names would have it
    # try to run the production functions as tests and error on their arguments
    # as missing fixtures. The names read correctly where they are defined --
    # they are the test-suite metric -- so the workaround belongs here.
    test_suite_databases as suite_databases,
)
from sieval.tasks.spider._spider_test_suite import (
    test_suite_match as suite_match,
)

DB_ID = "concert_singer"


def _make_db(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE singer (id int, name text, age int)")
    conn.executemany("INSERT INTO singer VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()


@pytest.fixture
def suite(tmp_path):
    """Two distilled databases for one db_id.

    On `v0` every singer is over 30, so `WHERE age > 30` and a bare select agree.
    On `v1` one is not, which is what separates them -- exactly how upstream's
    generated suites are meant to work.
    """
    root = tmp_path / "test_suite" / "database"
    _make_db(root / DB_ID / "concert_singer.sqlite", [(1, "Joe", 41), (2, "Ann", 52)])
    _make_db(root / DB_ID / "concert_singer_1.sqlite", [(1, "Joe", 41), (2, "Ann", 22)])
    return str(root)


@pytest.fixture
def shipped(tmp_path):
    """The single database the dataset ships, for the pre-2020 metrics.

    Deliberately the *coincidental* one: a prediction that only agrees here is
    the case the two metrics disagree about.
    """
    path = tmp_path / "database" / DB_ID / "concert_singer.sqlite"
    _make_db(path, [(1, "Joe", 41), (2, "Ann", 52)])
    return str(path)


@pytest.fixture
def tables_json(tmp_path):
    path = tmp_path / "tables.json"
    path.write_text(
        json.dumps(
            [
                {
                    "db_id": DB_ID,
                    "table_names_original": ["singer"],
                    "table_names": ["singer"],
                    "column_names_original": [
                        [-1, "*"],
                        [0, "id"],
                        [0, "name"],
                        [0, "age"],
                    ],
                    "column_names": [[-1, "*"], [0, "id"], [0, "name"], [0, "age"]],
                    "column_types": ["text", "number", "text", "number"],
                    "foreign_keys": [],
                    "primary_keys": [1],
                }
            ]
        )
    )
    return str(path)


# --- upstream's flags are pinned, not exposed -------------------------------


def test_upstream_cli_defaults_are_pinned():
    """A run configured either way is not comparable to a published score."""
    assert PLUG_VALUE is False
    assert KEEP_DISTINCT is False


# --- database discovery -----------------------------------------------------


def test_databases_are_found_in_sorted_order(suite):
    found = suite_databases(suite, DB_ID)
    assert [p.rsplit("/", 1)[1] for p in found] == [
        "concert_singer.sqlite",
        "concert_singer_1.sqlite",
    ]
    assert found == sorted(found)


def test_non_sqlite_files_are_excluded(suite, tmp_path):
    """The archive ships stray .txt/.csv under car_1/. Upstream's substring
    filter is what skips them, and widening it would add files to a comparison
    upstream does not make."""
    directory = tmp_path / "test_suite" / "database" / DB_ID
    (directory / "schema.txt").write_text("not a database")
    (directory / "data.csv").write_text("also not")
    assert len(suite_databases(suite, DB_ID)) == 2


def test_the_substring_filter_is_kept_not_a_suffix_test(suite, tmp_path):
    """Upstream tests `'.sqlite' in basename`, so a mid-name hit counts.

    Kept deliberately. Fails if the filter is 'cleaned up' to `endswith`, which
    would drop files from a comparison upstream makes.
    """
    directory = tmp_path / "test_suite" / "database" / DB_ID
    _make_db(directory / "concert_singer.sqlite.bak", [(1, "Joe", 41), (2, "Ann", 52)])
    assert len(suite_databases(suite, DB_ID)) == 3


# --- property 1: no parse gate ----------------------------------------------


@pytest.mark.parametrize(
    "prediction",
    [
        "SELECT name AS singer_name FROM singer",
        'SELECT "name" FROM singer',
        "SELECT name FROM (SELECT name FROM singer)",
        "SELECT name FROM singer WHERE name IS NOT NULL",
    ],
)
def test_a_prediction_the_parser_rejects_is_scored_on_its_results(suite, prediction):
    """The headline difference, on the same constructs `test__spider_exec.py`
    watches score 0. Each returns exactly the gold's rows and must score True.
    """
    matched, error = suite_match(suite, DB_ID, prediction, "SELECT name FROM singer")
    assert matched is True
    assert error is None


# --- property 2: agreement on every database --------------------------------


def test_a_coincidentally_correct_prediction_fails_on_a_second_database(suite):
    """The reason the archive exists, as a verdict.

    `WHERE age > 30` returns the gold's rows on `v0` and not on `v1`. Fails if
    the loop stops after one database, or if the AND becomes an OR.
    """
    matched, error = suite_match(
        suite,
        DB_ID,
        "SELECT name FROM singer WHERE age > 30",
        "SELECT name FROM singer",
    )
    assert matched is False
    assert error is None  # it ran fine; it just disagreed


def test_the_same_prediction_passes_the_pre_2020_metric(shipped, tables_json, suite):
    """The two metrics must actually disagree here, or the fixture proves nothing.

    Execution accuracy sees only the shipped database, where the coincidence
    holds, so it scores True while test-suite accuracy scores False. This is the
    gap the new metric closes, asserted end to end through `grade_one`.
    """
    out = grade_one(
        shipped,
        tables_json,
        suite,
        DB_ID,
        "SELECT name FROM singer WHERE age > 30",
        "SELECT name FROM singer",
    )
    assert out["execution"] is True
    assert out["test_suite"] is False


def test_an_equivalent_prediction_passes_on_every_database(suite):
    matched, error = suite_match(
        suite, DB_ID, "SELECT name FROM singer WHERE 1=1", "SELECT name FROM singer"
    )
    assert matched is True
    assert error is None


# --- row order is read off the gold -----------------------------------------


def test_order_is_significant_only_when_the_gold_orders(suite):
    """Upstream reads `order by` off the GOLD, so a prediction cannot opt into a
    laxer comparison by omitting one."""
    ordered_gold = "SELECT name FROM singer ORDER BY age"
    # Same rows, opposite order. Must fail, because the gold orders.
    wrong_order, _ = suite_match(
        suite, DB_ID, "SELECT name FROM singer ORDER BY age DESC", ordered_gold
    )
    assert wrong_order is False
    # With an unordered gold the same pair of results must compare equal.
    unordered, _ = suite_match(
        suite,
        DB_ID,
        "SELECT name FROM singer ORDER BY age DESC",
        "SELECT name FROM singer",
    )
    assert unordered is True


# --- failure postures -------------------------------------------------------


def test_a_prediction_that_will_not_run_scores_false_with_a_reason(suite):
    matched, error = suite_match(
        suite, DB_ID, "SELECT nope FROM singer", "SELECT name FROM singer"
    )
    assert matched is False
    assert error is not None
    assert "concert_singer.sqlite" in error


def test_a_gold_that_will_not_run_raises(suite):
    """A gold we cannot execute is our bug -- a stale archive, a bound of ours
    that binds -- so the sample must fail rather than score the model wrong."""
    with pytest.raises(ValueError, match="gold SQL failed on"):
        suite_match(suite, DB_ID, "SELECT name FROM singer", "SELECT nope FROM singer")


@pytest.mark.parametrize("prediction", ["", "   ", "\n"])
def test_a_blank_prediction_is_scored_not_raised(suite, prediction):
    """`extract_sql` returns nothing on a refusal or a truncated reply.

    Upstream cannot receive this input at all -- a blank line is a session
    boundary in its prediction file -- and `remove_distinct`, the first thing a
    blank string reaches, raises `IndexError` on the empty parse. Without the
    guard the sample lands in `fails` as `exception::IndexError` instead of
    scoring 0.
    """
    matched, error = suite_match(suite, DB_ID, prediction, "SELECT name FROM singer")
    assert matched is False
    assert error == "blank prediction"


def test_a_blank_prediction_does_not_match_a_gold_that_returns_no_rows(suite):
    """The reason the blank guard is a verdict and not just crash-avoidance.

    SQLite returns `[]` for empty SQL, so a blank prediction executed against a
    gold that legitimately selects nothing compares EQUAL on every database and
    scores **correct**. Fails if the guard is relaxed into a `try/except
    IndexError`, which fixes the crash and leaves this.
    """
    empty_gold = "SELECT name FROM singer WHERE age > 999"
    matched, _ = suite_match(suite, DB_ID, empty_gold, empty_gold)
    assert matched is True  # the gold really does return no rows, on both

    matched, error = suite_match(suite, DB_ID, "", empty_gold)
    assert matched is False
    assert error == "blank prediction"


def test_a_blank_gold_raises_rather_than_reporting_the_model_wrong(suite):
    """Unreachable on the pinned data; the posture has to hold anyway.

    sqlparse would raise `IndexError` from inside a vendored file, naming
    neither the sample nor the reason.
    """
    with pytest.raises(ValueError, match="gold SQL is blank"):
        suite_match(suite, DB_ID, "SELECT name FROM singer", "  ")


# --- the composed result ----------------------------------------------------


def test_grade_one_returns_every_metric_in_one_dispatch(shipped, tables_json, suite):
    out = grade_one(
        shipped,
        tables_json,
        suite,
        DB_ID,
        "SELECT name FROM singer",
        "SELECT name FROM singer",
    )
    assert out.keys() == {
        "exact_match",
        "execution",
        "hardness",
        "error",
        "parsed",
        "test_suite",
        "test_suite_error",
    }
    assert out["test_suite"] is True
    assert out["execution"] is True
    assert out["exact_match"] is True
    assert out["parsed"] is True
    assert out["error"] is None
    assert out["test_suite_error"] is None


def test_the_two_error_keys_are_independent(shipped, tables_json, suite, tmp_path):
    """The same prediction can run on one tree and not the other.

    Here the shipped database has the column and a distilled one does not, so
    `error` is None while `test_suite_error` is set. Folding them into one key
    would make that unreadable.
    """
    stripped = tmp_path / "test_suite" / "database" / DB_ID / "concert_singer_2.sqlite"
    stripped.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(stripped)
    conn.execute("CREATE TABLE singer (id int, name text)")  # no `age`
    conn.execute("INSERT INTO singer VALUES (1, 'Joe')")
    conn.commit()
    conn.close()

    out = grade_one(
        shipped,
        tables_json,
        suite,
        DB_ID,
        "SELECT name FROM singer WHERE age > 0",
        "SELECT name FROM singer",
    )
    assert out["error"] is None
    assert out["test_suite_error"] is not None
    assert "concert_singer_2.sqlite" in out["test_suite_error"]


# --- anchored against upstream ----------------------------------------------


@pytest.mark.parametrize(
    "pred",
    [
        "SELECT name FROM singer WHERE age > 30",  # equivalent to the gold
        "SELECT name FROM singer",  # agrees on v0 only
        "SELECT s.name AS singer_name FROM singer s WHERE s.age > 30",  # aliased
        "SELECT DISTINCT name FROM singer WHERE age > 30",  # DISTINCT is stripped
        'SELECT "name" FROM "singer" WHERE age > 30',  # quoted identifiers
        "SELECT name FROM singer WHERE age < 30",  # a different answer
        "SELECT nope FROM singer",  # will not run at all
    ],
)
def test_verdict_matches_upstream(suite, pred):
    """The headline's whole claim: hardened execution, upstream's comparison.

    Upstream is called with its own module globals, over the same directory of
    databases and at the same two flags, so an agreement here is between two
    implementations rather than between ours and a restatement of ours. Its
    `eval_exec_match` globs the directory of the database it is *handed*, which
    is why it takes the named `.sqlite` where ours takes the root.

    Three of these are the cases the metric exists for: the aliased and quoted
    predictions are ones the pre-2020 parser rejects outright, and the bare
    select agrees with the gold on the first database and not the second.
    """
    gold = "SELECT name FROM singer WHERE age > 30"
    named = os.path.join(suite, DB_ID, "concert_singer.sqlite")
    ours, _ = suite_match(suite, DB_ID, pred, gold)
    theirs = bool(eval_exec_match(named, pred, gold, PLUG_VALUE, KEEP_DISTINCT, False))
    assert ours is theirs
