"""Guards and bounds on how Spider 1.0's databases are opened and read.

Each guard is proved by deletion: the assertion must fail if the guard is
removed, or it has tested nothing.

These live apart from `test__spider_exec.py` because the module does: the
connection is shared by the prompt builder and both graders, and only the
graders pull the vendored SQL parser. Importing it from a grader module would
put nltk and sqlparse back at task-registration time, which
`test_import_discipline_family.py` is what pins.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import sqlite3

import pytest

from sieval.tasks.spider._spider_sqlite import (
    DEFAULT_DEADLINE_S,
    DEFAULT_MAX_ROWS,
    open_readonly,
    run_bounded,
)


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


# --- the module must not drag the optional group in --------------------------


def test_importing_the_connection_needs_no_grading_dependency():
    """Whole reason this module exists, asserted where a reader will look.

    `test_import_discipline_family.py` enforces it from the task's side, in a
    fresh interpreter; this states it locally, so moving `open_readonly` back
    into `_spider_exec` fails a test in the file it was moved out of.

    Read off the import statements rather than the source text: the docstring
    names nltk and sqlparse on purpose, to say why they must stay out.
    """
    import ast
    import inspect

    import sieval.tasks.spider._spider_sqlite as module

    tree = ast.parse(inspect.getsource(module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported == {"sqlite3", "time", "pathlib"}, imported


# --- guard 1: read-only -----------------------------------------------------


def test_write_is_rejected_by_the_connection(db):
    """A model can emit DDL/DML. Read-only must stop it at the driver."""
    conn = open_readonly(str(db))
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("DROP TABLE singer")
    conn.close()


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO singer VALUES (3, 'Eve')",
        "UPDATE singer SET name = 'x'",
        "DELETE FROM singer",
        "CREATE TABLE evil (a int)",
    ],
)
def test_every_write_vector_is_rejected(db, statement):
    """Not just DROP — DML and DDL alike must die at the driver."""
    conn = open_readonly(str(db))
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute(statement)
    conn.close()


# --- guard 2: no ATTACH -----------------------------------------------------


def test_attach_is_rejected(db, tmp_path):
    """`mode=ro` alone does not stop ATTACH — that is the escape hatch."""
    conn = open_readonly(str(db))
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        conn.execute(f"ATTACH DATABASE '{tmp_path / 'evil.db'}' AS evil")
    conn.close()


# --- guard 3: deadline and row cap ------------------------------------------


def test_deadline_aborts_a_runaway_query(db):
    """An unbounded recursive CTE must abort inside the worker, not hang it."""
    runaway = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) "
        "SELECT count(*) FROM c"
    )
    conn = open_readonly(str(db))
    try:
        with pytest.raises(sqlite3.OperationalError, match="interrupted"):
            run_bounded(conn, runaway, deadline_s=0.5, max_rows=DEFAULT_MAX_ROWS)
    finally:
        conn.close()


def test_row_cap_bounds_a_large_result(db):
    """The cap must fire on a result larger than it, before the whole fetch."""
    big = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c WHERE x<100000) "
        "SELECT x FROM c"
    )
    conn = open_readonly(str(db))
    try:
        with pytest.raises(RuntimeError, match="exceeded 10 rows"):
            run_bounded(conn, big, deadline_s=10.0, max_rows=10)
    finally:
        conn.close()


def test_the_progress_handler_is_cleared_after_a_statement(db):
    """Left installed, the next statement inherits an already-expired deadline.

    The connection outlives one `run_bounded` call -- the test-suite path runs
    gold and prediction on the same one -- so a handler that survives would
    abort the second query immediately. Fails if the `finally` is dropped.
    """
    conn = open_readonly(str(db))
    try:
        with pytest.raises(sqlite3.OperationalError, match="interrupted"):
            run_bounded(
                conn,
                "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) "
                "SELECT count(*) FROM c",
                deadline_s=0.2,
                max_rows=DEFAULT_MAX_ROWS,
            )
        # Same connection, deadline long since past: must still run.
        assert run_bounded(
            conn, "SELECT count(*) FROM singer", 5.0, DEFAULT_MAX_ROWS
        ) == [(2,)]
    finally:
        conn.close()


def test_default_bounds_do_not_bind_on_a_realistic_result():
    """The bounds must sit above real gold results, not truncate them.

    Measured over every dev gold on both grading paths: 20,662 rows / 0.486 s
    against the shipped databases, and 92,450 rows / 0.359 s across the 40,167
    executions of the distilled test suites. This pins the constants so a later
    'tightening' has to argue with a failing test rather than silently rescoring
    the benchmark.
    """
    assert DEFAULT_MAX_ROWS > 92_450
    assert DEFAULT_DEADLINE_S > 0.486


# --- text decoding ----------------------------------------------------------


def test_non_utf8_text_is_readable(tmp_path):
    """Spider's `wta_1.players.last_name` holds bytes that are not valid UTF-8.

    sqlite3's default text factory raises on them, which is why upstream dies on
    two dev examples rather than scoring them. Fails without the surrogateescape
    factory in `open_readonly`.
    """
    path = tmp_path / "latin.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (name text)")
    # x'41FF42' is 'A', an invalid continuation byte, then 'B'.
    conn.execute("INSERT INTO t VALUES (CAST(x'41FF42' AS TEXT))")
    conn.commit()
    conn.close()

    reader = open_readonly(str(path))
    try:
        (value,) = reader.execute("SELECT name FROM t").fetchone()
    finally:
        reader.close()
    # Round-trips losslessly, so two different bad byte sequences stay different.
    assert value.encode("utf-8", "surrogateescape") == b"\x41\xff\x42"


def test_two_different_bad_byte_sequences_stay_different(tmp_path):
    """What `surrogateescape` buys over `replace`, stated as a verdict.

    Under `errors="replace"` both rows decode to U+FFFD and compare EQUAL, so a
    wrong prediction would score correct. This is the assertion that makes the
    choice of error handler load-bearing rather than stylistic.
    """
    path = tmp_path / "two.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (id int, name text)")
    conn.execute("INSERT INTO t VALUES (1, CAST(x'FF' AS TEXT))")
    conn.execute("INSERT INTO t VALUES (2, CAST(x'FE' AS TEXT))")
    conn.commit()
    conn.close()

    reader = open_readonly(str(path))
    try:
        rows = reader.execute("SELECT name FROM t ORDER BY id").fetchall()
    finally:
        reader.close()
    assert rows[0] != rows[1]
