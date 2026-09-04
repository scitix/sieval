"""Guards and bounds on how a text-to-SQL task opens and reads a database.

Each guard is proved by deletion: the assertion must fail if the guard is
removed, or it has tested nothing.

These live here, beside the module, rather than under `spider/` where they were
written. The contract stopped being Spider 1.0's when Spider 2.0-lite's local
engine started running through it, and a guard tested only from one benchmark's
directory reads as that benchmark's — which is how the second copy of it came to
exist in the first place. Each benchmark's own *bounds* are tested next to that
benchmark, since a deadline and a row cap are measured against a corpus.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import sqlite3

import pytest

from sieval.tasks._sqlite_exec import open_readonly, run_bounded

#: Well above anything these fixtures return; a cap under test is passed
#: explicitly, so this stands in for "not the thing being measured".
UNBOUNDED = 1_000_000


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


# --- the module must not drag any optional group in --------------------------


def test_importing_the_connection_needs_no_grading_dependency():
    """Whole reason this module exists, asserted where a reader will look.

    `test_import_discipline_family.py` enforces it from each task's side, in a
    fresh interpreter; this states it locally, so moving `open_readonly` into a
    grader module fails a test in the file it was moved out of. Two benchmarks
    now depend on it: Spider 1.0's graders pull nltk and sqlparse, and Spider
    2.0's vendored evaluator pulls google.cloud — the prompt paths need none of
    the three.

    Read off the import statements rather than the source text: the docstring
    names those packages on purpose, to say why they must stay out.
    """
    import ast
    import inspect

    import sieval.tasks._sqlite_exec as module

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
            run_bounded(conn, runaway, deadline_s=0.5, max_rows=UNBOUNDED)
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
                max_rows=UNBOUNDED,
            )
        # Same connection, deadline long since past: must still run.
        assert run_bounded(conn, "SELECT count(*) FROM singer", 5.0, UNBOUNDED) == (
            ["count(*)"],
            [(2,)],
        )
    finally:
        conn.close()


# --- the column names, which is what the shared reader added -----------------


def test_column_names_come_back_with_the_rows(db):
    """A frame-building caller needs the header, and from the same cursor.

    Spider 2.0's engine builds a `DataFrame` from this, so taking the names off
    a second `PRAGMA` or off the SQL text would let the header and the rows
    describe different statements. Fails if `run_bounded` returns rows alone.
    """
    conn = open_readonly(str(db))
    try:
        columns, rows = run_bounded(
            conn, "SELECT name AS who, id FROM singer ORDER BY id", 5.0, UNBOUNDED
        )
    finally:
        conn.close()
    # The alias, not the underlying column: it is the cursor's own description.
    assert columns == ["who", "id"]
    assert rows == [("Joe", 1), ("Ann", 2)]


def test_a_statement_with_no_result_set_still_returns_a_header(db):
    """`cursor.description` is None for a non-query; the `or []` covers it.

    A model can write one — the extractor hands over whatever came back, not
    only SELECTs — and the caller unpacks two values unconditionally, so
    returning `None` here would be a TypeError at the call site rather than an
    empty frame. The pragma is one the connection is already in (read-only sets
    `query_only` for us), chosen because everything that would *change* the
    database is refused before `description` is reached at all.
    """
    conn = open_readonly(str(db))
    try:
        assert run_bounded(conn, "PRAGMA query_only = 1", 5.0, UNBOUNDED) == ([], [])
    finally:
        conn.close()


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
