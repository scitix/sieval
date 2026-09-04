"""Unit tests for the Spider 1.0 prompt builder.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import sqlite3

import pytest

import sieval.tasks.spider._spider_schema as schema_module
from sieval.tasks.spider._spider_schema import build_prompt


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "concert_singer.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE singer (id int, name text)")
    conn.execute("CREATE TABLE concert (cid int, singer_id int)")
    conn.executemany(
        "INSERT INTO singer VALUES (?, ?)",
        [(1, "Joe"), (2, "Ann"), (3, "Bo"), (4, "Cy")],
    )
    conn.commit()
    conn.close()
    return str(path)


def test_prompt_carries_every_create_table_statement(db):
    prompt = build_prompt(db, "How many singers do we have?")
    assert "CREATE TABLE singer" in prompt
    assert "CREATE TABLE concert" in prompt


def test_prompt_shows_exactly_three_example_rows(db):
    prompt = build_prompt(db, "How many singers do we have?")
    assert "3 example rows" in prompt
    assert "SELECT * FROM singer LIMIT 3;" in prompt
    assert "Joe" in prompt and "Ann" in prompt and "Bo" in prompt
    # The 4th row must not leak — the format is Select-3, not Select-all.
    assert "Cy" not in prompt


def test_prompt_states_the_dialect_and_the_question(db):
    prompt = build_prompt(db, "How many singers do we have?")
    assert "SQLite" in prompt
    assert "How many singers do we have?" in prompt


def test_prompt_asks_for_a_fenced_block(db):
    """The documented divergence: upstream ends in a bare `SELECT` for a
    completion model, which a chat turn cannot do."""
    prompt = build_prompt(db, "q?")
    assert "```sql" in prompt


def test_prompt_includes_column_headers_for_the_sample_rows(db):
    prompt = build_prompt(db, "q?")
    assert "id\tname" in prompt


def test_empty_table_still_renders_its_schema(tmp_path):
    """A table with no rows must not drop out of the prompt."""
    path = tmp_path / "empty.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE lonely (a int)")
    conn.commit()
    conn.close()
    prompt = build_prompt(str(path), "q?")
    assert "CREATE TABLE lonely" in prompt


def test_internal_sqlite_tables_are_excluded(tmp_path):
    """`sqlite_sequence` is an implementation detail, not part of the schema."""
    path = tmp_path / "auto.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (id integer primary key autoincrement, v text)")
    conn.execute("INSERT INTO t (v) VALUES ('x')")
    conn.commit()
    conn.close()
    prompt = build_prompt(str(path), "q?")
    assert "sqlite_sequence" not in prompt


def test_prompt_opens_the_database_read_only(db, monkeypatch):
    """Prompt building must not be a second, unhardened way into the data."""
    calls = []
    original = schema_module.open_readonly

    def spy(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(schema_module, "open_readonly", spy)
    build_prompt(db, "q?")
    # Asserting the count, not just the value: a refactor that stops routing
    # through this name would otherwise leave the test green.
    assert len(calls) == 1
    assert calls[0] == db
