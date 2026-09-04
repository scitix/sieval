"""Rajkumar et al. 2022 prompt construction for Spider 1.0.

The prompt convention is the "Create Table + Select 3" format from *Evaluating
the Text-to-SQL Capabilities of Large Language Models* (arXiv:2204.00498): each
table's ``CREATE TABLE`` statement followed by three example rows in a comment,
then the question. Spider predates LLM prompting and has no canonical prompt of
its own; this is the most-cited LLM-era convention, which is what makes
published numbers comparable.

**One deliberate divergence.** Upstream's prompt ends in a bare ``SELECT`` so a
completion model continues it. A chat turn cannot end mid-token, so the question
block closes with an instruction to answer in a fenced ``sql`` block instead. It
is the single reason a chat-mode score is not bit-comparable to the paper's
Codex figures; a completion-faithful ``_base_gen`` sibling is the place to match
those, not this file.

DDL comes from ``sqlite_master`` rather than upstream's ``tables.json`` so the
prompt shows the database as it actually is — types, constraints and all — which
is what the paper's format depends on. The database is opened through the same
hardened read-only connection the grader uses: prompt building must not become a
second, unguarded way into the data.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import sqlite3

from sieval.tasks._sqlite_exec import open_readonly

#: Number of sample rows shown per table — the "Select 3" half of the format.
N_EXAMPLE_ROWS = 3

_INSTRUCTION = (
    "-- Using valid SQLite, answer the following question "
    "for the tables provided above."
)
_CLOSER = "Return only the SQL query, in a ```sql code block."


def build_prompt(
    db_path: str, question: str, n_example_rows: int = N_EXAMPLE_ROWS
) -> str:
    """Render the schema-and-samples prompt for one question."""
    conn = open_readonly(db_path)
    try:
        tables = conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        blocks = [
            _table_block(conn, name, ddl, n_example_rows) for name, ddl in tables if ddl
        ]
    finally:
        conn.close()
    return "\n\n".join(blocks) + f"\n\n{_INSTRUCTION}\n-- {question}\n\n" + _CLOSER


def _table_block(conn: sqlite3.Connection, name: str, ddl: str, n_rows: int) -> str:
    """One table's DDL followed by its sample rows, as a SQL comment."""
    cursor = conn.execute(f"SELECT * FROM {_quote(name)} LIMIT {n_rows}")
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    rendered = "\n".join(
        "\t".join("" if value is None else str(value) for value in row) for row in rows
    )
    header = "\t".join(columns)
    return (
        f"{ddl.strip()}\n"
        f"/*\n{n_rows} example rows:\n"
        f"SELECT * FROM {name} LIMIT {n_rows};\n"
        f"{header}\n{rendered}\n*/"
    )


def _quote(identifier: str) -> str:
    """Double-quote an identifier, escaping embedded quotes."""
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'
