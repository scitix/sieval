"""Table construction, lifted from upstream ``lib/table.py``.

Only ``Table.create_table``'s body is vendored here, as the free function
``create_table``. That method is the sole part of upstream's ``lib/table.py`` on
the *evaluation* path: it is what built the shipped ``{split}.db`` files, so
reproducing it is what lets the engine run without them.

The rest of upstream's ``lib/table.py`` is deliberately not vendored, and the
omissions are listed so they read as choices rather than gaps:

* ``__repr__`` / ``query_str`` -- presentation only, and ``__repr__`` would pull
  ``tabulate`` in as a dependency for a method nothing calls.
* ``from_db`` / ``get_schema`` -- read a ``records`` database. Superseded by
  ``dbengine`` here, and ``from_db`` is dead upstream anyway (its first
  statement, ``schema_re.findall(table_info)[0] = [0].sql``, is a syntax-level
  impossibility that would raise on any call).
* ``generate_query`` / ``generate_queries`` -- upstream's dataset *construction*
  path (they sample random queries with ``random.choice``), not its evaluation
  path.

The ``lower=True`` default is load-bearing and must stay in step with
``dbengine.execute``: upstream lowercases string cells on INSERT here and
lowercases string condition values on SELECT there, so the comparison is
case-insensitive on both sides at once. Lowering one side only would silently
change every text condition's result.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import sqlite3


def get_id(table_id):
    """``lib/table.py::Table.get_id`` -- table id to SQL identifier."""
    return "table_{}".format(table_id.replace("-", "_"))


def create_table(conn: sqlite3.Connection, table_id, types, rows, lower=True):
    """Create and populate one table, mirroring ``Table.create_table``.

    Upstream issues ``CREATE TABLE {name} ({types})`` with ``col{i} {type}``
    pairs, then one parameterised ``INSERT`` per row, lowercasing ``str`` cells
    when ``lower``. Both statements are reproduced verbatim in shape; only the
    driver differs (``records``/SQLAlchemy named params -> ``sqlite3``
    qmark params), which does not change the SQL that reaches SQLite.

    The declared column types come from the dataset's own ``types`` column
    (``text``/``real``), which is what upstream fed in when it built the
    ``.db`` files -- verified equal to the declared schema of all 5,230 test
    and 2,716 dev tables in the shipped databases.
    """
    name = get_id(table_id)
    type_str = ", ".join("col{} {}".format(i, t) for i, t in enumerate(types))
    conn.execute("CREATE TABLE {name} ({types})".format(name=name, types=type_str))
    for row in rows:
        values = [v.lower() if lower and isinstance(v, str) else v for v in row]
        value_str = ", ".join("?" * len(values))
        conn.execute(
            "INSERT INTO {name} VALUES ({values})".format(name=name, values=value_str),
            values,
        )
