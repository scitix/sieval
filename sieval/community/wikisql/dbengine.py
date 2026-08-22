"""Execution engine, ported from upstream ``lib/dbengine.py``.

``DBEngine.execute`` is reproduced statement for statement: the same
``sqlite_master`` schema read, the same ``schema_re``/``num_re`` patterns, the
same ``babel`` numeric coercion for ``real`` columns, the same
``SELECT {agg}(col{sel}) AS result FROM {table} WHERE ...`` template, and the
same ``lower=True`` default. Three things differ, all of them stated here.

**1. Driver: ``records`` -> stdlib ``sqlite3``.** Upstream's ``records`` is an
archived SQLAlchemy wrapper (last release 2019) and is the only reason upstream
needs a database *file*. ``sqlite3`` speaks the same ``:name`` parameter style,
so the SQL text reaching SQLite is unchanged; only the row accessor differs
(``o.result`` -> ``row[0]``, over the same single-column ``AS result``
projection).

**2. Tables are built in memory, not read from a shipped ``.db``.** Upstream
distributes ``{split}.db`` inside ``data.tar.bz2`` and points ``DBEngine`` at
the file. This port rebuilds each table from the dataset's own
``types``/``rows`` via ``table.create_table`` -- the same code path that
produced those files. Verified equal, not assumed: every one of the 15,878 test
gold queries returns byte-identical results from the rebuild and from upstream's
shipped ``test.db`` (0 mismatches, 0 exceptions), the declared schema matches
the ``types`` column for all 5,230 test and 2,716 dev tables, and replaying
upstream's own ``test/example.pred.dev.jsonl.bz2`` through both paths yields the
same ``ex_accuracy``/``lf_accuracy`` to six decimals. It drops ~120 MB of binary
SQLite from the download and leaves the engine with no filesystem reach at all.

**3. Indices are validated before they are formatted into SQL.** This is the one
behavioural divergence, and it is an execution-safety stop rather than a repair.
Upstream interpolates ``select_index`` and ``col_index`` directly into the query
text and indexes ``agg_ops``/``cond_ops`` with whatever it is given. That is safe
for upstream, whose predictions come from a decoder over a closed output space,
and unsafe here, where they come from a chat model that can emit anything:

* a non-``int`` ``sel``/``col`` is string-formatted into the SQL text, so it is
  an injection point (``{"sel": "0 FROM x; DROP TABLE y; --"}``);
* ``bool`` formats as ``colTrue``, not ``col1``, so it is not the integer it
  looks like;
* a *negative* index is the quiet one: Python wraps, so ``agg_ops[-1]`` is
  ``AVG`` and ``cond_ops[-1]`` is ``OP``. That path raises nothing and scores a
  query the model did not ask for.

Rejected indices raise, which lands in the caller's guard -- the same
``except Exception`` upstream's ``evaluate.py`` already wraps every prediction
in, where an unexecutable prediction becomes a non-matching result and scores
wrong. So a malformed prediction is graded exactly as upstream grades it, and no
injection is reachable. Condition *values* are untouched: upstream already binds
them as parameters, never interpolates them, so they need no guard and get none.

``op == 3`` (``OP``) is deliberately *not* rejected. It is in upstream's
``cond_ops``, appears in no gold row of either split, and renders as invalid SQL
-- so SQLite raises and the caller's guard scores it wrong. That is upstream's
behaviour, and preserving it keeps a model that emits ``OP`` distinguishable
from one that emits an out-of-range integer.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import re
import sqlite3
from collections.abc import Sequence

from babel.numbers import NumberFormatError, parse_decimal

from .query import Query
from .table import create_table, get_id

schema_re = re.compile(r"\((.+)\)")
num_re = re.compile(r"[-+]?\d*\.\d+|\d+")


class InvalidQueryIndex(ValueError):
    """A ``sel``/``agg``/``col``/``op`` index that must not reach the SQL text.

    A distinct type so a caller can tell a rejected *prediction* apart from a
    genuine engine fault, and so tests can assert the guard fires rather than
    matching on a shared message. Callers still catch broadly -- upstream's
    ``evaluate.py`` treats every prediction-side exception alike.
    """


def _checked_index(value, upper: int, what: str) -> int:
    """Return *value* as an ``int`` in ``[0, upper)``, or raise.

    ``bool`` is rejected explicitly: it passes ``isinstance(v, int)`` but
    formats as ``colTrue``/``colFalse``, so it is not the index it appears to
    be.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidQueryIndex(
            f"{what} must be an int, got {type(value).__name__}: {value!r}"
        )
    if not 0 <= value < upper:
        raise InvalidQueryIndex(f"{what} out of range: {value} not in [0, {upper})")
    return value


class DBEngine:
    """Upstream's ``DBEngine``, over an in-memory SQLite connection.

    Construct with :meth:`from_table`. Closing is the caller's job; the class is
    a context manager so a task can scope one engine to one sample.
    """

    def __init__(self, conn: sqlite3.Connection, n_columns: int):
        self.conn = conn
        self.n_columns = n_columns

    @classmethod
    def from_table(
        cls, table_id: str, types: Sequence[str], rows: Sequence[Sequence]
    ) -> "DBEngine":
        """Build a one-table in-memory database, as upstream's ``.db`` was."""
        conn = sqlite3.connect(":memory:")
        create_table(conn, table_id, types, rows)
        return cls(conn, len(types))

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "DBEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def execute_query(self, table_id, query, *args, **kwargs):
        return self.execute(
            table_id,
            query.sel_index,
            query.agg_index,
            query.conditions,
            *args,
            **kwargs,
        )

    def execute(
        self, table_id, select_index, aggregation_index, conditions, lower=True
    ):
        if not table_id.startswith("table"):
            table_id = get_id(table_id)
        row = self.conn.execute(
            "SELECT sql from sqlite_master WHERE tbl_name = :name",
            {"name": table_id},
        ).fetchone()
        if row is None:
            raise LookupError(f"no such table: {table_id}")
        table_info = row[0]
        schema_str = schema_re.findall(table_info)[0]
        schema = {}
        for tup in schema_str.split(", "):
            c, t = tup.split()
            schema[c] = t

        # The guard described in the module docstring. Everything below this
        # point is upstream's, unchanged.
        select_index = _checked_index(select_index, self.n_columns, "sel")
        aggregation_index = _checked_index(aggregation_index, len(Query.agg_ops), "agg")

        select = "col{}".format(select_index)
        agg = Query.agg_ops[aggregation_index]
        if agg:
            select = "{}({})".format(agg, select)
        where_clause = []
        where_map = {}
        for col_index, op, val in conditions:
            col_index = _checked_index(col_index, self.n_columns, "cond column")
            op = _checked_index(op, len(Query.cond_ops), "cond operator")
            if lower and isinstance(val, str):
                val = val.lower()
            if schema["col{}".format(col_index)] == "real" and not isinstance(
                val, (int, float)
            ):
                try:
                    val = float(parse_decimal(val))
                except NumberFormatError:
                    val = float(num_re.findall(val)[0])
            where_clause.append(
                "col{} {} :col{}".format(col_index, Query.cond_ops[op], col_index)
            )
            where_map["col{}".format(col_index)] = val
        where_str = ""
        if where_clause:
            where_str = "WHERE " + " AND ".join(where_clause)
        query = "SELECT {} AS result FROM {} {}".format(select, table_id, where_str)
        out = self.conn.execute(query, where_map)
        return [o[0] for o in out]
