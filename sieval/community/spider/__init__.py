"""Vendored Spider 1.0 evaluator (taoyds/spider, Apache-2.0).

Pinned at b7b5b8c890cd30e35427348bb9eb8c6d1350ca7c. Both modules are upstream
byte-for-byte except ``evaluation.py:29``, whose flat ``from process_sql import
...`` cannot resolve inside a package and became a relative import.

``EMPTY_SQL`` is the one name upstream does not export: it inlines this literal
in ``evaluate()`` as the parse to score when a prediction will not parse. Lifted
here so the caller reuses upstream's fallback instead of re-typing it.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from .evaluation import (
    Evaluator,
    build_foreign_key_map_from_json,
    build_valid_col_units,
    eval_exec_match,
    rebuild_sql_col,
    rebuild_sql_val,
)
from .process_sql import Schema, get_schema, get_sql

#: Upstream's fallback parse for an unparseable prediction (``evaluation.py``,
#: inside ``evaluate()``). Scored rather than skipped, so a bad prediction is a
#: wrong answer instead of a missing one.
EMPTY_SQL = {
    "except": None,
    "from": {"conds": [], "table_units": []},
    "groupBy": [],
    "having": [],
    "intersect": None,
    "limit": None,
    "orderBy": [],
    "select": [False, []],
    "union": None,
    "where": [],
}

__all__ = [
    "EMPTY_SQL",
    "Evaluator",
    "Schema",
    "build_foreign_key_map_from_json",
    "build_valid_col_units",
    "eval_exec_match",
    "get_schema",
    "get_sql",
    "rebuild_sql_col",
    "rebuild_sql_val",
]
