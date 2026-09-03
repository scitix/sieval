"""Salesforce WikiSQL evaluation adaptation.

Source: https://github.com/salesforce/WikiSQL
Revision: cffb423077756d04c1bac5bcd45167c86903fbcb (archived 2025-10-06)
License: BSD-3-Clause

Upstream's evaluation surface is ``evaluate.py`` over ``lib/query.py`` +
``lib/dbengine.py``. It scores a prediction that is a **logical form** --
``{"sel": int, "agg": int, "conds": [[col, op, value], ...]}``, never SQL text --
and reports ``lf_accuracy`` (``Query.__eq__``) and ``ex_accuracy`` (both sides
run through ``DBEngine``, result lists compared). The query that runs is a
template over three integers, with condition values bound as parameters.

Local adaptations:

- ``query.py`` and ``common.py`` are byte-identical to upstream's ``lib/``
  copies but for one line: ``from lib.common import detokenize`` becomes
  ``from .common import detokenize``. Upstream's own quirks are kept, including
  ``Query.__eq__`` reading ``other.ordered`` rather than ``self.ordered`` and
  ``__hash__`` being unusable (it sorts ``__dict__`` items holding a list).
  ``count_lines`` is retained though only ``evaluate.py``'s progress bar used
  it -- dropping a helper upstream ships is as much a deviation as adding one.
- ``dbengine.py`` ports ``records``/SQLAlchemy to stdlib ``sqlite3``, builds
  tables in memory instead of opening a shipped ``.db``, and validates
  ``sel``/``agg``/``col``/``op`` before they reach the SQL text. Each is argued
  in that module's docstring; the index guard is the only behavioural change and
  it is an execution-safety stop, not a repair.
- ``table.py`` vendors only ``Table.create_table`` -- the part of upstream's
  ``lib/table.py`` on the evaluation path. That module's omissions are
  enumerated in its docstring.
- Not vendored at all: ``annotate.py`` (Stanza-based tokenisation, which
  upstream's own README declares unreproducible since Stanza's deprecation) and
  ``evaluate.py``'s ``__main__`` block, whose loop lives in the task's
  ``feedback``. ``Query.from_sequence`` / ``from_partial_sequence`` /
  ``from_tokenized_dict`` / ``from_generated_dict`` are kept in ``query.py``
  because they ship in the file, but they consume ``annotate.py``'s tokenised
  format and are unreachable from this port.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from .common import count_lines, detokenize
from .dbengine import DBEngine, InvalidQueryIndex
from .query import Query
from .table import create_table, get_id

__all__ = [
    "DBEngine",
    "InvalidQueryIndex",
    "Query",
    "count_lines",
    "create_table",
    "detokenize",
    "get_id",
]
