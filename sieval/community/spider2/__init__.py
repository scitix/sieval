"""Vendored Spider 2.0-lite evaluator (xlang-ai/Spider2, MIT).

Pinned at cafb867313aab4e674652054198f383cf4018943. ``evaluate.py`` is upstream
byte-for-byte — no deviation at all, unlike the Spider 1.0 mirror, because its
imports already resolve inside a package.

**Take the comparison from here, not from ``evaluate_utils.py``.** The repo ships
two copies of ``compare_pandas_table`` and they no longer agree: the one in
``evaluate.py`` is the live evaluator's and carries the 2025-10-29 accuracy fix
(a ``normalize`` that maps NaN to 0 before comparing, an early ``break`` once a
gold column finds no match, and an empty-``multi_gold`` guard). ``evaluate_utils``
has none of the three. Both are named the same and sit in the same directory, so
the stale one is the easy mistake — and it computes different verdicts.

What upstream's current lite evaluator does **not** ship is a Snowflake branch:
``evaluate_single_sql_instance`` routes ``bq``/``ga`` to BigQuery and ``local`` to
SQLite, and everything else falls through to "Unsupported instance id prefix".
That leaves 207 of the 547 instances unscoreable by upstream even though gold
results exist for all 547. sieval's Snowflake execution is therefore first-party
and lives in ``sieval.tasks._spider2_backends``, not here — there is no upstream
lite implementation to mirror.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from .evaluate import (
    compare_multi_pandas_table,
    compare_pandas_table,
    extract_sql_query,
    load_gold_csv,
    resolve_gold_paths,
)

__all__ = [
    "compare_multi_pandas_table",
    "compare_pandas_table",
    "extract_sql_query",
    "load_gold_csv",
    "resolve_gold_paths",
]
