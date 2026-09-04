"""Spider's post-2020 official metric: test-suite execution accuracy.

Upstream replaced execution accuracy in October 2020
(``taoyds/test-suite-sql-eval``) because the pre-2020 metric it is built on fails
in both directions, structurally rather than incidentally:

* **It scores through a parser.** ``eval_exec_match`` keys each result column by
  the *parsed* select unit, so a prediction the parser rejects is compared as an
  empty projection and scores 0 no matter what SQLite returned. That parser is a
  hand-written tokeniser over Spider's own gold dialect, and on the pinned dev
  data it accepts 100% of golds and 30-59% of model predictions.
* **One database cannot separate the queries.** A prediction that happens to
  agree with the gold on the shipped rows is indistinguishable from one that is
  right, so the metric over-credits.

This module implements the replacement. Two changes, both upstream's: **raw
result sets** compared by ``result_eq`` under bag semantics, searching column
permutations, with row order significant only when the gold has an ``ORDER BY``
and nothing parsed at all; and **a distilled test suite** — every query runs
against ~25-660 databases generated to distinguish neighbouring queries, and the
prediction must agree with the gold on *all* of them, which is what turns "agrees
here" into "equivalent".

The comparison is upstream's own bytes -- ``result_eq``, ``postprocess``,
``remove_distinct`` and ``replace_cur_year`` are imported, not reimplemented.
What this module supplies is the part that has to differ: **execution**, through
the same ``_sqlite_exec`` guards and ``_spider_sqlite`` bounds the pre-2020 path
runs under, so the deadline and the row cap have one implementation rather than
two that drift.
Upstream opens an unbounded read-write ``sqlite3.connect`` per query; here every
one of the ~39 databases per sample is opened ``mode=ro&immutable=1`` with
``ATTACH``/``DETACH`` denied.

Upstream's two flags are pinned to their CLI defaults, which is what its
published numbers use: ``plug_value=False`` (a predicted literal is the model's
responsibility) and ``keep_distinct=False`` (``DISTINCT`` stripped from both
sides). Pinned rather than exposed, because a run configured either way is not
comparable to a published Spider score.

**The unseeded RNG in upstream's ``get_constraint_permutation`` cannot move a
verdict.** It samples 20 random rows to shrink the column-permutation search
space, and every permutation it discards is discarded on a *sound* necessary
condition: a value present in one result column and absent from the other rules
that pairing out for any sample. So the true permutation, if one exists, is never
discarded, and if none exists the search returns ``False`` whatever it sampled.
Different draws change how long the search runs, never what it concludes — which
is why upstream's bytes are left unseeded rather than patched for
reproducibility. All three parts of that argument are pinned in
``tests/unit/community/test_spider_test_suite.py``.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import os

from sieval.community.spider_test_suite import (
    postprocess,
    remove_distinct,
    replace_cur_year,
    result_eq,
)
from sieval.tasks._sqlite_exec import open_readonly, run_bounded

from ._spider_exec import grade_one as grade_reference
from ._spider_sqlite import DEFAULT_DEADLINE_S, DEFAULT_MAX_ROWS

#: Upstream's ``--plug_value`` default: compare the prediction's own literals
#: rather than substituting the gold's.
PLUG_VALUE = False
#: Upstream's ``--keep_distinct`` default: strip ``DISTINCT`` from both sides.
KEEP_DISTINCT = False


def test_suite_databases(test_suite_db_dir: str, db_id: str) -> list[str]:
    """Every distilled database for *db_id*, in a deterministic order.

    Upstream globs ``os.listdir`` of the directory holding the named database and
    keeps basenames containing ``'.sqlite'`` -- a substring test, not a suffix
    one, and non-recursive. Both quirks are kept: the filter is what excludes the
    17 stray ``.txt``/``.csv`` files the archive ships under ``car_1/``, and
    widening it would silently add files to a comparison upstream does not make.

    The sort is ours. Upstream's directory order is filesystem-dependent, and
    because the loop below stops at the first database the prediction fails on,
    that order decides *which* failure is reported. It cannot change a verdict --
    the verdict is an AND over the whole set -- so sorting buys a reproducible
    diagnostic for free.
    """
    directory = os.path.join(test_suite_db_dir, db_id)
    return sorted(
        os.path.join(directory, basename)
        for basename in os.listdir(directory)
        if ".sqlite" in basename
    )


def test_suite_match(
    test_suite_db_dir: str,
    db_id: str,
    pred_sql: str,
    gold_sql: str,
    deadline_s: float = DEFAULT_DEADLINE_S,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> tuple[bool, str | None]:
    """Upstream ``eval_exec_match``, with execution hardened.

    Returns the verdict and, when the prediction would not run, the reason and
    the database it first failed on. Mirrors upstream statement for statement:
    post-process both sides, strip ``DISTINCT``, decide row-order significance
    from the *gold*, then require agreement on every database, stopping at the
    first disagreement.

    A gold that will not run **raises**, where upstream ``assert``s. Same posture
    as the pre-2020 path: a gold we cannot execute is our bug -- a missing
    database, a stale archive, a bound of ours that binds -- and scoring the
    sample would report the model wrong for it. Upstream reaches its assert only
    on databases it visits before the prediction fails; this keeps that order.

    A **blank** prediction is decided here rather than executed, the one place
    with no upstream behaviour to preserve: upstream reads predictions from a
    file in which a blank line is a *session boundary*, so it cannot receive one.
    Both available readings are wrong -- passing it through would score it
    against ``[]``, so an unextracted answer would come out **correct** against
    any gold returning no rows, and letting ``remove_distinct``'s ``IndexError``
    propagate would fail the sample rather than score it. So it is ``False`` with
    a reason, which is what the pre-2020 path already does via its empty parse.
    """
    if not pred_sql.strip():
        return False, "blank prediction"
    if not gold_sql.strip():
        # Unreachable on the pinned data -- all 1,034 dev golds parse -- but the
        # posture above has to hold for this gold too, and sqlparse's IndexError
        # names neither the sample nor the reason.
        raise ValueError(f"Spider gold SQL is blank for {db_id!r}")

    pred, gold = postprocess(pred_sql), postprocess(gold_sql)
    if not KEEP_DISTINCT:
        pred, gold = remove_distinct(pred), remove_distinct(gold)

    # Upstream's rule, comment included: an ORDER BY in the GOLD makes row order
    # significant. It is read off the gold alone, so a prediction cannot opt into
    # a laxer comparison by omitting one.
    order_matters = "order by" in gold.lower()

    # `replace_cur_year` is applied inside upstream's `exec_on_db_`, i.e. to both
    # sides and after the two rewrites above. Hoisted out of the per-database
    # loop here because it is a pure string substitution with no dependence on
    # which database is open, and ~39 identical calls per sample is the kind of
    # waste that becomes visible at 80k executions.
    pred, gold = replace_cur_year(pred), replace_cur_year(gold)

    for db_path in test_suite_databases(test_suite_db_dir, db_id):
        conn = open_readonly(db_path)
        try:
            try:
                # `result_eq` compares row tuples; the column names the shared
                # reader also returns belong to a caller building a frame.
                _, gold_rows = run_bounded(conn, gold, deadline_s, max_rows)
            except Exception as exc:
                raise ValueError(
                    f"Spider gold SQL failed on {os.path.basename(db_path)!r} "
                    f"for {db_id!r}: {type(exc).__name__}: {exc}"
                ) from exc
            try:
                _, pred_rows = run_bounded(conn, pred, deadline_s, max_rows)
            except Exception as exc:
                name = os.path.basename(db_path)
                return False, f"{type(exc).__name__} on {name}: {exc}"
        finally:
            conn.close()

        if not result_eq(gold_rows, pred_rows, order_matters=order_matters):
            return False, None

    return True, None


def grade_one(
    db_path: str,
    tables_json_path: str,
    test_suite_db_dir: str,
    db_id: str,
    pred_sql: str,
    gold_sql: str,
    deadline_s: float = DEFAULT_DEADLINE_S,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> dict:
    """Score one prediction on all three metrics in one worker dispatch.

    Module-level and picklable by name so ``run_cpu_bound`` can ship it to a
    worker process. Composed here rather than dispatched twice because the two
    metrics are independent but the *sample* is not: two dispatches would pickle
    the same arguments twice and pay two process hand-offs for one row, and a
    timeout would then have to be attributed to one of them.

    Returns ``_spider_exec.grade_one``'s keys -- ``exact_match``, ``execution``,
    ``hardness``, ``error`` -- plus ``test_suite`` and ``test_suite_error``. The
    two ``error`` keys are kept separate on purpose: they are the same prediction
    refusing to run against *different* databases, and a prediction can fail one
    while running fine on the other.
    """
    graded = grade_reference(
        db_path,
        tables_json_path,
        db_id,
        pred_sql,
        gold_sql,
        deadline_s=deadline_s,
        max_rows=max_rows,
    )
    test_suite, error = test_suite_match(
        test_suite_db_dir,
        db_id,
        pred_sql,
        gold_sql,
        deadline_s=deadline_s,
        max_rows=max_rows,
    )
    return graded | {"test_suite": test_suite, "test_suite_error": error}
