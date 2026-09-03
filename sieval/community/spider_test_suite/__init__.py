r"""Vendored Spider test-suite evaluator (taoyds/test-suite-sql-eval, Apache-2.0).

Pinned at e97acc546ecbee8fa27fa8dbf025ef61493a876c. Both modules are upstream
byte-for-byte except ``exec_eval.py:11``, whose flat ``from parse import ...``
cannot resolve inside a package and became a relative import.

This is upstream's official Spider metric since October 2020, and it replaces
the column-keyed comparison rather than extending it: the fork deletes
``eval_exec_match`` and ``res_map`` from ``evaluation.py`` and rebuilds the
comparison here, on raw result sets. That is why the two vendored trees coexist
-- ``community/spider`` still supplies exact set match and the hardness buckets,
which this fork leaves in place.

The names re-exported below are the whole scoring decision: ``result_eq`` and
its three text-normalisation helpers. Keeping them importable means the verdict
for every sample is computed by upstream's own bytes, and only *execution* --
which needs a read-only connection, an authorizer and a deadline that upstream
does not have -- is reimplemented on our side.

Two upstream names are deliberately not re-exported. ``parse.postprocess`` is a
different function from ``exec_eval.postprocess`` and is unused here; only the
latter is in the scoring path, so it takes the bare name. ``exec_eval.TIMEOUT``
and ``EXEC_TMP_DIR`` belong to upstream's own subprocess runner, which we do not
call.

``eval_exec_match`` is exported for the anchor test alone. It is upstream's
whole loop -- unbounded ``sqlite3.connect`` on every ``.sqlite`` beside the
named database, with an ``assert`` on the gold -- so it pins our port's verdicts
against upstream's, but it is not what grades a run.

Five of upstream's regex literals are unescaped (``"\s"``, ``"\d"``, ``"\."``),
which Python warns about at compile time. Editing them would buy silence with
five more deviations from upstream, so the filter goes here, in our own file,
and upstream's bytes stay as they are. It covers only the compile that happens
during this import; nothing at grade time is inside it.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    from .exec_eval import (
        eval_exec_match,
        postprocess,
        replace_cur_year,
        result_eq,
    )
    from .parse import remove_distinct

__all__ = [
    "eval_exec_match",
    "postprocess",
    "remove_distinct",
    "replace_cur_year",
    "result_eq",
]
