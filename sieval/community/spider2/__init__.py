"""Vendored Spider 2.0-lite evaluator (xlang-ai/Spider2, MIT).

Pinned at cafb867313aab4e674652054198f383cf4018943. ``evaluate.py`` is upstream
byte-for-byte — no deviation at all, unlike the Spider 1.0 mirror, because its
imports already resolve inside a package.

**Take the comparison from here, not from ``evaluate_utils.py``.** The repo ships
two ``compare_pandas_table`` copies and they no longer agree: ``evaluate.py``'s
is the live evaluator's and carries the 2025-10-29 accuracy fix (``normalize``
mapping NaN to 0, an early ``break`` once a gold column finds no match, an
empty-``multi_gold`` guard); ``evaluate_utils`` has none of the three. Same
name, same directory, different verdicts — the stale one is the easy mistake.

Upstream's current lite evaluator ships **no Snowflake branch**:
``evaluate_single_sql_instance`` routes ``bq``/``ga`` and ``local``, and
everything else falls through to "Unsupported instance id prefix", leaving 207
of 547 unscoreable despite gold existing for all of them. sieval's Snowflake
execution is therefore first-party, in ``sieval.tasks._spider2_backends``.

**Importing it has two side effects, and this module undoes both.** At module
scope ``evaluate.py`` — a command-line entry point as well as a library — does::

    sys.stdout = TeeOutput("log.txt")
    sys.stderr = sys.stdout

which truncates a ``log.txt`` that is not ours and redirects the whole process's
output for the rest of the run, swallowing every other task's logs in a shared
session. Fixing it upstream-side would mean editing a byte-identical file, so
the fix is here, at the only door in: the streams are saved and put back, the
handle upstream opened is closed, and an existing ``log.txt`` is moved aside and
back.

This is the one place it can go — ``evaluate`` is a submodule, so every route to
it runs this file first. When editing: whatever imports ``.evaluate`` must go
through :func:`_import_evaluate`.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import fcntl
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType

#: Upstream's hard-coded log name, resolved against whatever the current
#: directory happens to be at first import.
_LOG = Path("log.txt")

#: One lock per user, so two processes importing this package at the same moment
#: — the offload pool spawns workers that each import it on their first grade —
#: are not both moving one `log.txt` around. Without it the interleaving deletes
#: the file the other just restored. Kept in the temp directory rather than
#: beside `log.txt`: a lock file there would be litter in exactly the directory
#: this guard exists to leave untouched.
_LOCK = Path(tempfile.gettempdir()) / f"sieval-spider2-import-{os.getuid()}.lock"


def _acquire() -> int | None:
    """An exclusive lock descriptor, or ``None`` if one could not be taken.

    ``O_NOFOLLOW`` because the path is in a world-writable directory: a symlink
    planted there would otherwise redirect the open. Failing to lock degrades to
    an unserialised move rather than to a failed import — the race it guards is
    narrow, and refusing to import over it would be the larger breakage.
    """
    try:
        descriptor = os.open(_LOCK, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    except OSError:
        return None
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    return descriptor


def _import_evaluate() -> ModuleType:
    """Import upstream's evaluator without letting it keep what it takes."""
    lock = _acquire()
    stash: str | None = None
    try:
        if _LOG.exists():
            handle, stash = tempfile.mkstemp(dir=_LOG.parent, prefix="log.txt.sieval-")
            os.close(handle)
            os.replace(_LOG, stash)
        streams = (sys.stdout, sys.stderr)
        try:
            from . import evaluate

            return evaluate
        finally:
            hijacked = sys.stdout
            sys.stdout, sys.stderr = streams
            # `is not` rather than a truth test: an import that failed before
            # upstream's two assignments leaves the real stdout in place, and
            # closing *that* would take the process's output with it.
            if hijacked is not streams[0]:
                file = getattr(hijacked, "file", None)
                if file is not None:
                    file.close()
                # Nothing prints between the assignment and here — the rest of
                # `evaluate.py` is definitions — so this file is empty.
                _LOG.unlink(missing_ok=True)
            if stash is not None:
                os.replace(stash, _LOG)
    finally:
        if lock is not None:
            os.close(lock)


_evaluate = _import_evaluate()

compare_multi_pandas_table = _evaluate.compare_multi_pandas_table
compare_pandas_table = _evaluate.compare_pandas_table
extract_sql_query = _evaluate.extract_sql_query
load_gold_csv = _evaluate.load_gold_csv
resolve_gold_paths = _evaluate.resolve_gold_paths

__all__ = [
    "compare_multi_pandas_table",
    "compare_pandas_table",
    "extract_sql_query",
    "load_gold_csv",
    "resolve_gold_paths",
]
