"""How a task reads a verdict out of the code-eval service's ``msg``.

Every task that grades by executing a prediction reports a ``timeouts`` count and
has to decide, from a message string, whether the failure was the clock. That is
one contract with six readers, and it belongs to the service
(``vendor/code-evaluator``) rather than to any benchmark — which is what puts it
here instead of in one of their trees.

**The tail of a message is not vocabulary.** The service emits a small closed set
of shapes, but each interpolates the failing program's own output after the
colon, so a substring test for ``"timeout"`` matches all four of these::

    failed: subprocess timeout: 3.0s             a timeout
    failed: [CaseTimeout]                        a timeout
    failed: [TimeoutError] deadline exceeded     NOT one: the program raised
    failed: output ['timeout'] != expect ['ok']  NOT one: wrong answer

Both false positives are reachable — a LiveCodeBench comparison failure prints
the program's stdout verbatim, and any Python program may raise ``TimeoutError``
— and both land in the one bucket that means the program ran and was merely slow.

Only the diagnostic moves: ``correct`` comes from the service's boolean, never
from this string. That is why six copies of a substring test survived — nothing
they scored was wrong.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

#: **Every prefix is lowercased**, and a caller comparing against these tuples
#: must lowercase the message first. Only ``failed: [casetimeout]`` actually
#: depends on it, so a raw-message comparison still passes against the other
#: five and silently drops that one. :func:`is_timeout_message` normalizes for
#: you; reach for a tuple directly only to keep the two apart, as
#: ``multipl_e/_base.py`` does.
#:
#: A wall the service hit BEFORE the program ran:
#:
#: * ``failed: build timeout`` — a compile wall (``exec_lang``)
#: * ``failed: compile timeout:`` — LiveCodeBench's per-case compile budget
#:   (``exec_py_test``)
CODE_EVAL_BUILD_TIMEOUT_PREFIXES: tuple[str, ...] = (
    "failed: build timeout",
    "failed: compile timeout",
)

#: A wall the service hit while the program was RUNNING. Lowercased, as above.
#:
#: * ``failed: subprocess timeout:`` — the whole-submission wall
#:   (``exec_py_code``, ``exec_py_test``)
#: * ``failed: case timeout:`` — LiveCodeBench's per-case budget
#:   (``exec_py_test``)
#: * ``failed: [CaseTimeout]`` — that same per-case wall, arriving late. A
#:   delivered signal cannot be un-delivered, so an alarm landing past
#:   ``_unsafe_execute``'s own handler is formatted by the worker's outer
#:   ``except`` as a class name. **The one entry a program can forge**: the
#:   service's class is a ``BaseException``, which is why that handler names it
#:   explicitly, but a submitted program defining ``class
#:   CaseTimeout(Exception)`` is caught by ``_unsafe_execute_fn_call``'s
#:   ``except Exception`` and formatted into the same shape. Kept regardless —
#:   dropping it loses a real wall, the substring test this replaced counted the
#:   forgery too, and what the forgery costs is one diagnostic count.
#: * ``failed: timeout`` — the run wall for a non-Python ``lang`` (``exec_js``,
#:   ``exec_ts``, ``exec_lang``)
CODE_EVAL_RUN_TIMEOUT_PREFIXES: tuple[str, ...] = (
    "failed: subprocess timeout",
    "failed: case timeout",
    "failed: [casetimeout]",
    "failed: timeout",
)

#: Every prefix that means the SERVICE stopped the program on a clock.
#:
#: A union across executors — the spelling depends on which one answered — and
#: each is emitted only by a timeout path, so carrying one a task cannot
#: currently produce is harmless. This module's readers are all Python and see
#: neither ``failed: timeout`` nor ``failed: build timeout``; MultiPL-E sees
#: only those two, since each of its 24 languages routes through ``exec_js`` /
#: ``exec_ts`` / ``exec_lang`` and none of them is Python.
#:
#: Excluded, and each a live false positive under a substring test:
#: ``failed: [TimeoutError] ...`` (the program raised) and any
#: ``failed: output ... != expect ...`` quoting program output.
#:
#: **"The service stopped the clock" is not "belongs in ``timeouts``."** This
#: answers the first question only; the counter is the caller's call. MultiPL-E
#: routes ``failed: build timeout`` to ``n_build_errors`` rather than to
#: ``timeouts``, deliberately — the run never started, and build-versus-run is
#: the split its three keys exist to carry (``multipl_e/_base.py``). That is
#: what the two groups above are for: an adopting task takes the vocabulary and
#: still owes its own bucketing, where a bare :func:`is_timeout_message` would
#: move its build walls.
CODE_EVAL_TIMEOUT_PREFIXES: tuple[str, ...] = (
    CODE_EVAL_BUILD_TIMEOUT_PREFIXES + CODE_EVAL_RUN_TIMEOUT_PREFIXES
)


def is_timeout_message(msg: str | None) -> bool:
    """Whether *msg* says the service stopped the program on a clock.

    ``None`` is False rather than an error: a null ``msg`` is absent on disk, so
    a resumed record legitimately arrives without one.
    """
    if not msg:
        return False
    return msg.strip().lower().startswith(CODE_EVAL_TIMEOUT_PREFIXES)


#: How the service reports an uncaught exception: ``failed: [<Class>] <str(e)>``.
_EXC_MSG_PREFIX = "failed: ["


def exception_class_name(msg: str | None) -> str | None:
    """The exception class the service formatted into *msg*, lowercased.

    The class is a structured SLOT; everything after it is the program's own
    output. Reading the slot is what separates ``[MemoryError] ...`` from a
    program that merely printed the word. ``None`` when *msg* is not that shape.

    Returns the name rather than comparing it, so a caller can match a *family*.
    That matters: the counters reading this want subclasses too, and the class
    the service names is not always the builtin — ``ZipImportError`` is an
    ``ImportError``, ``OutOfMemoryError`` a ``MemoryError``. Testing for the
    exact builtin would drop them, which a bare substring test did not.
    """
    if not msg:
        return None
    text = msg.strip()
    if not text.lower().startswith(_EXC_MSG_PREFIX):
        return None
    end = text.find("]", len(_EXC_MSG_PREFIX))
    if end == -1:
        return None
    return text[len(_EXC_MSG_PREFIX) : end].lower()


__all__ = [
    "CODE_EVAL_BUILD_TIMEOUT_PREFIXES",
    "CODE_EVAL_RUN_TIMEOUT_PREFIXES",
    "CODE_EVAL_TIMEOUT_PREFIXES",
    "exception_class_name",
    "is_timeout_message",
]
