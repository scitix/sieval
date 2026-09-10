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

#: Message prefixes that mean the SERVICE stopped the program on a clock.
#:
#: A union across executors — the spelling depends on which one answered — and
#: each is emitted only by a timeout path, so carrying one a task cannot
#: currently produce is harmless:
#:
#: * ``failed: subprocess timeout:`` — the whole-submission wall
#:   (``exec_py_code``, ``exec_py_test``)
#: * ``failed: case timeout:`` / ``failed: compile timeout:`` — LiveCodeBench's
#:   per-case budgets (``exec_py_test``)
#: * ``failed: [CaseTimeout]`` — that same per-case wall, arriving late. A
#:   delivered signal cannot be un-delivered, so an alarm landing past
#:   ``_unsafe_execute``'s own handler is formatted by the worker's outer
#:   ``except`` as a class name. Service-internal ``BaseException``, so the name
#:   cannot come from submitted code.
#: * ``failed: timeout`` — the run wall for a non-Python ``lang`` (``exec_js``,
#:   ``exec_ts``, ``exec_lang``)
#: * ``failed: build timeout`` — a compile wall (``exec_lang``)
#:
#: The last two reach MultiPL-E, not this module's readers, which are all Python.
#:
#: Excluded, and each a live false positive under a substring test:
#: ``failed: [TimeoutError] ...`` (the program raised) and any
#: ``failed: output ... != expect ...`` quoting program output.
#:
#: **"The service stopped the clock" is not "belongs in ``timeouts``."** This
#: answers the first question only; the counter is the caller's call. MultiPL-E
#: routes ``failed: build timeout`` to ``n_build_errors`` rather than here,
#: deliberately — the run never started, and build-versus-run is the split its
#: three keys exist to carry (``multipl_e/_base.py``). A task adopting this
#: module gets the vocabulary, and still owes its own bucketing.
CODE_EVAL_TIMEOUT_PREFIXES: tuple[str, ...] = (
    "failed: subprocess timeout",
    "failed: case timeout",
    "failed: compile timeout",
    "failed: [casetimeout]",
    "failed: build timeout",
    "failed: timeout",
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
    "CODE_EVAL_TIMEOUT_PREFIXES",
    "exception_class_name",
    "is_timeout_message",
]
