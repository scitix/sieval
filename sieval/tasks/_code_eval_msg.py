"""How a task reads a verdict out of the code-eval service's ``msg`` field.

Every task that grades by executing a prediction reports a ``timeouts`` count,
and each one has to decide, from a message string, whether a failure was the
program running out of time. That decision is one contract with six readers, and
the contract belongs to the service (``vendor/code-evaluator``), not to any
benchmark — which is what puts it here rather than in one of their trees.

**The tail of a message is not vocabulary.** The service documents a small closed
set of message shapes, but each one interpolates the failing program's own output
after the colon::

    failed: subprocess timeout: 3.0s          <- a timeout
    failed: case timeout: 6.0s                <- a timeout
    failed: [TimeoutError] deadline exceeded  <- NOT a timeout: the program raised
    failed: output ['timeout'] != expect ['ok']   <- NOT a timeout: wrong answer

So the test is on the message's PREFIX. A substring test for ``"timeout"``
matches all four, and the two it should not match are both reachable: a
LiveCodeBench comparison failure prints the program's stdout verbatim, and any
Python program may raise ``TimeoutError``. Both then land under ``timeouts`` —
the one bucket that says the program ran and was merely slow, so a reader
investigates efficiency instead of the wrong answer or the exception in front of
them.

Only the diagnostic moves. ``correct`` comes from the service's boolean, never
from this string, so no headline metric depends on getting this right — which is
exactly why six copies of a substring test survived: nothing they scored was
wrong.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

#: Message prefixes that mean the SERVICE stopped the program on a clock.
#:
#: The union across executors, deliberately: each spelling is emitted only by a
#: timeout path, so a task reading a message it cannot currently produce is
#: harmless, while a task whose `lang` changes later keeps working.
#:
#: * ``failed: subprocess timeout:`` — the whole-submission wall
#:   (``exec_py_code``, ``exec_py_test``)
#: * ``failed: case timeout:`` / ``failed: compile timeout:`` — LiveCodeBench's
#:   per-case budgets (``exec_py_test``)
#: * ``failed: timeout`` — the js/ts wall (``exec_js``, ``exec_ts``); no in-tree
#:   task sends a non-default ``lang`` today, so this is the forward-looking one
#: * ``failed: build timeout`` — a compile wall (``exec_lang``)
#:
#: Not included, and each is a live false positive under a substring test:
#: ``failed: [TimeoutError] ...`` (the program raised), ``failed: compile error``,
#: and any ``failed: output ... != expect ...`` quoting program output.
CODE_EVAL_TIMEOUT_PREFIXES: tuple[str, ...] = (
    "failed: subprocess timeout",
    "failed: case timeout",
    "failed: compile timeout",
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


__all__ = ["CODE_EVAL_TIMEOUT_PREFIXES", "is_timeout_message"]
