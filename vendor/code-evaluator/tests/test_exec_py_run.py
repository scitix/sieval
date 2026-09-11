"""Execution-semantics tests for ``exec_py_run.execute_run``, called directly.

Covers what the isolated subprocess actually does: stdout capture, traceback
capture on an unhandled exception, stdout printed before a later raise
surviving it, timeout reporting, and statelessness across calls. Route-level
concerns -- the truncation flag, the non-python refusal, and the response
shape -- are tested through ``TestClient`` in ``test_code_runs.py`` instead:
those live in ``server.py``'s handler, not in ``execute_run`` itself.

Run from ``vendor/code-evaluator``:

    python -m pytest tests/ -q

AI-Generated Code - Claude Sonnet 5 (Anthropic)
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.exec_py_run import execute_run  # noqa: E402


def _run(code: str, **kwargs):
    """Drive one run on a fresh event loop.

    ``asyncio.run`` per call rather than an async test plugin: this suite has
    to stay runnable in the vendored service's own environment, which declares
    no test dependencies beyond pytest.
    """
    return asyncio.run(execute_run(code, **kwargs))


def test_returns_stdout():
    exit_code, out, err, timed_out, _stats = _run("print(6 * 7)")
    assert exit_code == 0
    assert out.strip() == "42"
    assert timed_out is False


def test_returns_traceback_on_raise():
    # A snippet that raises is a RESULT, not a failure of execute_run itself:
    # the call returns normally, with the traceback in stderr and a non-zero
    # exit code, so a caller (or the /code-runs route above this) can hand it
    # back to the model rather than treat it as a service error.
    exit_code, _out, err, timed_out, _stats = _run("1 / 0")
    assert exit_code != 0
    assert "ZeroDivisionError" in err
    assert timed_out is False


def test_stdout_survives_a_later_raise():
    # Output printed BEFORE the exception must come back. A model that printed
    # its intermediate result and then tripped over a typo has produced a
    # usable partial answer, and discarding it would make the tool look broken.
    exit_code, out, err, _timed_out, _stats = _run(
        "print('partial')\nraise ValueError('boom')"
    )
    assert out.strip() == "partial"
    assert "ValueError" in err
    assert exit_code != 0


def test_timeout_is_reported_not_raised():
    exit_code, out, err, timed_out, _stats = _run("while True: pass", timeout=1.0)
    assert timed_out is True
    # The process was killed rather than returning, so neither stream nor the
    # exit code can be trusted to reflect anything the snippet did.
    assert exit_code is None
    assert out == ""


def test_no_state_carries_between_calls():
    # v1 is stateless by contract: each call gets a fresh interpreter in a
    # fresh subprocess. If this ever starts passing, someone has added session
    # reuse without updating the contract.
    _run("_carried = 1")
    _exit_code, _out, err, _timed_out, _stats = _run("print(_carried)")
    assert "NameError" in err
