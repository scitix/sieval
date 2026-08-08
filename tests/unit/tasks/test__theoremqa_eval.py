"""
Unit tests for the TheoremQA numeric-answer evaluator.

Two properties, and they pull against each other, which is why both are here:
the evaluator must not execute model output, and it must still agree with the
bare ``eval`` upstream uses on every expression a real answer produces.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import builtins
import math
import time

import pytest

from sieval.tasks._theoremqa_eval import (
    _MAX_FACTORIAL,
    _MAX_NODES,
    UnsafeExpression,
    safe_eval,
)

#: Upstream's namespace: number_utils.py's module globals *plus* the real
#: builtins, which is what a bare ``eval(num)`` sees. Omitting "__builtins__"
#: is what restores the latter — Python injects them.
_UPSTREAM_GLOBALS = {
    "math": math,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "log": math.log,
    "pi": math.pi,
    "factorial": math.factorial,
    "exp": math.exp,
    "e": math.e,
    "E": 2.718,
}


# --------------------------------------------------------------- security ---

#: The payload from the bug report: a walk from a literal back to a live
#: ``open`` that a cleared ``__builtins__`` does not stop.
_ESCAPE = (
    "[c for c in ().__class__.__base__.__subclasses__() "
    "if c.__name__=='catch_warnings'][0]()._module.__builtins__['open']"
    "('{path}','w')"
)


def test_reported_escape_payload_is_refused(tmp_path):
    """The reported RCE is refused, and writes nothing."""
    target = tmp_path / "pwned"
    with pytest.raises(UnsafeExpression):
        safe_eval(_ESCAPE.format(path=target))
    assert not target.exists()


def test_cleared_builtins_eval_really_was_escapable(tmp_path):
    """The guard this replaced was not a sandbox — the reason for the change.

    Pins the premise rather than the fix: if this ever stops escaping, the
    justification for diverging from upstream has changed and should be re-read.
    """
    target = tmp_path / "pwned"
    builtins.eval(_ESCAPE.format(path=target), {"__builtins__": {}})  # noqa: S307
    assert target.exists()


@pytest.mark.parametrize(
    "expression",
    [
        "().__class__",
        "().__class__.__base__.__subclasses__()",
        "[c for c in (1, 2)][0]",
        "{c for c in (1, 2)}",
        "(c for c in (1, 2))",
        "[1, 2][0]",
        "lambda: 1",
        "__import__('os')",
        "open('x', 'w')",
        "'a string'",
        "f'{1}'",
        "1 if True else 2",
        "1 < 2",
        "True and False",
        "*[1],",
        "{'a': 1}",
        "math.__dict__",
        "math._x",
        "(x := 1)",
    ],
)
def test_execution_shapes_are_refused(expression):
    """Nothing that could reach an object or a statement is evaluable."""
    with pytest.raises((UnsafeExpression, SyntaxError)):
        safe_eval(expression)


def test_unsafe_expression_is_a_value_error():
    """Call sites catch broad `Exception`; this keeps the eval-era contract.

    A refusal has to land in the same `except` that a `SyntaxError` or a
    `NameError` from `eval` landed in, so the answer simply fails to become a
    number instead of failing the sample.
    """
    assert issubclass(UnsafeExpression, ValueError)


# ----------------------------------------------------------------- bounds ---


@pytest.mark.parametrize(
    "expression",
    [
        "factorial(2000000)",
        "9**9**9**9",
        "10**10**10",
        "(10**999)**999",
        "1 << 10**10",
    ],
)
def test_unbounded_computation_is_refused_promptly(expression):
    """Grading shares one event loop, so an unbounded answer stalls a session."""
    started = time.perf_counter()
    with pytest.raises(UnsafeExpression):
        safe_eval(expression)
    assert time.perf_counter() - started < 1.0


def test_node_count_is_bounded():
    with pytest.raises(UnsafeExpression, match="too large"):
        safe_eval("+".join(["1"] * (_MAX_NODES + 10)))


def test_factorial_is_bounded_but_usable():
    assert safe_eval(f"factorial({_MAX_FACTORIAL})") == math.factorial(_MAX_FACTORIAL)
    with pytest.raises(UnsafeExpression):
        safe_eval(f"factorial({_MAX_FACTORIAL + 1})")


# --------------------------------------------------------------- fidelity ---

#: Expressions drawn from the shapes the three call sites actually reach:
#: latex2sympy's `str()` output, and answers a model writes directly.
_FAITHFUL = [
    "1",
    "-1",
    "1.5",
    "-3.25e-7",
    "1 + 2",
    "3 - 4",
    "2*3",
    "7/2",
    "7//2",
    "7%2",
    "2**10",
    "-2**3",
    "+5",
    "(1 + 2)*3",
    "[1, 2, 3]",
    "(1, 2, 3)",
    "[0, 0]",
    "{0}",
    "{1, 2}",
    "sqrt(2)",
    "sin(0)",
    "cos(0)",
    "log(1)",
    "exp(1)",
    "pi",
    "e",
    "E",
    "factorial(5)",
    "math.sqrt(9)",
    "math.pi",
    "3.54*E - 7",
    "E/(-1 + E)",
    # builtins upstream exposed; the cleared-namespace eval lost every one
    "abs(-5)",
    "round(1.6)",
    "pow(2, 10)",
    "min(3, 1)",
    "max(3, 1)",
    "sum([1, 2, 3])",
    "int(2.9)",
    "float(3)",
    "divmod(7, 2)",
    "len([1, 2])",
    # bitwise: kept purely because upstream computes them
    "5 ^ 3",
    "5 & 3",
    "5 | 3",
    "1 << 4",
    "16 >> 2",
    "~5",
]


@pytest.mark.parametrize("expression", _FAITHFUL)
def test_matches_upstream_bare_eval(expression):
    """Same value as upstream's `eval(num)` for every evaluable shape."""
    expected = builtins.eval(expression, dict(_UPSTREAM_GLOBALS))  # noqa: S307
    assert safe_eval(expression) == expected


@pytest.mark.parametrize(
    "expression",
    ["a", "x_2", "oo", "Interval", "Integral", "unknown_name"],
)
def test_unknown_names_fail_as_upstream_does(expression):
    """Symbolic leftovers from latex2sympy: upstream raises NameError, so do we.

    The outcome, not the exception type, is the contract — both readings leave
    the answer un-numeric, which is what the call sites branch on.
    """
    with pytest.raises(UnsafeExpression, match="unknown name"):
        safe_eval(expression)
    with pytest.raises(NameError):
        builtins.eval(expression, dict(_UPSTREAM_GLOBALS))  # noqa: S307


def test_set_display_is_evaluated_not_refused():
    """Fidelity where safety does not object — the set-display defect stands.

    Refusing `{0}` would rescue a list answer latex2sympy folded into a set and
    score +2/800 on the measured run. That is a grader repair, so it belongs to
    a `_fixed` variant and deliberately does not happen under this name.
    """
    assert safe_eval("{0}") == {0}


def test_syntax_error_propagates_as_from_eval():
    with pytest.raises(SyntaxError):
        safe_eval("1 +")
