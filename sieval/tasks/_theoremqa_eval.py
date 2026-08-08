"""Evaluate a numeric answer expression without running it.

TheoremQA's official grader turns an extracted answer into a number with a bare
``eval(num)``. The answer is model output, so that is arbitrary code execution;
the usual mitigation -- ``eval(num, {"__builtins__": {}})`` -- does not close it,
because clearing the builtins does not remove *attribute traversal*, and the
standard ``catch_warnings`` route walks from a literal back to a live ``open``::

    [
        c
        for c in ().__class__.__base__.__subclasses__()
        if c.__name__ == "catch_warnings"
    ][0]()._module.__builtins__["open"]("x", "w")

This module replaces the interpreter with a tree walk. :func:`safe_eval` parses
the expression to an AST and evaluates an explicit allowlist of node types
itself; anything outside that list raises :exc:`UnsafeExpression`. Nothing is
executed, so there is no namespace to escape from and no sandbox to audit -- the
payload above dies at its first ``Attribute`` node, before any object is
reached.

Chosen over the ``parse_expr`` approach used for the UGMathBench grader
(:mod:`sieval.community.ugmathbench`) because that one *is* an execution path:
``parse_expr`` compiles and evaluates its transformed source, which is why it
needs a cleared namespace, a quote screen, and a power-tower pre-parse to be
safe. Those guards are load-bearing there because the grader wants a *symbolic*
comparison. Here the three call sites want a *number* out of a Python-ish
expression, so a tree walk answers the question directly and has no guard
surface to get wrong.

**Fidelity is not traded away for it.** The dialect is upstream's numeric
subset -- number literals, the arithmetic and bitwise operators, list / tuple /
set displays, calls to the allowlisted names in :data:`_NAMES`, and
``math.<name>`` reads. The builtins upstream's ``eval`` exposed (``abs``,
``round``, ``pow``, ``min``, ``max``, ``sum``, ``int``, ``float``, ``complex``,
``divmod``, ``len``) are back, as *values* in a namespace only a validated tree
can reach. Exposing them is safe here in a way it is not under ``eval``: the
danger was never ``abs``, it was the traversal that reaches everything else, and
no allowlisted node can express one. The cleared-builtins ``eval`` this replaces
lost all of them to a ``NameError``, so this is the *more* faithful of the two.

Where fidelity and safety did not conflict, fidelity won. Set displays and
bitwise operators are supported for no reason other than that upstream computes
them -- refusing set displays would score **2 more of 800 samples correct**, by
rescuing a list answer ``latex2sympy`` had folded into a set, and that is a
grader repair rather than a safety measure, so it is left to a ``_fixed``
variant and deliberately not taken here.

**Measured.** Replaying a stored 800-sample TheoremQA run (Qwen2.5-72B, 5-shot)
through all three readings -- upstream's bare ``eval``, the cleared-builtins
``eval``, and this walk -- reaches 706 expressions at the three call sites. All
706 agree: 586 evaluate to an identical value, 120 fail under both, and **zero**
are refused here but evaluated by upstream. Task accuracy is 44.625 under all
three. Closing the execution path cost nothing measurable.

Every evaluation is additionally bounded (:data:`_MAX_NODES`,
:data:`_MAX_RESULT_BITS`, :data:`_MAX_FACTORIAL`). Upstream has no bound at all:
``factorial(2000000)`` grades one sample for ~16 s, and grading shares one event
loop with every other task in the session, so an unbounded answer is a
session-wide stall rather than a slow sample. None of the three bounds is
reached by any of the 706 expressions above -- they are a ceiling on what an
answer may ask for, not a filter the benchmark runs into.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import ast
import math
import operator
from collections.abc import Callable
from math import cos, e, exp, factorial, log, pi, sin, sqrt

#: Upstream's ``E``. ``number_utils.py`` defines it as this truncated literal
#: rather than :data:`math.e`, and an answer written ``E`` is compared against
#: the truncated value there, so it is reproduced exactly.
E = 2.718

#: Largest argument :func:`factorial` will accept. Upstream applies no bound,
#: and ``factorial`` is in scope for every graded answer. The largest factorial
#: any TheoremQA reference needs is far below this; past it the call is refused
#: rather than allowed to run for minutes.
_MAX_FACTORIAL = 1000

#: Ceiling on the bit length of an integer power. ``a ** b`` is computed
#: eagerly, so ``9**9**9`` asks for a 370-million-digit integer and never
#: returns. The bound is on the *result*, estimated before computing it, so it
#: composes: a nested tower is refused at whichever level first crosses it.
#: ~3000 decimal digits, far above any answer this benchmark asks for.
_MAX_RESULT_BITS = 10_000

#: Ceiling on AST node count, bounding both parse depth and the number of
#: operations a single answer can ask for.
_MAX_NODES = 500


class UnsafeExpression(ValueError):
    """The expression is outside the evaluable subset, or exceeds a bound.

    A subclass of :exc:`ValueError` so the call sites' existing broad
    ``except Exception`` treats a refusal exactly as they already treat a
    ``SyntaxError`` or a ``NameError`` from ``eval``: the answer does not
    become a number, and the sample grades wrong.
    """


def _guarded_pow(base, power):
    """``base ** power``, refused when the result cannot be bounded."""
    # bits(base**power) == power * log2(|base|); checked before computing.
    if (
        isinstance(power, int)
        and isinstance(base, int)
        and base not in (0, 1, -1)
        and power > 0
        and power * base.bit_length() > _MAX_RESULT_BITS
    ):
        raise UnsafeExpression("integer power exceeds the size bound")
    if isinstance(power, float) and abs(power) > _MAX_RESULT_BITS:
        raise UnsafeExpression("float power exceeds the size bound")
    try:
        return operator.pow(base, power)
    except (OverflowError, MemoryError) as exc:
        raise UnsafeExpression(f"power overflowed: {exc}") from exc


def _guarded_factorial(value):
    if not isinstance(value, int) or value > _MAX_FACTORIAL:
        raise UnsafeExpression(f"factorial argument out of range: {value!r}")
    return factorial(value)


#: Names resolvable as bare identifiers. Upstream's ``eval`` saw
#: ``number_utils.py``'s module globals plus the builtins; this reproduces the
#: numeric part of both. ``factorial`` is wrapped to carry a bound.
_NAMES: dict[str, object] = {
    "math": math,
    "sqrt": sqrt,
    "sin": sin,
    "cos": cos,
    "log": log,
    "pi": pi,
    "factorial": _guarded_factorial,
    "exp": exp,
    "e": e,
    "E": E,
    # Builtins upstream exposed. Pure numeric functions: none of them can reach
    # an object the allowlisted nodes cannot already name.
    "abs": abs,
    "round": round,
    "pow": _guarded_pow,
    "min": min,
    "max": max,
    "sum": sum,
    "int": int,
    "float": float,
    "complex": complex,
    "divmod": divmod,
    "len": len,
    "True": True,
    "False": False,
    "None": None,
}


def _guarded_lshift(value, amount):
    """``value << amount``, refused when the result cannot be bounded."""
    if not isinstance(amount, int) or amount > _MAX_RESULT_BITS:
        raise UnsafeExpression(f"shift amount out of range: {amount!r}")
    return operator.lshift(value, amount)


_BIN_OPS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _guarded_pow,
    # Bitwise operators carry no escape and no size explosion (except a shift,
    # which is bounded). They are here for *fidelity*, not utility: upstream's
    # eval computes them, so omitting them would be a divergence that safety
    # does not ask for. `^` is the only one the measured corpus reaches, where
    # it is a model writing exponentiation in ASCII.
    ast.BitXor: operator.xor,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.LShift: _guarded_lshift,
    ast.RShift: operator.rshift,
}

_UNARY_OPS: dict[type, Callable] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
}


def _eval_node(node: ast.AST):
    if isinstance(node, ast.Constant):
        # Numbers and booleans only. A string constant is refused: it is the one
        # literal that can carry code into a function that re-parses its
        # argument, and no numeric answer needs one.
        if isinstance(node.value, (int, float, complex, bool)) or node.value is None:
            return node.value
        raise UnsafeExpression(f"literal of type {type(node.value).__name__}")

    if isinstance(node, ast.Name):
        try:
            return _NAMES[node.id]
        except KeyError:
            # Same outcome as upstream's NameError for an unknown identifier.
            raise UnsafeExpression(f"unknown name {node.id!r}") from None

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise UnsafeExpression(f"operator {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise UnsafeExpression(f"unary operator {type(node.op).__name__}")
        return op(_eval_node(node.operand))

    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(item) for item in node.elts)

    if isinstance(node, ast.List):
        return [_eval_node(item) for item in node.elts]

    if isinstance(node, ast.Set):
        # A set *display* only -- `ast.SetComp` stays refused, and it is the
        # comprehension, not the display, that the escape payload needs. Kept
        # for fidelity: upstream evaluates `{0}` to a set, and although a set
        # can never grade correct here (`compare_two_list` requires a `list`,
        # and `number_it` cannot floatify one), refusing it would rescue an
        # answer upstream loses. That is a grader repair, not a safety measure,
        # so it belongs to a `_fixed` variant rather than to this name.
        return {_eval_node(item) for item in node.elts}

    if isinstance(node, ast.Attribute):
        # Only ``math.<public name>``. Restricting the base to the literal name
        # ``math`` and rejecting a leading underscore is what keeps this from
        # being a traversal: ``math``'s public surface is functions and floats,
        # and the dunder path that reaches anything else is spelled with one.
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "math"
            and not node.attr.startswith("_")
            and hasattr(math, node.attr)
        ):
            return getattr(math, node.attr)
        raise UnsafeExpression("attribute access")

    if isinstance(node, ast.Call):
        if node.keywords:
            raise UnsafeExpression("keyword arguments")
        func = _eval_node(node.func)
        if not callable(func):
            raise UnsafeExpression("call of a non-callable")
        args = [_eval_node(arg) for arg in node.args]
        try:
            return func(*args)
        except UnsafeExpression:
            raise
        except (OverflowError, MemoryError) as exc:
            raise UnsafeExpression(f"call overflowed: {exc}") from exc

    raise UnsafeExpression(f"node {type(node).__name__}")


def safe_eval(expression: str):
    """Evaluate *expression* as a numeric expression, executing nothing.

    Drop-in for the ``eval(expression, _EVAL_GLOBALS)`` this replaces: it
    returns the same value for every expression inside the numeric subset, and
    raises for everything else, which the call sites already treat as "not a
    number".

    :raises UnsafeExpression: the expression is outside the subset or exceeds a
        bound.
    :raises SyntaxError: *expression* does not parse, as with ``eval``.
    """
    tree = ast.parse(expression, mode="eval")
    # Bound the walk before starting it, so a pathological but syntactically
    # legal answer is refused rather than walked.
    node_count = sum(1 for _ in ast.walk(tree))
    if node_count > _MAX_NODES:
        raise UnsafeExpression(f"expression too large: {node_count} nodes")
    return _eval_node(tree.body)
