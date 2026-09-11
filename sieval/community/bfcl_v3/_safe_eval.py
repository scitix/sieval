"""Evaluate a literal arithmetic expression without executing anything.

`resolve_ast_by_type` resolves each keyword argument of a decoded function call
into a Python value, and for `ast.BinOp` upstream does it with
``eval(ast.unparse(value))``. The node comes from a model's reply, so that line
runs model-authored code: a reply carrying
``f(x=__import__('os').system('...') + 0)`` parses to a `BinOp`, `eval` runs the
call, and the decoder returns an ordinary-looking number. Nothing in the run
looks unusual afterwards -- the sample grades, the score is plausible, and the
only trace is whatever the payload did.

:func:`safe_eval` replaces that one call. It walks the node instead of unparsing
it, so there is no string and no namespace to escape from, and it admits only
literals, their containers, and the arithmetic/bitwise operators. A `Name`,
`Call`, `Attribute`, `Subscript` or comprehension is refused outright, which is
the entire divergence: every expression upstream's `eval` computes *without*
executing something is computed here too, to the same value.

Three shapes are refused that do not execute, and all three are deliberate:

* An f-string, even one with no placeholders. `ast.literal_eval` draws the line
  in the same place, and the alternative is a `FormattedValue` walker guarding a
  shape no function argument has been observed to carry.
* An expression whose *cost* is unbounded -- see :data:`MAX_RESULT_SIZE`.
  Decoding runs inline on the session's event loop with no timeout around it
  (the grade timeout is in the `feedback` worker, a stage later), so
  ``f(x=9**9**9)`` would not return a wrong answer; it would stall every other
  sample in the run.
* ``%`` formatting on a string or bytes left operand. It belongs to the class
  above, but it is the one member of it that cannot be *screened*: every other
  eager operator's cost is a function of its operands' sizes, while a printf
  precision field sets the result size on its own -- ``"%.400000000f" % 1.0``
  asks for a 400MB string from two tiny inputs. There is nothing to measure
  before the fact, and :func:`_bounded` would see it only after the allocation
  it exists to prevent, so the shape is refused outright. Integer ``%`` is
  bounded by its right operand and stays admissible.

Refusal raises `ValueError`, which reaches the decoder's caller as a decode
failure and scores the sample wrong. That is the right outcome and not a
softening of it: a reply that puts executable code in an argument slot has not
produced the function call it was asked for, and upstream scores every other
unparseable reply the same way.

This is a restriction on what is evaluated, not a sandbox. It holds because the
accepted node types cannot name anything outside the expression; it would stop
holding the moment one that can is added.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import ast
import operator

#: Largest intermediate any step may produce, counted in bits for integers and
#: in items for everything sized. A bound is needed because the operators are
#: eager: `9 ** 9 ** 9` asks for a 369-million-digit integer and does not
#: return, and `'a' * 10**9` allocates a gigabyte before anything can inspect
#: it. 2**20 leaves a million-bit integer and a million-character string
#: admissible, against a largest integer literal of 13 digits (about 41 bits)
#: across the gold answers of all 2351 gradeable Python rows in the pinned
#: snapshot -- and no argument there reaches this module at all, since none of
#: them is written as an expression.
MAX_RESULT_SIZE = 1 << 20

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.BitAnd: operator.and_,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
    ast.Not: operator.not_,
}


def _size(value) -> int:
    """Footprint of an intermediate: bits for an integer, items for anything sized.

    `bool` is excluded from the integer arm on purpose -- it is an `int`
    subclass whose `bit_length` says nothing useful, and it is never the thing
    that grows.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return value.bit_length()
    try:
        return len(value)
    except TypeError:
        return 0


def _refuse_if_unbounded(op, left, right) -> None:
    """Screen the operators whose cost their operands' size does not bound.

    Checked *before* the operator runs, which is the only time it can be: the
    cost of `9 ** 9 ** 9` cannot be measured after the fact. Addition and the
    bitwise operators need no entry -- their result is at most as large as the
    sum of their inputs, so :func:`safe_eval`'s check on the value that comes
    back catches them one step later.

    String `%` is the one shape here that is refused rather than screened,
    because its result size comes from the format spec instead of from the
    operands -- see the module docstring.
    """
    grown = None
    if isinstance(op, ast.Mod) and isinstance(left, (str, bytes)):
        raise ValueError(
            "refusing `%` formatting on a str/bytes operand: a precision field "
            "sets the result size independently of both operands, so the cost "
            "cannot be bounded before the allocation"
        )
    if isinstance(op, (ast.Pow, ast.LShift)) and isinstance(right, int):
        if isinstance(right, bool) or right < 0:
            return
        # A float base overflows to `inf` cheaply, so only an integer one grows.
        if isinstance(op, ast.LShift):
            grown = _size(left) + right
        elif isinstance(left, int):
            grown = _size(left) * right
    elif isinstance(op, ast.Mult):
        # Sequence repetition. Two integers multiplied only *add* bit lengths,
        # so that case is left to the post-hoc check.
        for value, count in ((left, right), (right, left)):
            if (
                isinstance(count, int)
                and not isinstance(count, bool)
                and hasattr(value, "__len__")
            ):
                grown = len(value) * max(count, 0)
                break
    if grown is not None and grown > MAX_RESULT_SIZE:
        raise ValueError(
            f"refusing an expression whose result would exceed "
            f"{MAX_RESULT_SIZE} units ({type(op).__name__})"
        )


def safe_eval(node: ast.expr):
    """Value of a literal expression, computed without executing anything.

    Raises `ValueError` on any node that could name or call something outside
    the expression, and on any expression whose cost is unbounded.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(safe_eval(element) for element in node.elts)
    if isinstance(node, ast.List):
        return [safe_eval(element) for element in node.elts]
    if isinstance(node, ast.Set):
        return {safe_eval(element) for element in node.elts}
    if isinstance(node, ast.Dict):
        # A `**spread` entry has a `None` key and names something to spread, so
        # it is refused with everything else that names.
        if any(key is None for key in node.keys):
            raise ValueError("refusing a dict spread in a function argument")
        return {
            safe_eval(key): safe_eval(value)
            for key, value in zip(node.keys, node.values, strict=True)
            if key is not None  # unreachable past the guard; narrows the type
        }
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _bounded(_UNARY_OPS[type(node.op)](safe_eval(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = safe_eval(node.left)
        right = safe_eval(node.right)
        _refuse_if_unbounded(node.op, left, right)
        return _bounded(_BIN_OPS[type(node.op)](left, right))
    raise ValueError(
        f"refusing to evaluate {type(node).__name__} in a function argument"
    )


def _bounded(value):
    """Return `value`, or refuse it for being larger than :data:`MAX_RESULT_SIZE`.

    The companion to :func:`_refuse_if_unbounded`: that one screens the
    operators that can outrun their inputs, this one catches growth by
    accumulation, where every single step looked affordable.
    """
    if _size(value) > MAX_RESULT_SIZE:
        raise ValueError(
            f"refusing an intermediate larger than {MAX_RESULT_SIZE} units"
        )
    return value
