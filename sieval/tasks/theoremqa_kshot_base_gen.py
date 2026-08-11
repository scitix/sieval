"""
TheoremQA k-shot base generative task.

This implementation intentionally tracks the original TheoremQA vLLM
evaluation path: official short-form examples, matching stop tokens, and the
upstream answer_clean matcher. For score reproduction, configure the model
layer with the official decoding values: temperature=0, top_p=1, and
max_tokens=2048.
The few-shot prompt preserves official runtime artifacts, including the
approximation symbol and the control characters produced by non-raw LaTeX
escapes in the upstream examples.py.
Two common anomaly classes are therefore expected compatibility artifacts:
long chain-of-thought outputs can finish with reason="length", and repeated
"The answer is" triggers can be treated as ICL leakage by the upstream cleaner.

Known implementation deviations are execution safety, import fallback, and
typed groundtruth plumbing:

* **Numeric answers are evaluated, not executed.** Upstream turns an extracted
  answer into a number with a bare ``eval(num)`` on model output.
  :func:`safe_eval` below replaces the interpreter with an AST walk over an
  allowlist, which runs nothing and is bounded. This is the one *safety*
  divergence taken under the unqualified name rather than in a ``_fixed``
  variant -- see ``sieval/tasks/CLAUDE.md`` on where fidelity stops -- and it is
  measured at **zero**: replaying a stored 800-sample run through upstream's
  ``eval`` and through this walk reaches 706 expressions, all 706 agree, and
  accuracy is 44.625 either way. The earlier mitigation here, an ``eval`` with
  ``__builtins__`` cleared, was escapable (clearing builtins does not stop
  attribute traversal) and *also* silently dropped ``abs`` / ``round`` / ``pow``;
  the walk restores them, so it is the more faithful of the two as well as the
  safe one.

  The dialect is narrower than ``eval``'s outside the numeric subset the three
  call sites reach: comparisons, boolean and conditional expressions, dict
  displays, subscripts and string literals are all refused. Only the string
  literal is refused *for* safety -- it is the one literal that can carry code
  into a function that re-parses its argument -- and the rest are simply shapes
  no answer on this benchmark is written in.
* latex2sympy2 falls back to latex2sympy2_extended when the original package is
  unavailable.
* Numeric groundtruths are derived from the dataset's Answer_type field instead
  of the official loader's runtime Python types, to reproduce the same typed
  comparison inputs.

One upstream grader defect is reproduced rather than repaired, as the
unqualified name requires: ``latex2sympy`` folds a list answer such as
``[0, 0]`` into the set ``{0}``, and evaluating that set cements the corruption
where refusing it would let the original correct string survive to the
string-equality check. Repairing it is worth **+2 of 800 samples** (44.625 ->
44.875, no verdict moving right-to-wrong) and belongs to a ``_fixed`` variant,
which is deliberately left uncoined for a delta this size.

The Qwen2.5 technical report Table 2 lists TheoremQA as a 5-shot base-model
benchmark: Qwen2.5-72B scores 42.4, while 42.8 belongs to Qwen2-72B. By
matching the original TheoremQA runner strictly, this task reproduces a nearby
Qwen2.5-72B score under SiEval.

AI-Generated Code - GPT-5.5 (OpenAI)
"""

import ast
import asyncio
import contextlib
import math
import operator
import re
from collections.abc import Callable
from importlib import import_module
from math import cos, e, exp, factorial, log, pi, sin, sqrt
from typing import override

from loguru import logger

from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    ReferenceImpl,
    Task,
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
    sieval_task,
)
from sieval.core.tasks.metrics import (
    DENOMINATOR_FIELD,
    DENOMINATOR_JUDGED,
    SCORE_KEY_FIELD,
    first_rollout_correct,
    health_metrics,
    sampling_report,
)
from sieval.core.utils.offload import GRADE_TIMEOUT, run_cpu_bound
from sieval.datasets import TheoremQADatasetSample

from ._math_verify import normalize_vote


def _load_latex2sympy():
    # Official TheoremQA imports latex2sympy2. Python >=3.12 environments may
    # only have the compatible latex2sympy2_extended fallback, whose parser can
    # differ from upstream on edge cases.
    try:
        return import_module("latex2sympy2").latex2sympy
    except ModuleNotFoundError:
        return import_module("latex2sympy2_extended").latex2sympy


_LATEX2SYMPY: Callable[[str], object] | None = None


def _get_latex2sympy() -> Callable[[str], object]:
    global _LATEX2SYMPY
    if _LATEX2SYMPY is None:
        _LATEX2SYMPY = _load_latex2sympy()
    return _LATEX2SYMPY


# ---------------------------------------------------------------------------
# Numeric answer evaluation, without an interpreter.
#
# Upstream's number_utils.py turns an extracted answer into a number with a bare
# `eval(num)`. The answer is model output, so that is arbitrary code execution;
# the usual mitigation -- `eval(num, {"__builtins__": {}})`, which is what this
# task shipped before -- does not close it, because clearing the builtins does
# not remove *attribute traversal*, and the standard `catch_warnings` route
# walks from a literal back to a live `open`:
#
#     [c for c in ().__class__.__base__.__subclasses__()
#      if c.__name__=='catch_warnings'][0]()._module.__builtins__['open']('x','w')
#
# `safe_eval` parses to an AST and evaluates an explicit allowlist of node types
# itself. Nothing is executed, so there is no namespace to escape from and no
# sandbox to audit -- the payload above dies at its first `Attribute` node,
# before any object is reached. Preferred over the `parse_expr` route used for
# the UGMathBench grader (`sieval/community/ugmathbench.py`) because that one
# *is* an execution path, needing a cleared namespace, a quote screen and a
# power-tower pre-parse to be safe; those guards are load-bearing there because
# that grader wants a *symbolic* comparison, where the three call sites here
# want a *number* out of a Python-ish expression.
#
# Fidelity is not traded away for it, and the module docstring records the
# measurement: the dialect is upstream's numeric subset, and the builtins that
# upstream's `eval` exposed and the cleared namespace silently lost (abs, round,
# pow, min, max, sum, int, float, complex, divmod, len) are back, as *values* in
# a namespace only a validated tree can reach. Exposing them is safe here in a
# way it is not under `eval`: the danger was never `abs`, it was the traversal
# that reaches everything else, and no allowlisted node can express one.
# ---------------------------------------------------------------------------

#: Upstream's ``E``. ``number_utils.py`` defines it as this truncated literal
#: rather than :data:`math.e`, and an answer written ``E`` is compared against
#: the truncated value there, so it is reproduced exactly.
E = 2.718

#: Largest first argument :func:`factorial`, ``math.comb`` and ``math.perm``
#: will accept. Upstream bounds none of them, all three are in scope for every
#: graded answer, and each is reachable with a two-token literal: measured,
#: ``factorial(2000000)`` runs ~16 s and ``math.comb(2000000, 1000000)`` ~14 s.
#: Grading runs in a worker process under
#: :data:`~sieval.core.utils.offload.GRADE_TIMEOUT`, but that frees the
#: *caller* -- a pool cannot interrupt a running call, so an unbounded answer
#: holds its worker until it finishes, and enough of them empty the pool for
#: every task sharing it.
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

#: Ceiling on the length of a sequence built by *repetition*. A literal display
#: is already bounded by :data:`_MAX_NODES`; ``[0] * n`` is not, and seven nodes
#: build an 800 MB list at ``n = 10**8``. What this protects is not the sample
#: -- callers treat a ``MemoryError`` as "not a number", like any other refusal
#: -- but the pool: a worker killed by the OOM reaper reaches
#: :func:`~sieval.core.utils.offload.run_cpu_bound` as a ``BrokenExecutor``,
#: which drops *every* task in the session back to inline grading for the rest
#: of the run.
_MAX_SEQUENCE_LENGTH = 10_000


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


def _guarded_comb(n, k):
    """``math.comb(n, k)``, under the :data:`_MAX_FACTORIAL` ceiling.

    Bounded rather than refused: the ceiling is far above any answer this
    benchmark asks for, so upstream's function survives for every reachable
    comparison while the size-driven runaway does not.
    """
    if not isinstance(n, int) or n > _MAX_FACTORIAL:
        raise UnsafeExpression(f"comb argument out of range: {n!r}")
    return math.comb(n, k)


def _guarded_perm(n, k=None):
    """``math.perm(n, k)``, under the :data:`_MAX_FACTORIAL` ceiling."""
    if not isinstance(n, int) or n > _MAX_FACTORIAL:
        raise UnsafeExpression(f"perm argument out of range: {n!r}")
    return math.perm(n, k)


def _guarded_mul(left, right):
    """``left * right``, refused when it repeats a sequence past the bound."""
    for sequence, count in ((left, right), (right, left)):
        if isinstance(sequence, (list, tuple)) and isinstance(count, int):
            length = count * len(sequence)
            if length > _MAX_SEQUENCE_LENGTH:
                raise UnsafeExpression(f"sequence repetition of length {length}")
    return operator.mul(left, right)


def _guarded_lshift(value, amount):
    """``value << amount``, refused when the result cannot be bounded."""
    if not isinstance(amount, int) or amount > _MAX_RESULT_BITS:
        raise UnsafeExpression(f"shift amount out of range: {amount!r}")
    return operator.lshift(value, amount)


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
    # `True` / `False` / `None` need no entry: all three parse as `ast.Constant`
    # and are returned by that branch, never looked up here.
}

#: ``math`` members reachable through an attribute. An allowlist rather than
#: ``hasattr(math, ...)``, because the module is also the way *past* the bounds
#: above: ``math.factorial`` is not :func:`_guarded_factorial`, and
#: ``math.comb`` / ``math.perm`` are the same size-driven integer computation
#: under a different name. Replacing exactly those three with their bounded
#: wrappers makes both spellings of each agree; what is left of the public
#: surface returns floats or is a constant, and a float's only runaway is an
#: ``OverflowError`` the call site already converts.
#:
#: The sequence-consuming members (``prod``, ``fsum``, ``lcm``, ``dist``) need
#: no entry of their own -- their cost is the length of the sequence handed to
#: them, which :data:`_MAX_NODES` and :data:`_MAX_SEQUENCE_LENGTH` bound between
#: them. Built by filtering so the map tracks whatever ``math`` a given Python
#: version ships, with only the bounded names spelled out.
_MATH_ATTRS: dict[str, object] = {
    name: getattr(math, name) for name in dir(math) if not name.startswith("_")
} | {
    "factorial": _guarded_factorial,
    "comb": _guarded_comb,
    "perm": _guarded_perm,
}

_BIN_OPS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: _guarded_mul,
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
        # Converted on the same terms as a call: an operator that overflows or
        # runs the process out of memory is an answer outside the subset, not a
        # failed sample. Without this, `MemoryError` reaches the call sites as
        # itself -- still caught by their broad `except`, but indistinguishable
        # in a log from the pool dying.
        try:
            return op(_eval_node(node.left), _eval_node(node.right))
        except UnsafeExpression:
            raise
        except (OverflowError, MemoryError) as exc:
            raise UnsafeExpression(f"operator overflowed: {exc}") from exc

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
        # Only ``math.<allowlisted name>``. Restricting the base to the literal
        # name ``math`` is what keeps this from being a *traversal*: no other
        # object is reachable, and the dunder path that would reach one is not
        # in `_MATH_ATTRS`. The allowlist then carries the *size* bounds, which
        # the base restriction alone does not give -- see its definition.
        if isinstance(node.value, ast.Name) and node.value.id == "math":
            try:
                return _MATH_ATTRS[node.attr]
            except KeyError:
                raise UnsafeExpression(f"math member {node.attr!r}") from None
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


_DIRECT_ANSWER_TRIGGERS = ["The answer is:", "The answer is", "the answer is"]
_STOP_TOKENS = [
    "USER:",
    "ASSISTANT:",
    "### Instruction:",
    "Response:",
    "<start_of_turn>",
    "[INST]",
    "\n\nProblem",
    "Problem:",
]
_THEOREMQA_RUN_URL = "https://github.com/TIGER-AI-Lab/TheoremQA/blob/acfc9686aa9b49f3c8f189364a9a9ee9c53da039/run.py"  # noqa: E501

# These strings mirror the official runtime examples, including artifacts from
# upstream non-raw string escapes. The model-facing prompt is intentionally more
# important here than source readability.
_THEOREMQA_EXAMPLES: tuple[tuple[str, str], ...] = (
    (
        "In a 10 Gigabit Ethernet network, the average size of a frame is "
        "1500 bytes. If a burst of noise lasting 1ms interrupts the network, "
        "how many frames are lost?",
        "First, calculate the data rate in bytes/s:\n\n"
        "10 Gigabit/s * (1 Byte / 8 bits) = 1.25 * 10^9 Bytes/s\n\n"
        "Next, calculate the data loss in bytes due to the noise:\n\n"
        "1 ms * 1.25 * 10^9 Bytes/s = 1.25 * 10^6 Bytes\n\n"
        "Finally, divide the data loss by the average frame size to get the "
        "number of frames lost:\n\n"
        "1.25 * 10^6 Bytes / 1500 Bytes/frame \u2248 833.33 frames\n"
        "The answer is 833.33",
    ),
    (
        "Given x = 0.157, what is the value of x \\times "
        "\\frac{\\prod_{n=1}^\\infty (1 - \\frac{x^2}{n^2 \\pi^2})}"
        "{\\sin(x)}?",
        "To evaluate the expression $x \\times "
        "\\frac{\\prod_{n=1}^{\\infty} (1 - \\frac{x^2}{n^2 \\pi^2})}"
        "{\\sin(x)}$ given x = 0.157, we first recognize that the product "
        "in the numerator is related to the sine function through the "
        "Euler's reflection formula for the sine function, which can be "
        "expressed as:\n\n"
        "$$\\sin(x) = x \\prod_{n=1}^{\\infty} \\left(1 - "
        "\\frac{x^2}{n^2 \\pi^2}\\right)$$\n\n"
        "Therefore, the given expression simplifies to: $x \\times "
        "\\frac{\\sin(x)}{\\sin(x)}$\n\n"
        "Because sin(x) in the numerator and denominator cancels out, the "
        "expression simplifies further to just x.\n\n"
        "So, given x = 0.157, the value of the expression is 0.157. "
        "This result is derived from the properties of the sine function "
        "and does not require computational evaluation.\n"
        "The answer is 0.157",
    ),
    (
        "Consider the basis C of \\mathbb{R}^2 consisting of vectors "
        "u_1 = [2, 4] and u_2 = [1, -1]. If y = [8, 12], find the "
        "C-coordinate vector of y.",
        "The goal is to express y as a linear combination of the basis "
        "vectors of C, i.e., $y = a\\cdot u_1 + b\\cdot u_2$, where a and b "
        "are the scalar coefficients that we want to find. These coefficients "
        "will form the C-coordinate vector of y, which we'll denote as "
        "$[a, b]_C$.\n\n"
        "Given:\n"
        "- $u_1 = [2, 4]$,\n"
        "- $u_2 = [1, -1]$,\n"
        "- $y = [8, 12]$.\n\n"
        "We need to solve the system of linear equations:\n"
        "2a + 1b = 8\n"
        "4a - 1b = 12\n\n"
        "Let's solve this system of equations to find a and b.\n\n"
        "The solution to the system of equations is $a = \\frac{10}{3} and "
        "b = \\frac{4}{3}$. Therefore, the C-coordinate vector of y in the "
        "basis consisting of vectors u_1 = [2, 4] and u_2 = [1, -1] is "
        "$\\left[\\frac{10}{3}, \\frac{4}{3}\\right]_C$. \n"
        "Let's calculate the numerical value of "
        "$\\left[\frac{10}{3}, \frac{4}{3}\right]_C$ as [3.33, 1.33].\n"
        "The answer is [3.33, 1.33]",
    ),
    (
        "One can draw a simple, connected planar graph with 200 vertices and "
        "397 edges. Is this statement Trur or False?",
        "To determine the answer, we can use Euler's formula for planar "
        "graphs, which states that for any finite, connected, planar graph, "
        "$V - E + F = 2$, where V is the number of vertices, E is the number "
        "of edges, and F is the number of faces.\n\n"
        "Given the modified question, we have V = 200 vertices and E = 397 "
        "edges. We want to find if we can have a graph that satisfies these "
        "conditions, adhering to Euler's formula.\n\n"
        "First, let's rearrange Euler's formula to solve for F:  F = E - V + 2\n\n"
        "Substituting the given values: F = 397 - 200 + 2,  F = 199\n\n"
        "This means a graph with 200 vertices and 397 edges would have 199 "
        "faces. However, to determine the truth of this possibility, we "
        "should check if this graph doesn't violate any other planar graph "
        "constraints, particularly regarding the number of edges.\n\n"
        "For a simple, connected planar graph, there's also a relationship "
        "between vertices, edges, and faces given by the inequality: "
        "$E \\leq 3V - 6$\n\n"
        "Substituting V = 200 gives: $E \\leq 3*200 - 6 = 594$\n\n"
        "With E = 397, the condition $E \\leq 594$ is satisfied, meaning "
        "it's theoretically possible in terms of the edge condition for a "
        "planar graph.\n\n"
        "Therefore, one can draw a simple, connected planar graph with 200 "
        "vertices and 397 edges, resulting in 199 faces, without violating "
        "the conditions for it to be planar according to both Euler's formula "
        "and the constraint on the maximum number of edges.\n"
        "The answer is True",
    ),
    (
        "Given a finite group G, and a collection of permutations H on a set. "
        "Then (a) there always exists H such that G is isomorphic to H; "
        "(b) for any H, G is isomorphic to H; (c) G can never be isomorphic "
        "to H; (d) none of the above. Which option is correct?",
        "This is based on Cayley's theorem, which states that every group G "
        "is isomorphic to a subgroup of the symmetric group acting on G. \n"
        "In other words, for every finite group G, there exists a collection "
        "of permutations H (which in this context, can be thought of as the "
        "set of permutations representing the action of G on itself) such "
        "that G is isomorphic to H.\n\n"
        "Therefore, there always exists H such that G is isomorphic to H.\n"
        "The answer is (a)",
    ),
)

_DEFAULT_FEW_SHOT_COUNT = len(_THEOREMQA_EXAMPLES)


def floatify(num: str):
    try:
        num_float = float(num)
        if num_float.is_integer():
            return round(num_float)
        return num_float
    except Exception:
        return None


def within_eps(pred: float, gt: float):
    eps = abs(gt) * 0.04
    return gt - eps <= pred <= gt + eps


def clean_units(pred_str: str):
    """Clean the units in the number."""

    def convert_pi_to_number(code_string: str):
        code_string = code_string.replace("\\pi", "\u03c0")
        code_string = re.sub(r"(?<![\d}])\\?\u03c0", "3.14", code_string)
        code_string = re.sub(r"(\d)(\\?\u03c0)", r"\1*3.14", code_string)
        code_string = re.sub(r"\{(\\?\u03c0)\}", "3.14", code_string)
        code_string = re.sub(r"\*(\\?\u03c0)", "*3.14", code_string)
        return code_string

    pred_str = convert_pi_to_number(pred_str)
    pred_str = pred_str.replace("%", "/100")
    pred_str = pred_str.replace("$", "")
    pred_str = pred_str.replace("\u00a5", "")
    pred_str = pred_str.replace("\u00b0C", "")
    pred_str = pred_str.replace(" C", "")
    pred_str = pred_str.replace("\u00b0", "")
    return pred_str


def number_it(num):
    if isinstance(num, (int, float)):
        return num

    num = clean_units(num)
    with contextlib.suppress(Exception):
        num = str(_get_latex2sympy()(num))

    if floatify(num) is not None:
        return floatify(num)

    try:
        num_value = safe_eval(num)
        if isinstance(num_value, (list, tuple)):
            num_value = num_value[0]
        if floatify(num_value) is not None:
            return floatify(num_value)
        return None
    except Exception:
        return None


def compare_two_numbers(p, gt):
    try:
        if math.isnan(p):
            return False
        if isinstance(gt, int):
            return round(p) == gt
        return within_eps(pred=p, gt=gt)
    except Exception:
        return False


def compare_two_list(pred, gt):
    if not isinstance(pred, list):
        return False
    if len(pred) != len(gt):
        return False
    if any(not isinstance(x, (int, float)) for x in pred):
        return False
    pred = sorted(pred)
    gt = sorted(gt)
    return all(compare_two_numbers(p, g) for p, g in zip(pred, gt, strict=True))


def extract_theoremqa_answer(pred: str, answer_flag: bool = True):
    if any(option in pred.lower() for option in ["yes", "true"]):
        pred = "True"
    elif any(option in pred.lower() for option in ["no", "false"]):
        pred = "False"
    elif any(
        option in pred.lower() for option in ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    ):
        pass
    elif answer_flag:
        pred = pred.split("=")[-1].strip()
        pred = clean_units(pred)
        try:
            tmp = str(_get_latex2sympy()(pred))
            pred = str(safe_eval(tmp))
        except Exception:
            if re.match(r"-?[\d.]+\s\D+$", pred) or re.match(
                r"-?[\d.]+\s[^\s]+$",
                pred,
            ):
                pred = pred.split(" ")[0]
    else:
        preds = re.findall(r"-?\d*\.?\d+", pred)
        pred = preds[-1] if len(preds) >= 1 else ""

    return pred


def answer_clean(direct_answer_trigger_for_fewshot: list[str], pred: str):
    pred = pred.strip("\n")

    icl = False
    for trigger in direct_answer_trigger_for_fewshot:
        if pred.count(trigger) > 1:
            icl = True
    if icl:
        pred = pred.split("\n\n")[0]

    preds = re.split("|".join(direct_answer_trigger_for_fewshot), pred)
    if len(preds) > 1:
        answer_flag = True
        pred = preds[-1]
    else:
        answer_flag = False

    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ")

    pred = extract_theoremqa_answer(pred, answer_flag)
    pred = pred.rstrip(".").rstrip("/")

    return pred


def compare_answer_with_groundtruth(
    answer: str,
    groundtruth_str: str,
    groundtruth_num=None,
):
    if groundtruth_str.lower() in ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]:
        return groundtruth_str.lower() in answer.lower()
    if answer.lower() == groundtruth_str.lower():
        return True
    if groundtruth_num is not None:
        if isinstance(groundtruth_num, (int, float)):
            return compare_two_numbers(number_it(answer), groundtruth_num)
        if answer.startswith("(") and answer.endswith(")"):
            try:
                answer_list = list(safe_eval(answer))
                answer_list = [number_it(a) for a in answer_list]
            except Exception:
                return False
            return compare_two_list(answer_list, groundtruth_num)
        return False
    return False


def _get_short_format(qas: list[tuple[str, str]]):
    tmp = "You are supposed to provide a solution to a given problem.\n\n"
    for q, a in qas:
        tmp += f"\nProblem:\n{q}\nSolution:\n{a}\n"
    prefix = "\nProblem:\n{query}\nSolution:\n"

    return tmp, prefix


def _parse_groundtruth_num(answer: str, answer_type: str):
    if answer_type == "integer":
        return int(answer)
    if answer_type == "float":
        return float(answer)
    if answer_type == "list of integer":
        value = ast.literal_eval(answer)
        return [int(v) for v in value]
    if answer_type == "list of float":
        value = ast.literal_eval(answer)
        return [float(v) for v in value]
    return None


def _groundtruth_args(sample: TheoremQADatasetSample):
    answer = str(sample["Answer"])
    try:
        groundtruth_num = _parse_groundtruth_num(answer, sample["Answer_type"])
    except Exception:
        groundtruth_num = None
    return answer, groundtruth_num


def _normalize_n_shot(n_shot: int | None) -> int:
    if n_shot is None:
        return _DEFAULT_FEW_SHOT_COUNT
    if isinstance(n_shot, bool) or not isinstance(n_shot, int):
        raise TypeError(
            f"n_shot must be an int, got {type(n_shot).__name__}: {n_shot!r}"
        )
    if n_shot < 0:
        raise ValueError(f"n_shot must be >= 0, got {n_shot}")
    if n_shot > len(_THEOREMQA_EXAMPLES):
        raise ValueError(
            f"n_shot must be <= {len(_THEOREMQA_EXAMPLES)} because only "
            "that many built-in TheoremQA examples are available."
        )
    return n_shot


@sieval_task(
    name="theoremqa_kshot_base_gen",
    display_name="TheoremQA (k-shot, base generative)",
    description="TheoremQA k-shot ICL with official short prompt and matcher.",
    eval_mode=EvalMode.GEN,
    n_shot=_DEFAULT_FEW_SHOT_COUNT,
    tags=("english", "open-ended", "theorem-driven"),
    deps_group="math",
    model_type="gen",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="TIGER-AI-Lab/TheoremQA",
        url=_THEOREMQA_RUN_URL,
        notes=(
            "Prompt follows official short-form examples by default; n_shot can "
            "select any prefix of the built-in examples. answer_clean and "
            "numeric matching follow official utils.py/number_utils.py, with "
            "ONE deliberate SAFETY divergence, taken for execution safety "
            "rather than as a repair. UPSTREAM EXECUTES MODEL OUTPUT: "
            "number_utils.py turns "
            "an extracted answer into a number with a bare eval(num), which is "
            "arbitrary code execution driven by the model under test. This task "
            "evaluates the same expressions with an AST walk over an allowlist "
            "(safe_eval, in this module) that executes nothing. BOUNDS: node "
            "count; integer-power and shift size; sequence repetition; and the "
            "first argument to factorial, math.comb and math.perm, which are "
            "reached through an allowlist so that math.factorial(n) and "
            "factorial(n) carry the same ceiling. "
            "MEASURED DIVERGENCE: ZERO. Replaying a stored 800-sample run "
            "(Qwen2.5-72B, 5-shot, greedy) through upstream's bare eval and "
            "through this walk reaches 706 expressions across the three call "
            "sites; 586 evaluate to an identical value, 120 fail under both, "
            "and 0 are refused here but evaluated by upstream. Task accuracy is "
            "44.625 under both readings, and no sample's verdict differs. None "
            "of the bounds in force at that replay (node count, integer-power "
            "size, factorial argument) was reached by any of the 706; the "
            "sequence-repetition and math-spelled ceilings were added in review "
            "afterwards, and each sits orders of magnitude above the shapes "
            "this benchmark's answers are written in. Fidelity was preserved "
            "wherever safety did not object WITHIN the numeric subset the three "
            "call sites reach: set displays "
            "and bitwise operators are supported purely because upstream "
            "computes them, and the builtins that upstream's eval exposed "
            "(abs/round/pow/min/max/sum/int/float/complex/divmod/len) are "
            "available again -- an earlier mitigation, eval with __builtins__ "
            "cleared, was BOTH escapable (clearing builtins does not stop "
            "attribute traversal; the catch_warnings route reaches a live open) "
            "AND silently lost those names to a NameError. Outside that subset "
            "the dialect is narrower than eval's: comparisons, boolean and "
            "conditional expressions, dict displays, subscripts and string "
            "literals are refused, and only the string literal is refused for a "
            "safety reason. REPRODUCED DEFECT: "
            "latex2sympy folds a list answer such as [0, 0] into the set {0}, "
            "and evaluating that set cements the corruption; refusing set "
            "displays instead would score +2 of 800 samples correct (44.625 -> "
            "44.875) with no verdict moving right-to-wrong. That is a grader "
            "repair, not a safety measure, so it is NOT taken here and the "
            "_fixed name is left uncoined for a delta this size. GRADING: "
            "extraction and comparison both reach latex2sympy, which is "
            "synchronous and unbounded, so both run in a worker process "
            "(sieval/core/utils/offload.py) rather than on the event loop the "
            "session shares."
        ),
    ),
)
class TheoremQAKShotBaseGenTask(
    Task[
        TheoremQADatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        # `float | str`: the report carries `score_key`, which names a column
        # rather than measuring one.
        dict[str, float | str],
    ]
):
    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        *,
        n_shot: int | None = None,
        k: int = 1,
        n: int = 1,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        if k > n:
            raise ValueError(
                f"pass@{k} needs at least {k} sample(s) per problem, got n={n}."
            )
        self._k = k
        self._n = n
        self.n_shot = _normalize_n_shot(n_shot)
        self._prompt_no_input: str | None = None
        self._prompt_prefix: str | None = None

    @override
    async def setup(self) -> None:
        self._prompt_no_input, self._prompt_prefix = self._build_prompt_parts()

    @override
    async def preprocess(self, raw, ctx):
        prompt_no_input, prefix = self._get_prompt_parts()
        return build_prompt_record(
            prompt_no_input + prefix.format(query=raw["Question"]),
            reference=_groundtruth_args(raw)[0],
        )

    @override
    async def infer(self, pre, ctx):
        # `n` is the sampling budget `k` was validated against, so it has to
        # reach the model (sieval/tasks/CLAUDE.md, "n_shot vs k").
        return await self.model.agenerate(pre["prompt"], n=self._n, stop=_STOP_TOKENS)

    @override
    async def postprocess(self, inf, ctx):
        # answer_clean returns "" when nothing was extracted; None is the
        # protocol's spelling, and report()'s `empty` counter reads `extracted`
        # instead of comparing against "".
        #
        # A timed-out extraction lands in that same counter, so `empty` means
        # "no answer came out of extraction", not "the model wrote nothing".
        # Deliberate: the two are the same fact for every consumer of the metric
        # — no prediction to grade — and splitting them would add a second
        # counter that is zero on every run where the grader keeps up. The
        # warning in `_extract` is what distinguishes them when it matters.
        # Concurrent for the same reason as `feedback`: extraction is offloaded
        # and bounded per rollout, so awaiting in turn multiplies the ceiling.
        extracted = await asyncio.gather(
            *(self._extract(text, ctx) for text in inf.texts)
        )
        return build_prediction_record([text or None for text in extracted])

    async def _extract(self, text: str, ctx) -> str:
        # Extraction runs latex2sympy over model output, which is synchronous
        # and has no bound of its own, and every runner in the session shares
        # one event loop — so doing it here would stall every other task. A
        # worker process rather than a thread for exactly the reason
        # `run_cpu_bound` documents: a thread cannot be given up on, so one
        # unparseable answer would hold its slot for the rest of the run.
        try:
            return await run_cpu_bound(
                answer_clean, _DIRECT_ANSWER_TRIGGERS, text, timeout=GRADE_TIMEOUT
            )
        except TimeoutError:
            logger.warning(
                "Extracting sample {} exceeded {}s; recorded as not extracted. "
                "The response is likely a shape latex2sympy cannot parse "
                "quickly.",
                ctx.sample_id,
                GRADE_TIMEOUT,
            )
            return ""

    @override
    async def feedback(self, post, ctx):
        answer, groundtruth_num = _groundtruth_args(ctx.raw_sample)
        # Concurrent, not sequential: each grade is an offloaded CPU-bound call
        # with its own GRADE_TIMEOUT, so awaiting them in turn makes a sample's
        # worst case n x the timeout instead of one.
        verdicts = await asyncio.gather(
            *(
                self._grade(rollout, answer, groundtruth_num, ctx)
                for rollout in post["rollouts"]
            )
        )
        rollouts = [
            build_rollout_judgement(rollout["index"], verdict)
            for rollout, verdict in zip(post["rollouts"], verdicts, strict=True)
        ]
        return True, build_judgement_record(answer, rollouts)

    async def _grade(self, rollout, answer, groundtruth_num, ctx) -> bool:
        # `or ""` restores exactly what the comparator saw pre-migration.
        prediction = rollout.get("prediction") or ""
        # Offloaded on the same grounds as postprocess: the comparator reaches
        # latex2sympy through number_it.
        try:
            correct = await run_cpu_bound(
                compare_answer_with_groundtruth,
                prediction,
                answer,
                groundtruth_num,
                timeout=GRADE_TIMEOUT,
            )
        except TimeoutError:
            # Same contract as the rest of the grader: an answer that cannot be
            # graded is a wrong answer, not a failed run.
            logger.warning(
                "Grading sample {} exceeded {}s and was scored wrong.",
                ctx.sample_id,
                GRADE_TIMEOUT,
            )
            correct = False
        return bool(correct)

    @override
    async def report(self, finals, fails) -> dict[str, float | str]:
        count = len(finals)
        # First-rollout, because that is what this port was validated against.
        # `empty` reads the same population as the old `pred == ""` check:
        # postprocess maps that empty extraction to None, i.e. extracted=False.
        empty = sum(
            1
            for ctx in finals
            if not ((ctx.postprocess_result or {}).get("rollouts") or [{}])[0].get(
                "extracted"
            )
        )
        accuracy = 100 * first_rollout_correct(finals) / count if count else 0.0
        metrics: dict[str, float | str] = {
            "score": accuracy,
            "accuracy": accuracy,
            "fails": len(fails),
            "empty": empty,
            SCORE_KEY_FIELD: "accuracy",
            DENOMINATOR_FIELD: DENOMINATOR_JUDGED,
        }
        # Outside the gate: extraction health is a fact about the parser,
        # not the draw, and n=1 is where a stopped extractor hides longest.
        metrics |= health_metrics(finals)
        if self._n <= 1:
            return metrics
        # Over `len(finals)`, the denominator `accuracy` uses: this task
        # excludes failed samples where its siblings count them wrong.
        return metrics | sampling_report(
            finals,
            n=self._n,
            k=self._k,
            denominator=count,
            normalize=normalize_vote,
        )

    def _build_prompt_parts(self) -> tuple[str, str]:
        used_examples = list(_THEOREMQA_EXAMPLES[: self.n_shot])
        return _get_short_format(used_examples)

    def _get_prompt_parts(self) -> tuple[str, str]:
        if self._prompt_no_input is None or self._prompt_prefix is None:
            self._prompt_no_input, self._prompt_prefix = self._build_prompt_parts()
        return self._prompt_no_input, self._prompt_prefix
