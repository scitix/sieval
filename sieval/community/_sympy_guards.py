"""Guards for handing model output to sympy.

Both math graders in this package parse a model's answer with sympy, and sympy
parsing *is* an execution path. The three guards here are the same in both, and
have to stay the same: a new escape route closes in one place or the other
grader is left open. That shared contract is why they live here rather than in
either module.

The threat has three legs, and closing one without the others buys nothing:

1. ``parse_expr``'s default global namespace is built by
   ``exec("from sympy import *", ...)``, which also injects ``__builtins__`` --
   so model output gets ``__import__`` and ``open``. :func:`sympy_globals`
   removes them.
2. Sympy re-sympifies a *string* argument with its own default namespace, which
   has the builtins back, so a call carrying a string literal escapes leg 1.
   :func:`quotes_free` refuses the quote instead of the callee.
3. ``parse_expr`` evaluates as it parses, so ``9**9**9`` never returns.
   :func:`evaluable` screens it out with an unevaluated pre-parse.

**None of this makes a string safe to hand to ``sympify``, ``simplify``, ``N``
or ``S``.** Those take the default namespace, not the caller's, so they defeat
legs 1 and 2 on their own -- a payload needs no quote at all once ``__import__``
resolves (``__import__(chr(111)+chr(115))``). A caller must ensure a *parsed
sympy object* reaches them, never the raw text; refusing at the parse step and
then falling through to ``simplify(text)`` reopens exactly what was closed.

Callers are responsible for their own time bound. Leg 3 only screens the two
shapes common enough to be worth not spending a worker on; an eagerly-evaluating
sympy callable needs no exponent to be expensive (``primepi(10**12)`` takes 48 s,
``factorial(1000000)`` 3.8 s), and enumerating those is the same losing game as
allowlisting callees in leg 2. The bound that actually holds is
:data:`~sieval.core.utils.offload.GRADE_TIMEOUT` in the worker process.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

#: Largest integer exponent :func:`evaluable` will admit. Sympy computes
#: ``a**b`` eagerly, so a boxed ``9^9^9^9`` asks for a 370-million-digit integer
#: and never returns. Nothing either benchmark asks for comes close — the
#: largest exponent in UGMathBench's pinned references is three digits — so the
#: cap costs no reachable comparison.
MAX_EXPONENT = 10_000


def sympy_globals() -> dict:
    """Namespace for :func:`parse_expr`, with the builtins removed.

    ``parse_expr`` evaluates its input, and its *default* global namespace is
    built by ``exec("from sympy import *", ...)`` — which also injects
    ``__builtins__``. Since the string being parsed is model output, that hands
    a model ``__import__`` and ``open``: a boxed
    ``__import__('os').system(...)`` runs, and the grader still reports the slot
    wrong, so nothing in the run looks unusual.

    Clearing ``__builtins__`` closes that without narrowing the dialect. The
    sympy names have to stay: ``auto_symbol`` rewrites an unknown callable into
    ``Function('sin')``, so a namespace holding *only* the answer aliases fails
    every legitimate ``sin(pi*x/5)`` with ``NameError: name 'Function' is not
    defined``.

    This is a namespace restriction, not a sandbox — attribute access on sympy
    objects still resolves, and it only covers the namespace *this* parse runs
    in. It does not survive a nested parse, which is why :func:`quotes_free`
    exists.
    """
    namespace: dict = {}
    exec("from sympy import *", namespace)  # noqa: S102 - fixed literal, not input
    namespace["__builtins__"] = {}
    return namespace


def quotes_free(text: str) -> bool:
    """Is *text* free of the string literals that reopen the interpreter?

    :func:`sympy_globals` sanitizes the namespace the top-level parse runs in,
    and that is not enough on its own. Sympy re-sympifies a *string* argument
    with its own default namespace, which has the builtins back, so a call
    carrying a string literal escapes the restriction and runs::

        eval("__import__('os').system(...)")

    ``sympify``, ``S`` and ``N`` do the same thing, and ``auto_symbol`` turns
    any unrecognized name into a ``Function``, so the callee cannot be
    allowlisted — every function call is a potential carrier. What can be
    refused is the payload: without a quote there is no string literal for the
    nested parse to read, and the argument comes back as a sympy object
    (``eval(chr(112))`` evaluates ``chr`` symbolically and does nothing).

    This holds only for text reaching ``parse_expr`` with a cleared namespace.
    It is **not** sufficient for text handed straight to ``sympify`` / ``N`` /
    ``simplify``, where ``__import__`` resolves without any quote at all — see
    the module docstring.

    Nothing legitimate is lost, on evidence from both dialects: not one of
    UGMathBench's 42,064 gold slots on the pinned revision contains a quote
    (sympy source, where quotes have no meaning), and for deepseek's LaTeX it is
    the 6,319-sample replay in that module's deviations note. A refused
    prediction only loses this one reading, with the LaTeX and literal-equality
    paths still offered to the comparison.
    """
    return "'" not in text and '"' not in text


def evaluable(cleaned: str, local: dict | None = None, transformations=None) -> bool:
    """Is *cleaned* free of the two exponent shapes that never finish?

    Deliberately narrower than "would this terminate". ``parse_expr`` evaluates
    as it parses, so the check cannot run afterwards — by then the process is
    already computing. Parsing with ``evaluate=False`` first builds the tree
    without doing the arithmetic (microseconds even for the pathological cases),
    which is cheap enough to screen on.

    Rejected: a power whose exponent is itself a power (``9**9**9``, the tower
    shape), and an integer exponent above :data:`MAX_EXPONENT`. A left-nested
    ``(x**2)**3`` is fine and stays — only the right-nested tower explodes.

    What it does **not** screen is covered in the module docstring: an eagerly
    evaluating sympy callable needs no exponent to be expensive, and the bound
    that holds is the caller's worker timeout. This screen only buys back the
    two shapes common enough to be worth not spending a worker on.

    A rejected answer grades wrong rather than hanging the run. That is the
    correct trade for a grader: this pass only ever *upgrades* a verdict (it is
    reached after every other strategy said "not equal"), so refusing to run it
    can lose a point but can never invent one.
    """
    import sympy
    from sympy.parsing.sympy_parser import parse_expr
    from sympy.parsing.sympy_parser import (
        standard_transformations as _standard,
    )

    if "**" not in cleaned:
        return True
    try:
        tree = parse_expr(
            cleaned,
            local_dict=local,
            global_dict=sympy_globals(),
            transformations=_standard if transformations is None else transformations,
            evaluate=False,
        )
    except Exception:
        # Unparseable under this reading; the real parse will fail the same way
        # and is harmless. Screening is not the place to decide that.
        return True
    for node in sympy.preorder_traversal(tree):
        if not isinstance(node, sympy.Pow):
            continue
        exponent = node.exp
        if exponent.has(sympy.Pow):
            return False
        if exponent.is_Integer and abs(int(exponent)) > MAX_EXPONENT:
            return False
    return True
