# adapted from https://github.com/EnvCommons/IMO-Bench/blob/66b014f1b3799972ddfc32dbacea51b802586141/answer_verification.py
"""IMO-Bench (Google DeepMind) AnswerBench answer verification.

IMO-AnswerBench grades a short answer against the gold with ``math_verify`` and a
normalized-string fallback when either side cannot be parsed (or ``verify()``
raises). Vendored from the upstream ``answer_verification.py``.

``math_verify`` is imported lazily inside the functions so importing a task
module stays free of the optional ``[math]`` dependency (sieval import discipline);
upstream imports it at module top.
"""

import re


def parse_answer(answer: str) -> list:
    """Parse a math answer with LaTeX handling; returns [] if it can't be parsed."""
    from math_verify import parse

    try:
        parsed = parse(answer)
        # Handle potential LaTeX by wrapping in $ for a proper LaTeX environment.
        if not parsed:
            parsed = parse(f"$ {answer} $")
        return parsed
    except Exception:
        return []


def verify_math_answer(gold: str, pred: str) -> bool:
    """IMO-Bench equivalence: ``math_verify`` with a normalized-string fallback.

    Falls back to ``gold.strip().lower() == pred.strip().lower()`` when either
    side won't parse, or when ``verify()`` raises on a malformed/adversarial input.

    Upstream's signature is ``verify_math_answer(answer_one, answer_two)`` and its
    caller passes ``(model_answer, gold)``; sieval passes **gold first** because
    ``math_verify.verify`` is documented non-symmetric (gold, target). The order is
    immaterial for these short answers and keeps the call consistent with sieval's
    other math tasks.
    """
    from math_verify import verify

    parsed_gold = parse_answer(gold)
    parsed_pred = parse_answer(pred)
    # Fall back to normalized string comparison when math_verify can't parse.
    if not parsed_gold or not parsed_pred:
        return gold.strip().lower() == pred.strip().lower()
    try:
        return bool(verify(parsed_gold, parsed_pred))
    except Exception:
        return gold.strip().lower() == pred.strip().lower()


# ---------------------------------------------------------------------------
# sieval gen-mode wrapper (NOT upstream).
#
# Upstream AnswerBench is agentic: the agent submits a clean answer string via a
# tool call, so verify_math_answer above sees exactly the gold's format. In a
# non-agentic generative run the model writes its answer inside \boxed{}, often
# verbosely — "P(x) = -1 \quad\text{or}\quad P(x) = x+1", "-2(m-1)", "$2^{u-2}$",
# function-prefixed, $-wrapped, with \left/\right, or a multi-answer list.
# math_verify.parse then mis-parses these and marks correct answers wrong.
#
# verify_answer_gen normalizes the boxed answer into the shape an agent would
# submit and does a set-wise comparison for multi-answer golds, delegating every
# atomic equivalence check to the vendored verify_math_answer. The fast path is
# the verbatim upstream check, so this never grades *more strictly* than upstream.
# ---------------------------------------------------------------------------

_FN_PREFIX = re.compile(r"^\s*[A-Za-z]\s*\(\s*[A-Za-z0-9]\s*\)\s*=\s*")
_SEP_WORDS = re.compile(r"\\text\s*\{\s*(?:and|or)\s*\}|\b(?:and|or)\b")
_TEXT_ANNOT = re.compile(r"\\text\s*\{[^{}]*\}")
_LATEX_NOISE = re.compile(r"\\left|\\right|\\displaystyle|\\!|\\,|\\;|\\:")


def _normalize(s: str) -> str:
    s = s.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    s = s.replace("$", "")
    s = _SEP_WORDS.sub(",", s)  # "A or B" / "A and B" list separators -> comma
    s = _TEXT_ANNOT.sub(" ", s)  # drop \text{...} prose annotations
    s = _LATEX_NOISE.sub(" ", s)
    s = re.sub(r"\\quad|\\qquad", " ", s)
    s = re.sub(r"\s+", " ", s).strip().rstrip(".").strip()
    return s


def _split_top_level(s: str) -> list[str]:
    """Split on top-level commas only; commas inside (), [], {} stay (tuples)."""
    parts: list[str] = []
    depth = 0
    cur = ""
    for ch in s:
        if ch in "([{":
            depth += 1
            cur += ch
        elif ch in ")]}":
            depth = max(0, depth - 1)
            cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def _atom_equiv(a: str, b: str) -> bool:
    # Identity-only (modulo whitespace, and modulo a leading "f(x)=" prefix).
    # We deliberately do NOT call math_verify per split-item: math_verify.parse
    # grabs a coincidental number out of expression/equation answers (e.g. "n=3k"
    # -> 3), which produced false positives. Whole-answer math equivalence is still
    # handled by the verify_math_answer fast path in verify_answer_gen.
    #
    # An empty side (both reduced to "" by _normalize, e.g. "\\text{No}" vs
    # "\\text{Yes}") is never a match — upstream verify_math_answer returns False
    # there, and matching "" == "" would over-count. (Identical text answers are
    # already caught by the verify_math_answer fast path before we normalize.)
    if not a.strip() or not b.strip():
        return False
    if a.replace(" ", "") == b.replace(" ", ""):
        return True
    a2, b2 = _FN_PREFIX.sub("", a).strip(), _FN_PREFIX.sub("", b).strip()
    return (a2, b2) != (a, b) and a2.replace(" ", "") == b2.replace(" ", "")


def verify_answer_gen(gold: str, pred: str | None) -> bool:
    """Grade a generative (boxed) answer against gold, IMO-Bench style.

    Fast path is the verbatim official ``verify_math_answer``; only when that fails
    do we normalize and set-match, so we never grade more strictly than upstream.
    Kept intentionally conservative — prefer under- to over-counting; genuinely
    free-form / prose answers (which need the upstream agentic clean submission or
    an LLM judge) are left as-is.
    """
    if pred is None:
        return False
    if verify_math_answer(gold, pred):
        return True
    gold_items = _split_top_level(_normalize(gold))
    pred_items = _split_top_level(_normalize(pred))
    if not gold_items or not pred_items:
        return _atom_equiv(_normalize(gold), _normalize(pred))
    return all(any(_atom_equiv(x, y) for y in pred_items) for x in gold_items) and all(
        any(_atom_equiv(y, x) for x in gold_items) for y in pred_items
    )
