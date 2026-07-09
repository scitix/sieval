# adapted from https://github.com/EnvCommons/IMO-Bench/blob/66b014f1b3799972ddfc32dbacea51b802586141/answer_verification.py
"""IMO-Bench (Google DeepMind) AnswerBench answer verification.

IMO-AnswerBench grades a short answer against the gold with ``math_verify`` and a
normalized-string fallback when either side cannot be parsed (or ``verify()``
raises).

Authoritative benchmark: google-deepmind/superhuman (``imobench/``) +
imobench.github.io + arXiv 2511.01846. The OFFICIAL AnswerBench grader is an LLM
autograder (AnswerAutoGrader, Gemini 2.5 Pro; paper §2.3/§5.1); the paper
deliberately rejects symbolic/SymPy-style matching as too narrow, and the official
``imobench/`` dir ships DATA ONLY (no runnable grader code). Since there is no
official runnable grader, this file is vendored from ``EnvCommons/IMO-Bench`` — a
third-party OpenReward re-implementation whose deterministic ``verify_math_answer``
uses ``math_verify`` (the "No LLM graders / math_verify" wording is EnvCommons's
own README, NOT the paper). This deterministic grader is a deliberate, reproducible
SUBSTITUTE for the official LLM autograder (which sieval's reproducibility contract
precludes); it is strictly MORE conservative — it cannot grade prose / infinite-set
/ functional-family answers, so sieval UNDER-counts vs official. ``verify_math_answer``
/ ``parse_answer`` are behaviorally identical to the pinned EnvCommons source
(imports made lazy per sieval discipline), unchanged pinned-commit -> upstream HEAD.

``math_verify`` is imported lazily inside the functions so importing a task
module stays free of the optional ``[math]`` dependency (sieval import discipline);
upstream imports it at module top.

``verify_math_answer`` is the verbatim upstream grader. ``normalize_answer`` below
is a sieval-added *parsing* helper (not upstream) for the generative port: it
turns the model's verbose ``\\boxed{}`` answer into the clean string an agent would
submit, so the grader (math_verify) does all the equivalence itself.
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
# sieval gen-mode answer normalization (NOT upstream) — a PARSING concern.
#
# Upstream AnswerBench is agentic: the agent submits a clean answer string, so
# verify_math_answer / math_verify grade it directly. In our generative port the
# model writes the answer inside \boxed{} verbosely ($-wrapped, function-prefixed,
# \left/\right, ^\circ degrees, trailing "for any real c" qualifiers, \{...\}
# sets). normalize_answer reconstructs the clean string an agent would submit; the
# task then grades with the vendored verify_math_answer above, so math_verify does
# ALL the equivalence (commutativity, factoring, set-equality). No bespoke matching.
#
# Conservative by design (acceptance: 0 false positives): only a TRAILING
# ``\text{... for|where|such|with ...}`` qualifier is stripped, never inline
# ``\text`` (a piecewise ``\text{ if } x \ge 2`` is meaningful and is kept).
# ---------------------------------------------------------------------------

_DEGREE = re.compile(r"\^\{?\\circ\}?")
_SPACING = re.compile(r"\\left|\\right|\\displaystyle|\\!|\\,|\\;|\\:")
_TRAILING_QUALIFIER = re.compile(
    r"\\text\s*\{[^{}]*\b(?:for|where|such|with)\b[^{}]*\}\s*$", re.I
)
_FN_PREFIX = re.compile(r"^\s*[A-Za-z]\s*\(\s*[A-Za-z0-9]\s*\)\s*=\s*")


def normalize_answer(s: str | None) -> str | None:
    """Reconstruct the clean answer an agent would submit (parsing only — no math
    decisions), so the vendored ``verify_math_answer`` can judge equivalence.

    Strips ``$`` wrapping, ``^\\circ``, ``\\left``/``\\right``/spacing macros, a
    leading ``f(x)=`` function prefix and a trailing
    ``\\text{… for/where/such/with …}`` qualifier; rewrites ``\\{…\\}`` to ``{…}``
    which math_verify parses as a set. Returns ``None`` if nothing is left.
    """
    if s is None:
        return None
    s = s.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    s = s.replace("$", "")
    s = _DEGREE.sub("", s)
    s = _SPACING.sub(" ", s)
    s = _TRAILING_QUALIFIER.sub("", s)
    s = _FN_PREFIX.sub("", s)
    s = s.replace("\\{", "{").replace("\\}", "}")
    s = re.sub(r"\s+", " ", s).strip().rstrip(".").strip()
    return s or None
