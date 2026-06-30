# adapted from https://github.com/EnvCommons/IMO-Bench/blob/66b014f1b3799972ddfc32dbacea51b802586141/answer_verification.py
"""IMO-Bench (Google DeepMind) AnswerBench answer verification.

IMO-AnswerBench grades a short answer against the gold with ``math_verify`` and a
normalized-string fallback when either side cannot be parsed (or ``verify()``
raises). Vendored from the upstream ``answer_verification.py``.

``math_verify`` is imported lazily inside the functions so importing a task
module stays free of the optional ``[math]`` dependency (sieval import discipline);
upstream imports it at module top.
"""


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
