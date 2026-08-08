"""The math-verify comparison shared by the math-competition tasks.

Extracted rather than copied twelve times because the twelve call sites were
already byte-identical and share a contract that has to change together — gold
first, both sides ``$``-wrapped. The cost of that is worth stating plainly:
**editing this function rotates the verdicts of all twelve benchmarks at once**
(AIME x3, HMMT x3, Apex x2, BRUMO, CMIMC, SMT, MATH-500), where the duplication
it replaced let them drift apart deliberately. A change here needs the same
before/after count a scorer change in any one of them would need.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""


def verify_answer(gold: str, pred: str) -> bool:
    """Compare one answer pair. Module-level so a worker process can pickle it.

    Must run in a process: math-verify's ``signal.SIGALRM`` bound only arms on
    the main thread — criterion 1 in :mod:`sieval.core.utils.offload`.
    """
    from math_verify import parse, verify

    # math_verify.verify takes the gold answer first.
    return bool(verify(parse(gold), parse(pred)))
