"""The math-verify comparison shared by the math-competition tasks.

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
