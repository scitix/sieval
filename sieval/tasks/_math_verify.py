"""Answer comparison shared by the math-competition tasks.

Extracted rather than copied twelve times because the twelve call sites were
already byte-identical and share a contract that has to change together — gold
first, both sides ``$``-wrapped. The cost of that is worth stating plainly:
**editing** :func:`verify_answer` **rotates the verdicts of all twelve benchmarks
at once** (AIME x3, HMMT x3, Apex x2, BRUMO, CMIMC, SMT, MATH-500), where the
duplication it replaced let them drift apart deliberately. A change there needs
the same before/after count a scorer change in any one of them would need.

:func:`normalize_vote` is on the other side of that line: nothing grades through
it, so a change there moves ``maj@k`` and no verdict.

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


def normalize_vote(answer: str) -> str:
    """Cluster key for ``maj@k``, so ``\\dfrac{1}{2}`` and ``\\frac{1}{2}`` vote once.

    Uses the canonicalizer this family already applies to its golds, which puts
    votes and references in one space. It is **not** the grader: ``maj@k`` votes
    on strings where :func:`verify_answer` compares symbolically, so ``0.5`` and
    ``\\frac{1}{2}`` still split into two clusters. The residual bias is therefore
    downward — a real majority can be missed, a false one cannot be manufactured —
    which makes the reported ``maj@k`` a lower bound rather than an estimate.

    Total by construction. ``strip_string`` is written for curated references and
    indexes into the string it is repairing (``\\frac`` and ``\\sqrt`` at the very
    end of the input both raise ``IndexError``), whereas this runs on raw model
    output at the end of a scored run. An un-normalized vote key costs one
    cluster; an exception costs the whole report.
    """
    from sieval.community.math import strip_string

    try:
        return strip_string(answer)
    except Exception:
        return answer
