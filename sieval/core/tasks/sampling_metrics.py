"""Shared estimators for multi-sample generative tasks.

Three metrics, three different questions, and reporting only the first has been the
status quo:

``pass_at_k``
    The unbiased estimator from Chen et al. 2021 (Codex), ``1 - C(n-c, k) / C(n, k)``
    written as a running product to avoid overflow. ``pass@1`` is ``c / n`` -- the
    unbiased single-sample rate, NOT "the first sample's verdict" -- and ``pass@n`` is
    "solved at least once".

``majority_at_k``
    Self-consistency: is the modal ANSWER a correct one? This cannot be derived from
    verdicts alone. A problem answered right twice with one answer and wrong twice with
    two different answers is a majority WIN, and a verdict tally sees only 2/4.

``avg_at_k``
    The plain mean of the verdicts. Numerically identical to ``pass_at_k(n, c, 1)``;
    provided so a caller can say which question it is asking rather than relying on the
    identity.

Before this module, ``_pass_at_k`` was copy-pasted into 18 task files, byte-identical in
all of them, and two task families (UGMathBench, PlatinumBench) had no k/n support at
all -- so the metric a run could report depended on which task it happened to be.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for *c* correct out of *n* samples.

    Returns 0.0 when ``n < k`` -- unreachable through config, which rejects ``k > n``;
    only a model returning fewer choices than requested lands there.
    """
    if n <= 0 or n < k or c <= 0:
        return 0.0
    if c >= n:
        return 1.0
    prob_all_wrong = 1.0
    for i in range(k):
        prob_all_wrong *= (n - c - i) / (n - i)
    return 1.0 - prob_all_wrong


def avg_at_k(correct: Sequence[bool]) -> float:
    """Mean verdict over the samples. Equals :func:`pass_at_k` with ``k=1``."""
    return (sum(1 for x in correct if x) / len(correct)) if correct else 0.0


def majority_at_k(
    correct: Sequence[bool],
    answers: Sequence[str | None],
    *,
    normalize=None,
) -> float:
    """1.0 when the modal answer is a correct one, else 0.0.

    *answers* are compared as strings after *normalize* (identity by default). Callers
    that grade symbolically should pass their own normaliser: string equality splits
    ``1/2`` from ``0.5`` and will UNDER-count agreement, which biases this metric down,
    not up.

    A TIE IS NOT A MAJORITY. When two or more answers share the top count there is no
    consensus, and this returns 0.0. Breaking the tie toward whichever answer the
    sampler happened to emit first would make the metric depend on sample ORDER -- with
    four distinct answers it would report a majority win purely because the correct one
    was drawn first, and a re-run with the same model could report the opposite.

    Empty and missing answers do not vote: a sample that produced nothing is not
    evidence for the empty string.
    """
    if not correct or len(correct) != len(answers):
        return 0.0
    norm = normalize or (lambda x: x)
    votes: collections.Counter[str] = collections.Counter()
    for a in answers:
        s = norm(str(a)) if a is not None else ""
        if s.strip():
            votes[s] += 1
    if not votes:
        return 0.0
    ranked = votes.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return 0.0
    top = ranked[0][0]
    return 1.0 if any(ok and a is not None and norm(str(a)) == top
                      for ok, a in zip(correct, answers)) else 0.0


def sampling_metrics(
    correct: Sequence[bool],
    answers: Sequence[str | None] | None = None,
    k: int = 1,
    *,
    normalize=None,
) -> dict[str, float]:
    """``{"pass@1": …, "pass@k": …, "avg@k": …, "maj@k": …}`` for one problem.

    ``pass@k`` is omitted when ``k <= 1`` (it would duplicate ``pass@1``) and ``maj@k``
    when no answers are supplied, so a caller never has to publish a metric it cannot
    actually compute. A metric reported as 0.0 because its input was missing is
    indistinguishable from a real 0.0.
    """
    n = len(correct)
    c = sum(1 for x in correct if x)
    out = {"pass@1": pass_at_k(n, c, 1), "avg@k": avg_at_k(correct)}
    if k > 1:
        out[f"pass@{k}"] = pass_at_k(n, c, k)
    if answers is not None:
        out[f"maj@{k}"] = majority_at_k(correct, answers, normalize=normalize)
    return out
