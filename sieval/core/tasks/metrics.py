"""Shared estimators for multi-sample generative tasks.

One entry point, so the numbers stay comparable across tasks:

``pass@1``
    Unbiased single-sample rate (Chen et al. 2021). "What one draw is worth",
    NOT "the first sample's verdict".
``avg@k``
    Mean verdict over the draw. Numerically equal to ``pass@1``, kept separate
    because the two answer different questions.
``pass@k``
    Solved at least once in ``k`` draws -- an upper bound, so higher sampling
    variance can raise it while making a model worse to ship.
``maj@k``
    Self-consistency: is the modal ANSWER correct? Not derivable from verdicts
    alone -- right twice with one answer and wrong twice with two different ones
    is a majority WIN that a verdict tally sees as 2/4.

Keys carry a literal ``k``, never its value, so a leaderboard column does not
change identity when the budget does. The budget is reported once, as the ``n``
and ``k`` fields.

See RFC #74 (scitix/sieval) for the metric family and its scope.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import collections
from collections.abc import Callable, Sequence

#: Report key naming the metric ``score`` was taken from.
SCORE_KEY_FIELD = "score_key"


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for *c* correct out of *n* samples.

    ``1 - C(n-c, k) / C(n, k)``, as a running product to avoid overflow. Returns
    0.0 when ``n < k`` -- a model that returned fewer choices than requested,
    which :func:`count_short` exists to surface rather than let read as a zero.
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
    """Mean verdict over the draw. Equals :func:`pass_at_k` at ``k=1``."""
    return (sum(1 for x in correct if x) / len(correct)) if correct else 0.0


def majority_at_k(
    correct: Sequence[bool],
    answers: Sequence[str | None],
    *,
    normalize: Callable[[str], str] | None = None,
) -> float:
    """1.0 when the modal answer is a correct one, else 0.0.

    Answers are compared as strings after *normalize* (identity by default).
    Callers that grade symbolically should pass their own: string equality splits
    ``1/2`` from ``0.5``, which biases this metric down, not up.

    A TIE IS NOT A MAJORITY. Breaking ties toward whichever answer was emitted
    first would make the metric depend on sample order, so a re-run of the same
    model could report the opposite. (RFC #74 D.3 proposes a lowest-index
    tie-break; it is still order-dependent, so the stricter rule stands here.)

    Empty and missing answers do not vote.
    """
    if not correct or len(correct) != len(answers):
        return 0.0
    norm = normalize or (lambda x: x)

    def vote_key(answer: str | None) -> str:
        # Trim after normalising, or "18" and "18 " split a unanimous majority.
        return norm(str(answer)).strip() if answer is not None else ""

    votes: collections.Counter[str] = collections.Counter()
    for answer in answers:
        vote = vote_key(answer)
        if vote:
            votes[vote] += 1
    if not votes:
        return 0.0
    ranked = votes.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return 0.0
    top = ranked[0][0]
    winner_is_correct = any(
        ok and vote_key(answer) == top
        for ok, answer in zip(correct, answers, strict=True)
    )
    return 1.0 if winner_is_correct else 0.0


def rollout_metrics(
    correct: Sequence[bool],
    answers: Sequence[str | None] | None = None,
    k: int = 1,
    *,
    normalize: Callable[[str], str] | None = None,
) -> dict[str, float]:
    """Per-problem ``pass@1`` / ``avg@k`` / ``pass@k`` / ``maj@k``.

    A key is OMITTED rather than set to 0.0 when it cannot be computed, since a
    0.0 for lack of input is indistinguishable from a real one:

    * ``pass@k`` when ``k <= 1`` -- it would restate ``pass@1`` exactly.
    * ``maj@k`` without answers, or when ``k != len(correct)``: majority is
      defined over the whole draw, and sub-sampling it would need an estimator
      or a seed (RFC #74 D.2).
    """
    n = len(correct)
    c = sum(1 for x in correct if x)
    out = {"pass@1": pass_at_k(n, c, 1), "avg@k": avg_at_k(correct)}
    if k > 1:
        out["pass@k"] = pass_at_k(n, c, k)
    if answers is not None and k == n:
        out["maj@k"] = majority_at_k(correct, answers, normalize=normalize)
    return out


def count_short(observed: Sequence[int], n: int) -> int:
    """Samples that came back with fewer than the *n* rollouts requested.

    They score 0 for ``pass@k`` and bias every metric downward, so the count
    belongs in the report rather than in a log line (RFC #74 C).
    """
    return sum(1 for count in observed if count < n)


def aggregate(
    per_problem: Sequence[dict[str, float]],
    denominator: int,
    *,
    scale: float = 100.0,
) -> dict[str, float]:
    """Mean of the per-problem metrics over *denominator* problems.

    *denominator* is taken explicitly, not as ``len(per_problem)``: pass the one
    the task's headline metric uses, so these cover the same population as
    ``score``. The two differ whenever failed samples count as wrong.

    A key present for only SOME problems is dropped, not averaged -- summing what
    exists over a fixed denominator would turn a deliberate omission back into
    the 0.0 it was avoiding.
    """
    if denominator <= 0 or not per_problem:
        return {}
    totals: dict[str, float] = collections.defaultdict(float)
    counts: collections.Counter[str] = collections.Counter()
    for metrics in per_problem:
        for key, value in metrics.items():
            totals[key] += value
            counts[key] += 1
    complete = len(per_problem)
    return {
        key: total * scale / denominator
        for key, total in totals.items()
        if counts[key] == complete
    }
