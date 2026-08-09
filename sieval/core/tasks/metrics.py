"""Shared estimators for multi-sample generative tasks.

One entry point for every task that draws ``n`` rollouts per problem, so the
numbers stay comparable across tasks. Four metrics, four different questions --
reporting only the first has been the status quo:

``pass@1``
    The unbiased single-sample rate from Chen et al. 2021 (Codex),
    ``1 - C(n-c, k) / C(n, k)`` at ``k=1``, written as a running product to avoid
    overflow. This is "what one draw is worth", NOT "the first sample's verdict".

``avg@k``
    The plain mean of the verdicts over the drawn rollouts. Numerically equal to
    ``pass@1``, and kept as a separate key on purpose: the two answer different
    questions, and collapsing them because the arithmetic coincides would lose
    the distinction between "the value of one draw" and "what this run actually
    averaged". The ``n`` / ``k`` report fields say which budget it averaged over.

``pass@k``
    The optimistic direction -- solved at least once in ``k`` draws. An upper
    bound, so a model with higher sampling variance can score *higher* here while
    being worse to ship.

``maj@k``
    Self-consistency: is the modal ANSWER a correct one? This cannot be derived
    from verdicts alone -- a problem answered right twice with one answer and
    wrong twice with two different answers is a majority WIN, and a verdict tally
    sees only 2/4.

Key names carry a literal ``k``, not its value: ``pass@k``, never ``pass@4``. The
budget is reported once, in the ``n`` / ``k`` fields, so a reader never has to
reconcile a column name against a config and a leaderboard column does not change
identity when the budget does.

See RFC #74 (scitix/sieval) for the metric family and its intended scope.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import collections
from collections.abc import Callable, Sequence

#: Report key naming the metric ``score`` was taken from, so a reader never has
#: to guess which column the headline number came from.
SCORE_KEY_FIELD = "score_key"


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for *c* correct out of *n* samples.

    Returns 0.0 when ``n < k`` -- a model that returned fewer choices than were
    requested. Callers that sample should surface that case rather than let it
    read as a genuine zero; see :func:`count_short`.
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
    """Mean verdict over the drawn rollouts. Equals :func:`pass_at_k` at ``k=1``."""
    return (sum(1 for x in correct if x) / len(correct)) if correct else 0.0


def majority_at_k(
    correct: Sequence[bool],
    answers: Sequence[str | None],
    *,
    normalize: Callable[[str], str] | None = None,
) -> float:
    """1.0 when the modal answer is a correct one, else 0.0.

    *answers* are compared as strings after *normalize* (identity by default).
    Callers that grade symbolically should pass their own normaliser: string
    equality splits ``1/2`` from ``0.5`` and will UNDER-count agreement, which
    biases this metric down, not up.

    A TIE IS NOT A MAJORITY. When two or more answers share the top count there
    is no consensus, and this returns 0.0. Breaking the tie toward whichever
    answer the sampler happened to emit first would make the metric depend on
    sample ORDER -- with four distinct answers it would report a majority win
    purely because the correct one was drawn first, and a re-run with the same
    model could report the opposite.

    (RFC #74 D.3 proposes a lowest-rollout-index tie-break instead. That is
    deterministic given a stored rollout order, but still order-dependent, so the
    stricter rule stands here until the RFC settles.)

    Empty and missing answers do not vote: a sample that produced nothing is not
    evidence for the empty string.
    """
    if not correct or len(correct) != len(answers):
        return 0.0
    norm = normalize or (lambda x: x)

    def vote_key(answer: str | None) -> str:
        """One answer's vote key: normalised, then trimmed.

        Trimming after normalising matters -- otherwise ``"18"`` and ``"18 "``
        stand as two candidates and split a majority a reader would call
        unanimous.
        """
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

    Keys are literal; the budget is reported once in the task's ``n`` / ``k``
    fields rather than encoded into every column name.

    A key is OMITTED rather than set to 0.0 when it cannot be computed, because a
    metric that is 0.0 for lack of input is indistinguishable from a real 0.0:

    * ``pass@k`` when ``k <= 1`` -- it would restate ``pass@1`` exactly.
    * ``maj@k`` when no answers are supplied, or when ``k != len(correct)``.
      Majority is defined over the whole draw; scoring it at ``k < n`` would mean
      sub-sampling, which needs either an unbiased estimator or a seed -- and a
      seed in the metric layer is a fresh source of irreproducibility (RFC #74 D.2).
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
    """How many samples came back with fewer than the *n* rollouts requested.

    Short samples score 0 for ``pass@k`` and bias every metric downward. Upstream
    this is only ever a ``logger.warning``; returning it lets a task put the count
    in the report, where a reader can actually see it (RFC #74 C).
    """
    return sum(1 for count in observed if count < n)


def aggregate(
    per_problem: Sequence[dict[str, float]],
    denominator: int,
    *,
    scale: float = 100.0,
) -> dict[str, float]:
    """Mean of the per-problem metrics, over *denominator* problems.

    *denominator* is the task's own population: pass the same one the task's
    headline metric uses, so the sampling metrics cover the same set as ``score``.
    It is taken explicitly rather than derived from ``len(per_problem)`` because
    the two differ whenever failed samples count as wrong.

    A key present for only SOME problems is dropped, not averaged. The per-problem
    contract omits what it cannot compute, and summing the ones that do exist over
    a fixed denominator would quietly turn those omissions back into 0.0 -- the
    exact thing omitting them was protecting against.
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
