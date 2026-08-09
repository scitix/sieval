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

from loguru import logger

#: Report key naming the metric ``score`` was taken from.
SCORE_KEY_FIELD = "score_key"

#: Report key naming the POPULATION the headline is averaged over. Two values,
#: because reports split exactly two ways and the divergence is upstream-driven
#: rather than accidental (RFC #74 F): unifying them would change ``score`` for
#: eight tasks and break comparability with every stored number, so the
#: convention is made explicit instead.
DENOMINATOR_FIELD = "denominator_policy"

#: Every sample the run asked for -- ``finals + fails`` -- so a pipeline failure
#: counts as wrong. DeepSeek's full-set accuracy convention.
DENOMINATOR_REQUESTED = "requested"

#: Only the samples that produced a verdict; failures are excluded from the
#: denominator rather than counted against the model.
DENOMINATOR_JUDGED = "judged"


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


def pass_pow_k(n: int, c: int, k: int) -> float:
    """Unbiased ``pass^k``: all ``k`` of a random ``k``-subset correct.

    ``C(c, k) / C(n, k)``, the same hypergeometric family as :func:`pass_at_k`
    and its opposite direction. ``pass@k`` is an upper bound, so a model whose
    sampling variance grew can score HIGHER on it while being worse to ship;
    this is the one that falls when that happens, which is why a delivery check
    reads the pair rather than either alone (RFC #74, motivation 2).

    Same ``n < k`` guard: a short draw scores 0 here too, and ``n_short`` is what
    distinguishes that from a model that genuinely could not repeat itself.
    """
    if n <= 0 or n < k or k <= 0 or c < k:
        return 0.0
    probability = 1.0
    for i in range(k):
        probability *= (c - i) / (n - i)
    return probability


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
        # Reported only beside `pass@k`: at k=1 the two collapse onto `pass@1`
        # and three names for one number is not three pieces of evidence.
        out["pass^k"] = pass_pow_k(n, c, k)
    if answers is not None and k == n:
        out["maj@k"] = majority_at_k(correct, answers, normalize=normalize)
    return out


def zero_metrics(*, n: int, k: int, votes: bool = True) -> dict[str, float]:
    """The key set of a clean run at ``n`` / ``k``, valued zero.

    For the path where nothing was scored -- no samples, or every one of them
    failed. A column that exists only when a run happened to produce samples is
    worse than one reading 0.0: it turns "everything failed" into a KeyError in
    whatever reads the report, which is the shape a failed run is least able to
    afford.

    Derived by running :func:`rollout_metrics` over a draw of *n* wrong,
    unextracted rollouts rather than by listing the keys, so the empty path
    cannot drift from the populated one.

    *votes* mirrors whether the caller hands :func:`rollout_metrics` its answers.
    A task that never does -- the code family, where two correct programs are not
    one answer -- must not grow a ``maj@k`` column here that its scored path
    would never report.
    """
    return rollout_metrics([False] * n, [None] * n if votes else None, k=k)


def rollout_view(final) -> tuple[list[bool], list[str | None] | None]:
    """One judged sample's per-rollout verdicts and extracted answers.

    The verdicts come from the judgement, the answers from the prediction record
    -- two stages, so they can disagree on length. When they do, the answers come
    back as ``None`` rather than as a partial list, which makes
    :func:`rollout_metrics` omit ``maj@k`` instead of voting on a draw it cannot
    see whole. That happens for real: a run launched with
    ``record_each_stage=False`` and then resumed hydrates a judgement without the
    prediction record that produced it.

    Answers are read with ``.get``, not ``[]``: ``prediction=None`` means "could
    not extract" and serialization drops the key entirely, so on disk it is
    absent rather than null (``.claude/rules/records.md``).
    """
    verdicts = (final.feedback_result or {}).get("rollouts") or []
    correct = [bool(v.get("correct")) for v in verdicts]
    predictions = (final.postprocess_result or {}).get("rollouts") or []
    if len(predictions) != len(correct):
        return correct, None
    answers = [p.get("prediction") for p in predictions]
    return correct, [None if a is None else str(a) for a in answers]


def warn_unscored_rollouts(finals, *, knob: str) -> int:
    """Count -- and complain about -- draws the headline does not score.

    A task with no sampling budget of its own still RECEIVES one: ``agenerate``
    merges ``{**model_kwargs, **kwargs}``, so ``n`` set on the model reaches a
    task that passes none, and the extra choices are generated and paid for.
    Grading and recording them is strictly better than discarding them at
    ``inf.texts[0]``, but the headline still scores the first alone -- these
    benchmarks publish a single-draw number -- so the gap has to be said out
    loud rather than left to whoever reconciles the token bill.

    Late by design: this is a report-time count, not a per-sample warning, which
    would fire once per row. Raising instead would be wrong -- one model config
    legitimately serves a sampling math task and a single-draw MCQ task in the
    same run.
    """
    extra = sum(
        max(0, len((final.feedback_result or {}).get("rollouts") or []) - 1)
        for final in finals
    )
    if extra:
        logger.warning(
            "{} rollout(s) beyond the first were graded and recorded but do NOT "
            "reach this task's headline, which scores one draw per sample. They "
            "were generated and billed: set `n` per task ({}) rather than on the "
            "model if you did not mean to sample here.",
            extra,
            knob,
        )
    return extra


def count_unextracted(finals) -> int:
    """Rollouts whose answer could not be recovered from the response.

    Separates MODEL error from PARSER error: a run whose score dropped because
    the extractor stopped matching looks identical, in every other key, to one
    whose model got worse. Counted per rollout rather than per sample, since one
    bad draw in four is a different fact from four (RFC #74 C).

    ``extracted`` is the durable flag -- ``prediction=None`` is dropped by
    serialization, so on disk the key is absent rather than null.
    """
    return sum(
        1
        for final in finals
        for rollout in (final.postprocess_result or {}).get("rollouts") or []
        if not rollout.get("extracted")
    )


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


def first_rollout_correct(finals) -> int:
    """How many judged samples the FIRST rollout got right.

    The upstream-comparable count for a benchmark whose published number was
    generated once, greedily. Deliberately NOT ``pass@1``: under ``n > 1`` that
    is ``c/n`` over the whole draw -- a better estimator of the same quantity,
    but not the one the paper reports. A task aligned to such a number keeps this
    as its headline and reports the estimator beside it, which is also why the
    two must not be quietly merged.

    At ``n = 1`` they are the same number, so adopting a sampling budget never
    moves a score that was recorded without one.
    """
    count = 0
    for final in finals:
        rollouts = (final.feedback_result or {}).get("rollouts") or []
        if rollouts and rollouts[0].get("correct"):
            count += 1
    return count


def sampling_report(
    finals,
    *,
    n: int,
    k: int,
    denominator: int,
    normalize: Callable[[str], str] | None = None,
    votes: bool = True,
    unit: str = "sample",
) -> dict[str, float]:
    """Every sampling key a task reports, for one run's judged samples.

    The whole block, not a piece of it: read each sample, estimate per problem,
    average over *denominator*, then name the budget. Tasks differ in what they
    call their headline and which population they average over -- so
    *denominator* is a parameter (RFC #74 F) -- but not in any of the above, and
    a task assembling it by hand is how two columns stop meaning the same thing.

    ``pass@1`` is ALWAYS present, so a task whose headline is ``pass@1`` can read
    it back out at any *n* and merge the rest only when there was a draw to
    describe.

    *votes* off omits ``maj@k`` end to end: the code family has no single answer
    to vote on (RFC #74, "Out of scope"), and the key must be absent from the
    empty path too or a failed run grows a column a scored one never had.
    """
    per_problem: list[dict[str, float]] = []
    observed: list[int] = []
    for final in finals:
        correct, answers = rollout_view(final)
        observed.append(len(correct))
        per_problem.append(
            rollout_metrics(
                correct, answers if votes else None, k=k, normalize=normalize
            )
        )
    # `per_problem`, not `denominator`: a run whose every sample FAILED has a
    # non-zero denominator and nothing to aggregate, and needs the full key set
    # just as much as a run with no samples at all.
    rolled = (
        aggregate(per_problem, denominator)
        if per_problem
        else zero_metrics(n=n, k=k, votes=votes)
    )
    health = {"n_unextracted": float(count_unextracted(finals))}
    return rolled | budget_metrics(observed, n=n, k=k, unit=unit) | health


def budget_metrics(
    observed: Sequence[int], *, n: int, k: int, unit: str = "sample"
) -> dict[str, float]:
    """The sampling budget as report keys: ``n``, ``k``, ``n_short``.

    Reported once per run rather than folded into the metric names, which is what
    lets a key carry a literal ``k``. A run at ``n=4`` and a paper number at
    ``n=16`` otherwise land in the same column with nothing to tell them apart.

    Emitted together because they are read together -- ``n_short`` is how many
    ``observed`` draws came back below *n*, and it is meaningless without the *n*
    it is short of. *unit* names what was counted in the warning, since a task may
    sample something narrower than a sample (UGMathBench draws per *version*).
    """
    metrics = {"n": float(n), "k": float(k)}
    short = count_short(observed, n)
    metrics["n_short"] = float(short)
    if short:
        logger.warning(
            "{}/{} {}(s) came back with fewer than the requested n={} rollout(s); "
            "they contribute 0 to pass@k and bias every sampling metric downward.",
            short,
            len(observed),
            unit,
            n,
        )
    return metrics
