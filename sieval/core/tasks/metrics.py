"""Shared estimators for multi-sample generative tasks.

One entry point, so the numbers stay comparable across tasks. What each key
means and which pairs must be read together is in ``docs/guide/metrics.md``;
what a reader of THIS module needs is the three rules the code enforces:

* Keys carry a literal ``k``, never its value, so a leaderboard column keeps its
  identity when the budget changes. The budget is reported once, as ``n`` / ``k``.
* A key is omitted, never zeroed, when it cannot be computed -- a 0.0 meaning
  "not measurable" is indistinguishable from one meaning "measured, and zero".
* ``pass@1`` is ``c/n``, NOT the first sample's verdict. Tasks aligned to a
  published single-draw number keep that separately, via
  :func:`first_rollout_correct`.

See RFC #74 (scitix/sieval) for the metric family and its scope.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import collections
from collections.abc import Callable, Sequence

from loguru import logger

#: Report key naming the metric ``score`` was taken from.
SCORE_KEY_FIELD = "score_key"

#: Report key naming the POPULATION the headline is averaged over. Declared
#: rather than unified: the split is upstream-driven, and unifying it would
#: change ``score`` for eight tasks (RFC #74 F).
DENOMINATOR_FIELD = "denominator_policy"

#: Every sample the run asked for -- ``finals + fails`` -- so a pipeline failure
#: counts as wrong. DeepSeek's full-set accuracy convention.
DENOMINATOR_REQUESTED = "requested"

#: Only the samples that produced a verdict; failures are excluded from the
#: denominator rather than counted against the model.
DENOMINATOR_JUDGED = "judged"


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k: at least one of ``k`` correct, ``1 - C(n-c,k)/C(n,k)``.

    A running product, to avoid overflow. Returns 0.0 when ``n < k`` -- a short
    draw, which :func:`count_short` surfaces rather than let read as a zero.
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
    """Unbiased ``pass^k``: all ``k`` of a random ``k``-subset, ``C(c,k)/C(n,k)``.

    :func:`pass_at_k`'s opposite direction. That one is an upper bound and RISES
    with sampling variance, so a model that got less reliable can score better on
    it; this is what falls instead. Read the pair, never either alone.

    Same ``n < k`` guard, so a short draw scores 0 here too.
    """
    if n <= 0 or n < k or k <= 0 or c < k:
        return 0.0
    probability = 1.0
    for i in range(k):
        probability *= (c - i) / (n - i)
    return probability


def avg_at_n(correct: Sequence[bool]) -> float:
    """Mean verdict over the WHOLE draw -- ``n``, not ``k``.

    Spelled ``@n`` because it takes no ``k`` and does not vary with one: at
    ``n=4, k=2`` this averages four verdicts, where :func:`pass_at_k` estimates
    over two. Same suffix on two keys meaning two different things is how a
    reader concludes the pair is redundant.

    It coincides with ``pass@1`` on every boolean draw, and the two are still
    both reported, because they answer different questions: ``pass@1`` ESTIMATES
    the success rate of one draw, this MEASURES the mean of the draw that was
    paid for. They separate the moment a verdict stops being a bool.
    """
    return (sum(1 for x in correct if x) / len(correct)) if correct else 0.0


def majority_at_k(
    correct: Sequence[bool],
    answers: Sequence[str | None],
    *,
    normalize: Callable[[str], str] | None = None,
) -> float:
    """1.0 when the modal answer is a correct one, else 0.0.

    Answers are compared as strings after *normalize* (identity by default), so
    a caller that grades symbolically should pass its own -- string equality
    splits ``1/2`` from ``0.5``, biasing this down, never up.

    A TIE IS NOT A MAJORITY: breaking one toward whichever answer came first
    would make the metric depend on sample order, so a re-run could report the
    opposite. Empty and missing answers do not vote.
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


def self_consistency(
    answers: Sequence[str | None],
    *,
    normalize: Callable[[str], str] | None = None,
) -> float:
    """Share of the draw that landed in the modal ANSWER cluster.

    The dispersion metric ``maj@k`` cannot be, because that one is thresholded:
    4/4 agreeing and 3/4 agreeing both score 1.0. Only this moves when a
    converted or requantized model's answers widen without its mean changing.

    CORRECTNESS-BLIND: a consistently wrong model scores 1.0. Read it beside a
    correctness key, never instead of one.

    The denominator is the WHOLE draw, so an unextracted rollout drags it down --
    an unextracted rollout is not evidence of stability. That conflates model
    instability with parser failure, which is what :func:`count_unextracted`
    beside it separates.
    """
    if not answers:
        return 0.0
    norm = normalize or (lambda x: x)
    votes: collections.Counter[str] = collections.Counter()
    for answer in answers:
        if answer is None:
            continue
        vote = norm(str(answer)).strip()
        if vote:
            votes[vote] += 1
    if not votes:
        return 0.0
    return votes.most_common(1)[0][1] / len(answers)


def rollout_metrics(
    correct: Sequence[bool],
    answers: Sequence[str | None] | None = None,
    k: int = 1,
    *,
    normalize: Callable[[str], str] | None = None,
    n_requested: int | None = None,
) -> dict[str, float]:
    """Per-problem ``pass@1`` / ``avg@n`` / ``pass@k`` / ``pass^k`` / ``maj@k``.

    A key is OMITTED rather than set to 0.0 when it cannot be computed, since a
    0.0 for lack of input is indistinguishable from a real one:

    * ``pass@k`` and ``pass^k`` when ``k <= 1`` -- they would restate ``pass@1``.
    * ``maj@k`` without answers, or when ``k`` does not cover the whole REQUESTED
      draw. Majority is defined over the whole draw, and sub-sampling it would
      need an estimator or a seed (RFC #74 D.2).

    The gate is on the BUDGET, not on what arrived. A draw that came back short
    still votes: the count is run health, which every other key here treats as
    "compute it, annotate it with ``n_short``" rather than as grounds to withhold
    a column. ``self_consistency`` clusters the very same answers with the very
    same normalizer, so gating one on arrival and not the other would give two
    answers about whether one draw is fit to cluster.

    *n_requested* is the budget that was ASKED for, which is not always what came
    back. Without it the two are indistinguishable and a draw that truncated to
    exactly ``k`` looks like a full-budget one -- inverted, since ``k < n`` means
    the caller asked for a sub-sample majority that has no definition. It
    defaults to the observed count, the right reading for a caller that already
    knows the draw is complete (:func:`zero_metrics` synthesizes one).
    """
    n = len(correct)
    requested = n if n_requested is None else n_requested
    c = sum(1 for x in correct if x)
    out = {"pass@1": pass_at_k(n, c, 1), "avg@n": avg_at_n(correct)}
    if k > 1:
        out["pass@k"] = pass_at_k(n, c, k)
        # Reported only beside `pass@k`: at k=1 the two collapse onto `pass@1`
        # and three names for one number is not three pieces of evidence.
        out["pass^k"] = pass_pow_k(n, c, k)
    if answers is not None:
        out["self_consistency"] = self_consistency(answers, normalize=normalize)
        # Rejects a majority over a SUB-SAMPLE of the budget: at k=2, n=4 there
        # is no answer to "which 2", and picking needs a seed. Deliberately not
        # also gated on the arrived count -- see the docstring.
        if k == requested:
            out["maj@k"] = majority_at_k(correct, answers, normalize=normalize)
    return out


def zero_metrics(*, n: int, k: int, votes: bool = True) -> dict[str, float]:
    """The key set of a clean run at ``n`` / ``k``, valued zero.

    For the path where nothing was scored -- no samples, or every one of them
    failed. A column that exists only when a run produced samples turns
    "everything failed" into a KeyError downstream, which is the shape a failed
    run can least afford.

    Derived by running :func:`rollout_metrics` over a synthetic draw rather than
    by listing keys, so the empty path cannot drift from the populated one.
    *votes* mirrors whether the caller passes answers, so a task that never does
    cannot grow a ``maj@k`` here that its scored path would never report.
    """
    return rollout_metrics([False] * n, [None] * n if votes else None, k=k)


def rollout_view(final) -> tuple[list[bool], list[str | None] | None]:
    """One judged sample's per-rollout verdicts and extracted answers.

    Verdicts come from the judgement, answers from the prediction record -- two
    stages, so they can disagree on length. When they do the answers come back as
    ``None`` rather than a partial list, so ``maj@k`` is omitted instead of voted
    on a draw that cannot be seen whole. Real case: a run launched with
    ``record_each_stage=False`` and then resumed.

    Answers are read with ``.get``, not ``[]`` -- ``prediction=None`` is dropped
    by serialization, so on disk the key is absent (``.claude/rules/records.md``).
    """
    verdicts = (final.feedback_result or {}).get("rollouts") or []
    correct = [bool(v.get("correct")) for v in verdicts]
    predictions = (final.postprocess_result or {}).get("rollouts") or []
    if len(predictions) != len(correct):
        return correct, None
    answers = [p.get("prediction") for p in predictions]
    return correct, [None if a is None else str(a) for a in answers]


def warn_unscored_rollouts(finals, *, task: str) -> int:
    """Count -- and complain about -- draws the headline does not score.

    A task with no budget of its own still RECEIVES one: ``agenerate`` merges
    ``{**model_kwargs, **kwargs}``, so ``n`` set on the model reaches it and the
    extra choices are generated and paid for. They are graded and recorded, but
    the headline scores the first alone, so the gap is said out loud rather than
    left to whoever reconciles the token bill.

    Report-time rather than per-sample, which would fire once per row. Raising
    would be wrong: one model config legitimately serves a sampling math task
    and a single-draw MCQ task in the same run.

    The remedy names the MODEL config, which is the only place this ``n`` can
    have come from -- every task that emits this warning publishes a single-draw
    number and therefore takes no ``n`` argument to move it to.
    """
    extra = sum(
        max(0, len((final.feedback_result or {}).get("rollouts") or []) - 1)
        for final in finals
    )
    if extra:
        logger.warning(
            "{} rollout(s) beyond the first were graded and recorded but do NOT "
            "reach {}'s headline, which scores one draw per sample. They were "
            "generated and billed: this task publishes a single-draw number and "
            "takes no `n` of its own, so drop `n` from the MODEL config (or "
            "point this task at a model that does not set it) if you did not "
            "mean to sample here.",
            extra,
            task,
        )
    return extra


def count_unextracted(finals) -> int:
    """Rollouts whose answer could not be recovered from the response.

    Separates MODEL error from PARSER error: a run whose score dropped because
    the extractor stopped matching looks identical, in every other key, to one
    whose model got worse. Per rollout, not per sample -- one bad draw in four is
    a different fact from four (RFC #74 C).
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

    *denominator* is explicit, not ``len(per_problem)``: pass the one the task's
    headline uses, so these cover the same population as ``score``. The two
    differ whenever failed samples count as wrong.

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

    The upstream-comparable count for a benchmark whose published number was one
    greedy draw. Deliberately NOT ``pass@1``, which is ``c/n`` over the whole
    draw -- a better estimator of the same quantity, but not the one the paper
    reports, so the two are kept side by side rather than merged.

    At ``n = 1`` they coincide, so adopting a budget never moves a stored score.
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
    average over *denominator*, name the budget. Only *denominator* varies by
    task (RFC #74 F); a task assembling the rest by hand is how two columns stop
    meaning the same thing.

    ``pass@1`` is ALWAYS present, so a task whose headline it is can read it back
    out at any *n* and merge the rest only when there was a draw to describe.

    *votes* off omits ``maj@k`` end to end -- including on the empty path, or a
    failed run grows a column a scored one never had.
    """
    per_problem: list[dict[str, float]] = []
    observed: list[int] = []
    for final in finals:
        correct, answers = rollout_view(final)
        observed.append(len(correct))
        per_problem.append(
            rollout_metrics(
                correct,
                answers if votes else None,
                k=k,
                normalize=normalize,
                # The budget that was asked for. A sample that came back short
                # must not be voted on as though it were the whole draw.
                n_requested=n,
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
    return rolled | budget_metrics(observed, n=n, k=k, unit=unit)


def health_metrics(finals) -> dict[str, float]:
    """Extraction health, reported at EVERY budget -- including ``n = 1``.

    Deliberately not part of :func:`sampling_report`: ``n_unextracted`` measures
    the parser, not the draw, so gating it behind ``n > 1`` would withhold it
    from the default configuration -- the one where a silently-stopped extractor
    survives longest, because there is no second rollout to disagree with.

    Its own function rather than a key inside the sampling block, so a task with
    no sampling budget at all (the MCQ four) can still report it.
    """
    return {"n_unextracted": float(count_unextracted(finals))}


def budget_metrics(
    observed: Sequence[int], *, n: int, k: int, unit: str = "sample"
) -> dict[str, float]:
    """The sampling budget as report keys: ``n``, ``k``, ``n_short``.

    Reported once rather than folded into metric names, which is what lets a key
    carry a literal ``k``: a run at ``n=4`` and a paper number at ``n=16``
    otherwise land in the same column with nothing to tell them apart.

    Together because they are read together -- ``n_short`` is meaningless without
    the *n* it is short of. *unit* names what was counted in the warning, since a
    task may sample something narrower than a sample (UGMathBench, per *version*).
    """
    metrics = {"n": float(n), "k": float(k)}
    short = count_short(observed, n)
    metrics["n_short"] = float(short)
    # Only warn about a BUDGET that came up short. At n=1 "short" means zero
    # rollouts -- a pipeline failure, already visible as a sample that scored
    # nothing, and not a fact about sampling. Warning there would also fire for
    # only half the family, since the tasks whose headline is not `pass@1` skip
    # this block entirely at n=1.
    if short and n > 1:
        logger.warning(
            "{}/{} {}(s) came back with fewer than the requested n={} rollout(s); "
            "they contribute 0 to pass@k and bias every sampling metric downward.",
            short,
            len(observed),
            unit,
            n,
        )
    return metrics
