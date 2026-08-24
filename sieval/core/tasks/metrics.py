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
import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass

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


@dataclass(frozen=True)
class ProblemGrouping:
    """Which problem each judged sample belongs to, and how many there are.

    The two travel together because neither is usable alone: the keys say how to
    collapse the samples, and *n_problems* says what to divide by afterwards --
    read off the whole split, so a problem whose every copy failed still occupies
    a slot and the population does not shrink with run health.
    """

    keys: list[Hashable]
    n_problems: int


def problem_population(
    grouping: ProblemGrouping | None, finals, *, n_problems: int
) -> ProblemGrouping:
    """A grouping that reports ``n_problems`` in PROBLEMS, repeated split or not.

    For the tasks whose headline is averaged over ROLLOUTS rather than over
    samples. Their ``denominator`` is a rollout count, and passing it to
    :func:`interval_metrics` with no grouping would publish that rollout count as
    ``n_problems`` -- so the field would mean "problems x n" on an unrepeated
    split and "problems" once ``Dataset.repeat`` was applied, i.e. a different
    NOUN for the same task depending on its config. One field, two units, is what
    makes it uncomparable against a task reporting questions.

    So the grouping is always supplied for those tasks: *grouping* when the split
    is repeated (its ``n_problems`` already counts distinct problems), and
    otherwise one key per judged sample with *n_problems* given by the caller --
    the sample population, which is what one problem is there.

    This does NOT move any interval. ``n_problems`` is inert in the arithmetic:
    :func:`interval_metrics` scales each group's sum by ``n_problems/denominator``
    and then divides by ``n_problems``, so the factor cancels out of both ``p``
    and the variance. It corrects only what the field REPORTS.

    Its own function rather than four lines at each of the six call sites: they
    have to agree on the unit to stay comparable with each other and with the
    sample-denominator tasks, so the agreement lives in one place.
    """
    if grouping is not None:
        return grouping
    # POSITIONAL keys, not `sample_id`: `problem_groups` returning None means
    # "each judged sample is its own problem", and one key per position says
    # exactly that. Keying on `sample_id` would say "group by sample identity" --
    # the same thing only while those ids are distinct, and silently collapsing
    # two samples into one problem (halving the group count, and dropping the
    # interval entirely at two samples) if they ever are not.
    return ProblemGrouping(list(range(len(finals))), n_problems)


def wilson_interval(
    values: Sequence[float],
    denominator: int,
    *,
    z: float = 1.96,
    scale: float = 100.0,
) -> tuple[float, float] | None:
    """A 95% interval on ``sum(values) / denominator``, clustered on *values*.

    The resampling unit is one element of *values* -- one PROBLEM. Pooling the
    rollouts of one problem as independent trials understates the width, and
    understates it more the more the model varies per problem.

    Wilson on an effective sample size, ``m_eff = p(1-p)/Var``, rather than a Wald
    half-width: the half-width puts the lower bound below zero exactly where
    saturated and very hard sets live (a real 1/30 run reads ``3.33 +/- 6.42``),
    and the asymmetry near a bound is the part worth reporting. With boolean
    *values* and ``denominator == len(values)`` this reduces EXACTLY to the
    textbook Wilson interval -- which is why the variance below uses the
    population divisor, not ``m - 1``.

    *denominator* is the population the headline is averaged over, which is not
    ``len(values)`` whenever failed samples count as wrong. Those are DETERMINISTIC
    zeros: they enter the mean but contribute no variance, so the variance of
    ``sum/D`` over ``m`` random terms is ``m*s**2/D**2`` -- smaller than ``s**2/m``,
    while the mean is pulled down by the same zeros. Spelling it ``s**2/m`` would
    overstate the width on any run with failures (67% at ``D=50, m=30``).

    Returns ``None`` -- omitted, never zeroed -- when there is nothing to estimate:
    fewer than two problems, or no dispersion between them. At ``p`` exactly 0 or 1
    there is no dispersion either, but that is when a reader most needs the bound,
    so those fall back to the one-sided Clopper-Pearson limit over the ``m``
    OBSERVED units.

    At ``p == 0`` that limit is rescaled by ``m/D``, which is the same correction
    the variance above makes carried into the degenerate branch: Clopper-Pearson
    bounds the mean of the ``m`` random units, while the reported number is
    ``sum/D`` over a population that also holds ``D - m`` deterministic zeros --
    so the estimand is ``m/D`` of what the raw limit bounds. Without it a
    30-problem all-wrong run with 5 failures publishes ``[0, 11.57]`` where the
    bound on ``sum/D`` is ``[0, 9.92]``. The factor is exactly 1 whenever
    ``D == m``, so the limit stays EXACT in the textbook case and the reduction
    above is untouched. The saturated branch takes no such factor -- see the
    comment there.

    No randomness, so two readers of the same values compute the same interval
    (RFC #74 D refused a seed in this layer). Order-independent, which matters
    because a resumed run rebuilds its finals in manifest order.
    """
    m = len(values)
    if m < 2 or denominator <= 0:
        return None
    total = sum(values)
    p = total / denominator
    if p <= 0.0:
        # Clopper-Pearson bounds the mean of the m OBSERVED units; the reported
        # quantity is `sum/D`, which the `D - m` deterministic zeros scale by
        # `m/D`. Exactly 1.0 when `D == m`, so the textbook reduction is
        # unaffected; below 1 on a `requested` run with failures.
        return 0.0, scale * (m / denominator) * (1.0 - 0.025 ** (1 / m))
    if p >= 1.0:
        # NO `m/D` factor here, and not by omission: values on 0-1 sum to at most
        # `m`, so `p >= 1` forces `m >= D`, and the only sane case is `m == D`
        # where the factor is 1. Applying it would multiply a LOWER bound by
        # `m/D >= 1` and could push it past `scale`, inverting the interval.
        return scale * 0.025 ** (1 / m), scale
    mean = total / m
    # Population divisor: with the sample divisor `m_eff` lands on `m - 1` and the
    # reduction to plain Wilson is off by 0.34pp at 1/30.
    spread = sum((v - mean) ** 2 for v in values) / m
    variance = m * spread / (denominator * denominator)
    if variance <= 0.0:
        return None
    m_eff = p * (1.0 - p) / variance
    centre = (p + z * z / (2 * m_eff)) / (1 + z * z / m_eff)
    half = (
        z
        / (1 + z * z / m_eff)
        * math.sqrt(p * (1 - p) / m_eff + z * z / (4 * m_eff * m_eff))
    )
    return scale * max(0.0, centre - half), scale * min(1.0, centre + half)


#: Report key carrying the interval on the headline, as ``[lo, hi]``.
SCORE_CI_FIELD = "score_ci95"

#: Report key carrying the declared problem population the interval is quoted over.
PROBLEM_COUNT_FIELD = "n_problems"

#: Suffix that turns a metric's key into the key carrying THAT metric's interval:
#: ``pass@k`` -> ``pass@k_ci95``. ``SCORE_CI_FIELD`` is this rule applied to
#: ``score``, which is why the headline pair keeps its spelling.
CI_SUFFIX = "_ci95"

#: Report key declaring, per metric, WHICH population key that metric's interval
#: is clustered on -- ``{"pass@1": "n_problems"}``. The report's one nested
#: object, and the reason a report can carry more than one interval at all: two
#: metrics on two units both need a population, and one ``n_problems`` cannot say
#: which is whose. An interval whose unit is undeclared cannot be read, so the
#: estimators here emit the declaration WITH the interval, never beside it.
CI_UNITS_FIELD = "ci95_units"

#: Prefix every population-count key in this tree carries -- ``n_problems``,
#: ``n_turns``, ``n_versions``, ``n_subjects``. What a ``ci95_units`` entry names
#: is the SIZE of the population its metric is clustered on, so the value has to
#: be one of these: a value naming another metric points at a rate, and there is
#: no count beside the interval to read it against. The same prefix marks a count
#: apart from a rate everywhere else in a report.
COUNT_KEY_PREFIX = "n_"

#: How far two report values may sit apart and still be one number published
#: under two names, in points. A report may round the rate it prints -- SciTaRC
#: rounds to 2 dp -- while the bounds bracket the unrounded mean, and 2-dp
#: rounding of a percentage moves a value by at most this much. Wide enough to
#: absorb that, far narrower than any real difference between two metrics.
ALIAS_VALUE_TOLERANCE = 0.005


def ci_field(metric: str) -> str:
    """The report key carrying *metric*'s interval."""
    return f"{metric}{CI_SUFFIX}"


def _clustered_interval(
    values: Sequence[float],
    *,
    denominator: int,
    group_keys: Sequence[Hashable] | None,
    n_problems: int | None,
) -> tuple[tuple[float, float], int] | None:
    """The interval and the population it is declared over, or ``None``.

    The estimation half of :func:`interval_metrics` / :func:`metric_interval`,
    which differ only in the KEYS they publish it under. Sharing the arithmetic
    is what keeps a per-metric interval identical to the headline interval on the
    metric the headline was copied from.

    Raises:
        ValueError: if *group_keys* is given without *n_problems*, or does not
            carry one key per value. Both would silently mis-scale the interval.
    """
    if group_keys is not None:
        if n_problems is None:
            raise ValueError(
                "interval_metrics: group_keys needs n_problems beside it -- the "
                "collapsed values are scaled by it, so guessing would mis-scale "
                "the interval."
            )
        if len(group_keys) != len(values):
            raise ValueError(
                f"interval_metrics: group_keys must carry one key per value; got "
                f"{len(group_keys)} keys for {len(values)} values."
            )
        if denominator <= 0:
            # Both paths must agree that an empty population has no interval.
            # `wilson_interval` refuses one ungrouped, but it is handed
            # `population` here, not `denominator` -- so scaling by anything at
            # all would zero every unit, land on `p == 0`, and let the
            # Clopper-Pearson branch invent a bound over a mean of nothing.
            return None
        sums: dict[Hashable, float] = collections.defaultdict(float)
        for key, value in zip(group_keys, values, strict=True):
            sums[key] += value
        scale = n_problems / denominator
        units = [total * scale for total in sums.values()]
        population = n_problems
    else:
        units = list(values)
        population = denominator
    interval = wilson_interval(units, population)
    if interval is None:
        return None
    return interval, population


def interval_metrics(
    values: Sequence[float],
    *,
    denominator: int,
    group_keys: Sequence[Hashable] | None = None,
    n_problems: int | None = None,
    aliases: Sequence[str] = (),
) -> dict[str, float | list[float] | dict[str, str]]:
    """The headline's interval and the problem population it is declared over.

    ``n_problems`` is that DECLARED population -- the denominator of the estimand,
    reported as given. It is inert in the arithmetic: ``G/D`` scales the units and
    ``G`` is the divisor, so the factor cancels out of both ``p`` and the variance.
    The width is set by the number of groups actually observed, ``len(sums)``,
    which is smaller than ``n_problems`` on a run where every copy of some problem
    failed. The two coincide on a clean run.

    *values* are the PER-SAMPLE contributions to the headline, in the caller's own
    units -- whichever quantity that task's ``score`` is a mean of. Passed in rather
    than picked here, because only the task knows which of its metrics the interval
    belongs to, and a task publishing rates on two different axes must not have one
    guessed for it.

    *group_keys* collapses samples that are not independent problems -- the copies
    ``Dataset.repeat`` makes. With ``G`` problems and declared denominator ``D``, a
    group's summed value ``v`` becomes the per-problem unit ``v * G / D``, so the
    mean is ``sum(values) / D`` either way: collapsing widens the interval and
    leaves ``score`` bit-for-bit unchanged. Unrepeated, ``G == D`` and each unit is
    its own value.

    The three keys are emitted **whole or not at all**: an interval whose
    population is unknown cannot be read, a population with no interval beside it
    is a count nothing asked for, and an interval whose unit is undeclared cannot
    be told from one clustered on something else. All are omitted -- never
    zeroed -- whenever :func:`wilson_interval` has nothing to estimate, and on a
    *denominator* of zero, which is the one case the grouped path has to refuse
    for itself.

    That is a property of THIS fragment, not of the report it lands in: a report
    folding several fragments through :func:`merge_metrics` can publish
    ``n_problems`` with a sibling metric's interval and no ``score_ci95`` at all,
    when the headline has no dispersion and the sibling does. The pair rule
    applies per metric, and one metric's omission is not half of another's pair.

    *aliases* are the other key names the SAME number is published under -- the
    column ``score_key`` says the headline was copied from (``accuracy``,
    ``acc_norm``, ``exact_match``), and any second published route to it. Each
    gets ``<alias>_ci95`` carrying this interval and a declaration on the same
    population, because a consumer keyed on the column name has no way to know
    the interval it needs is filed under ``score``. A parameter rather than a
    second call: the alias interval must BE the headline's, and two calls with
    the same arguments are only equal until one of them is edited. It is for a
    true alias only -- a metric that is a different number, even a deterministic
    function of this one, is not one of these.

    This is :func:`metric_interval` pinned to the headline: metric ``score``,
    unit ``n_problems``. It spells those two out as its own literals rather than
    delegating, so the keys it publishes can be named by a reader that only parses
    the source -- which is how ``scripts/check_preflight.py`` checks that every
    task pairs its interval with a population. ``test_metrics`` pins the two
    functions against each other, so the second spelling cannot drift.

    Raises:
        ValueError: if *group_keys* is given without *n_problems*, or does not carry
            one key per value. Both would silently mis-scale the interval.
    """
    estimated = _clustered_interval(
        values,
        denominator=denominator,
        group_keys=group_keys,
        n_problems=n_problems,
    )
    if estimated is None:
        return {}
    (low, high), population = estimated
    fields: dict[str, float | list[float] | dict[str, str]] = {
        SCORE_CI_FIELD: [low, high],
        PROBLEM_COUNT_FIELD: float(population),
        CI_UNITS_FIELD: {"score": PROBLEM_COUNT_FIELD}
        | dict.fromkeys(aliases, PROBLEM_COUNT_FIELD),
    }
    for alias in aliases:
        # A fresh list per key, not one shared object: these land in one JSON
        # document, and a shared list is a shared mutable in every reader that
        # loads the report back and edits it.
        fields[ci_field(alias)] = [low, high]
    return fields


def metric_interval(
    metric: str,
    values: Sequence[float],
    *,
    denominator: int,
    group_keys: Sequence[Hashable] | None = None,
    n_problems: int | None = None,
    unit: str = PROBLEM_COUNT_FIELD,
    aliases: Sequence[str] = (),
) -> dict[str, float | list[float] | dict[str, str]]:
    """One metric's own interval, its population, and the unit it is clustered on.

    The general form of :func:`interval_metrics`, for every metric a report
    publishes beside its headline. *values* are that metric's PER-UNIT values --
    the quantity it is the mean of -- so a metric is a candidate here only when it
    is exactly ``sum(values) / denominator``. A pooled ratio of two sums and a
    nonlinear combination of aggregates are not, and get no interval rather than a
    plausible-looking wrong one.

    *unit* names the population key the interval is clustered on, and is what the
    report declares under ``ci95_units``. It defaults to ``n_problems`` because
    that is the unit of most metrics in this tree, but it is a parameter and not a
    constant: a rate per version, per graded sample or per session is a different
    population, and reusing a problem count for one of those narrows the interval
    by the same root-times an uncollapsed repeat does.

    *aliases* are further key names the same number is published under, on the
    same terms as :func:`interval_metrics`' -- IFEval publishes one prompt-level
    rate as both ``loose_prompt_level_accuracy`` and ``loose_accuracy``, and a
    consumer keyed on either has to find a companion. A true alias only: a
    metric that is a different number gets its own call, and one that is a
    deterministic function of this one gets nothing, since a mirrored bound reads
    as second evidence and is not.

    The interval, the population and the declaration are one return value, never
    three steps a caller can half-complete -- a ``<metric>_ci95`` whose unit
    nothing recorded is unreadable, and that is the failure this shape exists to
    make impossible.

    Raises:
        ValueError: if *unit* is not a population count (:data:`COUNT_KEY_PREFIX`).
            The population is written UNDER that key, so a *unit* naming a metric
            puts a count where a report holds a rate: later-wins in the fold, and
            the rate is gone. Refused here rather than left to
            :func:`interval_declaration_problems`, because that runs on the
            finished dict -- which the runner saves BEFORE it raises, so the
            corrupted rate would already be on disk. A declaration nobody can
            read is worth saving and refusing; a number that is wrong is not.
            Checked before anything is estimated, so a bad unit fails on every
            run rather than only on the ones with dispersion to report.
    """
    if not unit.startswith(COUNT_KEY_PREFIX):
        raise ValueError(
            f"metric_interval: unit {unit!r} is not a population count -- those "
            f"are spelled {COUNT_KEY_PREFIX!r}... in this tree. The population is "
            f"published under this key, so {metric!r}'s interval would replace "
            f"whatever the report holds at {unit!r} with a count. Two names for "
            "one number are `aliases`, not a unit pointing at another metric."
        )
    estimated = _clustered_interval(
        values,
        denominator=denominator,
        group_keys=group_keys,
        n_problems=n_problems,
    )
    if estimated is None:
        return {}
    (low, high), population = estimated
    names = [metric, *aliases]
    # A fresh list per key, not one shared object -- see `interval_metrics`.
    fields: dict[str, float | list[float] | dict[str, str]] = {
        ci_field(name): [low, high] for name in names
    }
    fields[unit] = float(population)
    fields[CI_UNITS_FIELD] = dict.fromkeys(names, unit)
    return fields


def merge_metrics(
    *fragments: Mapping[str, float | list[float] | dict[str, str]],
) -> dict[str, float | list[float] | dict[str, str]]:
    """Fold report fragments left to right, UNIONING their unit declarations.

    ``a | b`` replaces ``ci95_units`` wholesale, so a report that merges two
    fragments each carrying intervals keeps only the last one's declarations and
    every interval from the other loses its unit. That is silent -- the intervals
    are all still there -- so any report folding more than one interval-bearing
    fragment goes through here instead. Every other key keeps plain merge
    semantics: later wins.

    Fragments as the estimators return them. A report dict also carries its
    declaration STRINGS (``score_key``, ``denominator_policy``), which are not
    metrics and have no place in a fold; put that dict on the left of a plain
    ``|`` over this function's result.

    Raises:
        ValueError: when two fragments declare the same metric on two different
            population keys, or hand ``ci95_units`` something other than a map.
            One metric cannot be clustered on two units, and keeping the later
            one would publish an interval over a population it was not computed
            over.
    """
    merged: dict[str, float | list[float] | dict[str, str]] = {}
    units: dict[str, str] = {}
    for fragment in fragments:
        for key, value in fragment.items():
            if key != CI_UNITS_FIELD:
                merged[key] = value
                continue
            if not isinstance(value, dict):
                raise ValueError(
                    f"merge_metrics: {CI_UNITS_FIELD!r} must be a map of metric "
                    f"to population key; got {value!r}."
                )
            for metric, unit in value.items():
                declared = units.get(metric)
                if declared is not None and declared != unit:
                    raise ValueError(
                        f"merge_metrics: {metric!r} is declared on both "
                        f"{declared!r} and {unit!r}. One metric has one "
                        "population; two mean one of the intervals is quoted "
                        "over a unit it was not computed over."
                    )
                units[metric] = unit
    if units:
        merged[CI_UNITS_FIELD] = units
    return merged


def ungated_intervals(
    block: Mapping[str, float | list[float] | dict[str, str]],
    *,
    metrics: Sequence[str],
) -> dict[str, float | list[float] | dict[str, str]]:
    """*metrics*' intervals out of *block*, with their populations and their units.

    For a report that publishes only SOME of a block's columns. The sampling block
    is merged in only at ``n > 1``, because at ``n = 1`` the rest of it restates
    ``pass@1``; but ``pass@1`` itself and the headline copied from it ARE published
    at every budget, so their intervals are lifted out of that gate with them --
    ``ungated_intervals(rolled, metrics=("score", "pass@1"))``. Every other key's
    interval stays inside the gate with the metric it brackets, since an interval
    is never published for a metric that is not.

    Keyed on METRIC NAMES, not on interval keys: the interval key is derived with
    :func:`ci_field` and the population key is read off the kept ``ci95_units``
    entry, so a block spanning two units carries the right count for each. Keying
    on a fixed set of field names cannot -- it would copy one population and
    declare another, publishing an entry naming a count that is not there.

    ``ci95_units`` is trimmed to the metrics that came along: an entry for a
    withheld metric describes a key the report does not have. Every entry kept is
    *block*'s own, so a caller that later merges the whole block cannot lose one --
    the block declares a superset.

    Raises:
        ValueError: if *block* carries a requested interval with no unit declared,
            or declares one over a population key *block* itself does not write.
            Both mean the block was not built by the estimators here, and both
            would put an unreadable interval in a report.
    """
    declared = block.get(CI_UNITS_FIELD)
    units = declared if isinstance(declared, dict) else {}
    out: dict[str, float | list[float] | dict[str, str]] = {}
    kept: dict[str, str] = {}
    for metric in metrics:
        field = ci_field(metric)
        if field not in block:
            continue
        unit = units.get(metric)
        if unit is None:
            raise ValueError(
                f"ungated_intervals: {field!r} is in the block with no unit "
                f"declared for {metric!r}; an interval whose population is "
                "unnamed cannot be read."
            )
        if unit not in block:
            raise ValueError(
                f"ungated_intervals: {metric!r} is declared over {unit!r}, which "
                "the block does not write -- there would be no population to "
                "read the interval against."
            )
        out[field] = block[field]
        out[unit] = block[unit]
        kept[metric] = unit
    if not kept:
        # Whole or not at all: a population with no interval beside it is a count
        # nothing asked for, and an undeclared interval cannot be read.
        return {}
    out[CI_UNITS_FIELD] = kept
    return out


def interval_declaration_problems(report: Mapping[str, object]) -> list[str]:
    """Every way *report*'s intervals and their unit declarations disagree.

    The run-time half of the contract, for what no static reader can check: a
    per-metric interval key is built from a metric NAME
    (``metric_interval`` writes ``ci_field(metric)``), so a source scan cannot
    enumerate the keys a report will actually publish and cannot tell an
    undeclared interval from one it simply could not see. ``check_preflight.py``
    checks the declaration EXISTS; this checks it is complete and resolvable, on
    the finished dict, where a fragment folded with a plain ``|`` has already lost
    whatever it was going to lose.

    Five ways it can be wrong -- three of them a key a reader cannot use, two a
    key a reader would misread:

    * an interval with no entry -- nothing says which population it is over;
    * an entry naming a population key the report does not write;
    * an entry naming a metric the report does not publish -- a declaration about
      a column that is not there;
    * an entry whose value is not a population COUNT (:data:`COUNT_KEY_PREFIX`).
      ``ci95_units`` maps a metric to the SIZE of its population, so a value
      naming another metric resolves -- that key is in the report -- and still
      leaves the interval with nothing to be read against.
      :func:`metric_interval` refuses that unit outright, since it is what writes
      the count under it; this half covers the map a report hand-writes, where
      there is no estimator between the declaration and the reader;
    * two metrics declared on one unit, carrying one interval, publishing two
      different numbers (:func:`_alias_value_problems`).

    Returned rather than raised, so the caller decides when: the report is worth
    saving before anything about it is refused.
    """
    problems: list[str] = []
    declared = report.get(CI_UNITS_FIELD)
    units: dict[str, object] = {}
    if declared is not None:
        if not isinstance(declared, Mapping):
            return [
                f"{CI_UNITS_FIELD!r} is {type(declared).__name__}, not a map of "
                "metric to population key"
            ]
        # `str(metric)`: the report is about to be read back off disk as JSON,
        # where every key is a string anyway.
        units = {str(metric): unit for metric, unit in declared.items()}
    for key in report:
        if key == CI_UNITS_FIELD or not key.endswith(CI_SUFFIX):
            continue
        metric = key[: -len(CI_SUFFIX)]
        if metric not in units:
            problems.append(
                f"{key!r} has no {CI_UNITS_FIELD} entry for {metric!r}, so nothing "
                "says which population it is clustered on"
            )
    for metric, unit in units.items():
        if not isinstance(unit, str) or unit not in report:
            problems.append(
                f"{CI_UNITS_FIELD} declares {metric!r} over {unit!r}, which the "
                "report does not write"
            )
        elif not unit.startswith(COUNT_KEY_PREFIX):
            # `elif`: a unit that is absent has already been reported, and the
            # more specific complaint about a key that IS there is the useful one.
            problems.append(
                f"{CI_UNITS_FIELD} declares {metric!r} over {unit!r}, which is not "
                f"a population count -- those are spelled {COUNT_KEY_PREFIX!r}... "
                "in this tree. The unit is how many units the interval is "
                "clustered on, so naming another metric points at a rate"
            )
        if metric not in report:
            problems.append(
                f"{CI_UNITS_FIELD} declares {metric!r}, which the report does not "
                "publish"
            )
    return problems + _alias_value_problems(report, units)


def _alias_value_problems(
    report: Mapping[str, object], units: Mapping[str, object]
) -> list[str]:
    """Metrics sharing one interval on one unit whose published numbers differ.

    The alias rule, checked in the one place that can: ``aliases`` names the
    other keys ONE number is published under, and the emitters file the same
    bounds under each of them. They never see the values the report will publish,
    so a caller handing them a metric that is merely RELATED -- a complement, a
    rescale, a stricter variant -- gets a mirrored bound with no complaint, and a
    mirrored bound reads as second evidence when it is the same evidence twice.

    Two metrics on the same unit have equal bounds only when they have equal
    ``p``: Wilson is monotone in it. So equal bounds beside unequal values is
    that mistake and essentially nothing else -- near-tautological for a real
    alias, and specific to a mis-passed one. Compared against
    :data:`ALIAS_VALUE_TOLERANCE` rather than exactly, because a report may round
    the rate it prints while the bounds bracket the unrounded mean.

    Only *declared* metrics are paired: an undeclared interval is already a
    problem in its own right, and its unit is unknown, so nothing can be said
    about whom it shares a bound with.
    """
    problems: list[str] = []
    # Sorted, so a report with two mis-passed aliases names them in a stable
    # order rather than in dict order.
    metrics = sorted(units)
    for index, first in enumerate(metrics):
        for second in metrics[index + 1 :]:
            unit = units[first]
            if unit != units[second]:
                continue
            # The two INTERVALS, each a `[lo, hi]` list -- not two bounds.
            first_bounds = report.get(ci_field(first))
            second_bounds = report.get(ci_field(second))
            if (
                not isinstance(first_bounds, list)
                or not isinstance(second_bounds, list)
                or first_bounds != second_bounds
            ):
                continue
            first_value = report.get(first)
            second_value = report.get(second)
            if not isinstance(first_value, (int, float)) or not isinstance(
                second_value, (int, float)
            ):
                continue
            if abs(first_value - second_value) <= ALIAS_VALUE_TOLERANCE:
                continue
            problems.append(
                f"{first!r} and {second!r} are both declared on {unit!r} and carry "
                f"the same interval, but the report publishes {first_value} and "
                f"{second_value}. One interval under two names is a true ALIAS -- "
                "the same number, two keys; a metric that is a different number "
                "owns its own estimate, and one that is a deterministic function "
                "of another owns none"
            )
    return problems


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
    score_key: str | None = None,
    grouping: ProblemGrouping | None = None,
) -> dict[str, float | list[float] | dict[str, str]]:
    """Every sampling key a task reports, for one run's judged samples.

    The whole block, not a piece of it: read each sample, estimate per problem,
    average over *denominator*, name the budget. Only *denominator* varies by
    task (RFC #74 F); a task assembling the rest by hand is how two columns stop
    meaning the same thing.

    ``pass@1`` is ALWAYS present, so a task whose headline it is can read it back
    out at any *n* and merge the rest only when there was a draw to describe.

    *votes* off omits ``maj@k`` end to end -- including on the empty path, or a
    failed run grows a column a scored one never had.

    EVERY key here carries its own interval, on the very per-problem values
    :func:`aggregate` folded into it. All six are exact means over problems of a
    per-problem value, so none of them has to borrow another's: ``pass@k`` is not
    a rescaled ``pass@1``, and publishing one interval for a block of six would
    make five of them read as measured when only one was. They all sit on the same
    population, so the block still declares one ``n_problems``.

    No interval reaches a metric this block did not compute: each one is derived
    from the metric's own key, so the gates already in that key set -- ``pass@k``
    / ``pass^k`` at ``k > 1``, ``maj@k`` on a vote over the whole budget,
    ``self_consistency`` on votes -- are not restated and cannot fall out of step.

    The converse does NOT hold, and must not: a metric here can be published with
    no interval, because :func:`wilson_interval` has nothing to estimate from a
    single unit or from units that all agree. That is the older rule of this
    module -- omitted, never zeroed -- and it wins, since a zero-width interval
    would claim a certainty the run does not have. So a run with no dispersion
    reports six metrics and five intervals, and the empty path reports the full
    key set with none at all.

    *score_key* names which of this block's OWN columns -- ``pass@1``, ``avg@n``,
    and so on -- the task's headline is a mean of, so its interval can be
    published under ``score_ci95`` as well, for a reader keyed on the headline
    rather than on the column it came from. The two are the same estimate on the
    same values, not two measurements. Passed in rather than picked here, because
    only the caller knows which axis its headline is on; left ``None``, no
    ``score_ci95`` is emitted. Given a name this block did NOT compute, it RAISES
    rather than guessing: an interval silently attached to the wrong column would
    be a plausible-looking wrong number, worse than no number at all.

    The intervals are reported at EVERY budget the block runs at, including
    ``n = 1`` -- deliberately NOT gated behind ``n > 1`` the way most of this
    file's per-run keys are. They are at their WIDEST at ``n = 1``, which is
    exactly where a reader most needs them; gating them would withhold them from
    the default configuration. :func:`health_metrics` makes the same argument for
    ``n_unextracted``. A task that withholds the block itself at ``n = 1`` lifts
    the always-published ones out with :func:`ungated_intervals`.

    *grouping* collapses samples that are repeat copies of one problem before
    estimating -- see :func:`interval_metrics`. ``None`` when each sample is
    already its own problem.
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
    budget = budget_metrics(observed, n=n, k=k, unit=unit)
    if score_key is not None and score_key not in rolled:
        raise ValueError(
            f"sampling_report: score_key {score_key!r} names a column this block "
            f"does not compute; got {sorted(rolled)}. A headline pointing at a "
            f"missing column would report an interval on a different metric."
        )
    group_keys = None if grouping is None else grouping.keys
    n_problems = None if grouping is None else grouping.n_problems
    intervals: list[dict[str, float | list[float] | dict[str, str]]] = []
    if score_key is not None:
        # Subscript, not `.get(..., 0.0)`: the guard above proved the key is in
        # `rolled`, and `aggregate` drops a key absent from any entry -- so every
        # entry has it. If that ever stops holding, raise instead of padding the
        # gap with the 0.0 this module refuses everywhere else. Same for the
        # per-metric loop below, which reads only keys `rolled` kept.
        intervals.append(
            interval_metrics(
                [metrics[score_key] for metrics in per_problem],
                denominator=denominator,
                group_keys=group_keys,
                n_problems=n_problems,
            )
        )
    # `per_problem`, not `rolled`: on the empty path `rolled` is `zero_metrics`'
    # key set, which describes no problems at all and has no values to estimate
    # from. Iterating `rolled` otherwise is what makes an interval appear exactly
    # when its metric does -- the gates on `pass@k`, `maj@k` and
    # `self_consistency` are already in the keys.
    if per_problem:
        for metric in rolled:
            intervals.append(
                metric_interval(
                    metric,
                    [metrics[metric] for metrics in per_problem],
                    denominator=denominator,
                    group_keys=group_keys,
                    n_problems=n_problems,
                )
            )
    # `merge_metrics`, not `|`: every fragment above declares the unit of the one
    # metric it estimated, and a plain merge would keep only the last one's.
    return merge_metrics(rolled, budget, *intervals)


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
