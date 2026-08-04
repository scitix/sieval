"""Stage-output protocol: uniform record shapes for prompts, predictions, judgements.

Stage return types are free-form generics, so every task used to invent its own
shape for "the extracted answer" and "was it right", and nothing downstream could
read a sample without knowing which task wrote it. These records are the shared
contract; the builders below are the only supported way to construct them.

    preprocess  -> :class:`PromptRecord`
    infer       -> ``ModelOutput`` (already uniform; deliberately not re-boxed)
    postprocess -> :class:`PredictionRecord`
    feedback    -> :class:`JudgementRecord`

A verdict has three tiers, and a value in the wrong one is a value lost:
``correct``/``score`` are the headline (``correct`` being the only axis
comparable across tasks), ``metrics`` is every measured value by name, and
``extra`` is mechanism detail plus whatever raw material an aggregation needs. A
metric parked in ``extra`` is persisted but invisible to any reader that does not
already know the task.

Two properties are load-bearing and easy to break:

**Records are returned bare, never wrapped in ``TaskStageOutput``.** The runner
preserves that box as the stage value, so one boxed task would persist its result
under ``value`` with a ``__sieval_cls__`` marker while its peers stay flat.

**``obj_to_dict`` drops ``None``-valued keys**, so a ``None`` field is *absent* on
disk rather than null (``False`` and ``0`` survive). Hence ``extracted`` and the
``n_*`` counts are explicit rather than derived, and optional fields are read with
``.get()``.

Adoption is per-task: legacy shapes keep working, and
:func:`is_prediction_record` / :func:`is_judgement_record` are the single place
that decides which is which.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from collections.abc import Mapping, Sequence
from typing import Any, NotRequired, TypedDict

from sieval.core.types import JSONValue


class PromptRecord(TypedDict):
    """What ``preprocess`` hands to ``infer``, plus what analysis needs later.

    Attributes:
        prompt: The model input, in whatever shape the model kind takes --
            a chat ``messages`` list, or a plain string for base models.
            ``infer`` reads this key rather than the record.
        reference: Ground truth as known at prompt-build time, including one
            knowable only here (the correct letter after a per-sample choice
            permutation). Recorded even for a plain dataset field, so a prompt row
            reads on its own; coexisting with :attr:`JudgementRecord.reference`
            (the truth *as compared*) is intended, not redundant. Omit when the
            ground truth is not a value (a test suite, a rubric).
        extra: Task-specific prompt-side detail worth keeping (the permutation
            actually used, the constraint spec, a category label).
    """

    prompt: JSONValue
    reference: NotRequired[JSONValue | None]
    extra: NotRequired[dict]


class RolloutPrediction(TypedDict):
    """One rollout's extracted answer.

    Attributes:
        index: Position of this rollout within the sample (``0`` when ``n=1``).
        prediction: The extracted answer, or ``None`` when extraction failed --
            absent on disk in that case, so read ``extracted`` instead.
        extracted: Whether an answer was recovered. The durable signal.
        extra: Per-rollout detail from the extraction step.
    """

    index: int
    prediction: NotRequired[JSONValue | None]
    extracted: bool
    extra: NotRequired[dict]


class PredictionRecord(TypedDict):
    """What ``postprocess`` produced for one sample, across all its rollouts.

    Attributes:
        rollouts: One entry per model rollout, in rollout order.
        extra: Sample-level detail from the extraction step.
    """

    rollouts: list[RolloutPrediction]
    extra: NotRequired[dict]


class RolloutJudgement(TypedDict):
    """One rollout's verdict.

    Attributes:
        index: Position of this rollout within the sample, matching the
            corresponding :class:`RolloutPrediction`.
        correct: The headline binary verdict, and the only axis comparable across
            tasks -- ``n_correct`` derives from it. A multi-valued grade collapses
            to this and keeps its full form in ``extra``.
        score: Headline partial credit in ``[0, 1]``. Omitted by pass/fail
            verdicts -- absent is not zero.
        metrics: Every metric measured, by name, including the ones the headline
            points at. A task with co-equal metrics (IFEval strict *and* loose,
            HellaSwag ``acc`` *and* ``acc_norm``) records them all here and stays
            enumerable. Derive the headline *from* this mapping so the two cannot
            drift.
        extra: Mechanism detail, not metrics -- a grader's reply, a code runner's
            failure message, per-constraint results -- plus the raw counts an
            aggregation needs, since a per-sample rate cannot reconstruct a pooled
            one. Named for the mechanism: a string-compare verdict has no grader.
    """

    index: int
    correct: bool
    score: NotRequired[float]
    metrics: NotRequired[dict[str, bool | float]]
    extra: NotRequired[dict]


class JudgementRecord(TypedDict):
    """What ``feedback`` concluded for one sample.

    Sample-level facts live here rather than being repeated per rollout, so a
    ground truth is stored once regardless of ``n``.

    Attributes:
        reference: The ground truth as compared. ``None`` when the reference is
            a *procedure* rather than a value (a test suite, a rubric) -- in
            that case describe it in ``extra``. Absent on disk when ``None``.
        rollouts: One verdict per rollout, in rollout order.
        n_rollouts: Number of rollouts judged. Materialized (rather than left to
            ``len(rollouts)``) so a row stays self-describing once flattened.
        n_correct: How many of them were correct. With ``n_rollouts`` this is the
            sample-level pass rate.
        score: Sample-level partial credit in ``[0, 1]``, when the verdict has
            one. Reserved for genuine partial credit -- do not mirror
            ``n_correct / n_rollouts`` here.
        metrics: Sample-level counterpart of :attr:`RolloutJudgement.metrics`,
            same rule.
        extra: Sample-level detail (a category breakdown, a test-suite description
            for a procedural reference) and aggregation raw material -- not
            metrics.
    """

    reference: NotRequired[JSONValue | None]
    rollouts: list[RolloutJudgement]
    n_rollouts: int
    n_correct: int
    score: NotRequired[float]
    metrics: NotRequired[dict[str, bool | float]]
    extra: NotRequired[dict]


def build_prompt_record(
    prompt: JSONValue,
    *,
    reference: JSONValue | None = None,
    extra: dict | None = None,
) -> PromptRecord:
    """Build a :class:`PromptRecord`. Omits absent optional keys."""
    record: PromptRecord = {"prompt": prompt}
    if reference is not None:
        record["reference"] = reference
    if extra:
        record["extra"] = extra
    return record


def build_prediction_record(
    predictions: Sequence[JSONValue | None],
    *,
    extra: dict | None = None,
) -> PredictionRecord:
    """Build a :class:`PredictionRecord` from one extracted value per rollout.

    ``extracted`` is derived per rollout as ``prediction is not None``, so tasks
    must use ``None`` -- not ``""`` or ``-1`` -- for "could not extract".
    """
    rollouts: list[RolloutPrediction] = [
        {"index": index, "prediction": prediction, "extracted": prediction is not None}
        for index, prediction in enumerate(predictions)
    ]
    record: PredictionRecord = {"rollouts": rollouts}
    if extra:
        record["extra"] = extra
    return record


def _checked_metrics(metrics: Mapping[str, bool | float]) -> dict[str, bool | float]:
    """Reject metric values that would not survive serialization.

    A ``None`` metric would be *absent* on disk, turning "not measured" into
    "never existed" -- a bug at the call site, so it fails loud. Non-numeric
    values are rejected for the same reason: that is detail, and detail belongs
    in ``extra``.
    """
    unrecordable = sorted(k for k, v in metrics.items() if v is None)
    if unrecordable:
        raise ValueError(
            f"metric(s) {unrecordable} are None; None-valued keys are dropped on "
            "serialization, so the metric would be absent on disk rather than "
            "recorded. Omit it, or record a real value."
        )
    mistyped = sorted(
        k for k, v in metrics.items() if not isinstance(v, (bool, int, float))
    )
    if mistyped:
        raise ValueError(
            f"metric(s) {mistyped} are not bool/number; `metrics` holds measured "
            "values only. Put structured detail in `extra` instead."
        )
    return dict(metrics)


def build_rollout_judgement(
    index: int,
    correct: bool,
    *,
    score: float | None = None,
    metrics: Mapping[str, bool | float] | None = None,
    extra: dict | None = None,
) -> RolloutJudgement:
    """Build a :class:`RolloutJudgement`. Omits absent optional keys.

    Pass every metric measured as *metrics*, and derive *correct*/*score* from it.
    """
    judgement: RolloutJudgement = {"index": index, "correct": correct}
    if score is not None:
        judgement["score"] = score
    if metrics:
        judgement["metrics"] = _checked_metrics(metrics)
    if extra:
        judgement["extra"] = extra
    return judgement


def build_judgement_record(
    reference: JSONValue | None,
    rollouts: Sequence[RolloutJudgement],
    *,
    score: float | None = None,
    metrics: Mapping[str, bool | float] | None = None,
    extra: dict | None = None,
) -> JudgementRecord:
    """Build a :class:`JudgementRecord`, deriving ``n_rollouts`` / ``n_correct``."""
    entries = list(rollouts)
    record: JudgementRecord = {
        "reference": reference,
        "rollouts": entries,
        "n_rollouts": len(entries),
        "n_correct": sum(1 for entry in entries if entry["correct"]),
    }
    if score is not None:
        record["score"] = score
    if metrics:
        record["metrics"] = _checked_metrics(metrics)
    if extra:
        record["extra"] = extra
    return record


def is_prediction_record(value: Any) -> bool:
    """Whether *value* is a postprocess-stage protocol record.

    The place the legacy-vs-protocol distinction is decided, for both in-memory
    values and records rehydrated from disk (which come back as plain mappings).
    Callers that also accept a ``TaskStageOutput`` must unwrap first.

    Both record kinds carry ``rollouts``; only a judgement carries the
    materialized ``n_rollouts``, so its absence is what distinguishes a
    prediction from a judgement rather than merely from a legacy shape.
    """
    return (
        isinstance(value, Mapping) and "rollouts" in value and "n_rollouts" not in value
    )


def is_judgement_record(value: Any) -> bool:
    """Whether *value* is a feedback-stage protocol record.

    Keys on ``n_rollouts``, which :func:`build_judgement_record` always
    materializes (even for zero rollouts) and a prediction never has -- so this
    genuinely tells a judgement from a prediction, not just from a legacy shape.
    """
    return isinstance(value, Mapping) and "n_rollouts" in value
