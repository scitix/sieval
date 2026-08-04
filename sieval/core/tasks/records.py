"""Stage-output protocol: uniform record shapes for prompts, predictions, judgements.

A Task's stage return types are free-form generics, so historically every task
invented its own shape for "the extracted answer" and "was it right". Nothing
downstream could read a sample's answer, ground truth, or correctness without
knowing which task produced it. These record types are the shared contract that
fixes that, and the builders below are the only supported way to construct them.

Stage coverage:
    preprocess  -> :class:`PromptRecord`
    infer       -> ``ModelOutput`` (already uniform; deliberately not re-boxed)
    postprocess -> :class:`PredictionRecord`
    feedback    -> :class:`JudgementRecord`

Two properties are load-bearing and easy to break:

**Records are returned bare, never wrapped in ``TaskStageOutput``.** The runner
preserves that box as the stage value, so a single boxed task would persist its
result nested under ``value`` with a ``__sieval_cls__`` marker while its peers
persist a flat record -- exactly the divergence this protocol exists to remove.

**``obj_to_dict`` drops ``None``-valued keys**, so ``prediction: None`` and
``reference: None`` are *absent* on disk rather than present-and-null (``False``
and ``0`` survive -- the check is ``is not None``). That is why ``extracted`` and
``n_correct``/``n_rollouts`` are explicit fields instead of things a reader
derives from ``prediction is None``: the flags and counts are what survive the
wire. Read optional fields with ``.get()``.

Adoption is per-task and incremental: legacy shapes keep working, and
:func:`is_prediction_record` / :func:`is_judgement_record` are the single place
where "is this a protocol record?" is decided.

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
        reference: Ground truth as known at prompt-build time -- including one
            knowable *only* here, such as the correct letter after a per-sample
            choice permutation. Recorded even when it is a plain dataset field,
            so a prompt row is readable without joining to the feedback row;
            :attr:`JudgementRecord.reference` separately records the ground truth
            *as compared*, and the two are expected to coexist. Omit it when the
            ground truth is not a value at all (a test suite, a rubric).
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
        prediction: The extracted answer, or ``None`` when extraction failed.
            Absent on disk when ``None`` -- read ``extracted`` instead.
        extracted: Whether an answer was recovered from the model output.
            The durable signal, since a ``None`` prediction does not survive
            serialization.
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
        correct: The headline binary verdict. Multi-valued grades (a three-way
            CORRECT/INCORRECT/NOT_ATTEMPTED, a strict-vs-loose pair) collapse to
            this for cross-task comparability and keep their full form in
            ``extra``.
        score: Partial credit in ``[0, 1]``, when the verdict has a notion of
            one. Omitted by pass/fail verdicts -- absent is not zero.
        extra: Verdict-mechanism-specific detail -- an LLM grader's reply, a
            code runner's failure message and resource metrics, a constraint
            checker's per-constraint results. Named for the mechanism, not for a
            grader, since a string-compare or math-verify verdict has no grader.
    """

    index: int
    correct: bool
    score: NotRequired[float]
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
        extra: Sample-level verdict detail (a category breakdown, a test-suite
            description for a procedural reference).
    """

    reference: NotRequired[JSONValue | None]
    rollouts: list[RolloutJudgement]
    n_rollouts: int
    n_correct: int
    score: NotRequired[float]
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


def build_rollout_judgement(
    index: int,
    correct: bool,
    *,
    score: float | None = None,
    extra: dict | None = None,
) -> RolloutJudgement:
    """Build a :class:`RolloutJudgement`. Omits absent optional keys."""
    judgement: RolloutJudgement = {"index": index, "correct": correct}
    if score is not None:
        judgement["score"] = score
    if extra:
        judgement["extra"] = extra
    return judgement


def build_judgement_record(
    reference: JSONValue | None,
    rollouts: Sequence[RolloutJudgement],
    *,
    score: float | None = None,
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
