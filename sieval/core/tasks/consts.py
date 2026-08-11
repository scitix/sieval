from enum import StrEnum


class TaskStage(StrEnum):
    """Pipeline stage a sample can be in.

    Progression: INITIAL → PREPROCESSED → INFERRED → POSTPROCESSED → FEEDBACK → FINAL.
    FAILED is a terminal state reachable from any stage.
    """

    INITIAL = "initial"
    PREPROCESSED = "preprocessed"
    INFERRED = "inferred"
    POSTPROCESSED = "postprocessed"
    FEEDBACK = "feedback"
    FINAL = "final"
    FAILED = "failed"


class TaskAction(StrEnum):
    """Executable stage actions that advance a sample through the pipeline."""

    PREPROCESS = "preprocess"
    INFER = "infer"
    POSTPROCESS = "postprocess"
    FEEDBACK = "feedback"


ACTION_TO_RESULT_STAGE = {
    TaskAction.PREPROCESS: TaskStage.PREPROCESSED,
    TaskAction.INFER: TaskStage.INFERRED,
    TaskAction.POSTPROCESS: TaskStage.POSTPROCESSED,
    TaskAction.FEEDBACK: TaskStage.FEEDBACK,
}
STAGE_TO_RESULT_FIELD = {
    TaskStage.PREPROCESSED: "preprocess_result",
    TaskStage.INFERRED: "infer_result",
    TaskStage.POSTPROCESSED: "postprocess_result",
    TaskStage.FEEDBACK: "feedback_result",
}
STAGE_ORDER = [
    TaskStage.INITIAL,
    TaskStage.PREPROCESSED,
    TaskStage.INFERRED,
    TaskStage.POSTPROCESSED,
    TaskStage.FEEDBACK,
    TaskStage.FAILED,
    TaskStage.FINAL,
]
STAGE_RANK: dict[TaskStage, int] = {
    stage: rank for rank, stage in enumerate(STAGE_ORDER)
}
DEPENDENCY_STAGE_RANKS: tuple[tuple[TaskStage, int], ...] = tuple(
    (stage, STAGE_RANK[stage]) for stage in STAGE_TO_RESULT_FIELD
)
ERROR_ACTION_PREV_STAGE = {
    TaskAction.PREPROCESS: TaskStage.INITIAL,
    TaskAction.INFER: TaskStage.PREPROCESSED,
    TaskAction.POSTPROCESS: TaskStage.INFERRED,
    TaskAction.FEEDBACK: TaskStage.POSTPROCESSED,
}
ERROR_ACTION_CLEAR_FIELDS = {
    TaskAction.PREPROCESS: [
        "preprocess_result",
        "infer_result",
        "postprocess_result",
        "feedback_result",
    ],
    TaskAction.INFER: ["infer_result", "postprocess_result", "feedback_result"],
    TaskAction.POSTPROCESS: ["postprocess_result", "feedback_result"],
    TaskAction.FEEDBACK: ["feedback_result"],
}
#: Reason recorded when a stage raised :class:`NonRetriableSampleError`. One
#: fixed string, not ``non_retriable::<ClassName>``: the detail lives in the
#: exception's own message, which ``error_msg`` keeps.
NON_RETRIABLE_REASON = "non_retriable"

#: Reasons the loader refuses to roll back for another attempt. The two limit
#: reasons are terminal by exhaustion; ``NON_RETRIABLE_REASON`` is terminal by
#: declaration -- the task said the outcome is deterministic in its input.
ERROR_REASONS_NON_RETRIABLE = {"iteration_limit", "retry_limit", NON_RETRIABLE_REASON}


class NonRetriableSampleError(Exception):
    """A stage failure that re-running the sample cannot change.

    Raise this from a task stage when the sample cannot be completed for a
    reason fixed in its own input -- a ground truth that will not parse, a row
    the grader has no way to judge. The runner records
    :data:`NON_RETRIABLE_REASON` instead of ``exception::<ClassName>``, and the
    loader leaves such a sample ``FAILED`` on resume rather than rolling it back,
    so a deterministic defect is not re-inferred once per resume until
    ``max_retries`` runs out.

    The declaration is the *type*, not a name: ``core`` cannot enumerate task
    exception classes without knowing about ``sieval.tasks``, so the runner tests
    ``isinstance`` and any subclass inherits the behaviour.

    It says nothing about scoring. The sample is still ``FAILED``, and a task
    declaring ``DENOMINATOR_REQUESTED`` still counts it in the denominator --
    excluding a sample from the denominator would need a terminal state that
    does not exist yet.
    """
