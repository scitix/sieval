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
ERROR_REASONS_NON_RETRIABLE = {"iteration_limit", "retry_limit"}
