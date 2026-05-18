"""
Unit tests for sieval/core/tasks/consts.py.

Verifies enum values, mapping completeness and internal consistency:
- ACTION_TO_RESULT_STAGE covers every TaskAction
- STAGE_TO_RESULT_FIELD covers every stage that has a result
- ERROR_ACTION_PREV_STAGE covers every TaskAction
- ERROR_ACTION_CLEAR_FIELDS is monotonically shrinking toward later actions
- STAGE_ORDER contains every TaskStage value exactly once
- ERROR_REASONS_NON_RETRIABLE is a non-empty set of strings

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from sieval.core.tasks.consts import (
    ACTION_TO_RESULT_STAGE,
    ERROR_ACTION_CLEAR_FIELDS,
    ERROR_ACTION_PREV_STAGE,
    ERROR_REASONS_NON_RETRIABLE,
    STAGE_ORDER,
    STAGE_TO_RESULT_FIELD,
    TaskAction,
    TaskStage,
)


# ===================================================================
# Enum integrity
# ===================================================================
class TestEnumValues:
    def test_task_stage_str_values(self):
        assert str(TaskStage.INITIAL) == "initial"
        assert str(TaskStage.PREPROCESSED) == "preprocessed"
        assert str(TaskStage.INFERRED) == "inferred"
        assert str(TaskStage.POSTPROCESSED) == "postprocessed"
        assert str(TaskStage.FEEDBACK) == "feedback"
        assert str(TaskStage.FINAL) == "final"
        assert str(TaskStage.FAILED) == "failed"

    def test_task_action_str_values(self):
        assert str(TaskAction.PREPROCESS) == "preprocess"
        assert str(TaskAction.INFER) == "infer"
        assert str(TaskAction.POSTPROCESS) == "postprocess"
        assert str(TaskAction.FEEDBACK) == "feedback"


# ===================================================================
# ACTION_TO_RESULT_STAGE
# ===================================================================
class TestActionToResultStage:
    def test_all_actions_covered(self):
        for action in TaskAction:
            assert action in ACTION_TO_RESULT_STAGE, (
                f"TaskAction.{action} missing from ACTION_TO_RESULT_STAGE"
            )

    def test_correct_mappings(self):
        assert ACTION_TO_RESULT_STAGE[TaskAction.PREPROCESS] == TaskStage.PREPROCESSED
        assert ACTION_TO_RESULT_STAGE[TaskAction.INFER] == TaskStage.INFERRED
        assert ACTION_TO_RESULT_STAGE[TaskAction.POSTPROCESS] == TaskStage.POSTPROCESSED
        assert ACTION_TO_RESULT_STAGE[TaskAction.FEEDBACK] == TaskStage.FEEDBACK


# ===================================================================
# STAGE_TO_RESULT_FIELD
# ===================================================================
class TestStageToResultField:
    _expected = {
        TaskStage.PREPROCESSED: "preprocess_result",
        TaskStage.INFERRED: "infer_result",
        TaskStage.POSTPROCESSED: "postprocess_result",
        TaskStage.FEEDBACK: "feedback_result",
    }

    def test_all_result_stages_covered(self):
        for stage, field in self._expected.items():
            assert stage in STAGE_TO_RESULT_FIELD, (
                f"TaskStage.{stage} missing from STAGE_TO_RESULT_FIELD"
            )
            assert STAGE_TO_RESULT_FIELD[stage] == field

    def test_consistent_with_action_to_result_stage(self):
        """
        Result stages in ACTION_TO_RESULT_STAGE must appear in STAGE_TO_RESULT_FIELD
        """
        for action, stage in ACTION_TO_RESULT_STAGE.items():
            assert stage in STAGE_TO_RESULT_FIELD, (
                f"Stage {stage} (from action {action}) not in STAGE_TO_RESULT_FIELD"
            )


# ===================================================================
# ERROR_ACTION_PREV_STAGE
# ===================================================================
class TestErrorActionPrevStage:
    def test_all_actions_covered(self):
        for action in TaskAction:
            assert action in ERROR_ACTION_PREV_STAGE

    def test_correct_prev_stages(self):
        assert ERROR_ACTION_PREV_STAGE[TaskAction.PREPROCESS] == TaskStage.INITIAL
        assert ERROR_ACTION_PREV_STAGE[TaskAction.INFER] == TaskStage.PREPROCESSED
        assert ERROR_ACTION_PREV_STAGE[TaskAction.POSTPROCESS] == TaskStage.INFERRED
        assert ERROR_ACTION_PREV_STAGE[TaskAction.FEEDBACK] == TaskStage.POSTPROCESSED

    def test_prev_stage_is_result_of_prior_action(self):
        """
        prev_stage[action] must equal result_stage[prior_action], where applicable.
        """
        pipeline = [
            TaskAction.PREPROCESS,
            TaskAction.INFER,
            TaskAction.POSTPROCESS,
            TaskAction.FEEDBACK,
        ]
        for i, action in enumerate(pipeline[1:], start=1):
            prior_action = pipeline[i - 1]
            assert (
                ERROR_ACTION_PREV_STAGE[action] == ACTION_TO_RESULT_STAGE[prior_action]
            )


# ===================================================================
# ERROR_ACTION_CLEAR_FIELDS
# ===================================================================
class TestErrorActionClearFields:
    def test_all_actions_covered(self):
        for action in TaskAction:
            assert action in ERROR_ACTION_CLEAR_FIELDS

    def test_later_actions_clear_fewer_fields(self):
        """Earlier failures clear more downstream fields."""
        pipeline = [
            TaskAction.PREPROCESS,
            TaskAction.INFER,
            TaskAction.POSTPROCESS,
            TaskAction.FEEDBACK,
        ]
        counts = [len(ERROR_ACTION_CLEAR_FIELDS[a]) for a in pipeline]
        for i in range(len(counts) - 1):
            assert counts[i] > counts[i + 1], (
                f"Expected {pipeline[i]} to clear more fields than {pipeline[i + 1]}"
            )

    def test_feedback_clears_only_feedback_result(self):
        assert ERROR_ACTION_CLEAR_FIELDS[TaskAction.FEEDBACK] == ["feedback_result"]

    def test_preprocess_clears_all_result_fields(self):
        fields = ERROR_ACTION_CLEAR_FIELDS[TaskAction.PREPROCESS]
        for expected in [
            "preprocess_result",
            "infer_result",
            "postprocess_result",
            "feedback_result",
        ]:
            assert expected in fields


# ===================================================================
# STAGE_ORDER
# ===================================================================
class TestStageOrder:
    def test_contains_all_stages(self):
        for stage in TaskStage:
            assert stage in STAGE_ORDER, f"TaskStage.{stage} missing from STAGE_ORDER"

    def test_no_duplicates(self):
        assert len(STAGE_ORDER) == len(set(STAGE_ORDER))

    def test_initial_is_first(self):
        assert STAGE_ORDER[0] == TaskStage.INITIAL

    def test_final_is_last(self):
        assert STAGE_ORDER[-1] == TaskStage.FINAL

    def test_pipeline_order(self):
        """Core pipeline stages must appear in execution order."""
        ordered = [
            TaskStage.INITIAL,
            TaskStage.PREPROCESSED,
            TaskStage.INFERRED,
            TaskStage.POSTPROCESSED,
            TaskStage.FEEDBACK,
        ]
        indices = [STAGE_ORDER.index(s) for s in ordered]
        assert indices == sorted(indices)


# ===================================================================
# ERROR_REASONS_NON_RETRIABLE
# ===================================================================
class TestErrorReasonsNonRetriable:
    def test_is_non_empty_set(self):
        assert isinstance(ERROR_REASONS_NON_RETRIABLE, set)
        assert len(ERROR_REASONS_NON_RETRIABLE) > 0

    def test_contains_iteration_limit(self):
        assert "iteration_limit" in ERROR_REASONS_NON_RETRIABLE

    def test_contains_retry_limit(self):
        assert "retry_limit" in ERROR_REASONS_NON_RETRIABLE
