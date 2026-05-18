"""
Tests for TaskLoader._prepare_failed_retries: retry logic, stage reset,
field clearing, and record_each_stage behavior.

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

from sieval.core.tasks.consts import TaskAction, TaskStage
from sieval.core.tasks.context import TaskContext
from sieval.core.tasks.loader import TaskLoader

from .conftest import make_ctx, make_mock_task


class TestPrepareFailedRetries:
    def _make_loader(self, tmp_path):
        task = make_mock_task()
        return TaskLoader(task=task, root_dir=tmp_path / "run")

    def test_retriable_failure_resets_stage(self, tmp_path):
        loader = self._make_loader(tmp_path)
        ctx = make_ctx(
            0,
            TaskStage.FAILED,
            error_action=TaskAction.INFER,
            error_reason="exception::TimeoutError",
        )
        contexts = {0: ctx}
        loader._prepare_failed_retries(contexts, record_each_stage=True)

        updated = contexts[0]
        assert updated.stage == TaskStage.PREPROCESSED  # prev stage of INFER
        assert updated.retry_count == 1
        assert updated.error_action is None
        assert updated.error_reason is None

    def test_non_retriable_failure_unchanged(self, tmp_path):
        loader = self._make_loader(tmp_path)
        ctx = make_ctx(
            0,
            TaskStage.FAILED,
            error_action=TaskAction.FEEDBACK,
            error_reason="iteration_limit",
        )
        contexts = {0: ctx}
        loader._prepare_failed_retries(contexts, record_each_stage=True)
        assert contexts[0].stage == TaskStage.FAILED

    def test_no_error_action_unchanged(self, tmp_path):
        loader = self._make_loader(tmp_path)
        ctx = make_ctx(0, TaskStage.FAILED)
        contexts = {0: ctx}
        loader._prepare_failed_retries(contexts, record_each_stage=True)
        assert contexts[0].stage == TaskStage.FAILED

    def test_retry_increments_count(self, tmp_path):
        loader = self._make_loader(tmp_path)
        ctx = make_ctx(
            0,
            TaskStage.FAILED,
            error_action=TaskAction.PREPROCESS,
            error_reason="exception::ValueError",
            retry_count=2,
        )
        contexts = {0: ctx}
        loader._prepare_failed_retries(contexts, record_each_stage=True)
        assert contexts[0].retry_count == 3

    def test_record_each_stage_false_resets_to_initial(self, tmp_path):
        loader = self._make_loader(tmp_path)
        ctx = make_ctx(
            0,
            TaskStage.FAILED,
            error_action=TaskAction.POSTPROCESS,
            error_reason="exception::Error",
            iteration=0,
        )
        contexts = {0: ctx}
        loader._prepare_failed_retries(contexts, record_each_stage=False)

        updated = contexts[0]
        assert updated.stage == TaskStage.INITIAL
        assert updated.iteration == 0
        assert updated.preprocess_result is None
        assert updated.infer_result is None

    def test_record_each_stage_false_preserves_iteration(self, tmp_path):
        """With iteration > 0, should resume from last iteration start."""
        loader = self._make_loader(tmp_path)
        ctx = make_ctx(
            0,
            TaskStage.FAILED,
            iteration=2,
            error_action=TaskAction.INFER,
            error_reason="exception::Error",
        )
        contexts = {0: ctx}
        loader._prepare_failed_retries(contexts, record_each_stage=False)

        updated = contexts[0]
        assert updated.stage == TaskStage.INITIAL
        assert updated.iteration == 2

    def test_clears_correct_fields_for_infer_error(self, tmp_path):
        loader = self._make_loader(tmp_path)
        ctx = TaskContext(
            sample_id=0,
            raw_sample={"q": "test"},
            stage=TaskStage.FAILED,
            preprocess_result="pre",
            infer_result="inf",
            postprocess_result="post",
            error_action=TaskAction.INFER,
            error_reason="exception::Error",
        )
        contexts = {0: ctx}
        loader._prepare_failed_retries(contexts, record_each_stage=True)

        updated = contexts[0]
        assert updated.preprocess_result == "pre"  # preserved
        assert updated.infer_result is None  # cleared
        assert updated.postprocess_result is None  # cleared

    def test_retry_count_incremented_at_boundary(self, tmp_path):
        """_prepare_failed_retries always increments retry_count regardless of limit.

        The max_retries enforcement is the runner's responsibility. This test
        verifies that _prepare_failed_retries correctly increments retry_count
        to N+1, which is what the runner compares against max_retries.
        For max_retries=1: retry_count=1 is allowed (1 > 1 is False),
        retry_count=2 would be blocked (2 > 1 is True).
        """
        loader = self._make_loader(tmp_path)
        # Simulate a sample that has already used its one allowed retry
        ctx = make_ctx(
            0,
            TaskStage.FAILED,
            error_action=TaskAction.INFER,
            error_reason="exception::TimeoutError",
            retry_count=1,  # already retried once (max_retries=1 boundary)
        )
        contexts = {0: ctx}
        loader._prepare_failed_retries(contexts, record_each_stage=True)

        updated = contexts[0]
        # retry_count must be incremented to 2 so the runner can detect 2 > max_retries
        assert updated.retry_count == 2, (
            "retry_count must be incremented to 2 so runner can enforce max_retries=1 "
            "(runner blocks when retry_count > max_retries)"
        )
        assert updated.stage == TaskStage.PREPROCESSED

    def test_stage_meta_filter_failure_is_non_fatal(self, tmp_path):
        """
        If stage_meta filtering raises, the warning is logged and retry still
        proceeds.
        """

        loader = self._make_loader(tmp_path)
        ctx = make_ctx(
            0,
            TaskStage.FAILED,
            error_action=TaskAction.INFER,
            error_reason="exception::TimeoutError",
        )
        # Inject a bad stage_meta value that will cause TaskStage() to raise
        ctx = ctx.__class__(
            **{
                **{f: getattr(ctx, f) for f in ctx.__dataclass_fields__},
                "stage_meta": {"not_a_valid_stage": [{}]},
            }
        )
        contexts = {0: ctx}
        # Should not raise even when stage_meta filtering fails
        loader._prepare_failed_retries(contexts, record_each_stage=True)
        updated = contexts[0]
        # Retry still proceeds despite meta filter failure
        assert updated.stage != TaskStage.FAILED
        assert updated.retry_count == 1
