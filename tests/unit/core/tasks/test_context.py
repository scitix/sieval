"""
Tests for sieval.core.tasks.context — state machine, serialization, snapshot.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import pytest

from sieval.core.tasks.consts import TaskAction, TaskStage
from sieval.core.tasks.context import TaskStageMeta, TaskStageOutput


class TestTaskContextTransitions:
    """State machine transition tests."""

    def test_state_machine_happy_path(self, base_context):
        assert base_context.stage == TaskStage.INITIAL
        assert base_context.iteration == 0
        assert base_context.next_action() == TaskAction.PREPROCESS

        ctx = base_context.to_preprocessed("preprocessed_data")
        assert ctx.stage == TaskStage.PREPROCESSED
        assert ctx.preprocess_result == "preprocessed_data"
        assert ctx.next_action() == TaskAction.INFER

        ctx = ctx.to_inferred("inferred_data")
        assert ctx.stage == TaskStage.INFERRED
        assert ctx.infer_result == "inferred_data"
        assert ctx.next_action() == TaskAction.POSTPROCESS

        ctx = ctx.to_postprocessed("post_data")
        assert ctx.stage == TaskStage.POSTPROCESSED
        assert ctx.postprocess_result == "post_data"
        assert ctx.next_action() == TaskAction.FEEDBACK

        ctx = ctx.to_feedback("feedback_data")
        assert ctx.stage == TaskStage.FEEDBACK
        assert ctx.feedback_result == "feedback_data"
        assert ctx.next_action() is None

        ctx = ctx.to_final()
        assert ctx.stage == TaskStage.FINAL
        assert ctx.is_terminal()
        assert ctx.next_action() is None

    def test_failed_transition(self, base_context):
        ctx = base_context.to_failed(
            TaskAction.INFER, "api_error", "Connection refused"
        )
        assert ctx.stage == TaskStage.FAILED
        assert ctx.is_terminal()
        assert ctx.error_action == TaskAction.INFER
        assert ctx.error_reason == "api_error"
        assert ctx.error_msg == "Connection refused"

    def test_iterate(self, base_context):
        ctx = (
            base_context.to_preprocessed("pre")
            .to_inferred("inf")
            .to_postprocessed("post")
            .to_feedback("fb")
            .iterate()
        )
        assert ctx.iteration == 1
        assert ctx.stage == TaskStage.INITIAL
        # Previous results are preserved across iterations
        assert ctx.feedback_result == "fb"

    def test_frozen_dataclass(self, base_context):
        with pytest.raises(AttributeError):
            base_context.stage = TaskStage.FINAL

    def test_transition_preserves_immutability_of_original(self, base_context):
        """Transitioning creates a new context; original must remain unchanged."""
        ctx_pre = base_context.to_preprocessed("pre")
        assert base_context.stage == TaskStage.INITIAL
        assert base_context.preprocess_result is None
        assert ctx_pre.stage == TaskStage.PREPROCESSED
        assert ctx_pre.preprocess_result == "pre"

    def test_iterate_resets_stage_but_preserves_results(self, base_context):
        """iterate() resets to INITIAL with incremented iteration, keeping results."""
        ctx = (
            base_context.to_preprocessed("pre")
            .to_inferred("inf")
            .to_postprocessed("post")
            .to_feedback("fb")
            .iterate()
        )
        assert ctx.stage == TaskStage.INITIAL
        assert ctx.iteration == 1
        assert ctx.preprocess_result == "pre"
        assert ctx.infer_result == "inf"
        assert ctx.postprocess_result == "post"
        assert ctx.feedback_result == "fb"
        # Can transition again from INITIAL
        ctx2 = ctx.to_preprocessed("pre2")
        assert ctx2.stage == TaskStage.PREPROCESSED
        assert ctx2.preprocess_result == "pre2"
        assert ctx2.iteration == 1

    def test_to_failed_from_any_stage(self, base_context):
        """to_failed() should work from any non-terminal stage."""
        for transition_fn in [
            lambda c: c,
            lambda c: c.to_preprocessed("p"),
            lambda c: c.to_preprocessed("p").to_inferred("i"),
            lambda c: c.to_preprocessed("p").to_inferred("i").to_postprocessed("o"),
        ]:
            ctx = transition_fn(base_context)
            failed = ctx.to_failed(TaskAction.INFER, "error", "msg")
            assert failed.stage == TaskStage.FAILED
            assert failed.is_terminal()

    def test_next_action_terminal_stages(self, base_context):
        """FINAL and FAILED should return None for next_action."""
        final = base_context.to_final()
        assert final.next_action() is None
        failed = base_context.to_failed(TaskAction.PREPROCESS, "err", "msg")
        assert failed.next_action() is None


class TestTaskContextMeta:
    """Metadata accumulation during transitions."""

    def test_meta_attached_and_accumulates(self, base_context, sample_stage_meta):
        meta2: TaskStageMeta = {"timestamp": 2000.0, "timing_s": 2.0}
        ctx = base_context.to_preprocessed("pre", meta=sample_stage_meta).to_inferred(
            "inf", meta=meta2
        )
        assert "preprocessed" in ctx.stage_meta
        assert len(ctx.stage_meta["preprocessed"]) == 1
        assert ctx.stage_meta["preprocessed"][0]["timing_s"] == 1.5
        assert "inferred" in ctx.stage_meta
        assert ctx.stage_meta["inferred"][0]["timing_s"] == 2.0

    def test_no_meta_when_none(self, base_context):
        ctx = base_context.to_preprocessed("pre", meta=None)
        assert "preprocessed" not in ctx.stage_meta

    def test_record_stage_meta_and_append_history(
        self, base_context, sample_stage_meta
    ):
        ctx = base_context.record_stage_meta(TaskStage.INFERRED, sample_stage_meta)
        assert "inferred" in ctx.stage_meta
        assert len(ctx.stage_meta["inferred"]) == 1

        meta1: TaskStageMeta = {"timestamp": 1.0, "timing_s": 1.0}
        meta2: TaskStageMeta = {"timestamp": 2.0, "timing_s": 2.0}
        ctx = base_context.record_stage_meta(TaskStage.INFERRED, meta1)
        ctx = ctx.record_stage_meta(TaskStage.INFERRED, meta2)
        assert len(ctx.stage_meta["inferred"]) == 2


class TestTaskContextSerialize:
    """Serialization (non-snapshot mode)."""

    def test_minimal_and_with_results(self, base_context):
        d = base_context.serialize(store_type_metadata=False)
        assert d["sample_id"] == 0
        assert d["iteration"] == 0
        assert d["stage"] == "initial"

        ctx = base_context.to_preprocessed("pre").to_inferred("inf")
        d = ctx.serialize(store_type_metadata=False)
        assert d["preprocess_result"] == "pre"
        assert d["infer_result"] == "inf"
        assert "postprocess_result" not in d

    def test_meta_included_and_excluded(self, base_context, sample_stage_meta):
        ctx = base_context.to_preprocessed("pre", meta=sample_stage_meta)
        d = ctx.serialize(store_type_metadata=False, include_meta=True)
        assert "stage_meta" in d
        assert "preprocessed" in d["stage_meta"]

        d = ctx.serialize(store_type_metadata=False, include_meta=False)
        assert "stage_meta" not in d

    def test_error_fields(self, base_context):
        ctx = base_context.to_failed(TaskAction.INFER, "timeout", "Request timed out")
        d = ctx.serialize(store_type_metadata=False)
        assert d["error_action"] == "infer"
        assert d["error_reason"] == "timeout"
        assert d["error_msg"] == "Request timed out"

    def test_with_type_metadata(self, base_context, sample_model_output):
        ctx = base_context.to_preprocessed("pre").to_inferred(sample_model_output)
        d = ctx.serialize(store_type_metadata=True)
        assert d["infer_result"]["__sieval_cls__"] == "ModelOutput"


class TestTaskContextSnapshot:
    """Snapshot mode serialization."""

    def test_snapshot_only_keeps_current_stage(self, base_context):
        ctx = base_context.to_preprocessed("pre").to_inferred("inf").make_snapshot()
        d = ctx.serialize(store_type_metadata=False)
        # Snapshot should only include current stage's result
        assert "infer_result" in d
        assert "preprocess_result" not in d

    def test_snapshot_meta_behaviors(self, base_context, sample_stage_meta):
        ctx = (
            base_context.to_preprocessed("pre", meta=sample_stage_meta)
            .to_inferred("inf", meta={"timestamp": 2.0, "timing_s": 2.0})
            .make_snapshot()
        )
        d = ctx.serialize(store_type_metadata=False, include_meta=True)
        # Should have meta_last (last meta for current stage), not full stage_meta
        assert "meta_last" in d
        assert "stage_meta" not in d
        assert d["meta_last"]["timing_s"] == 2.0

        ctx = base_context.to_preprocessed("pre").make_snapshot()
        d = ctx.serialize(store_type_metadata=False, include_meta=False)
        assert "meta_last" not in d
        assert "stage_meta" not in d

    def test_snapshot_flag(self, base_context):
        ctx = base_context.make_snapshot()
        assert ctx.is_snapshot is True
        assert base_context.is_snapshot is False


class TestTaskRunMeta:
    def test_required_deterministic_field(self):
        from sieval.core.tasks.context import TaskRunMeta

        meta: TaskRunMeta = {"version": "0.1.0", "deterministic": False}
        assert meta["deterministic"] is False

        meta_true: TaskRunMeta = {"version": "0.1.0", "deterministic": True}
        assert meta_true["deterministic"] is True


class TestTaskStageOutputUnwrap:
    """TaskStageOutput wrapping behavior."""

    def test_wraps_value_and_meta(self, sample_stage_meta):
        tso = TaskStageOutput(value="answer")
        assert tso.value == "answer"
        assert tso.meta is None

        tso = TaskStageOutput(value=42, meta=sample_stage_meta)
        assert tso.value == 42
        assert tso.meta is not None
        assert tso.meta["timing_s"] == 1.5

    def test_frozen(self):
        tso = TaskStageOutput(value="x")
        with pytest.raises(AttributeError):
            tso.value = "y"  # type: ignore[invalid-assignment]
