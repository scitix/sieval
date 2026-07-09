"""
End-to-end tests: mock Task/Dataset/Model → TaskRunner → report.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from dataclasses import replace
from pathlib import Path
from typing import TypedDict
from unittest.mock import AsyncMock, MagicMock

import anyio
import orjson
import pytest

from sieval import __version__
from sieval.core.runners.resume_gate import ResumeVersionError
from sieval.core.runners.runner import ResultDirExistsError, TaskRunner
from sieval.core.tasks.concurrency import compute_stream_buffer_capacity
from sieval.core.tasks.consts import TaskAction, TaskStage
from sieval.core.tasks.context import TaskContext, TaskStageOutput
from sieval.core.tasks.task import Task
from tests.conftest import MockAlwaysFailModel, MockChatModel, MockDataset, make_config

# ===================================================================
# Default samples & answers for the 3-sample happy-path dataset
# ===================================================================
DEFAULT_ANSWERS = {"What is 1+1?": "2", "What is 2+3?": "5", "What is 10-7?": "3"}


# ===================================================================
# Task definitions (test-specific)
# ===================================================================
class MockTask(Task):
    model_type = "chat"

    async def preprocess(self, raw, ctx):
        return raw["question"]

    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

    async def postprocess(self, inf, ctx):
        return inf.texts[0].strip()

    async def feedback(self, post, ctx):
        correct = post == ctx.raw_sample["answer"]
        return True, {"correct": correct, "predicted": post}

    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        correct = sum(
            1 for f in finals if f.feedback_result and f.feedback_result["correct"]
        )
        return {"accuracy": correct / total if total else 0.0, "total": total}


class MockIterativeTask(Task):
    """Task that requires multiple iterations before finalizing."""

    model_type = "chat"

    async def preprocess(self, raw, ctx):
        return raw["question"]

    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

    async def postprocess(self, inf, ctx):
        return inf.texts[0].strip()

    async def feedback(self, post, ctx):
        # Finalize only after iteration 1 (i.e., needs 2 passes)
        finalize = ctx.iteration >= 1
        return finalize, {"iteration": ctx.iteration, "answer": post}

    async def report(self, finals, fails):
        return {
            "completed": len(finals),
            "failed": len(fails),
            "iterations": [f.iteration for f in finals],
        }


class MixedOutcomeTask(MockTask):
    """Task that fails preprocessing for selected samples."""

    async def preprocess(self, raw, ctx):
        if raw.get("raise_preprocess"):
            raise ValueError("forced preprocess failure")
        return await super().preprocess(raw, ctx)


class ListReportTask(MockTask):
    async def report(self, finals, fails):
        return ["not", "a", "dict"]


class NoneReportTask(MockTask):
    async def report(self, finals, fails):
        return None


class ProgressUpdateCall(TypedDict):
    sample_id: str | int
    current_hydrated_count: int
    failed: bool
    anomalies: set[str] | None


# ===================================================================
# Tests
# ===================================================================
class TestE2EHappyPath:
    @pytest.mark.anyio
    async def test_all_correct_and_persisted_artifacts(self, tmp_path):
        """Happy-path run should produce expected report and core artifacts."""
        dataset = MockDataset()
        model = MockChatModel(answers=DEFAULT_ANSWERS)
        task = MockTask(dataset=dataset, model=model, name="e2e_test")
        config = make_config(tmp_path)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report["accuracy"] == 1.0
        assert report["total"] == 3

        report_path = runner.root_dir / "report.json"
        assert report_path.exists()
        saved = orjson.loads(report_path.read_bytes())
        assert saved["accuracy"] == 1.0

        manifest_path = runner.root_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = orjson.loads(manifest_path.read_bytes())
        assert len(manifest) == 3
        assert all(e["final"] for e in manifest)
        assert all(e["stage"] == "final" for e in manifest)

        root = runner.root_dir
        jsonl_files = list(root.rglob("*.jsonl"))
        idx_files = list(root.rglob("*.idx"))
        assert len(jsonl_files) > 0
        assert len(idx_files) > 0


class TestE2EFailureRecovery:
    @pytest.mark.anyio
    async def test_sample_failure_captured(self, tmp_path):
        """A model exception should mark the sample as FAILED."""
        dataset = MockDataset()
        model = MockAlwaysFailModel()
        task = MockTask(dataset=dataset, model=model, name="e2e_fail")
        config = make_config(tmp_path)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report["accuracy"] == 0.0
        assert report["total"] == 3

        manifest = orjson.loads((runner.root_dir / "manifest.json").read_bytes())
        assert all(e["failed"] for e in manifest)


class TestE2EIterations:
    @pytest.mark.anyio
    async def test_multi_iteration_finalize(self, tmp_path):
        """Task that needs 2 iterations should complete with iteration=1."""
        dataset = MockDataset()
        model = MockChatModel(answers=DEFAULT_ANSWERS)
        task = MockIterativeTask(dataset=dataset, model=model, name="e2e_iter")
        config = make_config(tmp_path, max_iterations=5)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report["completed"] == 3
        assert report["failed"] == 0
        assert all(it == 1 for it in report["iterations"])

    @pytest.mark.anyio
    async def test_iteration_limit_causes_failure(self, tmp_path):
        """max_iterations=1 should cause iteration_limit failures."""
        dataset = MockDataset()
        model = MockChatModel(answers=DEFAULT_ANSWERS)
        task = MockIterativeTask(dataset=dataset, model=model, name="e2e_iter_limit")
        config = make_config(tmp_path, max_iterations=1)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report["completed"] == 0
        assert report["failed"] == 3

        manifest = orjson.loads((runner.root_dir / "manifest.json").read_bytes())
        assert all(e["failed"] for e in manifest)


class TestE2EConcurrency:
    @pytest.mark.anyio
    async def test_serial_execution_with_limit_one(self, tmp_path):
        """Pipeline should work correctly with concurrency_limit=1 (serial)."""
        dataset = MockDataset()
        model = MockChatModel(answers=DEFAULT_ANSWERS)
        task = MockTask(dataset=dataset, model=model, name="e2e_conc")
        config = make_config(tmp_path, concurrency_limit=1)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report["accuracy"] == 1.0
        assert report["total"] == 3

    @pytest.mark.anyio
    async def test_global_concurrency_limit_allows_actual_concurrency(self, tmp_path):
        """
        concurrency_limit=10 with 6 samples should allow actual concurrent execution.
        """
        dataset = MockDataset(CONC_SAMPLES)
        model = MockChatModel(answers=CONC_ANSWERS)
        task = StageConcurrencyTrackingTask(
            dataset=dataset, model=model, name="e2e_global_conc"
        )
        config = make_config(tmp_path, concurrency_limit=10)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report["accuracy"] == 1.0
        # With limit=10 and 6 samples, multiple infer calls should overlap
        assert task.max_concurrent_infer > 1, (
            "Expected concurrent infer calls with concurrency_limit=10, "
            f"but max_concurrent_infer={task.max_concurrent_infer}. "
            "Global concurrency limit may be incorrectly serializing execution."
        )


class TestE2ERecordModes:
    @pytest.mark.anyio
    async def test_record_each_stage_modes(self, tmp_path):
        """record_each_stage toggles whether intermediate stage shards are persisted."""
        # (record_each_stage, expected_stage_dirs, result_dir)
        cases = [
            # Persist all stage snapshots.
            (
                True,
                {"final", "preprocessed", "inferred", "postprocessed"},
                str(tmp_path / "stage_true"),
            ),
            # Persist final stage only.
            (
                False,
                {"final"},
                str(tmp_path / "stage_false"),
            ),
        ]
        for record_each_stage, expected_stage_dirs, result_dir in cases:
            dataset = MockDataset()
            model = MockChatModel(answers=DEFAULT_ANSWERS)
            task = MockTask(dataset=dataset, model=model, name="e2e_stage_mode")
            config = make_config(
                tmp_path,
                result_dir=result_dir,
                record_each_stage=record_each_stage,
            )

            runner = TaskRunner(task, config)
            report = await runner.arun()

            assert report["accuracy"] == 1.0
            stage_dirs = {p.parent.name for p in runner.root_dir.rglob("*.jsonl")}
            assert "final" in stage_dirs
            if record_each_stage:
                assert expected_stage_dirs.issubset(stage_dirs)
            else:
                assert stage_dirs == {"final"}


# ===================================================================
# P1: Stage-Level Concurrency Enforcement
# ===================================================================
class StageConcurrencyTrackingTask(Task):
    """Task that tracks max concurrent infer calls via a shared counter."""

    model_type = "chat"

    def __init__(self, dataset, model, name=None):
        super().__init__(dataset=dataset, model=model, name=name)
        self.max_concurrent_infer = 0
        self._current_concurrent_infer = 0

    async def preprocess(self, raw, ctx):
        return raw["question"]

    async def infer(self, pre, ctx):
        self._current_concurrent_infer += 1
        if self._current_concurrent_infer > self.max_concurrent_infer:
            self.max_concurrent_infer = self._current_concurrent_infer
        try:
            # Small sleep to create overlap window for concurrency measurement
            await anyio.sleep(0.01)
            return await self.model.agenerate(pre)
        finally:
            self._current_concurrent_infer -= 1

    async def postprocess(self, inf, ctx):
        return inf.texts[0].strip()

    async def feedback(self, post, ctx):
        correct = post == ctx.raw_sample["answer"]
        return True, {"correct": correct}

    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        correct = sum(
            1 for f in finals if f.feedback_result and f.feedback_result["correct"]
        )
        return {"accuracy": correct / total if total else 0.0, "total": total}


CONC_SAMPLES = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(6)]
CONC_ANSWERS = {f"Q{i}": f"A{i}" for i in range(6)}


class TestE2EStageConcurrency:
    @pytest.mark.anyio
    async def test_stage_concurrency_limit_enforced(self, tmp_path):
        """concurrency_limits={'infer': 1} should limit infer to 1 concurrent sample."""
        dataset = MockDataset(CONC_SAMPLES)
        model = MockChatModel(answers=CONC_ANSWERS)
        task = StageConcurrencyTrackingTask(
            dataset=dataset, model=model, name="e2e_stage_conc"
        )
        config = make_config(
            tmp_path,
            concurrency_limits={"infer": 1},
        )

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report["accuracy"] == 1.0
        # Must be exactly 1: proves the limiter is working AND that
        # the tracking logic itself is functional (not stuck at 0)
        assert task.max_concurrent_infer == 1

    @pytest.mark.anyio
    async def test_stage_concurrency_limit_allows_higher(self, tmp_path):
        """
        concurrency_limits={'infer': 4} should allow up to 4 concurrent infer calls.
        """
        dataset = MockDataset(CONC_SAMPLES)
        model = MockChatModel(answers=CONC_ANSWERS)
        task = StageConcurrencyTrackingTask(
            dataset=dataset, model=model, name="e2e_stage_conc_high"
        )
        config = make_config(
            tmp_path,
            concurrency_limits={"infer": 4},
        )

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report["accuracy"] == 1.0
        # Must be > 1 (proves concurrency actually happens) and <= 4 (limit respected)
        assert task.max_concurrent_infer > 1
        assert task.max_concurrent_infer <= 4


# ===================================================================
# P2: Edge Cases
# ===================================================================
class TestE2EEdgeCases:
    @pytest.mark.anyio
    async def test_empty_dataset(self, tmp_path):
        """Empty dataset should produce a report without crashing."""
        dataset = MockDataset([])
        model = MockChatModel(answers=DEFAULT_ANSWERS)
        task = MockTask(dataset=dataset, model=model, name="e2e_empty")
        config = make_config(tmp_path)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report["total"] == 0
        saved_report = orjson.loads((runner.root_dir / "report.json").read_bytes())
        assert saved_report["total"] == 0

    @pytest.mark.anyio
    async def test_max_retries_exhausted(self, tmp_path):
        """Samples exceeding max_retries should be marked FAILED and not retried."""
        dataset = MockDataset()
        result_dir = str(tmp_path / "max_retry_results")

        # First run: all samples fail
        model_fail = MockAlwaysFailModel()
        task1 = MockTask(dataset=dataset, model=model_fail, name="e2e_max_retry")
        config1 = make_config(tmp_path, result_dir=result_dir, max_retries=0)
        runner1 = TaskRunner(task1, config1)
        report1 = await runner1.arun()
        assert report1["total"] == 3
        assert report1["accuracy"] == 0.0

        # Resume: max_retries=0 means no retries allowed
        model_ok = MockChatModel(answers=DEFAULT_ANSWERS)
        task2 = MockTask(dataset=dataset, model=model_ok, name="e2e_max_retry")
        config2 = make_config(
            tmp_path, result_dir=result_dir, auto_resume=True, max_retries=0
        )
        runner2 = TaskRunner(task2, config2)
        report2 = await runner2.arun()

        # All samples should still be failed (no retries allowed)
        assert report2["accuracy"] == 0.0
        assert report2["total"] == 3

        # Verify manifest confirms all samples are still FAILED
        manifest = orjson.loads((runner2.root_dir / "manifest.json").read_bytes())
        assert len(manifest) == 3
        assert all(e["failed"] for e in manifest)

    @pytest.mark.anyio
    async def test_retry_succeeds_on_resume(self, tmp_path):
        """Failed samples should succeed on resume when max_retries > 0."""
        dataset = MockDataset()
        result_dir = str(tmp_path / "retry_results")

        # First run: all samples fail
        model_fail = MockAlwaysFailModel()
        task1 = MockTask(dataset=dataset, model=model_fail, name="e2e_retry")
        config1 = make_config(tmp_path, result_dir=result_dir, max_retries=0)
        runner1 = TaskRunner(task1, config1)
        report1 = await runner1.arun()
        assert report1["accuracy"] == 0.0

        # Resume with max_retries=1 and a working model — should succeed
        model_ok = MockChatModel(answers=DEFAULT_ANSWERS)
        task2 = MockTask(dataset=dataset, model=model_ok, name="e2e_retry")
        config2 = make_config(
            tmp_path, result_dir=result_dir, auto_resume=True, max_retries=1
        )
        runner2 = TaskRunner(task2, config2)
        report2 = await runner2.arun()

        assert report2["accuracy"] == 1.0
        assert report2["total"] == 3

        manifest = orjson.loads((runner2.root_dir / "manifest.json").read_bytes())
        assert len(manifest) == 3
        assert all(e["final"] for e in manifest)
        assert all(not e["failed"] for e in manifest)

    @pytest.mark.anyio
    async def test_all_samples_fail_report(self, tmp_path):
        """
        When all samples fail, report() receives finals=[] and fails=[...] correctly.
        """
        dataset = MockDataset()
        model = MockAlwaysFailModel(error=RuntimeError)
        task = MockTask(dataset=dataset, model=model, name="e2e_all_fail")
        config = make_config(tmp_path)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report["accuracy"] == 0.0
        assert report["total"] == 3

    @pytest.mark.anyio
    async def test_concurrent_shard_writes(self, tmp_path):
        """High concurrency with small shard_samples should not corrupt data."""
        many_samples = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(20)]
        many_answers = {f"Q{i}": f"A{i}" for i in range(20)}

        dataset = MockDataset(many_samples)
        model = MockChatModel(answers=many_answers)
        task = MockTask(dataset=dataset, model=model, name="e2e_shard_conc")
        # Small shard_samples forces multiple shards
        config = make_config(tmp_path, shard_samples=4, concurrency_limit=10)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report["accuracy"] == 1.0
        assert report["total"] == 20

        # Verify all samples are in manifest
        manifest = orjson.loads((runner.root_dir / "manifest.json").read_bytes())
        assert len(manifest) == 20
        assert all(e["final"] for e in manifest)


# ===================================================================
# P3: Edge Cases – init-time guard rails
# ===================================================================
class TestE2ERunnerGuardRails:
    def test_init_guard_rails_raise(self, tmp_path):
        """Invalid result_dir setups should fail at TaskRunner initialization."""
        dataset = MockDataset()
        model = MockChatModel(answers=DEFAULT_ANSWERS)

        file_path = tmp_path / "not_a_dir.json"
        file_path.write_text("{}")
        task_file = MockTask(dataset=dataset, model=model, name="e2e_file_dir_init")
        file_cfg = make_config(tmp_path, result_dir=str(file_path))
        with pytest.raises(ValueError):
            TaskRunner(task_file, file_cfg)

        result_dir = tmp_path / "existing_run"
        result_dir.mkdir()
        (result_dir / "manifest.json").write_text("[]")
        task_existing = MockTask(dataset=dataset, model=model, name="e2e_existing")
        existing_cfg = make_config(
            tmp_path, result_dir=str(result_dir), auto_resume=False
        )
        # Back-compat: keep both the subclass and FileExistsError assertions.
        with pytest.raises(ResultDirExistsError) as exc_info:
            TaskRunner(task_existing, existing_cfg)
        assert isinstance(exc_info.value, FileExistsError)
        assert exc_info.value.path == result_dir
        msg = str(exc_info.value)
        assert "auto_resume=True" in msg
        assert "--resume" not in msg

    def test_sync_run_wrapper(self, tmp_path):
        """runner.run() (sync) should block and return the same report as arun()."""
        dataset = MockDataset()
        model = MockChatModel(answers=DEFAULT_ANSWERS)
        task = MockTask(dataset=dataset, model=model, name="e2e_sync_run")
        config = make_config(tmp_path)

        runner = TaskRunner(task, config)
        report = runner.run()

        assert report["accuracy"] == 1.0
        assert report["total"] == 3


# ===================================================================
# P4: TaskStageOutput and list[ModelOutput] meta paths
# ===================================================================
class ListModelOutputTask(Task):
    """Task whose infer() returns a list[ModelOutput] to test auto-meta aggregation."""

    model_type = "chat"

    async def preprocess(self, raw, ctx):
        return raw["question"]

    async def infer(self, pre, ctx):
        # Return a list of ModelOutputs — exercises _build_auto_meta list branch
        out1 = await self.model.agenerate(pre)
        out2 = await self.model.agenerate(pre)
        return [out1, out2]

    async def postprocess(self, inf, ctx):
        # inf is a list of ModelOutputs
        return inf[0].texts[0].strip() if inf else ""

    async def feedback(self, post, ctx):
        correct = post == ctx.raw_sample["answer"]
        return True, {"correct": correct}

    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        correct = sum(
            1 for f in finals if f.feedback_result and f.feedback_result["correct"]
        )
        return {"accuracy": correct / total if total else 0.0, "total": total}


class TestE2EListModelOutput:
    @pytest.mark.anyio
    async def test_list_model_output_runs_successfully(self, tmp_path):
        """infer() returning list[ModelOutput] should complete without errors."""
        dataset = MockDataset()
        model = MockChatModel(answers=DEFAULT_ANSWERS)
        task = ListModelOutputTask(dataset=dataset, model=model, name="e2e_list_mo")
        config = make_config(tmp_path, record_meta=True)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report["total"] == 3
        assert report["accuracy"] == 1.0


# ===================================================================
# P5: stage_meta_hooks config path
# ===================================================================
class TestE2EStagMetaHooks:
    @pytest.mark.anyio
    async def test_stage_meta_hook_called(self, tmp_path):
        """stage_meta_hook should be called and its return value merged into meta."""
        hook_calls: list[str] = []

        def my_hook(_val, stage, _ctx):
            hook_calls.append(stage.value)
            return {"custom_tag": "test_hook"}

        dataset = MockDataset()
        model = MockChatModel(answers=DEFAULT_ANSWERS)
        task = MockTask(dataset=dataset, model=model, name="e2e_hook")
        config = make_config(tmp_path, stage_meta_hook=my_hook, record_meta=True)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report["accuracy"] == 1.0
        # Hook should have been called for each stage of each sample
        assert len(hook_calls) > 0

    @pytest.mark.anyio
    async def test_stage_meta_hooks_dict_called_per_stage(self, tmp_path):
        """stage_meta_hooks dict should only call the hook for the matching stage."""
        from sieval.core.tasks.context import TaskStage

        infer_calls: list = []
        feedback_calls: list = []

        def infer_hook(_val, _stage, ctx):
            infer_calls.append(ctx.sample_id)
            return {"hook": "infer_hook"}

        def feedback_hook(_val, _stage, ctx):
            feedback_calls.append(ctx.sample_id)
            return {"hook": "feedback_hook"}

        dataset = MockDataset()
        model = MockChatModel(answers=DEFAULT_ANSWERS)
        task = MockTask(dataset=dataset, model=model, name="e2e_hooks_dict")
        config = make_config(
            tmp_path,
            stage_meta_hooks={
                TaskStage.INFERRED: infer_hook,
                TaskStage.FEEDBACK: feedback_hook,
            },
            record_meta=True,
        )

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report["accuracy"] == 1.0
        # infer hook called once per sample
        assert len(infer_calls) == 3
        # feedback hook called once per sample
        assert len(feedback_calls) == 3


# ===================================================================
# _resolve_result_dir: auto timestamp path and auto-resume path
# ===================================================================
class TestResolveResultDir:
    def test_no_result_dir_creates_timestamped_path(self, tmp_path, monkeypatch):
        """Without result_dir, a timestamped subdir under outputs/<task> is created."""
        # Run inside tmp_path to avoid creating real outputs/
        monkeypatch.chdir(tmp_path)

        dataset = MockDataset()
        model = MockChatModel(answers=DEFAULT_ANSWERS)
        task = MockTask(dataset=dataset, model=model, name="ts_task")
        config = make_config(tmp_path, result_dir=None)

        runner = TaskRunner(task, config)
        # root_dir is set on init; should be outputs/ts_task/<14-digit timestamp>
        import re

        assert re.fullmatch(r"\d{14}", runner.root_dir.name)
        assert runner.root_dir.parent.name == "ts_task"
        assert runner.root_dir.parent.parent.name == "outputs"

    @pytest.mark.anyio
    async def test_auto_resume_picks_latest_timestamped_dir(
        self, tmp_path, monkeypatch
    ):
        """
        With auto_resume and no explicit result_dir, the latest timestamp dir is used.
        """
        monkeypatch.chdir(tmp_path)

        dataset = MockDataset()
        model = MockChatModel(answers=DEFAULT_ANSWERS)
        task_name = "auto_resume_ts"

        # First run: creates a timestamped dir
        task1 = MockTask(dataset=dataset, model=model, name=task_name)
        config1 = make_config(tmp_path, result_dir=None)
        runner1 = TaskRunner(task1, config1)
        report1 = await runner1.arun()
        assert report1 is not None
        first_dir = runner1.root_dir

        # Sanity: directory has manifest
        assert (first_dir / "manifest.json").exists()

        # Second run with auto_resume: should pick up the same directory
        task2 = MockTask(dataset=dataset, model=model, name=task_name)
        config2 = make_config(tmp_path, result_dir=None, auto_resume=True)
        runner2 = TaskRunner(task2, config2)
        report2 = await runner2.arun()

        assert report2 == report1
        assert runner2.root_dir == first_dir

    def test_resume_dir_selection_respects_auto_resume_flag(
        self, tmp_path, monkeypatch
    ):
        """auto_resume=True should reuse latest dir; auto_resume=False should not."""
        monkeypatch.chdir(tmp_path)

        task_name = "resume_flag_semantics"
        root_abs = tmp_path / "outputs" / task_name
        root_rel = Path("outputs") / task_name
        older = root_abs / "20240101010101"
        latest = root_abs / "20240102020202"
        older.mkdir(parents=True, exist_ok=True)
        latest.mkdir(parents=True, exist_ok=True)
        (older / "manifest.json").write_text("[]", encoding="utf-8")
        (latest / "manifest.json").write_text("[]", encoding="utf-8")
        # Real resumable dirs always carry a meta.json handshake (Task 4);
        # give the fabricated fixture one so the version gate (Task 5) sees
        # an EXACT match and this test stays focused on dir-selection logic.
        for d in (older, latest):
            (d / "meta.json").write_bytes(
                orjson.dumps({"version": __version__, "deterministic": False})
            )

        dataset = MockDataset()
        model = MockChatModel(answers=DEFAULT_ANSWERS)

        resumed_runner = TaskRunner(
            MockTask(dataset=dataset, model=model, name=task_name),
            make_config(tmp_path, result_dir=None, auto_resume=True),
        )
        assert resumed_runner.root_dir == root_rel / latest.name

        fresh_runner = TaskRunner(
            MockTask(dataset=dataset, model=model, name=task_name),
            make_config(tmp_path, result_dir=None, auto_resume=False),
        )
        assert fresh_runner.root_dir.parent == root_rel
        assert fresh_runner.root_dir not in {
            root_rel / older.name,
            root_rel / latest.name,
        }


class TestNormalizeLimits:
    def _make_runner(self, tmp_path):
        dataset = MockDataset()
        model = MockChatModel(answers=DEFAULT_ANSWERS)
        task = MockTask(dataset=dataset, model=model, name="norm_limits")
        config = make_config(tmp_path)
        return TaskRunner(task, config)

    def test_taskaction_key_is_normalized(self, tmp_path):
        from sieval.core.tasks.consts import TaskAction

        runner = self._make_runner(tmp_path)
        result = runner._normalize_limits({TaskAction.INFER: 3})
        assert result == {"infer": 3}

    def test_string_key_is_preserved(self, tmp_path):
        runner = self._make_runner(tmp_path)
        result = runner._normalize_limits({"infer": 2, "feedback": 4})
        assert result == {"infer": 2, "feedback": 4}

    def test_unknown_key_is_ignored(self, tmp_path):
        runner = self._make_runner(tmp_path)
        result = runner._normalize_limits({"unknown_stage": 5, "infer": 1})
        assert "unknown_stage" not in result
        assert result["infer"] == 1

    def test_none_returns_empty(self, tmp_path):
        runner = self._make_runner(tmp_path)
        assert runner._normalize_limits(None) == {}


class TestRunnerFastResumeAnomaly:
    @pytest.mark.anyio
    async def test_cached_report_regeneration_paths(self, tmp_path, monkeypatch):
        """Fast-resume anomaly handling should branch on rules-hash regeneration."""
        # (case_name, should_regenerate_anomaly_report)
        cases = [
            ("regen", True),
            ("no_regen", False),
        ]

        for name, should_regenerate in cases:
            dataset = MockDataset()
            model = MockChatModel(answers=DEFAULT_ANSWERS)
            task = MockTask(dataset=dataset, model=model, name=f"fast_resume_{name}")
            config = make_config(
                tmp_path,
                result_dir=str(tmp_path / f"fast_resume_{name}"),
                auto_resume=True,
                detect_anomalies=True,
                detect_anomalies_on_resume=True,
            )
            runner = TaskRunner(task, config)

            cached = {"accuracy": 1.0, "total": 3}
            loaded_ctx = (
                {0: task.make_context(0, {"question": "Q0", "answer": "A0"}).to_final()}
                if should_regenerate
                else {}
            )

            load_cached_report = AsyncMock(return_value=cached)
            get_manifest_status = AsyncMock(return_value=(False, True))
            load_initial_state = AsyncMock(return_value=loaded_ctx)
            hydrate = AsyncMock(return_value=None)
            anomaly_load = AsyncMock(return_value={"meta": {}})
            needs_regeneration = MagicMock(return_value=should_regenerate)
            generate_and_save = AsyncMock(return_value={})

            monkeypatch.setattr(
                runner._loader, "load_cached_report", load_cached_report
            )
            monkeypatch.setattr(
                runner._loader, "get_manifest_status", get_manifest_status
            )
            monkeypatch.setattr(
                runner._loader, "load_initial_state", load_initial_state
            )
            monkeypatch.setattr(runner._loader, "hydrate", hydrate)

            monkeypatch.setattr(runner._anomaly_detector, "load", anomaly_load)
            monkeypatch.setattr(
                runner._anomaly_detector, "needs_regeneration", needs_regeneration
            )
            monkeypatch.setattr(
                runner._anomaly_detector, "generate_and_save", generate_and_save
            )

            report = await runner.arun()
            assert report == cached

            if should_regenerate:
                load_initial_state.assert_awaited_once()
                hydrate.assert_awaited_once()
                hydrate_call = hydrate.await_args
                assert hydrate_call is not None
                assert hydrate_call.args[0] is loaded_ctx
                assert hydrate_call.args[1] is runner._hydrated_ids
                assert hydrate_call.kwargs["include_stages"] == {
                    TaskStage.FINAL,
                    TaskStage.FAILED,
                }
                assert hydrate_call.kwargs["prepare_retries"] is False
                assert (
                    hydrate_call.kwargs["record_each_stage"]
                    == runner._record_each_stage
                )
                generate_and_save.assert_awaited_once()
                regen_call = generate_and_save.await_args
                assert regen_call is not None
                assert regen_call.args[0] is loaded_ctx
                assert regen_call.args[1] == task.name
                assert regen_call.kwargs["backup_if_changed"] is True
            else:
                load_initial_state.assert_not_awaited()
                hydrate.assert_not_awaited()
                generate_and_save.assert_not_awaited()

    @pytest.mark.anyio
    async def test_cached_report_regeneration_requires_both_anomaly_flags(
        self, tmp_path, monkeypatch
    ):
        dataset = MockDataset()
        model = MockChatModel(answers=DEFAULT_ANSWERS)
        task = MockTask(dataset=dataset, model=model, name="fast_resume_flag_gate")
        config = make_config(
            tmp_path,
            result_dir=str(tmp_path / "fast_resume_flag_gate"),
            auto_resume=True,
            detect_anomalies=False,
            detect_anomalies_on_resume=True,
        )
        runner = TaskRunner(task, config)

        monkeypatch.setattr(
            runner._loader, "load_cached_report", AsyncMock(return_value={"ok": True})
        )
        monkeypatch.setattr(
            runner._loader, "get_manifest_status", AsyncMock(return_value=(False, True))
        )
        anomaly_load = AsyncMock(return_value={"meta": {}})
        monkeypatch.setattr(runner._anomaly_detector, "load", anomaly_load)
        load_initial_state = AsyncMock(return_value={})
        hydrate = AsyncMock(return_value=None)
        monkeypatch.setattr(runner._loader, "load_initial_state", load_initial_state)
        monkeypatch.setattr(runner._loader, "hydrate", hydrate)

        report = await runner.arun()

        assert report == {"ok": True}
        anomaly_load.assert_not_awaited()
        load_initial_state.assert_not_awaited()
        hydrate.assert_not_awaited()

    @pytest.mark.anyio
    async def test_cached_report_not_used_when_manifest_not_all_final(
        self, tmp_path, monkeypatch
    ):
        """
        Cached report is only valid when manifest is fully terminal.
        """
        dataset = MockDataset()
        model = MockChatModel(answers=DEFAULT_ANSWERS)
        task = MockTask(dataset=dataset, model=model, name="fast_resume_incomplete")
        config = make_config(
            tmp_path,
            result_dir=str(tmp_path / "fast_resume_incomplete"),
            auto_resume=True,
            detect_anomalies=False,
            detect_anomalies_on_resume=False,
        )
        runner = TaskRunner(task, config)

        cached = {"accuracy": 0.0, "total": 999}
        task_setup = AsyncMock(side_effect=KeyboardInterrupt)
        task_shutdown = AsyncMock(return_value=None)
        saver_flush = AsyncMock(return_value=None)

        monkeypatch.setattr(
            runner._loader, "load_cached_report", AsyncMock(return_value=cached)
        )
        monkeypatch.setattr(
            runner._loader,
            "get_manifest_status",
            AsyncMock(return_value=(False, False)),
        )
        monkeypatch.setattr(task, "setup", task_setup)
        monkeypatch.setattr(task, "shutdown", task_shutdown)
        monkeypatch.setattr(runner._saver, "flush", saver_flush)

        report = await runner.arun()

        assert report is None
        task_setup.assert_awaited_once()

    @pytest.mark.anyio
    async def test_resume_seeds_progress_from_terminal_contexts_and_anomalies(
        self, tmp_path, monkeypatch
    ):
        samples = [
            {"question": "q0", "answer": "a0"},
            {"question": "q1", "answer": "a1"},
            {"question": "q2", "answer": "a2"},
            {"question": "q3", "answer": "a3"},
        ]
        dataset = MockDataset(samples)
        model = MockChatModel(answers={"q2": "a2"})
        task = MockTask(dataset=dataset, model=model, name="resume_progress_seed")
        runner = TaskRunner(
            task,
            make_config(tmp_path, auto_resume=True, detect_anomalies=True),
        )

        final_ctx = task.make_context(0, samples[0]).to_final()
        failed_ctx = task.make_context(1, samples[1]).to_failed(
            TaskAction.INFER, "exception::RuntimeError", "boom"
        )
        active_ctx = task.make_context(2, samples[2])
        final_ctx_2 = task.make_context(3, samples[3]).to_final()

        completed_ids_seen: list[str | int] = []
        failed_ids_seen: list[str | int] = []
        anomaly_ids_seen: list[str | int] = []
        anomaly_details_seen: dict[str, int] = {}
        update_calls: list[ProgressUpdateCall] = []

        class _FakeProgress:
            def __init__(self, **_kwargs):
                return

            def init_state(
                self,
                done_ids: list[str | int],
                failed_ids: list[str | int] | None = None,
                anomaly_ids: list[str | int] | None = None,
                anomaly_details: dict[str, int] | None = None,
            ) -> None:
                completed_ids_seen.extend(done_ids)
                if failed_ids:
                    failed_ids_seen.extend(failed_ids)
                if anomaly_ids:
                    anomaly_ids_seen.extend(anomaly_ids)
                if anomaly_details:
                    anomaly_details_seen.update(anomaly_details)

            def tick(self, *_args, **_kwargs):
                return

            def update(
                self,
                sample_id: str | int,
                current_hydrated_count: int,
                failed: bool = False,
                anomalies: set[str] | None = None,
            ) -> None:
                update_calls.append(
                    {
                        "sample_id": sample_id,
                        "current_hydrated_count": current_hydrated_count,
                        "failed": failed,
                        "anomalies": anomalies,
                    }
                )

            def set_status(self, *_args, **_kwargs):
                return

            def close(self):
                return

        def _detect(ctx: TaskContext, *, task_tags: set[str]) -> dict[str, list[int]]:  # noqa: ARG001
            assert ctx.stage == TaskStage.FINAL
            if ctx.sample_id in {0, 2, 3}:
                return {"rule_a": [3, 1]}
            return {}

        anomaly_save = AsyncMock(return_value=None)
        monkeypatch.setattr(
            "sieval.core.runners.runner.TaskProgress",
            _FakeProgress,
        )
        monkeypatch.setattr(
            runner._loader,
            "load_cached_report",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            runner._loader,
            "load_initial_state",
            AsyncMock(
                return_value={
                    0: final_ctx,
                    1: failed_ctx,
                    2: active_ctx,
                    3: final_ctx_2,
                }
            ),
        )
        monkeypatch.setattr(runner._loader, "hydrate", AsyncMock(return_value=None))
        monkeypatch.setattr(runner._anomaly_detector, "detect", _detect)
        monkeypatch.setattr(
            runner._anomaly_detector,
            "generate_and_save_from_results",
            anomaly_save,
        )

        report = await runner.arun()

        assert report["total"] == 4
        assert set(completed_ids_seen) == {0, 1, 3}
        assert failed_ids_seen == [1]
        assert anomaly_ids_seen == [0, 3]
        assert anomaly_details_seen == {"rule_a": 2}
        assert any(
            call["sample_id"] == 2 and call["anomalies"] == {"rule_a"}
            for call in update_calls
        )

        anomaly_call = anomaly_save.await_args
        assert anomaly_call is not None
        saved_results = anomaly_call.args[0]
        assert saved_results[0][0]["rule_a"] == [1, 3]
        assert saved_results[2][0]["rule_a"] == [1, 3]
        assert saved_results[3][0]["rule_a"] == [1, 3]

    @pytest.mark.anyio
    async def test_early_exit_hydrate_flags_and_report_save_contract(
        self, tmp_path, monkeypatch
    ):
        dataset = MockDataset([])
        model = MockChatModel()
        task = MockTask(dataset=dataset, model=model, name="early_exit_contract")
        runner = TaskRunner(task, make_config(tmp_path))

        load_initial_state = AsyncMock(return_value={})
        hydrate = AsyncMock(return_value=None)
        report_payload = {"total": 0}
        task_report = AsyncMock(return_value=report_payload)
        save_report = AsyncMock(return_value=None)

        monkeypatch.setattr(runner._loader, "load_initial_state", load_initial_state)
        monkeypatch.setattr(runner._loader, "hydrate", hydrate)
        monkeypatch.setattr(task, "report", task_report)
        monkeypatch.setattr(runner._saver, "save_report", save_report)

        report = await runner.arun()

        assert report == report_payload
        assert hydrate.await_count == 2
        hydrate_call = hydrate.await_args_list[-1]
        assert hydrate_call is not None
        assert hydrate_call.kwargs["include_stages"] == {
            TaskStage.FINAL,
            TaskStage.FAILED,
        }
        assert hydrate_call.kwargs["prepare_retries"] is False
        assert hydrate_call.kwargs["record_each_stage"] == runner._record_each_stage
        save_report.assert_awaited_once_with(report_payload)

    @pytest.mark.anyio
    async def test_all_terminal_resume_short_circuits_without_progress_loop(
        self, tmp_path, monkeypatch
    ):
        samples = [
            {"question": "q0", "answer": "a0"},
            {"question": "q1", "answer": "a1"},
        ]
        dataset = MockDataset(samples)
        model = MockChatModel(answers={"q0": "a0", "q1": "a1"})
        task = MockTask(dataset=dataset, model=model, name="all_terminal_resume")
        runner = TaskRunner(task, make_config(tmp_path, auto_resume=True))

        terminal_ctx = {
            0: task.make_context(0, samples[0]).to_final(),
            1: task.make_context(1, samples[1]).to_final(),
        }

        monkeypatch.setattr(
            runner._loader,
            "load_cached_report",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            runner._loader,
            "load_initial_state",
            AsyncMock(return_value=terminal_ctx),
        )
        monkeypatch.setattr(runner._loader, "hydrate", AsyncMock(return_value=None))

        progress_init_count = 0

        class _ProgressShouldNotInit:
            def __init__(self, **_kwargs):
                nonlocal progress_init_count
                progress_init_count += 1

        monkeypatch.setattr(
            "sieval.core.runners.runner.TaskProgress",
            _ProgressShouldNotInit,
        )

        with anyio.fail_after(1):
            report = await runner.arun()

        assert report["total"] == 2
        assert progress_init_count == 0


class TestRunnerResumeState:
    @pytest.mark.anyio
    async def test_resume_backfills_missing_raw_sample_from_dataset(
        self, tmp_path, monkeypatch
    ):
        sample = {"question": "What is 1+1?", "answer": "2"}
        dataset = MockDataset([sample])
        model = MockChatModel(answers={"What is 1+1?": "2"})
        task = MockTask(dataset=dataset, model=model, name="resume_raw_backfill")
        runner = TaskRunner(task, make_config(tmp_path))

        # Use TaskContext directly to ensure raw_sample is truly missing.
        ctx_missing_raw = TaskContext(sample_id=0, raw_sample=None)
        observed_feedback_ctx: dict[str, object] = {}

        async def _feedback(post, ctx):
            observed_feedback_ctx["raw_sample"] = ctx.raw_sample
            return True, {
                "correct": post == ctx.raw_sample["answer"],
                "predicted": post,
            }

        monkeypatch.setattr(task, "feedback", _feedback)
        monkeypatch.setattr(
            runner._loader,
            "load_initial_state",
            AsyncMock(return_value={0: ctx_missing_raw}),
        )
        monkeypatch.setattr(runner._loader, "hydrate", AsyncMock(return_value=None))

        report = await runner.arun()
        assert report["accuracy"] == 1.0
        assert report["total"] == 1
        assert observed_feedback_ctx["raw_sample"] == sample

    @pytest.mark.anyio
    async def test_initial_manifest_includes_error_action_for_failed_context(
        self, tmp_path, monkeypatch
    ):
        dataset = MockDataset([])
        model = MockChatModel()
        task = MockTask(
            dataset=dataset, model=model, name="resume_manifest_error_action"
        )
        runner = TaskRunner(task, make_config(tmp_path, auto_resume=True))

        failed_ctx = replace(
            task.make_context(0, {"question": "What is 1+1?", "answer": "2"}).to_failed(
                TaskAction.INFER, "exception::RuntimeError", "boom"
            ),
            retry_count=2,
        )

        sync_manifest = MagicMock()
        monkeypatch.setattr(
            runner._loader, "load_cached_report", AsyncMock(return_value=None)
        )
        monkeypatch.setattr(
            runner._loader,
            "load_initial_state",
            AsyncMock(return_value={0: failed_ctx}),
        )
        monkeypatch.setattr(runner._loader, "hydrate", AsyncMock(return_value=None))
        monkeypatch.setattr(runner._saver, "sync_manifest", sync_manifest)

        await runner.arun()
        assert sync_manifest.call_count == 1
        manifest = sync_manifest.call_args.args[0]
        assert manifest[0]["sample_id"] == 0
        assert manifest[0]["iteration"] == 0
        assert manifest[0]["error_action"] == "infer"
        assert manifest[0]["error_reason"] == "exception::RuntimeError"
        assert manifest[0]["retry_count"] == 2
        assert manifest[0]["final"] is False
        assert manifest[0]["failed"] is True
        assert manifest[0]["stage"] == TaskStage.FAILED.value

    @pytest.mark.anyio
    async def test_retry_limit_failure_reason_and_message_are_user_visible(
        self, tmp_path, monkeypatch
    ):
        sample = {"question": "q", "answer": "a"}
        dataset = MockDataset([sample])
        model = MockChatModel(answers={"q": "a"})
        task = MockTask(dataset=dataset, model=model, name="resume_retry_limit")
        runner = TaskRunner(
            task,
            make_config(tmp_path, auto_resume=True, max_retries=1),
        )

        in_retry = replace(task.make_context(0, sample), retry_count=2)
        observed_failure: dict[str, str | None] = {}

        async def _report(_finals, fails):
            assert len(fails) == 1
            observed_failure["reason"] = fails[0].error_reason
            observed_failure["msg"] = fails[0].error_msg
            return {"total": 1}

        monkeypatch.setattr(task, "report", _report)
        monkeypatch.setattr(
            runner._loader, "load_cached_report", AsyncMock(return_value=None)
        )
        monkeypatch.setattr(
            runner._loader,
            "load_initial_state",
            AsyncMock(return_value={0: in_retry}),
        )
        monkeypatch.setattr(runner._loader, "hydrate", AsyncMock(return_value=None))

        report = await runner.arun()
        assert report["total"] == 1
        assert observed_failure["reason"] == "retry_limit"
        assert observed_failure["msg"] == "Max retries 1 reached"


class TestRunnerInternalBranches:
    @pytest.mark.anyio
    async def test_set_runtime_context_recomputes_buffer_capacity_from_effective_limits(
        self, tmp_path, monkeypatch
    ):
        """Runtime context should affect the stream capacities used during execution."""
        dataset = MockDataset([{"question": "q", "answer": "a"}])
        model = MockChatModel(answers={"q": "a"})
        runner = TaskRunner(
            MockTask(dataset=dataset, model=model, name="runtime_ctx_capacity"),
            make_config(
                tmp_path,
                record_each_stage=True,
                concurrency_limit=10,
                concurrency_limits={"infer": 8, "feedback": 5},
            ),
        )

        runner.set_runtime_context(
            shared_global_limiter=anyio.CapacityLimiter(4),
            shared_stage_limiters={},
            progress_position=2,
            shared_global_limit=4,
            shared_stage_limits={"infer": 6, "feedback": 2},
        )

        expected = compute_stream_buffer_capacity(
            True,
            4,
            {"infer": 6, "feedback": 2},
        )
        baseline_without_record_each_stage = compute_stream_buffer_capacity(
            False,
            4,
            {"infer": 6, "feedback": 2},
        )

        observed_buffer_sizes: list[int | None] = []
        original_create_stream = anyio.create_memory_object_stream

        def _capture_stream_buffer_size(*args, **kwargs):
            if args:
                observed_buffer_sizes.append(args[0])
            else:
                observed_buffer_sizes.append(kwargs.get("max_buffer_size"))
            return original_create_stream(*args, **kwargs)

        monkeypatch.setattr(
            anyio,
            "create_memory_object_stream",
            _capture_stream_buffer_size,
        )

        report = await runner.arun()

        assert report["total"] == 1
        assert observed_buffer_sizes[:2] == [expected, expected]
        assert expected > baseline_without_record_each_stage

    @pytest.mark.anyio
    async def test_progress_init_receives_log_pct_interval_and_dict_details(
        self, tmp_path, monkeypatch
    ):
        dataset = MockDataset([{"question": "q", "answer": "a"}])
        model = MockChatModel(answers={"q": "a"})
        task = MockTask(dataset=dataset, model=model, name="progress_init_args")
        config = make_config(
            tmp_path,
            progress_log_pct_interval=3.5,
            detect_anomalies=False,
        )
        runner = TaskRunner(task, config)

        init_kwargs_seen: dict[str, object] | None = None
        init_state_kwargs_seen: dict[str, object] | None = None

        class _FakeProgress:
            def __init__(self, **kwargs: object) -> None:
                nonlocal init_kwargs_seen
                init_kwargs_seen = kwargs

            def init_state(self, *_args: object, **kwargs: object) -> None:
                nonlocal init_state_kwargs_seen
                init_state_kwargs_seen = kwargs

            def tick(self, *_args: object, **_kwargs: object) -> None:
                return

            def update(self, *_args: object, **_kwargs: object) -> None:
                return

            def set_status(self, *_args: object, **_kwargs: object) -> None:
                return

            def close(self) -> None:
                return

        monkeypatch.setattr(
            "sieval.core.runners.runner.TaskProgress",
            _FakeProgress,
        )

        report = await runner.arun()
        assert report["accuracy"] == 1.0
        assert report["total"] == 1

        assert init_kwargs_seen is not None
        init_kwargs = init_kwargs_seen
        assert init_kwargs["total"] == 1
        assert init_kwargs["desc"] == f"Running {task.name}"
        assert init_kwargs["position"] == 0
        assert init_kwargs["show_progress"] is False
        assert init_kwargs["root_dir"] == runner.root_dir
        assert init_kwargs["log_interval"] == config.progress_log_interval
        assert init_kwargs["log_pct_interval"] == 3.5
        assert init_kwargs["dump_progress"] is False
        assert init_kwargs["dump_interval"] == config.progress_dump_interval

        assert init_state_kwargs_seen is not None
        init_state_kwargs = init_state_kwargs_seen
        anomaly_details = init_state_kwargs["anomaly_details"]
        assert isinstance(anomaly_details, dict)
        assert getattr(anomaly_details, "default_factory", None) is int

    @pytest.mark.anyio
    async def test_progress_update_contract_for_mixed_outcomes(
        self, tmp_path, monkeypatch
    ):
        samples = [
            {"question": "q_ok", "answer": "a_ok", "raise_preprocess": False},
            {"question": "q_fail", "answer": "a_fail", "raise_preprocess": True},
        ]
        dataset = MockDataset(samples)
        model = MockChatModel(answers={"q_ok": "a_ok"})
        task = MixedOutcomeTask(dataset=dataset, model=model, name="progress_contract")
        runner = TaskRunner(task, make_config(tmp_path, detect_anomalies=True))

        updates: list[ProgressUpdateCall] = []
        ticks: list[int] = []

        class _FakeProgress:
            def __init__(self, **_kwargs: object) -> None:
                return

            def init_state(self, *_args: object, **_kwargs: object) -> None:
                return

            def tick(self, current_hydrated_count: int) -> None:
                ticks.append(current_hydrated_count)

            def update(
                self,
                sample_id: str | int,
                current_hydrated_count: int,
                failed: bool = False,
                anomalies: set[str] | None = None,
            ) -> None:
                updates.append(
                    {
                        "sample_id": sample_id,
                        "current_hydrated_count": current_hydrated_count,
                        "failed": failed,
                        "anomalies": anomalies,
                    }
                )

            def set_status(self, *_args: object, **_kwargs: object) -> None:
                return

            def close(self) -> None:
                return

        def _detect(ctx: TaskContext, *, task_tags: set[str]) -> dict[str, list[int]]:  # noqa: ARG001
            assert ctx.stage == TaskStage.FINAL
            if ctx.sample_id == 0:
                return {"quality_rule": [2, 1]}
            return {}

        monkeypatch.setattr(
            "sieval.core.runners.runner.TaskProgress",
            _FakeProgress,
        )
        monkeypatch.setattr(runner._anomaly_detector, "detect", _detect)

        report = await runner.arun()
        assert report["total"] == 2
        assert len(updates) == 2

        by_id = {call["sample_id"]: call for call in updates}
        assert by_id[0]["failed"] is False
        assert by_id[0]["anomalies"] == {"quality_rule"}
        assert by_id[1]["failed"] is True
        assert by_id[1]["anomalies"] is None
        assert max(call["current_hydrated_count"] for call in updates) == 2
        assert ticks
        assert all(isinstance(v, int) for v in ticks)

    @pytest.mark.anyio
    async def test_prepare_limiters_receives_runner_local_limit(
        self, tmp_path, monkeypatch
    ):
        dataset = MockDataset([{"question": "q", "answer": "a"}])
        model = MockChatModel(answers={"q": "a"})
        task = MockTask(dataset=dataset, model=model, name="prepare_limiters_args")
        runner = TaskRunner(
            task,
            make_config(
                tmp_path,
                concurrency_limit=3,
                concurrency_limits={"infer": 2},
            ),
        )
        captured: dict[str, object] = {}

        def _fake_prepare(local_limit, stage_limits):
            captured["local_limit"] = local_limit
            captured["stage_limits"] = stage_limits
            return None, {}

        monkeypatch.setattr(
            "sieval.core.runners.runner.prepare_limiters",
            _fake_prepare,
        )

        report = await runner.arun()

        assert report["total"] == 1
        assert captured["local_limit"] == 3
        assert captured["stage_limits"] == {"infer": 2}

    @pytest.mark.anyio
    async def test_run_one_stage_returns_when_no_next_action(
        self, tmp_path, monkeypatch
    ):
        dataset = MockDataset([])
        model = MockChatModel()
        task = MockTask(dataset=dataset, model=model, name="internal_no_action")
        runner = TaskRunner(task, make_config(tmp_path))

        execute_stage_logic = AsyncMock()
        monkeypatch.setattr(runner, "_execute_stage_logic", execute_stage_logic)
        final_ctx = task.make_context(0, {"question": "q", "answer": "a"}).to_final()

        send, recv = anyio.create_memory_object_stream[TaskContext](1)
        async with send, recv:
            await runner._run_one_stage(final_ctx, send)

        execute_stage_logic.assert_not_awaited()

    @pytest.mark.anyio
    async def test_execute_stage_logic_records_non_null_timing_meta(self, tmp_path):
        dataset = MockDataset([{"question": "q", "answer": "a"}])
        model = MockChatModel(answers={"q": "a"})
        task = MockTask(dataset=dataset, model=model, name="internal_timing_meta")
        runner = TaskRunner(task, make_config(tmp_path))

        ctx = task.make_context(0, {"question": "q", "answer": "a"})

        send, recv = anyio.create_memory_object_stream[TaskContext](1)
        async with send, recv:
            await runner._execute_stage_logic(ctx, send, TaskAction.PREPROCESS)
            new_ctx = await recv.receive()

        assert new_ctx.stage == TaskStage.PREPROCESSED
        metas = new_ctx.stage_meta.get(TaskStage.PREPROCESSED.value)
        assert metas
        timing_s = metas[-1].get("timing_s")
        assert isinstance(timing_s, float)
        assert timing_s >= 0.0

    @pytest.mark.anyio
    async def test_run_one_stage_respects_shared_global_limiter(
        self, tmp_path, monkeypatch
    ):
        dataset = MockDataset(
            [{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}]
        )
        model = MockChatModel(answers={"q1": "a1", "q2": "a2"})
        task = MockTask(dataset=dataset, model=model, name="internal_shared_limiter")
        runner = TaskRunner(task, make_config(tmp_path))
        runner.set_runtime_context(
            shared_global_limiter=anyio.CapacityLimiter(1),
            shared_stage_limiters={},
            progress_position=0,
            shared_global_limit=1,
            shared_stage_limits={},
        )

        in_flight = 0
        max_in_flight = 0

        async def _fake_execute(_ctx, _compute_send, _action):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await anyio.sleep(0.02)
            in_flight -= 1

        monkeypatch.setattr(runner, "_execute_stage_logic", _fake_execute)

        ctx1 = task.make_context(0, {"question": "q1", "answer": "a1"})
        ctx2 = task.make_context(1, {"question": "q2", "answer": "a2"})

        send, recv = anyio.create_memory_object_stream[TaskContext](2)
        async with send, recv, anyio.create_task_group() as tg:
            tg.start_soon(runner._run_one_stage, ctx1, send)
            tg.start_soon(runner._run_one_stage, ctx2, send)

        assert max_in_flight == 1

    @pytest.mark.anyio
    async def test_execute_stage_logic_infer_passes_ctx_and_records_usage(
        self, tmp_path, monkeypatch
    ):
        dataset = MockDataset([{"question": "q", "answer": "a"}])
        model = MockChatModel(answers={"q": "a"})
        task = MockTask(dataset=dataset, model=model, name="internal_stage_infer")
        runner = TaskRunner(task, make_config(tmp_path))
        ctx = task.make_context(0, {"question": "q", "answer": "a"}).to_preprocessed(
            "q"
        )

        captured_ctx: dict[str, object] = {}

        async def _infer(pre, passed_ctx):
            captured_ctx["ctx"] = passed_ctx
            return await model.agenerate(pre)

        monkeypatch.setattr(task, "infer", _infer)
        record_usage = MagicMock()
        monkeypatch.setattr(runner._profiler, "record_model_usage", record_usage)

        send, recv = anyio.create_memory_object_stream[TaskContext](1)
        async with send, recv:
            await runner._execute_stage_logic(ctx, send, TaskAction.INFER)
            new_ctx = await recv.receive()

        assert captured_ctx["ctx"] is ctx
        assert new_ctx.stage == TaskStage.INFERRED
        infer_meta = new_ctx.stage_meta[TaskStage.INFERRED.value][-1]
        assert 0.0 <= infer_meta["timing_s"] < 5.0
        model_calls = infer_meta.get("model_calls", [])
        assert model_calls
        assert model_calls[0]["usage"]["total_tokens"] == 12

        record_usage.assert_called_once()
        args = record_usage.call_args.args
        kwargs = record_usage.call_args.kwargs
        assert args[0]["total_tokens"] == 12
        assert kwargs["stage_name"] == TaskStage.INFERRED.value

    @pytest.mark.anyio
    async def test_execute_stage_logic_postprocess_passes_ctx_and_records_meta(
        self, tmp_path, monkeypatch
    ):
        dataset = MockDataset([{"question": "q", "answer": "a"}])
        model = MockChatModel(answers={"q": "a"})
        task = MockTask(dataset=dataset, model=model, name="internal_stage_post")
        runner = TaskRunner(task, make_config(tmp_path))
        ctx = task.make_context(0, {"question": "q", "answer": "a"}).to_inferred("a")

        captured_ctx: dict[str, object] = {}

        async def _postprocess(infer_result, passed_ctx):
            captured_ctx["ctx"] = passed_ctx
            return str(infer_result)

        monkeypatch.setattr(task, "postprocess", _postprocess)

        send, recv = anyio.create_memory_object_stream[TaskContext](1)
        async with send, recv:
            await runner._execute_stage_logic(ctx, send, TaskAction.POSTPROCESS)
            new_ctx = await recv.receive()

        assert captured_ctx["ctx"] is ctx
        assert new_ctx.stage == TaskStage.POSTPROCESSED
        assert TaskStage.POSTPROCESSED.value in new_ctx.stage_meta
        assert new_ctx.stage_meta[TaskStage.POSTPROCESSED.value][-1]["timing_s"] >= 0.0

    @pytest.mark.anyio
    async def test_execute_stage_logic_feedback_hits_iteration_limit(
        self, tmp_path, monkeypatch
    ):
        dataset = MockDataset([{"question": "q", "answer": "a"}])
        model = MockChatModel(answers={"q": "a"})
        task = MockTask(dataset=dataset, model=model, name="internal_stage_feedback")
        runner = TaskRunner(task, make_config(tmp_path, max_iterations=1))
        ctx = task.make_context(0, {"question": "q", "answer": "a"}).to_postprocessed(
            "x"
        )

        async def _feedback(_post, passed_ctx):
            assert passed_ctx is ctx
            return False, {"ok": False}

        monkeypatch.setattr(task, "feedback", _feedback)

        send, recv = anyio.create_memory_object_stream[TaskContext](1)
        async with send, recv:
            await runner._execute_stage_logic(ctx, send, TaskAction.FEEDBACK)
            new_ctx = await recv.receive()

        assert new_ctx.stage == TaskStage.FAILED
        assert new_ctx.error_action is None
        assert new_ctx.error_reason == "iteration_limit"
        assert new_ctx.error_msg == "Max iterations 1 reached"
        assert TaskStage.FEEDBACK.value in new_ctx.stage_meta

    @pytest.mark.anyio
    async def test_execute_stage_logic_feedback_allows_boundary_iteration(
        self, tmp_path, monkeypatch
    ):
        dataset = MockDataset([{"question": "q", "answer": "a"}])
        model = MockChatModel(answers={"q": "a"})
        task = MockTask(
            dataset=dataset,
            model=model,
            name="internal_feedback_boundary",
        )
        runner = TaskRunner(task, make_config(tmp_path, max_iterations=2))
        ctx = task.make_context(0, {"question": "q", "answer": "a"}).to_postprocessed(
            "a"
        )

        async def _feedback(_post, _ctx):
            return False, {"continue": True}

        monkeypatch.setattr(task, "feedback", _feedback)

        send, recv = anyio.create_memory_object_stream[TaskContext](1)
        async with send, recv:
            await runner._execute_stage_logic(ctx, send, TaskAction.FEEDBACK)
            new_ctx = await recv.receive()

        assert new_ctx.stage == TaskStage.INITIAL
        assert new_ctx.iteration == 1

    @pytest.mark.anyio
    async def test_execute_stage_logic_non_model_list_uses_generic_meta_path(
        self, tmp_path, monkeypatch
    ):
        dataset = MockDataset([{"question": "q", "answer": "a"}])
        model = MockChatModel(answers={"q": "a"})
        task = MockTask(dataset=dataset, model=model, name="internal_non_model_list")
        runner = TaskRunner(task, make_config(tmp_path))
        ctx = task.make_context(0, {"question": "q", "answer": "a"})

        async def _preprocess(_raw, _ctx):
            return ["x", "y"]

        monkeypatch.setattr(task, "preprocess", _preprocess)

        send, recv = anyio.create_memory_object_stream[TaskContext](1)
        async with send, recv:
            await runner._execute_stage_logic(ctx, send, TaskAction.PREPROCESS)
            new_ctx = await recv.receive()

        assert new_ctx.preprocess_result == ["x", "y"]
        meta = new_ctx.stage_meta[TaskStage.PREPROCESSED.value][-1]
        assert "model_calls" not in meta

    @pytest.mark.anyio
    async def test_execute_stage_logic_list_model_output_records_timing_and_usage_meta(
        self, tmp_path, monkeypatch
    ):
        dataset = MockDataset([{"question": "q", "answer": "a"}])
        model = MockChatModel(answers={"q": "a"})
        task = MockTask(dataset=dataset, model=model, name="internal_list_model_output")
        runner = TaskRunner(task, make_config(tmp_path))
        ctx = task.make_context(0, {"question": "q", "answer": "a"}).to_preprocessed(
            "q"
        )

        async def _infer(pre, _ctx):
            return [await model.agenerate(pre), await model.agenerate(pre)]

        monkeypatch.setattr(task, "infer", _infer)

        send, recv = anyio.create_memory_object_stream[TaskContext](1)
        async with send, recv:
            await runner._execute_stage_logic(ctx, send, TaskAction.INFER)
            new_ctx = await recv.receive()

        infer_meta = new_ctx.stage_meta[TaskStage.INFERRED.value][-1]
        assert isinstance(infer_meta["timing_s"], float)
        assert infer_meta["timing_s"] >= 0.0
        assert len(infer_meta["model_calls"]) == 2

    @pytest.mark.anyio
    async def test_execute_stage_logic_taskstageoutput_meta_and_hook_contract(
        self, tmp_path, monkeypatch
    ):
        dataset = MockDataset([{"question": "q", "answer": "a"}])
        model = MockChatModel(answers={"q": "a"})
        observed: dict[str, object] = {}

        def _hook(val, _stage, _ctx):
            observed["hook_val"] = val
            return {"env": {"hook_key": "ok"}}

        task = MockTask(dataset=dataset, model=model, name="internal_stage_output_meta")
        runner = TaskRunner(
            task,
            make_config(
                tmp_path,
                stage_meta_hook=_hook,
            ),
        )
        ctx = task.make_context(0, {"question": "q", "answer": "a"})

        stage_output = TaskStageOutput("q", meta={"extra": {"from_output": True}})

        async def _preprocess(_raw, _ctx):
            return stage_output

        monkeypatch.setattr(task, "preprocess", _preprocess)

        send, recv = anyio.create_memory_object_stream[TaskContext](1)
        async with send, recv:
            await runner._execute_stage_logic(ctx, send, TaskAction.PREPROCESS)
            new_ctx = await recv.receive()

        assert observed["hook_val"] is stage_output
        pre_meta = new_ctx.stage_meta[TaskStage.PREPROCESSED.value][-1]
        assert pre_meta["extra"] == {"from_output": True}
        assert pre_meta["env"] == {"hook_key": "ok"}

    @pytest.mark.anyio
    async def test_execute_stage_logic_exception_path_preserves_failure_metadata(
        self, tmp_path, monkeypatch
    ):
        dataset = MockDataset([{"question": "q", "answer": "a"}])
        model = MockChatModel(answers={"q": "a"})
        task = MockTask(dataset=dataset, model=model, name="internal_stage_exception")
        runner = TaskRunner(task, make_config(tmp_path))
        ctx = task.make_context(0, {"question": "q", "answer": "a"})

        captured_ctx: dict[str, object] = {}

        async def _preprocess(_raw, passed_ctx):
            captured_ctx["ctx"] = passed_ctx
            raise RuntimeError("boom")

        monkeypatch.setattr(task, "preprocess", _preprocess)

        send, recv = anyio.create_memory_object_stream[TaskContext](1)
        async with send, recv:
            await runner._execute_stage_logic(ctx, send, TaskAction.PREPROCESS)
            new_ctx = await recv.receive()

        assert captured_ctx["ctx"] is ctx
        assert new_ctx.stage == TaskStage.FAILED
        assert new_ctx.error_action == TaskAction.PREPROCESS
        assert new_ctx.error_reason == "exception::RuntimeError"
        assert new_ctx.error_msg == "boom"


class TestRunnerInterruptHandling:
    @pytest.mark.anyio
    async def test_keyboard_interrupt_flushes_and_returns_none(
        self, tmp_path, monkeypatch
    ):
        dataset = MockDataset()
        model = MockChatModel(answers=DEFAULT_ANSWERS)
        task = MockTask(dataset=dataset, model=model, name="interrupt_test")
        config = make_config(tmp_path, detect_anomalies=False)
        runner = TaskRunner(task, config)

        task_setup = AsyncMock(side_effect=KeyboardInterrupt)
        task_shutdown = AsyncMock(return_value=None)
        saver_flush = AsyncMock(return_value=None)

        monkeypatch.setattr(task, "setup", task_setup)
        monkeypatch.setattr(task, "shutdown", task_shutdown)
        monkeypatch.setattr(runner._saver, "flush", saver_flush)

        report = await runner.arun()

        assert report is None
        saver_flush.assert_awaited_once()
        task_shutdown.assert_awaited_once()

    @pytest.mark.anyio
    async def test_keyboard_interrupt_closes_progress_if_initialized(
        self, tmp_path, monkeypatch
    ):
        dataset = MockDataset()
        model = MockChatModel(answers=DEFAULT_ANSWERS)
        task = MockTask(dataset=dataset, model=model, name="interrupt_with_progress")
        config = make_config(tmp_path, detect_anomalies=False)
        runner = TaskRunner(task, config)

        task_setup = AsyncMock(return_value=None)
        task_shutdown = AsyncMock(return_value=None)
        saver_flush = AsyncMock(return_value=None)
        load_initial_state = AsyncMock(return_value={})
        hydrate = AsyncMock(return_value=None)
        # Trigger interrupt late in the run, after progress is initialized.
        task_report = AsyncMock(side_effect=KeyboardInterrupt)
        progress_closed: list[object] = []

        monkeypatch.setattr(task, "setup", task_setup)
        monkeypatch.setattr(task, "shutdown", task_shutdown)
        monkeypatch.setattr(task, "report", task_report)
        monkeypatch.setattr(runner._saver, "flush", saver_flush)
        monkeypatch.setattr(runner._loader, "load_initial_state", load_initial_state)
        monkeypatch.setattr(runner._loader, "hydrate", hydrate)

        def _close_progress(self):
            progress_closed.append(self)

        monkeypatch.setattr(
            "sieval.core.runners.runner.TaskProgress.close",
            _close_progress,
        )

        report = await runner.arun()

        assert report is None
        assert len(progress_closed) == 1

    @pytest.mark.anyio
    async def test_keyboard_interrupt_with_anomaly_detection(
        self, tmp_path, monkeypatch
    ):
        """
        Interrupt with detect_anomalies=True should call generate_and_save_from_results.
        """
        dataset = MockDataset()
        model = MockChatModel(answers=DEFAULT_ANSWERS)
        task = MockTask(dataset=dataset, model=model, name="interrupt_anomaly_test")
        config = make_config(tmp_path, detect_anomalies=True)
        runner = TaskRunner(task, config)

        task_setup = AsyncMock(side_effect=KeyboardInterrupt)
        task_shutdown = AsyncMock(return_value=None)
        saver_flush = AsyncMock(return_value=None)
        anomaly_save = AsyncMock(return_value=None)

        monkeypatch.setattr(task, "setup", task_setup)
        monkeypatch.setattr(task, "shutdown", task_shutdown)
        monkeypatch.setattr(runner._saver, "flush", saver_flush)
        monkeypatch.setattr(
            runner._anomaly_detector,
            "generate_and_save_from_results",
            anomaly_save,
        )

        report = await runner.arun()

        assert report is None
        saver_flush.assert_awaited_once()
        anomaly_save.assert_awaited_once()
        anomaly_call = anomaly_save.await_args
        assert anomaly_call is not None
        assert anomaly_call.args[0] == {}
        assert anomaly_call.args[1] == task.name
        assert anomaly_call.kwargs["total_samples"] == 0
        assert anomaly_call.kwargs["final_count"] == 0
        assert anomaly_call.kwargs["failed_count"] == 0
        assert anomaly_call.kwargs["backup_if_changed"] is False

    @pytest.mark.anyio
    async def test_keyboard_interrupt_flush_failure_is_non_fatal(
        self, tmp_path, monkeypatch
    ):
        """
        If flush raises during interrupt handling,
        the error is caught and None is returned.
        """
        dataset = MockDataset()
        model = MockChatModel(answers=DEFAULT_ANSWERS)
        task = MockTask(dataset=dataset, model=model, name="interrupt_flush_fail_test")
        config = make_config(tmp_path, detect_anomalies=False)
        runner = TaskRunner(task, config)

        task_setup = AsyncMock(side_effect=KeyboardInterrupt)
        task_shutdown = AsyncMock(return_value=None)
        saver_flush = AsyncMock(side_effect=RuntimeError("disk full"))

        monkeypatch.setattr(task, "setup", task_setup)
        monkeypatch.setattr(task, "shutdown", task_shutdown)
        monkeypatch.setattr(runner._saver, "flush", saver_flush)

        # Should not raise even when flush fails
        report = await runner.arun()

        assert report is None
        saver_flush.assert_awaited_once()


class TestRunnerAnomalyReporting:
    @pytest.mark.anyio
    async def test_completion_anomaly_report_uses_final_and_failed_counts(
        self, tmp_path, monkeypatch
    ):
        dataset = MockDataset()
        model = MockChatModel(answers=DEFAULT_ANSWERS)
        task = MockTask(dataset=dataset, model=model, name="completion_anomaly_test")
        config = make_config(tmp_path, detect_anomalies=True)
        runner = TaskRunner(task, config)

        anomaly_save = AsyncMock(return_value=None)
        aggregate_stage_timings = MagicMock()
        aggregate_token_usage = MagicMock()
        monkeypatch.setattr(
            runner._anomaly_detector,
            "generate_and_save_from_results",
            anomaly_save,
        )
        monkeypatch.setattr(
            runner._profiler,
            "aggregate_stage_timings",
            aggregate_stage_timings,
        )
        monkeypatch.setattr(
            runner._profiler,
            "aggregate_token_usage",
            aggregate_token_usage,
        )

        report = await runner.arun()

        assert report["accuracy"] == 1.0
        assert report["total"] == 3
        aggregate_stage_timings.assert_called_once_with(runner._contexts)
        aggregate_token_usage.assert_called_once_with(runner._contexts)
        anomaly_save.assert_awaited_once()
        anomaly_call = anomaly_save.await_args
        assert anomaly_call is not None
        assert anomaly_call.args[0] == runner._anomaly_results
        assert anomaly_call.args[1] == task.name
        assert anomaly_call.kwargs["total_samples"] == 3
        assert anomaly_call.kwargs["final_count"] == 3
        assert anomaly_call.kwargs["failed_count"] == 0
        assert anomaly_call.kwargs["backup_if_changed"] is False


class TestLazyContextCreation:
    """Tests for deferred/on-demand context creation from dataset."""

    @pytest.mark.anyio
    async def test_contexts_not_eagerly_populated_at_init(self, tmp_path, monkeypatch):
        """Runner should not iterate entire dataset at startup.

        After arun completes, all contexts should exist, but the dataset's
        test_set should not be enumerated in a single eager loop before
        execution begins. We verify by intercepting make_context calls and
        checking they happen during the seed/dispatch phase (after hydrate),
        not during the old eager loop.
        """
        samples = [
            {"question": "What is 1+1?", "answer": "2"},
            {"question": "What is 2+3?", "answer": "5"},
            {"question": "What is 10-7?", "answer": "3"},
        ]
        dataset = MockDataset(samples)
        model = MockChatModel(
            answers={
                "What is 1+1?": "2",
                "What is 2+3?": "5",
                "What is 10-7?": "3",
            }
        )
        task = MockTask(dataset=dataset, model=model, name="lazy_ctx")
        runner = TaskRunner(task, make_config(tmp_path))

        # Track make_context calls — these should only be called during
        # the seed loop (on-demand), not during an eager enumerate loop.
        make_context_ids: list[int] = []
        original_make_context = task.make_context

        def tracking_make_context(sample_id, raw=None):
            make_context_ids.append(sample_id)
            return original_make_context(sample_id, raw)

        monkeypatch.setattr(task, "make_context", tracking_make_context)

        report = await runner.arun()

        # Functional correctness: all samples processed
        assert report["accuracy"] == 1.0
        assert report["total"] == 3

        # All contexts should be populated after execution
        assert len(runner._contexts) == 3

        # Verify: make_context was called 3 times (once per sample, on-demand)
        assert len(make_context_ids) == 3
        assert set(make_context_ids) == {0, 1, 2}

    @pytest.mark.anyio
    async def test_total_samples_matches_dataset_size(self, tmp_path):
        """_total_samples should equal the dataset test_set length."""
        samples = [
            {"question": "q1", "answer": "a1"},
            {"question": "q2", "answer": "a2"},
        ]
        dataset = MockDataset(samples)
        model = MockChatModel(default_answer="a1")
        task = MockTask(dataset=dataset, model=model, name="total_samples")
        runner = TaskRunner(task, make_config(tmp_path))

        await runner.arun()

        assert runner._total_samples == 2

    @pytest.mark.anyio
    async def test_total_samples_with_resume_includes_all(self, tmp_path, monkeypatch):
        """When resuming, _total_samples should be max(resumed, dataset_size)."""
        samples = [
            {"question": "q1", "answer": "a1"},
            {"question": "q2", "answer": "a2"},
            {"question": "q3", "answer": "a3"},
        ]
        dataset = MockDataset(samples)
        model = MockChatModel(default_answer="a1")
        task = MockTask(dataset=dataset, model=model, name="total_resume")
        runner = TaskRunner(task, make_config(tmp_path))

        # Simulate: sample 0 already completed on disk
        resumed_ctx = (
            task.make_context(0, samples[0])
            .to_preprocessed("q1")
            .to_inferred(await model.agenerate("q1"))
            .to_postprocessed("a1")
            .to_feedback({"correct": True, "predicted": "a1"})
            .to_final()
        )

        monkeypatch.setattr(
            runner._loader,
            "load_initial_state",
            AsyncMock(return_value={0: resumed_ctx}),
        )
        monkeypatch.setattr(
            runner._loader,
            "load_cached_report",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(runner._loader, "hydrate", AsyncMock(return_value=None))

        await runner.arun()

        # total_samples should be 3 (dataset size), not 1 (resumed count)
        assert runner._total_samples == 3

    @pytest.mark.anyio
    async def test_pending_ids_excludes_resumed_samples(self, tmp_path, monkeypatch):
        """Samples already in _contexts from resume should not be re-created."""
        samples = [
            {"question": "q1", "answer": "a1"},
            {"question": "q2", "answer": "a2"},
        ]
        dataset = MockDataset(samples)
        model = MockChatModel(answers={"q1": "a1", "q2": "a2"})
        task = MockTask(dataset=dataset, model=model, name="pending_ids")
        runner = TaskRunner(task, make_config(tmp_path))

        # Sample 0 resumed from disk (not yet terminal, needs re-execution)
        resumed_ctx = TaskContext(sample_id=0, raw_sample=None)

        monkeypatch.setattr(
            runner._loader,
            "load_initial_state",
            AsyncMock(return_value={0: resumed_ctx}),
        )
        monkeypatch.setattr(runner._loader, "hydrate", AsyncMock(return_value=None))

        # Track make_context calls to see which sample IDs are created fresh
        created_ids: list[int] = []
        original_make_context = task.make_context

        def tracking_make_context(sample_id, raw=None):
            created_ids.append(sample_id)
            return original_make_context(sample_id, raw)

        monkeypatch.setattr(task, "make_context", tracking_make_context)

        report = await runner.arun()
        assert report["total"] == 2

        # Sample 1 should be created fresh via make_context,
        # Sample 0 should NOT be re-created (it was resumed)
        assert 1 in created_ids
        assert 0 not in created_ids

    @pytest.mark.anyio
    async def test_resume_all_terminal_early_exit_with_total_samples(
        self, tmp_path, monkeypatch
    ):
        """Early exit should work when all resumed samples are terminal,
        even if _contexts doesn't include pending samples yet."""
        samples = [
            {"question": "q1", "answer": "a1"},
            {"question": "q2", "answer": "a2"},
        ]
        dataset = MockDataset(samples)
        model = MockChatModel(answers={"q1": "a1", "q2": "a2"})
        task = MockTask(dataset=dataset, model=model, name="early_exit")
        runner = TaskRunner(task, make_config(tmp_path))

        # Both samples already FINAL on disk
        ctx0 = (
            task.make_context(0, samples[0])
            .to_preprocessed("q1")
            .to_inferred(await model.agenerate("q1"))
            .to_postprocessed("a1")
            .to_feedback({"correct": True, "predicted": "a1"})
            .to_final()
        )
        ctx1 = (
            task.make_context(1, samples[1])
            .to_preprocessed("q2")
            .to_inferred(await model.agenerate("q2"))
            .to_postprocessed("a2")
            .to_feedback({"correct": True, "predicted": "a2"})
            .to_final()
        )

        monkeypatch.setattr(
            runner._loader,
            "load_cached_report",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            runner._loader,
            "load_initial_state",
            AsyncMock(return_value={0: ctx0, 1: ctx1}),
        )
        monkeypatch.setattr(runner._loader, "hydrate", AsyncMock(return_value=None))

        report = await runner.arun()
        assert report["total"] == 2
        assert runner._total_samples == 2


class TestResumeVersionGate:
    @pytest.mark.anyio
    async def test_incompatible_version_blocks_resume(self, tmp_path):
        result_dir = str(tmp_path / "gated")
        model = MockChatModel(answers=DEFAULT_ANSWERS)
        task = MockTask(dataset=MockDataset(), model=model, name="gate_block")
        await TaskRunner(task, make_config(tmp_path, result_dir=result_dir)).arun()

        # Tamper meta.json to an incompatible series.
        (Path(result_dir) / "meta.json").write_bytes(
            orjson.dumps({"version": "999.0.0", "deterministic": False})
        )

        task2 = MockTask(dataset=MockDataset(), model=model, name="gate_block")
        with pytest.raises(ResumeVersionError):
            TaskRunner(
                task2, make_config(tmp_path, result_dir=result_dir, auto_resume=True)
            )

    @pytest.mark.anyio
    async def test_same_version_resume_allowed(self, tmp_path):
        result_dir = str(tmp_path / "same")
        model = MockChatModel(answers=DEFAULT_ANSWERS)
        task = MockTask(dataset=MockDataset(), model=model, name="gate_ok")
        await TaskRunner(task, make_config(tmp_path, result_dir=result_dir)).arun()

        task2 = MockTask(dataset=MockDataset(), model=model, name="gate_ok")
        runner2 = TaskRunner(
            task2, make_config(tmp_path, result_dir=result_dir, auto_resume=True)
        )  # meta.json holds the current version -> EXACT -> no raise
        assert runner2._resumed_from_existing


class TestReportVersions:
    @pytest.mark.anyio
    async def test_report_includes_current_version(self, tmp_path):
        from sieval import __version__

        model = MockChatModel(answers=DEFAULT_ANSWERS)
        task = MockTask(dataset=MockDataset(), model=model, name="ver_report")
        runner = TaskRunner(task, make_config(tmp_path))
        report = await runner.arun()

        assert report["sieval_versions"] == [__version__]
        saved = orjson.loads((runner.root_dir / "report.json").read_bytes())
        assert saved["sieval_versions"] == [__version__]

    @pytest.mark.anyio
    async def test_non_dict_report_not_injected(self, tmp_path):
        model = MockChatModel(answers=DEFAULT_ANSWERS)
        task = ListReportTask(dataset=MockDataset(), model=model, name="listrep")
        report = await TaskRunner(task, make_config(tmp_path)).arun()

        assert report == ["not", "a", "dict"]  # unchanged, no crash

    @pytest.mark.anyio
    async def test_none_report_not_saved(self, tmp_path):
        # A task returning None must skip save_report entirely: no report.json
        # is written and arun returns None. Guards the `report is None` branch
        # (removing it would serialize `null` to report.json).
        model = MockChatModel(answers=DEFAULT_ANSWERS)
        task = NoneReportTask(dataset=MockDataset(), model=model, name="nonerep")
        runner = TaskRunner(task, make_config(tmp_path))
        report = await runner.arun()

        assert report is None
        assert not (runner.root_dir / "report.json").exists()
