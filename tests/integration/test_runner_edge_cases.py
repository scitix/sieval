"""
Integration tests: runner edge cases not covered by existing test files.

Covers:
  - Fast resume: auto_resume returns cached report when all samples are final
    and detect_anomalies=False.
  - Resume with failures: auto_resume re-processes failed samples instead of
    returning the cached report.
  - Anomaly report completeness on resume: anomalies.json includes samples that
    completed before the resume, not just newly completed samples.
  - dump_progress=True: progress.json is written to the result directory
    (TaskProgress._dump_state path via progress.py).

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

import json
from pathlib import Path
from typing import ClassVar

import pytest

from sieval.core.models import ModelOutput
from sieval.core.runners.runner import TaskRunner
from sieval.core.tasks.consts import TaskStage
from sieval.core.tasks.task import Task
from tests.conftest import (
    MockAlwaysFailModel,
    MockChatModel,
    MockCountingChatModel,
    MockDataset,
    make_config,
)

# ===================================================================
# Shared task definition reused across all test classes
# ===================================================================
EDGE_SAMPLES = [
    {"question": "E1", "answer": "A1"},
    {"question": "E2", "answer": "A2"},
    {"question": "E3", "answer": "A3"},
]

EDGE_ANSWERS = {"E1": "A1", "E2": "A2", "E3": "A3"}


class EdgeTask(Task):
    """Minimal single-turn task for edge-case runner tests."""

    model_type = "chat"
    tags: ClassVar[set[str]] = {"gen", "zero_shot"}

    async def preprocess(self, raw, ctx):
        return raw["question"]

    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

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
        return {
            "accuracy": correct / total if total else 0.0,
            "total": total,
            "completed": len(finals),
            "failed": len(fails),
        }


class CountingMockChatModel(MockCountingChatModel):
    pass


# ===================================================================
# TestRunnerAnomalyOnResume
# ===================================================================
class TestRunnerAnomalyOnResume:
    @pytest.mark.anyio
    async def test_full_resume_returns_cached_report(self, tmp_path):
        """
        After a complete run, a second run with auto_resume=True and
        detect_anomalies=False must return the exact same cached report
        without invoking the model again (fast-path at runner.py).
        """
        result_dir = str(tmp_path / "cached_report_results")
        dataset = MockDataset(EDGE_SAMPLES)

        # --- First run: complete all samples ---
        model1 = MockChatModel(answers=EDGE_ANSWERS)
        task1 = EdgeTask(dataset=dataset, model=model1, name="cached_report_test")
        config1 = make_config(tmp_path, result_dir=result_dir, detect_anomalies=False)
        runner1 = TaskRunner(task1, config1)
        report1 = await runner1.arun()

        assert report1 is not None
        assert report1["accuracy"] == 1.0
        assert report1["completed"] == 3

        # --- Second run: auto_resume with a counting model ---
        counting_model = CountingMockChatModel(answers=EDGE_ANSWERS)
        task2 = EdgeTask(
            dataset=dataset, model=counting_model, name="cached_report_test"
        )
        config2 = make_config(
            tmp_path,
            result_dir=result_dir,
            auto_resume=True,
            detect_anomalies=False,
        )
        runner2 = TaskRunner(task2, config2)
        report2 = await runner2.arun()

        # The cached report must be returned unchanged
        assert report2 is not None
        assert report2 == report1, (
            f"Expected cached report {report1!r}, got {report2!r}"
        )
        # Model must NOT have been called at all (fast-path hit)
        assert counting_model.call_count == 0, (
            f"Model was called {counting_model.call_count} times on fast resume — "
            "the cached report path was not taken."
        )

    @pytest.mark.anyio
    async def test_early_exit_hydrates_cross_stage_data(self, tmp_path):
        """
        When all samples are already terminal and the early-exit path is taken
        (runner.py step 5), hydrate() must still load cross-stage results from
        disk so that task.report() receives complete contexts.

        This test catches the regression where record_each_stage was not
        forwarded to the early-exit hydrate() call, causing infer_result and
        preprocess_result to remain None.
        """
        result_dir = str(tmp_path / "early_exit_hydrate_results")
        dataset = MockDataset(EDGE_SAMPLES)

        # --- First run: complete all samples with record_each_stage=True ---
        model = MockChatModel(answers=EDGE_ANSWERS)
        task1 = EdgeTask(dataset=dataset, model=model, name="early_exit_hydrate_test")
        config1 = make_config(
            tmp_path,
            result_dir=result_dir,
            detect_anomalies=False,
            record_each_stage=True,
            record_meta=True,
        )
        runner1 = TaskRunner(task1, config1)
        await runner1.arun()

        # --- Second run: auto_resume triggers early-exit path ---
        # All samples are FINAL, so step 5 fires immediately before the pipeline.
        # disable the cached-report fast-path by removing report.json so that
        # the code reaches step 5 (early-exit after hydration).
        report_file = Path(result_dir) / "report.json"
        report_file.unlink()

        counting_model = CountingMockChatModel(answers=EDGE_ANSWERS)
        task2 = EdgeTask(
            dataset=dataset,
            model=counting_model,
            name="early_exit_hydrate_test",
        )
        config2 = make_config(
            tmp_path,
            result_dir=result_dir,
            auto_resume=True,
            detect_anomalies=False,
            record_each_stage=True,
            record_meta=True,
        )
        runner2 = TaskRunner(task2, config2)
        report2 = await runner2.arun()

        # Model must not have been called — all samples were already FINAL
        assert counting_model.call_count == 0, (
            f"Model was called {counting_model.call_count} times — "
            "early-exit path should not invoke the model."
        )
        assert report2 is not None
        assert report2["completed"] == 3

        # Cross-stage data must have been loaded by the early-exit hydrate() call.
        # Verify from disk with a fresh loader — not from runner2._contexts (memory),
        # so this fails if hydration didn't actually read from disk.
        from sieval.core.tasks.loader import TaskLoader

        loader = TaskLoader(task=task2, root_dir=Path(result_dir))
        fresh_contexts = await loader.load_initial_state()
        hydrated: set = set()
        await loader.hydrate(
            fresh_contexts,
            hydrated,
            include_stages={TaskStage.FINAL},
            record_each_stage=True,
        )
        for ctx in fresh_contexts.values():
            assert ctx.stage == TaskStage.FINAL
            assert ctx.infer_result is not None, (
                f"ctx.infer_result is None for sample {ctx.sample_id} — "
                "cross-stage hydration did not persist to disk."
            )
            assert ctx.preprocess_result is not None, (
                f"ctx.preprocess_result is None for sample {ctx.sample_id}."
            )
            assert ctx.postprocess_result is not None, (
                f"ctx.postprocess_result is None for sample {ctx.sample_id}."
            )

    @pytest.mark.anyio
    async def test_full_resume_with_failures_does_not_return_cached(self, tmp_path):
        """
        When some samples failed in the first run, auto_resume must NOT
        return the cached report — it must re-process the failed samples.
        """
        result_dir = str(tmp_path / "partial_fail_results")
        dataset = MockDataset(EDGE_SAMPLES)

        # --- First run: all samples fail ---
        fail_model = MockAlwaysFailModel()
        task1 = EdgeTask(dataset=dataset, model=fail_model, name="partial_fail_test")
        config1 = make_config(tmp_path, result_dir=result_dir, detect_anomalies=False)
        runner1 = TaskRunner(task1, config1)
        report1 = await runner1.arun()

        assert report1["failed"] == 3

        # --- Second run: auto_resume with a model that succeeds ---
        ok_model = CountingMockChatModel(answers=EDGE_ANSWERS)
        task2 = EdgeTask(dataset=dataset, model=ok_model, name="partial_fail_test")
        config2 = make_config(
            tmp_path,
            result_dir=result_dir,
            auto_resume=True,
            detect_anomalies=False,
        )
        runner2 = TaskRunner(task2, config2)
        report2 = await runner2.arun()

        # The model must have been called for the failed samples
        assert ok_model.call_count > 0, (
            "Model should have been called to retry failed samples on resume."
        )
        assert report2["completed"] == 3
        assert report2["failed"] == 0

    @pytest.mark.anyio
    async def test_anomaly_report_includes_pre_resume_samples(self, tmp_path):
        """
        When resuming with detect_anomalies=True, the final anomalies.json
        must include anomalies detected in samples that completed before
        the resume, not just anomalies from newly completed samples.

        This test catches the bug where _anomaly_results was not rebuilt
        during resume initialization, causing pre-resume anomalies to be
        lost from the final report.
        """
        result_dir = str(tmp_path / "anomaly_resume_results")
        dataset = MockDataset(EDGE_SAMPLES)

        # --- Mock model that returns truncated outputs ---
        class MockTruncatedModel(MockChatModel):
            async def _agenerate_impl(self, prompt, **kwargs):
                result = await super()._agenerate_impl(prompt, **kwargs)
                # Return truncated finish_reason
                return ModelOutput(
                    model=result.model,
                    texts=result.texts,
                    finish_reasons=["length"] * len(result.texts),
                    usage=result.usage,
                    request_params=result.request_params,
                )

        # --- First run: complete 2 samples with truncated outputs ---
        class PartialFailTruncatedModel(MockTruncatedModel):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._call_count = 0

            async def _agenerate_impl(self, prompt, **kwargs):
                self._call_count += 1
                # Fail on third sample (E3)
                if self._call_count >= 3:
                    raise TimeoutError("Simulated failure on E3")
                return await super()._agenerate_impl(prompt, **kwargs)

        model1 = PartialFailTruncatedModel(answers=EDGE_ANSWERS)
        task1 = EdgeTask(dataset=dataset, model=model1, name="anomaly_resume_test")
        config1 = make_config(tmp_path, result_dir=result_dir, detect_anomalies=True)
        runner1 = TaskRunner(task1, config1)
        report1 = await runner1.arun()

        # First run: 2 completed (with truncated outputs), 1 failed
        assert report1["completed"] == 2
        assert report1["failed"] == 1

        # Check anomalies.json after first run
        anomaly_file = Path(result_dir) / "anomalies.json"
        assert anomaly_file.exists()

        with open(anomaly_file) as f:
            anomalies1 = json.load(f)

        # Should have 2 samples with truncated_output anomaly
        assert anomalies1["summary"]["anomaly_samples"] == 2
        assert "truncated_output" in anomalies1["summary"]["anomaly_sample_details"]
        assert anomalies1["summary"]["anomaly_sample_details"]["truncated_output"] == 2

        # --- Second run: resume and complete the failed sample ---
        model2 = MockTruncatedModel(answers=EDGE_ANSWERS)
        task2 = EdgeTask(dataset=dataset, model=model2, name="anomaly_resume_test")
        config2 = make_config(
            tmp_path,
            result_dir=result_dir,
            auto_resume=True,
            detect_anomalies=True,
        )
        runner2 = TaskRunner(task2, config2)
        report2 = await runner2.arun()

        # Second run: all 3 completed
        assert report2["completed"] == 3
        assert report2["failed"] == 0

        # Check anomalies.json after resume
        with open(anomaly_file) as f:
            anomalies2 = json.load(f)

        # FIXED: Should have 3 samples with truncated_output anomaly
        # (2 from first run + 1 from resume), not just 1
        assert anomalies2["summary"]["anomaly_samples"] == 3, (
            f"Expected 3 anomaly samples after resume, got "
            f"{anomalies2['summary']['anomaly_samples']}. "
            "Pre-resume anomalies were lost from the final report."
        )
        assert anomalies2["summary"]["anomaly_sample_details"]["truncated_output"] == 3

        # Verify all 3 sample IDs are in the report
        assert len(anomalies2["samples"]) == 3
        for sid in ["0", "1", "2"]:
            assert sid in anomalies2["samples"], (
                f"Sample {sid} missing from anomalies.json after resume"
            )


# ===================================================================
# TestRunnerProgress
# ===================================================================
class TestRunnerProgress:
    @pytest.mark.anyio
    async def test_progress_file_created_when_dump_progress_true(self, tmp_path):
        """
        When dump_progress=True, the runner must write a 'progress.json'
        inside the result directory during (or after) the run.

        This exercises TaskProgress._dump_state (progress.py)
        via the init_state(force=True) call and the final close() call.
        """
        result_dir = str(tmp_path / "progress_test_results")
        dataset = MockDataset(EDGE_SAMPLES)
        model = MockChatModel(answers=EDGE_ANSWERS)

        task = EdgeTask(dataset=dataset, model=model, name="progress_dump_test")
        config = make_config(
            tmp_path,
            result_dir=result_dir,
            dump_progress=True,
            # Use a very short dump interval so the file is written during the run
            progress_dump_interval=0.0,
        )
        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report is not None
        assert report["completed"] == 3

        # progress.json must exist in the result directory
        progress_file = Path(result_dir) / "progress.json"
        assert progress_file.exists(), (
            f"Expected progress.json at {progress_file} but it does not exist. "
            "dump_progress=True should always write this file."
        )

        # Basic structural validation of the progress file
        with open(progress_file) as f:
            state = json.load(f)

        assert "total" in state, "progress.json missing 'total' key"
        assert "completed" in state, "progress.json missing 'completed' key"
        assert "percent" in state, "progress.json missing 'percent' key"
        assert state["total"] == 3, (
            f"Expected total=3 in progress.json, got {state['total']}"
        )
        assert state["finished"] is True, (
            "Expected finished=True in final progress.json"
        )

    @pytest.mark.anyio
    async def test_progress_file_not_created_when_dump_progress_false(self, tmp_path):
        """
        When dump_progress=False (the default in make_config), no
        progress.json should be written.
        """
        result_dir = str(tmp_path / "no_progress_results")
        dataset = MockDataset(EDGE_SAMPLES)
        model = MockChatModel(answers=EDGE_ANSWERS)

        task = EdgeTask(dataset=dataset, model=model, name="no_progress_dump_test")
        # make_config defaults dump_progress=False
        config = make_config(tmp_path, result_dir=result_dir, dump_progress=False)
        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report is not None

        progress_file = Path(result_dir) / "progress.json"
        assert not progress_file.exists(), (
            "progress.json should NOT exist when dump_progress=False."
        )

    @pytest.mark.anyio
    async def test_progress_file_contains_correct_percent(self, tmp_path):
        """
        After a complete run with dump_progress=True, the progress.json
        must report percent == 100.0.
        """
        result_dir = str(tmp_path / "progress_pct_results")
        dataset = MockDataset(EDGE_SAMPLES)
        model = MockChatModel(answers=EDGE_ANSWERS)

        task = EdgeTask(dataset=dataset, model=model, name="progress_pct_test")
        config = make_config(
            tmp_path,
            result_dir=result_dir,
            dump_progress=True,
            progress_dump_interval=0.0,
        )
        runner = TaskRunner(task, config)
        await runner.arun()

        progress_file = Path(result_dir) / "progress.json"
        assert progress_file.exists()

        with open(progress_file) as f:
            state = json.load(f)

        assert state["percent"] == 100.0, (
            f"Expected percent=100.0 after full run, got {state['percent']}"
        )
        assert state["completed"] == state["total"]


# ===================================================================
# TestProfilePersistence
# ===================================================================
class TestProfilePersistence:
    """profile.json is written when profiling is enabled."""

    @pytest.mark.anyio
    async def test_profile_json_written_with_usage_enabled(self, tmp_path):
        cfg = make_config(tmp_path, profile_usage=True)
        task = EdgeTask(
            name="profile_test",
            dataset=MockDataset(EDGE_SAMPLES),
            model=MockChatModel(EDGE_ANSWERS),
        )
        runner = TaskRunner(task=task, config=cfg)
        await runner.arun()

        profile_path = runner.root_dir / "profile.json"
        assert profile_path.exists()

        import orjson

        data = orjson.loads(profile_path.read_bytes())
        assert data["meta"]["task_name"] == "profile_test"
        assert data["meta"]["config"]["profile_usage"] is True
        assert "token_usage" in data

    @pytest.mark.anyio
    async def test_profile_json_not_written_when_all_disabled(self, tmp_path):
        cfg = make_config(
            tmp_path,
            profile_io=False,
            profile_stages=False,
            profile_usage=False,
        )
        task = EdgeTask(
            name="no_profile_test",
            dataset=MockDataset(EDGE_SAMPLES),
            model=MockChatModel(EDGE_ANSWERS),
        )
        runner = TaskRunner(task=task, config=cfg)
        await runner.arun()

        assert not (runner.root_dir / "profile.json").exists()
