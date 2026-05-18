"""
Integration tests for basic resume functionality: completed run returns cached
report, record_each_stage variants, and partial-completion resume.

Pattern: Resume from partial completion
Covers: "Recovery Semantics", "Record Modes"

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

import pytest

from sieval.core.runners.runner import TaskRunner
from tests.conftest import (
    MockAlwaysFailModel,
    MockChatModel,
    MockDataset,
    MockSelectiveFailModel,
    make_config,
)

from .conftest import (
    PARTIAL_SAMPLES,
    RESUME_SAMPLES,
    CountingMockChatModel,
    ResumeTask,
)


class TestResumePartialCompletion:
    @pytest.mark.anyio
    async def test_resume_from_completed(self, tmp_path):
        """Run fully, then resume → should return cached report without model calls."""
        dataset = MockDataset(RESUME_SAMPLES)
        model = MockChatModel(answers={"Q1": "A1", "Q2": "A2", "Q3": "A3"})
        result_dir = str(tmp_path / "resume_results")

        task1 = ResumeTask(dataset=dataset, model=model, name="resume_test")
        config1 = make_config(tmp_path, result_dir=result_dir)
        runner1 = TaskRunner(task1, config1)
        report1 = await runner1.arun()
        assert report1["accuracy"] == 1.0

        counting_model = CountingMockChatModel(
            answers={"Q1": "A1", "Q2": "A2", "Q3": "A3"}
        )
        task2 = ResumeTask(dataset=dataset, model=counting_model, name="resume_test")
        config2 = make_config(tmp_path, result_dir=result_dir, auto_resume=True)
        runner2 = TaskRunner(task2, config2)
        report2 = await runner2.arun()

        assert report2 == report1
        assert counting_model.call_count == 0

    @pytest.mark.anyio
    async def test_resume_with_record_each_stage_true(self, tmp_path):
        """With record_each_stage=True, resume should find intermediate stage data."""
        dataset = MockDataset(RESUME_SAMPLES)
        model = MockChatModel(answers={"Q1": "A1", "Q2": "A2", "Q3": "A3"})
        result_dir = str(tmp_path / "resume_stage_true")

        task1 = ResumeTask(dataset=dataset, model=model, name="resume_s_true")
        config1 = make_config(tmp_path, result_dir=result_dir, record_each_stage=True)
        runner1 = TaskRunner(task1, config1)
        report1 = await runner1.arun()
        assert report1["accuracy"] == 1.0

        root = runner1.root_dir
        stage_dirs = {p.parent.name for p in root.rglob("*.jsonl")}
        assert "final" in stage_dirs
        assert len(stage_dirs) > 1

        task2 = ResumeTask(dataset=dataset, model=model, name="resume_s_true")
        config2 = make_config(
            tmp_path, result_dir=result_dir, record_each_stage=True, auto_resume=True
        )
        runner2 = TaskRunner(task2, config2)
        report2 = await runner2.arun()
        assert report2 == report1

    @pytest.mark.anyio
    async def test_resume_with_record_each_stage_false(self, tmp_path):
        """With record_each_stage=False, only final shards saved."""
        dataset = MockDataset(RESUME_SAMPLES)
        model = MockChatModel(answers={"Q1": "A1", "Q2": "A2", "Q3": "A3"})
        result_dir = str(tmp_path / "resume_stage_false")

        task1 = ResumeTask(dataset=dataset, model=model, name="resume_s_false")
        config1 = make_config(tmp_path, result_dir=result_dir, record_each_stage=False)
        runner1 = TaskRunner(task1, config1)
        report1 = await runner1.arun()
        assert report1["accuracy"] == 1.0

        root = runner1.root_dir
        stage_dirs = {p.parent.name for p in root.rglob("*.jsonl")}
        assert stage_dirs == {"final"}

        task2 = ResumeTask(dataset=dataset, model=model, name="resume_s_false")
        config2 = make_config(
            tmp_path, result_dir=result_dir, record_each_stage=False, auto_resume=True
        )
        runner2 = TaskRunner(task2, config2)
        report2 = await runner2.arun()
        assert report2 == report1


class TestResumePartialCompletionOnly:
    @pytest.mark.anyio
    async def test_resume_only_processes_failed_samples(self, tmp_path):
        """When 2/3 samples complete and 1 fails, resume should only re-process
        the failed sample — counting model must be called exactly once."""
        dataset = MockDataset(PARTIAL_SAMPLES)
        result_dir = str(tmp_path / "partial_only_results")

        fail_model = MockSelectiveFailModel(
            fail_samples={"Q3"},
            answers={"Q1": "A1", "Q2": "A2", "Q3": "A3"},
        )
        task1 = ResumeTask(dataset=dataset, model=fail_model, name="partial_only")
        config1 = make_config(tmp_path, result_dir=result_dir)
        runner1 = TaskRunner(task1, config1)
        report1 = await runner1.arun()

        assert report1["completed"] == 2
        assert report1["failed"] == 1

        counting_model = CountingMockChatModel(
            answers={"Q1": "A1", "Q2": "A2", "Q3": "A3"},
        )
        task2 = ResumeTask(dataset=dataset, model=counting_model, name="partial_only")
        config2 = make_config(tmp_path, result_dir=result_dir, auto_resume=True)
        runner2 = TaskRunner(task2, config2)
        report2 = await runner2.arun()

        assert report2["completed"] == 3
        assert report2["failed"] == 0
        assert counting_model.call_count == 1, (
            f"Expected 1 call (only for Q3), got {counting_model.call_count}. "
            "Already-completed samples are being re-processed on resume."
        )


class TestResumeFailedRetry:
    @pytest.mark.anyio
    async def test_failed_samples_in_manifest(self, tmp_path):
        """All-fail run records failures in manifest correctly."""
        dataset = MockDataset(RESUME_SAMPLES)
        model = MockAlwaysFailModel()
        task = ResumeTask(dataset=dataset, model=model, name="fail_manifest")
        config = make_config(tmp_path)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report["failed"] == 3
        assert report["completed"] == 0
