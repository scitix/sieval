"""
Integration tests for cross-stage data access after resume, iteration boundary
snapshots, max_retries enforcement, max_iterations enforcement, and heterogeneous
iteration completion.

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

import pytest

from sieval.core.runners.runner import TaskRunner
from tests.conftest import (
    MockAlwaysFailModel,
    MockChatModel,
    MockDataset,
    make_config,
)

from .conftest import (
    CROSS_STAGE_SAMPLES,
    HETERO_SAMPLES,
    ITER_LIMIT_SAMPLES,
    ITER_SAMPLES,
    CountingMockChatModel,
    CrossStageAccessTask,
    HeterogeneousIterTask,
    IterativeTask,
    NeverFinalizeTask,
    ResumeTask,
)

RETRY_SAMPLES = [
    {"question": "R1", "answer": "A1"},
    {"question": "R2", "answer": "A2"},
]


class TestResumeCrossStageAccess:
    @pytest.mark.anyio
    async def test_cross_stage_data_available_in_normal_run(self, tmp_path):
        """feedback() can access ctx.infer_result and ctx.preprocess_result."""
        dataset = MockDataset(CROSS_STAGE_SAMPLES)
        model = MockChatModel(answers={"Q1": "A1", "Q2": "A2"})
        task = CrossStageAccessTask(dataset=dataset, model=model, name="cross_normal")
        config = make_config(tmp_path, record_each_stage=True)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report["accuracy"] == 1.0
        assert report["all_had_infer"] is True
        assert report["all_had_preprocess"] is True

    @pytest.mark.anyio
    async def test_cross_stage_data_after_resume(self, tmp_path):
        """After resume with record_each_stage=True, feedback() can access
        ctx.infer_result from saved snapshots — data loaded from disk, not memory."""
        dataset = MockDataset(CROSS_STAGE_SAMPLES)
        result_dir = str(tmp_path / "cross_resume")

        fail_model = MockChatModel(answers={"Q1": "A1", "Q2": "A2"})
        task1 = CrossStageAccessTask(
            dataset=dataset,
            model=fail_model,
            name="cross_resume",
            fail_feedback_first_run=True,
        )
        config1 = make_config(tmp_path, result_dir=result_dir, record_each_stage=True)
        runner1 = TaskRunner(task1, config1)
        report1 = await runner1.arun()

        assert report1["failed"] == 2
        assert report1["completed"] == 0

        ok_model = MockChatModel(answers={"Q1": "A1", "Q2": "A2"})
        task2 = CrossStageAccessTask(
            dataset=dataset,
            model=ok_model,
            name="cross_resume",
            fail_feedback_first_run=False,
        )
        config2 = make_config(
            tmp_path, result_dir=result_dir, record_each_stage=True, auto_resume=True
        )
        runner2 = TaskRunner(task2, config2)
        report2 = await runner2.arun()

        assert report2["completed"] == 2
        assert report2["accuracy"] == 1.0
        assert report2["all_had_infer"] is True
        assert report2["all_had_preprocess"] is True

    @pytest.mark.anyio
    async def test_cross_stage_data_after_resume_record_each_stage_false(
        self, tmp_path
    ):
        """After resume with record_each_stage=False, feedback() can still access
        ctx.infer_result — loaded from the iteration-boundary snapshot, not per-stage
        shards. This exercises a different hydration code path than the True variant."""
        dataset = MockDataset(CROSS_STAGE_SAMPLES)
        result_dir = str(tmp_path / "cross_resume_no_stage")

        # First run: feedback fails so samples land in FAILED with infer data on disk
        fail_model = MockChatModel(answers={"Q1": "A1", "Q2": "A2"})
        task1 = CrossStageAccessTask(
            dataset=dataset,
            model=fail_model,
            name="cross_resume_no_stage",
            fail_feedback_first_run=True,
        )
        config1 = make_config(tmp_path, result_dir=result_dir, record_each_stage=False)
        runner1 = TaskRunner(task1, config1)
        report1 = await runner1.arun()

        assert report1["failed"] == 2
        assert report1["completed"] == 0

        # Second run: resume — feedback() must see infer_result from disk
        ok_model = MockChatModel(answers={"Q1": "A1", "Q2": "A2"})
        task2 = CrossStageAccessTask(
            dataset=dataset,
            model=ok_model,
            name="cross_resume_no_stage",
            fail_feedback_first_run=False,
        )
        config2 = make_config(
            tmp_path, result_dir=result_dir, record_each_stage=False, auto_resume=True
        )
        runner2 = TaskRunner(task2, config2)
        report2 = await runner2.arun()

        assert report2["completed"] == 2
        assert report2["accuracy"] == 1.0
        assert report2["all_had_infer"] is True, (
            "feedback() could not access ctx.infer_result after resume with "
            "record_each_stage=False — iteration-boundary hydration may be broken."
        )
        assert report2["all_had_preprocess"] is True


class TestIterationBoundarySnapshots:
    @pytest.mark.anyio
    async def test_iteration_boundary_saved_when_record_each_stage_false(
        self, tmp_path
    ):
        """record_each_stage=False + multi-iteration: boundary snapshots exist."""
        dataset = MockDataset(ITER_SAMPLES)
        model = MockChatModel(answers={"I1": "A1", "I2": "A2"})
        task = IterativeTask(dataset=dataset, model=model, name="iter_boundary")
        config = make_config(tmp_path, record_each_stage=False, max_iterations=5)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report["completed"] == 2
        assert all(it == 1 for it in report["iterations"])

        root = runner.root_dir
        stage_dirs = {
            (p.parent.parent.name, p.parent.name) for p in root.rglob("*.jsonl")
        }
        assert ("1", "initial") in stage_dirs
        assert ("1", "final") in stage_dirs

    @pytest.mark.anyio
    async def test_resume_from_iteration_boundary(self, tmp_path):
        """
        record_each_stage=False: failed samples resume from last iteration, not scratch.
        """
        dataset = MockDataset(ITER_SAMPLES)
        result_dir = str(tmp_path / "iter_resume")

        call_counts = {}

        class FailOnSecondIterModel(MockChatModel):
            async def _agenerate_impl(self, prompt, **kwargs):
                q = prompt if isinstance(prompt, str) else list(prompt)[-1]["content"]
                call_counts[q] = call_counts.get(q, 0) + 1
                if call_counts[q] == 2:
                    raise RuntimeError("Fail on iteration 1")
                return await super()._agenerate_impl(prompt, **kwargs)

        model1 = FailOnSecondIterModel(answers={"I1": "A1", "I2": "A2"})
        task1 = IterativeTask(dataset=dataset, model=model1, name="iter_resume")
        config1 = make_config(
            tmp_path, result_dir=result_dir, record_each_stage=False, max_iterations=5
        )
        runner1 = TaskRunner(task1, config1)
        report1 = await runner1.arun()
        assert report1["failed"] == 2

        model2 = MockChatModel(answers={"I1": "A1", "I2": "A2"})
        task2 = IterativeTask(dataset=dataset, model=model2, name="iter_resume")
        config2 = make_config(
            tmp_path,
            result_dir=result_dir,
            record_each_stage=False,
            max_iterations=5,
            auto_resume=True,
        )
        runner2 = TaskRunner(task2, config2)
        report2 = await runner2.arun()
        assert report2["completed"] == 2


class TestMaxRetries:
    @pytest.mark.anyio
    async def test_max_retries_marks_failed(self, tmp_path):
        """Samples that exceed max_retries are marked FAILED without calling model."""
        dataset = MockDataset(RETRY_SAMPLES)
        result_dir = str(tmp_path / "retry_results")

        task1 = ResumeTask(
            dataset=dataset, model=MockAlwaysFailModel(), name="retry_test"
        )
        config1 = make_config(tmp_path, result_dir=result_dir, max_retries=1)
        runner1 = TaskRunner(task1, config1)
        report1 = await runner1.arun()
        assert report1["failed"] == 2

        task2 = ResumeTask(
            dataset=dataset, model=MockAlwaysFailModel(), name="retry_test"
        )
        config2 = make_config(
            tmp_path, result_dir=result_dir, max_retries=1, auto_resume=True
        )
        runner2 = TaskRunner(task2, config2)
        report2 = await runner2.arun()
        assert report2["failed"] == 2

        counting_model = CountingMockChatModel(answers={"R1": "A1", "R2": "A2"})
        task3 = ResumeTask(dataset=dataset, model=counting_model, name="retry_test")
        config3 = make_config(
            tmp_path, result_dir=result_dir, max_retries=1, auto_resume=True
        )
        runner3 = TaskRunner(task3, config3)
        report3 = await runner3.arun()

        assert report3["failed"] == 2
        assert counting_model.call_count == 0

    @pytest.mark.anyio
    async def test_within_max_retries_succeeds(self, tmp_path):
        """Samples within max_retries that succeed on retry complete correctly."""
        dataset = MockDataset(RETRY_SAMPLES)
        result_dir = str(tmp_path / "retry_ok")

        task1 = ResumeTask(
            dataset=dataset, model=MockAlwaysFailModel(), name="retry_ok"
        )
        config1 = make_config(tmp_path, result_dir=result_dir, max_retries=3)
        runner1 = TaskRunner(task1, config1)
        await runner1.arun()

        ok_model = MockChatModel(answers={"R1": "A1", "R2": "A2"})
        task2 = ResumeTask(dataset=dataset, model=ok_model, name="retry_ok")
        config2 = make_config(
            tmp_path, result_dir=result_dir, max_retries=3, auto_resume=True
        )
        runner2 = TaskRunner(task2, config2)
        report2 = await runner2.arun()

        assert report2["completed"] == 2
        assert report2["failed"] == 0


class TestMaxIterations:
    @pytest.mark.anyio
    async def test_max_iterations_marks_failed(self, tmp_path):
        """Samples that reach max_iterations without finalizing are marked FAILED."""
        dataset = MockDataset(ITER_LIMIT_SAMPLES)
        model = MockChatModel(answers={"L1": "A1", "L2": "A2"})
        task = NeverFinalizeTask(dataset=dataset, model=model, name="iter_limit")
        config = make_config(tmp_path, max_iterations=3)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report["completed"] == 0
        assert report["failed"] == 2

    @pytest.mark.anyio
    async def test_max_iterations_one(self, tmp_path):
        """max_iterations=1 means no re-iteration allowed."""
        dataset = MockDataset(ITER_LIMIT_SAMPLES)
        model = MockChatModel(answers={"L1": "A1", "L2": "A2"})
        task = NeverFinalizeTask(dataset=dataset, model=model, name="iter_1")
        config = make_config(tmp_path, max_iterations=1)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report["completed"] == 0
        assert report["failed"] == 2


class TestHeterogeneousIteration:
    @pytest.mark.anyio
    async def test_samples_finalize_at_different_iterations(self, tmp_path):
        """Samples that finalize at different iterations all complete correctly."""
        dataset = MockDataset(HETERO_SAMPLES)
        model = MockChatModel(answers={"H1": "A1", "H2": "A2", "H3": "A3"})
        task = HeterogeneousIterTask(dataset=dataset, model=model, name="hetero_iter")
        config = make_config(tmp_path, max_iterations=5)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report["completed"] == 3
        assert report["failed"] == 0
        assert report["accuracy"] == 1.0
        assert report["iterations"] == [0, 1, 2]
