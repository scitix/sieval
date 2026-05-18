"""
Pattern: MultiTaskRunner — parallel task execution.

Covers "MultiTaskRunner" and "Concurrency Control".

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import pytest

from sieval.core.runners.multi_runner import MultiTaskRunner
from sieval.core.tasks.task import Task
from tests.conftest import MockChatModel, MockDataset, make_config

# ===================================================================
# Samples
# ===================================================================
SAMPLES_A = [
    {"question": "A1?", "answer": "1"},
    {"question": "A2?", "answer": "2"},
]
SAMPLES_B = [
    {"question": "B1?", "answer": "X"},
    {"question": "B2?", "answer": "Y"},
]
SAMPLES_C = [
    {"question": "C1?", "answer": "P"},
    {"question": "C2?", "answer": "Q"},
]


class SimpleTask(Task):
    """Minimal task for MultiTaskRunner tests."""

    model_type = "chat"

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
        return {"accuracy": correct / total if total else 0.0, "total": total}


class TestMultiTaskRunner:
    @pytest.mark.anyio
    async def test_two_tasks_parallel(self, tmp_path):
        """Two tasks run in parallel, each produces correct results."""
        dataset_a = MockDataset(SAMPLES_A)
        dataset_b = MockDataset(SAMPLES_B)
        model = MockChatModel(answers={"A1?": "1", "A2?": "2", "B1?": "X", "B2?": "Y"})

        task_a = SimpleTask(dataset=dataset_a, model=model, name="task_a")
        task_b = SimpleTask(dataset=dataset_b, model=model, name="task_b")

        multi = MultiTaskRunner(result_dir=str(tmp_path / "multi_results"))
        cfg = make_config(tmp_path, result_dir=None)
        multi.add_task(task_a, config=cfg)
        multi.add_task(task_b, config=cfg)

        results = await multi.arun()

        assert "task_a" in results
        assert "task_b" in results
        assert results["task_a"]["accuracy"] == 1.0
        assert results["task_b"]["accuracy"] == 1.0

    @pytest.mark.anyio
    async def test_shared_concurrency_limit(self, tmp_path):
        """Tasks share a global concurrency limit without deadlock."""
        dataset_a = MockDataset(SAMPLES_A)
        dataset_b = MockDataset(SAMPLES_B)
        model = MockChatModel(answers={"A1?": "1", "A2?": "2", "B1?": "X", "B2?": "Y"})

        task_a = SimpleTask(dataset=dataset_a, model=model, name="conc_a")
        task_b = SimpleTask(dataset=dataset_b, model=model, name="conc_b")

        multi = MultiTaskRunner(
            result_dir=str(tmp_path / "multi_conc"),
            concurrency_limit=2,  # Very tight limit
        )
        cfg = make_config(tmp_path, result_dir=None)
        multi.add_task(task_a, config=cfg)
        multi.add_task(task_b, config=cfg)

        results = await multi.arun()

        assert results["conc_a"]["accuracy"] == 1.0
        assert results["conc_b"]["accuracy"] == 1.0

    @pytest.mark.anyio
    async def test_one_task_fails_other_succeeds(self, tmp_path):
        """If one task has all failing samples, the other should still succeed."""
        dataset_a = MockDataset(SAMPLES_A)
        dataset_b = MockDataset(SAMPLES_B)

        good_model = MockChatModel(answers={"B1?": "X", "B2?": "Y"})

        class FailingModel(MockChatModel):
            async def _agenerate_impl(self, prompt, **kwargs):
                raise RuntimeError("Task A always fails")

        bad_model = FailingModel()

        task_a = SimpleTask(dataset=dataset_a, model=bad_model, name="fail_task")
        task_b = SimpleTask(dataset=dataset_b, model=good_model, name="good_task")

        multi = MultiTaskRunner(result_dir=str(tmp_path / "multi_fail"))
        cfg = make_config(tmp_path, result_dir=None)
        multi.add_task(task_a, config=cfg)
        multi.add_task(task_b, config=cfg)

        results = await multi.arun()

        # task_a should have 0% accuracy (all failed)
        assert results["fail_task"]["accuracy"] == 0.0
        # task_b should still succeed
        assert results["good_task"]["accuracy"] == 1.0

    def test_add_task_duplicate_name_raises(self, tmp_path):
        """Adding two tasks with the same name should raise ValueError."""
        dataset = MockDataset(SAMPLES_A)
        model = MockChatModel(answers={"A1?": "1", "A2?": "2"})

        task1 = SimpleTask(dataset=dataset, model=model, name="same_name")
        task2 = SimpleTask(dataset=dataset, model=model, name="same_name")

        multi = MultiTaskRunner(result_dir=str(tmp_path / "dup_name"))
        multi.add_task(task1)
        with pytest.raises(ValueError, match="already registered"):
            multi.add_task(task2)

    @pytest.mark.anyio
    async def test_three_tasks_parallel(self, tmp_path):
        """Three tasks run in parallel, each produces correct results."""
        dataset_a = MockDataset(SAMPLES_A)
        dataset_b = MockDataset(SAMPLES_B)
        dataset_c = MockDataset(SAMPLES_C)

        model = MockChatModel(
            answers={
                "A1?": "1",
                "A2?": "2",
                "B1?": "X",
                "B2?": "Y",
                "C1?": "P",
                "C2?": "Q",
            }
        )

        task_a = SimpleTask(dataset=dataset_a, model=model, name="task_a")
        task_b = SimpleTask(dataset=dataset_b, model=model, name="task_b")
        task_c = SimpleTask(dataset=dataset_c, model=model, name="task_c")

        multi = MultiTaskRunner(result_dir=str(tmp_path / "multi_three"))
        cfg = make_config(tmp_path, result_dir=None)
        multi.add_task(task_a, config=cfg)
        multi.add_task(task_b, config=cfg)
        multi.add_task(task_c, config=cfg)

        results = await multi.arun()

        assert "task_a" in results
        assert "task_b" in results
        assert "task_c" in results
        assert results["task_a"]["accuracy"] == 1.0
        assert results["task_b"]["accuracy"] == 1.0
        assert results["task_c"]["accuracy"] == 1.0

    def test_run_sync_blocking(self, tmp_path):
        """MultiTaskRunner.run() (sync) should work as a blocking wrapper."""
        dataset = MockDataset(SAMPLES_A)
        model = MockChatModel(answers={"A1?": "1", "A2?": "2"})
        task = SimpleTask(dataset=dataset, model=model, name="sync_run")
        cfg = make_config(tmp_path, result_dir=None)
        multi = MultiTaskRunner(result_dir=str(tmp_path / "sync_run"))
        multi.add_task(task, config=cfg)
        results = multi.run()
        assert "sync_run" in results

    def test_add_task_output_dir_collision_raises(self, tmp_path):
        """Two tasks that resolve to the same output dir should raise ValueError."""
        dataset = MockDataset(SAMPLES_A)
        model = MockChatModel()
        fixed_dir = str(tmp_path / "collision_dir")

        task1 = SimpleTask(dataset=dataset, model=model, name="coll_a")
        task2 = SimpleTask(dataset=dataset, model=model, name="coll_b")

        from sieval.core.runners.runner import TaskRunnerConfig

        cfg1 = TaskRunnerConfig(
            result_dir=fixed_dir,
            show_progress=False,
            detect_anomalies=False,
            profile_io=False,
            profile_stages=False,
            profile_usage=False,
            dump_progress=False,
        )
        # Use same result_dir for second task → same output dir → collision
        cfg2 = TaskRunnerConfig(
            result_dir=fixed_dir,
            show_progress=False,
            detect_anomalies=False,
            profile_io=False,
            profile_stages=False,
            profile_usage=False,
            dump_progress=False,
        )

        multi = MultiTaskRunner()
        multi.add_task(task1, config=cfg1)
        with pytest.raises(ValueError, match="collision"):
            multi.add_task(task2, config=cfg2)

    @pytest.mark.anyio
    async def test_stage_limits_via_task_action(self, tmp_path):
        """Passing concurrency_limits with TaskAction keys should work correctly."""
        from sieval.core.tasks.context import TaskAction

        dataset = MockDataset(SAMPLES_A)
        model = MockChatModel(answers={"A1?": "1", "A2?": "2"})
        task = SimpleTask(dataset=dataset, model=model, name="action_limits")
        cfg = make_config(tmp_path, result_dir=None)

        multi = MultiTaskRunner(
            result_dir=str(tmp_path / "action_limits"),
            concurrency_limit=10,
            concurrency_limits={TaskAction.INFER: 5},
        )
        multi.add_task(task, config=cfg)
        results = await multi.arun()
        assert "action_limits" in results
