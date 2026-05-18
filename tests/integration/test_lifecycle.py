"""
Pattern: Task lifecycle (setup/shutdown).

Covers "Task Structure" — setup() and shutdown() methods.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import pytest

from sieval.core.runners.runner import TaskRunner
from sieval.core.tasks.task import Task
from tests.conftest import MockChatModel, MockDataset, make_config

# ===================================================================
# Samples
# ===================================================================
SAMPLES = [
    {"question": "Q1", "answer": "A1"},
]


class LifecycleTrackingTask(Task):
    """Task that tracks setup/shutdown calls."""

    model_type = "chat"

    def __init__(self, dataset, model, name=None):
        super().__init__(dataset=dataset, model=model, name=name)
        self.setup_called = False
        self.shutdown_called = False
        self.setup_order = 0
        self.shutdown_order = 0
        self._call_counter = 0

    async def setup(self):
        self._call_counter += 1
        self.setup_called = True
        self.setup_order = self._call_counter

    async def shutdown(self):
        self._call_counter += 1
        self.shutdown_called = True
        self.shutdown_order = self._call_counter

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
        return {"accuracy": correct / total if total else 0.0}


class LifecycleFailingTask(LifecycleTrackingTask):
    """Task where infer always fails, to verify shutdown still runs."""

    async def infer(self, pre, ctx):
        raise RuntimeError("Intentional infer failure")


class SetupFailingTask(LifecycleTrackingTask):
    """Task where setup() raises, to verify shutdown still runs."""

    async def setup(self):
        await super().setup()
        raise RuntimeError("Intentional setup failure")


class ShutdownFailingTask(LifecycleTrackingTask):
    """Task where shutdown() raises, to verify runner still returns gracefully."""

    async def shutdown(self):
        await super().shutdown()
        raise RuntimeError("Intentional shutdown failure")


class TestLifecycle:
    @pytest.mark.anyio
    async def test_setup_called_before_run(self, tmp_path):
        """setup() should be called before pipeline starts."""
        dataset = MockDataset(SAMPLES)
        model = MockChatModel(answers={"Q1": "A1"})
        task = LifecycleTrackingTask(dataset=dataset, model=model, name="lc_setup")
        config = make_config(tmp_path)

        runner = TaskRunner(task, config)
        await runner.arun()

        assert task.setup_called is True

    @pytest.mark.anyio
    async def test_shutdown_called_after_run(self, tmp_path):
        """shutdown() should be called after pipeline completes."""
        dataset = MockDataset(SAMPLES)
        model = MockChatModel(answers={"Q1": "A1"})
        task = LifecycleTrackingTask(dataset=dataset, model=model, name="lc_shutdown")
        config = make_config(tmp_path)

        runner = TaskRunner(task, config)
        await runner.arun()

        assert task.shutdown_called is True

    @pytest.mark.anyio
    async def test_setup_before_shutdown(self, tmp_path):
        """setup() should be called before shutdown()."""
        dataset = MockDataset(SAMPLES)
        model = MockChatModel(answers={"Q1": "A1"})
        task = LifecycleTrackingTask(dataset=dataset, model=model, name="lc_order")
        config = make_config(tmp_path)

        runner = TaskRunner(task, config)
        await runner.arun()

        assert task.setup_called is True
        assert task.shutdown_called is True
        assert task.setup_order < task.shutdown_order

    @pytest.mark.anyio
    async def test_shutdown_called_on_error(self, tmp_path):
        """shutdown() should be called even when the pipeline encounters errors."""
        dataset = MockDataset(SAMPLES)
        model = MockChatModel(answers={"Q1": "A1"})
        task = LifecycleFailingTask(dataset=dataset, model=model, name="lc_error")
        config = make_config(tmp_path)

        runner = TaskRunner(task, config)
        # The task should still complete (with failures), not raise
        report = await runner.arun()

        assert task.setup_called is True
        assert task.shutdown_called is True
        assert task.setup_order < task.shutdown_order
        assert report is not None
        assert report["accuracy"] == 0.0

    @pytest.mark.anyio
    async def test_setup_failure_propagates(self, tmp_path):
        """If setup() raises, the error propagates to the caller.

        Current behavior: setup() runs outside the try/finally that guards
        shutdown(), so the exception propagates and shutdown() is NOT called.
        """
        dataset = MockDataset(SAMPLES)
        model = MockChatModel(answers={"Q1": "A1"})
        task = SetupFailingTask(dataset=dataset, model=model, name="lc_setup_fail")
        config = make_config(tmp_path)

        runner = TaskRunner(task, config)
        with pytest.raises(RuntimeError, match="Intentional setup failure"):
            await runner.arun()

        assert task.setup_called is True

    @pytest.mark.anyio
    async def test_shutdown_exception_propagates(self, tmp_path):
        """If shutdown() raises, the exception propagates to the caller.

        Current behavior: shutdown() runs in the finally block without its
        own try/except wrapper, so any exception it raises will propagate.
        """
        dataset = MockDataset(SAMPLES)
        model = MockChatModel(answers={"Q1": "A1"})
        task = ShutdownFailingTask(
            dataset=dataset, model=model, name="lc_shutdown_fail"
        )
        config = make_config(tmp_path)

        runner = TaskRunner(task, config)
        with pytest.raises(RuntimeError, match="Intentional shutdown failure"):
            await runner.arun()

        assert task.setup_called is True
        assert task.shutdown_called is True
