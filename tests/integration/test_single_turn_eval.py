"""
Pattern: Single-turn evaluation with multi-reference answers.

Covers "Multi-Reference Evaluation" common pattern.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import pytest

from sieval.core.runners.runner import TaskRunner
from sieval.core.tasks.task import Task
from tests.conftest import MockChatModel, MockDataset, make_config

# ===================================================================
# Multi-Reference Samples
# ===================================================================
MULTI_REF_SAMPLES = [
    {"question": "Capital of France?", "golds": ["Paris", "paris"]},
    {"question": "3+4?", "golds": ["7", "seven"]},
    {"question": "Color of sky?", "golds": ["blue", "Blue", "azure"]},
]


class MockMultiRefTask(Task):
    """Task where feedback checks against multiple acceptable answers."""

    model_type = "chat"

    async def preprocess(self, raw, ctx):
        return raw["question"]

    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

    async def postprocess(self, inf, ctx):
        return inf.texts[0].strip()

    async def feedback(self, post, ctx):
        golds = ctx.raw_sample["golds"]
        correct = any(post == gold for gold in golds)
        return True, {"correct": correct}

    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        correct = sum(
            1 for f in finals if f.feedback_result and f.feedback_result["correct"]
        )
        return {"accuracy": correct / total if total else 0.0, "total": total}


class TestMultiReferenceEvaluation:
    @pytest.mark.anyio
    async def test_multi_ref_correct(self, tmp_path):
        """Model answers match one of the acceptable gold answers."""
        dataset = MockDataset(MULTI_REF_SAMPLES)
        model = MockChatModel(
            answers={
                "Capital of France?": "Paris",
                "3+4?": "7",
                "Color of sky?": "blue",
            }
        )
        task = MockMultiRefTask(dataset=dataset, model=model, name="multi_ref_correct")
        config = make_config(tmp_path)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report is not None
        assert report["accuracy"] == 1.0
        assert report["total"] == 3

    @pytest.mark.anyio
    async def test_multi_ref_alternate_gold(self, tmp_path):
        """Model answers match a non-first gold answer."""
        dataset = MockDataset(MULTI_REF_SAMPLES)
        model = MockChatModel(
            answers={
                "Capital of France?": "paris",  # second gold
                "3+4?": "seven",  # second gold
                "Color of sky?": "azure",  # third gold
            }
        )
        task = MockMultiRefTask(dataset=dataset, model=model, name="multi_ref_alt")
        config = make_config(tmp_path)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report is not None
        assert report["accuracy"] == 1.0

    @pytest.mark.anyio
    async def test_multi_ref_incorrect(self, tmp_path):
        """Model answers don't match any gold answer."""
        dataset = MockDataset(MULTI_REF_SAMPLES)
        model = MockChatModel(default_answer="wrong_answer")
        task = MockMultiRefTask(dataset=dataset, model=model, name="multi_ref_wrong")
        config = make_config(tmp_path)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report is not None
        assert report["accuracy"] == 0.0
        assert report["total"] == 3
