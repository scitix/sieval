"""
Pattern: LLM-as-Judge evaluation.

Covers "LLM-as-Judge" common pattern.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import pytest

from sieval.core.runners.runner import TaskRunner
from sieval.core.tasks import TaskStageOutput
from sieval.core.tasks.consts import TaskStage
from sieval.core.tasks.loader import TaskLoader
from sieval.core.tasks.task import Task
from sieval.core.utils.meta import build_stage_meta
from tests.conftest import MockChatModel, MockDataset, MockJudgeModel, make_config

# ===================================================================
# Samples
# ===================================================================
LLM_JUDGE_SAMPLES = [
    {"question": "Capital of France?", "gold": "Paris"},
    {"question": "3+4?", "gold": "7"},
]


class MockLLMJudgeTask(Task):
    """Task where feedback uses a judge model to evaluate correctness."""

    model_type = "chat"

    def __init__(self, dataset, model, judge_model, name=None):
        super().__init__(dataset=dataset, model=model, name=name)
        self._judge_model = judge_model

    async def preprocess(self, raw, ctx):
        return raw["question"]

    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

    async def postprocess(self, inf, ctx):
        return inf.texts[0].strip()

    async def feedback(self, post, ctx):
        judge_prompt = (
            f"Is this correct?\nAnswer: {post}\nGold: {ctx.raw_sample['gold']}"
        )
        judge_output = await self._judge_model.agenerate(judge_prompt)
        correct = "yes" in judge_output.texts[0].lower()

        feedback = {"correct": correct, "judge_output": judge_output.texts[0]}
        meta = build_stage_meta(judge_output)
        return True, TaskStageOutput(value=feedback, meta=meta)

    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        correct = sum(
            1
            for f in finals
            if f.feedback_result and f.feedback_result.value["correct"]
        )
        return {"accuracy": correct / total if total else 0.0, "total": total}


class TestLLMJudge:
    @pytest.mark.anyio
    async def test_judge_correct(self, tmp_path):
        """Judge returns 'yes' → all correct."""
        dataset = MockDataset(LLM_JUDGE_SAMPLES)
        model = MockChatModel(answers={"Capital of France?": "Paris", "3+4?": "7"})
        judge = MockJudgeModel(verdict="yes")
        task = MockLLMJudgeTask(
            dataset=dataset, model=model, judge_model=judge, name="judge_correct"
        )
        config = make_config(tmp_path)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report is not None
        assert report["accuracy"] == 1.0

    @pytest.mark.anyio
    async def test_judge_incorrect(self, tmp_path):
        """Judge returns 'no' → all incorrect."""
        dataset = MockDataset(LLM_JUDGE_SAMPLES)
        model = MockChatModel(default_answer="wrong")
        judge = MockJudgeModel(verdict="no")
        task = MockLLMJudgeTask(
            dataset=dataset, model=model, judge_model=judge, name="judge_wrong"
        )
        config = make_config(tmp_path)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report is not None
        assert report["accuracy"] == 0.0

    @pytest.mark.anyio
    async def test_judge_metadata_captured(self, tmp_path):
        """Judge model metadata should be captured via TaskStageOutput.

        Loads from disk with a fresh TaskLoader to verify persistence,
        not just in-memory state.
        """
        dataset = MockDataset(LLM_JUDGE_SAMPLES)
        model = MockChatModel(answers={"Capital of France?": "Paris", "3+4?": "7"})
        judge = MockJudgeModel(verdict="yes")
        task = MockLLMJudgeTask(
            dataset=dataset, model=model, judge_model=judge, name="judge_meta"
        )
        config = make_config(tmp_path, record_meta=True, record_each_stage=True)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report is not None

        # Load from disk with a fresh loader (no in-memory state)
        loader = TaskLoader(task=task, root_dir=runner.root_dir)
        contexts = await loader.load_initial_state()
        hydrated: set = set()
        await loader.hydrate(
            contexts,
            hydrated,
            include_stages={TaskStage.FINAL},
            record_each_stage=True,
        )

        # Verify feedback_result is a TaskStageOutput with meta from disk
        for ctx in contexts.values():
            if ctx.stage == TaskStage.FINAL and ctx.feedback_result is not None:
                assert isinstance(ctx.feedback_result, TaskStageOutput)
                # stage_meta should have "feedback" entries with model_calls
                fb_meta_list = ctx.stage_meta.get("feedback", [])
                assert len(fb_meta_list) > 0
                # Check that model_calls from judge are present
                last_meta = fb_meta_list[-1]
                assert "model_calls" in last_meta
                assert len(last_meta["model_calls"]) > 0
