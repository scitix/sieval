"""
Pattern: LLM-as-Judge evaluation.

Covers "LLM-as-Judge" common pattern.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import pytest

from sieval.core.runners.runner import TaskRunner
from sieval.core.tasks import (
    TaskStageOutput,
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
)
from sieval.core.tasks.consts import TaskStage
from sieval.core.tasks.loader import TaskLoader
from sieval.core.tasks.task import Task
from sieval.core.utils.meta import build_stage_meta
from sieval.core.utils.serialization import obj_to_dict
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


class ProtocolJudgeTask(Task):
    """LLM-judged task returning a BARE JudgementRecord, per the protocol.

    The stage-output protocol forbids wrapping a record in TaskStageOutput, which
    is how MockLLMJudgeTask above hands the judge's ModelOutput to the runner. A
    protocol task has no such channel, so the grader's output is persisted inside
    the record instead -- and the runner has to read it back from there.
    """

    model_type = "chat"

    def __init__(self, dataset, model, judge_model, name=None):
        super().__init__(dataset=dataset, model=model, name=name)
        self._judge_model = judge_model

    async def preprocess(self, raw, ctx):
        return build_prompt_record(raw["question"], reference=raw["gold"])

    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"])

    async def postprocess(self, inf, ctx):
        return build_prediction_record([inf.texts[0].strip() or None])

    async def feedback(self, post, ctx):
        gold = ctx.raw_sample["gold"]
        rollouts = []
        for rollout in post["rollouts"]:
            out = await self._judge_model.agenerate(
                f"Is this correct?\nAnswer: {rollout['prediction']}\nGold: {gold}"
            )
            rollouts.append(
                build_rollout_judgement(
                    rollout["index"],
                    "yes" in out.texts[0].lower(),
                    extra={"grader_output": obj_to_dict(out, add_type=False)},
                )
            )
        return True, build_judgement_record(gold, rollouts)

    async def report(self, finals, fails):
        correct = sum(f.feedback_result["n_correct"] for f in finals)
        return {"accuracy": correct / len(finals) if finals else 0.0}


class TestProtocolJudgeProfiling:
    @pytest.mark.anyio
    async def test_grader_spend_reaches_the_profiler(self, tmp_path):
        """Grader tokens land in the FEEDBACK stage's usage, not just on disk.

        Before this was routed, the profiler read only task-supplied
        `TaskStageMeta["model_calls"]`, which a bare record cannot carry -- so a
        judge's tokens were persisted in the record and missing from profile.json.
        """
        dataset = MockDataset(LLM_JUDGE_SAMPLES)
        model = MockChatModel(answers={"Capital of France?": "Paris", "3+4?": "7"})
        judge = MockJudgeModel(verdict="yes")
        task = ProtocolJudgeTask(
            dataset=dataset, model=model, judge_model=judge, name="protocol_judge"
        )
        runner = TaskRunner(task, make_config(tmp_path, profile_usage=True))
        report = await runner.arun()
        assert report["accuracy"] == 1.0

        usage = runner._profiler.to_dict()["token_usage"]
        feedback = usage[TaskStage.FEEDBACK.value]
        # MockJudgeModel reports 20 in / 1 out per call, one call per sample.
        assert feedback["input"]["total"] == 20 * len(LLM_JUDGE_SAMPLES)
        assert feedback["output"]["total"] == len(LLM_JUDGE_SAMPLES)
        # The candidate model's own spend still belongs to `infer`, not feedback.
        assert usage[TaskStage.INFERRED.value]["input"]["total"] > 0

    @pytest.mark.anyio
    async def test_the_recorded_call_is_the_grader_not_the_candidate(self, tmp_path):
        """The feedback stage must attribute its call to the judge model."""
        dataset = MockDataset(LLM_JUDGE_SAMPLES)
        model = MockChatModel(answers={"Capital of France?": "Paris", "3+4?": "7"})
        judge = MockJudgeModel(verdict="yes")
        task = ProtocolJudgeTask(
            dataset=dataset, model=model, judge_model=judge, name="protocol_judge_meta"
        )
        runner = TaskRunner(task, make_config(tmp_path))
        await runner.arun()

        ctx = next(iter(runner._contexts.values()))
        calls = ctx.stage_meta[TaskStage.FEEDBACK.value][-1]["model_calls"]
        assert [c["model"]["model"] for c in calls] == ["mock-judge"]
        assert calls[0]["usage"]["input_tokens"] == 20
