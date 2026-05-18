"""
Pattern: Pass@k evaluation with n>1 sampling.

Covers "Pass@k Evaluation" common pattern.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import pytest

from sieval.core.runners.runner import TaskRunner
from sieval.core.tasks.task import Task
from tests.conftest import MockChatModel, MockDataset, make_config

# ===================================================================
# Samples
# ===================================================================
PASS_AT_K_SAMPLES = [
    {"question": "What is 1+1?", "answer": "2"},
    {"question": "What is 2+3?", "answer": "5"},
]


class MockPassAtKTask(Task):
    """Task that evaluates n>1 samples per prompt (pass@k)."""

    model_type = "chat"

    def __init__(self, dataset, model, name=None, k=1, n=1):
        super().__init__(dataset=dataset, model=model, name=name)
        self._k = k
        self._n = n

    async def preprocess(self, raw, ctx):
        return raw["question"]

    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre, n=self._n)

    async def postprocess(self, inf, ctx):
        return inf.texts  # Return all n samples

    async def feedback(self, post, ctx):
        feedbacks = [{"correct": p == ctx.raw_sample["answer"]} for p in post]
        return True, feedbacks  # Always finalize (no retry)

    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        if total == 0:
            return {"score": 0.0}

        pass_at_1_total = 0.0
        pass_at_k_total = 0.0
        for f in finals:
            feedbacks = f.feedback_result
            n_samples = len(feedbacks)
            correct_num = sum(1 for fb in feedbacks if fb["correct"])
            pass_at_1_total += self._pass_at_k(n_samples, correct_num, 1)
            if self._k > 1:
                pass_at_k_total += self._pass_at_k(n_samples, correct_num, self._k)

        pass_at_1 = pass_at_1_total * 100 / total
        metrics = {"score": pass_at_1, "pass@1": pass_at_1}
        if self._k > 1:
            metrics[f"pass@{self._k}"] = pass_at_k_total * 100 / total
        return metrics

    def _pass_at_k(self, n: int, c: int, k: int) -> float:
        if n < k:
            return 0.0
        if c == 0:
            return 0.0
        prob_all_wrong = 1.0
        for i in range(k):
            prob_all_wrong *= (n - c - i) / (n - i)
        return 1.0 - prob_all_wrong


class TestPassAtK:
    @pytest.mark.anyio
    async def test_pass_at_1_all_correct(self, tmp_path):
        """n=1, all answers correct → 100%."""
        dataset = MockDataset(PASS_AT_K_SAMPLES)
        model = MockChatModel(answers={"What is 1+1?": "2", "What is 2+3?": "5"})
        task = MockPassAtKTask(dataset=dataset, model=model, name="pass_1", k=1, n=1)
        config = make_config(tmp_path)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report is not None
        assert report["score"] == 100.0

    @pytest.mark.anyio
    async def test_pass_at_k_all_correct(self, tmp_path):
        """n=3, all 3 answers correct → pass@1=100%, pass@2=100%."""
        dataset = MockDataset(PASS_AT_K_SAMPLES)
        model = MockChatModel(
            answers={"What is 1+1?": ["2", "2", "2"], "What is 2+3?": ["5", "5", "5"]}
        )
        task = MockPassAtKTask(
            dataset=dataset, model=model, name="pass_k_all", k=2, n=3
        )
        config = make_config(tmp_path)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report is not None
        assert report["pass@1"] == 100.0
        assert report["pass@2"] == 100.0

    @pytest.mark.anyio
    async def test_pass_at_k_partial(self, tmp_path):
        """n=3, 1 of 3 correct → pass@1 < 100% but > 0%."""
        dataset = MockDataset(PASS_AT_K_SAMPLES)
        model = MockChatModel(
            answers={
                "What is 1+1?": ["2", "wrong", "wrong"],
                "What is 2+3?": ["wrong", "5", "wrong"],
            }
        )
        task = MockPassAtKTask(
            dataset=dataset, model=model, name="pass_k_partial", k=2, n=3
        )
        config = make_config(tmp_path)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report is not None
        assert 0 < report["pass@1"] < 100.0
        # pass@2 should be higher than pass@1 with partial correct
        assert report["pass@2"] > report["pass@1"]

    @pytest.mark.anyio
    async def test_pass_at_k_none_correct(self, tmp_path):
        """n=3, 0 correct → pass@1=0%, pass@2=0%."""
        dataset = MockDataset(PASS_AT_K_SAMPLES)
        model = MockChatModel(default_answer="wrong")
        task = MockPassAtKTask(
            dataset=dataset, model=model, name="pass_k_zero", k=2, n=3
        )
        config = make_config(tmp_path)

        runner = TaskRunner(task, config)
        report = await runner.arun()

        assert report is not None
        assert report["pass@1"] == 0.0
        assert report["pass@2"] == 0.0
