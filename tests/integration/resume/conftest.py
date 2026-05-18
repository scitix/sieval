"""
Shared task definitions and helpers for tests/integration/resume/.

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

from sieval.core.tasks.task import Task
from tests.conftest import MockCountingChatModel

# ===================================================================
# Shared Samples
# ===================================================================
RESUME_SAMPLES = [
    {"question": "Q1", "answer": "A1"},
    {"question": "Q2", "answer": "A2"},
    {"question": "Q3", "answer": "A3"},
]

PARTIAL_SAMPLES = RESUME_SAMPLES

CROSS_STAGE_SAMPLES = [
    {"question": "Q1", "answer": "A1"},
    {"question": "Q2", "answer": "A2"},
]

ITER_SAMPLES = [
    {"question": "I1", "answer": "A1"},
    {"question": "I2", "answer": "A2"},
]

ITER_LIMIT_SAMPLES = [
    {"question": "L1", "answer": "A1"},
    {"question": "L2", "answer": "A2"},
]

HETERO_SAMPLES = [
    {"question": "H1", "answer": "A1"},
    {"question": "H2", "answer": "A2"},
    {"question": "H3", "answer": "A3"},
]


# ===================================================================
# Shared Task Implementations
# ===================================================================
class ResumeTask(Task):
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
        return {
            "accuracy": correct / total if total else 0.0,
            "total": total,
            "completed": len(finals),
            "failed": len(fails),
        }


class CountingMockChatModel(MockCountingChatModel):
    pass


class CrossStageAccessTask(Task):
    model_type = "chat"

    def __init__(self, dataset, model, name=None, fail_feedback_first_run=False):
        super().__init__(dataset=dataset, model=model, name=name)
        self._fail_feedback_first_run = fail_feedback_first_run
        self._feedback_call_count = 0

    async def preprocess(self, raw, ctx):
        return raw["question"]

    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

    async def postprocess(self, inf, ctx):
        return inf.texts[0].strip()

    async def feedback(self, post, ctx):
        self._feedback_call_count += 1
        assert ctx.infer_result is not None, "infer_result should be accessible"
        assert ctx.preprocess_result is not None, (
            "preprocess_result should be accessible"
        )

        if self._fail_feedback_first_run and self._feedback_call_count <= len(
            CROSS_STAGE_SAMPLES
        ):
            raise RuntimeError("Simulated feedback failure on first run")

        correct = post == ctx.raw_sample["answer"]
        return True, {
            "correct": correct,
            "had_infer": ctx.infer_result is not None,
            "had_preprocess": ctx.preprocess_result is not None,
        }

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
            "all_had_infer": all(
                f.feedback_result.get("had_infer") for f in finals if f.feedback_result
            ),
            "all_had_preprocess": all(
                f.feedback_result.get("had_preprocess")
                for f in finals
                if f.feedback_result
            ),
        }


class IterativeTask(Task):
    model_type = "chat"

    async def preprocess(self, raw, ctx):
        return raw["question"]

    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

    async def postprocess(self, inf, ctx):
        return inf.texts[0].strip()

    async def feedback(self, post, ctx):
        finalize = ctx.iteration >= 1
        return finalize, {"answer": post, "iteration": ctx.iteration}

    async def report(self, finals, fails):
        return {
            "completed": len(finals),
            "failed": len(fails),
            "iterations": [f.iteration for f in finals],
        }


class NeverFinalizeTask(Task):
    model_type = "chat"

    async def preprocess(self, raw, ctx):
        return raw["question"]

    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

    async def postprocess(self, inf, ctx):
        return inf.texts[0].strip()

    async def feedback(self, post, ctx):
        return False, {"answer": post}

    async def report(self, finals, fails):
        return {"completed": len(finals), "failed": len(fails)}


class HeterogeneousIterTask(Task):
    model_type = "chat"
    _finalize_at = {"H1": 0, "H2": 1, "H3": 2}

    async def preprocess(self, raw, ctx):
        return raw["question"]

    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

    async def postprocess(self, inf, ctx):
        return inf.texts[0].strip()

    async def feedback(self, post, ctx):
        q = ctx.raw_sample["question"]
        target_iter = self._finalize_at.get(q, 0)
        finalize = ctx.iteration >= target_iter
        correct = post == ctx.raw_sample["answer"]
        return finalize, {"correct": correct, "iteration": ctx.iteration}

    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        correct = sum(
            1 for f in finals if f.feedback_result and f.feedback_result["correct"]
        )
        return {
            "accuracy": correct / total if total else 0.0,
            "completed": len(finals),
            "failed": len(fails),
            "iterations": sorted(f.iteration for f in finals),
        }
