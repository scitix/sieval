"""HMMT February 2026 zero-shot generative task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from typing import TypedDict, override

from loguru import logger
from openai.types.chat import ChatCompletionUserMessageParam

from sieval.community.matharena import HMMT_INSTRUCTION, build_prompt, extract_answer
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import HMMTFeb2026DatasetSample


class Feedback(TypedDict):
    correct: bool
    answer: str


@sieval_task(
    name="hmmt_feb_2026_0shot_gen",
    display_name="HMMT Feb 2026 (0-shot, generative)",
    description=(
        "HMMT February 2026 — Harvard-MIT Mathematics Tournament, 33 problems."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended"),
    deps_group="math",
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="matharena",
        url="https://github.com/eth-sri/matharena/blob/a11194deff8c67a232974a383795e8a2776b4c6f/configs/competitions/hmmt/hmmt_feb_2026.yaml",
        notes=(
            "MathArena-aligned: boxed prompt, last-boxed extraction; "
            "equivalence via math-verify."
        ),
    ),
)
class HMMTFeb2026ZeroShotGenTask(
    Task[
        HMMTFeb2026DatasetSample,
        list[ChatCompletionUserMessageParam],
        ModelOutput,
        list[str | None],
        list[Feedback],
        dict[str, float],
    ],
):
    def __init__(self, dataset, model, name: str | None = None, k: int = 1, n: int = 1):
        super().__init__(dataset=dataset, model=model, name=name)
        self._k = k
        self._n = n

    @override
    async def preprocess(self, raw, ctx):
        return [
            {
                "role": "user",
                "content": build_prompt(HMMT_INSTRUCTION, raw["question"]),
            },
        ]

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre, n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        # MathArena-aligned: last \boxed{}; non-strict -> fall back to last integer.
        return [extract_answer(choice, strict_parsing=False) for choice in inf.texts]

    @override
    async def feedback(self, post, ctx):
        from math_verify import parse, verify

        feedbacks: list[Feedback] = []
        ground_truth = ctx.raw_sample["answer"]
        for pred in post:
            if pred is None:
                feedbacks.append({"correct": False, "answer": ground_truth})
                continue
            pred_with_env = f"${pred}$"
            ref_with_env = f"${ground_truth}$"
            try:
                parsed_pred = parse(pred_with_env)
                parsed_ref = parse(ref_with_env)
                # math_verify.verify expects the gold answer as the first arg.
                correct = verify(parsed_ref, parsed_pred)
            except Exception as e:
                logger.warning("Feedback failed for sample {}: {}", ctx.sample_id, e)
                correct = False
            feedbacks.append({"correct": correct, "answer": ground_truth})
        return True, feedbacks

    @override
    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        if total == 0:
            return {"score": 0.0, "fails": len(fails)}

        pass_at_1_total = 0.0
        pass_at_k_total = 0.0
        for f in finals:
            feedbacks = f.feedback_result
            n_samples = len(feedbacks)
            correct_num = sum(1 for f in feedbacks if f["correct"])
            pass_at_1_total += self._pass_at_k(n_samples, correct_num, 1)
            if self._k > 1:
                pass_at_k_total += self._pass_at_k(n_samples, correct_num, self._k)

        pass_at_1 = pass_at_1_total * 100 / total
        metrics = {"score": pass_at_1, "fails": len(fails), "pass@1": pass_at_1}
        if self._k > 1:
            metrics[f"pass@{self._k}"] = pass_at_k_total * 100 / total
        return metrics

    def _pass_at_k(self, n: int, c: int, k: int) -> float:
        if n < k:
            return 0.0
        if c == 0:
            return 0.0
        # Formula: 1 - product_{i=0}^{k-1} (n - c - i) / (n - i)
        # This calculates the probability that all k samples are wrong
        prob_all_wrong = 1.0
        for i in range(k):
            prob_all_wrong *= (n - c - i) / (n - i)
        return 1.0 - prob_all_wrong
