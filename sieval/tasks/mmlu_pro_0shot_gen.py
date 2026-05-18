import re
from collections import defaultdict
from typing import TypedDict, override

from openai.types.chat import ChatCompletionUserMessageParam

from sieval.community.mmlu_pro import CHOICES, QUERY_TEMPLATE
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import MMLUProDatasetSample


class Feedback(TypedDict):
    correct: bool
    category: str
    answer: str


@sieval_task(
    name="mmlu_pro_0shot_gen",
    display_name="MMLU-Pro (0-shot, generative)",
    description="MMLU-Pro — harder MCQ with 10 options, filtered for reasoning.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "multiple-choice"),
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="opencompass",
        url="https://github.com/open-compass/opencompass/blob/568572803ab108eb0e2ae73b770d965b7de078de/opencompass/configs/datasets/mmlu_pro/mmlu_pro_0shot_cot_gen_08c1de.py",
        notes="QUERY_TEMPLATE / CHOICES adapted from opencompass 0-shot CoT config.",
    ),
)
class MMLUProZeroShotGenTask(
    Task[
        MMLUProDatasetSample,
        list[ChatCompletionUserMessageParam],
        ModelOutput,
        str,
        Feedback,
        dict[str, float],
    ]
):
    @override
    async def preprocess(self, raw, ctx):
        options_str = ""
        for i, opt in enumerate(raw["options"]):
            if opt == "N/A":
                continue
            option = f"{CHOICES[i]}. {opt}\n"
            options_str += option
        return [
            {
                "role": "user",
                "content": QUERY_TEMPLATE.format(
                    question=raw["question"], options_str=options_str.strip()
                ),
            }
        ]

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

    @override
    async def postprocess(self, inf, ctx):
        match = re.search(
            r"(?i)ANSWER\s*:\s*([A-P])", inf.texts[0]
        )  # n=1, only one choice
        return match.group(1) if match else ""

    @override
    async def feedback(self, post, ctx):
        answer = ctx.raw_sample["answer"]
        category = ctx.raw_sample["category"]
        return True, {"correct": post == answer, "category": category, "answer": answer}

    @override
    async def report(self, finals, fails):
        correct_num = 0
        category_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
        for ctx in finals:
            correct = ctx.feedback_result["correct"]
            category = ctx.feedback_result["category"]
            if correct:
                correct_num += 1
                category_metrics[category]["correct"] += 1
            category_metrics[category]["total"] += 1

        score = 100 * correct_num / len(finals) if finals else 0.0
        results = {"score": score}
        for category, metrics in category_metrics.items():
            category_score = (
                100 * metrics["correct"] / metrics["total"]
                if metrics["total"] > 0
                else 0.0
            )
            results[f"score_{category}"] = category_score
        results["fails"] = len(fails)
        return results
