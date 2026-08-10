import re
from collections import defaultdict
from typing import override

from sieval.community.mmlu_pro import CHOICES, QUERY_TEMPLATE
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    ReferenceImpl,
    Task,
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
    sieval_task,
)
from sieval.core.tasks.metrics import (
    DENOMINATOR_FIELD,
    DENOMINATOR_JUDGED,
    SCORE_KEY_FIELD,
    health_metrics,
    warn_unscored_rollouts,
)
from sieval.datasets import MMLUProDatasetSample


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
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        # `float | str`: the report carries `score_key`, which names a column
        # rather than measuring one.
        dict[str, float | str],
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
        return build_prompt_record(
            [
                {
                    "role": "user",
                    "content": QUERY_TEMPLATE.format(
                        question=raw["question"], options_str=options_str.strip()
                    ),
                }
            ],
            reference=raw["answer"],
            extra={"category": raw["category"]},
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"])

    @override
    async def postprocess(self, inf, ctx):
        # Every choice the model returned, not `texts[0]`: this task has no
        # sampling budget of its own, but a model-level `n` still reaches it,
        # and a draw that was paid for should not be dropped on the floor.
        predictions = []
        for text in inf.texts:
            match = re.search(r"(?i)ANSWER\s*:\s*([A-P])", text)
            predictions.append(match.group(1) if match else None)
        return build_prediction_record(predictions)

    @override
    async def feedback(self, post, ctx):
        answer = ctx.raw_sample["answer"]
        category = ctx.raw_sample["category"]
        rollouts = [
            build_rollout_judgement(
                rollout["index"], rollout.get("prediction") == answer
            )
            for rollout in post["rollouts"]
        ]
        return True, build_judgement_record(
            answer,
            rollouts,
            extra={"category": category},
        )

    @override
    async def report(self, finals, fails):
        # The FIRST rollout's verdict, per category and overall: this benchmark
        # publishes a single-draw number, so scoring the whole draw would
        # restate it.
        warn_unscored_rollouts(finals, task="mmlu_pro_0shot_gen")
        correct_num = 0
        category_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
        for ctx in finals:
            # One `or {}` for both reads, or the guard on the first is a promise
            # the second breaks two lines later.
            judgement = ctx.feedback_result or {}
            verdicts = judgement.get("rollouts") or []
            correct = bool(verdicts) and verdicts[0]["correct"]
            category = (judgement.get("extra") or {}).get("category", "other")
            if correct:
                correct_num += 1
                category_metrics[category]["correct"] += 1
            category_metrics[category]["total"] += 1

        score = 100 * correct_num / len(finals) if finals else 0.0
        results: dict[str, float | str] = {"score": score}
        for category, metrics in category_metrics.items():
            category_score = (
                100 * metrics["correct"] / metrics["total"]
                if metrics["total"] > 0
                else 0.0
            )
            results[f"score_{category}"] = category_score
        results["fails"] = len(fails)
        results[SCORE_KEY_FIELD] = "score"
        results[DENOMINATOR_FIELD] = DENOMINATOR_JUDGED
        return results | health_metrics(finals)
