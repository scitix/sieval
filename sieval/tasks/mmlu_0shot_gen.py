import re
from collections import defaultdict
from typing import override

from sieval.community.simple_evals.common import (
    MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,
    MULTILINGUAL_ANSWER_REGEXES,
    QUERY_TEMPLATE_MULTICHOICE,
    normalize_extracted_answer,
    normalize_response,
)
from sieval.community.simple_evals.mmlu_eval import subject2category
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
from sieval.datasets import MMLUDatasetSample


@sieval_task(
    name="mmlu_0shot_gen",
    display_name="MMLU (0-shot, generative)",
    description=(
        "Massive Multitask Language Understanding — 57 academic subjects, MCQ."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "multiple-choice"),
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="simple-evals",
        url="https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/mmlu_eval.py",
        notes=(
            "0-shot generative MMLU with letter extraction; scoring aligned "
            "with simple-evals, data loaded from cais/mmlu (same 14042-item "
            "test set)."
        ),
    ),
)
class MMLUZeroShotGenTask(
    Task[
        MMLUDatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        dict[str, float],
    ]
):
    @override
    async def preprocess(self, raw, ctx):
        choices = raw["choices"]
        data = {
            "Question": raw["question"],
            "A": choices[0],
            "B": choices[1],
            "C": choices[2],
            "D": choices[3],
        }
        subject = raw.get("subject", "unknown")
        return build_prompt_record(
            [
                {"role": "user", "content": QUERY_TEMPLATE_MULTICHOICE.format(**data)},
            ],
            reference="ABCD"[raw["answer"]],
            extra={
                "subject": subject,
                "category": subject2category.get(subject, "other"),
            },
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"])

    @override
    async def postprocess(self, inf, ctx):
        response_text = normalize_response(inf.texts[0])  # n=1, only one choice
        extracted_answer = ""
        for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
            regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
            match = re.search(regex, response_text)
            if match:
                extracted_answer = normalize_extracted_answer(match.group(1))
                break
        # No regex matched -> None, so `extracted` reports the miss. The letter
        # comparison below is unaffected: neither "" nor None equals a gold letter.
        return build_prediction_record([extracted_answer or None])

    @override
    async def feedback(self, post, ctx):
        answer = "ABCD"[ctx.raw_sample["answer"]]
        subject = ctx.raw_sample.get("subject", "unknown")
        category = subject2category.get(subject, "other")
        prediction = post["rollouts"][0]["prediction"]
        return True, build_judgement_record(
            answer,
            [build_rollout_judgement(0, prediction == answer)],
            extra={"subject": subject, "category": category},
        )

    @override
    async def report(self, finals, fails):
        correct_num = 0
        category_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
        for ctx in finals:
            correct = ctx.feedback_result["rollouts"][0]["correct"]
            category = ctx.feedback_result["extra"]["category"]
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
