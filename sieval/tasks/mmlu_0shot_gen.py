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
from sieval.core.tasks.metrics import (
    DENOMINATOR_FIELD,
    DENOMINATOR_JUDGED,
    SCORE_KEY_FIELD,
    health_metrics,
    warn_unscored_rollouts,
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
    reference_kind="value",
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
        # `float | str`: the report carries `score_key`, which names a column
        # rather than measuring one.
        dict[str, float | str],
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
        # Every choice the model returned, not `texts[0]`: this task has no
        # sampling budget of its own, but a model-level `n` still reaches it,
        # and a draw that was paid for should not be dropped on the floor.
        #
        # No regex matched -> None, so `extracted` reports the miss. The letter
        # comparison below is unaffected: neither "" nor None equals a gold letter.
        return build_prediction_record(
            [self._extract(text) or None for text in inf.texts]
        )

    @staticmethod
    def _extract(text: str) -> str:
        response_text = normalize_response(text)
        for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
            regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
            match = re.search(regex, response_text)
            if match:
                return normalize_extracted_answer(match.group(1))
        return ""

    @override
    async def feedback(self, post, ctx):
        answer = "ABCD"[ctx.raw_sample["answer"]]
        subject = ctx.raw_sample.get("subject", "unknown")
        category = subject2category.get(subject, "other")
        rollouts = [
            build_rollout_judgement(
                rollout["index"], rollout.get("prediction") == answer
            )
            for rollout in post["rollouts"]
        ]
        return True, build_judgement_record(
            answer,
            rollouts,
            extra={"subject": subject, "category": category},
        )

    @override
    async def report(self, finals, fails):
        # The FIRST rollout's verdict, per category and overall: this benchmark
        # publishes a single-draw number, so scoring the whole draw would
        # restate it.
        warn_unscored_rollouts(finals, task="mmlu_0shot_gen")
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
