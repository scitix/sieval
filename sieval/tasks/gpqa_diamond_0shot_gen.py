import random
import re
from typing import override

from sieval.community.simple_evals.common import (
    ANSWER_PATTERN_MULTICHOICE,
    QUERY_TEMPLATE_MULTICHOICE,
)
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
    first_rollout_correct,
    health_metrics,
    warn_unscored_rollouts,
)
from sieval.datasets import GPQADiamondDatasetSample


@sieval_task(
    name="gpqa_diamond_0shot_gen",
    display_name="GPQA-Diamond (0-shot, generative)",
    description="Graduate-level science MCQ — diamond subset, 198 questions.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "multiple-choice", "graduate-level"),
    model_type="chat",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="simple-evals",
        url="https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/gpqa_eval.py",
        notes="Permutation + seed logic aligned with simple-evals.",
    ),
)
class GPQADiamondZeroShotGenTask(
    Task[
        GPQADiamondDatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        # `float | str`: the report carries `score_key`, which names a column
        # rather than measuring one.
        dict[str, float | str],
    ]
):
    """GPQA-Diamond 0-shot chat generation with shuffled answer choices.

    Aligns with OpenAI simple-evals' ``GPQAEval``: a single sequential
    ``Random(seed)`` pre-computes one permutation per virtual sample, and
    ``n_repeats`` controls how many times each question is evaluated with
    different answer orderings (default 4, same as simple-evals).
    """

    def __init__(
        self, dataset, model, name: str | None = None, seed: int = 0, n_repeats: int = 4
    ):
        expanded = dataset.repeat(n_repeats) if n_repeats > 1 else dataset
        super().__init__(dataset=expanded, model=model, name=name)
        # Pre-compute all permutations with a single sequential RNG,
        # replicating the exact sequence from simple-evals' Random(seed).
        n = len(expanded.test_set)
        rng = random.Random(seed)
        self._permutations = [rng.sample(range(4), 4) for _ in range(n)]

    @override
    async def preprocess(self, raw, ctx):
        permutation = self._permutations[ctx.sample_id]
        choices_list = [
            raw["Correct Answer"],
            raw["Incorrect Answer 1"],
            raw["Incorrect Answer 2"],
            raw["Incorrect Answer 3"],
        ]
        shuffled_choices = [choices_list[i] for i in permutation]
        correct_index = shuffled_choices.index(raw["Correct Answer"])
        correct_answer_letter = "ABCD"[correct_index]
        data = {
            "Question": raw["Question"],
            "A": shuffled_choices[0],
            "B": shuffled_choices[1],
            "C": shuffled_choices[2],
            "D": shuffled_choices[3],
            "Answer": correct_answer_letter,
        }
        return build_prompt_record(
            [
                {"role": "user", "content": QUERY_TEMPLATE_MULTICHOICE.format(**data)},
            ],
            reference=correct_answer_letter,
            extra={"permutation": permutation},
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
            match = re.search(ANSWER_PATTERN_MULTICHOICE, text)
            predictions.append(match.group(1) if match else None)
        return build_prediction_record(predictions)

    @override
    async def feedback(self, post, ctx):
        reference = ctx.preprocess_result["reference"]
        rollouts = [
            build_rollout_judgement(
                rollout["index"], rollout.get("prediction") == reference
            )
            for rollout in post["rollouts"]
        ]
        first = post["rollouts"][0].get("prediction") if post["rollouts"] else None
        return True, build_judgement_record(
            reference,
            rollouts,
            # `chars` measures the extracted letter, not the model's answer, so
            # it is always 0 or 1 and nothing reads it. Kept to keep this a pure
            # refactor; drop it once someone confirms no external consumer.
            extra={"chars": len(first or "")},
        )

    @override
    async def report(self, finals, fails):
        # The FIRST rollout's verdict. Not `n_correct`: that reads an int as a
        # bool, and would silently become pass@n. This benchmark publishes a
        # single-draw number, so scoring the whole draw would restate it.
        warn_unscored_rollouts(finals, task="gpqa_diamond_0shot_gen")
        count = first_rollout_correct(finals)
        score = 100 * count / len(finals) if finals else 0.0
        return {
            "score": score,
            "fails": len(fails),
            SCORE_KEY_FIELD: "score",
            DENOMINATOR_FIELD: DENOMINATOR_JUDGED,
        } | health_metrics(finals)
