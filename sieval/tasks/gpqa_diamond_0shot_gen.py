import random
import re
from typing import TypedDict, override

from openai.types.chat import ChatCompletionUserMessageParam

from sieval.community.simple_evals.common import (
    ANSWER_PATTERN_MULTICHOICE,
    QUERY_TEMPLATE_MULTICHOICE,
)
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import GPQADiamondDatasetSample


class Preprocessed(TypedDict):
    msg: list[ChatCompletionUserMessageParam]
    answer: str  # correct answer letter after permutation


class Feedback(TypedDict):
    correct: bool
    chars: int


@sieval_task(
    name="gpqa_diamond_0shot_gen",
    display_name="GPQA-Diamond (0-shot, generative)",
    description="Graduate-level science MCQ — diamond subset, 198 questions.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "multiple-choice", "graduate-level"),
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="simple-evals",
        url="https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/gpqa_eval.py",
        notes="Permutation + seed logic aligned with simple-evals.",
    ),
)
class GPQADiamondZeroShotGenTask(
    Task[
        GPQADiamondDatasetSample,
        Preprocessed,
        ModelOutput,
        str,
        Feedback,
        dict[str, float],
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
        return {
            "msg": [
                {"role": "user", "content": QUERY_TEMPLATE_MULTICHOICE.format(**data)},
            ],
            "answer": correct_answer_letter,
        }

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["msg"])

    @override
    async def postprocess(self, inf, ctx):
        match = re.search(
            ANSWER_PATTERN_MULTICHOICE, inf.texts[0]
        )  # n=1, only one choice
        return match.group(1) if match else ""

    @override
    async def feedback(self, post, ctx):
        return True, {
            "correct": post == ctx.preprocess_result["answer"],
            "chars": len(post),
        }

    @override
    async def report(self, finals, fails):
        count = sum(1 for ctx in finals if ctx.feedback_result["correct"])
        score = 100 * count / len(finals) if finals else 0.0
        return {"score": score, "fails": len(fails)}
