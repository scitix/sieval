import re
from typing import TypedDict, override

from openai.types.chat import ChatCompletionUserMessageParam

from sieval.community.simple_evals.common import ANSWER_PATTERN
from sieval.community.simple_evals.drop_eval import (
    FEW_SHOT_TEMPLATE,
    QUERY_TEMPLATE,
    drop_metric,
)
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import DROPDatasetSample


class Feedback(TypedDict):
    em: float
    f1: float
    ref_text: str


@sieval_task(
    name="drop_kshot_gen",
    display_name="DROP (few-shot, generative)",
    description="Discrete Reasoning Over Paragraphs — reading-comprehension benchmark.",
    eval_mode=EvalMode.GEN,
    n_shot=3,
    tags=("english", "open-ended"),
    deps_group="drop",
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="simple-evals",
        url="https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/drop_eval.py",
        notes="drop_metric vendored from simple-evals; prompt templates are sieval-local.",  # noqa: E501
    ),
)
class DROPFewShotGenTask(
    Task[
        DROPDatasetSample,
        list[ChatCompletionUserMessageParam],
        ModelOutput,
        str,
        Feedback,
        dict[str, float],
    ]
):
    def __init__(
        self, dataset, model, name: str | None = None, k: int = 3, sep: str = "\n\n"
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._k = k
        self._sep = sep

    @override
    async def preprocess(self, raw, ctx):
        few_shot_examples = self.dataset.retrieve_samples(
            self._k,
            split="train",
            mode="random",
            seed=42,
        )
        few_shot_str = self._sep.join(
            [
                FEW_SHOT_TEMPLATE.format(
                    context=ex["context"], completion=ex["completion"]
                )
                for ex in few_shot_examples
            ]
        )
        return [
            {
                "role": "user",
                "content": QUERY_TEMPLATE.format(
                    few_shot_str=few_shot_str, context=raw["context"]
                ),
            }
        ]

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

    @override
    async def postprocess(self, inf, ctx):
        match = re.search(ANSWER_PATTERN, inf.texts[0])  # n=1, only one choice
        return match.group(1) if match else inf.texts[0]

    @override
    async def feedback(self, post, ctx):
        ref = ctx.raw_sample["ref_text"]
        answers = ref.split("|")
        em, f1 = drop_metric(post, answers)
        return True, {"em": em, "f1": f1, "ref_text": ref}

    @override
    async def report(self, finals, fails):
        count = len(finals)
        total_em = sum(ctx.feedback_result["em"] for ctx in finals)
        total_f1 = sum(ctx.feedback_result["f1"] for ctx in finals)
        avg_em = total_em / count * 100 if count > 0 else 0
        avg_f1 = total_f1 / count if count > 0 else 0
        return {"score": avg_f1, "fails": len(fails), "em": avg_em, "f1": avg_f1}
