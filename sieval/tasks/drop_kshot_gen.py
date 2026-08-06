import re
from typing import override

from sieval.community.simple_evals.common import ANSWER_PATTERN
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
from sieval.datasets import DROPDatasetSample

_FEWSHOT_SPLIT = "train"
_FEWSHOT_SEED = 42


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
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        dict[str, float],
    ]
):
    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        n_shot: int = 3,
        sep: str = "\n\n",
    ):
        if n_shot < 0:
            raise ValueError(f"n_shot must be >= 0, got {n_shot}")
        super().__init__(dataset=dataset, model=model, name=name)
        self.n_shot = n_shot
        self._sep = sep
        # "" is a legitimate value (the n_shot=0 prefix), so None is the
        # not-yet-built sentinel, mirroring the ARC tasks.
        self._few_shot_str: str | None = None

    @override
    async def setup(self) -> None:
        # Drawn once here (setup runs before any sample) rather than per
        # preprocess: the seed is fixed, so every per-sample call was already
        # returning this same set. Building it here also aborts a too-short
        # pool before any inference spend, like the other few-shot tasks.
        self._few_shot_str = self._build_few_shot_str()

    @override
    async def preprocess(self, raw, ctx):
        from sieval.community.simple_evals.drop_eval import QUERY_TEMPLATE

        few_shot_str = (
            self._few_shot_str
            if self._few_shot_str is not None
            else self._build_few_shot_str()
        )
        return build_prompt_record(
            [
                {
                    "role": "user",
                    "content": QUERY_TEMPLATE.format(
                        few_shot_str=few_shot_str, context=raw["context"]
                    ),
                }
            ],
            # DROP golds are a `|`-separated set of acceptable answers.
            reference=raw["ref_text"],
        )

    def _build_few_shot_str(self) -> str:
        if self.n_shot == 0:
            return ""
        split = self.dataset.dataset_dict.get(_FEWSHOT_SPLIT)
        if split is None:
            raise ValueError(
                "DROP few-shot generative task requires a "
                f"{_FEWSHOT_SPLIT!r} split for few-shot examples."
            )
        # retrieve_samples truncates to the split length, which would render
        # fewer shots than meta.json reports with nothing on
        # disk saying so. Fail instead, as the other few-shot tasks do.
        if len(split) < self.n_shot:
            raise ValueError(
                "DROP few-shot generative task requires at least "
                f"{self.n_shot} examples in split {_FEWSHOT_SPLIT!r}; "
                f"found {len(split)}."
            )
        # Lazy, and after the guards: drop_eval pulls scipy, and importing this
        # module for registration must not (pinned by a test).
        from sieval.community.simple_evals.drop_eval import FEW_SHOT_TEMPLATE

        examples = self.dataset.retrieve_samples(
            self.n_shot,
            split=_FEWSHOT_SPLIT,
            mode="random",
            seed=_FEWSHOT_SEED,
        )
        return self._sep.join(
            FEW_SHOT_TEMPLATE.format(context=ex["context"], completion=ex["completion"])
            for ex in examples
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"])

    @override
    async def postprocess(self, inf, ctx):
        match = re.search(ANSWER_PATTERN, inf.texts[0])  # n=1, only one choice
        # No pattern match falls back to the whole response, exactly as before;
        # only a wholly empty response has no answer at all.
        return build_prediction_record(
            [(match.group(1) if match else inf.texts[0]) or None]
        )

    @override
    async def feedback(self, post, ctx):
        from sieval.community.simple_evals.drop_eval import drop_metric

        ref = ctx.raw_sample["ref_text"]
        answers = ref.split("|")
        prediction = post["rollouts"][0].get("prediction") or ""
        em, f1 = drop_metric(prediction, answers)
        # DROP's two published metrics fit the headline natively -- exact match is
        # the binary verdict, F1 the partial credit -- but both are also named in
        # `metrics`, so a generic reader can enumerate them without knowing that
        # this task's `score` happens to be F1. report() pools from `metrics`, so
        # the aggregate cannot drift from the values recorded here.
        metrics: dict[str, bool | float] = {"em": em, "f1": f1}
        return True, build_judgement_record(
            ref,
            [build_rollout_judgement(0, bool(em), score=f1, metrics=metrics)],
            score=f1,
            metrics=metrics,
        )

    @override
    async def report(self, finals, fails):
        count = len(finals)
        total_em = sum(ctx.feedback_result["metrics"]["em"] for ctx in finals)
        total_f1 = sum(ctx.feedback_result["metrics"]["f1"] for ctx in finals)
        avg_em = total_em / count * 100 if count > 0 else 0
        avg_f1 = total_f1 / count if count > 0 else 0
        return {"score": avg_f1, "fails": len(fails), "em": avg_em, "f1": avg_f1}
