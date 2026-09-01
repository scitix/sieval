import re
from typing import override

from loguru import logger

from sieval.community.simple_evals.common import ANSWER_PATTERN
from sieval.community.simple_evals.math_eval import QUERY_TEMPLATE
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
    DENOMINATOR_REQUESTED,
    SCORE_KEY_FIELD,
    health_metrics,
    sampling_report,
    ungated_intervals,
)
from sieval.core.utils.offload import GRADE_TIMEOUT, run_cpu_bound
from sieval.datasets import AIME2025DatasetSample

from ._math_verify import normalize_vote, verify_answer


@sieval_task(
    name="aime_2025_0shot_gen",
    display_name="AIME 2025 (0-shot, generative)",
    description=(
        "AIME 2025 — American Invitational Mathematics Examination, 30 problems."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended"),
    deps_group="math",
    model_type="chat",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="simple-evals",
        url="https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/math_eval.py",
        notes="Math postprocess + ANSWER_PATTERN extraction aligned with simple-evals.",
    ),
)
class AIME2025ZeroShotGenTask(
    Task[
        AIME2025DatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        # `float | str`: the report carries `score_key`, which names a column
        # rather than measuring one; `list[float]` carries an interval, and
        # `dict[str, str]` the `ci95_units` map naming each interval's unit.
        dict[str, float | str | list[float] | dict[str, str]],
    ],
):
    def __init__(self, dataset, model, name: str | None = None, k: int = 1, n: int = 1):
        super().__init__(dataset=dataset, model=model, name=name)
        if k > n:
            raise ValueError(
                f"pass@{k} needs at least {k} sample(s) per problem, got n={n}. "
                "Raise the task arg `n` (tasks.<name>.args.n) to at least k — "
                "setting `n` on the model is silently overridden call-time."
            )
        self._k = k
        self._n = n

    @override
    async def preprocess(self, raw, ctx):
        return build_prompt_record(
            [
                {
                    "role": "user",
                    "content": QUERY_TEMPLATE.format(problem=raw["question"]),
                },
            ],
            reference=raw["answer"],
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        predictions = []
        for choice in inf.texts:
            match = re.search(ANSWER_PATTERN, choice)
            predictions.append(match.group(1).strip() if match else None)
        return build_prediction_record(predictions)

    @override
    async def feedback(self, post, ctx):
        rollouts = []
        ground_truth = ctx.raw_sample["answer"]
        for rollout in post["rollouts"]:
            pred = rollout.get("prediction")
            if pred is None:
                rollouts.append(build_rollout_judgement(rollout["index"], False))
                continue
            pred_with_env = f"${pred}$"
            ref_with_env = f"${ground_truth}$"
            try:
                correct = await run_cpu_bound(
                    verify_answer, ref_with_env, pred_with_env, timeout=GRADE_TIMEOUT
                )
            except TimeoutError:
                # A grade that could not be COMPUTED IN TIME is a wrong answer,
                # not a failed run: the contract the whole math family keeps, and
                # `report` counts fails in the denominator, so the accuracy is
                # the same either way. Every OTHER exception now propagates and
                # the sample lands in `fails` as `exception::<class>` -- a grader
                # that is broken rather than slow (a dead worker, an optional
                # dependency missing from the environment) must not read as a
                # model that answered wrongly.
                logger.warning(
                    "Grading sample {} exceeded {}s and was scored wrong; the "
                    "prediction is likely a shape the grader cannot bound.",
                    ctx.sample_id,
                    GRADE_TIMEOUT,
                )
                correct = False
            rollouts.append(build_rollout_judgement(rollout["index"], correct))
        return True, build_judgement_record(ground_truth, rollouts)

    @override
    async def report(self, finals, fails):
        total = len(finals) + len(fails)
        rolled = sampling_report(
            finals,
            n=self._n,
            k=self._k,
            denominator=total,
            normalize=normalize_vote,
            score_key="pass@1",
            grouping=self.problem_groups(finals),
        )
        # Read back out of the shared block, so `score` cannot drift from it.
        pass_at_1 = rolled["pass@1"]
        metrics: dict[str, float | str | list[float] | dict[str, str]] = {
            "score": pass_at_1,
            "fails": len(fails),
            "pass@1": pass_at_1,
            SCORE_KEY_FIELD: "pass@1",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
        }
        # Outside the n>1 gate, because the metrics they bracket are: `pass@1`
        # is published at every budget, and so is the headline copied from it.
        metrics |= ungated_intervals(rolled, metrics=("score", "pass@1"))
        if self._n > 1:
            # At n=1 the rest only restates `pass@1`.
            metrics.update(rolled)
        # Outside the gate: extraction health is a fact about the parser, not
        # about the draw, and n=1 is where a stopped extractor hides longest.
        return metrics | health_metrics(finals)
