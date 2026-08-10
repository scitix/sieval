"""CMIMC 2025 zero-shot generative task.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from typing import override

from loguru import logger

from sieval.community.matharena import CMIMC_INSTRUCTION, build_prompt, extract_answer
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
)
from sieval.core.utils.offload import GRADE_TIMEOUT, run_cpu_bound
from sieval.datasets import CMIMC2025DatasetSample

from ._math_verify import normalize_vote, verify_answer


@sieval_task(
    name="cmimc_2025_0shot_gen",
    display_name="CMIMC 2025 (0-shot, generative)",
    description=(
        "CMIMC 2025 — Carnegie Mellon Informatics and Mathematics Competition, "
        "40 problems."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended"),
    deps_group="math",
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="matharena",
        url="https://github.com/eth-sri/matharena/blob/a11194deff8c67a232974a383795e8a2776b4c6f/configs/competitions/cmimc/cmimc_2025.yaml",
        notes=(
            "MathArena-aligned: boxed prompt, last-boxed extraction; equivalence "
            "via math-verify. PROMPT: the only ported competition whose upstream "
            "`instruction` goes beyond the plain boxed line — it also dictates a "
            "trailing `### Final answer` section, so this task uses "
            "matharena.CMIMC_INSTRUCTION rather than the shared BOXED_INSTRUCTION. "
            "REPEATS: upstream publishes at 4 runs/problem (`--n 4`); this task "
            "defaults to n=1, so pass n=4 as `tasks.<name>.args.n` to compare "
            "against matharena.ai — setting `n` on the model is overridden "
            "call-time, and k>n is rejected at construction. DEVIATION: golds are "
            "normalized by sieval.community.math.strip_string; matharena does not. "
            "PROMPT COHORT: this task sends the pinned config's `instruction`, which "
            "upstream later changed without re-running earlier rows — only 4 of 35 "
            "models on matharena.ai's CMIMC table (640/5,600 rollouts) match it; the "
            "rest carry a leading `Please reason step by step, and `, and 1,088 "
            "rollouts predate the `### Final answer` section entirely, the widest "
            "prompt spread of any ported competition. That confounds any delta "
            "against an older row rather than being a defect; measured once, on "
            "brumo_2025, at 0.8 pp — below that set's sampling noise. VALIDATED: "
            "replaying MathArena/cmimc_2025_outputs (5,600 rollouts) reproduces "
            "upstream's `correct` on 99.1%, inside the 96.2-99.7% band of the "
            "shipped AIME/HMMT ports; 31 of the 51 residual disagreements would "
            "close if sieval vendored upstream's normalize_string (it deliberately "
            "does not)."
        ),
    ),
)
class CMIMC2025ZeroShotGenTask(
    Task[
        CMIMC2025DatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        # `float | str`: the report carries `score_key`, which names a column
        # rather than measuring one.
        dict[str, float | str],
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
                    "content": build_prompt(CMIMC_INSTRUCTION, raw["problem"]),
                },
            ],
            reference=raw["answer"],
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        # MathArena-aligned: last \boxed{}; non-strict -> fall back to last integer.
        # list_answer mirrors grader.py's `gold_answer_is_list`: a comma in the gold
        # switches extraction to join the boxes on the model's final line instead of
        # keeping only the last one.
        raw = ctx.raw_sample
        list_answer = raw is not None and "," in raw["answer"]
        return build_prediction_record(
            [
                extract_answer(choice, strict_parsing=False, list_answer=list_answer)
                for choice in inf.texts
            ]
        )

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
            except Exception as e:
                logger.warning("Feedback failed for sample {}: {}", ctx.sample_id, e)
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
        )
        # Read back out of the shared block, so `score` cannot drift from it.
        pass_at_1 = rolled["pass@1"]
        metrics: dict[str, float | str] = {
            "score": pass_at_1,
            "fails": len(fails),
            "pass@1": pass_at_1,
            SCORE_KEY_FIELD: "pass@1",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
        }
        if self._n > 1:
            # At n=1 the rest only restates `pass@1`.
            metrics.update(rolled)
        # Outside the gate: extraction health is a fact about the parser, not
        # about the draw, and n=1 is where a stopped extractor hides longest.
        return metrics | health_metrics(finals)
