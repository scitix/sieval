"""MathArena Apex 2025 zero-shot generative task.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from typing import override

from loguru import logger

from sieval.community.matharena import BOXED_INSTRUCTION, build_prompt, extract_answer
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
from sieval.datasets import Apex2025DatasetSample

from ._math_verify import normalize_vote, verify_answer


@sieval_task(
    name="apex_2025_0shot_gen",
    display_name="MathArena Apex 2025 (0-shot, generative)",
    description=(
        "MathArena Apex 2025 — 12 problems curated from 2025 competitions to be "
        "very hard for models."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended"),
    deps_group="math",
    model_type="chat",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="matharena",
        url="https://github.com/eth-sri/matharena/blob/a11194deff8c67a232974a383795e8a2776b4c6f/configs/competitions/apex/apex_2025.yaml",
        notes=(
            "MathArena-aligned: boxed prompt, last-boxed extraction; equivalence "
            "via math-verify. REPEATS: set n=16 to compare against matharena.ai — "
            "NOT the family's usual 4. Apex publishes at a higher repeat count than "
            "the rest of the suite (over apex_2025_outputs: 35 of 46 scored models "
            "at 16 runs/problem, 10 at 8, one at 4), which is how a 12-problem set "
            "gets a usable score. Pass it as `tasks.<name>.args.n`; this task still "
            "defaults to n=1, setting `n` on the model is overridden call-time, and "
            "k>n is rejected at construction. DEVIATION: golds are normalized by "
            "sieval.community.math.strip_string; matharena does not. SMALL N: 12 "
            "problems, so one problem moves the score by 8.3 points — read it "
            "alongside a full-size sibling, not on its own. OVERLAP: 3 of the 12 are "
            "byte-identical to smt_2025 problems 8, 42 and 43, so a run covering "
            "both datasets scores those twice; those are the only three, reconciled "
            "against every `source` string. apex_shortlist_2025 is an easier-band "
            "sibling (~50% vs this set's ~5%), NOT a superset — the two share no "
            "problem. PROMPT COHORT: this task sends the pinned config's "
            "`instruction`, which upstream later changed without re-running earlier "
            "rows — 25 of 46 models on matharena.ai's Apex table (3,830/7,717 "
            "rollouts) match it and the rest carry a leading `Please reason step by "
            "step, and `, so the published table mixes two prompts in roughly equal "
            "parts. That confounds any delta against an older row rather than being "
            "a defect; measured once, on brumo_2025, at 0.8 pp — below that set's "
            "sampling noise. VALIDATED: replaying MathArena/apex_2025_outputs (7,717 "
            "rollouts) reproduces upstream's `correct` on 99.9% — the highest of any "
            "ported MathArena competition."
        ),
    ),
)
class Apex2025ZeroShotGenTask(
    Task[
        Apex2025DatasetSample,
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
                    "content": build_prompt(BOXED_INSTRUCTION, raw["problem"]),
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
