"""HMMT November 2025 zero-shot generative task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
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
from sieval.core.utils.offload import GRADE_TIMEOUT, run_cpu_bound
from sieval.datasets import HMMTNov2025DatasetSample

from ._math_verify import verify_answer


@sieval_task(
    name="hmmt_nov_2025_0shot_gen",
    display_name="HMMT Nov 2025 (0-shot, generative)",
    description=(
        "HMMT November 2025 — Harvard-MIT Mathematics Tournament, 30 problems."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended"),
    deps_group="math",
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="matharena",
        url="https://github.com/eth-sri/matharena/blob/a11194deff8c67a232974a383795e8a2776b4c6f/configs/competitions/hmmt/hmmt_nov_2025.yaml",
        notes=(
            "MathArena-aligned: boxed prompt, last-boxed extraction; equivalence "
            "via math-verify. REPEATS: upstream publishes at 4 runs/problem "
            "(`--n 4`); this task defaults to n=1, so pass n=4 as "
            "`tasks.<name>.args.n` to compare against matharena.ai — setting `n` on "
            "the model is overridden call-time, and k>n is rejected at construction. "
            "DEVIATION: golds are normalized by sieval.community.math.strip_string; "
            "matharena does not. PROMPT COHORT: this task sends the pinned config's "
            "`instruction`, which upstream later changed without re-running earlier "
            "rows — only 5 of 22 models on matharena.ai's HMMT Nov 2025 table "
            "(600/2,640 rollouts) match it; the rest carry a leading `Please reason "
            "step by step, and `. That confounds any delta against an older row "
            "rather than being a defect, the live gemini-3-flash-preview delta below "
            "included; measured once, on brumo_2025, at 0.8 pp — below that set's "
            "sampling noise. VALIDATED against official MathArena: replaying its "
            "published 2,640 outputs (22 models x 30 problems x 4 runs) through this "
            "task's grading path agrees with the upstream grader on 99.51% and "
            "reproduces 16/22 model scores exactly; Gemini 3 Flash is 93.33% three "
            "ways (published, upstream grader, sieval grader). A live sieval run of "
            "gemini-3-flash-preview scored 95.00% vs the published 93.33% — sampling "
            "variance, not a grading difference: both graders agree on 120/120 of "
            "sieval's own outputs."
        ),
    ),
)
class HMMTNov2025ZeroShotGenTask(
    Task[
        HMMTNov2025DatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        dict[str, float],
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
        if total == 0:
            # Same key set as the populated path, so `pass@1` never KeyErrors.
            return self._metrics(0.0, 0.0, len(fails))

        pass_at_1_total = 0.0
        pass_at_k_total = 0.0
        short = 0
        for f in finals:
            judgement = f.feedback_result
            n_samples = judgement["n_rollouts"]
            if n_samples < self._k:
                short += 1
            correct_num = judgement["n_correct"]
            pass_at_1_total += self._pass_at_k(n_samples, correct_num, 1)
            if self._k > 1:
                pass_at_k_total += self._pass_at_k(n_samples, correct_num, self._k)

        if short:
            logger.warning(
                "{}/{} sample(s) returned fewer than k={} choices (model produced "
                "fewer than the requested n={}) and contribute 0 to pass@{}.",
                short,
                len(finals),
                self._k,
                self._n,
                self._k,
            )

        return self._metrics(
            pass_at_1_total * 100 / total,
            pass_at_k_total * 100 / total,
            len(fails),
        )

    def _metrics(
        self, pass_at_1: float, pass_at_k: float, fails: int
    ) -> dict[str, float]:
        # Single source of truth for the report key set — both branches route here.
        metrics = {"score": pass_at_1, "fails": fails, "pass@1": pass_at_1}
        if self._k > 1:
            metrics[f"pass@{self._k}"] = pass_at_k
        return metrics

    def _pass_at_k(self, n: int, c: int, k: int) -> float:
        if n < k:
            # Unreachable by config (__init__ rejects k > n); only a model that
            # returned fewer choices than requested lands here, and report() warns.
            return 0.0
        if c == 0:
            return 0.0
        # Formula: 1 - product_{i=0}^{k-1} (n - c - i) / (n - i)
        # This calculates the probability that all k samples are wrong
        prob_all_wrong = 1.0
        for i in range(k):
            prob_all_wrong *= (n - c - i) / (n - i)
        return 1.0 - prob_all_wrong
