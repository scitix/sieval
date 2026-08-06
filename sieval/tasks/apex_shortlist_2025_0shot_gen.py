"""MathArena Apex Shortlist 2025 zero-shot generative task.

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
from sieval.datasets import ApexShortlist2025DatasetSample


@sieval_task(
    name="apex_shortlist_2025_0shot_gen",
    display_name="MathArena Apex Shortlist 2025 (0-shot, generative)",
    description=(
        "MathArena Apex Shortlist 2025 — 47 problems where frontier models "
        "score around 50%."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended"),
    deps_group="math",
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="matharena",
        url="https://github.com/eth-sri/matharena/blob/a11194deff8c67a232974a383795e8a2776b4c6f/configs/competitions/apex/shortlist_2025.yaml",
        notes=(
            "MathArena-aligned: boxed prompt, last-boxed extraction; equivalence "
            "via math-verify. REPEATS: upstream publishes at 4 runs/problem "
            "(`--n 4`); this task defaults to n=1, so pass n=4 as "
            "`tasks.<name>.args.n` to compare against matharena.ai — setting `n` on "
            "the model is overridden call-time, and k>n is rejected at construction. "
            "DEVIATION: golds are normalized by sieval.community.math.strip_string; "
            "matharena does not. SYMBOLIC GOLDS: problems 8, 12, 14 and 39 answer in "
            "a free variable (n-1, n(n-1)/2, 2n(n-1), 4N^3+9N^2+6N+1), so they lean "
            "on math-verify's symbolic equivalence where the rest of the set needs "
            "only numeric — and the non-strict last-integer fallback can never "
            "satisfy them (upstream is `strict_parsing: false` here too, so the "
            "fallback itself is parity). OVERLAP: 5 of the 47 duplicate sibling "
            "datasets, so a run covering them scores those twice — 29 = brumo_2025 "
            "problem 30 and 27/28 = hmmt_feb_2025 problems 19/20 byte-for-byte, plus "
            "25/26 = aime_2025 problems 14/15, which match only after normalization "
            "(same statement and gold; that loader mirrors opencompass/AIME2025, "
            "which re-typesets). Reconciled against every `source` string: the two it "
            "attributes to the HMMT 2025 team round are NOT duplicates, because "
            "hmmt_feb_2025 carries the individual rounds only. Not a superset of "
            "apex_2025 either — the two share no problem. PROMPT COHORT: this task "
            "sends the pinned config's `instruction`, which upstream later changed "
            "without re-running earlier rows — 30 of 42 models on matharena.ai's "
            "shortlist table (4,813/9,659 rollouts) match it; the rest carry a "
            "leading `Please reason step by step, and `. That confounds any delta "
            "against an older row rather than being a defect; measured once, on "
            "brumo_2025, at 0.8 pp — below that set's sampling noise. VALIDATED: "
            "replaying MathArena/apex_shortlist_outputs (9,659 rollouts) reproduces "
            "upstream's `correct` on 99.4%, inside the 96.2-99.7% band of the shipped "
            "AIME/HMMT ports; 43 of the 62 residual disagreements would close if "
            "sieval vendored upstream's normalize_string (it deliberately does not) — "
            "this set's expression-valued golds invite `X = <expr>` and prose-wrapped "
            "boxes, which the integer-answer AIME/HMMT sets almost never produce."
        ),
    ),
)
class ApexShortlist2025ZeroShotGenTask(
    Task[
        ApexShortlist2025DatasetSample,
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
        from math_verify import parse, verify

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
                parsed_pred = parse(pred_with_env)
                parsed_ref = parse(ref_with_env)
                # math_verify.verify expects the gold answer as the first arg.
                correct = verify(parsed_ref, parsed_pred)
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
