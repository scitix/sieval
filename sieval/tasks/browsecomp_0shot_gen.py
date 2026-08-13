"""BrowseComp — 0-shot generative, LLM-autorater graded.

Faithful generative port of BrowseComp (OpenAI, Wei et al., 2025,
arXiv:2504.12516): the model answers a hard, multi-hop, live-web question in an
``Explanation / Exact Answer / Confidence`` block, and a separate **LLM
autorater** grades the free-form answer against the gold as ``correct: yes|no``
→ CORRECT / INCORRECT. The headline metric is accuracy.

Same shape as ``simpleqa_verified_0shot_gen`` (short-answer QA + LLM judge, runs
unchanged on any chat model), differing only in (a) the ``QUERY_TEMPLATE``
wrapper, (b) BrowseComp's yes/no grader prompt, and (c) accuracy instead of
SimpleQA's F1/NOT_ATTEMPTED (BrowseComp has no NOT_ATTEMPTED bucket — a
non-answer is simply INCORRECT).

The grader is supplied via the ``grader`` task arg (a model-config dict, or a
pre-built Model); the official autorater is gpt-4.1-2025-04-14. As with the
other autorater-graded tasks, correctness depends on a real grader model whose
version sieval cannot pin the way it pins a Hub revision, so for reproducibility
pin the grader model and set ``temperature: 0``; each sample's grade, echoed
confidence and the autorater's whole ``ModelOutput`` (``extra.grader_output``:
reply, reasoning, usage, finish reasons, model id) are persisted on the
judgement record — see :meth:`feedback` for why the raw output is kept whole.

NOTE: BrowseComp is designed to require web browsing; a plain (closed-book)
model scores near-zero — this task grades whatever answer the model returns, so
the browsing capability itself must be provided by the model/scaffold.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from collections.abc import Mapping
from typing import override

from sieval.community.browsecomp import (
    GRADER_TEMPLATE,
    QUERY_TEMPLATE,
    aggregate_metrics,
    parse_confidence,
    parse_grade,
)
from sieval.core.models import ChatModel, Model, ModelOutput
from sieval.core.tasks import (
    GRADER_OUTPUT_KEY,
    EvalMode,
    InputKind,
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    ReferenceImpl,
    RequirementContext,
    RolloutJudgement,
    Task,
    TaskModelRequirement,
    TaskRequirements,
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
)
from sieval.core.utils.serialization import obj_to_dict
from sieval.datasets import BrowseCompDatasetSample


@sieval_task(
    name="browsecomp_0shot_gen",
    display_name="BrowseComp (0-shot, generative)",
    description="Hard live-web browsing QA graded yes/no by an LLM autorater.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "browsing", "deep-research", "open-ended"),
    model_type="chat",
    status="experimental",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="browsecomp",
        url="https://arxiv.org/abs/2504.12516",
        notes=(
            "Generative port of BrowseComp (OpenAI, arXiv:2504.12516) — 1,266 "
            "hard live-web questions with short verifiable answers. QUERY_TEMPLATE "
            "and GRADER_TEMPLATE (HLE-derived yes/no autorater) are verbatim from "
            "openai/simple-evals@652c89d0 browsecomp_eval.py; grade parsing matches "
            "upstream except it reads the yes/no capture group (upstream compares "
            "its whole match to bare 'yes', a latent bug). Headline metric = "
            "accuracy. "
            "Grader is a "
            "REAL LLM (official autorater: gpt-4.1-2025-04-14) supplied via the "
            "`grader` task arg on its own api_base/api_key. REPRODUCIBILITY: scores "
            "depend on the grader endpoint's model version (not pinnable like a Hub "
            "revision) — pin the grader model + temperature=0; the per-sample grade/"
            "confidence and the autorater's full ModelOutput (extra."
            "grader_output: reply, reasoning, usage, finish_reasons, model id) "
            "are persisted — the reply being the only evidence of a verdict a "
            "re-grade need not reproduce, and (parse_grade defaults to "
            "INCORRECT) the only way to tell format drift from a real negative; "
            "finish_reasons separates a reasoning grader that spent its budget "
            "thinking from an empty API response. "
            "BrowseComp requires "
            "browsing: closed-book models score near-zero — validated at 0.316% "
            "(4/1266) for gemma-3-27b-it with the gpt-4.1 autorater — so a "
            "meaningful score needs a browsing-capable model/scaffold."
        ),
    ),
)
class BrowseCompZeroShotGenTask(
    Task[
        BrowseCompDatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        # `float | str`: the report carries `score_key`, which names a column
        # rather than measuring one.
        dict[str, float | str],
    ]
):
    @classmethod
    @override
    def model_requirements_for(
        cls, context: RequirementContext
    ) -> tuple[TaskModelRequirement, ...]:
        candidate = super().model_requirements_for(context)
        grader = cls._bind_role_requirement(
            context,
            "grader",
            TaskRequirements(input=InputKind.CHAT),
        )
        return candidate + grader

    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        grader: Mapping | Model | None = None,
        n: int = 1,
        models_by_role: Mapping[str, Model] | None = None,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._n = n
        self._grader = self._resolve_role_model(
            "grader",
            grader,
            models_by_role,
            build=self._build_grader,
        )

    @staticmethod
    def _build_grader(grader: Mapping | Model | None) -> Model:
        """Resolve the ``grader`` task arg into a Model (mapping → ChatModel).

        Grading is mandatory — there is no deterministic fallback — so ``None``
        raises.
        """
        if isinstance(grader, Model):
            return grader
        if isinstance(grader, Mapping):
            return ChatModel(**grader)
        raise ValueError(
            "BrowseComp requires an LLM grader. Pass `grader:` in the task args "
            "— a model-config dict such as "
            "{model: gpt-4.1, api_base: ..., api_key: ..., temperature: 0}."
        )

    @override
    async def preprocess(self, raw, ctx):
        return build_prompt_record(
            [
                {
                    "role": "user",
                    "content": QUERY_TEMPLATE.format(Question=raw["problem"]),
                }
            ],
            reference=raw["answer"],
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        # Open-ended answer: the response *is* the answer. A blank response
        # normalizes to None so `extracted` stays a real signal; the grader still
        # receives "" and rates it.
        return build_prediction_record(
            [text if text.strip() else None for text in inf.texts]
        )

    @override
    async def feedback(self, post, ctx):
        """Grade every rollout, recording the autorater's full output.

        ``extra["grader_output"]`` is the autorater's whole ``ModelOutput``
        flattened to a plain dict, replacing #51's flat ``grader_reply`` (response
        content only). Nothing is hand-picked, so ``finish_reasons`` now separates
        a reasoning grader that spent its budget thinking from an empty API
        response, and a future ``ModelOutput`` field is captured for free.

        ``grade`` is the parsed verdict on top of that raw output -- task logic,
        not model output, hence a separate key -- and ``confidence`` is the
        grader-echoed number kept for post-hoc calibration. Neither measures
        whether the answer was right, so both stay in ``extra``, not ``metrics``.
        """
        raw = ctx.raw_sample
        question = raw["problem"]
        gold = raw["answer"]

        rollouts: list[RolloutJudgement] = []
        for rollout in post["rollouts"]:
            predicted = rollout.get("prediction") or ""
            prompt = GRADER_TEMPLATE.format(
                question=question,
                correct_answer=gold,
                response=predicted,
            )
            out = await self._grader.agenerate(prompt)
            reply = out.texts[0] if out.texts else ""
            grade = parse_grade(reply)
            rollouts.append(
                build_rollout_judgement(
                    rollout["index"],
                    grade == "CORRECT",
                    extra={
                        "grade": grade,
                        "confidence": parse_confidence(reply),
                        GRADER_OUTPUT_KEY: obj_to_dict(out, add_type=False),
                    },
                )
            )
        return True, build_judgement_record(gold, rollouts)

    @override
    async def report(self, finals, fails):
        graded = [
            r["extra"]["grade"]
            for f in finals
            for r in (f.feedback_result or {}).get("rollouts", [])
        ]
        # Pipeline failures (exhausted retries) never produced a gradeable
        # answer; BrowseComp has no NOT_ATTEMPTED bucket, so count each failed
        # sample's requested attempts as INCORRECT — accuracy spans the full
        # requested set, matching the official metric.
        grades = graded + ["INCORRECT"] * (self._n * len(fails))
        m = aggregate_metrics(grades)
        return {
            "score": m["accuracy"] * 100,
            "accuracy": m["accuracy"] * 100,
            "correct": m["is_correct"] * 100,
            "incorrect": m["is_incorrect"] * 100,
            "n_graded": len(graded),
            "fails": len(fails),
            SCORE_KEY_FIELD: "accuracy",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
            # There is no NOT_ATTEMPTED bucket here, so an empty response scores
            # INCORRECT alongside a wrong answer; this is the count that tells
            # them apart. Deliberately only `health_metrics` and not the rest of
            # the sampling block: RFC #74 defers `pass@k` / `maj@k` for the
            # LLM-judged family, while this one measures extraction rather than
            # the draw and is outside that gate.
        } | health_metrics(finals)
