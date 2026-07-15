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
confidence, and grader model id are persisted in the feedback record.

NOTE: BrowseComp is designed to require web browsing; a plain (closed-book)
model scores near-zero — this task grades whatever answer the model returns, so
the browsing capability itself must be provided by the model/scaffold.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from collections.abc import Mapping
from typing import TypedDict, override

from openai.types.chat import ChatCompletionUserMessageParam

from sieval.community.browsecomp import (
    GRADER_TEMPLATE,
    QUERY_TEMPLATE,
    aggregate_metrics,
    parse_confidence,
    parse_grade,
)
from sieval.core.models import ChatModel, Model, ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import BrowseCompDatasetSample


class GradeFeedback(TypedDict):
    grade: str  # "CORRECT" | "INCORRECT"
    confidence: int  # grader-echoed confidence 0-100 (for post-hoc calibration)
    gold: str
    predicted: str
    grader_model: str


@sieval_task(
    name="browsecomp_0shot_gen",
    display_name="BrowseComp (0-shot, generative)",
    description="Hard live-web browsing QA graded yes/no by an LLM autorater.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "browsing", "deep-research", "open-ended"),
    model_type="chat",
    status="experimental",
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
            "confidence and grader model id are persisted. BrowseComp requires "
            "browsing: closed-book models score near-zero — validated at 0.316% "
            "(4/1266) for gemma-3-27b-it with the gpt-4.1 autorater — so a "
            "meaningful score needs a browsing-capable model/scaffold."
        ),
    ),
)
class BrowseCompZeroShotGenTask(
    Task[
        BrowseCompDatasetSample,
        list[ChatCompletionUserMessageParam],
        ModelOutput,
        list[str],
        list[GradeFeedback],
        dict[str, float],
    ]
):
    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        grader: Mapping | Model | None = None,
        n: int = 1,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._n = n
        self._grader = self._build_grader(grader)

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
        return [
            {"role": "user", "content": QUERY_TEMPLATE.format(Question=raw["problem"])}
        ]

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre, n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        return list(inf.texts)

    @override
    async def feedback(self, post, ctx):
        raw = ctx.raw_sample
        question = raw["problem"]
        gold = raw["answer"]
        grader_model = self._grader.meta()["model"]

        feedbacks: list[GradeFeedback] = []
        for predicted in post:
            prompt = GRADER_TEMPLATE.format(
                question=question,
                correct_answer=gold,
                response=predicted,
            )
            out = await self._grader.agenerate(prompt)
            reply = out.texts[0] if out.texts else ""
            feedbacks.append(
                {
                    "grade": parse_grade(reply),
                    "confidence": parse_confidence(reply),
                    "gold": gold,
                    "predicted": predicted,
                    "grader_model": grader_model,
                }
            )
        return True, feedbacks

    @override
    async def report(self, finals, fails):
        graded = [fb["grade"] for f in finals for fb in (f.feedback_result or [])]
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
        }
