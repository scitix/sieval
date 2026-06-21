
"""Shared base classes for the RULER 0-shot task family.

RULER tasks are thin — the prompt is fully synthesized in the dataset loader, so
every task just sends the prompt and scores the reply. Uses the chat endpoint
(``ChatModel``, where the prompt is wrapped in a user turn and the serving
framework applies the model's chat template).

Scoring lives in mixins so it is shared across endpoints; prompt construction and
the stage plumbing (preprocess/infer/postprocess) live on the endpoint base.
Base classes stay undecorated — only concrete tasks register via ``@sieval_task``.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from abc import ABC
from typing import TypedDict

from openai.types.chat import ChatCompletionMessageParam

from sieval.community.ruler.eval.constants import (
    string_match_all,
    string_match_part,
)
from sieval.core.models import ModelOutput
from sieval.core.tasks import Task
from sieval.datasets.ruler import thinking_prefill


class RulerRecallSample(TypedDict):
    """Structural bound for recall-style RULER samples (NIAH/VT/CWE/FWE).

    Mirrors the dataset row schema: the body, the split-off answer cue, and the
    reference answers (``outputs``). The loaders also emit ``index``/``length``
    (and NIAH ``token_position_answer``), which the task does not read.
    """

    input: str
    answer_prefix: str
    outputs: list[str]


class RecallFeedback(TypedDict):
    prediction: str
    references: list[str]


class QaFeedback(TypedDict):
    prediction: str
    references: list[str]


# --- Scoring mixins (endpoint-agnostic: feedback + report only) ---------------


class _RecallScoringMixin:
    """RULER ``string_match_all``: per-sample mean recall over references, ×100."""

    async def feedback(self, post, ctx):
        refs = ctx.raw_sample["outputs"]
        pred = post
        # Collect individual prediction-reference pairs for batch scoring
        return True, {"prediction": pred, "references": refs}

    async def report(self, finals, fails):
        if not finals:
            return {"score": 0.0, "fails": len(fails)}
        preds = [ctx.feedback_result["prediction"] for ctx in finals]
        refs = [ctx.feedback_result["references"] for ctx in finals]
        score = string_match_all(preds, refs)
        return {"score": score, "fails": len(fails)}


class _QaScoringMixin:
    """RULER ``string_match_part``: best-match over references, batch-wide.

    ``feedback`` carries each prediction + its references forward; the
    authoritative metric runs once over the whole batch in ``report`` to match
    upstream exactly.
    """

    async def feedback(self, post, ctx):
        return True, {"prediction": post, "references": ctx.raw_sample["outputs"]}

    async def report(self, finals, fails):
        preds = [ctx.feedback_result["prediction"] for ctx in finals]
        refs = [ctx.feedback_result["references"] for ctx in finals]
        score = string_match_part(preds, refs) if finals else 0.0
        return {"score": score, "fails": len(fails)}


# --- Endpoint bases (stage plumbing; prompt built by `_build_prompt`) ----------


class _ChatGenBase[TSample, TFeedback](
    Task[
        TSample,
        list[ChatCompletionMessageParam],
        ModelOutput,
        str,
        TFeedback,
        dict[str, float],
    ],
    ABC,
):
    """Chat endpoint: user turn carries the body, an assistant turn prefills the
    RULER answer cue so the model *continues* it instead of re-answering.

    The prefill only works if the serving framework keeps the final assistant
    turn open instead of closing it and appending a fresh generation prompt. That
    is opt-in per deployment via the model's ``extra_body``
    (``continue_final_message`` / ``add_generation_prompt`` for vLLM / SGLang) —
    set in the run config, not here, so it composes with the rest of
    ``extra_body`` instead of overwriting it.
    """

    def __init__(self, dataset, model, name: str | None = None):
        super().__init__(dataset=dataset, model=model, name=name)

    async def preprocess(self, raw, ctx):
        assistant_content = raw["answer_prefix"]

        # Prefill any model-specific assistant-turn placeholder (e.g. Qwen3's
        # empty <think></think> block when thinking is disabled) so the model
        # continues from the answer cue. The dataset loader reserves token budget
        # for the same string via the shared ``thinking_prefill`` helper.
        extra_body = self.model._kwargs.get("extra_body", {})
        enable_thinking = extra_body.get("enable_thinking", True)
        assistant_content = (
            f"{thinking_prefill(self.model._model, enable_thinking)}{assistant_content}"
        )

        return [
            {"role": "user", "content": self._build_prompt(raw)},
            {"role": "assistant", "content": assistant_content},
        ]

    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

    async def postprocess(self, inf, ctx):
        return inf.texts[0]

    def _build_prompt(self, raw) -> str:
        raise NotImplementedError


# --- Prompt-shape mixins ------------------------------------------------------


class _PromptMixin:
    def _build_prompt(self, raw) -> str:
        # The body only — the answer cue (``answer_prefix``) is sent as a separate
        # assistant prefill turn (see `_ChatGenBase.preprocess`) so the model
        # continues from it rather than treating it as part of the user question.
        return raw["input"]


# --- Leaf bases (scoring × prompt × endpoint) ---------------------------------


class RulerRecallGenTask[TSample: RulerRecallSample](
    _RecallScoringMixin, _PromptMixin, _ChatGenBase[TSample, RecallFeedback]
):
    """Recall-style RULER task over the chat endpoint."""


class RulerQaGenTask[TSample](
    _QaScoringMixin, _PromptMixin, _ChatGenBase[TSample, QaFeedback]
):
    """QA-style RULER task over the chat endpoint."""
