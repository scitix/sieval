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

from openai.types.chat import ChatCompletionUserMessageParam

from sieval.community.ruler.eval.constants import string_match_part
from sieval.core.models import ModelOutput
from sieval.core.tasks import Task


class RulerRecallSample(TypedDict):
    """Structural bound for recall-style RULER samples (NIAH/VT/CWE/FWE)."""

    prompt: str
    answer: list[str]


class RecallFeedback(TypedDict):
    score: float


class QaFeedback(TypedDict):
    prediction: str
    references: list[str]


# --- Scoring mixins (endpoint-agnostic: feedback + report only) ---------------


class _RecallScoringMixin:
    """RULER ``string_match_all``: per-sample mean recall over references, ×100."""

    async def feedback(self, post, ctx):
        refs = ctx.raw_sample["answer"]
        pred = post.lower()
        score = sum(1.0 for r in refs if r.lower() in pred) / len(refs)
        return True, {"score": score}

    async def report(self, finals, fails):
        count = len(finals)
        total = sum(ctx.feedback_result["score"] for ctx in finals)
        avg = total / count * 100 if count > 0 else 0.0
        return {"score": avg, "fails": len(fails)}


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
        list[ChatCompletionUserMessageParam],
        ModelOutput,
        str,
        TFeedback,
        dict[str, float],
    ],
    ABC,
):
    """Chat endpoint: wrap the synthesized prompt in a single user turn."""

    def __init__(self, dataset, model, name: str | None = None):
        super().__init__(dataset=dataset, model=model, name=name)

    async def preprocess(self, raw, ctx):
        return [{"role": "user", "content": self._build_prompt(raw)}]

    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

    async def postprocess(self, inf, ctx):
        return inf.texts[0]

    def _build_prompt(self, raw) -> str:
        raise NotImplementedError




# --- Prompt-shape mixins ------------------------------------------------------


class _RecallPromptMixin:
    def _build_prompt(self, raw) -> str:
        return raw["prompt"]


class _QaPromptMixin:
    def _build_prompt(self, raw) -> str:
        # RULER stores the prompt split into body + answer cue; the model sees
        # them concatenated (mirrors the original RULER jsonl `input + answer_prefix`).
        return raw["input"] + raw["answer_prefix"]


# --- Leaf bases (scoring × prompt × endpoint) ---------------------------------


class RulerRecallGenTask[TSample: RulerRecallSample](
    _RecallScoringMixin, _RecallPromptMixin, _ChatGenBase[TSample, RecallFeedback]
):
    """Recall-style RULER task over the chat endpoint."""


class RulerQaGenTask[TSample](
    _QaScoringMixin, _QaPromptMixin, _ChatGenBase[TSample, QaFeedback]
):
    """QA-style RULER task over the chat endpoint."""
