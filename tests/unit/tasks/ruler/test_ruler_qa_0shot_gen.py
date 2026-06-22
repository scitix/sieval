from typing import Any

import pytest

from sieval.core.tasks.context import TaskContext
from sieval.tasks.ruler.ruler_qa_0shot_gen import RulerQaZeroShotGenTask

# feedback/report read only ctx + args (never `self`), so they can be invoked
# as unbound methods with self=None — no dataset/model construction needed.
# `_SELF` is typed Any so passing it as `self` type-checks without per-line ignores.
_SELF: Any = None


class _StubModel:
    """Minimal stand-in for the chat model `preprocess` reads. Non-reasoning
    name + no `enable_thinking` → `thinking_prefill` returns "" (general case)."""

    _model = "test-model"
    _kwargs: dict = {}


@pytest.mark.anyio
async def test_preprocess_splits_body_and_answer_prefix():
    """Body goes in the user turn; the answer cue is an assistant prefill turn."""
    raw = {
        "input": "Document 1: foo. Question: q?",
        "answer_prefix": " Answer:",
        "outputs": ["x"],
    }
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    # preprocess resolves `_build_prompt` via MRO and reads `self.model` to decide
    # whether to prefill a model-specific placeholder. A non-reasoning stub model
    # exercises the general case: no prefill, so the answer cue passes through.
    task = RulerQaZeroShotGenTask.__new__(RulerQaZeroShotGenTask)
    task._model = _StubModel()
    pre = await RulerQaZeroShotGenTask.preprocess(task, raw, ctx)
    assert pre == [
        {"role": "user", "content": "Document 1: foo. Question: q?"},
        {"role": "assistant", "content": " Answer:"},
    ]


@pytest.mark.anyio
async def test_feedback_carries_prediction_and_references():
    raw = {"outputs": ["Paris", "the capital"]}
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    finalize, fb = await RulerQaZeroShotGenTask.feedback(
        _SELF, "The answer is paris.", ctx
    )
    assert finalize is True
    assert fb == {
        "prediction": "The answer is paris.",
        "references": ["Paris", "the capital"],
    }


def _final_ctx(prediction: str, references: list[str]) -> TaskContext:
    ctx = TaskContext(sample_id=0, raw_sample={"outputs": references})
    return ctx.to_feedback({"prediction": prediction, "references": references})


@pytest.mark.anyio
async def test_report_uses_max_over_references():
    """RULER QA uses string_match_part (best-match): a single reference present
    earns full credit, unlike NIAH's string_match_all mean."""
    # Sample 1: one of two refs present → counts as 1.0 under max.
    # Sample 2: no ref present → 0.0. Batch score = (1.0 + 0.0)/2 * 100 = 50.0.
    finals = [
        _final_ctx("the answer is paris.", ["Paris", "the capital"]),
        _final_ctx("Berlin", ["Paris", "London"]),
    ]
    report = await RulerQaZeroShotGenTask.report(_SELF, finals, [])
    assert report["score"] == 50.0
    assert report["fails"] == 0


@pytest.mark.anyio
async def test_report_all_correct_is_100():
    finals = [
        _final_ctx("paris", ["Paris"]),
        _final_ctx("london", ["London"]),
    ]
    report = await RulerQaZeroShotGenTask.report(_SELF, finals, [])
    assert report["score"] == 100.0


@pytest.mark.anyio
async def test_report_empty_is_zero():
    report = await RulerQaZeroShotGenTask.report(_SELF, [], [])
    assert report["score"] == 0.0
