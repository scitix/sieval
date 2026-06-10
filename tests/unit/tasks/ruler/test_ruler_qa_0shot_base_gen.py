import pytest

from sieval.core.tasks.context import TaskContext
from sieval.tasks.ruler.ruler_qa_0shot_base_gen import RulerQaZeroShotBaseGenTask

# feedback/report read only ctx + args (never `self`), so they can be invoked
# as unbound methods with self=None — no dataset/model construction needed.


@pytest.mark.anyio
async def test_preprocess_returns_raw_string():
    # Base/completion variant: input + answer_prefix is a single raw prompt
    # string the model continues — NOT a chat message list.
    raw = {
        "input": "Document 1: foo. Question: q?",
        "answer_prefix": " Answer:",
        "outputs": ["x"],
    }
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    # preprocess delegates to the shared `_build_prompt` (resolved via MRO), so it
    # needs a real `self`; an uninitialized instance suffices (no dataset/model).
    task = RulerQaZeroShotBaseGenTask.__new__(RulerQaZeroShotBaseGenTask)
    pre = await RulerQaZeroShotBaseGenTask.preprocess(task, raw, ctx)
    assert pre == "Document 1: foo. Question: q? Answer:"


@pytest.mark.anyio
async def test_feedback_carries_prediction_and_references():
    raw = {"outputs": ["Paris", "the capital"]}
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    finalize, fb = await RulerQaZeroShotBaseGenTask.feedback(
        None, "The answer is paris.", ctx
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
    report = await RulerQaZeroShotBaseGenTask.report(None, finals, [])
    assert report["score"] == 50.0
    assert report["fails"] == 0


@pytest.mark.anyio
async def test_report_all_correct_is_100():
    finals = [
        _final_ctx("paris", ["Paris"]),
        _final_ctx("london", ["London"]),
    ]
    report = await RulerQaZeroShotBaseGenTask.report(None, finals, [])
    assert report["score"] == 100.0


@pytest.mark.anyio
async def test_report_empty_is_zero():
    report = await RulerQaZeroShotBaseGenTask.report(None, [], [])
    assert report["score"] == 0.0
