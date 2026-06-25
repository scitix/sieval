"""Tests for the unified RulerZeroShotGenTask.

feedback/report read only ctx + args (never self), so they can be invoked as
unbound methods with self=None.  _SELF typed Any keeps the type-checker happy.
"""

from typing import Any

import pytest

from sieval.core.tasks.context import TaskContext
from sieval.tasks.ruler_0shot_gen import RulerZeroShotGenTask

_SELF: Any = None


class _StubModel:
    """Minimal stand-in for the chat model preprocess reads."""

    _model = "test-model"
    _kwargs: dict = {}


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_preprocess_splits_body_and_answer_prefix():
    raw = {
        "input": "find the needle",
        "answer_prefix": " Answer:",
        "outputs": ["42"],
        "subtask": "niah_single_1",
        "context_length": 4096,
    }
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    task = RulerZeroShotGenTask.__new__(RulerZeroShotGenTask)
    task._model = _StubModel()
    pre = await RulerZeroShotGenTask.preprocess(task, raw, ctx)
    assert pre == [
        {"role": "user", "content": "find the needle"},
        {"role": "assistant", "content": " Answer:"},
    ]


# ---------------------------------------------------------------------------
# feedback
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_feedback_carries_all_fields():
    raw = {
        "input": "p",
        "answer_prefix": "",
        "outputs": ["Alpha", "Beta"],
        "subtask": "niah_single_1",
        "context_length": 4096,
    }
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    finalize, fb = await RulerZeroShotGenTask.feedback(_SELF, "alpha found", ctx)
    assert finalize is True
    assert fb == {
        "prediction": "alpha found",
        "references": ["Alpha", "Beta"],
        "subtask": "niah_single_1",
        "context_length": 4096,
    }


@pytest.mark.anyio
async def test_feedback_carries_qa_subtask():
    raw = {
        "input": "q",
        "answer_prefix": "",
        "outputs": ["Paris"],
        "subtask": "qa_squad",
        "context_length": 8192,
    }
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    _, fb = await RulerZeroShotGenTask.feedback(_SELF, "paris", ctx)
    assert fb["subtask"] == "qa_squad"
    assert fb["context_length"] == 8192


# ---------------------------------------------------------------------------
# report helpers
# ---------------------------------------------------------------------------


def _ctx(*, prediction: str, references: list[str], subtask: str, ctx_len: int) -> TaskContext:
    raw = {
        "input": "x",
        "answer_prefix": "",
        "outputs": references,
        "subtask": subtask,
        "context_length": ctx_len,
    }
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    return ctx.to_feedback(
        {
            "prediction": prediction,
            "references": references,
            "subtask": subtask,
            "context_length": ctx_len,
        }
    )


# ---------------------------------------------------------------------------
# report — basic correctness
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_report_recall_single_cell():
    # Both refs present → string_match_all = 100.
    finals = [
        _ctx(prediction="alpha beta", references=["Alpha", "Beta"], subtask="niah_single_1", ctx_len=4096),
    ]
    report = await RulerZeroShotGenTask.report(_SELF, finals, [])
    assert report["score"] == pytest.approx(100.0)
    assert report["score_4k"] == pytest.approx(100.0)
    assert report["score_niah_single_1_4k"] == pytest.approx(100.0)
    assert report["fails"] == 0


@pytest.mark.anyio
async def test_report_qa_subtask_uses_string_match_part():
    # string_match_part: sample 1 has "paris" in prediction → 1.0; sample 2 → 0.0.
    # batch = 0.5 * 100 = 50.0
    finals = [
        _ctx(prediction="the answer is paris", references=["Paris"], subtask="qa_squad", ctx_len=4096),
        _ctx(prediction="berlin", references=["London"], subtask="qa_squad", ctx_len=4096),
    ]
    report = await RulerZeroShotGenTask.report(_SELF, finals, [])
    assert report["score_qa_squad_4k"] == pytest.approx(50.0)


@pytest.mark.anyio
async def test_report_aggregates_multiple_lengths():
    # Two lengths: 4k (score=100) and 8k (score=0). Overall = mean(100, 0) = 50.
    finals = [
        _ctx(prediction="alpha", references=["Alpha"], subtask="niah_single_1", ctx_len=4096),
        _ctx(prediction="nothing", references=["Alpha"], subtask="niah_single_1", ctx_len=8192),
    ]
    report = await RulerZeroShotGenTask.report(_SELF, finals, [])
    assert report["score_4k"] == pytest.approx(100.0)
    assert report["score_8k"] == pytest.approx(0.0)
    assert report["score"] == pytest.approx(50.0)


@pytest.mark.anyio
async def test_report_per_length_mean_averages_present_subtasks():
    # 4k: niah_single_1=100, vt=0 → mean=50. Only 1 length → overall=50.
    finals = [
        _ctx(prediction="alpha", references=["Alpha"], subtask="niah_single_1", ctx_len=4096),
        _ctx(prediction="wrong", references=["Alpha"], subtask="vt", ctx_len=4096),
    ]
    report = await RulerZeroShotGenTask.report(_SELF, finals, [])
    assert report["score_4k"] == pytest.approx(50.0)
    assert report["score"] == pytest.approx(50.0)


@pytest.mark.anyio
async def test_report_empty_returns_zero():
    report = await RulerZeroShotGenTask.report(_SELF, [], [])
    assert report["score"] == 0.0
    assert report["fails"] == 0


@pytest.mark.anyio
async def test_report_fails_counted():
    finals = [
        _ctx(prediction="alpha", references=["Alpha"], subtask="niah_single_1", ctx_len=4096),
    ]
    report = await RulerZeroShotGenTask.report(_SELF, finals, ["fail1", "fail2"])
    assert report["fails"] == 2


@pytest.mark.anyio
async def test_report_key_format_uses_len_tag():
    finals = [
        _ctx(prediction="alpha", references=["Alpha"], subtask="niah_multiquery", ctx_len=131072),
    ]
    report = await RulerZeroShotGenTask.report(_SELF, finals, [])
    assert "score_128k" in report
    assert "score_niah_multiquery_128k" in report
