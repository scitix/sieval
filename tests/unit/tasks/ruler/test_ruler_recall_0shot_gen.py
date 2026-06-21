"""Tests for the recall-style RULER tasks (NIAH/VT/CWE/FWE).

All four share :class:`RulerRecallGenTask`; scoring is RULER ``string_match_all``
(per-sample mean recall over reference answers, averaged across samples × 100).
``preprocess``/``feedback``/``report`` read only ctx + args (never ``self``)
except ``preprocess`` which resolves ``_build_prompt`` via MRO — so they run as
unbound methods, with an uninitialized instance where ``self`` is needed (no
dataset/model construction).
"""

import pytest

from sieval.core.tasks.context import TaskContext
from sieval.tasks.ruler.ruler_cwe_kshot_gen import RulerCweFewShotGenTask
from sieval.tasks.ruler.ruler_fwe_0shot_gen import RulerFweZeroShotGenTask
from sieval.tasks.ruler.ruler_niah_0shot_gen import RulerNiahZeroShotGenTask
from sieval.tasks.ruler.ruler_vt_kshot_gen import RulerVtFewShotGenTask

# Every recall task inherits the same pipeline from RulerRecallGenTask; running
# the shared assertions against all four guards against an accidental override.
RECALL_TASKS = [
    RulerNiahZeroShotGenTask,
    RulerVtFewShotGenTask,
    RulerCweFewShotGenTask,
    RulerFweZeroShotGenTask,
]


class _StubModel:
    """Minimal stand-in for the chat model `preprocess` reads. Non-reasoning
    name + no `enable_thinking` → `thinking_prefill` returns "" (general case)."""

    _model = "test-model"
    _kwargs: dict = {}


@pytest.mark.anyio
@pytest.mark.parametrize("task_cls", RECALL_TASKS)
async def test_preprocess_splits_body_and_answer_prefix(task_cls):
    """Body goes in the user turn; the answer cue is an assistant prefill turn."""
    raw = {
        "input": "find the magic number",
        "answer_prefix": " The magic number is",
        "outputs": ["123"],
    }
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    # preprocess resolves `_build_prompt` via MRO and reads `self.model` to decide
    # whether to prefill a model-specific placeholder. A non-reasoning stub model
    # exercises the general case: no prefill, so the answer cue passes through.
    task = task_cls.__new__(task_cls)
    task._model = _StubModel()
    pre = await task_cls.preprocess(task, raw, ctx)
    assert pre == [
        {"role": "user", "content": "find the magic number"},
        {"role": "assistant", "content": " The magic number is"},
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("task_cls", RECALL_TASKS)
async def test_feedback_carries_prediction_and_references(task_cls):
    """feedback forwards the prediction + references (from ``outputs``); scoring
    happens batch-wide in ``report``."""
    raw = {"input": "p", "answer_prefix": "", "outputs": ["Alpha", "Beta"]}
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    finalize, fb = await task_cls.feedback(None, "the answer mentions alpha", ctx)
    assert finalize is True
    assert fb == {
        "prediction": "the answer mentions alpha",
        "references": ["Alpha", "Beta"],
    }


def _final_ctx(prediction: str, references: list[str]) -> TaskContext:
    ctx = TaskContext(
        sample_id=0,
        raw_sample={"input": "p", "answer_prefix": "", "outputs": references},
    )
    return ctx.to_feedback({"prediction": prediction, "references": references})


@pytest.mark.anyio
async def test_report_means_recall_and_scales_to_100():
    # string_match_all averages per-sample recall (fraction of refs present), ×100.
    # S1: both of 2 refs present → 1.0; S2: 1 of 2 → 0.5; S3: 0 of 1 → 0.0.
    # mean(1.0, 0.5, 0.0) * 100 = 50.0
    finals = [
        _final_ctx("alpha and beta", ["Alpha", "Beta"]),
        _final_ctx("only alpha here", ["Alpha", "Beta"]),
        _final_ctx("nothing", ["Gamma"]),
    ]
    report = await RulerNiahZeroShotGenTask.report(None, finals, [])
    assert report["score"] == pytest.approx(50.0)
    assert report["fails"] == 0


@pytest.mark.anyio
async def test_report_empty_is_zero():
    report = await RulerVtFewShotGenTask.report(None, [], [])
    assert report["score"] == 0.0
