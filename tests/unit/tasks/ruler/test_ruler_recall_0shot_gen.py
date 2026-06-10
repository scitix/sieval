"""Tests for the recall-style RULER tasks (NIAH/VT/CWE/FWE).

All four share :class:`RulerRecallGenTask`; scoring is RULER ``string_match_all``
(per-sample mean recall over reference answers, averaged across samples × 100).
``preprocess``/``feedback``/``report`` read only ctx + args (never ``self``), so
they run as unbound methods with ``self=None`` — no dataset/model construction.
"""

import pytest

from sieval.core.tasks.context import TaskContext
from sieval.tasks.ruler.ruler_cwe_0shot_gen import RulerCweZeroShotGenTask
from sieval.tasks.ruler.ruler_fwe_0shot_gen import RulerFweZeroShotGenTask
from sieval.tasks.ruler.ruler_niah_0shot_gen import RulerNiahZeroShotGenTask
from sieval.tasks.ruler.ruler_vt_0shot_gen import RulerVtZeroShotGenTask

# Every recall task inherits the same pipeline from RulerRecallGenTask; running
# the shared assertions against all four guards against an accidental override.
RECALL_TASKS = [
    RulerNiahZeroShotGenTask,
    RulerVtZeroShotGenTask,
    RulerCweZeroShotGenTask,
    RulerFweZeroShotGenTask,
]


@pytest.mark.anyio
@pytest.mark.parametrize("task_cls", RECALL_TASKS)
async def test_preprocess_passes_prompt_through(task_cls):
    raw = {"prompt": "find the magic number", "answer": ["123"]}
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    # preprocess delegates to the shared `_build_prompt` (resolved via MRO), so it
    # needs a real `self`; an uninitialized instance suffices (no dataset/model).
    pre = await task_cls.preprocess(task_cls.__new__(task_cls), raw, ctx)
    assert pre == [{"role": "user", "content": "find the magic number"}]


@pytest.mark.anyio
@pytest.mark.parametrize("task_cls", RECALL_TASKS)
async def test_feedback_scores_partial_recall(task_cls):
    # 1 of 2 references present (case-insensitive) → 0.5 recall.
    raw = {"prompt": "p", "answer": ["Alpha", "Beta"]}
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    finalize, fb = await task_cls.feedback(None, "the answer mentions alpha", ctx)
    assert finalize is True
    assert fb == {"score": 0.5}


def _final_ctx(score: float) -> TaskContext:
    ctx = TaskContext(sample_id=0, raw_sample={"prompt": "p", "answer": ["x"]})
    return ctx.to_feedback({"score": score})


@pytest.mark.anyio
async def test_report_means_recall_and_scales_to_100():
    # string_match_all averages per-sample recall, then × 100.
    finals = [_final_ctx(1.0), _final_ctx(0.5), _final_ctx(0.0)]
    report = await RulerNiahZeroShotGenTask.report(None, finals, [])
    assert report["score"] == pytest.approx(50.0)
    assert report["fails"] == 0


@pytest.mark.anyio
async def test_report_empty_is_zero():
    report = await RulerVtZeroShotGenTask.report(None, [], [])
    assert report["score"] == 0.0
