"""Tests for the recall-style RULER base-model tasks (NIAH/VT/CWE/FWE, completion).

Same scoring as the chat recall tasks (``string_match_all``), but ``preprocess``
returns the raw prompt **string** (fed to a GenModel), not a chat message list.
``preprocess`` delegates to the shared ``_build_prompt`` (resolved via MRO), so it
needs a real ``self``; an uninitialized instance suffices (no dataset/model).
"""

import pytest

from sieval.core.tasks.context import TaskContext
from sieval.tasks.ruler.ruler_cwe_0shot_base_gen import RulerCweZeroShotBaseGenTask
from sieval.tasks.ruler.ruler_fwe_0shot_base_gen import RulerFweZeroShotBaseGenTask
from sieval.tasks.ruler.ruler_niah_0shot_base_gen import RulerNiahZeroShotBaseGenTask
from sieval.tasks.ruler.ruler_vt_0shot_base_gen import RulerVtZeroShotBaseGenTask

RECALL_BASE_TASKS = [
    RulerNiahZeroShotBaseGenTask,
    RulerVtZeroShotBaseGenTask,
    RulerCweZeroShotBaseGenTask,
    RulerFweZeroShotBaseGenTask,
]


@pytest.mark.anyio
@pytest.mark.parametrize("task_cls", RECALL_BASE_TASKS)
async def test_preprocess_returns_raw_prompt_string(task_cls):
    raw = {"prompt": "find the magic number", "answer": ["123"]}
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    pre = await task_cls.preprocess(task_cls.__new__(task_cls), raw, ctx)
    # Completion endpoint: a raw string, NOT a chat message list.
    assert pre == "find the magic number"


@pytest.mark.anyio
@pytest.mark.parametrize("task_cls", RECALL_BASE_TASKS)
async def test_feedback_scores_partial_recall(task_cls):
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
    finals = [_final_ctx(1.0), _final_ctx(0.5), _final_ctx(0.0)]
    report = await RulerNiahZeroShotBaseGenTask.report(None, finals, [])
    assert report["score"] == pytest.approx(50.0)
    assert report["fails"] == 0
