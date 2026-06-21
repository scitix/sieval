"""RULER VT (variable tracking) few-shot generative task.

The prompt is fully synthesized in ``RulerVtDataset.load()``, which (mirroring
upstream RULER) always prepends one in-context demonstration — hence
``n_shot=1``, not 0. This task is thin: send the prompt, then score by
substring recall (RULER ``string_match_all`` — the mean over reference variable
names of whether each appears in the prediction). All pipeline logic lives in
:class:`~sieval.tasks.ruler._base.RulerRecallGenTask`.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    sieval_task,
)
from sieval.datasets import RulerVtDatasetSample
from sieval.tasks.ruler._base import RulerRecallGenTask


@sieval_task(
    name="ruler_vt_kshot_gen",
    display_name="RULER VT (few-shot, generative)",
    description="RULER variable tracking: trace multi-hop variable assignments.",
    eval_mode=EvalMode.GEN,
    n_shot=1,
    tags=("english", "open-ended", "long-context"),
    deps_group="ruler",
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="opencompass",
        url="https://github.com/open-compass/opencompass/blob/a4b54048ae8759fa342d3efa1df5b53865518804/opencompass/datasets/ruler/ruler_vt.py",
        notes="Synthesis + substring-recall scoring ported from OpenCompass RULER.",
    ),
)
class RulerVtFewShotGenTask(RulerRecallGenTask[RulerVtDatasetSample]):
    pass
