"""RULER NIAH 0-shot generative task.

The prompt is fully synthesized in ``RulerNiahDataset.load()``, so this task is
thin: pass the prompt to the model, then score by substring recall — the mean
over reference answers of whether each appears (case-insensitively) in the
prediction. Mirrors OpenCompass ``RulerNiahEvaluator``. All pipeline logic lives
in :class:`~sieval.tasks.ruler._base.RulerRecallGenTask`.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    sieval_task,
)
from sieval.datasets import RulerNiahDatasetSample
from sieval.tasks.ruler._base import RulerRecallGenTask


@sieval_task(
    name="ruler_niah_0shot_gen",
    display_name="RULER NIAH (0-shot, generative)",
    description="RULER needle-in-a-haystack: retrieve magic values from long context.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended", "long-context"),
    deps_group="ruler",
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="opencompass",
        url="https://github.com/open-compass/opencompass/blob/a4b54048ae8759fa342d3efa1df5b53865518804/opencompass/datasets/ruler/ruler_niah.py",
        notes="Synthesis + substring-recall scoring ported from OpenCompass RULER.",
    ),
)
class RulerNiahZeroShotGenTask(RulerRecallGenTask[RulerNiahDatasetSample]):
    pass
