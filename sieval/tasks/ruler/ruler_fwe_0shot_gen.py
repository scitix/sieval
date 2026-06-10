"""RULER FWE (frequent words extraction) 0-shot generative task.

The prompt is fully synthesized in ``RulerFweDataset.load()``, so this task is
thin: send the prompt, then score by substring recall (RULER ``string_match_all``
— the mean over the three reference coded words of whether each appears in the
prediction). All pipeline logic lives in
:class:`~sieval.tasks.ruler._base.RulerRecallGenTask`.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    sieval_task,
)
from sieval.datasets import RulerFweDatasetSample
from sieval.tasks.ruler._base import RulerRecallGenTask


@sieval_task(
    name="ruler_fwe_0shot_gen",
    display_name="RULER FWE (0-shot, generative)",
    description="RULER frequent words extraction: report the top-3 coded words.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended", "long-context"),
    deps_group="ruler",
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="opencompass",
        url="https://github.com/open-compass/opencompass/blob/a4b54048ae8759fa342d3efa1df5b53865518804/opencompass/datasets/ruler/ruler_fwe.py",
        notes="Synthesis + substring-recall scoring ported from OpenCompass RULER.",
    ),
)
class RulerFweZeroShotGenTask(RulerRecallGenTask[RulerFweDatasetSample]):
    pass
