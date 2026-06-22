"""RULER NIAH 0-shot generative task.

The prompt is fully synthesized in ``RulerNiahDataset.load()``, so this task is
thin: pass the prompt to the model, then score by substring recall (RULER
``string_match_all`` — the mean over reference answers of whether each appears,
case-insensitively, in the prediction). All pipeline logic lives in
:class:`~sieval.tasks.ruler._base.RulerRecallGenTask`.

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
        source="NVIDIA/RULER",
        url="https://github.com/NVIDIA/RULER/blob/ab17b7853df4e0a30b78cd5d2b463ac7dff6ee13/scripts/eval/synthetic/constants.py",
        notes="This task mirrors RULER's scoring (string_match_all, vendored in "
        "community/ruler/eval). Prompt synthesis lives in the RulerNiahDataset "
        "loader, ported from RULER's scripts/data/synthetic/niah.py.",
    ),
)
class RulerNiahZeroShotGenTask(RulerRecallGenTask[RulerNiahDatasetSample]):
    pass
