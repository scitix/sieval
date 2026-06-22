"""RULER QA 0-shot generative task (chat endpoint).

The prompt (question + distractor documents) is fully synthesized in
``RulerQaDataset.load()``, so this task is thin: send the prompt, then score with
RULER's own ``string_match_part`` metric (best-match: any reference answer present
counts — ``max`` over references, vs the recall ``string_match_all`` mean used by
NIAH/VT/CWE/FWE). All pipeline logic lives in
:class:`~sieval.tasks.ruler._base.RulerQaGenTask`.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    sieval_task,
)
from sieval.datasets import RulerQaDatasetSample
from sieval.tasks.ruler._base import RulerQaGenTask


@sieval_task(
    name="ruler_qa_0shot_gen",
    display_name="RULER QA (0-shot, generative)",
    description="RULER multi-doc QA: answer over many distractor documents.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended", "long-context"),
    deps_group="ruler",
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="NVIDIA/RULER",
        url="https://github.com/NVIDIA/RULER/blob/ab17b7853df4e0a30b78cd5d2b463ac7dff6ee13/scripts/eval/synthetic/constants.py",
        notes="This task mirrors RULER's scoring (string_match_part, vendored in "
        "community/ruler/eval). Prompt synthesis lives in the RulerQaDataset loader, "
        "ported from RULER's scripts/data/synthetic/qa.py.",
    ),
)
class RulerQaZeroShotGenTask(RulerQaGenTask[RulerQaDatasetSample]):
    pass
