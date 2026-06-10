"""RULER QA 0-shot generative task (chat endpoint).

The prompt (question + distractor documents) is fully synthesized in
``RulerQaDataset.load()``, so this task is thin: send the prompt, then score with
RULER's own ``string_match_part`` metric (best-match: any reference answer present
counts — ``max`` over references, vs the recall ``string_match_all`` mean used by
NIAH/VT/CWE/FWE). All pipeline logic lives in
:class:`~sieval.tasks.ruler._base.RulerQaGenTask`; see its docstring for the
chat-vs-completion endpoint split (the completion variant is
``RulerQaZeroShotBaseGenTask``).

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
        source="opencompass",
        url="https://github.com/open-compass/opencompass/blob/a4b54048ae8759fa342d3efa1df5b53865518804/opencompass/datasets/ruler/ruler_qa.py",
        notes="Scoring uses RULER's own string_match_part (vendored in "
        "sieval.community.ruler.eval.constants); synthesis ported from OpenCompass.",
    ),
)
class RulerQaZeroShotGenTask(RulerQaGenTask[RulerQaDatasetSample]):
    pass
