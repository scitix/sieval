"""RULER QA 0-shot base-model task (completion endpoint).

Same synthesis + ``string_match_part`` scoring as the chat task
:class:`~sieval.tasks.ruler.ruler_qa_0shot_gen.RulerQaZeroShotGenTask`, but the
raw ``input + answer_prefix`` string is fed verbatim to a ``GenModel``
(completions API) and the model continues it — faithful to original NVIDIA
RULER, which feeds the answer cue ("... Answer:") to a base model for
continuation rather than folding it into a chat turn. All pipeline logic lives in
:class:`~sieval.tasks.ruler._base.RulerQaBaseGenTask`.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    sieval_task,
)
from sieval.datasets import RulerQaDatasetSample
from sieval.tasks.ruler._base import RulerQaBaseGenTask


@sieval_task(
    name="ruler_qa_0shot_base_gen",
    display_name="RULER QA (0-shot, base/completion)",
    description="RULER multi-doc QA via completions: continue input+answer_prefix "
    "as raw text.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended", "long-context"),
    deps_group="ruler",
    model_type="gen",
    reference_impl=ReferenceImpl(
        source="github",
        url="https://github.com/NVIDIA/RULER/blob/ab17b7853df4e0a30b78cd5d2b463ac7dff6ee13/scripts/data/synthetic/qa.py",
        notes="Original NVIDIA RULER feeds raw input + answer_prefix to a base "
        "model via completion; scoring uses RULER's string_match_part.",
    ),
)
class RulerQaZeroShotBaseGenTask(RulerQaBaseGenTask[RulerQaDatasetSample]):
    pass
