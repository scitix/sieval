"""RULER FWE 0-shot base-model task (completion endpoint).

Same synthesis + substring-recall scoring as the chat task
:class:`~sieval.tasks.ruler.ruler_fwe_0shot_gen.RulerFweZeroShotGenTask`, but the
raw ``prompt`` is fed verbatim to a ``GenModel`` (completions API) and the model
continues it — faithful to original NVIDIA RULER's base-model evaluation.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    sieval_task,
)
from sieval.datasets import RulerFweDatasetSample
from sieval.tasks.ruler._base import RulerRecallBaseGenTask


@sieval_task(
    name="ruler_fwe_0shot_base_gen",
    display_name="RULER FWE (0-shot, base/completion)",
    description="RULER frequent words extraction: report the top-3 coded words.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended", "long-context"),
    deps_group="ruler",
    model_type="gen",
    reference_impl=ReferenceImpl(
        source="github",
        url="https://github.com/NVIDIA/RULER/blob/ab17b7853df4e0a30b78cd5d2b463ac7dff6ee13/scripts/data/synthetic/freq_words_extraction.py",
        notes="Original NVIDIA RULER evaluates base models via completion; "
        "substring-recall scoring.",
    ),
)
class RulerFweZeroShotBaseGenTask(RulerRecallBaseGenTask[RulerFweDatasetSample]):
    pass
