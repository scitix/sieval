"""RULER NIAH 0-shot base-model task (completion endpoint).

Same synthesis + substring-recall scoring as the chat task
:class:`~sieval.tasks.ruler.ruler_niah_0shot_gen.RulerNiahZeroShotGenTask`, but
the raw ``prompt`` is fed verbatim to a ``GenModel`` (completions API) and the
model continues it — faithful to original NVIDIA RULER, which evaluates base
models via text continuation rather than a chat turn.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    sieval_task,
)
from sieval.datasets import RulerNiahDatasetSample
from sieval.tasks.ruler._base import RulerRecallBaseGenTask


@sieval_task(
    name="ruler_niah_0shot_base_gen",
    display_name="RULER NIAH (0-shot, base/completion)",
    description="RULER needle-in-a-haystack: retrieve magic values from long context.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended", "long-context"),
    deps_group="ruler",
    model_type="gen",
    reference_impl=ReferenceImpl(
        source="github",
        url="https://github.com/NVIDIA/RULER/blob/ab17b7853df4e0a30b78cd5d2b463ac7dff6ee13/scripts/data/synthetic/niah.py",
        notes="Original NVIDIA RULER evaluates base models via completion "
        "(raw input + answer_prefix continuation); substring-recall scoring.",
    ),
)
class RulerNiahZeroShotBaseGenTask(RulerRecallBaseGenTask[RulerNiahDatasetSample]):
    pass
