"""RULER CWE (common words extraction) few-shot generative task.

The prompt is fully synthesized in ``RulerCweDataset.load()``, which (mirroring
upstream RULER's ``num_fewshot=1`` default) prepends one in-context
demonstration — hence ``n_shot=1``, not 0. This task is thin: send the prompt,
then score by substring recall (RULER ``string_match_all`` — the mean over
reference common words of whether each appears in the prediction). All pipeline
logic lives in :class:`~sieval.tasks.ruler._base.RulerRecallGenTask`.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    sieval_task,
)
from sieval.datasets import RulerCweDatasetSample
from sieval.tasks.ruler._base import RulerRecallGenTask


@sieval_task(
    name="ruler_cwe_kshot_gen",
    display_name="RULER CWE (few-shot, generative)",
    description="RULER common words extraction: report the most frequent words.",
    eval_mode=EvalMode.GEN,
    n_shot=1,
    tags=("english", "open-ended", "long-context"),
    deps_group="ruler",
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="NVIDIA/RULER",
        url="https://github.com/NVIDIA/RULER/blob/ab17b7853df4e0a30b78cd5d2b463ac7dff6ee13/scripts/eval/synthetic/constants.py",
        notes="This task mirrors RULER's scoring (string_match_all, vendored in "
        "community/ruler/eval). Prompt synthesis lives in the RulerCweDataset loader, "
        "ported from RULER's scripts/data/synthetic/common_words_extraction.py.",
    ),
)
class RulerCweFewShotGenTask(RulerRecallGenTask[RulerCweDatasetSample]):
    pass
