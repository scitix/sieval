"""MultiPL-E MBPP, base-model completion protocol.

Same protocol as the HumanEval sibling over MBPP's problems, in the 23
languages upstream translated MBPP to (no Dart). Protocol, grading and the
traps are documented once in :mod:`sieval.tasks.multipl_e._base`.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from typing import ClassVar

from sieval.core.tasks import EvalMode, ReferenceImpl, sieval_task
from sieval.datasets import MultiPLEMbppDatasetSample

from ._base import (
    COMPLETION_NOTES,
    MBPP_SUITE_NOTES,
    MULTIPL_E_UPSTREAM_URL,
    MultiPLECompletionTask,
)


@sieval_task(
    name="multipl_e_mbpp_0shot_base_gen",
    display_name="MultiPL-E MBPP (0-shot, base generative)",
    description=(
        "MBPP in 23 languages for base models, graded by compiling and "
        "running the program."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("multilingual", "code-exec", "base-model"),
    model_type="gen",
    status="experimental",
    reference_kind="procedure",
    reference_impl=ReferenceImpl(
        source="MultiPL-E",
        url=MULTIPL_E_UPSTREAM_URL,
        notes=COMPLETION_NOTES + MBPP_SUITE_NOTES,
    ),
)
class MultiPLEMbppZeroShotBaseGenTask(
    MultiPLECompletionTask[MultiPLEMbppDatasetSample]
):
    suite: ClassVar[str] = "mbpp"
    eval_source: ClassVar[str] = "mbpp"
