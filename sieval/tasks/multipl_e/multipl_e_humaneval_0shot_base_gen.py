"""MultiPL-E HumanEval, base-model completion protocol.

Upstream's primary path: the prompt is a partial program, the model continues
it, and the completed program is compiled and run against the suite's
assertions. Protocol, grading and the traps are documented once in
:mod:`sieval.tasks.multipl_e._base`.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from typing import ClassVar

from sieval.core.tasks import EvalMode, ReferenceImpl, sieval_task
from sieval.datasets import MultiPLEHumanEvalDatasetSample

from ._base import COMPLETION_NOTES, MULTIPL_E_UPSTREAM_URL, MultiPLECompletionTask


@sieval_task(
    name="multipl_e_humaneval_0shot_base_gen",
    display_name="MultiPL-E HumanEval (0-shot, base generative)",
    description=(
        "HumanEval in 24 languages for base models, graded by compiling and "
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
        notes=COMPLETION_NOTES,
    ),
)
class MultiPLEHumanEvalZeroShotBaseGenTask(
    MultiPLECompletionTask[MultiPLEHumanEvalDatasetSample]
):
    suite: ClassVar[str] = "humaneval"
    eval_source: ClassVar[str] = "human-eval"
