"""MultiPL-E HumanEval, chat-model protocol.

Follows upstream's ``dataset_builder/chat_completions.py``, which exists
because the original completion prompts "are not compatible with chat-only
models". The model is asked for the whole program back and the graded program
is built from ITS copy of the prefix, not the dataset's. Protocol, grading and
the traps are documented once in :mod:`sieval.tasks.multipl_e._base`.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from typing import ClassVar

from sieval.core.tasks import EvalMode, ReferenceImpl, sieval_task
from sieval.datasets import MultiPLEHumanEvalDatasetSample

from ._base import CHAT_NOTES, CHAT_UPSTREAM_URL, MultiPLEChatTask


@sieval_task(
    name="multipl_e_humaneval_0shot_gen",
    display_name="MultiPL-E HumanEval (0-shot, generative)",
    description=(
        "HumanEval in 24 languages for chat models, graded by compiling and "
        "running the program."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("multilingual", "code-exec"),
    model_type="chat",
    status="experimental",
    reference_kind="procedure",
    reference_impl=ReferenceImpl(
        source="MultiPL-E",
        url=CHAT_UPSTREAM_URL,
        notes=CHAT_NOTES,
    ),
)
class MultiPLEHumanEvalZeroShotGenTask(
    MultiPLEChatTask[MultiPLEHumanEvalDatasetSample]
):
    suite: ClassVar[str] = "humaneval"
    eval_source: ClassVar[str] = "human-eval"
