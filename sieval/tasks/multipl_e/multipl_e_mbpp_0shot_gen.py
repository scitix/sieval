"""MultiPL-E MBPP, chat-model protocol.

Same protocol as the HumanEval chat sibling over MBPP's problems, in the 23
languages upstream translated MBPP to (no Dart). The graded program is built
from the model's own copy of the prefix, not the dataset's — see
:mod:`sieval.tasks.multipl_e._base` for why that is upstream's rule.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from typing import ClassVar

from sieval.core.tasks import EvalMode, ReferenceImpl, sieval_task
from sieval.datasets import MultiPLEMbppDatasetSample

from ._base import (
    CHAT_NOTES,
    CHAT_UPSTREAM_URL,
    MBPP_SUITE_NOTES,
    MultiPLEChatTask,
)


@sieval_task(
    name="multipl_e_mbpp_0shot_gen",
    display_name="MultiPL-E MBPP (0-shot, generative)",
    description=(
        "MBPP in 23 languages for chat models, graded by compiling and "
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
        notes=CHAT_NOTES + MBPP_SUITE_NOTES,
    ),
)
class MultiPLEMbppZeroShotGenTask(MultiPLEChatTask[MultiPLEMbppDatasetSample]):
    suite: ClassVar[str] = "mbpp"
    eval_source: ClassVar[str] = "mbpp"
