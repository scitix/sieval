"""AIME 2025 with a Python execution tool, text (fenced-block) protocol.

The same 30 problems, the same prompt template, the same answer pattern and the
same verifier as the no-tool task -- the model may additionally run Python while
solving. Reported as its own task because the affordance changes what is being
measured, and the pair is only comparable because nothing else differs.

No upstream publishes a tool-augmented AIME column, so this task's anchor is the
paired difference against its no-tool sibling rather than an external number.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from typing import override

from sieval.community.simple_evals.common import ANSWER_PATTERN
from sieval.community.simple_evals.math_eval import QUERY_TEMPLATE
from sieval.core.tasks import EvalMode, ReferenceImpl, sieval_task
from sieval.datasets import AIME2025DatasetSample

from ._math_tool_base import MathToolTask


@sieval_task(
    name="aime_2025_0shot_gen_tool",
    display_name="AIME 2025 (0-shot, generative, code tool)",
    # 82 chars: the registry caps a description at 100, and the decorator
    # validates it while the class is being DECORATED -- i.e. at import time,
    # long before any record is read. Not a warning at any point.
    description=(
        "AIME 2025 with a Python tool during the solve, graded exactly as the "
        "no-tool task."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended", "tool-use"),
    deps_group="math",
    model_type="chat",
    reference_kind="value",
    status="experimental",
    reference_impl=ReferenceImpl(
        source="simple-evals",
        url="https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/math_eval.py",
        notes=(
            "Prompt template, answer extraction and grading are simple-evals', "
            "identical to aime_2025_0shot_gen. The tool affordance is not "
            "upstream's: no upstream publishes a tool-augmented AIME column, so "
            "this task is anchored on the paired difference against the no-tool "
            "task (same data, same extractor, same verifier, same n) rather than "
            "on an external score. The protocol prompt is pinned in "
            "_math_tool_base.TOOL_SYSTEM_PROMPT; editing it invalidates any "
            "previously measured difference. Execution runs on the code-evaluator "
            "/code-runs route, one fresh interpreter per call."
        ),
    ),
)
class AIME2025ZeroShotGenToolTask(MathToolTask[AIME2025DatasetSample]):
    query_template = QUERY_TEMPLATE
    answer_pattern = ANSWER_PATTERN

    @override
    def _question(self, raw) -> str:
        return raw["question"]

    @override
    def _reference(self, raw) -> str:
        return raw["answer"]
