"""
PlatinumBench SingleOp 0-shot generative task.

Scores the cleaned revision of SingleOp: 150 of 159 questions survive
PlatinumBench's re-annotation. Single-operation arithmetic word problems — the
easiest of the five math subsets, and therefore the one where a nonzero error
count is hardest to explain away as difficulty.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from sieval.core.tasks import EvalMode, ReferenceImpl, sieval_task

from ._base import (
    PLATINUM_REFERENCE_NOTES,
    PLATINUM_UPSTREAM_URL,
    PlatinumMathGenTask,
)


@sieval_task(
    name="platinum_singleop_0shot_gen",
    display_name="PlatinumBench SingleOp (0-shot, generative)",
    description=(
        "Label-noise-cleaned SingleOp: 150 single-operation arithmetic problems."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "math-word-problems", "open-ended"),
    model_type="chat",
    status="stable",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="MadryLab/platinum-benchmarks",
        url=PLATINUM_UPSTREAM_URL,
        notes=(
            "Subset 'singleop' of madrylab/platinum-bench: 150 rows kept of 159 "
            f"(9 rejected). {PLATINUM_REFERENCE_NOTES}"
        ),
    ),
)
class PlatinumSingleOpZeroShotGenTask(PlatinumMathGenTask):
    subset = "singleop"
