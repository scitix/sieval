"""
PlatinumBench MultiArith 0-shot generative task.

Scores the cleaned revision of MultiArith: 170 of 174 questions survive
PlatinumBench's re-annotation. Multi-step arithmetic word problems, so a
frontier model's remaining errors here are the interesting ones.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from sieval.core.tasks import EvalMode, ReferenceImpl, sieval_task

from ._base import (
    PLATINUM_REFERENCE_NOTES,
    PLATINUM_UPSTREAM_URL,
    PlatinumMathGenTask,
)


@sieval_task(
    name="platinum_multiarith_0shot_gen",
    display_name="PlatinumBench MultiArith (0-shot, generative)",
    description=(
        "Label-noise-cleaned MultiArith: 170 multi-step arithmetic word problems."
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
            "Subset 'multiarith' of madrylab/platinum-bench: 170 rows kept of 174 "
            f"(4 rejected). {PLATINUM_REFERENCE_NOTES}"
        ),
    ),
)
class PlatinumMultiArithZeroShotGenTask(PlatinumMathGenTask):
    subset = "multiarith"
