"""
PlatinumBench SingleEq 0-shot generative task.

Scores the cleaned revision of SingleEq: 100 of 109 questions survive
PlatinumBench's re-annotation. Word problems solvable by a single equation. The
dataset config is spelled ``singleq`` even though upstream's own comment calls it
``singleeq``; the config name is what selects the data, so it wins here.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from sieval.core.tasks import EvalMode, ReferenceImpl, sieval_task

from ._base import (
    PLATINUM_REFERENCE_NOTES,
    PLATINUM_UPSTREAM_URL,
    PlatinumMathGenTask,
)


@sieval_task(
    name="platinum_singleq_0shot_gen",
    display_name="PlatinumBench SingleEq (0-shot, generative)",
    description="Label-noise-cleaned SingleEq: 100 single-equation math word problems.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "math-word-problems", "open-ended"),
    model_type="chat",
    status="stable",
    reference_impl=ReferenceImpl(
        source="MadryLab/platinum-benchmarks",
        url=PLATINUM_UPSTREAM_URL,
        notes=(
            "Subset 'singleq' of madrylab/platinum-bench: 100 rows kept of 109 "
            f"(9 rejected). {PLATINUM_REFERENCE_NOTES}"
        ),
    ),
)
class PlatinumSingleEqZeroShotGenTask(PlatinumMathGenTask):
    subset = "singleq"
