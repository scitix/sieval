"""
PlatinumBench SVAMP 0-shot generative task.

Scores the cleaned revision of SVAMP: 265 of 300 sampled questions survive
PlatinumBench's re-annotation. SVAMP probes robustness by varying the surface
form of simple arithmetic word problems, which is also why it had the most
rejected rows of the five math subsets (35).

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from sieval.core.tasks import EvalMode, ReferenceImpl, sieval_task

from ._base import (
    PLATINUM_REFERENCE_NOTES,
    PLATINUM_UPSTREAM_URL,
    PlatinumMathGenTask,
)


@sieval_task(
    name="platinum_svamp_0shot_gen",
    display_name="PlatinumBench SVAMP (0-shot, generative)",
    description="Label-noise-cleaned SVAMP: 265 varied arithmetic word problems.",
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
            "Subset 'svamp' of madrylab/platinum-bench: 265 rows kept of 300 "
            f"(35 rejected). {PLATINUM_REFERENCE_NOTES}"
        ),
    ),
)
class PlatinumSVAMPZeroShotGenTask(PlatinumMathGenTask):
    subset = "svamp"
