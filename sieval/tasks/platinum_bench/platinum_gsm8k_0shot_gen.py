"""
PlatinumBench GSM8K 0-shot generative task.

Scores the cleaned revision of GSM8K's test split: 268 of 300 sampled questions
survive PlatinumBench's re-annotation. Distinct from ``gsm8k_0shot_gen``, which
runs the full uncleaned split through the DeepSeek-Math protocol — different
questions, different prompt, different extractor, not comparable.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from sieval.core.tasks import EvalMode, ReferenceImpl, sieval_task

from ._base import (
    PLATINUM_REFERENCE_NOTES,
    PLATINUM_UPSTREAM_URL,
    PlatinumMathGenTask,
)


@sieval_task(
    name="platinum_gsm8k_0shot_gen",
    display_name="PlatinumBench GSM8K (0-shot, generative)",
    description="Label-noise-cleaned GSM8K: 268 grade-school math word problems.",
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
            "Subset 'gsm8k' of madrylab/platinum-bench: 268 rows kept of 300 "
            f"(32 rejected). {PLATINUM_REFERENCE_NOTES}"
        ),
    ),
)
class PlatinumGSM8KZeroShotGenTask(PlatinumMathGenTask):
    subset = "gsm8k"
