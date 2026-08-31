"""
MATH-P-Simple 0-shot generative task.

The control arm of MATH-Perturb: 279 MATH level-5 problems edited so the
*original* solution method still applies. Read against
``math_perturb_hard_0shot_gen`` over the same ``problem_id`` set — the gap
between the two is the benchmark's point, and it is only interpretable when both
ran under the same model and the same decoding.

Protocol, grading, measured fidelity and the traps are documented once in
:mod:`sieval.tasks._math_perturb_base`.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from sieval.core.tasks import EvalMode, ReferenceImpl, sieval_task
from sieval.datasets import MATHPerturbSimpleDatasetSample

from ._math_perturb_base import (
    MATH_PERTURB_REFERENCE_NOTES,
    MATH_PERTURB_UPSTREAM_URL,
    MathPerturbZeroShotGenTask,
)


@sieval_task(
    name="math_perturb_simple_0shot_gen",
    display_name="MATH-P-Simple (0-shot, generative)",
    description=(
        "MATH-P-Simple 0-shot CoT eval, scored overall and by seed split and subject."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended", "robustness"),
    deps_group="math",
    model_type="chat",
    status="experimental",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="Kaffaljidhmah2/MATH-Perturb",
        url=MATH_PERTURB_UPSTREAM_URL,
        notes=(
            "MATH-P-Simple, 279 rows (164 train-seeded / 115 test-seeded). "
            "Table 1 targets (All): o1-mini 94.98, Gemini-2.0-flash-thinking-exp "
            "91.04, GPT-4o 62.01, Qwen2.5-Math-7B-Instruct 51.61, "
            "Llama-3.1-8B-Instruct 31.54, Gemma-2-9b-it 27.60. This is the arm "
            "that should NOT move much: the paper's finding is that most models "
            "lose only a little here versus Original, while losing 10-25 points "
            "on MATH-P-Hard, so a large Simple drop is evidence about the "
            "harness rather than about the model. Two published rows invert even "
            "that (MAmmoTH2-8B 17.92 Simple vs 12.90 Original, "
            "Phi-3.5-mini-instruct 28.67 vs 26.16), so non-monotonicity alone is "
            f"not a defect signal. {MATH_PERTURB_REFERENCE_NOTES}"
        ),
    ),
)
class MATHPerturbSimpleZeroShotGenTask(
    MathPerturbZeroShotGenTask[MATHPerturbSimpleDatasetSample]
):
    pass
