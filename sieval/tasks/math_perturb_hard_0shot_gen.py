"""
MATH-P-Hard 0-shot generative task.

The treatment arm of MATH-Perturb: 279 MATH level-5 problems edited so the
original solution method *no longer applies*. Every model the paper benchmarks
drops 10–25 points here relative to the unperturbed problems, which is the
result the benchmark exists to produce. Read against
``math_perturb_simple_0shot_gen`` over the same ``problem_id`` set, under the
same model and decoding.

Protocol, grading, measured fidelity and the traps are documented once in
:mod:`sieval.tasks._math_perturb_base`.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from sieval.core.tasks import EvalMode, ReferenceImpl, sieval_task
from sieval.datasets import MATHPerturbHardDatasetSample

from ._math_perturb_base import (
    MATH_PERTURB_REFERENCE_NOTES,
    MATH_PERTURB_UPSTREAM_URL,
    MathPerturbZeroShotGenTask,
)


@sieval_task(
    name="math_perturb_hard_0shot_gen",
    display_name="MATH-P-Hard (0-shot, generative)",
    description=(
        "MATH-P-Hard 0-shot CoT eval, scored overall and by seed split and subject."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended", "robustness"),
    deps_group="math",
    model_type="chat",
    status="stable",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="Kaffaljidhmah2/MATH-Perturb",
        url=MATH_PERTURB_UPSTREAM_URL,
        notes=(
            "MATH-P-Hard, 279 rows (164 train-seeded / 115 test-seeded), "
            "row-aligned with MATH-P-Simple by problem_id. Table 1 targets "
            "(All): o1-mini 78.49, Gemini-2.0-flash-thinking-exp 78.14, GPT-4o "
            "39.43, Qwen2.5-Math-7B-Instruct 27.24, Llama-3.1-8B-Instruct 10.04, "
            "Gemma-2-9b-it 11.83. The number this benchmark is for, and the one "
            "that is far below both siblings for every published row -- a Hard "
            "score near its Simple score means the harness, not the model. The "
            "seed-split cells carry the paper's second claim: the drop comes "
            "mainly from train-seeded problems, which the model may have "
            f"memorized. {MATH_PERTURB_REFERENCE_NOTES}"
        ),
    ),
)
class MATHPerturbHardZeroShotGenTask(
    MathPerturbZeroShotGenTask[MATHPerturbHardDatasetSample]
):
    pass
