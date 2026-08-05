"""
Hendrycks MATH few-shot base-model task, aligned with DeepSeek-Math evaluation.

Strict port of DeepSeek-Math's ``math-cot-test`` path (pinned commit
``b8b0f8ce``): the Minerva 4-shot CoT prompt (``MinervaMathPrompt``,
``Problem:\\n...\\n\\nSolution:\\n``, stop ``["\\nProblem:"]``); DeepSeek's own
answer extraction (``extract_math_few_shot_cot_answer`` ->
``extract_math_answer`` -> ``extract_answer`` + DeepSeek ``strip_string``); and
``eval_math`` -> ``is_correct`` -> ``math_equal``. Extraction returns a LIST of
answers (multi-answer questions) and ``eval_math`` set-matches the predicted and
reference answer lists. All of this lives verbatim in
``sieval.community.deepseek_math``.

The shot count is fixed at 4 (the exemplars are a single baked-in prompt string
upstream; there is no per-shot knob), and the reference answer is extracted from
the ``solution`` column the same way DeepSeek's ``process_math_test`` does.

Deviation from upstream ``process_math_test``: reference extraction is not
wrapped in a ``try/except`` that drops the sample — a failed boxed extraction
counts as a wrong answer, not a dropped one. Immaterial in practice (all 5,000
test rows carry a ``\\boxed`` answer). Extraction/equivalence deviations
(dropped debug prints, unused ``timeout`` path) are in the community docstring.

Repro decoding (greedy, matching DeepSeek's ``run_cot_eval.py``): temperature=0,
top_p=1, max_gen_toks=1024.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from typing import override

from sieval.community.deepseek_math import (
    STOP_WORDS,
    eval_math,
    extract_math_answer,
    extract_math_few_shot_cot_answer,
    format_prompt,
)
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    ReferenceImpl,
    Task,
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
    sieval_task,
)
from sieval.datasets import HendrycksMathDatasetSample

N_SHOT = 4


@sieval_task(
    name="hendrycks_math_kshot_base_gen",
    display_name="Hendrycks MATH (few-shot, base generative)",
    description="Full Hendrycks MATH DeepSeek-Math 4-shot CoT base-model eval.",
    eval_mode=EvalMode.GEN,
    n_shot=N_SHOT,
    tags=("english", "open-ended", "base-model"),
    deps_group="math",
    model_type="gen",
    reference_impl=ReferenceImpl(
        source="DeepSeek-Math",
        url="https://github.com/deepseek-ai/DeepSeek-Math/tree/b8b0f8ce093d80bf8e9a641e44142f06d092c305/evaluation",
        notes=(
            "math-cot-test path: MinervaMathPrompt 4-shot, "
            "extract_math_few_shot_cot_answer (list-valued) + eval_math/math_equal."
        ),
    ),
)
class HendrycksMathFewShotBaseGenTask(
    Task[
        HendrycksMathDatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        dict[str, float],
    ]
):
    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        *,
        stop: tuple[str, ...] = tuple(STOP_WORDS),
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._stop = stop

    @override
    async def preprocess(self, raw, ctx):
        return build_prompt_record(
            format_prompt(raw["problem"], ""),
            # The gold is derived from the solution at judgement time (the same
            # extractor upstream uses), so it is recorded there, not here.
        )

    @override
    async def infer(self, pre, ctx):
        if self._stop:
            return await self.model.agenerate(pre["prompt"], stop=list(self._stop))
        return await self.model.agenerate(pre["prompt"])

    @override
    async def postprocess(self, inf, ctx):
        text = inf.texts[0] if inf.texts else ""
        # Empty extraction -> None so `extracted` reports the miss; feedback
        # restores "" so the grader sees exactly what it saw pre-migration.
        return build_prediction_record(
            [
                extract_math_few_shot_cot_answer(ctx.raw_sample["problem"], text, "cot")
                or None
            ]
        )

    @override
    async def feedback(self, post, ctx):
        reference = extract_math_answer(
            ctx.raw_sample["problem"], ctx.raw_sample["solution"], "cot"
        )
        prediction = post["rollouts"][0]["prediction"] or ""
        correct = bool(eval_math({"prediction": prediction, "answer": reference}))
        return True, build_judgement_record(
            reference, [build_rollout_judgement(0, correct)]
        )

    @override
    async def report(self, finals, fails):
        # Accuracy over the full requested set (finals + fails), matching the
        # gsm8k-0shot-gen DeepSeek-Math sibling and DeepSeek's full-set accuracy:
        # a pipeline failure counts as wrong, not as an excluded sample.
        total = len(finals) + len(fails)
        if total == 0:
            return {"score": 0.0, "fails": len(fails), "accuracy": 0.0}
        correct_num = sum(
            1 for ctx in finals if ctx.feedback_result["rollouts"][0]["correct"]
        )
        accuracy = 100 * correct_num / total
        return {
            "score": accuracy,
            "fails": len(fails),
            "accuracy": accuracy,
        }
