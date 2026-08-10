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

import asyncio
from typing import override

from loguru import logger

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
from sieval.core.tasks.metrics import (
    DENOMINATOR_FIELD,
    DENOMINATOR_REQUESTED,
    SCORE_KEY_FIELD,
    first_rollout_correct,
    health_metrics,
    sampling_report,
)
from sieval.core.utils.offload import GRADE_TIMEOUT, run_cpu_bound
from sieval.datasets import HendrycksMathDatasetSample

from ._math_verify import normalize_vote

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
            "extract_math_few_shot_cot_answer (list-valued) + eval_math/math_equal, "
            "vendored in sieval.community.deepseek_math with ONE divergence, "
            "taken for execution safety rather than as a repair: upstream's "
            "symbolic_equal hands model output to a bare parse_expr and, when "
            "parsing fails, to N as raw text -- both sympify with "
            "a namespace carrying __builtins__, so an answer of "
            "__import__('os').system(...) runs while the sample still grades "
            "wrong (not simplify(a-b), which raises TypeError first). Here the "
            "parse is guarded and an unparseable answer refuses the comparison "
            "instead. MEASURED DIVERGENCE: ZERO -- replaying the "
            "full 5000-sample stored run (Qwen2.5-72B) through upstream's "
            "reading and this one gives identical verdicts on every sample, "
            "61.2600 either way, and 60.0200 either way with parse_latex "
            "disabled -- the adversarial case, since it sends every comparison "
            "down the guarded path (1622 of 5000 fall through to the refusal). "
            "The exponent pre-parse also declines a right-nested ** tower and an "
            "exponent above 10000, which upstream evaluates; parse_latex reads "
            "those spellings first, so the zero covers them too. "
            "NOTE ON parse_latex: sympy 1.14's LaTeX grammar needs "
            "antlr4-python3-runtime 4.11.0, pinned in the [math] extra. Without "
            "it parse_latex raises into upstream's bare except and the symbolic "
            "path silently never runs, costing 1.24 pp (61.26 -> 60.02) on this "
            "benchmark with no signal in any log -- the two figures above are "
            "the same run measured with and without it."
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
        # `float | str`: the report carries `score_key`, which names a column
        # rather than measuring one.
        dict[str, float | str],
    ]
):
    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        *,
        k: int = 1,
        n: int = 1,
        stop: tuple[str, ...] = tuple(STOP_WORDS),
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        if k > n:
            raise ValueError(
                f"pass@{k} needs at least {k} sample(s) per problem, got n={n}."
            )
        self._k = k
        self._n = n
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
        # `n` is the sampling budget `k` was validated against, so it has to
        # reach the model (sieval/tasks/CLAUDE.md, "n_shot vs k").
        if self._stop:
            return await self.model.agenerate(
                pre["prompt"], n=self._n, stop=list(self._stop)
            )
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        # Empty extraction -> None so `extracted` reports the miss; feedback
        # restores "" so the grader sees exactly what it saw pre-migration.
        return build_prediction_record(
            [
                extract_math_few_shot_cot_answer(ctx.raw_sample["problem"], text, "cot")
                or None
                for text in inf.texts
            ]
        )

    @override
    async def feedback(self, post, ctx):
        reference = extract_math_answer(
            ctx.raw_sample["problem"], ctx.raw_sample["solution"], "cot"
        )
        # Concurrent, not sequential: each grade is an offloaded CPU-bound call
        # with its own GRADE_TIMEOUT, so awaiting them in turn makes a sample's
        # worst case n x the timeout instead of one.
        verdicts = await asyncio.gather(
            *(self._grade(rollout, reference, ctx) for rollout in post["rollouts"])
        )
        rollouts = [
            build_rollout_judgement(rollout["index"], verdict)
            for rollout, verdict in zip(post["rollouts"], verdicts, strict=True)
        ]
        return True, build_judgement_record(reference, rollouts)

    async def _grade(self, rollout, reference, ctx) -> bool:
        prediction = rollout.get("prediction") or ""
        # `math_equal` runs `parse_latex` + `simplify`: ~11 ms typical, 1.7 s
        # worst case — measured on *reference* data, and `simplify` on arbitrary
        # model output has no ceiling. Reached with `timeout=False`, so nothing
        # else bounds it: criterion 2 in `core/utils/offload.py`.
        try:
            correct = bool(
                await run_cpu_bound(
                    eval_math,
                    {"prediction": prediction, "answer": reference},
                    timeout=GRADE_TIMEOUT,
                )
            )
        except TimeoutError:
            # An answer that cannot be graded is a wrong answer, not a failed
            # run — the contract every sibling math grader keeps. Letting this
            # propagate would land the sample in `fails` instead, which reads as
            # an infrastructure failure and is one of the signals a run is
            # promoted on. The accuracy is identical either way (`report` counts
            # fails in the denominator), so the only thing at stake is whether
            # the number means what it says.
            logger.warning(
                "Grading sample {} exceeded {}s and was scored wrong; the "
                "prediction is likely a shape `simplify` cannot bound.",
                ctx.sample_id,
                GRADE_TIMEOUT,
            )
            correct = False
        return bool(correct)

    @override
    async def report(self, finals, fails):
        # Accuracy over the full requested set (finals + fails), matching the
        # gsm8k-0shot-gen DeepSeek-Math sibling and DeepSeek's full-set accuracy:
        # a pipeline failure counts as wrong, not as an excluded sample.
        total = len(finals) + len(fails)
        # First-rollout, because that is what DeepSeek-Math published (one
        # greedy draw). The sampling metrics below never touch it.
        accuracy = 100 * first_rollout_correct(finals) / total if total else 0.0
        metrics: dict[str, float | str] = {
            "score": accuracy,
            "fails": len(fails),
            "accuracy": accuracy,
            SCORE_KEY_FIELD: "accuracy",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
        }
        # Outside the gate: extraction health is a fact about the parser,
        # not the draw, and n=1 is where a stopped extractor hides longest.
        metrics |= health_metrics(finals)
        if self._n <= 1:
            return metrics
        return metrics | sampling_report(
            finals,
            n=self._n,
            k=self._k,
            denominator=total,
            normalize=normalize_vote,
        )
