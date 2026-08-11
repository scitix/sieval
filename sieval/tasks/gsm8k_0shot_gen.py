"""
GSM8K 0-shot generative task, aligned with DeepSeek-Math evaluation.

Strict port of DeepSeek-Math's ``gsm8k-test`` zero-shot (CoT, instruct/chat)
path (pinned commit ``b8b0f8ce``, ``configs/zero_shot_test_configs.json``):

* Prompt (``run_subset_parallel.py::markup_question``, language="en", task="cot"):
  the user turn is ``{question}`` followed by ``"\\nPlease reason step by step,
  and put your final answer within \\boxed{}."``; the serving backend applies the
  model's own chat template (``apply_chat_template``, add_generation_prompt=True
  — see ``replicate/predict_instruct.py``).
* Answer extraction: DeepSeek's ``extract_last_single_answer`` is exactly
  ``extract_answer(reasoning, exhaust=False)`` — last ``\\boxed{...}`` if present,
  else text after ``"he answer is"``, else the last number; then ``strip_string``
  normalization. We call ``extract_answer(..., exhaust=False)`` directly.
* Scoring: DeepSeek's ``eval_last_single_answer`` is ``is_correct`` (numeric
  isclose with %-variants, then sympy symbolic fallback). We call ``is_correct``
  directly. ``score`` is this accuracy.

All extraction/scoring lives verbatim in ``sieval.community.deepseek_math``
(vendored byte-faithfully from DeepSeek-Math's ``answer_extraction.py`` /
``eval_utils.py`` / ``eval_script.py`` at the pinned commit).

Deviations from the DeepSeek-Math repo (documented, not silent):

* Gold answer: DeepSeek's bundled ``datasets/gsm8k/test.jsonl`` stores ``answer``
  as the bare post-``####`` number. This task loads ``openai/gsm8k`` (the
  GSM8KDataset source), so the gold is derived the same way ``process_gsm8k_test``
  does: ``answer.split("####")[-1].strip()`` with commas removed. Questions are
  identical.
* The chat template is applied by the inference backend (sglang/vLLM serving the
  instruct checkpoint) rather than in-process, as in DeepSeek's harness.

Comparison target: DeepSeek-LLM-7B-Chat GSM8K = 63.0 (DeepSeek LLM report,
Table 6, 0-shot). That number is for DeepSeek-LLM-7B-Chat while this pipeline is
DeepSeek-Math's; both share the answer-extraction lineage. The model under test,
its prompt rendering, and its chat template govern how close the score lands.

Repro decoding (model-layer assets — set via ``models:`` / ``infer_args``, not
in this code): greedy ``temperature=0``, ``top_p=1.0``, ``max_tokens=1024``,
stop = the model's EOS only (DeepSeek's ``run_cot_eval.py`` SamplingParams for
zero-shot CoT).

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import asyncio
from typing import override

from loguru import logger

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
from sieval.datasets import GSM8KDatasetSample

from ._math_verify import normalize_vote

# Verbatim from run_subset_parallel.py::markup_question (language="en",
# task="cot"): f"{content}\nPlease reason step by step, and put your final
# answer within " + "\\boxed{}."
COT_INSTRUCTION = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)


def _gold_answer(answer: str) -> str:
    # DeepSeek process_gsm8k_test gold: item['answer'].replace(',', ''); for the
    # openai/gsm8k schema that bare number is answer.split('####')[-1].strip().
    return answer.split("####")[-1].strip().replace(",", "")


@sieval_task(
    name="gsm8k_0shot_gen",
    display_name="GSM8K (0-shot, generative)",
    description="GSM8K 0-shot chat-model eval aligned with the DeepSeek-Math pipeline.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "math-word-problems", "open-ended"),
    deps_group="math",
    model_type="chat",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="deepseek-ai/DeepSeek-Math",
        url=(
            "https://github.com/deepseek-ai/DeepSeek-Math/tree/b8b0f8ce093d80bf8e9a641e44142f06d092c305/evaluation"
        ),
        notes=(
            "gsm8k-test zero-shot CoT protocol: user turn = question + "
            '"Please reason step by step, and put your final answer within '
            '\\boxed{}.", chat template applied by the serving backend; '
            "extract_answer(exhaust=False) (= extract_last_single_answer) and "
            "is_correct/math_equal (= eval_last_single_answer) scoring are "
            "vendored byte-for-byte in sieval.community.deepseek_math, with ONE "
            "divergence, taken for execution safety rather than as a repair: "
            "upstream's symbolic_equal hands model output to a bare parse_expr "
            "and, when parsing fails, to N as raw text -- both sympify with a "
            "namespace carrying __builtins__, so an answer of "
            "__import__('os').system(...) runs while the sample still grades "
            "wrong (not simplify(a-b), which raises TypeError first). Here the "
            "parse is guarded and an unparseable answer refuses the comparison "
            "instead. MEASURED DIVERGENCE: ZERO -- replaying "
            "this benchmark's full 1319-sample stored run (deepseek-llm-7b-chat) "
            "through upstream's reading and this one gives identical verdicts on "
            "every sample, 63.3813 either way, and 63.3055 either way with "
            "parse_latex disabled (the case that forces every comparison down "
            "the guarded path). The exponent pre-parse also declines a "
            "right-nested ** tower and an exponent above 10000, which upstream "
            "evaluates; parse_latex reads those spellings first, so the zero "
            "covers them too. Gold "
            "derived from openai/gsm8k like process_gsm8k_test "
            "(answer.split('####')[-1], commas removed). NOTE ON parse_latex: "
            "sympy 1.14's LaTeX grammar needs antlr4-python3-runtime 4.11.0, "
            "pinned in the [math] extra. Without it parse_latex raises into "
            "upstream's bare except and the symbolic path silently never runs -- "
            "worth 0.08 pp here and 1.24 pp on MATH, so an environment missing "
            "it scores lower for a reason no log reports."
        ),
    ),
)
class GSM8KZeroShotGenTask(
    Task[
        GSM8KDatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        # `float | str`: the report carries `score_key`, which names a column
        # rather than measuring one.
        dict[str, float | str],
    ]
):
    def __init__(self, dataset, model, name: str | None = None, k: int = 1, n: int = 1):
        super().__init__(dataset=dataset, model=model, name=name)
        if k > n:
            raise ValueError(
                f"pass@{k} needs at least {k} sample(s) per problem, got n={n}."
            )
        self._k = k
        self._n = n

    @override
    async def preprocess(self, raw, ctx):
        return build_prompt_record(
            [
                {"role": "user", "content": raw["question"] + COT_INSTRUCTION},
            ],
            reference=_gold_answer(raw["answer"]),
        )

    @override
    async def infer(self, pre, ctx):
        # `n` is the sampling budget `k` was validated against, so it has to
        # reach the model (sieval/tasks/CLAUDE.md, "n_shot vs k").
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        from sieval.community.deepseek_math import extract_answer

        # extract_answer returns "" when nothing was found; None is the protocol's
        # spelling of that, and feedback restores "" for the grader.
        return build_prediction_record(
            [extract_answer(text, exhaust=False) or None for text in inf.texts]
        )

    @override
    async def feedback(self, post, ctx):
        gold = _gold_answer(ctx.raw_sample["answer"])
        # Concurrent, not sequential: each grade is an offloaded CPU-bound call
        # with its own GRADE_TIMEOUT, so awaiting them in turn makes a sample's
        # worst case n x the timeout instead of one.
        verdicts = await asyncio.gather(
            *(self._grade(rollout, gold, ctx) for rollout in post["rollouts"])
        )
        rollouts = [
            build_rollout_judgement(rollout["index"], verdict)
            for rollout, verdict in zip(post["rollouts"], verdicts, strict=True)
        ]
        return True, build_judgement_record(gold, rollouts)

    async def _grade(self, rollout, gold: str, ctx) -> bool:
        from sieval.community.deepseek_math import is_correct

        # `or ""` restores exactly what the grader saw pre-migration.
        prediction = rollout.get("prediction") or ""
        # `math_equal` runs `parse_latex` + `simplify`: ~11 ms typical, 1.7 s
        # worst case — measured on *reference* data, and `simplify` on arbitrary
        # model output has no ceiling. Reached with `timeout=False`, so nothing
        # else bounds it: criterion 2 in `core/utils/offload.py`.
        try:
            correct = await run_cpu_bound(
                is_correct,
                {"prediction": prediction, "answer": gold},
                timeout=GRADE_TIMEOUT,
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
        # math-0shot-gen family and DeepSeek's full-set accuracy: a pipeline
        # failure counts as wrong, not as an excluded sample.
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
