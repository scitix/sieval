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

from typing import TypedDict, override

from openai.types.chat import ChatCompletionUserMessageParam

from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import GSM8KDatasetSample

# Verbatim from run_subset_parallel.py::markup_question (language="en",
# task="cot"): f"{content}\nPlease reason step by step, and put your final
# answer within " + "\\boxed{}."
COT_INSTRUCTION = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)


class Feedback(TypedDict):
    correct: bool
    answer: str
    prediction: str


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
            "vendored byte-for-byte in sieval.community.deepseek_math. Gold "
            "derived from openai/gsm8k like process_gsm8k_test "
            "(answer.split('####')[-1], commas removed)."
        ),
    ),
)
class GSM8KZeroShotGenTask(
    Task[
        GSM8KDatasetSample,
        list[ChatCompletionUserMessageParam],
        ModelOutput,
        str,
        Feedback,
        dict[str, float],
    ]
):
    @override
    async def preprocess(self, raw, ctx):
        return [
            {"role": "user", "content": raw["question"] + COT_INSTRUCTION},
        ]

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

    @override
    async def postprocess(self, inf, ctx):
        from sieval.community.deepseek_math import extract_answer

        text = inf.texts[0] if inf.texts else ""
        return extract_answer(text, exhaust=False)

    @override
    async def feedback(self, post, ctx):
        from sieval.community.deepseek_math import is_correct

        gold = _gold_answer(ctx.raw_sample["answer"])
        correct = is_correct({"prediction": post, "answer": gold})
        return True, {"correct": correct, "answer": gold, "prediction": post}

    @override
    async def report(self, finals, fails):
        # Accuracy over the full requested set (finals + fails), matching the
        # math-0shot-gen family and DeepSeek's full-set accuracy: a pipeline
        # failure counts as wrong, not as an excluded sample.
        total = len(finals) + len(fails)
        if total == 0:
            return {"score": 0.0, "fails": len(fails), "accuracy": 0.0}
        correct_num = sum(1 for ctx in finals if ctx.feedback_result["correct"])
        accuracy = 100 * correct_num / total
        return {"score": accuracy, "fails": len(fails), "accuracy": accuracy}
