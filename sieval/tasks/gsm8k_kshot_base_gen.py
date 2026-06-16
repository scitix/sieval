"""
GSM8K few-shot base-model generative task.

The reported `score` and `exact_match` are strict GSM8K EM: predictions are
scored only when the model emits a `#### N` final answer. This matches the
original GSM8K dataset.py answer delimiter and the lm-eval-harness
`strict-match` filter. The prompt examples and strict extractor must stay in
lockstep around that `####` final-answer format.

The secondary `flexible_exact_match` metric mirrors the lm-eval-harness
`flexible-extract` last-number filter; it is not part of the original GSM8K
official metric.

The comparison target is DeepSeek-V3 Table 3: Qwen2.5-72B-Base GSM8K 8-shot
EM = 88.3. DeepSeek-V3 does not specify whether it used strict-match or
flexible-match extraction, so this task states and reports strict-match EM.
Default k=8 is chosen for that DeepSeek-V3 comparison.

AI-Generated Code - GPT-5.5 (OpenAI)
"""

import re
from typing import TypedDict, override

from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import GSM8KDatasetSample

N_SHOT = 8
DEFAULT_MAX_TOKENS = 2048
DEFAULT_FEWSHOT_SEED = 1234
STOP_SEQUENCES = ("Question:", "</s>", "<|im_end|>")

_STRICT_ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
_FLEXIBLE_ANSWER_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


class Feedback(TypedDict):
    correct: bool
    flexible_correct: bool
    answer: str
    prediction: str
    flexible_prediction: str
    extraction_method: str
    flexible_extraction_method: str


class Prediction(TypedDict):
    answer: str
    flexible_answer: str
    extraction_method: str
    flexible_extraction_method: str


def _format_example(sample: GSM8KDatasetSample, include_answer: bool) -> str:
    prompt = f"Question: {sample['question']}\nAnswer:"
    if include_answer:
        prompt += f" {sample['answer']}\n\n"
    return prompt


def _normalize_exact_match(text: str) -> str:
    text = re.sub(r",", "", text)
    text = re.sub(r"\$", "", text)
    text = re.sub(r"\.$", "", text.strip())
    return text.strip().lower()


def _extract_strict_answer(text: str) -> str:
    match = _STRICT_ANSWER_RE.search(text)
    return match.group(1).strip() if match else ""


def _extract_flexible_answer(text: str) -> str:
    matches = _FLEXIBLE_ANSWER_RE.findall(text)
    if not matches:
        return ""
    return next((part.strip() for part in matches[-1] if part), "")


def _extract_answer(text: str) -> tuple[str, str]:
    strict = _extract_strict_answer(text)
    if strict:
        return _normalize_exact_match(strict), "strict-match"
    return "", "none"


def _extract_flexible_match(text: str) -> tuple[str, str]:
    flexible = _extract_flexible_answer(text)
    if flexible:
        return _normalize_exact_match(flexible), "flexible-extract"
    return "", "none"


@sieval_task(
    name="gsm8k_kshot_base_gen",
    display_name="GSM8K (few-shot, base generative)",
    description="GSM8K few-shot base-model strict-match EM evaluation.",
    eval_mode=EvalMode.GEN,
    n_shot=N_SHOT,
    tags=("english", "math-word-problems", "open-ended", "base-model"),
    model_type="gen",
    reference_impl=ReferenceImpl(
        source="lm-evaluation-harness",
        url=(
            "https://github.com/EleutherAI/lm-evaluation-harness/blob/1dd931087362abba74e0375c8c631295559f48b2/lm_eval/tasks/gsm8k/gsm8k.yaml"
        ),
        notes=(
            "Uses lm-eval-harness strict-match extraction, aligned with the "
            "original GSM8K dataset.py #### answer delimiter. Also reports "
            "lm-eval-harness flexible-extract as a secondary metric, not as "
            "the original GSM8K official metric. Default k=8 follows the "
            "DeepSeek-V3 Table 3 comparison target (Qwen2.5-72B-Base GSM8K "
            "8-shot EM = 88.3); DeepSeek-V3 does not specify strict vs "
            "flexible extraction."
        ),
    ),
)
class GSM8KFewShotBaseGenTask(
    Task[
        GSM8KDatasetSample,
        str,
        ModelOutput,
        Prediction,
        Feedback,
        dict[str, float],
    ]
):
    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        *,
        k: int = N_SHOT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        fewshot_split: str = "train",
        fewshot_seed: int = DEFAULT_FEWSHOT_SEED,
        stop: tuple[str, ...] = STOP_SEQUENCES,
    ):
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        super().__init__(dataset=dataset, model=model, name=name)
        self._k = k
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._fewshot_split = fewshot_split
        self._fewshot_seed = fewshot_seed
        self._stop = stop
        self._fewshot_examples: list[GSM8KDatasetSample] | None = None

    @override
    async def setup(self) -> None:
        self._fewshot_examples = self._sample_fewshot_examples()

    @override
    async def preprocess(self, raw, ctx):
        examples = self._get_fewshot_examples()
        return "".join(_format_example(ex, include_answer=True) for ex in examples) + (
            _format_example(raw, include_answer=False)
        )

    @override
    async def infer(self, pre, ctx):
        kwargs: dict[str, object] = {
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if self._stop:
            kwargs["stop"] = list(self._stop)
        return await self.model.agenerate(pre, **kwargs)

    @override
    async def postprocess(self, inf, ctx):
        text = inf.texts[0] if inf.texts else ""
        answer, extraction_method = _extract_answer(text)
        flexible_answer, flexible_extraction_method = _extract_flexible_match(text)
        return {
            "answer": answer,
            "flexible_answer": flexible_answer,
            "extraction_method": extraction_method,
            "flexible_extraction_method": flexible_extraction_method,
        }

    @override
    async def feedback(self, post, ctx):
        answer, _ = _extract_answer(ctx.raw_sample["answer"])
        return True, {
            "correct": post["answer"] == answer,
            "flexible_correct": post["flexible_answer"] == answer,
            "answer": answer,
            "prediction": post["answer"],
            "flexible_prediction": post["flexible_answer"],
            "extraction_method": post["extraction_method"],
            "flexible_extraction_method": post["flexible_extraction_method"],
        }

    @override
    async def report(self, finals, fails):
        count = len(finals)
        if count == 0:
            return {
                "score": 0.0,
                "fails": len(fails),
                "exact_match": 0.0,
                "flexible_exact_match": 0.0,
            }
        correct_num = sum(1 for ctx in finals if ctx.feedback_result["correct"])
        flexible_correct_num = sum(
            1 for ctx in finals if ctx.feedback_result["flexible_correct"]
        )
        exact_match = 100 * correct_num / count
        flexible_exact_match = 100 * flexible_correct_num / count
        return {
            "score": exact_match,
            "fails": len(fails),
            "exact_match": exact_match,
            "flexible_exact_match": flexible_exact_match,
        }

    def _get_fewshot_examples(self) -> list[GSM8KDatasetSample]:
        if self._fewshot_examples is None:
            self._fewshot_examples = self._sample_fewshot_examples()
        return self._fewshot_examples

    def _sample_fewshot_examples(self) -> list[GSM8KDatasetSample]:
        split = self.dataset.dataset_dict.get(self._fewshot_split)
        if split is None:
            raise ValueError(
                "GSM8K few-shot base generative task requires a "
                f"{self._fewshot_split!r} split for few-shot examples."
            )
        if len(split) < self._k:
            raise ValueError(
                "GSM8K few-shot base generative task requires at least "
                f"{self._k} examples in split {self._fewshot_split!r}; "
                f"found {len(split)}."
            )
        if self._k == 0:
            return []
        return self.dataset.retrieve_samples(
            self._k,
            split=self._fewshot_split,
            mode="random",
            seed=self._fewshot_seed,
        )
