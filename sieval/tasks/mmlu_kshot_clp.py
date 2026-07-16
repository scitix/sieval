"""
MMLU few-shot base-model clp task, aligned with hendrycks/test evaluate_flan.py.

Scoring: the reference takes one forward pass over the prompt ending in
``Answer:``, restricts the next-token logits to the four letter tokens
``A/B/C/D``, softmaxes and predicts the argmax. This task reproduces the
*prediction* over the OpenAI-style completions API with a single
``alogprobs(echo=False)`` call, reading the next token's ``top_logprobs`` and
argmax-ing over A/B/C/D. That is the identical prediction: softmax is monotonic
and its normalizer is constant across the four letters. This is OpenCompass
``clp`` (conditional log-prob, single inference), not ``ppl`` (full-sequence
perplexity of multi-token continuations).

Few-shot: the first ``k`` ``dev`` examples of the *same subject*, in dataset
order (matching evaluate_flan ``dev_df[:k]`` per subject). Prompt text
(``_format_subject`` / ``_format_example``) is copied verbatim from the
reference and pinned by tests.

Scoring requires all of A/B/C/D in the returned top-k and raises otherwise, so a
too-small top-k surfaces as a sample failure rather than a best-of-present
guess. Faithful reproduction needs a top-k that always includes the four option
letters, so ``DEFAULT_LOGPROBS`` defaults to 100 (SGLang serves 100 by default;
on vLLM start with ``--max-logprobs 100``, whose default of 20 can drop them).

Target: 85.0 — Qwen2.5-72B Base, MMLU 5-shot, from the DeepSeek-V3 report
Table 6 (base-model comparison). Cross-check only: that table is scored under
DeepSeek's own harness, so a small gap from this clp reproduction is expected
rather than a reproduction failure.

Deviations from the reference:
- No client-side 2048-token truncation (the reference drops shots when the
  tokenized prompt overflows); there is no tokenizer at this layer.
- ``score`` is the micro-average over the finalized set (denominator
  ``len(finals)``), matching the reference ``weighted_acc`` and the CLP siblings
  ``cmmlu_kshot_clp`` / ``mmmlu_kshot_clp``: infra failures are reported
  separately (``fails``) rather than scored wrong. Per-category accuracies are
  reported alongside.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from collections import defaultdict
from typing import Any, TypedDict, override

from sieval.community.simple_evals.mmlu_eval import subject2category
from sieval.core.datasets import Dataset
from sieval.core.models import Model, ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.core.utils.ppl import choice_scores_from_top_logprobs
from sieval.datasets import MMLUDatasetSample

CHOICES = ("A", "B", "C", "D")
DEFAULT_N_SHOT = 5
# Must be large enough that all of A/B/C/D land in the returned top-k. SGLang
# serves 100 by default; vLLM needs `--max-logprobs 100` (its default is 20).
DEFAULT_LOGPROBS = 100


class Feedback(TypedDict):
    correct: bool
    subject: str
    category: str
    answer: str
    prediction: str


def _format_subject(subject: str) -> str:
    """Reference ``format_subject``: ``abstract_algebra`` -> `` abstract algebra``."""
    return " " + " ".join(subject.split("_"))


def _format_example(sample: MMLUDatasetSample, *, include_answer: bool) -> str:
    prompt = sample["question"]
    for label, choice in zip(CHOICES, sample["choices"], strict=True):
        prompt += f"\n{label}. {choice}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {CHOICES[sample['answer']]}\n\n"
    return prompt


@sieval_task(
    name="mmlu_kshot_clp",
    display_name="MMLU (few-shot, base clp)",
    description="MMLU few-shot MCQ, base-model next-token A/B/C/D logprob scoring.",
    eval_mode=EvalMode.CLP,
    n_shot=DEFAULT_N_SHOT,
    tags=("english", "multiple-choice", "base-model"),
    model_type="gen",
    reference_impl=ReferenceImpl(
        source="hendrycks/test",
        url=(
            "https://github.com/hendrycks/test/blob/"
            "4450500f923c49f1fb1dd3d99108a0bd9717b660/evaluate_flan.py"
        ),
        notes=(
            "Next-token A/B/C/D scoring via one alogprobs(echo=False) call, "
            "argmax over the option tokens in top_logprobs; equivalent to the "
            "reference softmax over the restricted letter logits (OpenCompass "
            "clp). Few-shot = first k dev examples per subject in dataset "
            "order. Requires all of A/B/C/D in the top-k and fails the sample "
            "otherwise (default logprobs=100; vLLM needs --max-logprobs 100, "
            "default 20; SGLang serves 100). Micro-average score over the "
            "finalized set (infra failures reported separately, not scored "
            "wrong) + per-category via subject2category, matching evaluate_flan "
            "weighted_acc and the CLP siblings cmmlu/mmmlu. No client-side "
            "2048-token truncation. Validated: Qwen2.5-72B base, 5-shot, "
            "14042/14042, score 86.11 (target 85.0), 0 fails, deterministic "
            "across two runs."
        ),
    ),
)
class MMLUFewShotCLPTask(
    Task[
        MMLUDatasetSample,
        str,
        ModelOutput,
        str,
        Feedback,
        dict[str, float],
    ]
):
    def __init__(
        self,
        dataset: Dataset[MMLUDatasetSample],
        model: Model[Any],
        name: str | None = None,
        *,
        k: int = DEFAULT_N_SHOT,
        logprobs: int = DEFAULT_LOGPROBS,
        fewshot_split: str = "dev",
    ):
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        if logprobs < 1:
            raise ValueError(f"logprobs must be >= 1, got {logprobs}")
        super().__init__(dataset=dataset, model=model, name=name)
        self._k = k
        self._logprobs = max(logprobs, len(CHOICES))
        self._fewshot_split = fewshot_split
        self._few_shot_by_subject: dict[str, list[MMLUDatasetSample]] = {}
        self._few_shot_prompt_by_subject: dict[str, str] = {}

    @override
    async def setup(self) -> None:
        self._ensure_few_shot_pool()

    def _ensure_few_shot_pool(self) -> None:
        if self._few_shot_by_subject or self._k == 0:
            return
        split = self.dataset.dataset_dict.get(self._fewshot_split)
        if split is None:
            raise ValueError(
                "MMLU few-shot clp task requires a "
                f"{self._fewshot_split!r} split for few-shot examples."
            )
        for sample in split:
            subject = str(sample.get("subject") or "unknown")
            self._few_shot_by_subject.setdefault(subject, []).append(sample)

    def _select_examples(self, subject: str) -> list[MMLUDatasetSample]:
        if self._k == 0:
            return []
        self._ensure_few_shot_pool()
        return list(self._few_shot_by_subject.get(subject, []))[: self._k]

    def _build_few_shot_prompt(self, subject: str) -> str:
        cached = self._few_shot_prompt_by_subject.get(subject)
        if cached is not None:
            return cached
        prompt = (
            "The following are multiple choice questions (with answers) about"
            f"{_format_subject(subject)}.\n\n"
        )
        for example in self._select_examples(subject):
            prompt += _format_example(example, include_answer=True)
        self._few_shot_prompt_by_subject[subject] = prompt
        return prompt

    @override
    async def preprocess(self, raw, ctx):
        subject = raw.get("subject") or "unknown"
        return self._build_few_shot_prompt(subject) + _format_example(
            raw, include_answer=False
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.alogprobs(
            pre,
            max_tokens=1,
            logprobs=self._logprobs,
            echo=False,
        )

    @override
    async def postprocess(self, inf, ctx):
        scores, all_present = choice_scores_from_top_logprobs(inf.top_logprobs, CHOICES)
        if not all_present:
            missing = [label for label in CHOICES if scores[label] == float("-inf")]
            raise RuntimeError(
                "MMLU top_logprobs missing option token(s) "
                f"{missing}; increase logprobs (got top-k of {self._logprobs}) "
                "or raise the server's max-logprobs so all of A/B/C/D are "
                "returned."
            )
        return max(scores.items(), key=lambda item: item[1])[0]

    @override
    async def feedback(self, post, ctx):
        answer = CHOICES[ctx.raw_sample["answer"]]
        subject = ctx.raw_sample.get("subject", "unknown")
        category = subject2category.get(subject, "other")
        return True, {
            "correct": post == answer,
            "subject": subject,
            "category": category,
            "answer": answer,
            "prediction": post,
        }

    @override
    async def report(self, finals, fails):
        correct_num = 0
        category_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
        for ctx in finals:
            correct = ctx.feedback_result["correct"]
            category = ctx.feedback_result["category"]
            if correct:
                correct_num += 1
                category_metrics[category]["correct"] += 1
            category_metrics[category]["total"] += 1

        # CLP-family convention (cmmlu / mmmlu): the score denominator is the
        # finalized set; infra failures are reported separately (``fails``), not
        # scored wrong.
        score = 100 * correct_num / len(finals) if finals else 0.0
        results = {"score": score}
        for category, metrics in category_metrics.items():
            results[f"score_{category}"] = (
                100 * metrics["correct"] / metrics["total"]
                if metrics["total"] > 0
                else 0.0
            )
        results["fails"] = len(fails)
        return results
