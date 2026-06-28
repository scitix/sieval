"""C-Eval few-shot next-token logprob task (base models).

Replicates the non-CoT path of the C-Eval ``evaluator_series`` LLaMA evaluator
(``code/evaluator_series/evaluators/llama.py``): a per-subject few-shot header,
``k`` ``dev`` exemplars, then the question with its four options and a trailing
``"答案："``. Each option letter A/B/C/D is scored by its conditional next-token
log-probability and the argmax is taken — equivalent to the reference's
single-pass ``softmax(logits[A,B,C,D])`` since argmax is invariant under softmax.

Deviations from the reference:
- Eval split is ``test`` (its labels are now public); ``evaluator_series/eval.py``
  scored ``val``. Selectable via the dataset's ``eval_split``.
- The per-letter logprob comes from four ``echo=True`` completion calls (one per
  candidate) rather than one full-vocab forward pass, because the backend is an
  OpenAI-compatible completions API. Equivalent for the argmax.
- ``evaluator_series`` reports one subject's accuracy per run and does not
  aggregate; ``score`` here is the macro-average over the 52 subjects (mean of
  per-subject accuracy), matching the C-Eval paper's "average accuracy over all
  the subjects".
- The few-shot header uses the English subject key (e.g. ``operating_system``),
  matching ``evaluator_series/eval.py`` (``subject_name=args.subject``), not the
  Chinese-name variant used by C-Eval's other evaluator.

Decoding is deterministic: argmax over candidate log-probabilities, no sampling
(``temperature=0``, ``max_tokens=1``, ``echo=True``); ``top_p`` / ``max_gen_toks``
do not apply.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from collections import defaultdict
from typing import TypedDict, override

from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    TaskStageOutput,
    sieval_task,
)
from sieval.core.utils.meta import build_stage_meta
from sieval.core.utils.ppl import extract_option_logprob
from sieval.datasets import CEvalDatasetSample

CHOICES = ("A", "B", "C", "D")
_FEWSHOT_SPLIT = "dev"


class Feedback(TypedDict):
    correct: bool
    pred: str
    answer: str
    subject: str


@sieval_task(
    name="c_eval_kshot_ppl",
    display_name="C-Eval (few-shot, next-token logprob)",
    description="C-Eval few-shot MCQ with CEval LLaMA next-token logprob scoring.",
    eval_mode=EvalMode.PPL,
    n_shot=5,
    tags=("chinese", "multiple-choice"),
    model_type="gen",
    reference_impl=ReferenceImpl(
        source="ceval",
        url="https://raw.githubusercontent.com/hkust-nlp/ceval/cba65ae93bcf189149ced9f66ae0c958201faed9/code/evaluator_series/evaluators/llama.py",
        notes=(
            "Mirrors the non-CoT LLaMA evaluator: per-subject few-shot header "
            "with the English subject key, dev exemplars, and next-token "
            "A/B/C/D logprob argmax (equivalent to softmax(logits[A,B,C,D])). "
            "Eval split is the released test set (the reference scored val); "
            "score is the macro-average over the 52 subjects."
        ),
    ),
)
class CEvalFewShotPPLTask(
    Task[
        CEvalDatasetSample,
        CEvalDatasetSample,
        TaskStageOutput[dict[str, float]],
        str,
        Feedback,
        dict[str, float],
    ]
):
    """C-Eval few-shot next-token logprob evaluation for base models."""

    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        *,
        k: int = 5,
        logprobs: int = 5,
    ):
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        super().__init__(dataset=dataset, model=model, name=name)
        self._k = k
        self._logprobs = logprobs
        self._few_shot_cache: dict[str, str] = {}
        self._few_shot_by_subject: dict[str, list[CEvalDatasetSample]] = {}

    @override
    async def setup(self) -> None:
        # Build every per-subject few-shot prefix once here, not per sample.
        if self._k <= 0:
            return
        self._ensure_few_shot_pool()
        for subject in self._few_shot_by_subject:
            self._build_few_shot_prompt(subject)

    def _ensure_few_shot_pool(self) -> None:
        """Group ``dev`` exemplars by subject (idempotent)."""
        if self._few_shot_by_subject or self._k <= 0:
            return
        dataset_dict = self.dataset.dataset_dict
        if _FEWSHOT_SPLIT not in dataset_dict:
            return
        # retrieve_samples() selects across a whole split and cannot express
        # C-Eval's same-subject, fixed-order few-shot, so group dev by subject.
        for sample in dataset_dict[_FEWSHOT_SPLIT]:
            self._few_shot_by_subject.setdefault(sample["subject"], []).append(sample)

    def _select_examples(self, subject: str) -> list[CEvalDatasetSample]:
        """First k same-subject dev exemplars, in dataset order (matches upstream)."""
        self._ensure_few_shot_pool()
        return list(self._few_shot_by_subject.get(subject, []))[: self._k]

    def _format_example(
        self, sample: CEvalDatasetSample, *, include_answer: bool = True
    ) -> str:
        example = sample["question"]
        for choice in CHOICES:
            example += f"\n{choice}. {sample[choice]}"
        example += "\n答案："
        if include_answer:
            example += f"{sample['answer']}\n\n"
        return example

    def _build_few_shot_prompt(self, subject: str) -> str:
        """Build (and cache) the few-shot prefix for a subject."""
        if self._k <= 0:
            return ""
        cached = self._few_shot_cache.get(subject)
        if cached is not None:
            return cached

        prompt = f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"
        for example in self._select_examples(subject):
            prompt += self._format_example(example, include_answer=True)

        self._few_shot_cache[subject] = prompt
        return prompt

    def _build_prompt(
        self, sample: CEvalDatasetSample, option_label: str | None = None
    ) -> str:
        """Build the prompt, optionally with a candidate answer-label continuation."""
        few_shot = self._build_few_shot_prompt(sample["subject"])
        question = self._format_example(sample, include_answer=False)
        return few_shot + question + (option_label or "")

    @override
    async def preprocess(self, raw, ctx):
        return raw

    @override
    async def infer(self, pre, ctx):
        """Score each option letter by its conditional next-token logprob."""
        scores: dict[str, float] = {}
        model_outputs = []

        for label in CHOICES:
            prompt = self._build_prompt(pre, label)
            # echo/max_tokens/logprobs are structural to logprob scoring (not
            # user decode prefs): echo returns the appended letter's logprob.
            lp_out = await self.model.alogprobs(
                prompt, max_tokens=1, logprobs=self._logprobs, echo=True
            )
            logprob = extract_option_logprob(
                lp_out.logprobs_tokens or [], lp_out.logprobs or [], label
            )
            scores[label] = logprob if logprob is not None else float("-inf")
            model_outputs.append(lp_out)

        return TaskStageOutput(value=scores, meta=build_stage_meta(*model_outputs))

    @override
    async def postprocess(self, inf, ctx):
        """Select the option with the highest log-probability."""
        if not inf.value:
            return ""
        return max(inf.value.items(), key=lambda item: item[1])[0]

    @override
    async def feedback(self, post, ctx):
        raw = ctx.raw_sample
        answer = raw["answer"]
        return True, {
            "correct": post == answer,
            "pred": post,
            "answer": answer,
            "subject": raw["subject"],
        }

    @override
    async def report(self, finals, fails):
        # score == macro-average: the mean of per-subject accuracy (the C-Eval
        # paper's "average over all subjects"). Failures are reported separately.
        by_subject: dict[str, list[bool]] = defaultdict(list)
        for ctx in finals:
            fb = ctx.feedback_result
            if fb is not None:
                by_subject[fb["subject"]].append(fb["correct"])
        if not by_subject:
            return {"score": 0.0, "fails": len(fails), "macro_accuracy": 0.0}
        per_subject_acc = [sum(c) / len(c) for c in by_subject.values()]
        score = 100 * sum(per_subject_acc) / len(per_subject_acc)
        return {"score": score, "fails": len(fails), "macro_accuracy": score}
