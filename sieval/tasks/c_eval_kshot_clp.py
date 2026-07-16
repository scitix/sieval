"""C-Eval few-shot conditional-log-prob (CLP) task (base models).

Replicates the non-CoT path of the C-Eval ``evaluator_series`` LLaMA evaluator
(``code/evaluator_series/evaluators/llama.py``): a per-subject few-shot header,
``k`` ``dev`` exemplars, then the question with its four options and a trailing
``"答案："``. The answer is the argmax over the A/B/C/D next-token log-probs from
a single inference (``EvalMode.CLP``) — equivalent to the reference's
single-pass ``softmax(logits[A,B,C,D])`` since softmax is monotonic.

Scoring reads one next token's ``top_logprobs`` and argmaxes over A/B/C/D
(matching option tokens by stripped string). It *requires all four* option
tokens to be in the returned top-k and raises otherwise, so a too-small top-k
fails the sample loudly instead of argmax-ing a subset.

Infra requirement for faithful reproduction: the serving backend must return a
top-k large enough to always include A/B/C/D. SGLang serves ``logprobs=100`` out
of the box; on vLLM start with ``--max-logprobs 100`` (default 20 can drop
option tokens). ``DEFAULT_LOGPROBS`` is 100 to match.

Deviations from the reference:
- Eval split is ``test`` (its labels are now public); ``evaluator_series/eval.py``
  scored ``val``. Selectable via the dataset's ``eval_split``.
- Uses the OpenAI-compatible API ``top_logprobs`` as a substitute for the
  official raw last-token-logits argmax; equivalent while all four option tokens
  are in the top-k.
- ``evaluator_series`` is per-subject and defines no cross-subject aggregation.
  ``score`` is the macro-average over the 52 subjects, following the C-Eval
  paper (Table 3: "average accuracy over all the subjects"; the reported 66.4
  for GPT-4 equals the unweighted mean of the 52 per-subject accuracies). This
  differs from lm-eval ``ceval-valid``, which micro-averages (``weight_by_size``)
  on the val split. Comparison target: DeepSeek-V3 Table 3, C-Eval 5-shot 89.2
  (Qwen2.5-72B-Base).
- The few-shot header uses the English subject key (e.g. ``operating_system``),
  matching ``evaluator_series/eval.py`` (``subject_name=args.subject``), not the
  Chinese-name variant used by C-Eval's other evaluator.

Decoding is deterministic: argmax over the next-token option log-probs, no
sampling (``max_tokens=1``, ``temperature=0``); ``top_p`` / ``max_gen_toks`` do
not apply.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from collections import defaultdict
from typing import TypedDict, override

from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.core.utils.ppl import choice_scores_from_top_logprobs
from sieval.datasets import CEvalDatasetSample

CHOICES = ("A", "B", "C", "D")
_FEWSHOT_SPLIT = "dev"
# Must be large enough that all of A/B/C/D land in the returned top-k. The
# validated run used 100; vLLM needs `--max-logprobs 100` (default 20).
DEFAULT_LOGPROBS = 100

# Subject → topic category from the C-Eval paper (Table 8 / Figure 1). Category
# scores are macro-averaged within each category, matching the paper's
# per-category "Average" columns; the overall score is the macro over all 52.
CEVAL_CATEGORY_SUBJECTS = {
    "STEM": (
        "advanced_mathematics",
        "college_chemistry",
        "college_physics",
        "college_programming",
        "computer_architecture",
        "computer_network",
        "discrete_mathematics",
        "electrical_engineer",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_mathematics",
        "high_school_physics",
        "metrology_engineer",
        "middle_school_biology",
        "middle_school_chemistry",
        "middle_school_mathematics",
        "middle_school_physics",
        "operating_system",
        "probability_and_statistics",
        "veterinary_medicine",
    ),
    "Social Science": (
        "business_administration",
        "college_economics",
        "education_science",
        "high_school_geography",
        "high_school_politics",
        "mao_zedong_thought",
        "marxism",
        "middle_school_geography",
        "middle_school_politics",
        "teacher_qualification",
    ),
    "Humanities": (
        "art_studies",
        "chinese_language_and_literature",
        "high_school_chinese",
        "high_school_history",
        "ideological_and_moral_cultivation",
        "law",
        "legal_professional",
        "logic",
        "middle_school_history",
        "modern_chinese_history",
        "professional_tour_guide",
    ),
    "Other": (
        "accountant",
        "basic_medicine",
        "civil_servant",
        "clinical_medicine",
        "environmental_impact_assessment_engineer",
        "fire_engineer",
        "physician",
        "plant_protection",
        "sports_science",
        "tax_accountant",
        "urban_and_rural_planner",
    ),
}


class Feedback(TypedDict):
    correct: bool
    pred: str
    answer: str
    subject: str


@sieval_task(
    name="c_eval_kshot_clp",
    display_name="C-Eval (few-shot, next-token logprob)",
    description="C-Eval few-shot MCQ with CEval LLaMA next-token logprob scoring.",
    eval_mode=EvalMode.CLP,
    n_shot=5,
    tags=("chinese", "multiple-choice"),
    model_type="gen",
    reference_impl=ReferenceImpl(
        source="ceval",
        url="https://raw.githubusercontent.com/hkust-nlp/ceval/cba65ae93bcf189149ced9f66ae0c958201faed9/code/evaluator_series/evaluators/llama.py",
        notes=(
            "Mirrors the non-CoT LLaMA evaluator: per-subject few-shot header "
            "with the English subject key, dev exemplars, and next-token "
            "A/B/C/D argmax via one-call top_logprobs (equivalent to the "
            "reference's softmax(logits[A,B,C,D]) while all four option tokens "
            "are in top-k; requires them present and fails otherwise). Faithful "
            "reproduction needs a top-k covering A/B/C/D (default logprobs=100; "
            "vLLM --max-logprobs 100, default 20; SGLang serves 100). Eval split "
            "is the released test set (reference scored val). Score is the "
            "macro-average over the 52 subjects (C-Eval paper Table 3), unlike "
            "lm-eval ceval-valid which micro-averages (weight_by_size). "
            "Comparison target: DeepSeek-V3 Table 3, C-Eval 5-shot 89.2 "
            "(Qwen2.5-72B-Base); the validated Qwen2.5-72B-Base 5-shot run on "
            "the test split scores 90.19 (macro)."
        ),
    ),
)
class CEvalFewShotCLPTask(
    Task[
        CEvalDatasetSample,
        str,
        ModelOutput,
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
        logprobs: int = DEFAULT_LOGPROBS,
    ):
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        if logprobs < 1:
            raise ValueError(f"logprobs must be >= 1, got {logprobs}")
        super().__init__(dataset=dataset, model=model, name=name)
        self._k = k
        self._logprobs = max(logprobs, len(CHOICES))
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
            raise ValueError(
                "C-Eval few-shot task requires a "
                f"{_FEWSHOT_SPLIT!r} split for few-shot examples."
            )
        # retrieve_samples() selects across a whole split and cannot express
        # C-Eval's same-subject, fixed-order few-shot, so group dev by subject.
        for sample in dataset_dict[_FEWSHOT_SPLIT]:
            self._few_shot_by_subject.setdefault(sample["subject"], []).append(sample)

    def _select_examples(self, subject: str) -> list[CEvalDatasetSample]:
        """First k same-subject dev exemplars, in dataset order (matches upstream)."""
        self._ensure_few_shot_pool()
        examples = self._few_shot_by_subject.get(subject, [])
        if len(examples) < self._k:
            raise ValueError(
                f"C-Eval subject {subject!r} has {len(examples)} dev exemplars; "
                f"need k={self._k}."
            )
        return list(examples)[: self._k]

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

    def _build_prompt(self, sample: CEvalDatasetSample) -> str:
        few_shot = self._build_few_shot_prompt(sample["subject"])
        return few_shot + self._format_example(sample, include_answer=False)

    @override
    async def preprocess(self, raw, ctx):
        return self._build_prompt(raw)

    @override
    async def infer(self, pre, ctx):
        # One next-token top_logprobs call (max_tokens=1, echo=False); these are
        # structural to CLP scoring, not user decode prefs.
        return await self.model.alogprobs(
            pre, max_tokens=1, logprobs=self._logprobs, echo=False
        )

    @override
    async def postprocess(self, inf, ctx):
        """Argmax over A/B/C/D; fail loudly if any option token is missing."""
        scores, all_present = choice_scores_from_top_logprobs(inf.top_logprobs, CHOICES)
        if not all_present:
            missing = [c for c in CHOICES if scores[c] == float("-inf")]
            raise RuntimeError(
                "C-Eval top_logprobs missing option token(s) "
                f"{missing}; increase logprobs (got top-k of {self._logprobs}) "
                "or raise the server's max-logprobs so all of A/B/C/D are "
                "returned."
            )
        return max(scores.items(), key=lambda item: item[1])[0]

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
        # Per-subject accuracy → macro over subjects (the C-Eval paper's
        # "average accuracy over all the subjects"); per-category scores are the
        # macro-average within each category. Infra failures are reported
        # separately in `fails`, not scored wrong (CLP-family convention).
        by_subject: dict[str, list[bool]] = defaultdict(list)
        for ctx in finals:
            fb = ctx.feedback_result
            by_subject[fb["subject"]].append(fb["correct"])

        subject_acc = {
            subject: 100 * sum(correct) / len(correct)
            for subject, correct in by_subject.items()
        }
        overall = sum(subject_acc.values()) / len(subject_acc) if subject_acc else 0.0

        metrics: dict[str, float] = {
            "score": overall,
            "fails": float(len(fails)),
            "overall": overall,
        }
        # Only report categories with evaluated subjects, so a subject-subset
        # run omits absent categories instead of a misleading 0.0.
        for category, subjects in CEVAL_CATEGORY_SUBJECTS.items():
            available = [subject_acc[s] for s in subjects if s in subject_acc]
            if available:
                metrics[category.lower().replace(" ", "_")] = sum(available) / len(
                    available
                )
        return metrics
