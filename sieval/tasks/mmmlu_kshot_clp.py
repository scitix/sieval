"""MMMLU full multilingual k-shot base-model CLP task.

Protocol: follows the EleutherAI lm-evaluation-harness ``openai-mmmlu``
organization (locale -> category -> subject, size-weighted aggregation) with
Hendrycks ``evaluate.py`` completion scoring: one next token, ``top_logprobs``,
and space-prefixed ``" A"``/``" B"``/``" C"``/``" D"`` choices.  This is
base-model completion scoring, not the simple-evals chat/instruct protocol.

Deviations from those references:
    - OpenAI MMMLU ships only translated test CSVs, so few-shot examples are
      drawn from the test split (as in lm-evaluation-harness); the first ``k``
      rows per ``(Locale, Subject)`` are reserved and removed from the scored
      test split so no item appears in its own demonstration pool.
    - Language selection is a dataset concern: configure ``MMMLUDataset`` with
      ``args.locales`` (e.g. ``[zh_cn]`` for a Chinese run).  There is no
      separate ZH-CN task; Chinese evaluation is this protocol restricted to
      one locale.
    - The references define no sampling protocol.  Optional efficient
      evaluation (e.g. the Qwen3 technical report's 10% MMMLU setting) is
      offered via ``sample_fraction``/``sample_seed``/``sample_by``, applied in
      ``setup()`` before few-shot reservation so scoring, reservation, and
      aggregation are identical for full and sampled runs.

Reproduction decoding: greedy ``temperature=0`` with ``max_tokens=1`` and
top-``logprobs`` scoring.  These are structural to ``alogprobs`` (next-token
logprob scoring with no text generation), not free decoding knobs.

Infra requirement: the serving backend must return a top-k large enough to
include all of ``" A"``/``" B"``/``" C"``/``" D"``.  Unlike the CMMLU sibling
(which fails loud), a missing option is filled with ``-100`` *silently* (per
Hendrycks ``evaluate.py``), so a too-small server top-k depresses the score with
no error.  ``logprobs`` defaults to 100; SGLang serves 100 out of the box, but
on vLLM start the server with ``--max-logprobs 100`` (its default is 20).

AI-Generated Code - GPT-5-Codex (OpenAI)
"""

import math
import random
from collections.abc import Sequence
from typing import TypedDict, cast, override

from sieval.core.datasets import Dataset
from sieval.core.models import Model
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    TaskContext,
    TaskStageOutput,
    sieval_task,
)
from sieval.core.utils.meta import build_stage_meta
from sieval.datasets import MMMLUDatasetSample

CHOICES = ("A", "B", "C", "D")
DEFAULT_N_SHOT = 5
OFFICIAL_MISSING_LOGPROB = -100.0
MMMLU_SAMPLE_BY = ("locale", "locale_subject")


class OfficialScores(TypedDict):
    logprobs: dict[str, float]
    probs: dict[str, float]


class Feedback(TypedDict):
    correct: bool
    pred: str
    answer: str
    subject: str
    category: str
    locale: str
    prob_A: float
    prob_B: float
    prob_C: float
    prob_D: float


class _MetricCounts(TypedDict):
    correct: int
    total: int


def _softmax(scores: Sequence[float]) -> list[float]:
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    denominator = sum(exp_scores)
    return [score / denominator for score in exp_scores]


def _format_subject(subject: str, locale_display_name: str) -> str:
    subject_text = " ".join(subject.split("_"))
    return f"{subject_text} ({locale_display_name})"


def _format_example(sample: MMMLUDatasetSample, *, include_answer: bool = True) -> str:
    prompt = sample["Question"].strip()
    for choice, text in zip(
        CHOICES,
        (sample["A"], sample["B"], sample["C"], sample["D"]),
        strict=True,
    ):
        prompt += f"\n{choice}. {text}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {sample['Answer']}\n\n"
    return prompt


def _logprobs_to_official_scores(
    top_logprobs: list[dict[str, float]],
) -> OfficialScores:
    final_top_logprobs = top_logprobs[-1] if top_logprobs else {}
    logprobs = {
        choice: final_top_logprobs.get(f" {choice}", OFFICIAL_MISSING_LOGPROB)
        for choice in CHOICES
    }
    probs = dict(
        zip(CHOICES, _softmax([logprobs[choice] for choice in CHOICES]), strict=True)
    )
    return {"logprobs": logprobs, "probs": probs}


def _metric_score(counts: _MetricCounts) -> float:
    return counts["correct"] * 100 / counts["total"] if counts["total"] else 0.0


def _add_metric(
    metrics: dict[str, _MetricCounts],
    key: str,
    *,
    correct: bool,
) -> None:
    counts = metrics.setdefault(key, {"correct": 0, "total": 0})
    counts["total"] += 1
    if correct:
        counts["correct"] += 1


def _normalize_sample_fraction(sample_fraction: object) -> float | None:
    if sample_fraction is None:
        return None
    if isinstance(sample_fraction, bool) or not isinstance(
        sample_fraction, int | float
    ):
        raise ValueError(
            "MMMLU sample_fraction must be a number in the interval (0, 1]."
        )
    fraction = float(sample_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            "MMMLU sample_fraction must be a number in the interval (0, 1]."
        )
    return fraction


def _normalize_sample_seed(sample_seed: object) -> int:
    if isinstance(sample_seed, bool) or not isinstance(sample_seed, int):
        raise ValueError("MMMLU sample_seed must be an integer.")
    return sample_seed


def _normalize_sample_by(sample_by: str) -> str:
    normalized = str(sample_by).strip().lower()
    if normalized not in MMMLU_SAMPLE_BY:
        allowed = ", ".join(MMMLU_SAMPLE_BY)
        raise ValueError(f"MMMLU sample_by must be one of: {allowed}.")
    return normalized


@sieval_task(
    name="mmmlu_kshot_clp",
    display_name="MMMLU (k-shot, CLP)",
    description="OpenAI MMMLU multilingual k-shot MCQ with weighted groups.",
    eval_mode=EvalMode.CLP,
    n_shot=DEFAULT_N_SHOT,
    tags=("multilingual", "multiple-choice"),
    model_type="gen",
    reference_impl=ReferenceImpl(
        source="lm-evaluation-harness + hendrycks/test",
        url=(
            "https://github.com/EleutherAI/lm-evaluation-harness/tree/1dd931087362abba74e0375c8c631295559f48b2/lm_eval/tasks/openai-mmmlu"
        ),
        notes=(
            "MMMLU locale/category/subject organization and size-weighted "
            "(micro) aggregation from lm-evaluation-harness; Hendrycks "
            "evaluate.py completion scoring via a single alogprobs call that "
            "reads the next token's top-logprobs over the space-prefixed option "
            "tokens ' A'/' B'/' C'/' D' (a missing option is filled with -100 "
            "per Hendrycks, not failed loudly; empirically ~0.009% of samples). "
            "Because the -100 fill is silent, faithful reproduction needs a "
            "server top-k that returns all four option tokens: SGLang serves 100 "
            "by default; on vLLM start with --max-logprobs 100 (default 20). "
            "Validated on Qwen2.5-72B (sglang): full 14-locale 5-shot = 76.13, "
            "zh_cn = 82.43. Those runs used the pre-rename PPL/base_gen "
            "packaging at an out-of-tree SHA; the scoring mechanism (single "
            "alogprobs call) is identical to this clp task, so the numbers "
            "carry over."
        ),
    ),
)
class MMMLUKShotClpTask(
    Task[
        MMMLUDatasetSample,
        str,
        TaskStageOutput[OfficialScores],
        str,
        Feedback,
        dict[str, float],
    ]
):
    """Full MMMLU evaluation with weighted locale/category/subject reporting."""

    def __init__(
        self,
        dataset: Dataset[MMMLUDatasetSample],
        model: Model[str],
        name: str | None = None,
        *,
        k: int = DEFAULT_N_SHOT,
        fewshot_split: str = "test",
        logprobs: int = 100,
        sample_fraction: float | None = None,
        sample_seed: int = 0,
        sample_by: str = "locale_subject",
    ):
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        if logprobs < 1:
            raise ValueError(f"logprobs must be >= 1, got {logprobs}")
        super().__init__(dataset=dataset, model=model, name=name)
        self._k = k
        self._fewshot_split = fewshot_split
        self._logprobs = logprobs
        self._sample_fraction = _normalize_sample_fraction(sample_fraction)
        self._sample_seed = _normalize_sample_seed(sample_seed)
        self._sample_by = _normalize_sample_by(sample_by)
        self._fewshot_by_group: dict[tuple[str, str], list[MMMLUDatasetSample]] | None
        self._fewshot_by_group = None
        self._fewshot_source_indices: set[int] = set()
        self._eval_split_excludes_fewshot = False
        self._sampling_applied = False
        self._prompt_cache: dict[tuple[str, str], str] = {}

    @override
    async def setup(self) -> None:
        self._apply_sampling()
        if self._k > 0:
            self._ensure_fewshot_pool()

    @override
    async def preprocess(
        self,
        raw: MMMLUDatasetSample,
        ctx: TaskContext[
            MMMLUDatasetSample,
            str,
            TaskStageOutput[OfficialScores],
            str,
            Feedback,
        ],
    ) -> str:
        train_prompt = self._build_train_prompt(raw)
        prompt_end = _format_example(raw, include_answer=False)
        return train_prompt + prompt_end

    @override
    async def infer(
        self,
        pre: str,
        ctx: TaskContext[
            MMMLUDatasetSample,
            str,
            TaskStageOutput[OfficialScores],
            str,
            Feedback,
        ],
    ) -> TaskStageOutput[OfficialScores]:
        # CLP reads only the generated next token's top-logprobs, never the
        # echoed prompt (echo is for scoring a supplied continuation, i.e. true
        # PPL). echo=False is bit-identical here, faster, and prefix-cache-safe.
        output = await self.model.alogprobs(
            pre,
            max_tokens=1,
            logprobs=self._logprobs,
            echo=False,
            temperature=0.0,
        )
        scores = _logprobs_to_official_scores(output.top_logprobs or [])
        return TaskStageOutput(value=scores, meta=build_stage_meta(output))

    @override
    async def postprocess(
        self,
        inf: TaskStageOutput[OfficialScores],
        ctx: TaskContext[
            MMMLUDatasetSample,
            str,
            TaskStageOutput[OfficialScores],
            str,
            Feedback,
        ],
    ) -> str:
        logprobs = inf.value["logprobs"]
        return max(CHOICES, key=lambda choice: logprobs[choice])

    @override
    async def feedback(
        self,
        post: str,
        ctx: TaskContext[
            MMMLUDatasetSample,
            str,
            TaskStageOutput[OfficialScores],
            str,
            Feedback,
        ],
    ) -> tuple[bool, Feedback]:
        raw = ctx.raw_sample
        if raw is None:
            return True, {
                "correct": False,
                "pred": post,
                "answer": "",
                "subject": "",
                "category": "",
                "locale": "",
                "prob_A": 0.0,
                "prob_B": 0.0,
                "prob_C": 0.0,
                "prob_D": 0.0,
            }
        probs = ctx.infer_result.value["probs"] if ctx.infer_result else {}
        answer = raw["Answer"]
        return True, {
            "correct": post == answer,
            "pred": post,
            "answer": answer,
            "subject": self._subject(raw),
            "category": self._category(raw),
            "locale": self._locale(raw),
            "prob_A": probs.get("A", 0.0),
            "prob_B": probs.get("B", 0.0),
            "prob_C": probs.get("C", 0.0),
            "prob_D": probs.get("D", 0.0),
        }

    @override
    async def report(
        self,
        finals: list[
            TaskContext[
                MMMLUDatasetSample,
                str,
                TaskStageOutput[OfficialScores],
                str,
                Feedback,
            ]
        ],
        fails: list[
            TaskContext[
                MMMLUDatasetSample,
                str,
                TaskStageOutput[OfficialScores],
                str,
                Feedback,
            ]
        ],
    ) -> dict[str, float]:
        # Infra failures (e.g. transient ReadError) are reported separately and
        # excluded from the denominator, matching cmmlu_kshot_base_gen /
        # theoremqa: a scored-but-degenerate sample stays a final and counts as
        # wrong, but an unscored infra fail must not depress accuracy.
        total = len(finals)
        if total == 0:
            return {"score": 0.0, "score_mmmlu": 0.0, "fails": float(len(fails))}

        metric_counts: dict[str, _MetricCounts] = {}
        correct_total = 0

        for ctx in finals:
            if ctx.feedback_result is None:
                correct = False
                locale = "unknown"
                category = "unknown"
                subject = "unknown"
            else:
                correct = ctx.feedback_result["correct"]
                locale = ctx.feedback_result["locale"]
                category = ctx.feedback_result["category"]
                subject = ctx.feedback_result["subject"]
            if correct:
                correct_total += 1
            self._add_group_metrics(
                metric_counts,
                locale=locale,
                category=category,
                subject=subject,
                correct=correct,
            )

        score = correct_total * 100 / total
        metrics: dict[str, float] = {
            "score": score,
            "score_mmmlu": score,
            "fails": float(len(fails)),
        }
        for key, counts in sorted(metric_counts.items()):
            metrics[key] = _metric_score(counts)
        return metrics

    def _build_train_prompt(self, sample: MMMLUDatasetSample) -> str:
        group = self._group(sample)
        cached = self._prompt_cache.get(group)
        if cached is not None:
            return cached

        subject = self._subject(sample)
        locale_display_name = self._locale_display_name(sample)
        prompt = (
            "The following are multiple choice questions (with answers) about "
            f"{_format_subject(subject, locale_display_name)}.\n\n"
        )
        for example in self._fewshot_examples_for(group):
            prompt += _format_example(example)

        self._prompt_cache[group] = prompt
        return prompt

    def _apply_sampling(self) -> None:
        # Idempotent like _ensure_fewshot_pool / _exclude_fewshot_examples: the
        # runner calls setup() once, but a second call must not sample a sample.
        if self._sampling_applied:
            return
        if self._sample_fraction is None:
            return

        test_split = self.dataset.dataset_dict.get("test")
        if test_split is None or len(test_split) == 0:
            return

        groups: dict[tuple[str, ...], list[int]] = {}
        for index, row in enumerate(test_split):
            groups.setdefault(self._sample_key(row), []).append(index)

        sampled_indices: list[int] = []
        for key in sorted(groups):
            indices = groups[key].copy()
            rng = random.Random(
                f"{self._sample_seed}:{self._sample_by}:{'|'.join(key)}"
            )
            rng.shuffle(indices)
            sample_size = max(1, math.ceil(len(indices) * self._sample_fraction))
            sampled_indices.extend(indices[:sample_size])

        self.dataset.dataset_dict["test"] = test_split.select(sampled_indices)
        self._sampling_applied = True

    def _sample_key(self, row: dict[str, object]) -> tuple[str, ...]:
        locale = str(row.get("Locale", "")).strip()
        if self._sample_by == "locale":
            return (locale,)
        return (locale, str(row.get("Subject", "")).strip())

    def _ensure_fewshot_pool(self) -> None:
        if self._k <= 0:
            self._fewshot_by_group = {}
            return

        if self._fewshot_by_group is None:
            self._fewshot_by_group = self._collect_fewshot_by_group()

        if self._fewshot_split == "test":
            self._exclude_fewshot_examples_from_eval_split()

    def _collect_fewshot_by_group(
        self,
    ) -> dict[tuple[str, str], list[MMMLUDatasetSample]]:
        split = self.dataset.dataset_dict.get(self._fewshot_split)
        if split is None:
            raise ValueError(
                "MMMLU official k-shot evaluation requires a "
                f"{self._fewshot_split!r} split with same-locale few-shot examples."
            )

        fewshot_by_group: dict[tuple[str, str], list[MMMLUDatasetSample]] = {}
        group_totals: dict[tuple[str, str], int] = {}
        source_indices: set[int] = set()
        for index, row in enumerate(split):
            sample = cast(MMMLUDatasetSample, row)
            group = self._group(sample)
            group_totals[group] = group_totals.get(group, 0) + 1
            examples = fewshot_by_group.setdefault(group, [])
            if len(examples) < self._k:
                examples.append(sample)
                if self._fewshot_split == "test":
                    source_indices.add(index)

        if self._fewshot_split == "test":
            for group, examples in fewshot_by_group.items():
                if len(examples) < self._k:
                    raise ValueError(
                        "MMMLU official k-shot evaluation requires at least "
                        f"{self._k} {self._fewshot_split!r} examples for locale "
                        f"{group[0]!r}, subject {group[1]!r}; found {len(examples)}."
                    )

            for group, total in group_totals.items():
                if total <= self._k:
                    raise ValueError(
                        "MMMLU test-split few-shot evaluation requires at least "
                        f"{self._k + 1} test examples for locale {group[0]!r}, "
                        f"subject {group[1]!r} so reserved few-shot examples can "
                        f"be excluded from the scored test set; found {total}."
                    )

        self._fewshot_source_indices = source_indices
        return fewshot_by_group

    def _exclude_fewshot_examples_from_eval_split(self) -> None:
        if self._eval_split_excludes_fewshot:
            return

        test_split = self.dataset.dataset_dict.get("test")
        if test_split is None:
            raise ValueError(
                "MMMLU test-split few-shot evaluation requires a 'test' split "
                "to exclude reserved few-shot examples from scoring."
            )

        eval_indices = [
            index
            for index in range(len(test_split))
            if index not in self._fewshot_source_indices
        ]
        if len(eval_indices) == len(test_split):
            self._eval_split_excludes_fewshot = True
            return

        self.dataset.dataset_dict["test"] = test_split.select(eval_indices)
        self._eval_split_excludes_fewshot = True

    def _fewshot_examples_for(self, group: tuple[str, str]) -> list[MMMLUDatasetSample]:
        if self._k <= 0:
            return []
        if self._fewshot_by_group is None:
            self._ensure_fewshot_pool()

        fewshot_by_group = self._fewshot_by_group
        if fewshot_by_group is None:
            raise RuntimeError("MMMLU few-shot pool was not initialized.")

        examples = fewshot_by_group.get(group, [])
        if len(examples) < self._k:
            raise ValueError(
                "MMMLU official k-shot evaluation requires at least "
                f"{self._k} {self._fewshot_split!r} examples for locale "
                f"{group[0]!r}, subject {group[1]!r}; found {len(examples)}."
            )
        return examples[: self._k]

    def _add_group_metrics(
        self,
        metric_counts: dict[str, _MetricCounts],
        *,
        locale: str,
        category: str,
        subject: str,
        correct: bool,
    ) -> None:
        _add_metric(metric_counts, f"score_locale_{locale}", correct=correct)
        _add_metric(
            metric_counts,
            f"score_locale_{locale}_category_{category}",
            correct=correct,
        )
        _add_metric(
            metric_counts,
            f"score_locale_{locale}_subject_{subject}",
            correct=correct,
        )

    def _group(self, sample: MMMLUDatasetSample) -> tuple[str, str]:
        return (self._locale(sample), self._subject(sample))

    def _subject(self, sample: MMMLUDatasetSample) -> str:
        subject = str(sample.get("Subject", "")).strip()
        if not subject:
            raise ValueError("MMMLU official k-shot evaluation requires Subject.")
        return subject

    def _category(self, sample: MMMLUDatasetSample) -> str:
        category = str(sample.get("Category", "")).strip()
        if not category:
            raise ValueError("MMMLU official k-shot evaluation requires Category.")
        return category

    def _locale(self, sample: MMMLUDatasetSample) -> str:
        locale = str(sample.get("Locale", "")).strip()
        if not locale:
            raise ValueError("MMMLU official k-shot evaluation requires Locale.")
        return locale

    def _locale_display_name(self, sample: MMMLUDatasetSample) -> str:
        display_name = str(sample.get("LocaleDisplayName", "")).strip()
        return display_name or self._locale(sample)
