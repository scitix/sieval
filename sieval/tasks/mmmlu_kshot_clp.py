"""MMMLU full multilingual k-shot base-model CLP task.

Protocol: follows the EleutherAI lm-evaluation-harness ``openai-mmmlu``
organization (locale -> category -> subject, size-weighted aggregation) with
Hendrycks ``evaluate.py`` completion scoring: one next token, ``top_logprobs``,
and space-prefixed ``" A"``/``" B"``/``" C"``/``" D"`` choices.  This is
base-model completion scoring, not the simple-evals chat/instruct protocol.

Deviations from those references:
    - OpenAI MMMLU ships only translated test CSVs, so few-shot examples are
      drawn from the test split (as in lm-evaluation-harness); the first ``n_shot``
      rows per ``(Locale, Subject)`` are reserved and removed from the scored
      test split so no item appears in its own demonstration pool.
    - Language selection is a dataset concern: configure ``MMMLUDataset`` with
      ``args.locales`` (e.g. ``[zh_cn]`` for a Chinese run).  There is no
      separate ZH-CN task; Chinese evaluation is this protocol restricted to
      one locale.
    - The references define no sampling protocol.  Efficient evaluation (e.g.
      the Qwen3 technical report's 10% MMMLU setting) is a *dataset* operation,
      not a task argument — the task used to carry its own stratified sampler::

          operations:
            - stratified_sample:
                {by: [Locale, Subject], fraction: 0.1, min_per_group: 6, seed: 0}

      Stratify by ``[Locale, Subject]``, the same key few-shot reservation uses.
      Reservation needs *more* than ``n_shot`` rows in every ``(Locale, Subject)``
      cell, so the budget has to keep at least ``n_shot + 1`` per cell — hence
      ``min_per_group``, and a *fraction* large enough for the smallest subject
      (MMLU's smallest test subjects hold 100 rows, so ``0.1`` clears the default
      ``n_shot=5``; ``0.05`` does not, and fails loudly).  Stratifying by
      ``[Locale]`` alone applies no per-subject floor: it can leave a cell short
      (a hard error) or drop one entirely, which is *silent* — the subject simply
      never reaches the scored split and vanishes from the size-weighted
      aggregation.  Do not sample this task that way.

      Operations run when the dataset is built, so the subsample still precedes
      few-shot reservation exactly as before, and the scoring, reservation and
      aggregation *mechanism* is identical for full and sampled runs.

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
from collections.abc import Sequence
from typing import TypedDict, cast, override

from sieval.core.datasets import Dataset
from sieval.core.models import Model
from sieval.core.tasks import (
    EvalMode,
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    ReferenceImpl,
    Task,
    TaskContext,
    TaskStageOutput,
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
    sieval_task,
)
from sieval.core.utils.meta import build_stage_meta
from sieval.datasets import MMMLUDatasetSample

CHOICES = ("A", "B", "C", "D")
DEFAULT_N_SHOT = 5
OFFICIAL_MISSING_LOGPROB = -100.0


class OfficialScores(TypedDict):
    logprobs: dict[str, float]
    probs: dict[str, float]


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
        PromptRecord,
        TaskStageOutput[OfficialScores],
        PredictionRecord,
        JudgementRecord,
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
        n_shot: int = DEFAULT_N_SHOT,
        fewshot_split: str = "test",
        logprobs: int = 100,
    ):
        if n_shot < 0:
            raise ValueError(f"n_shot must be >= 0, got {n_shot}")
        if logprobs < 1:
            raise ValueError(f"logprobs must be >= 1, got {logprobs}")
        super().__init__(dataset=dataset, model=model, name=name)
        self.n_shot = n_shot
        self._fewshot_split = fewshot_split
        self._logprobs = logprobs
        self._fewshot_by_group: dict[tuple[str, str], list[MMMLUDatasetSample]] | None
        self._fewshot_by_group = None
        self._fewshot_source_indices: set[int] = set()
        self._eval_split_excludes_fewshot = False
        self._prompt_cache: dict[tuple[str, str], str] = {}

    @override
    async def setup(self) -> None:
        if self.n_shot > 0:
            self._ensure_fewshot_pool()

    @override
    async def preprocess(
        self,
        raw: MMMLUDatasetSample,
        ctx: TaskContext[
            MMMLUDatasetSample,
            PromptRecord,
            TaskStageOutput[OfficialScores],
            PredictionRecord,
            JudgementRecord,
        ],
    ) -> PromptRecord:
        train_prompt = self._build_train_prompt(raw)
        prompt_end = _format_example(raw, include_answer=False)
        return build_prompt_record(
            train_prompt + prompt_end,
            reference=raw["Answer"],
            extra={
                "locale": self._locale(raw),
                "category": self._category(raw),
                "subject": self._subject(raw),
            },
        )

    @override
    async def infer(
        self,
        pre: PromptRecord,
        ctx: TaskContext[
            MMMLUDatasetSample,
            PromptRecord,
            TaskStageOutput[OfficialScores],
            PredictionRecord,
            JudgementRecord,
        ],
    ) -> TaskStageOutput[OfficialScores]:
        # CLP reads only the generated next token's top-logprobs, never the
        # echoed prompt (echo is for scoring a supplied continuation, i.e. true
        # PPL). echo=False is bit-identical here, faster, and prefix-cache-safe.
        output = await self.model.alogprobs(
            # A record's `prompt` is JSONValue (whatever shape the model kind
            # takes); this is a base-model task, so it is always the assembled
            # string preprocess built.
            cast(str, pre["prompt"]),
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
            PromptRecord,
            TaskStageOutput[OfficialScores],
            PredictionRecord,
            JudgementRecord,
        ],
    ) -> PredictionRecord:
        logprobs = inf.value["logprobs"]
        return build_prediction_record(
            [max(CHOICES, key=lambda choice: logprobs[choice])]
        )

    @override
    async def feedback(
        self,
        post: PredictionRecord,
        ctx: TaskContext[
            MMMLUDatasetSample,
            PromptRecord,
            TaskStageOutput[OfficialScores],
            PredictionRecord,
            JudgementRecord,
        ],
    ) -> tuple[bool, JudgementRecord]:
        raw = ctx.raw_sample
        prediction = post["rollouts"][0].get("prediction")
        if raw is None:
            # No sample to grade against. The empty grouping keys are what the
            # pre-migration shape recorded, and report() maps them to "unknown".
            return True, build_judgement_record(
                "",
                [build_rollout_judgement(0, False)],
                extra={"subject": "", "category": "", "locale": ""},
            )
        probs = ctx.infer_result.value["probs"] if ctx.infer_result else {}
        answer = raw["Answer"]
        # The per-option probabilities are the mechanism behind the argmax, not
        # metrics measuring the answer, so they stay in `extra` (and as one
        # mapping rather than four flat prob_A..prob_D keys).
        return True, build_judgement_record(
            answer,
            [
                build_rollout_judgement(
                    0,
                    prediction == answer,
                    extra={"probs": {c: probs.get(c, 0.0) for c in CHOICES}},
                )
            ],
            extra={
                "subject": self._subject(raw),
                "category": self._category(raw),
                "locale": self._locale(raw),
            },
        )

    @override
    async def report(
        self,
        finals: list[
            TaskContext[
                MMMLUDatasetSample,
                PromptRecord,
                TaskStageOutput[OfficialScores],
                PredictionRecord,
                JudgementRecord,
            ]
        ],
        fails: list[
            TaskContext[
                MMMLUDatasetSample,
                PromptRecord,
                TaskStageOutput[OfficialScores],
                PredictionRecord,
                JudgementRecord,
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
                correct = ctx.feedback_result["rollouts"][0]["correct"]
                locale = ctx.feedback_result["extra"]["locale"]
                category = ctx.feedback_result["extra"]["category"]
                subject = ctx.feedback_result["extra"]["subject"]
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

    def _ensure_fewshot_pool(self) -> None:
        if self.n_shot <= 0:
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
            if len(examples) < self.n_shot:
                examples.append(sample)
                if self._fewshot_split == "test":
                    source_indices.add(index)

        if self._fewshot_split == "test":
            for group, examples in fewshot_by_group.items():
                if len(examples) < self.n_shot:
                    raise ValueError(
                        "MMMLU official k-shot evaluation requires at least "
                        f"{self.n_shot} {self._fewshot_split!r} examples for locale "
                        f"{group[0]!r}, subject {group[1]!r}; found {len(examples)}."
                    )

            for group, total in group_totals.items():
                if total <= self.n_shot:
                    raise ValueError(
                        "MMMLU test-split few-shot evaluation requires at least "
                        f"{self.n_shot + 1} test examples for locale {group[0]!r}, "
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
        if self.n_shot <= 0:
            return []
        if self._fewshot_by_group is None:
            self._ensure_fewshot_pool()

        fewshot_by_group = self._fewshot_by_group
        if fewshot_by_group is None:
            raise RuntimeError("MMMLU few-shot pool was not initialized.")

        examples = fewshot_by_group.get(group, [])
        if len(examples) < self.n_shot:
            raise ValueError(
                "MMMLU official k-shot evaluation requires at least "
                f"{self.n_shot} {self._fewshot_split!r} examples for locale "
                f"{group[0]!r}, subject {group[1]!r}; found {len(examples)}."
            )
        return examples[: self.n_shot]

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
