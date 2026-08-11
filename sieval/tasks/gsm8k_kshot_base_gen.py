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
Default n_shot=8 is chosen for that DeepSeek-V3 comparison.

Decoding parameters such as max_tokens and temperature are model-owned in
SiEval; this task only forwards stop sequences coupled to the prompt format.

AI-Generated Code - GPT-5.5 (OpenAI)
"""

import re
from typing import override

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
    DENOMINATOR_JUDGED,
    SCORE_KEY_FIELD,
    health_metrics,
    sampling_report,
)
from sieval.datasets import GSM8KDatasetSample

from ._math_verify import normalize_vote

N_SHOT = 8
DEFAULT_FEWSHOT_SEED = 1234
STOP_SEQUENCES = ("Question:", "</s>", "<|im_end|>")

_STRICT_ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
_FLEXIBLE_ANSWER_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


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


def _named(finals, metric: str) -> int:
    """How many judged samples the FIRST rollout got right on *metric*.

    Reads the sample-level `metrics` block, which `feedback` fills from rollout 0
    -- the axis lm-eval-harness measured. Tolerant of a missing block so a run
    interrupted mid-feedback reports a number rather than a KeyError.
    """
    return sum(
        1
        for ctx in finals
        if ((ctx.feedback_result or {}).get("metrics") or {}).get(metric)
    )


@sieval_task(
    name="gsm8k_kshot_base_gen",
    display_name="GSM8K (few-shot, base generative)",
    description="GSM8K few-shot base-model strict-match EM evaluation.",
    eval_mode=EvalMode.GEN,
    n_shot=N_SHOT,
    tags=("english", "math-word-problems", "open-ended", "base-model"),
    model_type="gen",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="lm-evaluation-harness",
        url=(
            "https://github.com/EleutherAI/lm-evaluation-harness/blob/1dd931087362abba74e0375c8c631295559f48b2/lm_eval/tasks/gsm8k/gsm8k.yaml"
        ),
        notes=(
            "Uses lm-eval-harness strict-match extraction, aligned with the "
            "original GSM8K dataset.py #### answer delimiter. Also reports "
            "lm-eval-harness flexible-extract as a secondary metric, not as "
            "the original GSM8K official metric. Default n_shot=8 follows the "
            "DeepSeek-V3 Table 3 comparison target (Qwen2.5-72B-Base GSM8K "
            "8-shot EM = 88.3); DeepSeek-V3 does not specify strict vs "
            "flexible extraction."
        ),
    ),
)
class GSM8KFewShotBaseGenTask(
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
    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        *,
        n_shot: int = N_SHOT,
        fewshot_split: str = "train",
        fewshot_seed: int = DEFAULT_FEWSHOT_SEED,
        k: int = 1,
        n: int = 1,
        stop: tuple[str, ...] = STOP_SEQUENCES,
    ):
        if n_shot < 0:
            raise ValueError(f"n_shot must be >= 0, got {n_shot}")
        if k > n:
            raise ValueError(
                f"pass@{k} needs at least {k} sample(s) per problem, got n={n}."
            )
        super().__init__(dataset=dataset, model=model, name=name)
        self._k = k
        self._n = n
        self.n_shot = n_shot
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
        prompt = "".join(
            _format_example(ex, include_answer=True) for ex in examples
        ) + (_format_example(raw, include_answer=False))
        return build_prompt_record(prompt, reference=_extract_answer(raw["answer"])[0])

    @override
    async def infer(self, pre, ctx):
        # Keep `stop` out of the kwargs when unset so it can't clobber the
        # model's configured stop via the `{**self._kwargs, **kwargs}` merge.
        # `n` always goes: it is the sampling budget `k` was validated against
        # (sieval/tasks/CLAUDE.md, "n_shot vs k").
        if self._stop:
            return await self.model.agenerate(
                pre["prompt"], n=self._n, stop=list(self._stop)
            )
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        # The flexible match is a second extraction RULE over the same
        # response, not a second rollout. PER-ROLLOUT because it is a fact
        # about one response: in the sample-level slot it would silently mean
        # "rollout 0's" the moment n > 1.
        predictions: list = []
        extras: list[dict | None] = []
        for text in inf.texts:
            answer, extraction_method = _extract_answer(text)
            flexible_answer, flexible_extraction_method = _extract_flexible_match(text)
            predictions.append(answer or None)
            extras.append(
                {
                    "flexible_prediction": flexible_answer,
                    "extraction_method": extraction_method,
                    "flexible_extraction_method": flexible_extraction_method,
                }
            )
        return build_prediction_record(predictions, extras=extras)

    @override
    async def feedback(self, post, ctx):
        answer, _ = _extract_answer(ctx.raw_sample["answer"])
        # Strict and flexible extraction are co-equal published metrics over one
        # response, so both are named in `metrics`; `correct` is DERIVED from the
        # strict one (the headline `exact_match`) so the two cannot drift.
        rollouts = []
        for rollout in post["rollouts"]:
            extra = rollout.get("extra") or {}
            metrics: dict[str, bool | float] = {
                "exact_match": (rollout.get("prediction") or "") == answer,
                "flexible_exact_match": extra.get("flexible_prediction") == answer,
            }
            rollouts.append(
                build_rollout_judgement(
                    rollout["index"],
                    bool(metrics["exact_match"]),
                    metrics=metrics,
                    extra={
                        "extraction_method": extra.get("extraction_method"),
                        "flexible_extraction_method": extra.get(
                            "flexible_extraction_method"
                        ),
                    },
                )
            )
        # Sample-level `metrics` stays the FIRST rollout's, which is the axis the
        # lm-eval-harness numbers this port reproduces were measured on.
        first = dict(rollouts[0]["metrics"]) if rollouts else {}
        return True, build_judgement_record(answer, rollouts, metrics=first or None)

    @override
    async def report(self, finals, fails):
        count = len(finals)
        # First-rollout, because that is what lm-eval-harness published (one
        # greedy draw). The sampling metrics below never touch them.
        correct_num = _named(finals, "exact_match")
        flexible_correct_num = _named(finals, "flexible_exact_match")
        exact_match = 100 * correct_num / count if count else 0.0
        flexible_exact_match = 100 * flexible_correct_num / count if count else 0.0
        metrics: dict[str, float | str] = {
            "score": exact_match,
            "fails": len(fails),
            "exact_match": exact_match,
            "flexible_exact_match": flexible_exact_match,
            SCORE_KEY_FIELD: "exact_match",
            DENOMINATOR_FIELD: DENOMINATOR_JUDGED,
        }
        # Outside the gate: extraction health is a fact about the parser,
        # not the draw, and n=1 is where a stopped extractor hides longest.
        metrics |= health_metrics(finals)
        if self._n <= 1:
            return metrics
        # The sampling family rides on `correct`, derived from the strict
        # `exact_match`, so it describes the HEADLINE metric only --
        # `flexible_exact_match` would need its own verdict axis.
        # Over `len(finals)`: this task excludes failed samples.
        return metrics | sampling_report(
            finals,
            n=self._n,
            k=self._k,
            denominator=count,
            normalize=normalize_vote,
        )

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
        if len(split) < self.n_shot:
            raise ValueError(
                "GSM8K few-shot base generative task requires at least "
                f"{self.n_shot} examples in split {self._fewshot_split!r}; "
                f"found {len(split)}."
            )
        if self.n_shot == 0:
            return []
        return self.dataset.retrieve_samples(
            self.n_shot,
            split=self._fewshot_split,
            mode="random",
            seed=self._fewshot_seed,
        )
