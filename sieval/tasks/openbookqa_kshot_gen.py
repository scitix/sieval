"""OpenBookQA k-shot generative task (instruct/chat models).

Accuracy metric; the predicted option letter is extracted with OpenCompass's
``first_option_postprocess(options="ABCD")``. That extractor and the prompt
template are vendored from OpenCompass ``obqa_gen_9069e4`` (``main`` variant)
in ``sieval.community.openbookqa``. This task targets implementation parity
with that config (prompt + extraction), not a specific published accuracy.

Deviations from the OpenCompass reference (``obqa_gen_9069e4``):
  - OpenCompass uses ``ZeroRetriever`` (0-shot). At ``n_shot=0`` the prompt and
    extraction match upstream; ``n_shot>0`` is a sieval extension with no upstream
    counterpart — the few-shot block is the first ``n_shot`` ``train`` rows (fixed
    indices), each with its ``answerKey`` appended. By default they are packed
    into one user turn (the lm-eval/OpenCompass default); when
    ``fewshot_as_multiturn`` is set they are rendered as alternating
    user/assistant turns instead (lm-eval's ``fewshot_as_multiturn``).
    At ``n_shot>0`` generation is bounded by a stop sequence (the next example's
    ``Question:`` header) so a verbose run-on cannot emit a later high-priority
    extractor match that overrides the real answer; ``n_shot=0`` is left unbounded to
    match the upstream 0-shot config.
  - Only the ``main`` variant is implemented; the ``additional``/``fact1``
    ("Given the fact: ...") prompt variant is not used.
  - Choices map to A–D by position (``choices["text"][0..3]``), matching
    OpenCompass ``OBQADataset``.

Repro decoding: greedy ``temperature=0``, ``top_p=1``. ``obqa_gen_9069e4`` sets
no ``max_out_len``, so ``max_gen_toks`` follows the model/run config rather than
a task-pinned value.

AI-Generated Code - Opus 4.8 (Anthropic)
"""

from typing import override

from openai.types.chat import ChatCompletionMessageParam

from sieval.community.openbookqa import (
    OBQA_OPTIONS,
    OBQA_PROMPT_TEMPLATE,
    first_option_postprocess,
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
from sieval.datasets import OpenBookQADatasetSample

DEFAULT_N_SHOT = 0
FEWSHOT_SEP = "\n\n"
# Coupled to the few-shot block: each packed example begins with "Question:".
# Applied only at n_shot>0 to bound verbose run-on (see infer); n_shot=0 is
# left unbounded.
STOP_SEQUENCES = ("\nQuestion:",)


def _format_question(sample: OpenBookQADatasetSample) -> str:
    texts = sample["choices"]["text"]
    return OBQA_PROMPT_TEMPLATE.format(
        question_stem=sample["question_stem"],
        A=texts[0],
        B=texts[1],
        C=texts[2],
        D=texts[3],
    )


@sieval_task(
    name="openbookqa_kshot_gen",
    display_name="OpenBookQA (k-shot, generative)",
    description="OpenBookQA elementary-science MCQ, generative letter extraction.",
    eval_mode=EvalMode.GEN,
    n_shot=DEFAULT_N_SHOT,
    tags=("english", "science", "multiple-choice"),
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="opencompass",
        url="https://github.com/open-compass/opencompass/blob/5767b74899806c0c37efdc5529ffea01e7340e48/opencompass/configs/datasets/obqa/obqa_gen_9069e4.py",
        notes=(
            "Prompt template (main variant) and first_option_postprocess "
            "vendored from OpenCompass. At n_shot=0 the prompt and extraction match "
            "the upstream 0-shot config; n_shot>0 is a sieval extension (fixed "
            "first n_shot train rows)."
        ),
    ),
)
class OpenBookQAFewShotGenTask(
    Task[
        OpenBookQADatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        dict[str, float],
    ]
):
    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        *,
        n_shot: int = DEFAULT_N_SHOT,
        fewshot_split: str = "train",
        fewshot_as_multiturn: bool = False,
        stop: tuple[str, ...] = STOP_SEQUENCES,
    ):
        if n_shot < 0:
            raise ValueError(f"n_shot must be >= 0, got {n_shot}")
        super().__init__(dataset=dataset, model=model, name=name)
        self.n_shot = n_shot
        self._fewshot_split = fewshot_split
        self._fewshot_as_multiturn = fewshot_as_multiturn
        self._stop = stop
        self._fewshot_prefix: str = ""
        self._fewshot_turns: list[ChatCompletionMessageParam] = []

    @override
    async def setup(self) -> None:
        examples = self._retrieve_fewshot()
        if self._fewshot_as_multiturn:
            turns: list[ChatCompletionMessageParam] = []
            for ex in examples:
                turns.append({"role": "user", "content": _format_question(ex)})
                turns.append({"role": "assistant", "content": ex["answerKey"]})
            self._fewshot_turns = turns
        else:
            rendered = [f"{_format_question(ex)} {ex['answerKey']}" for ex in examples]
            self._fewshot_prefix = (
                FEWSHOT_SEP.join(rendered) + FEWSHOT_SEP if rendered else ""
            )

    @override
    async def preprocess(self, raw, ctx):
        query = _format_question(raw)
        # Annotated because the two branches infer different message-list types,
        # and a record's `prompt` is JSONValue -- whatever shape the model kind takes.
        prompt: list
        if self._fewshot_as_multiturn:
            prompt = [*self._fewshot_turns, {"role": "user", "content": query}]
        else:
            prompt = [{"role": "user", "content": self._fewshot_prefix + query}]
        return build_prompt_record(prompt, reference=raw["answerKey"])

    @override
    async def infer(self, pre, ctx):
        # At n_shot>0 the packed few-shot block primes verbose models to run on and
        # re-answer bundled examples; because the extractor scans the whole text
        # by pattern priority (not first-by-position), a trailing match can then
        # override the real leading answer. Bound generation at the next example
        # boundary. n_shot=0 stays unbounded to match the upstream 0-shot config.
        if self.n_shot > 0 and self._stop:
            return await self.model.agenerate(pre["prompt"], stop=list(self._stop))
        return await self.model.agenerate(pre["prompt"])

    @override
    async def postprocess(self, inf, ctx):
        # n=1, only one choice. No option found -> None, so `extracted` reports
        # the miss; "" and None both fail the letter comparison below.
        return build_prediction_record(
            [first_option_postprocess(inf.texts[0], OBQA_OPTIONS) or None]
        )

    @override
    async def feedback(self, post, ctx):
        answer = ctx.raw_sample["answerKey"]
        prediction = post["rollouts"][0].get("prediction")
        return True, build_judgement_record(
            answer, [build_rollout_judgement(0, prediction == answer)]
        )

    @override
    async def report(self, finals, fails):
        correct = sum(
            1 for ctx in finals if ctx.feedback_result["rollouts"][0]["correct"]
        )
        accuracy = 100 * correct / len(finals) if finals else 0.0
        # `score` is the headline; `accuracy` names the metric behind it
        # (% of finalized samples whose extracted letter equals answerKey),
        # mirroring how gsm8k/drop surface their metric alongside `score`.
        return {"score": accuracy, "fails": len(fails), "accuracy": accuracy}

    def _retrieve_fewshot(self) -> list[OpenBookQADatasetSample]:
        if self.n_shot <= 0:
            return []
        split = self.dataset.dataset_dict.get(self._fewshot_split)
        if split is None:
            raise ValueError(
                "OpenBookQA few-shot generative task requires a "
                f"{self._fewshot_split!r} split for few-shot examples."
            )
        # retrieve_samples clips out-of-range indices, which would render fewer
        # shots than meta.json reports with nothing on disk
        # saying so. Fail instead, as the other few-shot tasks do. Called from
        # setup(), so this aborts before any inference spend.
        if len(split) < self.n_shot:
            raise ValueError(
                "OpenBookQA few-shot generative task requires at least "
                f"{self.n_shot} examples in split {self._fewshot_split!r}; "
                f"found {len(split)}."
            )
        return self.dataset.retrieve_samples(
            self.n_shot,
            split=self._fewshot_split,
            mode="fixed",
            indices=list(range(self.n_shot)),
        )
