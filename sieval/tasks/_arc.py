"""
Shared helpers for ARC multiple-choice base-model tasks.

Two scoring protocols (see ``sieval/tasks/CLAUDE.md`` §ppl-vs-clp):
- ``ppl`` (separation): score each option's full TEXT as a continuation of a
  bare ``"Answer:"`` and pick the highest unconditionally-normalized likelihood.
- ``clp`` (options): list the options ``A/B/C/...`` in the prompt and pick the
  option letter with the highest next-token log-prob from ``top_logprobs``.

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

from collections.abc import Mapping, Sequence
from typing import Any

from sieval.core.datasets import Dataset
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
)
from sieval.core.utils.ppl import total_logprob

DEFAULT_FEWSHOT_SEED = 1234

# Unconditional-normalization context (Brown et al. 2020) for the ppl protocol:
# score options relative to their likelihood under a bare "Answer:" prefix.
ARC_UNCOND_CONTEXT = "Answer:"

# clp top-k: must be large enough that every option letter is in top_logprobs.
# SGLang serves 100 by default; on vLLM start with --max-logprobs 100.
DEFAULT_CLP_LOGPROBS = 100


def choice_label(index: int) -> str:
    return chr(ord("A") + index)


def choice_text(choices: list[str], index: int | None) -> str:
    """The option text at *index*, or ``""`` when there is no such option.

    Accepts ``None`` because a protocol prediction is ``None`` when scoring could
    not pick an option; that reads the same as an out-of-range index here.
    """
    if index is None:
        return ""
    return choices[index] if 0 <= index < len(choices) else ""


# --------------------------------------------------------------------------- #
# ppl (separation): full-text continuation + unconditional normalization
# --------------------------------------------------------------------------- #
def format_arc_ppl_context(question: str) -> str:
    """Cloze scoring context: the question then a bare ``"Answer:"`` cue.

    Options are NOT listed — each option's full TEXT is scored as the
    continuation (DeepSeek separation / lm-eval ``multiple_choice`` layout).
    The caller appends ``" <option text>"`` per candidate.
    """
    return f"Question: {question}\nAnswer:"


def build_arc_ppl_fewshot_prefix(examples: Sequence[Mapping[str, Any]]) -> str:
    """Join answered exemplars, each ``"Question: {q}\\nAnswer: {correct text}\\n\\n"``.

    The exemplar answer is the correct option's full text (not a letter),
    matching the scored continuation format. Depends only on the exemplar set,
    so callers build it once in ``setup()`` and reuse it across every sample.
    """
    blocks: list[str] = []
    for example in examples:
        choices = list(example["choices"])
        answer = int(example["answer"])
        blocks.append(f"Question: {example['question']}\nAnswer: {choices[answer]}\n\n")
    return "".join(blocks)


# --------------------------------------------------------------------------- #
# clp (options): labeled choices + next-token letter log-prob
# --------------------------------------------------------------------------- #
def format_arc_clp_item(
    question: str, choices: Sequence[str], answer: int | None = None
) -> str:
    """Format one item as labeled options ending in ``"Answer:"``.

    Few-shot exemplars pass *answer* → trailing ``"Answer: <letter>"``; the
    target passes ``answer=None`` → ``"Answer:"`` and the next-token letter is
    scored from ``top_logprobs``.
    """
    body = "\n".join(
        f"{choice_label(index)}. {choice}" for index, choice in enumerate(choices)
    )
    block = f"Question: {question}\n{body}\nAnswer:"
    if answer is not None:
        block += f" {choice_label(answer)}"
    return block


def build_arc_clp_fewshot_prefix(examples: Sequence[Mapping[str, Any]]) -> str:
    """Join labeled exemplars, each ending ``"Answer: <letter>\\n\\n"``."""
    return "".join(
        format_arc_clp_item(
            str(example["question"]),
            list(example["choices"]),
            int(example["answer"]),
        )
        + "\n\n"
        for example in examples
    )


# --------------------------------------------------------------------------- #
# shared: few-shot sampling + accuracy report
# --------------------------------------------------------------------------- #
def sample_arc_fewshot(
    dataset: Dataset[Any],
    k: int,
    fewshot_split: str,
    fewshot_seed: int,
) -> list[Any]:
    """Draw *k* deterministic few-shot exemplars from *fewshot_split*."""
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if k == 0:
        return []
    split = dataset.dataset_dict.get(fewshot_split)
    if split is None:
        raise ValueError(
            "ARC few-shot base task requires a "
            f"{fewshot_split!r} split for few-shot examples."
        )
    if len(split) < k:
        raise ValueError(
            "ARC few-shot base task requires at least "
            f"{k} examples in split {fewshot_split!r}; found {len(split)}."
        )
    return dataset.retrieve_samples(
        k,
        split=fewshot_split,
        mode="random",
        seed=fewshot_seed,
    )


def echoed_logprob(output: ModelOutput) -> float:
    """Sum the logprobs of the ECHOED INPUT tokens only (exclude generation).

    ``alogprobs(echo=True)`` returns the echoed prompt tokens' logprobs followed
    by the generated token(s) (``max_tokens >= 1``, since sglang rejects 0). Only
    the echoed input is the scored sequence; the trailing generated token differs
    between the conditional and unconditional calls and would NOT cancel, so it
    must be dropped. Slice to ``usage.input_tokens`` (the echoed length) before
    summing so the shared context cancels exactly in ``cond - uncond``. Raises if
    ``usage`` is absent rather than fall back to a full-sequence sum (which would
    silently reintroduce the trailing-token bug this function exists to prevent).
    """
    if output.usage is None:
        raise RuntimeError(
            "echoed_logprob requires usage.input_tokens to drop the generated "
            "token from the echoed sequence, but the response carries no usage."
        )
    input_tokens = output.usage["input_tokens"]
    total, _ = total_logprob(
        (output.logprobs_tokens or [])[:input_tokens],
        (output.logprobs or [])[:input_tokens],
    )
    return total


# --------------------------------------------------------------------------- #
# shared: stage-output protocol records
# --------------------------------------------------------------------------- #
# All four ARC tasks (challenge/easy x ppl/clp) build the same three records from
# the same fields, so they are built here rather than four times over. `arc_report`
# reads the judgement, so a task that built its own shape would silently disagree
# with it -- the coupling that makes this module, not the task file, the migration
# unit.
def arc_prompt_record(prompt: str, raw: Mapping[str, Any]) -> PromptRecord:
    """The assembled prompt plus the gold option index.

    The gold is recorded here because ``raw_sample`` is never serialized: without
    it a prompt row on disk has no ground truth at all.
    """
    return build_prompt_record(prompt, reference=int(raw["answer"]))


def arc_prediction_record(index: int | None) -> PredictionRecord:
    """The single scored option index. ``None`` means no option could be picked."""
    return build_prediction_record([index])


def arc_judgement_record(
    prediction: int | None, raw: Mapping[str, Any]
) -> JudgementRecord:
    """The verdict, carrying both option texts so a row reads without the dataset.

    The prediction and reference are option *indices*; the matching texts are the
    only thing on disk that says what an index means, so they ride along as
    mechanism detail.
    """
    answer = int(raw["answer"])
    choices = list(raw["choices"])
    return build_judgement_record(
        answer,
        [
            build_rollout_judgement(
                0,
                prediction == answer,
                extra={"prediction_choice": choice_text(choices, prediction)},
            )
        ],
        extra={"answer_choice": choice_text(choices, answer)},
    )


def arc_report(
    finals,
    fails,
) -> dict[str, float]:
    correct_num = sum(
        1 for ctx in finals if ctx.feedback_result["rollouts"][0]["correct"]
    )
    acc = 100 * correct_num / len(finals) if finals else 0.0
    return {"score": acc, "acc": acc, "fails": len(fails)}
