"""RULER 0-shot generative task.

Handles all 13 RULER subtasks in a single class. The scoring branch is chosen
per sample in ``feedback()`` based on ``subtask``:

- recall subtasks (NIAH × 8, VT, CWE, FWE): ``string_match_all``
- QA subtasks (qa_squad, qa_hotpotqa): ``string_match_part``

``report()`` groups by ``(context_length, subtask)`` to emit:
- per-cell scores: ``score_{subtask}_{len_tag}``
- per-length 13-task means: ``score_{len_tag}``
- overall headline: ``score``

The prompt is fully synthesized in the dataset loader; this task just sends
it and scores the reply. The RULER answer-cue (``answer_prefix``) is appended
directly to the user message so the model produces the answer inline without
needing a prefilled assistant turn.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from abc import ABC
from collections import defaultdict
from typing import TypedDict

from openai.types.chat import ChatCompletionMessageParam

from sieval.community.ruler.eval.constants import (
    string_match_all,
    string_match_part,
)
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets.ruler import RulerDatasetSample, len_tag, thinking_prefill

_QA_SUBTASKS: frozenset[str] = frozenset({"qa_squad", "qa_hotpotqa"})


class RulerFeedback(TypedDict):
    prediction: str
    references: list[str]
    subtask: str
    context_length: int


class _ChatGenBase[TSample, TFeedback](
    Task[
        TSample,
        list[ChatCompletionMessageParam],
        ModelOutput,
        str,
        TFeedback,
        dict[str, float],
    ],
    ABC,
):
    def __init__(self, dataset, model, name: str | None = None):
        super().__init__(dataset=dataset, model=model, name=name)

    async def preprocess(self, raw, ctx):
        # Support both message patterns:
        # 1. User-message pattern: answer_prefix appended to user message
        # 2. Assistant-message pattern: answer_prefix in prefilled assistant turn
        #
        # Detection logic:
        # - If both flags in extra_body → assistant pattern
        # - Otherwise → user message pattern (default)
        extra_body = self.model._kwargs.get("extra_body", {})
        # Detect prefill mode: both flags must be set explicitly to enable prefill
        # - continue_final_message=True: continue from assistant's last message
        # - add_generation_prompt=False: suppress default generation prompt
        # Both must match for assistant-pattern; otherwise defaults to user-message
        use_assistant_prefill = extra_body.get(
            "continue_final_message", False
        ) and not extra_body.get("add_generation_prompt", True)

        if use_assistant_prefill:
            # Assistant-message pattern: prefilled turn with thinking placeholder
            enable_thinking = extra_body.get("enable_thinking", False)
            prefill = thinking_prefill(self.model._model, enable_thinking)
            assistant_content = f"{prefill}{raw['answer_prefix']}"
            return [
                {"role": "user", "content": raw["input"]},
                {"role": "assistant", "content": assistant_content},
            ]
        else:
            # User-message pattern: answer_prefix appended to user message (default)
            return [
                {"role": "user", "content": raw["input"] + raw["answer_prefix"]},
            ]

    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

    async def postprocess(self, inf, ctx):
        return inf.texts[0]


@sieval_task(
    name="ruler_0shot_gen",
    display_name="RULER (0-shot, generative)",
    description=(
        "RULER long-context benchmark: 13 subtasks (NIAH×8, VT, CWE, FWE, QA×2)."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended", "long-context"),
    deps_group="ruler",
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="NVIDIA/RULER",
        url="https://github.com/NVIDIA/RULER/blob/ab17b7853df4e0a30b78cd5d2b463ac7dff6ee13/scripts/eval/synthetic/constants.py",
        notes="Scoring mirrors RULER's string_match_all (recall) and "
        "string_match_part (QA), vendored in community/ruler/eval.",
    ),
)
class RulerZeroShotGenTask(_ChatGenBase[RulerDatasetSample, RulerFeedback]):
    async def feedback(self, post: str, ctx) -> tuple[bool, RulerFeedback]:
        return True, {
            "prediction": post,
            "references": ctx.raw_sample["outputs"],
            "subtask": ctx.raw_sample["subtask"],
            "context_length": ctx.raw_sample["context_length"],
        }

    async def report(self, finals: list, fails: list) -> dict[str, float | int]:
        cells: dict[tuple[int, str], list[tuple[str, list[str]]]] = defaultdict(list)
        for ctx in finals:
            fb: RulerFeedback = ctx.feedback_result
            cells[(fb["context_length"], fb["subtask"])].append(
                (fb["prediction"], fb["references"])
            )

        cell_scores: dict[tuple[int, str], float] = {}
        for (ctx_len, subtask), samples in cells.items():
            preds = [p for p, _ in samples]
            refs = [r for _, r in samples]
            score = (
                string_match_part(preds, refs)
                if subtask in _QA_SUBTASKS
                else string_match_all(preds, refs)
            )
            cell_scores[(ctx_len, subtask)] = score

        by_length: dict[int, list[float]] = defaultdict(list)
        for (ctx_len, _), score in cell_scores.items():
            by_length[ctx_len].append(score)
        length_means = {ctx_len: sum(s) / len(s) for ctx_len, s in by_length.items()}

        overall = (
            sum(length_means.values()) / len(length_means) if length_means else 0.0
        )

        result: dict[str, float | int] = {"score": overall, "fails": len(fails)}
        for ctx_len, mean_score in sorted(length_means.items()):
            result[f"score_{len_tag(ctx_len)}"] = mean_score
        for (ctx_len, subtask), score in sorted(cell_scores.items()):
            result[f"score_{subtask}_{len_tag(ctx_len)}"] = score
        return result
