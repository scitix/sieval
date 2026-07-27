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
it and scores the reply. The RULER answer-cue (``answer_prefix``) is prefilled
in an assistant turn (when continue_final_message: true is set), validated
against NVIDIA/RULER at commit ab17b78. An alternative user-message append
mode is supported for thinking model compatibility.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

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
        notes=(
            "Scoring mirrors RULER's string_match_all (recall) and "
            "string_match_part (QA), vendored in community/ruler/eval. Verified "
            "byte-exact against NVIDIA/RULER at ab17b78 and reproduces published "
            "recall on a Qwen3-8B run (repro table in the PR). Generation is "
            "capped per subtask to upstream's tokens_to_generate (via the sample's "
            "gen_budget, see infer()), and HotpotQA document ordering matches "
            "upstream's sorted(set()). Remaining divergences, each score-neutral:\n"
            "1. Paul Graham essay corpus: the BYO generator concatenates essays "
            "in the pinned URL-list order, while upstream groups them repo-then-"
            "html (glob order within each group), so the full-corpus bytes differ. "
            "The generator also escapes markdown via html2text's escape_snob=True "
            "(gen_paul_graham_essays.py), while upstream's escape_all is a no-op "
            "(unescaped), so per-essay bytes differ too. NIAH/VT needles and QA "
            "answers don't depend on filler-text content or order, so recall is "
            "unchanged.\n"
            "2. CWE word count is capped at the wonderwords pool, which fills "
            ">=98% only at <=32k; at 64k/128k CWE underfills. The stable contract "
            "is scoped to context lengths <=32k; 64k/128k are experimental.\n"
            "3. HotpotQA (_read_hotpotqa in datasets/ruler/_qa.py) is sourced from "
            "the HF mirror (hotpotqa/hotpot_qa), not upstream's derived json, and "
            "questions are taken in the HF dataset's native row order and sliced "
            "to the first N. Per-index question selection can therefore differ "
            "from upstream even though document ordering (sorted(set()), used for "
            "distractor indexing) matches. Scoring is unaffected: each sample's "
            "answer is looked up from its own question, not from upstream's index.\n"
            "Separately, not a divergence: CWE recall drops sharply under thinking "
            "mode (e.g. non-thinking 8k ~98.3 vs thinking 8k ~69.1). Upstream's "
            "tokens_to_generate=120 cap for this subtask assumes a terse, "
            "list-only answer; thinking-mode predictions are more verbose and get "
            "truncated mid-list before the answer completes. This reproduces "
            "upstream's own cap, so a low thinking-mode CWE score is expected, "
            "not a bug."
        ),
    ),
)
class RulerZeroShotGenTask(
    Task[
        RulerDatasetSample,
        list[ChatCompletionMessageParam],
        ModelOutput,
        str,
        RulerFeedback,
        dict[str, float],
    ]
):
    async def preprocess(self, raw, ctx):  # noqa: ARG002
        # Support both message patterns:
        # 1. User-message pattern: answer_prefix appended to user message
        # 2. Assistant-message pattern: answer_prefix in prefilled assistant turn
        #
        # Detection logic:
        # - If both flags in extra_body → assistant pattern
        # - Otherwise → user message pattern (default)
        model_meta = self.model.meta()
        extra_body: dict = model_meta.get("default_params", {}).get(  # type: ignore[assignment]
            "extra_body", {}
        )
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
            prefill = thinking_prefill(model_meta["model"], enable_thinking)
            assistant_content = f"{prefill}{raw['answer_prefix']}"
            return [
                {"role": "user", "content": raw["input"]},
                {"role": "assistant", "content": assistant_content},
            ]
        else:
            # User-message pattern: answer_prefix appended to user message (default)
            return [
                {
                    "role": "user",
                    "content": raw["input"] + raw["answer_prefix"],
                },
            ]

    async def infer(self, pre, ctx):
        # Cap generation per subtask, matching NVIDIA/RULER's tokens_to_generate
        # (128 NIAH / 30 VT / 120 CWE / 50 FWE / 32 QA). The dataset stamps the
        # per-subtask budget on each sample (`gen_budget` = base + any thinking
        # overhead), so one class serving all 13 subtasks applies the right cap
        # without a per-subtask YAML infer_args.
        max_tokens = ctx.raw_sample["gen_budget"]

        # Qwen3 extended thinking adaptation: allocate max_tokens based on
        # context_length. (Diverges from upstream RULER's single-budget approach)
        #
        # During dataset generation, gen_budget was computed differently based on
        # context_length (see _shared.py:tokens_to_generate for rationale):
        # - Small contexts (!=128k): gen_budget excludes think_budget
        # - Large contexts (128k): gen_budget includes think_budget
        #
        # Now at inference, restore think_budget for small contexts where it was
        # omitted.
        if (
            ctx.raw_sample.get("enable_thinking", False)
            and "think_budget" in ctx.raw_sample
            and ctx.raw_sample.get("context_length") != 131072
        ):
            # Small context: gen_budget didn't account for thinking tokens,
            # so add think_budget to ensure sufficient generation capacity.
            max_tokens += ctx.raw_sample["think_budget"]
        # Large context: gen_budget already includes think_budget, no adjustment needed.

        return await self.model.agenerate(pre, max_tokens=max_tokens)

    async def postprocess(self, inf, ctx):  # noqa: ARG002
        return inf.texts[0]

    async def feedback(self, post: str, ctx) -> tuple[bool, RulerFeedback]:  # noqa: ARG002
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
