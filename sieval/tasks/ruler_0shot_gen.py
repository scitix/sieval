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

from sieval.community.ruler.eval.constants import (
    string_match_all,
    string_match_part,
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
from sieval.core.tasks.metrics import (
    DENOMINATOR_FIELD,
    DENOMINATOR_JUDGED,
    SCORE_KEY_FIELD,
)
from sieval.datasets.ruler import RulerDatasetSample, len_tag, thinking_prefill

_QA_SUBTASKS: frozenset[str] = frozenset({"qa_squad", "qa_hotpotqa"})


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
    reference_kind="value",
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
            "NIAH/VT needles and QA answers don't depend on filler-text content or "
            "order, so recall is unchanged.\n"
            "2. HotpotQA (_read_hotpotqa in datasets/ruler/_qa.py) is sourced from "
            "the HF mirror (hotpotqa/hotpot_qa), not upstream's derived json, and "
            "questions are taken in the HF dataset's native row order and sliced "
            "to the first N. Per-index question selection can therefore differ "
            "from upstream even though document ordering (sorted(set()), used for "
            "distractor indexing) matches. Scoring is unaffected: each sample's "
            "answer is looked up from its own question, not from upstream's index.\n"
            "Separately, a real divergence with material score impact — not the "
            "harmless cap-reproduction it might look like. CWE's 124-token "
            "answer budget (120 upstream base + 4 Qwen3 <think> tag overhead) "
            "is upstream's own tokens_to_generate=120, inherited byte-exact; "
            "sieval did not shrink it. But upstream calibrated 120 against its "
            "own terse, non-thinking answers — it has no thinking mode, so it "
            "never tested this budget against thinking-mode's more verbose "
            "output. A 10-item markdown list needs roughly 150 tokens, so the "
            "answer gets cut off mid-list (finish_reason=length) before naming "
            "the top-10. Measured truncation rate and recall split by completed "
            "vs truncated, across six thinking runs (n=500 each):\n"
            "  length:            4k     8k    16k    32k    64k   128k\n"
            "  truncated/500:    469    482    484    468    437    363\n"
            "  recall completed: 100.0  88.9   95.6   82.8   55.9   13.0\n"
            "  recall truncated:  83.1  68.3   53.8   56.9   42.9   14.4\n"
            "niah_single_1 and vt truncate 0/500 on these same runs, so this is "
            "CWE-specific verbosity, not a general thinking-mode issue. At "
            "4k-64k, completed recall stays far above truncated recall — real "
            "budget starvation. At 128k the gap closes (13.0 vs 14.4) — that "
            "length is genuinely capability-limited, not truncation-limited.\n"
            "Score impact: replacing 4k-64k CWE scores with their "
            "completed-only recall (i.e. as if untruncated) moves the "
            "thinking-mode mean from 86.86 to ~88.3 — the gap from the 84.4 "
            "reference widens from +2.91% to ~+4.6%, outside the <3% band this "
            "PR otherwise holds to. Stated plainly, not as expected behavior. "
            "Whether to give the answer a larger budget (breaking byte-exact "
            "alignment with upstream's 120) is a separate decision; this note "
            "exists so the tradeoff is visible."
        ),
    ),
)
class RulerZeroShotGenTask(
    Task[
        RulerDatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        # `float | str`: the report carries `score_key`, which names a column
        # rather than measuring one. `int` matches what `report` already returns
        # (`fails`), which this slot had drifted from.
        dict[str, float | int | str],
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
        # default_params values are JSONValue, so a misconfigured YAML can put a
        # non-mapping under extra_body; narrow instead of suppressing the type
        # error, otherwise the bad config surfaces as an AttributeError in here.
        raw_extra_body = model_meta.get("default_params", {}).get("extra_body")
        extra_body: dict = raw_extra_body if isinstance(raw_extra_body, dict) else {}
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
            prompt = [
                {"role": "user", "content": raw["input"]},
                {"role": "assistant", "content": assistant_content},
            ]
        else:
            # User-message pattern: answer_prefix appended to user message (default)
            prompt = [
                {
                    "role": "user",
                    "content": raw["input"] + raw["answer_prefix"],
                },
            ]
        return build_prompt_record(
            prompt,
            reference=list(raw["outputs"]),
            extra={
                "subtask": raw["subtask"],
                "context_length": raw["context_length"],
            },
        )

    async def infer(self, pre, ctx):
        # Cap generation per subtask, matching NVIDIA/RULER's tokens_to_generate
        # (128 NIAH / 30 VT / 120 CWE / 50 FWE / 32 QA). The dataset stamps the
        # per-subtask budget on each sample (`gen_budget` = base + any thinking
        # overhead), so one class serving all 13 subtasks applies the right cap
        # without a per-subtask YAML infer_args.
        max_tokens = ctx.raw_sample["gen_budget"]

        # Qwen3 extended thinking adaptation (diverges from upstream RULER's
        # single-budget approach): whether gen_budget already covers think_budget
        # is a packing-time decision the loader owns, so read the flag it stamped
        # rather than re-deriving it here. Re-deriving is what let the two sides
        # disagree once `reserve_think_budget` became configurable — a reserving
        # dataset would get think_budget added a second time, overflowing the very
        # window the reserve was meant to fit.
        # `_stamp` writes think_budget and think_budget_reserved together, so
        # indexing the flag directly is safe here and turns a future loader that
        # forgets to stamp it into a loud KeyError rather than a silently wrong
        # budget.
        if (
            ctx.raw_sample.get("enable_thinking", False)
            and "think_budget" in ctx.raw_sample
            and not ctx.raw_sample["think_budget_reserved"]
        ):
            # Not reserved while packing: the prompt fills max_seq_length leaving
            # room only for the answer, so thinking tokens come out of the serving
            # window's headroom and must be added to max_tokens here.
            max_tokens += ctx.raw_sample["think_budget"]
        # Reserved while packing: gen_budget already includes think_budget.

        return await self.model.agenerate(pre, max_tokens=max_tokens)

    async def postprocess(self, inf, ctx):  # noqa: ARG002
        # Open-ended retrieval: the response *is* the answer. A blank response
        # normalizes to None so `extracted` stays a real signal; the matcher below
        # still sees "" and scores it 0.
        text = inf.texts[0]
        return build_prediction_record([text if text.strip() else None])

    async def feedback(self, post, ctx) -> tuple[bool, JudgementRecord]:
        """Score this sample against its references.

        RULER upstream only ever produced a per-CELL number: `string_match_*`
        take whole lists and average internally, so before this migration no
        sample carried a verdict of its own. Both metrics decompose exactly --
        `string_match_all` is the mean of per-sample `matched / len(refs)` and
        `string_match_part` the mean of per-sample `any(matched)` -- so the
        per-sample term is computed here and report() averages the stored values.
        Same arithmetic, same floats, and now every row is inspectable.
        """
        prediction = post["rollouts"][0].get("prediction") or ""
        references = list(ctx.raw_sample["outputs"])
        subtask = ctx.raw_sample["subtask"]
        lowered = prediction.lower()
        hits = [1.0 if r.lower() in lowered else 0.0 for r in references]
        if subtask in _QA_SUBTASKS:
            # string_match_part: credit for finding ANY reference.
            score = max(hits) if hits else 0.0
        else:
            # string_match_all: the fraction of references found.
            score = sum(hits) / len(hits) if hits else 0.0
        return True, build_judgement_record(
            references,
            [
                build_rollout_judgement(
                    0,
                    # The strict binary reading: every reference the metric asks
                    # for was found. `score` carries the partial credit.
                    score == 1.0,
                    score=score,
                )
            ],
            score=score,
            extra={
                "subtask": subtask,
                "context_length": ctx.raw_sample["context_length"],
                "n_references": len(references),
            },
        )

    async def report(self, finals: list, fails: list) -> dict[str, float | int | str]:
        # Cell scores still go through the VENDORED string_match_* functions on the
        # whole cell, unchanged. The per-sample `score` on each judgement is the
        # decomposed term of the same formula, recorded for inspectability, but it
        # is deliberately NOT what is aggregated here: re-deriving the cell mean in
        # this file would fork upstream's scoring, and a vendored metric that no
        # longer runs is a reproduction that drifts silently.
        cells: dict[tuple[int, str], list[tuple[str, list[str]]]] = defaultdict(list)
        for ctx in finals:
            fb = ctx.feedback_result
            # `.get()`, not `[]`: postprocess normalizes a blank response to None,
            # and a None field is ABSENT once the record round-trips through disk --
            # which is exactly what this reads on the resume-report path.
            prediction = ctx.postprocess_result["rollouts"][0].get("prediction") or ""
            cells[(fb["extra"]["context_length"], fb["extra"]["subtask"])].append(
                (prediction, list(fb["reference"]))
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

        # The headline is a macro over context lengths, so no single `score_*`
        # breakdown key is its source — `score` names itself. The denominator is
        # the judged set: a cell is scored over the samples that reached it, and a
        # pipeline failure is reported in `fails` rather than folded into a cell as
        # a zero-scoring sample.
        result: dict[str, float | int | str] = {
            "score": overall,
            "fails": len(fails),
            SCORE_KEY_FIELD: "score",
            DENOMINATOR_FIELD: DENOMINATOR_JUDGED,
        }
        for ctx_len, mean_score in sorted(length_means.items()):
            result[f"score_{len_tag(ctx_len)}"] = mean_score
        for (ctx_len, subtask), score in sorted(cell_scores.items()):
            result[f"score_{subtask}_{len_tag(ctx_len)}"] = score
        return result
