"""Multi-IF zero-shot generative task (multi-turn, multilingual IFEval).

One sample is one whole conversation. ``infer`` walks its turns in order,
appending the model's own reply before sending the next user turn, so turn *t*
is answered with turns *1..t-1* in context -- upstream's ``old_prompt +
old_response + new_prompt``, which it reaches by re-running the whole set once
per ``--steps`` value and writing the growing conversation back to a CSV. Doing
it in one pass makes every turn of a conversation a single unit of work, which
is what lets one run report all three turns.

Grading is per turn, against that turn's *cumulative* constraint list (turn 3
carries turns 1 and 2's constraints too), under IFEval's strict and loose
readings. Both are co-equal published metrics, so both are recorded.

Upstream's headline per turn is a plain mean of four numbers -- strict and loose
x prompt-level and instruction-level -- reported per language and over all of
them (``turn_1_all_languages_overall``, and so on). The paper quotes these as
fractions (o1-preview 0.877 at turn 1, 0.707 at turn 3); this task reports
percentages, matching its IFEval and IFBench siblings.

Deviations from the official Multi-IF evaluation:

- The 56 conversations with no third turn get no third generation. Upstream
  sends them a literal ``"None"`` prompt and then drops the row when scoring
  turn 3, so the metrics are unchanged and the tokens are not spent.
- Bootstrap confidence intervals are not computed. Upstream resamples every
  language cell 10,000 times via scipy; nothing in SiEval consumes the interval,
  and the per-sample verdicts needed to recompute one are all on disk.
- ``score`` is the mean of the three turns' all-language overalls. Upstream
  emits one report per turn and never reduces them to a single number, but a
  task needs one headline; every component is in the report.

Two upstream defects are tracked rather than repaired, because the unqualified
name must measure what upstream measures. Both make the score nondeterministic:

- Two conversations (6 of 13,447 turn-cells, 0.04%) carry kwargs the checker
  rejects -- ``keywords:letter_frequency`` with ``letter="#"``, and
  ``keywords:frequency`` with no ``keyword``. On a rejected value
  ``build_description`` falls back to ``random.choice`` over the alphabet and
  grades the response against a letter nobody asked for, freshly drawn per call.
- ``langdetect`` is unseeded, so detection -- which selects the word- and
  sentence-counting algorithm behind every length constraint -- can vary between
  runs on short or mixed-script text.

Verified against upstream's own ``metrics_gen`` on 535 conversations spanning
all eight languages and all 56 two-turn rows: 3,098 strict/loose follow-lists
and every per-language ``overall`` agree exactly, once those two conversations
are set aside. They cannot agree with anything, upstream included.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import json
from typing import override

from openai.types.chat import ChatCompletionMessageParam

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
from sieval.core.types import JSONValue
from sieval.datasets import MultiIFDatasetSample

# The two IFEval readings, both published. Neither is subordinate: upstream's
# per-turn headline averages them together rather than picking one.
_GRADES = ("strict", "loose")

# The pooled cell every published Multi-IF number is quoted against; kept
# distinct from the CSV's own `language` values, which are English names.
_ALL_LANGUAGES = "all_languages"


@sieval_task(
    name="multi_if_0shot_gen",
    display_name="Multi-IF (0-shot, generative)",
    description=(
        "Multi-IF — 4,501 three-turn, eight-language instruction-following "
        "conversations extending IFEval."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("multilingual", "multi-turn", "open-ended"),
    deps_group="multi-if",
    model_type="chat",
    # The grader and aggregation are verified against upstream's own evaluator
    # (see the module docstring), but no live run has reproduced a published
    # number yet -- Multi-IF publishes paper scores only, not the per-model
    # inference dumps that let MathArena and PlatinumBench ports claim `stable`
    # by replay. Faithful by construction until a run lands within a stated band.
    status="experimental",
    reference_impl=ReferenceImpl(
        source="facebookresearch/Multi-IF",
        url="https://github.com/facebookresearch/Multi-IF/blob/1cdb53ed18499ad729e0766e5d3099dd5344406f/metrics.py",
        notes=(
            "Multi-IF's own multilingual fork of the IFEval checkers is vendored "
            "(sieval.community.multi_if); the google-research IFEval sibling is "
            "NOT interchangeable with it. Upstream drives one pass per turn "
            "(--steps 1 2 3), one sample per turn; this task walks all three in "
            "one pass. Upstream reports fractions, this task percentages. "
            "Grading matches upstream's metrics_gen exactly — verified on 535 "
            "conversations across all 8 languages (3,098 follow-lists, all "
            "per-language overalls) — except for two conversations upstream "
            "cannot grade reproducibly itself: kwargs it rejects "
            "(letter='#'; missing keyword) send build_description to an unseeded "
            "random.choice. langdetect is likewise unseeded upstream, and picks "
            "the counting algorithm behind every length constraint. Both defects "
            "are tracked, not repaired, per the unqualified-name rule; fixing "
            "either needs a `_fixed` variant with a measured delta."
        ),
    ),
)
class MultiIFZeroShotGenTask(
    Task[
        MultiIFDatasetSample,
        PromptRecord,
        list[ModelOutput],
        PredictionRecord,
        JudgementRecord,
        dict[str, float],
    ]
):
    @override
    async def preprocess(self, raw, ctx):
        turns = raw["turns"]
        return build_prompt_record(
            # Only the opening turn can be built ahead of inference; the rest
            # depend on what the model says. `infer` sends this verbatim and
            # grows the conversation from it, so the recorded prompt is provably
            # the one the model saw.
            [{"role": "user", "content": turns[0]["prompt"]}],
            # The "ground truth" is the constraint set each turn must satisfy,
            # innermost list ordered to match that turn's kwargs.
            reference=[list(turn["instruction_id_list"]) for turn in turns],
            extra={
                "key": raw["key"],
                "language": raw["language"],
                # Says how many prompts this sample really takes, so the single
                # recorded message cannot be misread as the whole input.
                "n_turns": len(turns),
                # The later turns' user text, recorded because it is part of the
                # sample's input and is otherwise only in the raw dataset.
                "later_turn_prompts": [turn["prompt"] for turn in turns[1:]],
            },
        )

    @override
    async def infer(self, pre, ctx):
        turns = ctx.raw_sample["turns"]
        # Seeded from the record rather than rebuilt, so what was persisted and
        # what was sent cannot drift.
        messages: list[ChatCompletionMessageParam] = list(pre["prompt"])
        outputs: list[ModelOutput] = []

        for turn in turns:
            if outputs:
                previous = outputs[-1]
                # An aborted or filtered response has no choices. Feed the empty
                # assistant turn through anyway: the conversation continues (as
                # upstream's does) and the missing answer scores zero on its own
                # turn rather than failing the whole sample.
                messages.append(
                    {
                        "role": "assistant",
                        "content": previous.texts[0] if previous.texts else "",
                    }
                )
                messages.append({"role": "user", "content": turn["prompt"]})
            # n=1 per turn: a second sample would fork the conversation, and
            # every later turn would have to be answered once per branch.
            outputs.append(await self.model.agenerate(messages, n=1))

        # Returned bare so the runner sums token usage across all turns into the
        # stage meta -- it special-cases `list[ModelOutput]` for exactly this.
        return outputs

    @override
    async def postprocess(self, inf, ctx):
        # One rollout per conversation, not one per turn: turns are sequential
        # parts of a single answer, while a rollout is one of `n` independent
        # samples of the same prompt. Per-turn responses ride inside it.
        texts = [output.texts[0] if output.texts else "" for output in inf]
        # Annotated because `list` is invariant: an inferred
        # `list[dict[str, int | str]]` is not a `list[JSONValue]`.
        responses: list[JSONValue] = [
            {"turn": index, "response": text}
            for index, text in enumerate(texts, start=1)
        ]
        # `None` only when no turn produced anything, so `extracted` stays a real
        # signal -- a partly-blank conversation is a real answer that scores
        # badly. This is also what keeps the empty-postprocess anomaly rule
        # meaningful here, since the rules that read `infer` unwrap a single
        # ModelOutput and skip a list.
        any_text = any(text.strip() for text in texts)
        return build_prediction_record([responses if any_text else None])

    @override
    async def feedback(self, post, ctx):
        # Graded here rather than in report() so every turn's verdict is on disk
        # and inspectable. Both graders are pure per-response.
        from sieval.community.multi_if.evaluation_lib import (
            gen_acc_loose,
            gen_acc_strict,
        )

        graders = {"strict": gen_acc_strict, "loose": gen_acc_loose}
        raw = ctx.raw_sample
        turns = raw["turns"]
        responses = post["rollouts"][0].get("prediction") or []
        by_turn = {r["turn"]: r["response"] for r in responses}

        metrics: dict[str, bool | float] = {}
        detail: dict[str, dict] = {}
        for index, turn in enumerate(turns, start=1):
            instruction_ids = list(turn["instruction_id_list"])
            payload = {
                "response": by_turn.get(index, ""),
                "instruction_id_list": instruction_ids,
                # Each element is JSON-encoded upstream and stays that way in the
                # dataset (see its module docstring); decoded here, immediately
                # before `build_description` consumes it.
                "kwargs": [json.loads(kwarg) for kwarg in turn["kwargs"]],
            }
            detail[f"turn_{index}"] = {"instruction_id_list": instruction_ids}
            for grade in _GRADES:
                followed = list(graders[grade](payload)["follow_instruction_list"])
                metrics[f"turn_{index}_{grade}_follow_all"] = all(followed)
                metrics[f"turn_{index}_{grade}_instruction_level"] = (
                    sum(followed) / len(followed) if followed else 0.0
                )
                # The raw per-constraint outcomes, which report() pools; a
                # per-sample rate cannot reconstruct a pooled one.
                detail[f"turn_{index}"][grade] = {"follow_instruction_list": followed}

        # Derived from `metrics`, not recomputed, so the headline cannot disagree
        # with the set. `correct` is the strictest reading the benchmark offers:
        # every constraint honoured in every turn.
        correct = all(
            bool(metrics[f"turn_{index}_strict_follow_all"])
            for index in range(1, len(turns) + 1)
        )
        score = sum(
            float(metrics[f"turn_{index}_strict_instruction_level"])
            for index in range(1, len(turns) + 1)
        ) / len(turns)
        return True, build_judgement_record(
            [list(turn["instruction_id_list"]) for turn in turns],
            [build_rollout_judgement(0, correct, score=score, metrics=metrics)],
            score=score,
            metrics=metrics,
            extra={
                "key": raw["key"],
                "language": raw["language"],
                "n_turns": len(turns),
                **detail,
            },
        )

    @override
    async def report(self, finals, fails):
        judgements = [f.feedback_result for f in finals]
        results: dict[str, float] = {"fails": len(fails)}

        # Every language present, plus the pooled cell. Sorted so the report's
        # key order does not depend on which sample finished first.
        languages = sorted({str(j["extra"]["language"]) for j in judgements})
        turn_overalls: list[float] = []

        for turn in (1, 2, 3):
            # Only conversations that *have* this turn count toward it -- 56 rows
            # have no third turn, and upstream skips them rather than scoring
            # them zero.
            present = [j for j in judgements if j["extra"]["n_turns"] >= turn]
            if not present:
                continue
            results[f"turn_{turn}_prompts_number"] = len(present)

            for language in (_ALL_LANGUAGES, *languages):
                cell = (
                    present
                    if language == _ALL_LANGUAGES
                    else [j for j in present if j["extra"]["language"] == language]
                )
                if not cell:
                    continue
                components = []
                for grade in _GRADES:
                    followed = [
                        j["extra"][f"turn_{turn}"][grade]["follow_instruction_list"]
                        for j in cell
                    ]
                    prompt_level = sum(1 for f in followed if all(f)) / len(followed)
                    # Pooled from raw counts, not averaged from the per-sample
                    # rates -- the two differ when turns carry different
                    # constraint counts, and here they always do.
                    instruction_total = sum(len(f) for f in followed)
                    instruction_level = (
                        sum(sum(f) for f in followed) / instruction_total
                        if instruction_total
                        else 0.0
                    )
                    components += [prompt_level, instruction_level]
                    if language == _ALL_LANGUAGES:
                        # The four components are only broken out for the pooled
                        # cell; per language upstream publishes just `overall`.
                        results[f"turn_{turn}_{grade}_prompt_level_accuracy"] = (
                            prompt_level * 100
                        )
                        results[f"turn_{turn}_{grade}_instruction_level_accuracy"] = (
                            instruction_level * 100
                        )

                # Upstream's per-turn headline: the plain mean of strict and
                # loose x prompt-level and instruction-level.
                overall = sum(components) / len(components) * 100
                results[f"turn_{turn}_{language}_overall"] = overall
                if language == _ALL_LANGUAGES:
                    turn_overalls.append(overall)

        # Upstream reports each turn separately; a task needs one headline, and
        # the benchmark's subject is how following degrades across turns, so the
        # mean over turns is the summary that does not privilege one of them.
        results["score"] = (
            sum(turn_overalls) / len(turn_overalls) if turn_overalls else 0.0
        )
        return results
