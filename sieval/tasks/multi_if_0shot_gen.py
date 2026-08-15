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
- A turn whose request fails outright ends the walk instead of failing the whole
  conversation. The turns already answered keep their verdicts; the unreached
  ones are *absent*, not zero, so an infrastructure failure cannot read as a
  model that stopped following instructions. Upstream keeps the earlier turns
  too, for free, by running one pass per turn.
  ``turn_{t}_prompts_number`` therefore counts the conversations that actually
  reached turn *t* -- identical to upstream's count on a clean run, and honest
  about a degraded one.
- ``n`` is pinned to 1 per turn, so a configured ``n > 1`` is ignored rather
  than honoured: a second sample would fork the conversation, and every later
  turn would then have to be answered once per branch.

Two upstream defects are tracked rather than repaired, because the unqualified
name must measure what upstream measures. Both make the score nondeterministic:

- Two conversations (6 of 13,447 turn-cells, 0.04%) carry kwargs the checker
  rejects -- ``keywords:letter_frequency`` with ``letter="#"``, and
  ``keywords:frequency`` with an empty ``keyword``. On a rejected value
  ``build_description`` falls back to ``random.choice`` over the alphabet and
  grades the response against a letter nobody asked for, freshly drawn per call.
- ``langdetect`` is unseeded, so detection -- which selects the word- and
  sentence-counting algorithm behind every length constraint -- can vary between
  runs on short or mixed-script text.

Verified against upstream's own ``metrics_gen`` on 535 conversations spanning
all eight languages and all 56 two-turn rows: 3,098 strict/loose follow-lists
and every per-language ``overall`` agree exactly, once those two conversations
are set aside. They cannot agree with anything, upstream included.

Verified again on two full live runs, by re-grading their own responses with
upstream's graders: 128,258 per-constraint comparisons, 15 disagreements
(0.012%), every one at a langdetect-routed checker or at the rejected-kwargs
row -- none at a deterministic one. Re-grading identical responses three times
moves ``score`` by 0.012, against the +-0.4 that conversation sampling
contributes, which is the quantitative case for tracking those two defects
rather than repairing them.

What remains open is a *published* number, and the obstacle is the anchor. The
only servable model with a first-party Multi-IF figure is Qwen3-32B (Qwen3
Technical Report, arXiv:2505.09388: 73.0 thinking / 70.7 non-thinking). Full
runs at that report's own sampling knobs land at 78.97 and 78.73, and no single
reduction closes both figures -- turn-3-only comes nearest yet inverts the
published ordering, while the report's own Table 11 puts OpenAI-o1 at 48.8
where the Multi-IF paper puts o1-preview at 78.9. ``reference_impl.notes``
carries the numbers; ``status`` stays ``experimental`` until an anchor with a
stated protocol exists.

Infra: grading needs NLTK's ``punkt_tab`` data (see ``_ensure_punkt_tab``),
which this task downloads once if it is absent. Pre-stage it for offline runs.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import json
from functools import cache
from typing import override

from loguru import logger
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
from sieval.core.tasks.metrics import (
    DENOMINATOR_FIELD,
    DENOMINATOR_JUDGED,
    SCORE_KEY_FIELD,
)
from sieval.core.types import JSONValue
from sieval.datasets import MultiIFDatasetSample

# The two IFEval readings, both published. Neither is subordinate: upstream's
# per-turn headline averages them together rather than picking one.
_GRADES = ("strict", "loose")

# The pooled cell every published Multi-IF number is quoted against; kept
# distinct from the CSV's own `language` values, which are English names.
_ALL_LANGUAGES = "all_languages"

# The headline cell: pooled over languages *and* over turns. Named to match the
# per-turn cells it averages (`turn_{t}_all_languages_overall`), so the family
# reads as one series rather than as a total bolted onto its parts.
_ALL_TURNS_KEY = "all_turns_all_languages_overall"


@cache
def _download_punkt_tab_once() -> None:
    import nltk

    nltk.download("punkt_tab", quiet=True)


def _ensure_punkt_tab() -> None:
    """Make NLTK's ``punkt_tab`` data available before any grading happens.

    Two checkers reach NLTK, and between them they cover 926 of the 4,501
    conversations: ``length_constraints:number_sentences`` loads the sentence
    tokenizer, and ``change_case:capital_word_frequency`` calls
    ``nltk.word_tokenize``. Both resolve through **punkt_tab** on nltk >= 3.9 --
    including the ``nltk:tokenizers/punkt/english.pickle`` path, whose name still
    says ``punkt``. Staging the legacy ``punkt`` package alone satisfies neither.

    Without it a fifth of the set dies one ``LookupError`` at a time, deep inside
    the grader, and because that fails the whole conversation it also shortens
    every *earlier* turn's denominator -- a wrong score rather than a loud stop.

    Lives here, not in ``sieval/community/multi_if/``, for two reasons: the
    vendored checkers stay byte-identical to upstream, and a helper with a single
    caller belongs in its caller's module. The IFBench sibling splits the job: it
    downloads from inside its vendored module (upstream ships that helper; ours
    does not) and verifies from its task layer, because that helper swallows its
    own failure.
    """
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
        return
    except LookupError:
        pass
    # At most one download attempt per process: an offline run must not spend a
    # network timeout per sample discovering the same absence 4,501 times.
    _download_punkt_tab_once()
    # Re-check instead of trusting the download's return value, so a run that
    # cannot get the resource stops here naming it, rather than surfacing later
    # as an opaque per-sample failure.
    nltk.data.find("tokenizers/punkt_tab")


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
    # The grader and aggregation are verified exactly against upstream's own
    # evaluator, now on two full live runs as well as offline (see the module
    # docstring). What is still unreproduced is a *published number*, and the
    # reason is no longer "nobody has tried": the one first-party anchor that
    # exists for a servable model cannot be reproduced under any single
    # reduction, and is internally inconsistent with Multi-IF's own paper. That
    # is a property of the anchor, not of this port -- but `stable` claims a
    # reproduction, so it stays out of reach until an anchor with a stated
    # protocol (or a replayable inference dump, as MathArena and PlatinumBench
    # have) turns up. See `reference_impl.notes` for the measurements.
    status="experimental",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="facebookresearch/Multi-IF",
        url="https://github.com/facebookresearch/Multi-IF/blob/1cdb53ed18499ad729e0766e5d3099dd5344406f/metrics.py",
        notes=(
            "Multi-IF's own multilingual fork of the IFEval checkers is vendored "
            "(sieval.community.multi_if); the google-research IFEval sibling is "
            "NOT interchangeable with it. Upstream drives one pass per turn "
            "(--steps 1 2 3), one sample per turn; this task walks all three in "
            "one pass. Upstream reports fractions, this task percentages. "
            "Grading matches upstream's metrics_gen exactly, checked twice: "
            "offline on 535 conversations across all 8 languages (3,098 "
            "follow-lists), and on two full live runs by re-grading their own "
            "responses with upstream's graders — 128,258 per-constraint "
            "comparisons, 15 disagreements (0.012%), every one of them at a "
            "langdetect-routed checker (change_case:english_capital x10, "
            "length_constraints:number_sentences/number_words x1 each) or at the "
            "rejected-kwargs row (keywords:letter_frequency x3). Zero "
            "disagreements at any deterministic checker; re-derived per-language "
            "cells agree to 6e-02, `score` to 0.003. "
            "The two defects upstream cannot grade reproducibly itself: kwargs it "
            "rejects (letter='#' in 1122:18:en; empty keyword in 2616:4:zh — 6 of "
            "13,447 turn-cells) send build_description to an unseeded "
            "random.choice, and langdetect is likewise unseeded and picks the "
            "counting algorithm behind every length constraint. Measured cost of "
            "both, re-grading identical responses 3x on the full set: 12 of "
            "26,894 turn-cells flip and `score` spans 0.012 — two orders of "
            "magnitude under the +-0.4 sd that conversation sampling contributes. "
            "Tracked, not repaired, per the unqualified-name rule; fixing either "
            "needs a `_fixed` variant with a measured delta. That variant now "
            "exists as multi_if_0shot_gen_fixed, and it addresses the first only "
            "in part: it keeps the letter the 1122:18:en row names, and repairs "
            "two further checkers, but the empty-keyword row 2616:4:zh stays as "
            "upstream grades it and langdetect still routes the counting "
            "algorithm behind every length constraint in both tasks. "
            "PUBLISHED-NUMBER RESIDUAL (open). The only servable model carrying a "
            "first-party Multi-IF figure is Qwen3-32B: Qwen3 Technical Report "
            "(arXiv:2505.09388) Table 13 Thinking 73.0, Table 14 Non-thinking "
            "70.7. Full 4,501-conversation runs at that report's own sampling "
            "knobs, 0 failures, denominators 4501/4501/4445, give `score` 78.73 "
            "(non-thinking) and 78.97 (thinking) — +8.03 and +5.97. No single "
            "reduction closes both: turn-3-only is nearest (71.38, +0.68; 71.23, "
            "-1.77) and is mechanically plausible because upstream emits one "
            "report per turn, so an integrator running --steps 3 would publish "
            "exactly that cell — but it contradicts the published ordering, since "
            "the two arms here are statistically indistinguishable (0.24 apart, "
            "bootstrap sd 0.4) where the report has Thinking +2.3. Upstream's own "
            "driver default max_new_tokens=1024 accounts for part of the level: a "
            "paired 600-conversation arm at that cap truncates 12.23% of "
            "turn-cells and loses 2.66 points, about a third of the gap. That the "
            "anchor is not the benchmark's protocol is visible in the report "
            "itself — its Table 11 gives OpenAI-o1 48.8 where the Multi-IF paper "
            "gives o1-preview 78.9 (three-turn average) and 70.7 (turn 3)."
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
        # `float | str`: the report carries `score_key` and `denominator_policy`,
        # which name a column and a population rather than measuring one.
        dict[str, float | str],
    ]
):
    def _instruction_dict(self) -> dict[str, type] | None:
        """Registry the checkers are looked up in; ``None`` is the vendored one.

        The single seam ``multi_if_0shot_gen_fixed`` needs. Everything else about
        how a conversation is walked, graded and pooled is shared, so overriding
        this cannot make the two tasks differ in any other way.
        """
        return None

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

        for index, turn in enumerate(turns, start=1):
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
            try:
                # n=1 per turn: a second sample would fork the conversation, and
                # every later turn would have to be answered once per branch.
                outputs.append(await self.model.agenerate(messages, n=1))
            except Exception as exc:
                if not outputs:
                    # Nothing was answered, so there is nothing to salvage: fail
                    # the sample exactly as a single-turn task's would.
                    raise
                # A request that never came back ends the walk. Failing the whole
                # conversation instead would delete the turns that *did* answer
                # from their own denominators as well, moving turn 1 and turn 2's
                # published numbers because turn 3's request timed out. Broad on
                # purpose -- any failure that reaches here has already exhausted
                # `max_retries` -- and never silent: the warning names the turn.
                logger.warning(
                    "Multi-IF conversation {} failed at turn {} of {}; keeping "
                    "the {} turn(s) already answered and ending the walk: {}",
                    ctx.raw_sample["key"],
                    index,
                    len(turns),
                    len(outputs),
                    exc,
                )
                break

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
        # signal -- a partly-blank conversation is a real answer that scores badly,
        # and that is what `detect_extraction_failure` reads.
        any_text = any(text.strip() for text in texts)
        return build_prediction_record(
            [responses if any_text else None],
            extra={
                # How many turns were answered -- which `prediction` cannot carry,
                # because it collapses to `None` when every turn came back blank.
                # A walk cut short by a failed request and a conversation that
                # answered every turn with "" are different facts, and only this
                # tells them apart once the record is on disk.
                "n_answered": len(texts),
                # Per-turn finish reasons. `detect_truncated_output` now reads a
                # list, so it does flag the rollout -- but it reports *that* the
                # conversation truncated, not *where*, since its indices are rollout
                # positions. A three-turn sample dragging its own history along is
                # the likeliest shape here to hit `max_tokens`, so which turn did it
                # is worth having on disk next to the verdict.
                # `or []`: the field is optional on ModelOutput, so a backend that
                # does not report one leaves it None rather than empty.
                "finish_reasons": [list(output.finish_reasons or []) for output in inf],
            },
        )

    @override
    async def feedback(self, post, ctx):
        # Graded here rather than in report() so every turn's verdict is on disk
        # and inspectable. Both graders are pure per-response.
        from sieval.community.multi_if.evaluation_lib import (
            gen_acc_loose,
            gen_acc_strict,
        )

        _ensure_punkt_tab()

        graders = {"strict": gen_acc_strict, "loose": gen_acc_loose}
        raw = ctx.raw_sample
        turns = raw["turns"]
        responses = post["rollouts"][0].get("prediction") or []
        by_turn = {r["turn"]: r["response"] for r in responses}
        # Grade only the turns that were reached. On a clean run that is all of
        # them; a walk ended early by a failed request leaves the rest ungraded
        # rather than scoring them zero, so the missing answer does not read as a
        # model that stopped following instructions.
        n_answered = int(post.get("extra", {}).get("n_answered", len(turns)))
        graded = turns[:n_answered]

        metrics: dict[str, bool | float] = {}
        detail: dict[str, dict] = {}
        instruction_dict = self._instruction_dict()
        for index, turn in enumerate(graded, start=1):
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
                graded_turn = graders[grade](payload, instruction_dict=instruction_dict)
                followed = list(graded_turn["follow_instruction_list"])
                metrics[f"turn_{index}_{grade}_follow_all"] = all(followed)
                metrics[f"turn_{index}_{grade}_instruction_level"] = (
                    sum(followed) / len(followed) if followed else 0.0
                )
                # The raw per-constraint outcomes, which report() pools; a
                # per-sample rate cannot reconstruct a pooled one.
                detail[f"turn_{index}"][grade] = {"follow_instruction_list": followed}

        # Derived from `metrics`, not recomputed, so the headline cannot disagree
        # with the set. `correct` is the strictest reading the benchmark offers:
        # every constraint honoured in every turn -- which a conversation that
        # never reached its last turn cannot claim, however well the rest scored.
        correct = n_answered >= len(turns) and all(
            bool(metrics[f"turn_{index}_strict_follow_all"])
            for index in range(1, len(graded) + 1)
        )
        # Averaged over the turns that were graded, matching how `report` pools:
        # an unreached turn is absent from the denominator, not a zero in it.
        score = sum(
            float(metrics[f"turn_{index}_strict_instruction_level"])
            for index in range(1, len(graded) + 1)
        ) / len(graded)
        return True, build_judgement_record(
            # Ground truth is the whole sample's constraint set, so it lists every
            # turn the dataset ships -- including any the walk did not reach.
            [list(turn["instruction_id_list"]) for turn in turns],
            [build_rollout_judgement(0, correct, score=score, metrics=metrics)],
            score=score,
            metrics=metrics,
            extra={
                "key": raw["key"],
                "language": raw["language"],
                "n_turns": len(turns),
                # What `report` counts turns by. Equal to `n_turns` on a clean
                # run; lower only where the walk ended early.
                "n_answered": len(graded),
                **detail,
            },
        )

    @override
    async def report(self, finals, fails):
        judgements = [f.feedback_result for f in finals]
        results: dict[str, float | str] = {"fails": len(fails)}

        # Every language present, plus the pooled cell. Sorted so the report's
        # key order does not depend on which sample finished first.
        languages = sorted({str(j["extra"]["language"]) for j in judgements})
        turn_overalls: list[float] = []
        # Read off the run instead of hardcoded, so the turn count lives only
        # where the dataset reshapes the CSV into a `turns` list.
        max_turns = max((int(j["extra"]["n_answered"]) for j in judgements), default=0)

        for turn in range(1, max_turns + 1):
            # Only conversations that answered this turn count toward it -- 56
            # rows ship no third turn and upstream skips them rather than scoring
            # them zero, and a walk ended early by a failed request stops here for
            # the same reason.
            present = [j for j in judgements if int(j["extra"]["n_answered"]) >= turn]
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
        #
        # Named rather than only assigned to `score`: every other turn-and-language
        # cell this report writes has a name, and the pooled-over-turns one is the
        # single cell a reader is most likely to want to cite or re-derive. It also
        # lets `score_key` name a column that exists -- a headline computed inline
        # can only be pointed at by inventing a key or by crowning one turn, and
        # crowning turn 3 would bake in a guess about what published tables quote
        # (see `reference_impl.notes`) rather than report upstream's arithmetic.
        results[_ALL_TURNS_KEY] = (
            sum(turn_overalls) / len(turn_overalls) if turn_overalls else 0.0
        )
        results["score"] = results[_ALL_TURNS_KEY]
        results[SCORE_KEY_FIELD] = _ALL_TURNS_KEY
        # `judged`: `report` pools over the judgements it was handed, and a sample
        # that never produced one is counted in `fails` instead of entering a
        # denominator as a followed-nothing conversation. The same reading applies
        # within a sample -- a turn the walk never reached is absent from that
        # turn's denominator, which is why `turn_{t}_prompts_number` is reported.
        results[DENOMINATOR_FIELD] = DENOMINATOR_JUDGED
        return results
