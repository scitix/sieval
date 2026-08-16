"""SysBench — 0-shot generative, multi-turn system-message following, LLM-judged.

SysBench (arXiv:2408.10943) asks whether a model keeps obeying its **system prompt**
across a 5-turn conversation. Each turn carries a checklist of atomic constraints drawn
from the system prompt, and a grader LLM answers 是/否 per constraint. The three
published rates nest: **CSR** (constraints met, pooled over every constraint in the
run — *not* the mean of the per-turn rates, which turns of unequal constraint count
make a different number), **ISR** (turns meeting every one), **SSR** (the mean
normalised count of consecutive turns, from the session's start, that met every one —
*not* the fraction of sessions whose turns all do). Both readings that the paper's
prose leaves open are settled in
:func:`~sieval.community.sysbench.aggregate_metrics` against the scripts that
generated the published tables. CSR is the headline; ISR and SSR fall off faster,
being conjunctions over ~2.4 constraints and over a growing prefix.

**One sample is one whole session.** ``infer`` walks the five turns in order, appending
the model's own reply before sending the next user turn — upstream's
``eval_system_bench.py``, whose ``do_infer`` grows one ``messages`` list across the
conversation. So a model that derails on turn 2 answers turn 3 from its own derailed
history, and SSR is a property of one rollout rather than a reassembly over independent
turns. That is the protocol behind every published SysBench number.

``history="ground_truth"`` selects upstream's *other* script,
``eval_system_bench_with_gt.py``, which replaces the history with the dataset's
reference answers, so no turn is answered from the model's own earlier mistakes —
the history is still supplied, it is just not the model's. That is the paper's §4.5
investigative experiment (Figure 5, "we replace the historical model response with the
ground truth"), not its headline — it measures per-turn adherence with conversational
recovery factored out. The report names the mode it ran in, because the two numbers
differ systematically and nothing else in a result directory would tell them apart.

The grader is supplied via the ``grader`` task arg (a model-config dict, or a pre-built
Model, on its own ``api_base``/``api_key``). It is called **once per turn**, on the same
cumulative message list the model saw, and every one of those ``ModelOutput`` objects is
persisted under ``extra.grader_output`` — see :meth:`feedback`.

**The grader is the measurement, and it is not pinnable the way a Hub revision is.**
Upstream graded with GPT-4o and reports no decoding settings. Every deterministic lane
can be re-derived from a persisted response plus a reference; this one cannot, so the
reply is the only durable evidence of a verdict. Pin the grader model, set
``temperature: 0`` where the endpoint honours it, and name the grader beside every
number this task produces. A single run is indicative; two runs with different graders
are not comparable.

**The prompt is Chinese and so is the benchmark.** The grader has to read Chinese
constraint text and a Chinese response.

Deviations / by-design behavior worth knowing:

* ``n`` is pinned to 1 and the task takes no ``n`` argument: a second sample would fork
  the conversation, and every later turn would have to be answered once per branch.
* A turn whose request fails outright ends the walk instead of failing the session. The
  turns already answered keep their verdicts; for ``csr`` and ``isr`` the unreached ones
  are then **absent** from the denominator, not zero, so an infrastructure failure
  cannot read as a model that stopped following its system prompt.
  ``turn_{t}_n_turns`` counts the sessions that actually reached turn *t*.
  ``ssr`` and ``turn_{t}_isr_cumulative`` are the exception, and have to be: they
  measure a *prefix*, so an unreached turn keeps the session's declared length as the
  denominator and simply shortens the run. Dropping it there would let a failure at
  turn 3 of 5 report a perfect session.
* A constraint with no readable verdict scores **not satisfied** but is counted in
  ``n_grader_unparsed``; a turn where *nothing* parsed is counted again in
  ``n_grader_unparsed_turns``. Upstream instead retries the grader up to 10 times
  and lets the session fail, so where upstream loses 5 turns this loses 1
  constraint — and says so. ``csr`` counts an unresolved constraint as unsatisfied,
  matching upstream's denominator, which makes it a **floor**; ``csr_graded`` is the
  same pooled rate over the turns actually graded and ``ungradeable_rate`` sizes the
  gap. Read all three or a grader outage reads as a weaker model.
* An empty response still goes to the grader: a SysBench constraint is frequently a
  prohibition ("不要提及…"), so short-circuiting an empty reply to zero would be a
  *different* claim from upstream's, which grades what it is given.
* **Upstream's one rule-based check is not ported**: it is unreachable dead code whose
  call site also swaps its arguments, so upstream grades every constraint by judge. The
  measurement and the reasoning are in :mod:`sieval.community.sysbench`; reviving the
  rule would move CSR by at most ~2.8 points and is a scoring change, not a repair.

The judge prompt is byte-identical to upstream's ``get_eval_pattern`` — verified by
running both over all 2,500 turns of the released set, 0 differences.

Comparison targets — paper Table 2, CSR / ISR / SSR, GPT-4o grader, 500 sessions /
2,500 turns / 5,952 constraints: GPT-4o 87.1 / 76.4 / 54.4; GPT-4-Turbo-20240409 86.5 /
76.6 / 53.2; Claude-3-Opus 85.0 / 74.1 / 51.8; Qwen2.5-72B-Instruct 80.4 / 66.2 / 42.8;
Qwen2-72B-Instruct 79.0 / 64.1 / 41.6; GLM-4-9B-Chat 64.2 / 44.0 / 25.9; GPT-3.5-Turbo
61.6 / 43.2 / 20.8. The best published SSR is 54.4. Align on CSR first — ISR and SSR
amplify both real differences and grader noise.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

import json
from collections import defaultdict
from collections.abc import Mapping
from typing import override

from loguru import logger
from openai.types.chat import ChatCompletionMessageParam

from sieval.community.sysbench import (
    aggregate_metrics,
    aggregate_turn,
    build_judge_prompt,
    parse_verdict,
    session_prefixes,
)
from sieval.core.models import ChatModel, Model, ModelOutput
from sieval.core.tasks import (
    GRADER_OUTPUT_KEY,
    EvalMode,
    InputKind,
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    ReferenceImpl,
    RequirementContext,
    Task,
    TaskModelRequirement,
    TaskRequirements,
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
)
from sieval.core.types import JSONValue
from sieval.core.utils.serialization import obj_to_dict
from sieval.datasets import SysBenchDatasetSample

#: The two protocols upstream ships, one script each. ``model`` is the headline.
_HISTORY_MODES = ("model", "ground_truth")

#: Upstream's six constraint categories, for the per-type breakdown the paper reports.
#: An unlisted type keys under its own Chinese name rather than being dropped, so a
#: seventh category added upstream shows up in the report instead of vanishing from it.
_CONSTRAINT_TYPES = {
    "动作约束": "action",
    "内容约束": "content",
    "背景约束": "background",
    "角色约束": "role",
    "格式约束": "format",
    "风格约束": "style",
}


@sieval_task(
    name="sysbench_0shot_gen",
    display_name="SysBench (0-shot, generative)",
    description=(
        "Multi-turn Chinese system-message following; per-turn constraint "
        "checklists graded by an LLM judge."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("chinese", "multi-turn", "instruction-following", "open-ended"),
    model_type="chat",
    # The judge prompt is verified byte-identical to upstream's over all 2,500 turns,
    # and the data path against the released file. What is unverified is a published
    # NUMBER: every Table 2 entry was graded by GPT-4o at unstated decoding settings, so
    # reproducing one would be reproducing a grader, not a protocol. `stable` claims a
    # reproduction, so this stays experimental until a run against a named grader is
    # anchored to something.
    status="experimental",
    # `value`, not `procedure`: the judgement records a reference -- the per-turn
    # criteria id lists -- and that is the operational criterion, not whether the gold
    # looks scalar. Same reading as IFEval, which records the instruction ids its
    # checkers run and is likewise a `value`.
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="PKU-Baichuan-MLSystemLab/SysBench",
        url=(
            "https://github.com/PKU-Baichuan-MLSystemLab/SysBench/blob/"
            "627ffa8010d00e270426975b33b1fb7a0a635602/eval_system_bench.py"
        ),
        notes=(
            "Port of SysBench (arXiv:2408.10943) at commit 627ffa80. PROTOCOL: the "
            "unqualified task is upstream's HEADLINE script eval_system_bench.py — "
            "one session is one sample, and do_infer appends the model's OWN reply "
            "to `messages` before asking the next turn, so turn t is answered from "
            "the model's own history and SSR is a property of one rollout. "
            "history='ground_truth' selects eval_system_bench_with_gt.py instead, "
            "which is the paper's SS4.5 / Figure 5 ablation (GT history, turns "
            "independent), NOT its headline; the mode is named in the report. n is "
            "pinned to 1 — a second sample would fork the conversation. METRICS: "
            "headline CSR, with ISR (turns satisfying ALL) and SSR alongside. "
            "CSR IS POOLED OVER CONSTRAINTS, NOT AVERAGED OVER TURNS: turns carry "
            "1-11 constraints (mean 2.38), so sum(satisfied)/sum(constraints) and "
            "mean(satisfied/constraints) are different numbers. Paper SS3.3's 1/mn "
            "prefactor describes the ISR/SSR outer average over turns; the published "
            "CSR tables come from plot/tab6_csr_full.py, which reads 遵循数量 over "
            "约束总量 off the per-constraint-type sheet (wired to the reported CSR by "
            "plot/eval_output.py), and upstream's own shipped analysis workbooks "
            "carry that division as their 遵循率 row (GPT-4o: 5183/5952 = 87.08, "
            "published 87.1). MEASURED over the 14 shipped per-turn outputs: pooled "
            "reproduces published CSR at 1dp for all 7 models whose workbook matches "
            "its published run (87.080/86.542/85.013/78.999/78.881/64.247/61.626 vs "
            "87.1/86.5/85.0/79.0/78.9/64.2/61.6); the turn-averaged variant "
            "reproduces 0 of 14, sits up to 1.903 points off (mean 0.895), and "
            "inverts two adjacent published pairs (Qwen2-72B vs GLM-4, GLM-4-9B vs "
            "GPT-3.5). It is still reported, as csr_macro, because it is the finer "
            "signal when the turn is the unit of interest. ISR and SSR over the same "
            "7 reproduce at 1dp 5/7 and 6/7 (GLM-4's workbook disagrees with its own "
            "published row on both). SSR IS NOT A "
            "SESSION-LEVEL ALL-OR-NOTHING RATE: paper SS3.3 defines it as (1/mn) "
            "summed over EVERY prefix length alpha, i.e. the mean normalised count "
            "of consecutive fully-satisfied turns from the session's start, which "
            "upstream computes in utils.analysis_eval_results as "
            "cal_continuous_avg/5 (count_continuous_round holds one count per "
            "alpha). Reading it as 'sessions whose turns all do' reports only the "
            "alpha=n term, the smallest, and lands ~20 points under Table 2. An "
            "unreached turn shortens the prefix rather than leaving the "
            "denominator, so a walk that died at turn 3 cannot report a perfect "
            "session. Plus the paper's align/misalign split -- Table 3 reports ISR, "
            "so isr_align/isr_misalign are the comparable pair and csr_* is carried "
            "beside them -- its six constraint types, and per-turn-position cells in "
            "both readings: turn_{t}_isr is that position on its own, while "
            "turn_{t}_isr_cumulative is Table 4's R_t (sessions still unbroken "
            "through turn t) and averages to ssr by construction. "
            "GRADING: one grader call per turn "
            "on the same cumulative message list the model saw; the judge prompt is "
            "byte-identical to upstream's get_eval_pattern, verified over all 2,500 "
            "turns of the released set (0 differences). Upstream parses the reply "
            "with eval(reply[7:-3]) — Python eval, not json.loads, which is what "
            "makes the bare-integer-key dict it asks for parse at all — then asserts "
            "the verdict keys equal the criteria keys and retries up to 10x (temp 0 "
            "for five, then 0.5) before failing the session. This port reads the "
            "verdicts by regex (it does not execute grader text, and does not depend "
            "on the fence being byte-exact) and scores an unresolved constraint "
            "not-satisfied, counted in n_grader_unparsed: one constraint lost "
            "and visible, rather than five turns lost. UPSTREAM'S ONE RULE-BASED "
            "CHECK IS NOT PORTED: utils.character_count asserts its own regex "
            "matches are empty immediately after computing them, so its only "
            "reachable return is the -1 'not my rule' sentinel (run.sh uses plain "
            "python, no -O), and BOTH eval scripts call it as "
            "character_count(criteria_content, answer) against a def of "
            "character_count(answer, criteria_content) — arguments swapped, so the "
            "regexes scan the model's reply. Both defects point the same way: "
            "upstream grades EVERY constraint by judge, which is what its published "
            "numbers measure. Reviving the rule claims 13 of 459 constraints (2.83%) "
            "on a measured subset, bounding the CSR difference at 2.83 points — a "
            "scoring change to declare, not a repair to slip in. THE GRADER IS THE "
            "MEASUREMENT: upstream used GPT-4o and states no decoding settings; pin "
            "the grader, set temperature=0 where honoured, and name it beside every "
            "number. Every per-turn grader ModelOutput is persisted at "
            "extra.grader_output. csr keeps an ungradeable turn's constraints in the "
            "denominator with none of them satisfied (upstream's denominator) so it "
            "is a FLOOR; read it with csr_graded — the same pooled rate over the "
            "turns that were graded — and ungradeable_rate. UPSTREAM DECLARES NO "
            "LICENSE — see the dataset "
            "module. Paper Table 2 (GPT-4o grader; 500 sessions / 2,500 turns / "
            "5,952 constraints) CSR/ISR/SSR: GPT-4o 87.1/76.4/54.4, "
            "GPT-4-Turbo-20240409 86.5/76.6/53.2, Claude-3-Opus 85.0/74.1/51.8, "
            "Qwen2.5-72B-Instruct 80.4/66.2/42.8, Qwen2-72B-Instruct 79.0/64.1/41.6, "
            "GLM-4-9B-Chat 64.2/44.0/25.9, GPT-3.5-Turbo-20231106 61.6/43.2/20.8. "
            "Best published SSR is 54.4."
        ),
    ),
)
class SysBenchZeroShotGenTask(
    Task[
        SysBenchDatasetSample,
        PromptRecord,
        list[ModelOutput],
        PredictionRecord,
        JudgementRecord,
        # `float | str`: the report carries `score_key`, `denominator_policy` and
        # `history_mode`, which name things rather than measure them.
        dict[str, float | str],
    ]
):
    @classmethod
    @override
    def model_requirements_for(
        cls, context: RequirementContext
    ) -> tuple[TaskModelRequirement, ...]:
        candidate = super().model_requirements_for(context)
        grader = cls._bind_role_requirement(
            context,
            "grader",
            TaskRequirements(input=InputKind.CHAT),
        )
        return candidate + grader

    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        grader: Mapping | Model | None = None,
        history: str = "model",
        models_by_role: Mapping[str, Model] | None = None,
    ):
        # Checked first: `history` is a literal typed into a YAML config, and a
        # typo in it should say so rather than surface behind whatever model or
        # grader resolution happens to complain about first.
        if history not in _HISTORY_MODES:
            raise ValueError(
                f"history must be one of {_HISTORY_MODES}, got {history!r}. "
                "'model' is upstream's headline protocol (eval_system_bench.py); "
                "'ground_truth' is the paper's SS4.5 ablation "
                "(eval_system_bench_with_gt.py) and is not comparable to published "
                "SysBench numbers."
            )
        super().__init__(dataset=dataset, model=model, name=name)
        self._history = history
        self._grader = self._resolve_role_model(
            "grader",
            grader,
            models_by_role,
            build=self._build_grader,
        )

    @staticmethod
    def _build_grader(grader: Mapping | Model | None) -> Model:
        """Resolve the ``grader`` task arg into a Model.

        Accepts a pre-built Model (tests / advanced configs) or a model-config mapping
        (the YAML path). Grading is mandatory — SysBench constraints are natural
        language, so there is no deterministic fallback — and ``None`` raises.
        """
        if isinstance(grader, Model):
            return grader
        if isinstance(grader, Mapping):
            return ChatModel(**grader)
        raise ValueError(
            "SysBench requires an LLM grader. Pass `grader:` in the task args — a "
            "model-config dict such as {model: gpt-4o, api_base: ..., api_key: ..., "
            "temperature: 0}. The grader must read Chinese."
        )

    @override
    async def preprocess(self, raw, ctx):
        turns = json.loads(raw["turns_json"])
        return build_prompt_record(
            # Only the system prompt and the opening turn can be built ahead of
            # inference; the rest depend on what the model says. `infer` sends this
            # verbatim and grows the conversation from it, so the recorded prompt is
            # provably the one the model saw first.
            [
                {"role": "system", "content": raw["system_prompt"]},
                {"role": "user", "content": turns[0]["user"]},
            ],
            # The "ground truth" is the constraint checklist each turn must satisfy,
            # one list of criterion ids per turn.
            reference=[list(turn["criteria"]) for turn in turns],
            extra={
                "session_id": raw["session_id"],
                "domain": raw["domain"],
                "scenario": raw["scenario"],
                # Says how many prompts this sample really takes, so the single
                # recorded opening cannot be misread as the whole input.
                "n_turns": len(turns),
                "alignments": [turn["alignment"] for turn in turns],
                # The later turns' user text, recorded because it is part of the
                # sample's input and is otherwise only in the raw dataset.
                "later_turn_prompts": [turn["user"] for turn in turns[1:]],
            },
        )

    @override
    async def infer(self, pre, ctx):
        raw = ctx.raw_sample
        turns = json.loads(raw["turns_json"])
        # Seeded from the record rather than rebuilt, so what was persisted and what
        # was sent cannot drift.
        messages: list[ChatCompletionMessageParam] = list(pre["prompt"])
        outputs: list[ModelOutput] = []

        for index, turn in enumerate(turns, start=1):
            if outputs:
                previous = outputs[-1]
                # THE protocol switch. `model` carries the model's own reply forward
                # (upstream's headline); `ground_truth` substitutes the dataset's
                # reference answer (upstream's with-GT ablation), which is why a
                # derailed turn does not contaminate the next one in that mode.
                if self._history == "ground_truth":
                    content = turns[index - 2]["assistant"]
                else:
                    # An aborted or filtered response has no choices. Feed the empty
                    # assistant turn through anyway: the conversation continues (as
                    # upstream's does) and the missing answer scores zero on its own
                    # turn rather than failing the whole session.
                    content = previous.texts[0] if previous.texts else ""
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": turn["user"]})
            try:
                # n=1 per turn: a second sample would fork the conversation, and every
                # later turn would have to be answered once per branch.
                outputs.append(await self.model.agenerate(messages, n=1))
            except Exception as exc:
                if not outputs:
                    # Nothing was answered, so there is nothing to salvage: fail the
                    # sample exactly as a single-turn task's would.
                    raise
                # A request that never came back ends the walk. Failing the whole
                # session instead would delete the turns that *did* answer from their
                # own denominators too, moving turn 1 and 2's numbers because turn 3
                # timed out. Broad on purpose -- anything reaching here has already
                # exhausted `max_retries` -- and never silent: the warning names the
                # turn.
                logger.warning(
                    "SysBench session {} failed at turn {} of {}; keeping the {} "
                    "turn(s) already answered and ending the walk: {}",
                    raw["session_id"],
                    index,
                    len(turns),
                    len(outputs),
                    exc,
                )
                break

        # Returned bare so the runner sums token usage across all turns into the stage
        # meta -- it special-cases `list[ModelOutput]` for exactly this.
        return outputs

    @override
    async def postprocess(self, inf, ctx):
        # One rollout per session, not one per turn: the turns are sequential parts of
        # a single answer, while a rollout is one of `n` independent samples of the
        # same prompt. Per-turn responses ride inside it.
        #
        # Open-ended: the response IS the answer, so there is no extraction step.
        texts = [output.texts[0] if output.texts else "" for output in inf]
        # Annotated because `list` is invariant: an inferred
        # `list[dict[str, int | str]]` is not a `list[JSONValue]`.
        responses: list[JSONValue] = [
            {"turn": index, "response": text}
            for index, text in enumerate(texts, start=1)
        ]
        # `None` only when no turn produced anything, so `extracted` stays a real
        # signal -- a partly-blank session is a real answer that scores badly.
        any_text = any(text.strip() for text in texts)
        return build_prediction_record(
            [responses if any_text else None],
            extra={
                # How many turns were answered -- which `prediction` cannot carry,
                # because it collapses to `None` when every turn came back blank. A
                # walk cut short by a failed request and a session that answered every
                # turn with "" are different facts, and only this tells them apart once
                # the record is on disk.
                "n_answered": len(texts),
                # Per-turn finish reasons. A five-turn sample dragging its own history
                # along is the likeliest shape here to hit `max_tokens`, so which turn
                # did it is worth having next to the verdict. `or []`: the field is
                # optional on ModelOutput.
                "finish_reasons": [list(o.finish_reasons or []) for o in inf],
            },
        )

    @override
    async def feedback(self, post, ctx):
        """Grade each answered turn with one grader call, on the history it saw.

        The grader is a model, so its outputs are persisted the way model outputs are:
        ``extra[GRADER_OUTPUT_KEY]`` is the **list** of its whole ``ModelOutput``
        objects, one per turn, flattened to plain dicts (``add_type=False``, keeping the
        judgement record uniformly plain-dict). Nothing is hand-picked, so the replies
        survive — the only durable evidence of a verdict set on a lane whose grader
        cannot be pinned, and the only way to separate grader format drift from a
        genuine failure. It is a list because this task grades five turns per rollout;
        ``iter_grader_outputs`` reads either shape, so grader spend still reaches
        ``profile.json``.
        """
        raw = ctx.raw_sample
        turns = json.loads(raw["turns_json"])
        responses = post["rollouts"][0].get("prediction") or []
        by_turn = {r["turn"]: r["response"] for r in responses}
        # Grade only the turns that were reached. On a clean run that is all of them; a
        # walk ended early leaves the rest ungraded rather than scoring them zero.
        n_answered = int(post.get("extra", {}).get("n_answered", len(turns)))
        graded = turns[:n_answered]

        # Rebuilt exactly as `infer` built it, so the judge sees the history the model
        # saw -- upstream grades from its own `infer_results` for the same reason.
        messages: list[dict] = [{"role": "system", "content": raw["system_prompt"]}]
        grader_outputs: list[dict] = []
        detail: dict[str, dict] = {}
        metrics: dict[str, bool | float] = {}
        per_turn: list[tuple[float, bool]] = []

        for index, turn in enumerate(graded, start=1):
            answer = by_turn.get(index, "")
            criteria = turn["criteria"]
            criteria_ids = list(criteria)
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": answer})

            out = await self._grader.agenerate(
                [
                    {
                        "role": "user",
                        "content": build_judge_prompt(messages, criteria),
                    }
                ]
            )
            grader_outputs.append(obj_to_dict(out, add_type=False))
            reply = out.texts[0] if out.texts else ""
            verdicts = parse_verdict(reply, criteria_ids)
            csr, all_satisfied, n_satisfied, n_grader_unparsed = aggregate_turn(
                verdicts
            )

            # A turn upstream left unlabelled carries no criteria, so it has nothing to
            # satisfy: it keeps its place in the walk (it shaped the history of the
            # turns after it) but stays out of every rate, which is what `report` does
            # with it too. Counting it would score a 0.0 nobody asked for.
            if criteria_ids:
                per_turn.append((csr, all_satisfied))
                metrics[f"turn_{index}_csr"] = csr
                metrics[f"turn_{index}_all_satisfied"] = all_satisfied
            detail[f"turn_{index}"] = {
                # Raw per-constraint outcomes, which `report` pools; a per-turn rate
                # cannot reconstruct a pooled one, and the type is what the paper's
                # constraint-category breakdown needs.
                "criterion_verdicts": verdicts,
                "criterion_types": {
                    cid: criteria[cid].get("criteria_type", "") for cid in criteria_ids
                },
                "alignment": turn["alignment"],
                "n_criteria": len(criteria_ids),
                "n_satisfied": n_satisfied,
                "n_grader_unparsed": n_grader_unparsed,
                # Constraints the grader resolved. 0 means the reply was unreadable,
                # which `csr` alone cannot distinguish from a turn that satisfied
                # nothing.
                "n_graded": len(criteria_ids) - n_grader_unparsed,
            }

            # Swap in the protocol's history for the NEXT turn's judge prompt: under
            # the with-GT ablation the model never saw its own previous answer, so the
            # judge must not either.
            if self._history == "ground_truth":
                messages[-1] = {"role": "assistant", "content": turn["assistant"]}

        # The session's own CSR and ISR, over the turns that were graded.
        score = sum(t[0] for t in per_turn) / len(per_turn) if per_turn else 0.0
        # The strictest reading the benchmark offers -- every constraint honoured in
        # every turn -- which a session that never reached its last turn cannot claim,
        # however well the rest scored. NOT the session's SSR contribution: SSR counts
        # the leading run of fully-satisfied turns, so a session that keeps four turns
        # and loses the fifth contributes 4/5 there and `False` here.
        # `bool(per_turn)`: `all(())` is True, so a session whose turns carried no
        # criteria at all would otherwise be crowned for satisfying nothing.
        correct = (
            bool(per_turn) and n_answered >= len(turns) and all(t[1] for t in per_turn)
        )
        return True, build_judgement_record(
            [list(turn["criteria"]) for turn in turns],
            [
                build_rollout_judgement(
                    0,
                    correct,
                    score=score,
                    metrics=metrics,
                    extra={GRADER_OUTPUT_KEY: grader_outputs},
                )
            ],
            score=score,
            metrics=metrics,
            extra={
                "session_id": raw["session_id"],
                "domain": raw["domain"],
                "n_turns": len(turns),
                # What `report` counts turns by. Equal to `n_turns` on a clean run;
                # lower only where the walk ended early.
                "n_answered": len(graded),
                "history_mode": self._history,
                **detail,
            },
        )

    @override
    async def report(self, finals, fails):
        # Everything here derives from the persisted judgement records, so a result dir
        # can be re-reported without the model or the grader.
        turns: list[tuple[int, int, int, int]] = []
        # [satisfied constraints, total constraints] over turns actually graded.
        graded: list[int] = [0, 0]
        n_graded_turns = 0
        # [satisfied, constraints, full turns, turns] per bucket. Raw counts, because
        # `csr_*` pools over constraints and `isr_*` averages over turns -- the two
        # denominators differ, so neither can be recovered from the other's rate.
        # The paper splits this axis by ISR (Table 3); CSR is carried alongside
        # because it is the finer of the two.
        by_alignment: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        by_position: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        # Each session's DECLARED turn count, which is `ssr`'s denominator -- an
        # unreached turn must shorten the prefix, not leave the denominator.
        session_n_turns: dict[int, int] = {}
        # Pooled from raw counts, not averaged from per-turn rates: turns carry
        # different constraint counts, so the two differ.
        type_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        n_grader_unparsed = 0
        unparsed_turns = 0
        history_modes: set[str] = set()

        for position, f in enumerate(finals):
            judgement = f.feedback_result or {}
            extra = judgement.get("extra") or {}
            history_modes.add(str(extra.get("history_mode", "")))
            # `feedback` always writes this. The fallback guards a hand-edited or
            # truncated record, and is per-record on purpose: `ssr` groups turns by
            # this id, so a shared default would silently fuse those records into one
            # session and report a rate over the wrong denominator.
            raw_id = extra.get("session_id")
            session_id = int(raw_id) if raw_id is not None else -position - 1
            # `n_turns` is what the session HAS; `n_answered` is what it reached.
            # They differ exactly when a walk ended early, which is the case `ssr`
            # must not reward -- so the denominator comes from the former.
            session_n_turns[session_id] = int(
                extra.get("n_turns") or extra.get("n_answered", 0)
            )
            for index in range(1, int(extra.get("n_answered", 0)) + 1):
                turn = extra.get(f"turn_{index}") or {}
                n_turn_criteria = int(turn.get("n_criteria", 0))
                if not n_turn_criteria:
                    continue
                verdicts = turn.get("criterion_verdicts") or {}
                types = turn.get("criterion_types") or {}
                n_satisfied = int(turn.get("n_satisfied", 0))
                turn_unparsed = int(turn.get("n_grader_unparsed", 0))
                full = n_satisfied == n_turn_criteria

                turns.append((session_id, index, n_satisfied, n_turn_criteria))
                for bucket in (
                    by_position[index],
                    by_alignment[str(turn.get("alignment", ""))],
                ):
                    bucket[0] += n_satisfied
                    bucket[1] += n_turn_criteria
                    bucket[2] += full
                    bucket[3] += 1
                n_grader_unparsed += turn_unparsed
                if turn_unparsed >= n_turn_criteria:
                    unparsed_turns += 1
                else:
                    graded[0] += n_satisfied
                    graded[1] += n_turn_criteria
                    n_graded_turns += 1
                for cid, verdict in verdicts.items():
                    # Coerced: the loader normalises `alignment` but stores
                    # `criteria` raw, so `criteria_type: null` would key this
                    # bucket under None and make the `sorted()` below raise --
                    # losing the report, not one constraint. Absent and null
                    # both mean untyped here.
                    raw_type = str(types.get(cid) or "")
                    bucket = type_totals[_CONSTRAINT_TYPES.get(raw_type, raw_type)]
                    bucket[0] += bool(verdict)
                    bucket[1] += 1

        m = aggregate_metrics(turns, session_n_turns)
        n_turns = int(m["n_turns"])
        metrics: dict[str, float | str] = {
            "score": m["csr"] * 100,
            # Pooled over constraints, which is what Table 2 reports -- NOT the mean
            # of the per-turn rates, which turns of unequal constraint count make a
            # different number (`csr_macro`, reported beside it).
            "csr": m["csr"] * 100,
            "csr_macro": m["csr_macro"] * 100,
            "isr": m["isr"] * 100,
            "ssr": m["ssr"] * 100,
            "n_turns": float(n_turns),
            "n_sessions": m["n_sessions"],
            "n_criteria_graded": m["n_criteria"],
            # Grader format drift, kept out of the rate it would be invisible inside:
            # these constraints scored not-satisfied.
            "n_grader_unparsed": float(n_grader_unparsed),
            "n_grader_unparsed_turns": float(unparsed_turns),
            # `csr` counts an ungradeable turn as 0.0 -- upstream's denominator, so it
            # stays the headline -- which makes it a floor. These two say how far.
            "csr_graded": (graded[0] / graded[1] * 100) if graded[1] else 0.0,
            "ungradeable_rate": (
                (n_turns - n_graded_turns) / n_turns * 100 if n_turns else 0.0
            ),
            "fails": float(len(fails)),
            SCORE_KEY_FIELD: "csr",
            # `judged`: a session that never produced a judgement is counted in `fails`
            # rather than entering the denominator as a followed-nothing session. The
            # same reading applies within a session -- an unreached turn is absent from
            # its position's denominator, which is why `turn_{t}_n_turns` is reported.
            DENOMINATOR_FIELD: DENOMINATOR_JUDGED,
            # Which of upstream's two scripts this run reproduces. Named because the
            # numbers differ systematically and nothing else in a result directory
            # would tell them apart; `mixed` can only mean a resumed run whose config
            # changed under it, which is worth seeing rather than averaging away.
            "history_mode": (
                history_modes.pop() if len(history_modes) == 1 else "mixed"
            ),
        }

        # The paper's alignment split: `misalign` turns ask for something the system
        # prompt forbids, so the gap between these two is the part of the score that is
        # actually about system-message priority.
        #
        # `isr_*` is the comparable pair -- Table 3 splits ISR, not CSR, which is also
        # what upstream computes (`analysis_eval_results` appends the all-satisfied
        # indicator `是否可用` to its align buckets). `csr_*` is kept beside it because
        # it is the finer signal, but it is NOT the published column.
        # Every rate below is reported with the count it was divided by. A breakdown
        # cell is not readable without one: `csr_misalign` over 4 constraints and over
        # 400 are the same number and a different claim, and the reader who wants to
        # pool the cells back to the headline -- which is upstream's own `tab6_csr_full`
        # pairing -- cannot weight them without the denominators. `turn_{t}_n_turns`
        # already did this for one of the four families here; the other three, and the
        # per-position CSR's own denominator, are the rest of that convention.
        for alignment, (satisfied, criteria, full_turns, cells) in sorted(
            by_alignment.items()
        ):
            if alignment and cells:
                metrics[f"csr_{alignment}"] = satisfied / criteria * 100
                metrics[f"isr_{alignment}"] = full_turns / cells * 100
                # Two denominators, not one: `csr_*` pools over constraints while
                # `isr_*` averages over turns, so neither is recoverable from the
                # other's rate -- the same reason `by_alignment` carries both counts.
                metrics[f"csr_{alignment}_n_criteria"] = float(criteria)
                metrics[f"isr_{alignment}_n_turns"] = float(cells)

        # Per-turn-position cells: SysBench's subject is how adherence decays across a
        # conversation, and the headline averages that decay away. Computed here rather
        # than through `aggregate_metrics`, which would also return an `ssr` over a
        # single position -- a prefix rate whose prefix is missing, i.e. meaningless.
        for position, (satisfied, criteria, full_turns, cells) in sorted(
            by_position.items()
        ):
            metrics[f"turn_{position}_csr"] = satisfied / criteria * 100
            # Turn *t* on its own, over the sessions that reached it. NOT Table 4's
            # `R_t`, which is cumulative -- see `turn_{t}_isr_cumulative` below. The
            # two diverge fast: upstream's shipped GPT-4o run reads 82.8/66.2/51.2/
            # 40.4/31.6 cumulative (mean 54.4, its published SSR) against a
            # per-position 82.8/77.2/74.0/74.8/73.0 that barely decays, so a run read
            # against the wrong one looks like a different model.
            metrics[f"turn_{position}_isr"] = full_turns / cells * 100
            metrics[f"turn_{position}_n_turns"] = float(cells)
            # `turn_{t}_csr`'s own denominator, which is NOT `n_turns`: turns carry
            # different constraint counts, so the position's constraint total is a
            # separate number and the one the CSR cell was divided by.
            metrics[f"turn_{position}_n_criteria"] = float(criteria)

        # Table 4's `R_t`: the fraction of sessions whose first *t* turns ALL satisfied
        # every constraint -- upstream's `plot/tab4_turn.py`, which credits position
        # `turn` only while `accmulated_turn == turn`. Denominator is the sessions
        # declaring at least *t* turns, so a session that never had turn *t* is absent
        # rather than counted against. `ssr` is the mean of this series when every
        # session declares the same length, which is the released set's shape.
        prefix_cells: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for prefix, n_declared in session_prefixes(turns, session_n_turns).values():
            for position in range(1, n_declared + 1):
                prefix_cells[position][0] += position <= prefix
                prefix_cells[position][1] += 1
        for position, (hits, sessions) in sorted(prefix_cells.items()):
            metrics[f"turn_{position}_isr_cumulative"] = hits / sessions * 100
            # A third denominator, different again from the two above: sessions that
            # DECLARE at least `position` turns, where `turn_{t}_n_turns` counts turns
            # that were reached and graded. A walk that died mid-session separates them,
            # which is the case this series exists to punish -- so reporting one as if
            # it were the other would hide exactly the sessions R_t is about.
            metrics[f"turn_{position}_isr_cumulative_n_sessions"] = float(sessions)

        # Per constraint category, pooled over constraints rather than turns.
        #
        # `total` cannot be 0 -- a bucket is created by the increment that fills it --
        # so the live half of the old `if name and total` guard was `name`, and what it
        # dropped was every constraint upstream left untyped. That drop was SILENT and
        # it breaks the invariant the type cells exist to support: the cells pooled to
        # less than the headline, with nothing in the report saying by how much or that
        # anything was missing. A category is still not invented for them -- upstream's
        # six are upstream's six -- so the residual is reported as a count instead.
        n_criteria_untyped = 0
        for name, (satisfied, total) in sorted(type_totals.items()):
            if not name:
                n_criteria_untyped += total
                continue
            metrics[f"csr_type_{name}"] = satisfied / total * 100
            metrics[f"csr_type_{name}_n_criteria"] = float(total)
        # Always emitted, including as 0.0: a reader who has to distinguish "none were
        # untyped" from "this build does not report it" is back to guessing, and a key
        # that appears only in the bad case is a key nobody has a baseline for.
        metrics["n_criteria_untyped"] = float(n_criteria_untyped)
        # The same silence on the other axis, and it takes BOTH counts for the
        # reason the alignment cells take two denominators: an unlabelled turn
        # (which lands under "", so the guard above skips it) leaves `csr_*` short
        # by constraints and `isr_*` short by turns, and neither count sizes the
        # other. `.get`, because indexing this defaultdict would create the bucket
        # as a side effect of asking whether it exists.
        unaligned = by_alignment.get("", (0, 0, 0, 0))
        metrics["n_criteria_unaligned"] = float(unaligned[1])
        metrics["n_turns_unaligned"] = float(unaligned[3])

        return metrics | health_metrics(finals)
