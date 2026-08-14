"""ComplexConstraints — 0-shot generative, LLM-judge graded against a rubric.

Generative port of ComplexConstraints (Mehta et al., 2026, arXiv:2606.09118): the
model answers one realistic multi-constraint instruction, and a separate **LLM
judge** grades the free-form response against that prompt's 10-40 atomic rubric
criteria, one PASS/FAIL verdict each. The headline metric is the **task pass
rate** — the fraction of prompts whose response satisfies *every* criterion —
which is what the paper's public 75-prompt leaderboard reports (its Table 1).
The paper's other metric, the mean per-criterion pass rate, is reported
alongside it in both the macro (published) and pooled-micro readings.

Upstream ships **no evaluation code and no judge prompt** — the paper defines the
metrics and nothing more — so the rubric prompt and verdict parsing are authored
by this port (see ``sieval.community.complex_constraints``), which is why the task
ships ``status="experimental"``.

**Which judge produced Table 1 is unknown, and measured it is a capability floor
rather than a choice between equals.** The paper names GPT-5-mini as the
per-criterion judge for *training* only — "Per-criterion satisfaction judgments
are produced by GPT-5-mini during ComplexConstraints training and CoreCraft
training" — and never says what graded the leaderboard, so the judge is an
unpinned degree of freedom in the comparison itself. Re-grading one identical set
of responses with four judges across two vendors, changing nothing else: three
capable ones (a frontier model and two Gemini tiers) landed within **51.6–53.3**
and agreed with *each other* on 98.8–99.0% of the 1,559 criteria — about how well
any one of them agrees with itself on a repeat — while a mini-tier judge sat ~27
points low, and one-way: it alone fails ~141 criteria against ~35 in the other
direction, concentrated on criteria 1.8× longer than average that enumerate an
exclusion list to cross-check. This template tells the judge to fail what it
cannot verify, so a judge below the floor produces one-way false FAILs by
construction. **Choose the grader by capability, not by name or price** — a
flash-tier judge matched a pro-tier one to 1.3 points.

**The consensus sits above the board**, so the floor does not close the gap:
51.6–53.3 against Table 1's **38.7** for the same candidate means this port's
authored rubric grades *more leniently* than whatever produced the leaderboard,
and that ~13-point residual is open — one more reason to ship ``experimental``.
What does reproduce is the **ordering**: three models spanning the board ranked
51.56 > 18.33 > 5.78 against Table 1's 38.7 > 26.7 > 4.9, adjacent gaps 32.3 and
11.7 points even at worst case. Prefer rank agreement over a single-point score
match when reporting alignment here.

The judge is supplied via the ``grader`` task arg (a model-config dict, or a
pre-built Model, on its own ``api_base``/``api_key``); each rollout's
per-criterion verdicts and the judge's whole ``ModelOutput``
(``extra.grader_output``: reply, reasoning, usage, finish reasons, model id) are
persisted — see :meth:`feedback`.

**A sampling judge makes a single run indicative only, and this task cannot
average that away**: grading is one judge call per rollout, and ``n`` repeats the
*candidate*, not the judge. So pin the grader model — sieval cannot pin its
version the way it pins a Hub revision — and set ``temperature: 0`` *where the
endpoint honours it*. Measured on a gateway that fixes ``temperature`` at 1 for
its whole ``gpt-5.x`` family: one identical judge on one identical set of 75
responses moved the headline over **20.0–29.3**, left **22 of 75 prompts flipping
between pass and fail** and **14.8% of criteria undecided** across four runs (a
stronger judge was far steadier — 99.5% per-criterion self-agreement against
91.9%). ``temperature: 0`` is necessary but not sufficient: measured at it, one
judge repeated *exactly* while another still spanned 2.7 points. Repeat the
grading and report a spread; a lone number is a draw, not a measurement.

Budget tokens on **both** sides. The grader needs room for one verdict line per
criterion — up to 40 — *after* whatever reasoning it emits first; truncated
mid-block it leaves the tail unparsed, which is counted (``n_grader_unparsed``) and
scored not-satisfied, biasing the score **down**, and nothing flags it since the
``truncated_output`` anomaly rule reads the *candidate*'s finish reasons, never
the grader's. The candidate has the mirror failure: a reasoning model can spend
the whole allowance thinking and return empty content, scoring zero criteria —
observed at ``max_tokens: 16000``, 15 of 75 prompts empty, about 20 points off
that arm. There ``truncated_output`` *does* fire and ``n_unextracted`` counts them.

Deviations / by-design behavior worth knowing:

* **One judge call per rollout**, grading all of that prompt's criteria as an
  indexed list rather than one call per criterion (upstream states no call
  structure): batching keeps a rollout's whole verdict set in the single
  ``ModelOutput`` the runner's grader-spend accounting expects, and the indexing
  makes misalignment detectable instead of silently shifting verdicts.
* A criterion with no readable verdict is scored **not satisfied** — an
  unreadable verdict must never inflate a score — but counted separately as
  ``n_grader_unparsed``, so judge format drift stays distinguishable from a model that
  failed the rubric.
* The template and parser are **not tuned to one judge family**: across four
  judges from two vendors, 13 gradings of 1,559 criteria each, every verdict
  parsed — ``n_grader_unparsed`` was 0 every time, with no change to the template.
* **The grader prompt is hardened in two port-authored ways**, both scoring-
  relevant: its format example reads ``<verdict>`` and the parser rejects a
  PASS/FAIL alternation as a verdict, so a judge restating its instructions lands
  in ``n_grader_unparsed`` rather than scoring a full rubric; and it declares the
  RESPONSE block material to be graded, never instruction.
* An empty/whitespace response satisfies **zero** criteria **without** invoking
  the judge; ``extra.grader_output`` is absent there because no call was made,
  and the prediction rollout's ``extracted: false`` identifies it independently.
* Pipeline failures (exhausted retries) count as task failures satisfying zero
  criteria, weighted by ``n``, so all three rates span the full requested set.

Reproduction decoding: ``n`` (repeats) is a **task arg** — set it in
``tasks.<name>.args.n``. The leaderboard states no repeat count, so the port
defaults to ``n=1``; ``infer`` forwards ``n`` call-time and call-time wins over
model config, so setting ``n`` on the *model* is silently overridden. Some
gateways fix ``n`` at 1 for whole model families (the same ones that fix
``temperature``). Comparison target is the leaderboard Table 1 snapshots
(https://surgehq.ai/benchmarks/complex-constraints) — snapshot 2026-06-03: Gemini
3.1 Pro 40.4, GPT-5.5 38.7, Claude Opus 4.8 34.9 task pass %. The live board moves
and the paper pins no version of it, so align against that snapshot: as of
2026-08-10 it reads 43.7 / 44.4 / 34.2 and has begun splitting rows by reasoning
effort (GPT-5.5 default 44.4, High 48.9, xHigh 49.5), which Table 1's rows do not
state — so record the effort a comparison run used.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from collections.abc import Mapping
from typing import override

from sieval.community.complex_constraints import (
    aggregate_metrics,
    build_grader_prompt,
    parse_verdicts,
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
    RolloutJudgement,
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
    DENOMINATOR_REQUESTED,
    SCORE_KEY_FIELD,
    health_metrics,
)
from sieval.core.utils.serialization import obj_to_dict
from sieval.datasets import ComplexConstraintsDatasetSample


@sieval_task(
    name="complex_constraints_0shot_gen",
    display_name="ComplexConstraints (0-shot, generative)",
    description=(
        "Multi-constraint instruction following; rubric graded by an LLM judge."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "instruction-following", "open-ended"),
    model_type="chat",
    status="experimental",
    reference_kind="procedure",
    reference_impl=ReferenceImpl(
        source="complex-constraints",
        url="https://arxiv.org/abs/2606.09118",
        notes=(
            "Generative port of ComplexConstraints (Surge AI, arXiv:2606.09118) "
            "— 75 multi-constraint prompts (CIF-001..CIF-075) with 10-40 atomic "
            "rubric criteria each (1,559 total), graded by rubric rather than "
            "exact match. NO UPSTREAM EVAL CODE AND NO UPSTREAM JUDGE PROMPT: "
            "the paper defines the metrics, but the judge template, decoding "
            "settings and call structure are unstated, so the rubric prompt and "
            "verdict parsing are AUTHORED BY THIS PORT "
            "(sieval.community.complex_constraints) — hence status=experimental. "
            "METRICS: headline = task pass rate (response satisfies EVERY "
            "criterion), the metric the paper's leaderboard reports (Table 1); "
            "also criterion_pass_rate_macro (the paper's 'mean per-criterion "
            "pass rate', Table 3) and _micro (pooled), which differ because "
            "criteria counts vary 10-40 per prompt. GRADING: one judge call per "
            "rollout, verdicts as an indexed PASS/FAIL list; an unreadable "
            "verdict scores not-satisfied but is counted as n_grader_unparsed; empty "
            "responses satisfy zero criteria without invoking the judge. Give "
            "the grader max_tokens for up to 40 verdict lines AFTER its "
            "reasoning — a truncated grader biases the score DOWN. ALIGNMENT "
            "(measured 2026-08-10, 25 gradings; details in the module "
            "docstring): the ORDERING reproduces exactly — three models spanning "
            "the board rank 51.56 > 18.33 > 5.78 against Table 1's 38.7 > 26.7 > "
            "4.9, adjacent gaps 32.3 and 11.7 points even at worst case — but "
            "magnitudes sit ~13 points HIGH, and that residual is NOT judge "
            "choice: three capable judges across two vendors land on 51.6-53.3 "
            "and agree with each other about as well as each agrees with itself, "
            "so all of them sit ABOVE the board and this port's authored rubric "
            "grades more leniently than whatever produced it. Open; prefer rank "
            "agreement over a single-point score match. THE JUDGE IS A "
            "CAPABILITY FLOOR, NOT A NAME OR A PRICE: a flash-tier judge matched "
            "a pro-tier one to 1.3 points while a mini-tier one sat ~27 low and "
            "ONE-WAY — this template says fail what you cannot verify, so a "
            "sub-floor judge produces one-way false FAILs by construction. "
            "REPRODUCIBILITY: pin the grader model and set temperature=0 where "
            "the endpoint honours it — necessary but NOT sufficient (at t=0 one "
            "judge repeated exactly, another still spanned 2.67); where it is "
            "not honoured (one gateway fixes temperature=1 across its gpt-5.x "
            "family) the same judge on the same responses spanned 20.0-29.3 over "
            "four runs, so repeat the grading and report a spread. `n` is a task "
            "arg (tasks.<name>.args.n), not a model arg, and repeats the "
            "CANDIDATE, not the judge. Per-rollout verdicts and the judge's full "
            "ModelOutput (extra.grader_output) are persisted. Leaderboard "
            "snapshot 2026-06-03 "
            "(https://surgehq.ai/benchmarks/complex-constraints; the live board "
            "moves): Gemini 3.1 Pro 40.4, GPT-5.5 38.7, Claude Opus 4.8 34.9 "
            "task pass %."
        ),
    ),
)
class ComplexConstraintsZeroShotGenTask(
    Task[
        ComplexConstraintsDatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        # `float | str`: the report carries `score_key`, which names a column
        # rather than measuring one.
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
        n: int = 1,
        models_by_role: Mapping[str, Model] | None = None,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._n = n
        self._grader = self._resolve_role_model(
            "grader",
            grader,
            models_by_role,
            build=self._build_grader,
        )

    @staticmethod
    def _build_grader(grader: Mapping | Model | None) -> Model:
        """Resolve the ``grader`` task arg into a Model.

        Accepts a pre-built Model (used by tests / advanced configs) or a
        model-config mapping (the YAML path, e.g.
        ``{model: gpt-5-mini, api_base: ..., temperature: 0}``). Grading is
        mandatory — the rubric is natural language, so there is no deterministic
        fallback — and ``None`` raises.
        """
        if isinstance(grader, Model):
            return grader
        if isinstance(grader, Mapping):
            return ChatModel(**grader)
        raise ValueError(
            "ComplexConstraints requires an LLM grader. Pass `grader:` in the "
            "task args — a model-config dict such as "
            "{model: gpt-5-mini, api_base: ..., api_key: ..., temperature: 0}."
        )

    @override
    async def preprocess(self, raw, ctx):
        # No `reference`: the ground truth is a *rubric* (a procedure), not a
        # value. It goes to `extra` instead, once per sample — the judgement's
        # per-criterion verdicts are index-aligned to this list.
        return build_prompt_record(
            [{"role": "user", "content": raw["prompt"]}],
            extra={
                "benchmark_id": raw["benchmark_id"],
                "criteria": list(raw["criteria"]),
                "use_case": raw["use_case"],
                "instruction_type": raw["instruction_type"],
                "prompt_style": raw["prompt_style"],
            },
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        # Open-ended task: the response *is* the answer, so no extraction step.
        # Normalizing a blank to None keeps `extracted` a real signal AND is
        # exactly the empty-response condition feedback() short-circuits on --
        # one notion of "no answer", spelled once.
        return build_prediction_record(
            [text if text.strip() else None for text in inf.texts]
        )

    @override
    async def feedback(self, post, ctx):
        """Grade every rollout against the full rubric in one judge call.

        The judge is a model, so its output is persisted the way any model output
        is: ``extra["grader_output"]`` is its whole ``ModelOutput`` flattened to a
        plain dict (``add_type=False``, so the judgement record stays uniformly
        plain-dict). Nothing is hand-picked, so no field is silently dropped and
        the reply survives — the only durable evidence of a verdict set that a
        re-grade need not reproduce, and the only way to tell judge format drift
        (``n_grader_unparsed``) from a response that genuinely failed the rubric.

        ``extra["criterion_verdicts"]`` is one ``True``/``False``/``None`` per
        criterion, index-aligned to the prompt record's ``criteria`` — task
        logic on top of the raw reply, which is why it is a separate key. It is
        absent entirely on the empty-response path, where no judge call is made.
        """
        raw = ctx.raw_sample
        criteria = list(raw["criteria"])
        n_criteria = len(criteria)

        rollouts: list[RolloutJudgement] = []
        for rollout in post["rollouts"]:
            predicted = rollout.get("prediction")
            if predicted is None:
                # An empty response satisfies nothing, and asking the judge to
                # confirm that costs a call per rollout and invites a spurious
                # PASS on a criterion phrased as a prohibition ("should not
                # mention X"). No call, so no grader output to record -- its
                # ABSENCE is the durable signal here.
                rollouts.append(
                    build_rollout_judgement(
                        rollout["index"],
                        False,
                        score=0.0,
                        metrics={"task_pass": False, "criterion_pass_rate": 0.0},
                        extra={
                            "n_criteria": n_criteria,
                            "n_satisfied": 0,
                            "n_grader_unparsed": 0,
                        },
                    )
                )
                continue

            out = await self._grader.agenerate(
                build_grader_prompt(raw["prompt"], predicted, criteria)
            )
            reply = out.texts[0] if out.texts else ""
            verdicts = parse_verdicts(reply, n_criteria)
            n_satisfied = sum(1 for verdict in verdicts if verdict)
            n_grader_unparsed = sum(1 for verdict in verdicts if verdict is None)
            # Both published readings are co-equal metrics, so both go in
            # `metrics`; the headline merely points at task_pass. Derived from
            # the mapping, not recomputed, so the two cannot drift.
            metrics: dict[str, bool | float] = {
                "task_pass": n_criteria > 0 and n_satisfied == n_criteria,
                "criterion_pass_rate": n_satisfied / n_criteria if n_criteria else 0.0,
            }
            rollouts.append(
                build_rollout_judgement(
                    rollout["index"],
                    bool(metrics["task_pass"]),
                    score=float(metrics["criterion_pass_rate"]),
                    metrics=metrics,
                    extra={
                        "criterion_verdicts": verdicts,
                        "n_criteria": n_criteria,
                        "n_satisfied": n_satisfied,
                        "n_grader_unparsed": n_grader_unparsed,
                        GRADER_OUTPUT_KEY: obj_to_dict(out, add_type=False),
                    },
                )
            )

        # Sample-level partial credit: the mean criterion pass rate across
        # rollouts. Genuine partial credit rather than a mirror of
        # n_correct/n_rollouts, which already records the task-pass side.
        score = (
            sum(float(r["metrics"]["criterion_pass_rate"]) for r in rollouts)
            / len(rollouts)
            if rollouts
            else 0.0
        )
        return True, build_judgement_record(
            # The rubric is a procedure, not a value; `extra` describes it and
            # the criterion texts live once on the prompt record.
            None,
            rollouts,
            score=score,
            extra={
                "benchmark_id": raw["benchmark_id"],
                "n_criteria": n_criteria,
                "use_case": raw["use_case"],
                "instruction_type": raw["instruction_type"],
                "prompt_style": raw["prompt_style"],
            },
        )

    @override
    async def report(self, finals, fails):
        graded = [
            rollout
            for f in finals
            for rollout in (f.feedback_result or {}).get("rollouts", [])
        ]
        # Pooled from raw per-rollout counts rather than averaged from the
        # per-rollout rates -- the two differ when prompts carry different
        # criteria counts, which is exactly the macro/micro split below.
        units = [(r["extra"]["n_satisfied"], r["extra"]["n_criteria"]) for r in graded]

        # Pipeline failures (exhausted retries) never produced a gradeable
        # response; each failed sample stands in for its n requested attempts,
        # satisfying zero of its criteria, so all three rates span the full
        # requested set rather than only the successfully-graded subset. The
        # rubric size comes from the raw sample when the context still carries
        # one; without it the attempt still counts as a task failure but adds
        # nothing to the pooled denominator.
        for f in fails:
            n_criteria = len(f.raw_sample["criteria"]) if f.raw_sample else 0
            units.extend([(0, n_criteria)] * self._n)

        m = aggregate_metrics(units)
        return {
            "score": m["task_pass_rate"] * 100,
            "task_pass_rate": m["task_pass_rate"] * 100,
            "criterion_pass_rate_macro": m["criterion_pass_rate_macro"] * 100,
            "criterion_pass_rate_micro": m["criterion_pass_rate_micro"] * 100,
            "n_graded": len(graded),
            "n_criteria_graded": sum(r["extra"]["n_criteria"] for r in graded),
            # Judge format drift, kept out of the rates it would otherwise be
            # invisible inside: these criteria scored not-satisfied.
            "n_grader_unparsed": sum(r["extra"]["n_grader_unparsed"] for r in graded),
            "fails": len(fails),
            SCORE_KEY_FIELD: "task_pass_rate",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
            # `n_graded` counts the short-circuited empty responses too, so it
            # cannot separate "the judge failed every criterion" from "the model
            # returned nothing". `n_unextracted` is that second count, and the
            # two failure modes read identically without it. Deliberately only
            # `health_metrics` and not the rest of the sampling block: RFC #74
            # defers `pass@k` / `maj@k` for the LLM-judged family, while this one
            # measures the parser rather than the draw and is outside that gate.
        } | health_metrics(finals)
