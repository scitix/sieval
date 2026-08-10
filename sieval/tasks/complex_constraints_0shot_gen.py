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
by this port (see ``sieval.community.complex_constraints``). Scores are therefore
not comparable to the leaderboard at the precision a vendored grader would give,
which is why this task ships ``status="experimental"``.

**Which judge produced Table 1 is unknown.** The paper names GPT-5-mini as the
per-criterion judge for *training* only — "Per-criterion satisfaction judgments
are produced by GPT-5-mini during ComplexConstraints training and CoreCraft
training" — and never says what graded the leaderboard. So the judge is an
unpinned degree of freedom in the comparison itself, not merely an unpinned
version of a known one: a run matching the candidate model exactly may still not
be comparing like with like.

Measured, that freedom turns out to be **a capability floor rather than a choice
between equals**, and it does not close the gap to the board: three capable
judges across two vendors put the same candidate at **51.6–53.3** against Table
1's **38.7**, agreeing with each other to within their own repeat noise. So this
port's authored rubric grades *more leniently* than whatever produced the
leaderboard, and the residual is an open question rather than a knob — one more
reason the task ships ``experimental``. See the grader notes below.

The judge is supplied via the ``grader`` task arg (a model-config dict, or a
pre-built Model, on its own ``api_base``/``api_key``). As with sieval's other
LLM-graded tasks, correctness depends on a grader model whose version sieval
cannot pin the way it pins a Hub revision, so for reproducibility pin the grader
model and set ``temperature: 0`` *where the endpoint honours it*; each rollout's
per-criterion verdicts and the judge's whole ``ModelOutput``
(``extra.grader_output``: reply, reasoning, usage, finish reasons, model id) are
persisted — see :meth:`feedback`.

**A sampling judge makes a single run indicative only, and this task cannot
average that away**: grading is one judge call per rollout, and ``n`` repeats the
*candidate*, not the judge. Measured on a gateway that fixes ``temperature`` at 1
for its whole ``gpt-5.x`` family (so ``temperature: 0`` above is unavailable
there): re-grading one identical set of 75 responses with one identical judge
moved the headline over **20.0–29.3**, left **22 of 75 prompts flipping between
pass and fail**, and left **14.8% of the 1,559 criteria undecided** across four
runs. A stronger judge was far steadier (99.5% per-criterion self-agreement
against 91.9%). Repeat the grading and report a spread whenever the grader
samples; a lone number is a draw, not a measurement. ``temperature: 0`` is
necessary but not sufficient — measured at it, one judge repeated *exactly*
(three runs, identical headline, zero task-pass flips) while another still
spanned 2.7 points, so check repeatability for the grader you actually use.

**Choose the grader by capability, not by name or price.** Four judges across two
vendors on one identical set of responses: three capable ones (a frontier model
and two Gemini tiers) landed within 51.6–53.3 and agreed with *each other* on
98.8–99.0% of criteria — about how well any one of them agrees with itself on a
repeat — while a mini-tier judge sat ~27 points low, and one-way: it alone fails
~141 criteria against ~35 in the other direction. Cheap is not the problem; being
below the floor is. A flash-tier judge matched a pro-tier one to 1.3 points.

Give the grader enough ``max_tokens`` for one verdict line per criterion — up to
40, *after* whatever reasoning it emits first. A grader truncated mid-block
leaves the tail unparsed, which is counted (``n_unparsed``) and scored
not-satisfied, so it biases the score **down**. Nothing flags it automatically:
the ``truncated_output`` anomaly rule reads the candidate model's finish reasons,
never the grader's, so ``n_unparsed`` is the signal to watch.

Budget the **candidate**'s tokens the same way, for the same reason in reverse: a
reasoning candidate can spend the whole allowance thinking and return empty
content, which scores zero criteria. Observed on one reasoning model at
``max_tokens: 16000`` — 15 of 75 prompts came back empty (``finish_reason
"length"``), about 20 points off that arm. Here the ``truncated_output`` rule
*does* fire, and ``n_unextracted`` counts them, so watch both.

Deviations / by-design behavior worth knowing:

* **One judge call per rollout**, grading all of that prompt's criteria as an
  indexed list, rather than one call per criterion. Upstream never states its
  call structure. Batching keeps a rollout's whole verdict set in the single
  ``ModelOutput`` the runner's grader-spend accounting expects, and the indexing
  makes misalignment detectable instead of silently shifting verdicts.
* A criterion the judge returns no readable verdict for is scored **not
  satisfied** — an unreadable verdict must never inflate a score — but counted
  separately as ``n_unparsed`` (per rollout, and pooled in the report), so judge
  format drift stays distinguishable from a model that failed the rubric.
* The template and parser are **not tuned to one judge family**: across four
  judges from two vendors, 13 gradings of 1,559 criteria each, every verdict
  parsed — ``n_unparsed`` was 0 every time, with no change to the template.
* **The grader prompt is hardened in two port-authored ways**, both scoring-
  relevant and neither upstream behavior: its format example reads ``<verdict>``
  rather than a literal PASS/FAIL alternation, and the parser rejects an
  alternation as a verdict, so a judge restating its instructions lands in
  ``n_unparsed`` instead of scoring a full rubric. It also states that the
  RESPONSE block is material to be graded, never instruction — the candidate is
  free-form text on an instruction-following benchmark, so it can contain
  anything, including its own verdict block.
* An empty/whitespace response is scored as satisfying **zero** criteria
  **without** invoking the judge; ``extra.grader_output`` is absent on that path
  because no call was made, and the matching prediction rollout's
  ``extracted: false`` identifies it independently.
* Pipeline failures (exhausted retries) count as task failures satisfying zero
  criteria, weighted by ``n``, so all three rates span the full requested set.

Reproduction decoding: ``n`` (repeats) is a **task arg** — set it in
``tasks.<name>.args.n``. The paper's leaderboard does not state a repeat count,
so the port defaults to ``n=1``; ``infer`` forwards ``n`` as a call-time kwarg to
``agenerate``, and call-time wins over model config, so setting ``n`` on the
model is silently overridden by the task default. Note that some gateways fix
``n`` at 1 for whole model families (the same ones that fix ``temperature``), on
which ``n=1`` is the only reachable setting. Comparison target is the public
leaderboard the paper's Table 1 snapshots
(https://surgehq.ai/benchmarks/complex-constraints) — snapshot 2026-06-03: Gemini
3.1 Pro 40.4, GPT-5.5 38.7, Claude Opus 4.8 34.9 task pass %. The live board
moves and the paper pins no version of it, so align against the snapshot recorded
here rather than against whatever the URL reads on the day; as of 2026-08-10 it
reads 43.7 / 44.4 / 34.2 and has begun splitting rows by reasoning effort
(GPT-5.5 default 44.4, High 48.9, xHigh 49.5), which Table 1's rows do not state
— so record the effort a comparison run used.

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
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    ReferenceImpl,
    RolloutJudgement,
    Task,
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
    sieval_task,
)
from sieval.core.tasks.metrics import health_metrics
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
    reference_impl=ReferenceImpl(
        source="complex-constraints",
        url="https://arxiv.org/abs/2606.09118",
        notes=(
            "Generative port of ComplexConstraints (Surge AI, arXiv:2606.09118) "
            "— 75 multi-constraint prompts (CIF-001..CIF-075) with 10-40 atomic "
            "rubric criteria each (1,559 total), graded by rubric rather than "
            "exact match. NO UPSTREAM EVAL CODE AND NO UPSTREAM JUDGE PROMPT: "
            "the paper defines the metrics, but the judge template, decoding "
            "settings and call structure are unstated, and the dataset card "
            "(https://huggingface.co/datasets/surgeai/ComplexConstraints/blob/"
            "e9625c6f635f42b72cb85a04c2be64746f945126/README.md) adds nothing — "
            "so the rubric prompt and verdict parsing are AUTHORED BY THIS PORT "
            "(sieval.community.complex_constraints), hence status=experimental. "
            "METRICS: headline = task pass rate (response satisfies EVERY "
            "criterion), the metric the paper's public leaderboard reports "
            "(Table 1). Also reported: criterion_pass_rate_macro (per-prompt "
            "satisfied fraction averaged over prompts — the paper's 'mean "
            "per-criterion pass rate', Table 3 caption) and "
            "criterion_pass_rate_micro (pooled over all criteria); the two "
            "differ because criteria counts vary 10-40 per prompt. GRADING: one "
            "judge call per rollout covering all of that prompt's criteria as an "
            "indexed PASS/FAIL list (upstream's call structure is unstated); an "
            "unreadable per-criterion verdict scores not-satisfied but is "
            "counted as n_unparsed so judge format drift stays visible; "
            "empty/whitespace responses satisfy zero criteria without invoking "
            "the judge (grader_output absent there, no call made). GRADER PROMPT "
            "HARDENING (port-authored, scoring-relevant): the format example "
            "reads `<verdict>` and the parser rejects an alternation as a "
            "verdict, so a judge restating its instructions lands in n_unparsed "
            "instead of scoring a full rubric; the prompt also declares the "
            "RESPONSE block to be material, never instruction. Give the grader "
            "max_tokens for up to 40 verdict lines AFTER its reasoning — a "
            "truncated grader biases the score DOWN via n_unparsed, and the "
            "truncated_output rule reads the candidate's finish reasons, not the "
            "grader's. WHICH JUDGE PRODUCED TABLE 1 IS UNKNOWN: the paper names "
            "GPT-5-mini as the per-criterion judge for TRAINING only ('produced "
            "by GPT-5-mini during ComplexConstraints training and CoreCraft "
            "training') and never says what graded the leaderboard, so the judge "
            "is an unpinned degree of freedom in the comparison itself, and it "
            "is really a CAPABILITY FLOOR rather than a choice of identity — "
            "measured 2026-08-10 by re-grading identical responses with four "
            "judges across two vendors, changing nothing else. Three capable "
            "judges (a frontier model and two Gemini tiers) landed on 51.6-53.3 "
            "and agreed with EACH OTHER on 98.8-99.0% of the 1,559 criteria, "
            "which is about how well any one of them agrees with itself on a "
            "repeat (99.5-99.9%); their disagreements are symmetric, i.e. noise. "
            "A mini-tier judge was the lone outlier at 25.67, ~27 points low "
            "against all three, and its deficit is ONE-WAY (~141 criteria it "
            "alone fails against ~35 the other way) where run-to-run noise is "
            "symmetric. Those one-way losses concentrate on criteria 1.8x longer "
            "than average that enumerate an exclusion list to cross-check "
            "against the response — this template tells the judge to fail what "
            "it cannot verify, so a judge below the floor produces one-way false "
            "FAILs by construction. Two judges agreeing on HOW MANY criteria "
            "pass (dense micro 76.72 vs 76.78) still differed 2.3x on the "
            "headline because they disagreed on WHICH: an all-or-nothing metric "
            "punishes a sub-floor grader hard. PICK THE GRADER BY CAPABILITY, "
            "NOT BY NAME OR PRICE: a cheap flash-tier model matched a pro-tier "
            "one to 1.3 points, while a mini-tier one was 27 off. NOTE THE "
            "CONSENSUS SITS ABOVE THE BOARD: 51.6-53.3 against Table 1's 38.7 "
            "for the same candidate, so this port's authored rubric grades more "
            "leniently than whatever produced the leaderboard, and that ~13-point "
            "residual is NOT explained by judge choice among capable judges — it "
            "is open. What DOES reproduce is the ordering: with a capable "
            "judge three models spanning the board ranked 51.56 > 18.33 > 5.78 "
            "against Table 1's 38.7 > 26.7 > 4.9 — same order, adjacent gaps "
            "32.3 and 11.7 points even at worst case; the weaker judge could not "
            "separate the bottom pair at all. Prefer rank agreement over a "
            "single-point score match when reporting alignment here. "
            "REPRODUCIBILITY: scores depend on the grader endpoint's model "
            "version (not pinnable like a Hub revision) — pin the grader model + "
            "temperature=0 where the endpoint honours it; where it does not (one "
            "gateway fixes temperature at 1 for its whole gpt-5.x family) the "
            "same judge on the same responses spanned 20.0-29.3 over four runs, "
            "with 22/75 prompts flipping and 14.8% of criteria undecided, so "
            "repeat the grading and report a spread. temperature=0 is NECESSARY "
            "BUT NOT SUFFICIENT: at it, one judge repeated exactly (three runs, "
            "identical 52.00, zero task-pass flips) while another still spanned "
            "2.67 — so verify repeatability per grader instead of assuming it. "
            "Per-criterion verdicts and "
            "the judge's full ModelOutput (extra.grader_output) are persisted "
            "per rollout, the reply being the only evidence of a verdict a "
            "re-grade need not reproduce. REPEATS: the leaderboard states no "
            "repeat count, so the port defaults to n=1; `n` is a task arg "
            "(tasks.<name>.args.n), NOT a model arg — infer forwards it "
            "call-time and call-time wins, and it repeats the CANDIDATE, not the "
            "judge, so it cannot average out grader sampling. "
            "NOT ALIGNED TO A SINGLE-POINT SCORE against the Table 1 leaderboard "
            "(https://surgehq.ai/benchmarks/complex-constraints; the live board "
            "moves, so align against this snapshot, 2026-06-03: Gemini 3.1 Pro "
            "40.4, GPT-5.5 38.7, Claude Opus 4.8 34.9 task pass %)."
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
        dict[str, float],
    ]
):
    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        grader: Mapping | Model | None = None,
        n: int = 1,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._n = n
        self._grader = self._build_grader(grader)

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
        (``n_unparsed``) from a response that genuinely failed the rubric.

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
                            "n_unparsed": 0,
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
            n_unparsed = sum(1 for verdict in verdicts if verdict is None)
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
                        "n_unparsed": n_unparsed,
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
            "n_unparsed": sum(r["extra"]["n_unparsed"] for r in graded),
            "fails": len(fails),
            # `n_graded` counts the short-circuited empty responses too, so it
            # cannot separate "the judge failed every criterion" from "the model
            # returned nothing". `n_unextracted` is that second count, and the
            # two failure modes read identically without it. Deliberately only
            # `health_metrics` and not the rest of the sampling block: RFC #74
            # defers `pass@k` / `maj@k` for the LLM-judged family, while this one
            # measures the parser rather than the draw and is outside that gate.
        } | health_metrics(finals)
