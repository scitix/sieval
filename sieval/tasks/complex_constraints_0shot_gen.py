"""ComplexConstraints — 0-shot generative, LLM-judge graded against a rubric.

Generative port of ComplexConstraints (Mehta et al., 2026, arXiv:2606.09118): the
model answers one realistic multi-constraint instruction, and a separate **LLM
judge** grades the free-form response against that prompt's 10-40 atomic rubric
criteria, one PASS/FAIL verdict each. The headline metric is the **task pass
rate** — the fraction of prompts whose response satisfies *every* criterion —
which is what the paper's public 75-prompt leaderboard reports (its Table 1).
The paper's other metric, the mean per-criterion pass rate, is reported
alongside it in both the macro (published) and pooled-micro readings.

Upstream ships **no evaluation code and no judge prompt** — the paper names
GPT-5-mini as the judge and defines the metrics, nothing more — so the rubric
prompt and verdict parsing are authored by this port (see
``sieval.community.complex_constraints``). Scores are therefore not comparable to
the leaderboard at the precision a vendored grader would give, which is why this
task ships ``status="experimental"``.

The judge is supplied via the ``grader`` task arg (a model-config dict, or a
pre-built Model, on its own ``api_base``/``api_key``). As with sieval's other
LLM-graded tasks, correctness depends on a grader model whose version sieval
cannot pin the way it pins a Hub revision, so for reproducibility pin the grader
model and set ``temperature: 0``; each rollout's per-criterion verdicts and the
judge's whole ``ModelOutput`` (``extra.grader_output``: reply, reasoning, usage,
finish reasons, model id) are persisted — see :meth:`feedback`.

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
model is silently overridden by the task default. Comparison target is the
paper's Table 1 leaderboard (snapshot 2026-06-03; Gemini 3.1 Pro 40.4, GPT-5.5
38.7, Claude Opus 4.8 34.9 task pass %).

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
            "the paper names GPT-5-mini as the per-criterion judge and defines "
            "the metrics, but the template, decoding settings and call structure "
            "are unstated, and the dataset card "
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
            "the judge (grader_output absent there, no call made). "
            "REPRODUCIBILITY: scores depend on the grader endpoint's model "
            "version (not pinnable like a Hub revision) — pin the grader model + "
            "temperature=0; per-criterion verdicts and the judge's full "
            "ModelOutput (extra.grader_output) are persisted per rollout, the "
            "reply being the only evidence of a verdict a re-grade need not "
            "reproduce. REPEATS: the leaderboard states no repeat count, so the "
            "port defaults to n=1; `n` is a task arg (tasks.<name>.args.n), NOT "
            "a model arg — infer forwards it call-time and call-time wins. "
            "NOT YET VALIDATED against the Table 1 leaderboard (snapshot "
            "2026-06-03: Gemini 3.1 Pro 40.4, GPT-5.5 38.7, Claude Opus 4.8 "
            "34.9 task pass %)."
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
        }
