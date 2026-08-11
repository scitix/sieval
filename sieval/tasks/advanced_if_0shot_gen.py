"""AdvancedIF — 0-shot generative, rubric-graded by an LLM judge.

AdvancedIF (Meta, Hu et al., 2025, arXiv:2511.10507) probes instruction
following past the verifiable-constraint regime IFEval and IFBench cover: every
prompt is expert-written and paired with a human-curated rubric of yes/no
checks, and the three aspects it spans -- complex single-turn instructions
(6+ per prompt), instructions carried across a multi-turn dialog, and system-
prompt steerability -- are graded by an LLM judge rather than by checkers.

The model answers the conversation's final user turn; a separate **grader**
model then answers every rubric question and declares whether the response
satisfied all of them. Headline metric is the overall pass rate: the share of
samples where the grader answered yes to that declaration.

The grader is supplied via the ``grader`` task arg (a model-config dict, or a
pre-built Model, on its own ``api_base``/``api_key``). Upstream's judge is
o3-mini-2025-01-31 with ``max_completion_tokens=32768`` and
``response_format={"type": "json_object"}`` and **no temperature** -- it stores
one but never sends it, and a reasoning-model grader rejects an explicit value.
Set those in the grader's model config, not here; sieval does not force
``response_format`` on the request, since an endpoint that rejects the field
would fail the whole run, and the reply parser tolerates fenced JSON anyway.

Upstream's judge routing is reproduced with its defect intact: the
system-steerability judge is selected on a ``benchmark_name`` the released
dataset never contains, so every row -- including the 507 system-prompt ones --
is graded by the user-instruction judge. The unqualified task name tracks
upstream, defects included; see :mod:`sieval.community.advanced_if`, which also
documents the two grader-reply deviations.

Running this task needs an upstream checkout: the judge prompts are CC-BY-NC-4.0
and are not redistributed with sieval, so point ``SIEVAL_ADVANCED_IF_SRC`` at
your own clone (digest-checked against the pinned commit). The data is
CC-BY-NC-4.0 too.

AI-Generated Code - Claude Opus 5 (Anthropic)
"""

from collections.abc import Mapping
from typing import override

from sieval.community.advanced_if import (
    aggregate_metrics,
    compose_judge_prompt,
    count_all_checks,
    count_in_range_passes,
    load_judge_prompts,
    parse_conversation,
    parse_judgement,
    parse_rubrics,
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
from sieval.datasets import AdvancedIFDatasetSample


@sieval_task(
    name="advanced_if_0shot_gen",
    display_name="AdvancedIF (0-shot, generative)",
    description=(
        "Instruction following graded against expert-written rubrics by an LLM judge."
    ),
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended"),
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="facebookresearch/AdvancedIF",
        url=(
            "https://github.com/facebookresearch/AdvancedIF/blob/"
            "f9d30137c4139d4d9af260ae28108b5afae828c0/judge.py"
        ),
        notes=(
            "Port of AdvancedIF (Meta, arXiv:2511.10507), 1,645 prompts across "
            "complex_if_single_turn_v5 (402), system_steerability_v2 (507) and "
            "carried_context_multi_turn_eval_v5 (736). Headline score is the "
            "overall pass rate (share of samples the grader marked "
            "SATISFIED_ALL_REQUIREMENTS=yes), the number the paper reports; "
            "micro_pass_rate is the co-published rubric-level rate; "
            "macro_pass_rate is sieval's, NOT published, averaging the "
            "per-sample rate so every sample weighs the same. "
            "LICENSING: upstream ships judge.py under CC-BY-NC-4.0, "
            "incompatible with sieval's Apache-2.0 tree, so the prompts are NOT "
            "vendored -- the operator stages a checkout and points "
            "SIEVAL_ADVANCED_IF_SRC at it, digest-checked against commit "
            "f9d30137c4139d4d9af260ae28108b5afae828c0 (which also makes them "
            "byte-exact by construction). The dataset is CC-BY-NC-4.0 too, so "
            "running this benchmark accepts those terms either way. "
            "UPSTREAM DEFECT (reproduced, not corrected): upstream routes to the "
            "system-steerability judge on benchmark_name == "
            "'if_system_steerability_oss', a value the released dataset never "
            "contains (it ships 'system_steerability_v2'), so all 507 "
            "system-prompt rows are graded by the plain user-instruction judge "
            "and the CLI's --task choices match zero rows; "
            "processor.process_file's own docstring uses the released spelling, "
            "so the if_*_oss literals are what went stale. Kept verbatim so the "
            "unqualified name tracks upstream. Correcting it is MEASURED at "
            "+0.00 on the SS pass rate (507 rows, same responses and grader, "
            "judge prompt 5,893 -> 31,598 chars) while flipping 13.8% of "
            "individual verdicts, so no _fixed variant ships -- it would matter "
            "only for per-sample analysis. "
            "DEVIATIONS (two, both in grader-reply handling): (1) a fenced/"
            "prose-wrapped JSON verdict is recovered, since sieval cannot force "
            "response_format on an arbitrary endpoint and a fenced reply would "
            "otherwise score zero everywhere -- the same scan also recovers a "
            "verdict wrapped in a JSON array, which upstream fails the row on; "
            "(2) a non-string rubric answer is stringified instead of failing "
            "the row -- with the declaration NO that moves micro_pass_rate "
            "alone, with YES it also moves the "
            "headline (upstream fails the row, this port honours the "
            "declaration). (1) does fire in practice, on endpoints that drop "
            "response_format; (2) never fired in 1,644 live gradings. "
            "GRADER is a REAL LLM (upstream: o3-mini-2025-01-31, "
            "max_completion_tokens=32768, response_format=json_object) supplied "
            "via the `grader` task arg; pin it, as its version is not pinnable "
            "the way a Hub revision is. Send NO temperature: upstream stores "
            "self.temperature but _call_openai_api never passes it, and a "
            "reasoning-model grader rejects an explicit value outright "
            "('temperature, top_p and n are fixed at 1'), o3-mini included -- "
            "setting it fails every grader call, and a wholly failed grading "
            "stage still reports score 0.0, which reads as a score rather than a "
            "floor. Give the grader a real thinking budget: at one endpoint's "
            "default effort the same grader emitted 0-char reasoning and ~97 "
            "output tokens for up to 20 rubrics and scored 17 points lower than "
            "itself at high effort. Its full ModelOutput and per-rubric answers "
            "are persisted under the judgement record's `extra`. "
            "VALIDATION: the port is verified against upstream, but NO published "
            "number is reproducible. Verified model-free -- judge prompts "
            "byte-identical to upstream's own _compose_prompt on all 1,645 rows "
            "for both judge classes; per-row verdicts identical to upstream's "
            "real evaluate() on 21,385 (row, reply) pairs across 13 reply "
            "shapes; pooled rates exact on every schema-conforming shape; paper "
            "Table 1 reproduced from these loaders (counts 402/736/507, avg "
            "criteria 7.44/6.08/9.81, avg turns 1.00/7.69/11.21). NOT "
            "reproducible -- the paper's judge and every model it evaluates "
            "(GPT-5, GPT-4.1, Gemini 2.5/3 Pro, Claude 4 Sonnet, Llama 4 "
            "Maverick) are legacy or retired, and judge choice alone spans ~29 "
            "points because the headline is all-or-nothing over ~7.6 rubrics: "
            "raising judge effort moved per-rubric micro only 87.2->91.8 but "
            "all-pass 48.0->65.1, a 3.4x amplification. Judge instability on "
            "identical rows: 86.6% per-sample self-agreement (same judge, same "
            "config), 69.1% across effort settings, 61.8% across families. The "
            "best available judge put gpt-5.1 at macro 76.7 against the paper's "
            "GPT-5 77.9, but with per-aspect errors of +5.1/+0.1/-8.8 that "
            "merely cancel -- not agreement, and must not be quoted as such. "
            "Upstream releases no calibration set (its Table 4 human labels are "
            "unpublished), so a replacement judge cannot be checked against "
            "human agreement first; pin and report the judge beside every score "
            "so runs stay comparable to each other."
        ),
    ),
    # The routing and scoring kernel is verified against upstream's own code
    # (see VALIDATION above), but no published number is reproducible: the
    # paper's judge and evaluated models have aged out, and judge choice moves
    # this metric ~29 points. Stays experimental until either the original judge
    # is reachable or a substitute validation bar is agreed for judge-dependent
    # benchmarks whose judge has been retired.
    status="experimental",
)
class AdvancedIFZeroShotGenTask(
    Task[
        AdvancedIFDatasetSample,
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
        # Validate the checkout here, not at the first grade. Discovered in
        # feedback() it costs a whole generation pass to learn: every grade then
        # raises, and a wholly failed grading stage still reports 0.0 -- the
        # floor that reads as a score. `@cache`d, so this is one read per run,
        # and construction already fails without a grader anyway.
        load_judge_prompts()

    @staticmethod
    def _build_grader(grader: Mapping | Model | None) -> Model:
        """Resolve the ``grader`` task arg into a Model.

        Accepts a pre-built Model (tests / advanced configs) or a model-config
        mapping (the YAML path). Rubric grading is the only scorer AdvancedIF
        has -- there is no deterministic fallback -- so ``None`` raises.
        """
        if isinstance(grader, Model):
            return grader
        if isinstance(grader, Mapping):
            return ChatModel(**grader)
        raise ValueError(
            "AdvancedIF requires an LLM grader. Pass `grader:` in the task "
            "args — a model-config dict such as {model: o3-mini, api_base: "
            "..., api_key: ..., max_completion_tokens: 32768}. Send no "
            "temperature: a reasoning-model grader rejects an explicit value."
        )

    @override
    async def preprocess(self, raw, ctx):
        messages = parse_conversation(raw["conversation_history"])
        # No `reference`: the ground truth is a rubric (a procedure), not a
        # value. The rubric itself rides in `extra` so a prompt row is readable
        # on its own and feedback() need not re-decode the raw sample.
        return build_prompt_record(
            messages,
            extra={
                "benchmark_name": raw["benchmark_name"],
                "rubrics": parse_rubrics(raw["prompt_metadata"]),
            },
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        # Open-ended: the response *is* the answer, so no extraction step. A
        # blank response normalizes to None so `extracted` stays a real signal;
        # the grader still sees "" and will fail its rubrics.
        return build_prediction_record(
            [text if text.strip() else None for text in inf.texts]
        )

    @override
    async def feedback(self, post, ctx):
        """Grade every rollout against its rubrics, recording the judge's reply.

        ``extra["grader_output"]`` is the grader's whole ``ModelOutput``, nothing
        hand-picked. That matters more here than for a short-answer autorater: an
        unparseable reply scores the same as a failing one, so only the raw text
        separates a grader that broke format from a response that genuinely
        missed every rubric.

        Per-rubric answers and the two raw counts also live in the rollout's
        ``extra``, because the published micro rate pools over the grader's own
        answer keys -- which a per-sample rate cannot reconstruct.
        """
        prompt_extra = ctx.preprocess_result["extra"]
        benchmark_name = prompt_extra["benchmark_name"]
        rubrics = prompt_extra["rubrics"]
        messages = ctx.preprocess_result["prompt"]

        rollouts: list[RolloutJudgement] = []
        for rollout in post["rollouts"]:
            response = rollout.get("prediction") or ""
            out = await self._grader.agenerate(
                compose_judge_prompt(benchmark_name, messages, response, rubrics)
            )
            reply = out.texts[0] if out.texts else ""
            judgement = parse_judgement(reply)

            if judgement is None:
                # Upstream's failed-row path: counts against the pass rate,
                # contributes no rubrics to the pooled micro rate.
                checks: dict[str, str] = {}
                satisfied_all = False
                n_checks, n_checks_passed = 0, 0
                pass_rate = 0.0
            else:
                checks = judgement.rubrics_check
                satisfied_all = judgement.satisfied_all
                n_checks, n_checks_passed = count_all_checks(checks)
                pass_rate = count_in_range_passes(checks, rubrics) / max(
                    len(rubrics), 1
                )

            rollouts.append(
                build_rollout_judgement(
                    rollout["index"],
                    satisfied_all,
                    score=pass_rate,
                    metrics={
                        "satisfied_all": satisfied_all,
                        "rubric_level_pass_rate": pass_rate,
                    },
                    extra={
                        "judge_parsed": judgement is not None,
                        "rubrics_check": checks,
                        "n_checks": n_checks,
                        "n_checks_passed": n_checks_passed,
                        GRADER_OUTPUT_KEY: obj_to_dict(out, add_type=False),
                    },
                )
            )

        # `max(..., 1)` for the same reason the grader's reply is read with a
        # guard: a backend can return a ModelOutput with no texts, which leaves
        # `postprocess` with zero rollouts to grade and nothing to average.
        mean_pass_rate = sum(r["score"] for r in rollouts) / max(len(rollouts), 1)
        return True, build_judgement_record(
            # Rubric grading is a procedure, not a value to compare against.
            None,
            rollouts,
            score=mean_pass_rate,
            extra={"benchmark_name": benchmark_name, "n_rubrics": len(rubrics)},
        )

    @override
    async def report(self, finals, fails):
        """Pool the two published rates -- plus the macro -- overall and per aspect.

        Reads the persisted verdicts rather than ``raw_sample``, so the report
        survives a resume. Pipeline failures (exhausted retries) never produced a
        gradeable answer; each failed sample's requested rollouts count as
        non-passes so the headline spans the full requested set, matching
        upstream's denominator (which likewise includes rows its judge failed).

        ``macro_pass_rate`` is sieval's, not published -- it lifts the per-sample
        rate to ``report.json``, where it would otherwise stop at the shard data.

        ``n_judge_unparsed`` is lifted for the same reason, and matters more:
        every metric here treats an unparseable grader reply exactly like a
        response that missed every rubric, so without the count a broken grader
        and a bad model are the same number. Reported per aspect too, since a
        grader tends to break format on the long multi-turn rows first.
        """
        by_benchmark: dict[str, list[dict]] = {}
        verdicts: list[dict] = []
        for final in finals:
            judgement = final.feedback_result or {}
            benchmark_name = judgement.get("extra", {}).get("benchmark_name", "unknown")
            for rollout in judgement.get("rollouts", []):
                extra = rollout.get("extra", {})
                verdict = {
                    "satisfied_all": rollout["correct"],
                    "n_checks": extra.get("n_checks", 0),
                    "n_checks_passed": extra.get("n_checks_passed", 0),
                    # `score` is the same number `metrics` carries under
                    # `rubric_level_pass_rate`, and feedback() always writes it,
                    # so it is read without a default: a rename cannot silently
                    # report a macro rate of zero.
                    "rubric_pass_rate": rollout["score"],
                    # Absent means an older record, not a broken grade -- default
                    # to parsed so the diagnostic under-reports rather than
                    # inventing failures.
                    "judge_unparsed": not extra.get("judge_parsed", True),
                }
                verdicts.append(verdict)
                by_benchmark.setdefault(benchmark_name, []).append(verdict)

        n_graded = len(verdicts)
        # A failed sample has no verdict to attribute to an aspect, so it lands
        # in the overall rates only; the per-aspect rates below cover graded
        # rollouts, and `fails` reports the shortfall.
        failed = [
            {
                "satisfied_all": False,
                "n_checks": 0,
                "n_checks_passed": 0,
                "rubric_pass_rate": 0.0,
                # Not a grader-parse failure: no grade was ever attempted. It is
                # `fails` that reports these.
                "judge_unparsed": False,
            }
        ] * (self._n * len(fails))

        overall = aggregate_metrics(verdicts + failed)
        results: dict[str, float] = {
            "score": overall["overall_pass_rate"],
            "overall_pass_rate": overall["overall_pass_rate"],
            "micro_pass_rate": overall["micro_pass_rate"],
            "macro_pass_rate": overall["macro_pass_rate"],
            "n_rubric_checks": overall["n_rubric_checks"],
            "n_judge_unparsed": overall["n_judge_unparsed"],
            "n_graded": float(n_graded),
            "fails": len(fails),
        }
        for benchmark_name, group in sorted(by_benchmark.items()):
            aspect = aggregate_metrics(group)
            results[f"{benchmark_name}_pass_rate"] = aspect["overall_pass_rate"]
            results[f"{benchmark_name}_micro_pass_rate"] = aspect["micro_pass_rate"]
            results[f"{benchmark_name}_macro_pass_rate"] = aspect["macro_pass_rate"]
            results[f"{benchmark_name}_n_judge_unparsed"] = aspect["n_judge_unparsed"]
            results[f"{benchmark_name}_n_graded"] = aspect["n_samples"]
        return results
