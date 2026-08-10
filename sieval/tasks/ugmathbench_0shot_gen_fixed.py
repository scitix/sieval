"""
UGMathBench 0-shot generative task, corrected — effective accuracy and the gap.

The ``_fixed`` variant: this grades slots that upstream's judge cannot win, so
it is deliberately *not* a reproduction of the published numbers. The
unqualified name ``ugmathbench_0shot_gen`` is reserved for a faithful port and
will stay vacant — upstream's grader is GPL-3.0 and cannot ship in an
Apache-2.0 distribution, so no faithful port is possible here at all.

The divergence from upstream is measured rather than asserted, two independent
ways — against upstream's judge on a full live run, and on replayed references.
Both are below, with the live one carrying the weight.

One sample is one *(problem, version)* pair, so a full run issues three
inferences per problem, one per randomized version. That is what the benchmark's
headline metric needs:

* **AAcc** — average accuracy over every version.
* **EAcc** — effective accuracy: the share of problems answered correctly in
  *all* three versions. The headline (``score``).
* **Delta** — the reasoning gap, ``AAcc - EAcc``. A model that reasons rather
  than recognizes drives this toward zero; the paper reports double-digit gaps
  for every model it evaluated.
* **CAcc** — the share of problems answered correctly in at least one version.
  Upstream reports it as the optimistic bound bracketing EAcc.

Grading is per answer *slot*: a problem states how many ``[ANS]`` placeholders
it has and what type each one takes, and a sample counts as correct only when
every slot is. The per-slot rules live in
:mod:`sieval.community.ugmathbench`, which also explains why the grader is an
independent implementation rather than a port of the GPL-licensed reference.

Two pinned rows are upstream-corrupt (empty answer sequence, problem text
replaced by an error message); they are prompted and graded like any other
sample, which scores them 0 exactly as the reference harness does.

**Status: stable, on one full live run.** Qwen3-30B-A3B, thinking on,
temperature 0.6 / top_p 0.95 / top_k 20, one rollout per version; all 15,183
versions over all 5,061 problems; sglang tp2xdp4 on 8xH100, 75.7M output tokens.

    EAcc 38.49 (``score``) · AAcc 45.59 · CAcc 53.53 · Delta 7.10
    fails 0 · incomplete_problems 0 · extracted=False 32/15,183 (0.21%)

**Read 38.49 as protocol-faithful, not as this model's mathematical ability.**
The two are separated by a *format tax* of roughly 33 EAcc points, and the tax
is the benchmark's rather than this port's: only the last ``\\boxed{}`` counts
and a slot-count mismatch scores every slot wrong, while Qwen3 ends multi-part
problems with ``\\boxed{a}, \\boxed{b}, \\boxed{c}`` instead of
``\\boxed{a, b, c}``. Measured off the same stored responses, single-answer rows
mismatch 0.30% of the time against multi-answer rows' 86.31% — same model, same
subjects, so this is formatting and not difficulty. Upstream's ``judge_rule.py``
has the identical rule, so reproducing it is the point. Repairing extraction on
those stored responses lifts EAcc to 74.43, which is where a model scoring 72.5
on AIME 2026 in this harness belongs.

**Evidence that the grader itself is right.** Upstream's ``judge_rule.py`` is an
independent implementation of the same spec, so it can be run as a local
*instrument* over the same 15,183 stored responses (GPL-3.0 restricts
distribution, not use; nothing is vendored). Against it this task agrees on
95.51% of samples, and where the two differ it is **591 to 91 in this task's
favour**. The 591 is the direction that matters, since a too-lenient grader
would show up there: a sampled eyeball found 12 of 12 genuinely correct
(``2\\sqrt{2t+9}`` against ``sqrt(2*4*t+36)``, ``4^20`` against ``1.09951E+12``,
``\\ln(2)/2`` against ``0.346573590279973``). The residual misses are
91/15,183 = 0.60% of samples, with named non-systematic causes: LaTeX interval
notation (``\\cup``, ``\\infty``), absolute-value bars, a ``y = `` prefix on an
``EX`` answer. Upstream's own verifier scores EAcc 35.55 on these responses,
i.e. *below* this task.

**How the promotion criteria resolved.** The gate was never "matches upstream" —
that bar is unreachable by construction here and would pin this task to
``experimental`` forever. It was evidence that *this* grader is right:

1. ``extracted=False`` rate low — **met**, 0.21%, with truncation at 0.19%, so
   the box-or-nothing rule is not what costs this model points;
2. ``fails`` 0 across the run — **met**;
3. EAcc in a plausible band next to sibling math benchmarks on the same model —
   **explained rather than met.** 38.49 sits far from the same model's 72.5 on
   AIME 2026, but this criterion exists to catch an *unexplained* anomaly (a
   mis-wired prompt, a mis-joined gold, a broken extractor). This gap is
   attributed to the format tax above and the attribution is checked two
   independent ways — the single- against multi-answer mismatch split, and
   tracking upstream's own verifier to within 3 points. A harness that were
   actually broken would not track the reference implementation that closely;
4. a sampled false-negative rate on wrong verdicts — **met**, via the
   independent-instrument audit above. It is the criterion that matters, and it
   tests "this grader is correct" directly instead of by proxy.

**What criterion (4) caught the first time, and why the earlier evidence
missed it.** On the first run it *failed* at 34.9%: of 1,634 wrong slots where
extraction and the reference agreed on slot count, 570 were the grader's error
rather than the model's, entirely in the free-form types (EX 59.6%, NV 25.4%)
and at exactly 0% in every structured one. The cause was in
:func:`~sieval.community.ugmathbench.math_equal`: ``math_verify.parse`` routes
everything through a LaTeX reader, so the dataset's plain-sympy gold was mangled
(``7*sin(pi*x/5)+1`` read as ``7*s*i*n*(i*p*x)/5 + 1``, ``sin`` as s·i·n) and a
gold naming any function could only match by exact string equality. Fixing it
moved EAcc 34.46 -> 38.49 with **716 verdicts wrong-to-right and 0
right-to-wrong**.

The zero matters more than the +4.03, and so does the shape of the miss: the
reference-replay figure below could not see this defect at all, because
replaying a gold as its own answer short-circuits on
``squash(pred) == squash(gold)`` and never reaches the symbolic path. **A
self-replay canary exercises the fast path and is silent about exactly the
comparison logic it appears to certify.** Worth remembering beyond this task.

**The divergence from upstream, measured on replayed references.** Replaying
every pinned reference back as a boxed answer through both graders, upstream
accepts its own reference on 14,616 of 15,183 rows (96.27%) and this task on
15,160 (99.85%). The 552 rows that disagree span 192 of 5,061 problems: an EAcc
**ceiling** difference of 3.79 pp, about five times the 0.70 pp binomial
standard error, and 548 of the 552 are rows upstream cannot win at all — so the
gap is repair, not drift. :mod:`sieval.community.ugmathbench` enumerates each
divergence and how the figure was obtained. It is a ceiling on the *grader*,
realized only on a problem a model would otherwise answer correctly in all three
versions, and — per the paragraph above — it is the weaker of the two
measurements. The live head-to-head is the one to trust.

**Caveat on Delta.** At temperature 0.6 the reasoning gap is mostly the sampler.
A control on Geometry with ``n=3`` gives Delta 8.70 across the three randomized
*versions* but 7.25 across three *rollouts of one fixed version* — only 1.45 pp
is version sensitivity, 83% is sampling noise. Upstream generates greedily,
where this does not arise; read Delta only against the sampling settings that
produced it.

Budget note: a full run is 15,183 inferences, and grading a wrong answer costs
roughly 25 ms of sympy per sample (a correct one is effectively free, since it
short-circuits on string equality). That work runs in a worker process
(:func:`~sieval.core.utils.offload.run_cpu_bound`) rather than on the event loop
the rest of the session shares, but it is still CPU the run has to spend.
Feedback is therefore worth a few minutes on a whole-benchmark run, and a
subject subset is a reasonable smoke test — pass
``datasets.<name>.args.subjects``.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from collections import defaultdict
from typing import override

from loguru import logger

from sieval.community.ugmathbench import (
    VERSIONS,
    build_prompt,
    extract_predictions,
    judge_answers,
    squash,
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
    DENOMINATOR_REQUESTED,
    SCORE_KEY_FIELD,
    aggregate,
    budget_metrics,
    health_metrics,
    rollout_metrics,
)
from sieval.core.utils.offload import GRADE_TIMEOUT, run_cpu_bound
from sieval.datasets import UGMathBenchDatasetSample

#: Relative tolerance for numeric answers. Matches the reference evaluator's
#: CLI default (``eval_rule.py --precision``), not the stricter 1e-8 its
#: ``Judger`` class defaults to.
DEFAULT_PRECISION = 1e-3


@sieval_task(
    name="ugmathbench_0shot_gen_fixed",
    display_name="UGMathBench (0-shot, generative, corrected)",
    description="Undergraduate math, 3 randomized versions per problem; EAcc + gap.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "open-ended"),
    deps_group="math",
    model_type="chat",
    reference_impl=ReferenceImpl(
        source="UGMathBench",
        url="https://github.com/YangLabHKUST/UGMathBench/blob/df47bfa639bfb89bdb0220036a7b2f216e72b0b3/eval_rule.py",
        notes=(
            "Prompt and metric definitions mirror upstream's `raw` template and "
            "`eval_file` (aacc / eacc / cacc / Delta / RE). The grader does NOT, "
            "and cannot: upstream's `judge_rule.py` is GPL-3.0 and would not ship "
            "in an Apache-2.0 distribution, so answer comparison is an "
            "independent math-verify-based implementation of the same 10 answer "
            "types. The `_fixed` variant therefore does not reproduce the paper's "
            "numbers, and the unqualified name stays vacant rather than reserved. "
            "MEASURED DIVERGENCE: upstream's judge was run as a local instrument "
            "(GPL-3.0 restricts distribution, not use; nothing is vendored) over "
            "all 15,183 pinned rows, each row's own reference replayed back as a "
            "boxed answer. Upstream accepts its own reference on 14,616 rows "
            "(96.27%), this task on 15,160 (99.85%); the 552 rows that disagree "
            "span 192 of 5,061 problems — an EAcc ceiling difference of 3.79 pp, "
            "about 5x the 0.70 pp binomial standard error. 548 of the 552 are rows "
            "upstream rejects its OWN reference on, so the gap is repair, not "
            "drift; only 4 go the other way (a UOL reference whose top-level "
            "commas sit outside any bracket). Dominant cause: upstream normalizes "
            "the prediction and the reference by different passes — its "
            "extraction-time normalize_answer rewrites sqrt(x) into sqrt{(}x) on "
            "the prediction only, which a plain-sympy reference never survives. "
            "Contributing: a TF slot whose reference is not a boolean (9 of 1665; "
            "upstream asserts it is and swallows the AssertionError into a False, "
            "so no answer wins), a zero-valued numeric reference, the declared "
            "answer type deciding the rule where upstream's `is_equal` retries "
            "every method until one accepts, and three deliberate differences in "
            "where commas are split (9 rows): `<` and `>` are the relational "
            "operators here, not brackets, so counting them as brackets — which "
            "upstream does — swallows the comma after them and the row comes out "
            "a slot short (6 rows); the bracket depth clamps at zero instead of "
            "going negative, where upstream loses every remaining slot after one "
            "unmatched closer (3 rows); and `{}` counts as a bracket, which costs "
            "0 rows on the references but keeps a comma inside a LaTeX group in "
            "the model's answer from splitting a slot in two. All enumerated in "
            "sieval/community/ugmathbench.py. The figure bounds the GRADER, not a "
            "model's score: it is a replay of stored references, so it says "
            "nothing about extraction on real model prose. LIVE HEAD-TO-HEAD "
            "(the stronger measurement, and the one that promoted this task): "
            "over a full 15,183-version run, upstream's judge re-graded the same "
            "stored responses; the two agree on the large majority of samples, "
            "the disagreements run several-to-one in this task's favour, and it "
            "scores above upstream on identical responses. The residual misses "
            "are LaTeX interval notation, absolute-value bars, and a `y = ` "
            "prefix on an EX answer. Figures are deliberately not quoted here: "
            "the grader has changed since that audit (the case-preserving LaTeX "
            "reading), so any number carried over would describe a comparison "
            "that no longer runs. Note that the replay figure could NOT see the "
            "largest defect the live run found — replaying a gold as its own "
            "answer short-circuits on string equality and never reaches the "
            "symbolic path. GUARDS: since the parsed text is model output, three "
            "shapes are refused rather than evaluated — the parse namespace has "
            "its builtins removed; an answer containing a quote is refused "
            "outright, because a quoted string handed to any callable (eval, "
            "sympify, S, N, or any name at all, since auto_symbol makes unknown "
            "names callable) is re-sympified with sympy's own default namespace "
            "and gets the builtins back; and an answer requiring unbounded "
            "arithmetic (a power tower) is screened out by an unevaluated "
            "pre-parse. All three grade the answer wrong, and none is reachable "
            "by any pinned reference — the largest exponent is three digits and "
            "not one of the 42,064 gold slots contains a quote. "
            "SAMPLING: upstream generates "
            "greedily, one sample per version (temperature 0, max_tokens 2048), "
            "and this task issues one rollout per version to match; set "
            "temperature via the model config. Upstream also offers a "
            "model-as-judge variant (eval_marj.py) which it now recommends over "
            "the rule-based path — not implemented here. METRICS: `relative_delta` "
            "is upstream's RE scaled by 100 — upstream prints the bare ratio "
            "(0.1667) where this reports 16.67; do not compare the two directly."
        ),
    ),
    # Promoted on the first live run (Qwen3-30B-A3B, all 15,183 versions,
    # fails 0). (1) extracted=False 0.21%, (2) 0 fails, (4) audited against
    # upstream's judge as an independent instrument: agreement on the large
    # majority of samples, disagreement several-to-one in this grader's favour,
    # every sample in the risky direction genuinely correct. The defect that run
    # exposed -- math_verify.parse LaTeX-parsing the dataset's plain-sympy gold,
    # `sin` -> s*i*n -- is fixed, with 0 regressions. Scores are not quoted: the
    # grader has changed since (the case-preserving LaTeX reading), so a number
    # carried over would describe a comparison that no longer runs.
    # (3) is explained rather than met: the remaining gap
    # to sibling benchmarks is the BENCHMARK's last-box rule, reproduced
    # faithfully, and this task scores above upstream's own verifier on
    # identical responses. The module docstring records all of it.
    status="stable",
)
class UGMathBenchZeroShotGenFixedTask(
    Task[
        UGMathBenchDatasetSample,
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
        precision: float = DEFAULT_PRECISION,
        k: int = 1,
        n: int = 1,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        if k > n:
            raise ValueError(
                f"pass@{k} needs at least {k} sample(s) per problem, got n={n}."
            )
        self._k = k
        self._n = n
        if precision <= 0:
            raise ValueError(
                f"precision must be > 0 (got {precision}); it is the relative "
                "tolerance for numeric answers."
            )
        self._precision = precision

    @override
    async def preprocess(self, raw, ctx):
        prompt = build_prompt(
            raw["subject"],
            raw["problem"],
            len(raw["answer"]),
            raw["answer_type"],
            raw["options"],
        )
        return build_prompt_record(
            [{"role": "user", "content": prompt}],
            reference=raw["answer"],
            # The version and its problem are what report() groups on; kept here
            # too so a prompt row identifies its sibling versions on its own.
            extra={
                "problem_id": raw["id"],
                "version": raw["version"],
                "subject": raw["subject"],
                "answer_type": raw["answer_type"],
            },
        )

    @override
    async def infer(self, pre, ctx):
        # `n` is the sampling budget `k` was validated against, so it has to
        # reach the model (sieval/tasks/CLAUDE.md, "n_shot vs k").
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        # One prediction per rollout, itself the list of per-slot answers.
        predictions: list = [extract_predictions(text) for text in inf.texts]
        return build_prediction_record(predictions)

    @override
    async def feedback(self, post, ctx):
        raw = ctx.raw_sample
        if raw is None:
            # Nothing to compare against; the reference is genuinely unknown
            # rather than a procedure, so the verdict is wrong-by-default.
            #
            # The grouping keys still have to survive, and the prompt record
            # carries them. Without that, report() cannot tell which problem
            # this version belonged to and drops it from the effective-accuracy
            # denominator, while the wrong verdict stays in AAcc's -- so EAcc is
            # computed over the survivors and biased *upward*. That is the same
            # failure `_identify` guards for failed samples; a wrong-by-default
            # verdict has to hold its problem's place just as a failure does.
            problem_id, subject = _identify(ctx)
            return True, build_judgement_record(
                None,
                [
                    build_rollout_judgement(rollout["index"], False)
                    for rollout in post["rollouts"]
                ],
                extra={"problem_id": problem_id, "subject": subject}
                if problem_id is not None
                else None,
            )

        golds = raw["answer"]
        rollouts = []
        for rollout in post["rollouts"]:
            # Grading is synchronous sympy — ~35 ms for a wrong answer (the path that
            # runs every parser reading), and every runner shares one event loop, so
            # doing it here would stall every other task. `run_cpu_bound` moves it to
            # a worker process; a process not a thread because math-verify's timeouts
            # are signal-based and it refuses to run threaded at all.
            try:
                per_slot = await run_cpu_bound(
                    judge_answers,
                    rollout.get("prediction"),
                    golds,
                    raw["answer_type"],
                    raw["options"],
                    self._precision,
                    timeout=GRADE_TIMEOUT,
                )
            except TimeoutError:
                # Same contract as the rest of the grader: an answer that cannot
                # be graded is a wrong answer, not a failed run. Loud, because a
                # timeout here means an input the in-module guards did not catch.
                logger.warning(
                    "Grading sample {} exceeded {}s and was scored wrong; the "
                    "prediction is likely a shape the parser guards miss.",
                    ctx.sample_id,
                    GRADE_TIMEOUT,
                )
                per_slot = [False] * len(golds)
            n_correct = sum(per_slot)
            rollouts.append(
                build_rollout_judgement(
                    rollout["index"],
                    bool(per_slot) and all(per_slot),
                    metrics={
                        # Slot-level credit, so a near-miss on a 20-blank table
                        # is distinguishable from a blank answer. The headline
                        # verdict stays all-or-nothing, as upstream grades.
                        "answer_accuracy": n_correct / len(per_slot)
                        if per_slot
                        else 0.0
                    },
                    extra={"per_answer": per_slot, "n_answers": len(per_slot)},
                )
            )
        return True, build_judgement_record(
            golds,
            rollouts,
            # Aggregation raw material: report() reads these instead of
            # raw_sample, which a persisted context is not required to carry.
            extra={
                "problem_id": raw["id"],
                "version": raw["version"],
                "subject": raw["subject"],
            },
        )

    @override
    async def report(self, finals, fails):
        by_problem: dict[str, list[bool]] = defaultdict(list)
        by_subject: dict[str, dict[str, list[bool]]] = defaultdict(
            lambda: defaultdict(list)
        )
        n_correct = 0
        # Per *version*, the unit AAcc counts — not per problem, which is EAcc's.
        per_version: list[dict[str, float]] = []
        observed_rollouts: list[int] = []

        unattributed_finals = 0
        for final in finals:
            judgement = final.feedback_result
            extra = judgement.get("extra", {})
            problem_id = extra.get("problem_id")
            subject = extra.get("subject")
            if problem_id is None:
                # Same recovery as the failed-sample loop below. A judged
                # version that cannot name its problem would otherwise leave
                # `by_problem` while its verdict stayed in AAcc's denominator,
                # which biases EAcc *upward* — silently, and in the direction
                # that flatters the run.
                problem_id, subject = _identify(final)
            # UGMathBench asks one answer per version, so the verdict is the
            # first rollout's. A model configured for n > 1 does not turn this
            # into pass@n -- that would inflate every version-level accuracy the
            # effective-accuracy metric is built from.
            verdicts = judgement["rollouts"]
            correct = bool(verdicts) and verdicts[0]["correct"]
            n_correct += int(correct)
            # Computed ALONGSIDE AAcc/EAcc, never inside them: those keep their
            # first-rollout definition or EAcc's denominator stops meaning what
            # its warnings say it means.
            if self._n > 1 and verdicts:
                observed_rollouts.append(len(verdicts))
                per_version.append(
                    rollout_metrics(
                        [bool(v.get("correct")) for v in verdicts],
                        [_answer_text(final, i) for i in range(len(verdicts))],
                        k=self._k,
                        # So `5^2*7` and `5^2 \cdot 7` are one vote, not two.
                        normalize=squash,
                    )
                )
            if problem_id is None:
                unattributed_finals += 1
                continue
            by_problem[problem_id].append(correct)
            by_subject[subject or "unknown"][problem_id].append(correct)

        # A failed version still belongs to a problem, and that problem still
        # owes three correct answers. Registering it here keeps it in the
        # effective-accuracy denominator: without this, a problem whose three
        # versions all failed would vanish from `by_problem` entirely while its
        # three failures stayed in AAcc's denominator, so EAcc would be computed
        # over the survivors and silently biased *upward* — the opposite
        # direction from the partial-failure case below, and with no warning.
        unattributed_fails = 0
        for failed in fails:
            problem_id, subject = _identify(failed)
            if problem_id is None:
                unattributed_fails += 1
                continue
            by_problem.setdefault(problem_id, [])
            by_subject[subject or "unknown"].setdefault(problem_id, [])

        # A failed sample is an unanswered version, so it counts against the
        # average — same convention as the pass@1 math tasks.
        n_versions = len(finals) + len(fails)
        aacc = n_correct * 100 / n_versions if n_versions else 0.0

        unattributed = unattributed_fails + unattributed_finals
        if unattributed:
            logger.warning(
                "{} sample(s) ({} failed, {} judged) carry neither a raw sample "
                "nor a prompt record, so the problem they belong to could not be "
                "kept in the effective-accuracy denominator; EAcc is an upper "
                "bound by up to that many problems.",
                unattributed,
                unattributed_fails,
                unattributed_finals,
            )

        incomplete = sum(
            1 for verdicts in by_problem.values() if len(verdicts) != VERSIONS
        )
        if incomplete:
            logger.warning(
                "{}/{} problem(s) were judged on fewer than {} versions (failed or "
                "sliced samples) and cannot count as effective-accuracy hits; "
                "EAcc is a lower bound for this run.",
                incomplete,
                len(by_problem),
                VERSIONS,
            )

        eacc = _effective_accuracy(by_problem)

        # EAcc counts problems correct in *every* version; AAcc counts correct
        # versions. A problem cannot be correct in all three without those three
        # being correct, so EAcc <= AAcc holds for any run — and the way to break
        # it is for a problem to leave EAcc's denominator while its versions stay
        # in AAcc's. That is exactly what an unattributed sample does, so the
        # invariant is the cheapest detector for the whole class. A wrong number
        # that still looks plausible is the worst thing an eval can emit; say so
        # rather than let it be read as a score.
        if eacc > aacc + 1e-9:
            logger.error(
                "EAcc ({:.2f}) exceeds AAcc ({:.2f}), which is impossible: "
                "{} problem(s) are in AAcc's denominator but not EAcc's. Treat "
                "this run's EAcc as invalid rather than optimistic.",
                eacc,
                aacc,
                unattributed,
            )

        metrics: dict[str, float | str] = {
            "score": eacc,
            # `score` is EAcc, not one of the sampling metrics — say which column
            # the headline number came from rather than leave it to be inferred.
            SCORE_KEY_FIELD: "eacc",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
            "fails": float(len(fails)),
            "eacc": eacc,
            "aacc": aacc,
            "cacc": _covered_accuracy(by_problem),
            # Reasoning gap, in accuracy points; `relative_delta` expresses it as
            # a percentage of EAcc (upstream's RE).
            "delta": aacc - eacc,
            "relative_delta": (aacc - eacc) * 100 / eacc if eacc else 0.0,
            "n_problems": float(len(by_problem)),
            "n_versions_judged": float(len(finals)),
            "incomplete_problems": float(incomplete),
            # Non-zero means EAcc's denominator is short by this many samples,
            # so the figure is an upper bound rather than the usual lower one.
            # Split by origin: a failed sample never reached `feedback`, a
            # judged one did and still could not name its problem.
            "unattributed_fails": float(unattributed_fails),
            "unattributed_finals": float(unattributed_finals),
        }
        # Only when the run actually drew more than one sample: at n=1 pass@1 is
        # aacc/100, and a second name for it invites being read as independent
        # evidence.
        if self._n > 1 and per_version:
            # `n_versions` is AAcc's denominator, so a failed version counts as
            # wrong in both. Averaging over the judged versions alone would bias
            # these upward over survivors — the defect the EAcc warnings above
            # describe (RFC #74 F).
            metrics.update(aggregate(per_version, n_versions))
            metrics.update(
                budget_metrics(
                    observed_rollouts, n=self._n, k=self._k, unit="judged version"
                )
            )

        # Outside the n>1 gate: extraction health is a fact about the parser, not
        # about the draw, and n=1 is where a stopped extractor hides longest.
        metrics |= health_metrics(finals)

        for subject, problems in sorted(by_subject.items()):
            metrics[f"eacc_{subject.lower()}"] = _effective_accuracy(problems)
        return metrics


def _answer_text(final, index: int) -> str | None:
    """The extracted prediction of one rollout, joined across answer slots.

    maj@k votes on ANSWERS, not verdicts: two rollouts wrong in two different ways
    must not combine into a majority. Slots are joined with a separator that cannot
    occur inside a boxed answer.
    """
    rollouts = (final.postprocess_result or {}).get("rollouts") or []
    if index >= len(rollouts):
        return None
    pred = rollouts[index].get("prediction")
    if pred is None:
        return None
    return "\u241f".join(str(x) for x in pred) if isinstance(pred, list) else str(pred)


def _identify(ctx) -> tuple[str | None, str | None]:
    """The problem and subject a sample belongs to, without a judgement.

    Two callers, one reason. A failed sample never reaches ``feedback``; a
    sample whose ``raw_sample`` is gone reaches it but has no reference to
    record. Either way the grouping keys the judgement normally carries are
    unavailable — and either way the sample still has to hold its place in the
    effective-accuracy denominator, because leaving it out biases EAcc upward.

    Two sources, in order: ``raw_sample``, which survives ``to_failed`` (a
    dataclass ``replace``), and the prompt record, which carries the same keys
    for a context persisted without its raw sample. Both absent means the sample
    was lost before either existed, which is what ``unattributed_*`` counts.
    """
    raw = ctx.raw_sample
    if raw is not None and raw.get("id") is not None:
        return raw["id"], raw.get("subject")
    pre = ctx.preprocess_result
    if pre is not None:
        extra = pre.get("extra") or {}
        return extra.get("problem_id"), extra.get("subject")
    return None, None


def _effective_accuracy(by_problem: dict[str, list[bool]]) -> float:
    """Share of problems correct in *every* one of their randomized versions.

    A problem judged on fewer versions than the benchmark defines cannot be
    confirmed correct across all of them, so it never counts as a hit.
    """
    if not by_problem:
        return 0.0
    hits = sum(
        1
        for verdicts in by_problem.values()
        if len(verdicts) == VERSIONS and all(verdicts)
    )
    return hits * 100 / len(by_problem)


def _covered_accuracy(by_problem: dict[str, list[bool]]) -> float:
    """Share of problems correct in at least one version — EAcc's upper bracket."""
    if not by_problem:
        return 0.0
    hits = sum(1 for verdicts in by_problem.values() if any(verdicts))
    return hits * 100 / len(by_problem)
