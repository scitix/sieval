"""
UGMathBench 0-shot generative task, corrected — effective accuracy and the gap.

The ``_fixed`` variant: this grades slots that upstream's judge cannot win, so
it is deliberately *not* a reproduction of the published numbers. The
unqualified name ``ugmathbench_0shot_gen`` is reserved for a faithful port and
will stay vacant — upstream's grader is GPL-3.0 and cannot ship in an
Apache-2.0 distribution, so no faithful port is possible here at all.

The divergence is measured, not asserted. Replaying every pinned reference back
as a boxed answer through both graders, upstream accepts its own reference on
14,616 of 15,183 rows (96.27%) and this task on 15,154 (99.81%). The 546 rows
that disagree span 190 of 5,061 problems: an EAcc **ceiling** difference of
3.75 pp, about five times the 0.70 pp binomial standard error, and 542 of the
546 are rows upstream cannot win at all. So the gap is large *and* it is repair
— :mod:`sieval.community.ugmathbench` enumerates each divergence and how the
figure was obtained.

Because it is a ceiling, it is realized only on a problem a model would
otherwise answer correctly in all three versions; and because it comes from
replayed references, it bounds the grader, not any model's score.

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

**Why ``status="experimental"``, and what promotes it.** Not because the grader
diverges from upstream — that is what the ``_fixed`` name and the measured
figure above are for. It is experimental for the ordinary reason ``arc_*`` and
``ifbench`` are: no served model has ever run this task. Everything known about
the grader comes from replaying stored references, which says nothing about how
extraction behaves on real model prose.

"Matches upstream" cannot be the promotion gate here — upstream's grader is
GPL-3.0, so that bar is unreachable by construction and would pin this task to
``experimental`` forever. The gate is instead evidence that *this* grader is
right, which a single live run produces:

1. the ``extracted=False`` rate is low — replay cannot test this, since it feeds
   perfectly boxed answers;
2. ``fails`` is 0 across the run, so the long inference stream held;
3. EAcc/AAcc/CAcc land in a plausible band next to sibling math benchmarks on
   the same model — *not* next to the paper, which used a different ruler;
4. a sampled audit of wrong verdicts reports a **false-negative rate** — of N
   slots graded wrong, how many the grader got wrong rather than the model.

(4) is the one that matters, and it is stronger evidence for "a correct grader"
than a reproduction table would have been, because it tests that claim directly
instead of by proxy. Record the numbers here when promoting.

Budget note: a full run is 15,183 inferences, and grading a wrong answer costs
roughly 25 ms of synchronous sympy per sample (a correct one is effectively
free, since it short-circuits on string equality). Feedback is therefore worth
a few minutes on a whole-benchmark run, and a subject subset is a reasonable
smoke test — pass ``datasets.<name>.args.subjects``.

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
            "(96.27%), this task on 15,154 (99.81%); the 546 rows that disagree "
            "span 190 of 5,061 problems — an EAcc ceiling difference of 3.75 pp, "
            "about 5x the 0.70 pp binomial standard error. 542 of the 546 are rows "
            "upstream rejects its OWN reference on, so the gap is repair, not "
            "drift; only 4 go the other way (a UOL reference whose top-level "
            "commas sit outside any bracket). Dominant cause: upstream normalizes "
            "the prediction and the reference by different passes — its "
            "extraction-time normalize_answer rewrites sqrt(x) into sqrt{(}x) on "
            "the prediction only, which a plain-sympy reference never survives. "
            "Contributing: a TF slot whose reference is not a boolean (9 of 1665; "
            "upstream asserts it is and swallows the AssertionError into a False, "
            "so no answer wins), a zero-valued numeric reference, and the declared "
            "answer type deciding the rule where upstream's `is_equal` retries "
            "every method until one accepts. All enumerated in "
            "sieval/community/ugmathbench.py. The figure bounds the GRADER, not a "
            "model's score: it is a replay of stored references, so it says "
            "nothing about extraction on real model prose — which is why status "
            "is experimental until a live run. SAMPLING: upstream generates "
            "greedily, one sample per version (temperature 0, max_tokens 2048), "
            "and this task issues one rollout per version to match; set "
            "temperature via the model config. Upstream also offers a "
            "model-as-judge variant (eval_marj.py) which it now recommends over "
            "the rule-based path — not implemented here. METRICS: `relative_delta` "
            "is upstream's RE scaled by 100 — upstream prints the bare ratio "
            "(0.1667) where this reports 16.67; do not compare the two directly."
        ),
    ),
    # Not because the grader diverges from upstream — that is what the `_fixed`
    # name and the measured figure above carry. Experimental for the ordinary
    # reason `arc_*` and `ifbench` are: no served model has run this task yet, so
    # every claim here rests on replayed references. The module docstring states
    # what a live run has to show to promote this to "stable".
    status="experimental",
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
    ):
        super().__init__(dataset=dataset, model=model, name=name)
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
        return await self.model.agenerate(pre["prompt"])

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
            return True, build_judgement_record(
                None,
                [
                    build_rollout_judgement(rollout["index"], False)
                    for rollout in post["rollouts"]
                ],
            )

        golds = raw["answer"]
        rollouts = []
        for rollout in post["rollouts"]:
            per_slot = judge_answers(
                rollout.get("prediction"),
                golds,
                raw["answer_type"],
                raw["options"],
                self._precision,
            )
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

        for final in finals:
            judgement = final.feedback_result
            extra = judgement.get("extra", {})
            problem_id = extra.get("problem_id")
            # UGMathBench asks one answer per version, so the verdict is the
            # first rollout's. A model configured for n > 1 does not turn this
            # into pass@n -- that would inflate every version-level accuracy the
            # effective-accuracy metric is built from.
            verdicts = judgement["rollouts"]
            correct = bool(verdicts) and verdicts[0]["correct"]
            n_correct += int(correct)
            if problem_id is None:
                continue
            by_problem[problem_id].append(correct)
            by_subject[extra.get("subject", "unknown")][problem_id].append(correct)

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

        if unattributed_fails:
            logger.warning(
                "{} failed sample(s) carry neither a raw sample nor a prompt "
                "record, so the problem they belong to could not be kept in the "
                "effective-accuracy denominator; EAcc is an upper bound by up to "
                "that many problems.",
                unattributed_fails,
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
        metrics: dict[str, float] = {
            "score": eacc,
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
            # Non-zero means EAcc's denominator is short by this many problems,
            # so the figure is an upper bound rather than the usual lower one.
            "unattributed_fails": float(unattributed_fails),
        }
        for subject, problems in sorted(by_subject.items()):
            metrics[f"eacc_{subject.lower()}"] = _effective_accuracy(problems)
        return metrics


def _identify(ctx) -> tuple[str | None, str | None]:
    """The problem and subject a sample belongs to, without a judgement.

    A failed sample never reaches ``feedback``, so the grouping keys the
    judgement carries are unavailable — but it still has to hold its place in
    the effective-accuracy denominator. Two sources, in order: ``raw_sample``,
    which survives ``to_failed`` (a dataclass ``replace``), and the prompt
    record, which carries the same keys for a context persisted without its raw
    sample. Both absent means the sample failed before either existed.
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
