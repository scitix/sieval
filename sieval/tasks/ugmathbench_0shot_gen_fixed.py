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
figure above are for.

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
instead of by proxy.

**The live run has now happened, and (4) failed. The task stays experimental
for a real reason instead of a procedural one.** Qwen3-30B-A3B, thinking on,
temperature 0.6 / top_p 0.95 / top_k 20, one rollout per version; the full
15,183 versions over all 5,061 problems; sglang tp2xdp4 on 8xH100; 75.7M output
tokens. Headline: **EAcc 34.46, AAcc 40.87, CAcc 48.07, Delta 6.42**, with
``fails`` 0 and ``incomplete_problems`` 0.

* (1) **passes** — ``extracted=False`` on 32/15,183 (0.21%); truncation 29
  (0.19%) at ``max_tokens`` 32768, so the box-or-nothing extraction rule is not
  what is costing this model points.
* (2) **passes** — 0 failed samples.
* (3) **fired, as a diagnostic should.** 34.46 on undergraduate coursework math
  is not a plausible neighbour of the *same* model's 72.5 on AIME 2026 and 51.5
  on HMMT Feb 2026 (same harness, same sampling). Undergraduate coursework is
  not harder than AIME. The criterion caught a measurement problem, and (4)
  names it.
* (4) **fails** — of 1,634 wrong slots drawn from the bucket where extraction
  and the reference agree on slot count, **570 (34.9%) are this grader's error,
  not the model's**, confirmed by substituting the same random values into both
  sides' free symbols. They sit entirely in the free-form types (EX 59.6%,
  NV 25.4%, OE 7/10) and at exactly **0%** in every structured type — OL, UOL,
  MCS, MCM, TF, INT, EQ.

**Root cause of (4), and it is in this module's**
:func:`~sieval.community.ugmathbench.math_equal`.
``math_verify.parse`` routes every string through a LaTeX reader, so the
dataset's plain-sympy *gold* is mangled: ``7*sin(pi*x/5)+1`` parses as
``7*s*i*n*(i*p*x)/5 + 1``, reading ``sin`` as s·i·n and ``pi`` as p·i. A gold
containing any function name can then only match by exact string equality.

That is also why the 99.81% replay figure above could not see it: replaying a
reference as its own answer short-circuits on ``_squash(pred) == _squash(gold)``
and never reaches the symbolic path. **A self-replay canary validates the fast
path only** — it is silent about precisely the comparison logic it appears to
certify.

**That defect is now fixed.** :mod:`sieval.community.ugmathbench` reads the gold
with sympy's own parser as well as the LaTeX one, and falls back to equivalence
by substitution over a fixed probe ladder. Re-grading the same 15,183 stored
responses with extraction held fixed — so the delta is the grader's alone and
not a more permissive extractor's:

    EAcc 34.46 -> **38.43**, AAcc 40.87 -> 45.52, CAcc 48.07 -> 53.45;
    705 samples wrong-to-right, and **0 right-to-wrong**.

The zero matters more than the +3.97: the pass runs only after every other
strategy has said "not equal", so it is structurally incapable of breaking a
comparison that already worked, and the run confirms it empirically. For
reference, upstream's own verifier scores 35.55 on these same responses, so the
repaired grader is no longer the lenient or the strict one — it is above both
its previous self and the reference implementation.

**What still blocks "stable", and it is a maintainer's call, not a measurement.**
Two things:

1. Criterion (3) is still unmet on the raw number — 38.43 against AIME 72.5 —
   and the whole remaining gap is the format tax below, which is the
   *benchmark's* rule, faithfully reproduced. Whether a protocol-faithful score
   that a formatting convention dominates counts as "a plausible band" is a
   judgement about what this task is for, not something another run settles.
2. Criterion (4) can no longer be measured the way it was. The audit that found
   the defect used numeric substitution; the grader now uses numeric
   substitution. Auditing it with the same method would be measuring a method
   against itself. A residual audit needs an independent instrument — a manual
   sample, or upstream's judge — and a sampled eyeball of what remains wrong is
   dominated by genuine model errors, with a thin tail of prediction-side
   parse failures (a ``y = `` prefix on an ``EX`` answer, a trailing ``dx``).

**Reading the headline number.** EAcc 34.46 is protocol-faithful and comparable
to the paper's ruler, but it is not this model's mathematical ability. Two
measurement effects separate the two, both quantified off the same stored
responses with no extra inference:

* **The format tax, ~+33.5 EAcc, and it is the benchmark's, not ours.** Only the
  last ``\\boxed{}`` is kept and a slot-count mismatch scores every slot wrong.
  Qwen3 ends multi-part problems with ``\\boxed{a}, \\boxed{b}, \\boxed{c}``
  rather than ``\\boxed{a, b, c}`` and loses answers it got right. Single-answer
  rows mismatch 0.30% of the time; multi-answer rows 86.31% — same model, same
  subjects, so this is formatting, not difficulty. Upstream's ``judge_rule.py``
  has the identical rule, so this task reproduces it deliberately.
* **The (4) defect, ~+6.5 EAcc.**

Repairing both, exactly rather than by extrapolation: 34.46 -> 67.97 -> 74.43,
which does sit next to AIME 72.5 as criterion (3) expects.

**Swapping in upstream's verifier does not fix this.** Run as a local instrument
over the same 15,183 responses (GPL-3.0 restricts distribution, not use),
upstream scores EAcc 35.55 against this task's 34.46 — **+1.09**, with 94.95%
per-sample agreement and disagreement in *both* directions (upstream wins 486,
this grader wins 280, the latter being the unwinnable-slot repair documented
above). It carries the same format tax and recovers little of the equivalence
gap, while being unshippable here on licence.

**Caveat on Delta.** At temperature 0.6 the reasoning gap is mostly the sampler.
A control on Geometry with ``n=3`` gives Delta 8.70 across the three randomized
*versions* but 7.25 across three *rollouts of one fixed version* — only 1.45 pp
is version sensitivity, 83% is sampling noise. Upstream generates greedily,
where this does not arise; read Delta only against the sampling settings that
produced it.

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
    # name and the measured figure above carry. The live run has happened
    # (Qwen3-30B-A3B, all 15,183 versions, fails 0); it failed promotion
    # criterion 4, and the defect it found — `math_verify.parse` LaTeX-parsing
    # the dataset's plain-sympy gold, `sin` -> s*i*n — is now FIXED, worth
    # EAcc 34.46 -> 38.43 with 0 regressions.
    # Still experimental, for two reasons a further run cannot settle: (3) is
    # unmet on the raw number because the benchmark's own last-box rule
    # dominates it, and (4) now needs an audit instrument independent of the
    # substitution the grader itself adopted. Both are maintainer calls; the
    # module docstring lays them out.
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
