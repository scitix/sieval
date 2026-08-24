# Report Metrics

Every evaluation writes a `report.json` per task. This page documents its keys.

Two rules govern the whole file:

- **A key never spells the value of `k`** — the column is `pass@k`, not `pass@4`,
  so it keeps its identity when the budget changes. The budget is reported once,
  as the `n` and `k` fields.
- **A key is omitted, never zeroed, when it cannot be computed.** A `0.0` meaning
  "not measurable" is indistinguishable from one meaning "measured, and it was
  zero". If a key you expect is missing, something below says why.

## Always present

| Key | Meaning |
| --- | --- |
| `score` | The headline. What a leaderboard ranks on. |
| `score_key` | Which other key `score` was copied from — `pass@1`, `accuracy`, `exact_match`, … |
| `fails` | Samples that failed the pipeline (an error, not a wrong answer). |
| `denominator_policy` | Which population the headline is averaged over. |
| `score_ci95` | 95% interval on `score`, as `[lo, hi]` |
| `n_problems` | Declared problem population — of whichever metrics `ci95_units` puts on it, usually the headline |
| `ci95_units` | Which population count each interval in the report is clustered on |

The first four are on every report (one documented exception below).
`score_ci95` / `n_problems` are on every task whose headline is a mean over
problems, which is most of them but not all: a headline that is a rate **pooled
over constraints**, a **nonlinear ratio of aggregates**, or **absent
altogether** is not such a mean, and those reports carry no headline interval. A
headline that is a **macro-average over strata** is a mean over *those* — the
three that ship it carry `score_ci95` beside `n_subjects` / `n_subsets` rather
than `n_problems`. So a report may carry no interval at all, which is not the
same as a zero-width one. It is never zeroed to stand in for a missing
one. `ci95_units` is there whenever the report carries **any** interval, and
absent when it carries none.

`ci95_units` is the one **nested object** in `report.json`. Every other value is a
scalar or a flat list — a number, a string, an interval as `[lo, hi]`, the
variable-length `sieval_versions`, and `null` where a task omits a value in band
(`hle_0shot_gen`'s `calibration_error`). Consumers that read `score` and pass the
rest through are unaffected.

`denominator_policy` is `requested` (`finals + fails`, so a pipeline failure
counts as **wrong**) or `judged` (`finals` only, failures excluded). The split is
upstream-convention-driven, and unifying it would change `score` for eight tasks
— so it is declared rather than resolved. Two tasks are directly comparable only
when this field agrees; when `fails` is 0 the two coincide.

`score_key` and `denominator_policy` are enforced rather than conventional:
`scripts/check_preflight.py --check check_report_declarations` fails a task whose
`report()` omits either, whose policy is a word other than the two above, or
whose `score_key` names a column the report does not contain — nothing reads
`score_key` at run time, so that last one would otherwise go unnoticed. The same
check fails a report whose **source** writes `score_ci95` without `n_problems`, or
the reverse, and one that writes any `*_ci95` key without writing `ci95_units` at
all. Read that as a property of the check, not as an invariant of the format: it
is an exact match on the literal keys a source scan can see, and a headline
clustered on strata pairs with its own count instead (three reports do — see
"whole or not at all" below). Those are counted separately in the check's own PASS
line, so the two figures reconcile rather than leaving three reports unexplained.

Whether that map is *complete* is checked at run time instead, when the report is
finished: a per-metric interval key is built from a metric name, so no source scan
can enumerate the keys a report will publish. The run **saves `report.json` and
then fails** on any of five things — an interval with no `ci95_units` entry; an
entry naming a metric or a population key the report does not write; an entry
whose population is not a **count** key (`n_…`), which is how a value naming
another *metric* is caught, since that key really is in the report; and two
metrics declared on one unit that carry one interval while publishing two
different numbers, which is what a metric passed as an alias of something it is
not looks like from the outside. Saved first on purpose — the artifacts of a
finished run are worth keeping — and raised rather than logged, because an
ignorable warning is how every earlier undeclared-key defect shipped.

The count-key rule is also enforced **before** any of that, by the estimator
itself, and that ordering is the point: the population is published *under* the
key the unit names, so a unit naming a metric replaces that metric's rate with a
count. Save-then-raise is right for a declaration a reader cannot use — the file
stays inspectable — and wrong for a value that is wrong, so a report with such a
unit is never built and never reaches disk. What the run-time half still covers is
a `ci95_units` map written by hand, where no estimator sits between the
declaration and the reader.

One task is the documented exception, and only to the `score` half:
`t_eval_before_calling_0shot_gen` publishes one rate per axis and no headline, so
it emits neither `score` nor `score_key` — there is nothing for the latter to
name. It still declares `denominator_policy`.

## Intervals

### The interval on the headline

`score_ci95` is a 95% interval on `score`; `n_problems` is the **declared** problem
population it is clustered over — the denominator of the estimand, reported as it
was declared. The width itself is measured over the problems the run actually
observed, which is the same set on a clean run and a smaller one when every copy
of some problem failed: gpqa with 8 wholly-failed questions still reports
`n_problems` 198, over a width measured on the 190 that came back. Read it as the
population the number is *about*, not as a count of what was measured.

They arrive **whole or not at all** — an interval whose population is unknown
cannot be read, a population with no interval beside it is a count nothing asked
for, and an interval whose unit is undeclared cannot be told from one clustered on
something else — so an interval, the count it is clustered on, and the
`ci95_units` entry naming that count appear together. All are omitted, never
zeroed, when there is nothing to estimate: fewer than two problems, or no spread
between them.

**Read that per metric, not per report** — a report is not malformed for
breaking either half of it at the report level, and three shapes in this tree do:

- `score_ci95` with **no `n_problems`**: AGIEval, C-Eval and CMMLU cluster their
  headline on strata, so the count beside it is `n_subsets` / `n_subjects`. The
  pair is with *its own* population, and copying `n_problems` there would be the
  error, not the fix.
- `n_problems` with **no `score_ci95`**: `simpleqa_verified_0shot_gen`'s headline
  is a nonlinear F1 and gets no interval, while its three buckets do — the count
  is theirs.
- a metric with **no interval at all**, whenever its own values had no spread,
  beside a sibling that has one.

So pair `<metric>_ci95` with the count `ci95_units` names **for that metric**, and
treat every one of the three keys as independently optional.

It is a Wilson interval on an effective sample size, so it stays inside 0–100 and
is asymmetric near a bound; at a `score` of exactly 0 or 100 it falls back to the
exact one-sided Clopper-Pearson limit over problems, which is where a reader most
needs a bound. Nothing in it is random, so two readers of the same run compute
the same interval, and a resumed run computes the same one as a fresh one.

### One interval per metric

`score_ci95` is not the only interval a report can carry. A metric that can carry
one carries **its own**, under its own key: `<metric>_ci95`, so `pass@k` gets
`pass@k_ci95`. `score_ci95` is that same rule applied to `score`.

Each of those needs to say which population it was clustered on, and one
`n_problems` cannot answer for several metrics at once — a rate per problem and a
rate per graded turn are not the same estimand. So the report declares it, one
entry per metric:

```json
{
  "pass@1": 41.2, "pass@1_ci95": [36.1, 46.5],
  "pass@k": 63.8, "pass@k_ci95": [58.2, 69.1],
  "n_problems": 500,
  "ci95_units": {"pass@1": "n_problems", "pass@k": "n_problems"}
}
```

One population key per **unit**, not per metric: metrics sharing a unit share the
count. The value of every `ci95_units` entry is a **population count** the same
report writes — always an `n_…` key — so a reader that finds an interval can
always find the size of the population it is quoted over. It never points at
another metric: two names for one number are handled where the interval is
computed, not by an entry redirecting one key to another.

**Which metrics have one today is partial, and this is where it stands.** The
headline of 52 of the 58 task reports — 49 of them over `n_problems`, and
AGIEval, C-Eval and CMMLU over the strata their macro averages
(`n_subsets` / `n_subjects`); the column `score_key` names it was copied
from, under that column's own name; every sampling key the shared block publishes
— `pass@1`, `avg@n`, `pass@k`, `pass^k`, `maj@k`, `self_consistency` — on the 28
tasks that route through it, plus UGMathBench's copies of those six on its
*version* axis; and the co-equal second rate of the tasks that
publish one on the problem unit (HellaSwag's `acc`, DROP's `em`, the second
extraction rule on each of the GSM8K and GSM1k base tasks —
`flexible_exact_match` beside GSM8K's `exact_match`, `strict_exact_match` beside
GSM1k's `flexible_exact_match` — IFEval's and IFBench's other
grade, UGMathBench's `cacc`, AdvancedIF's `macro_pass_rate`,
ComplexConstraints' `criterion_pass_rate_macro`, SciTaRC's `exact_match` and
`partial`, SimpleQA-Verified's `correct` / `incorrect` / `not_attempted`).

Beyond the problem unit — and beyond the `n_subjects` / `n_subsets` above — six
more populations carry one: SysBench's per-turn rates (`csr_macro`, `isr`,
`ungradeable_rate`) over `n_turns` and its `ssr` over `n_sessions`; every
`t_eval_before_calling` axis over `n_graded`, with the `*_parsed` triple over
`n_parsed`; UGMathBench's `aacc` and its six sampling keys over `n_versions`; and
GSM+'s `score_wo_critical_thinking` over `n_problems_wo_critical_thinking`.

Not yet: the pooled-ratio and nonlinear families below, every
`score_<category>` / per-subject breakdown key, and the stratum macros of
`ruler` / `iheval` — held deliberately, not for want of a count, because at 5
context lengths or 9 subtasks the interval would describe between-stratum spread
over a handful of terms.

So **`<metric>_ci95` is optional even when `<metric>` is there**, and a consumer
has to treat it that way. A metric can be published with no interval for two
reasons: no estimator has been wired to it yet (every metric outside the groups
above), or there was nothing to estimate on this run — fewer than two units,
or no dispersion between them, which is when `wilson_interval` returns nothing
rather than a zero-width interval claiming certainty the run does not have. A run
where every sample failed publishes all six sampling metrics and no interval at
all. The implication only runs one way: an interval is **never** present for a
metric that is absent.

Three rules decide whether a metric is a candidate at all:

- It has to be **exactly** a mean over some unit of a per-unit value. A rate
  **pooled over two sums** (SysBench's `csr`, the IFEval-family
  `*_instruction_level_accuracy`) and a **nonlinear combination of aggregates**
  (SimpleQA-Verified's `f1`, HLE's `calibration_error`, UGMathBench's `delta`) are
  neither, and get no interval rather than an estimator that does not fit them.
- A **deterministic transform** of another metric gets none. Its bounds would be
  the other metric's mirrored, which reads as two independent measurements of one
  thing. `aa_lcr_0shot_gen`'s `incorrect` is literally `1 - accuracy`, so it gets
  none. `browsecomp_0shot_gen`'s is written as an independent bucket count and
  still gets none, because `parse_grade` returns `CORRECT` or `INCORRECT` and
  nothing else and a pipeline failure stands in as `INCORRECT` — so the two rates
  sum to 100 on every input that task can produce. It becomes a third estimand
  only if a `NOT_ATTEMPTED` bucket arrives, which is the difference from
  `simpleqa_verified_0shot_gen`, where three buckets are real.
- A true **alias** — the same number under a second key name — does get one, so a
  consumer keyed on either name finds a companion. That covers `score_ci95` and
  `<column>_ci95` on every task whose `score` is a copy of another column
  (`accuracy`, `acc_norm`, `exact_match`, `pass@1`, …), and IFEval's
  `strict_accuracy` beside `strict_prompt_level_accuracy`. There is **one
  estimate, published under every name**: the emitter takes the alias names as a
  parameter, so the bounds are computed once rather than once per key and cannot
  drift apart. Each key still carries its own list, so they are equal values and
  not a shared object — nothing that reads the report back and edits one bound
  changes another. The finished report is checked against this: two metrics on one
  unit sharing one interval have to publish the same number, within the half a
  hundredth a rate rounded to 2 dp can sit from the unrounded mean the bounds
  bracket. Two metrics on one unit have equal bounds only when they have equal
  `p`, so that check is a tautology for a real alias and fires on a metric handed
  to the parameter that is not one — **with one blind spot**: it runs on the
  finished report, so a wrongly-aliased metric that *also* has its own
  `metric_interval` call later in the same fold is overwritten by that call before
  the check sees it. What it catches is the case that actually publishes a
  mirrored bound: a sibling with no estimate of its own.

A withheld metric takes its interval with it: a task that withholds the sampling
block at `n = 1` withholds those metrics' intervals too, and keeps the ones for
what it still publishes. The converse does not hold — a published metric may have
no interval, per the reading above.

`hle_0shot_gen`'s `confidence_interval` is not part of this and is not listed in
`ci95_units` — it is upstream's own pooled half-width, described below.

### What it estimates — and the two things it does not

Three different questions get called "the confidence interval". This ships
exactly one of them:

| Question | Estimand | Shipped |
| --- | --- | --- |
| Would this number hold on another problem set from the same distribution? | between-problem variance | **yes — this is `score_ci95`** |
| Would a re-run of this exact config move the score? | between-invocation variance | no |
| Is run A different from run B on the same problems? | paired per-problem delta | no |

Re-run variance is absent because **no single run can estimate it**, and that was
measured rather than assumed. The candidate was the closed-form rollout standard
error — the within-run quantity that would have had to answer it. It is exactly
calibrated *within* a run: split-half over 12 real runs gives a mean z² of 1.05,
against 1.00 for a perfectly calibrated interval. Across runs it is **2.0× too
narrow**: over 6 identical-config replicate pairs the mean z² is 4.18, and 4 of
those 6 re-runs landed outside their own nominal 95% interval. An estimator that
is calibrated inside a run and half the width it should be between runs cannot be
relabelled as a re-run interval, so `score_ci95` does not claim to be one.
Run-to-run drift within roughly 3 pp is treated as acceptable and is not pursued
— but read that as a working tolerance, not a bound: those 6 replicate pairs
averaged −1.70 pp and the largest was −3.54 pp, so a gate set at 3 pp will fire
on a genuine identical-config re-run.

### Width is not comparable across budgets

At `n = 1` the interval necessarily carries rollout noise as well as
between-problem noise: one draw per problem cannot separate them. It sheds that
as `n` rises. Measured on one real 30-problem run:

| `n` | half-width |
| --- | --- |
| 1 | ±11.65 |
| 4 | ±8.51 |
| 8 | ±7.36 |
| 64 | ±6.68 |

It flattens out as `n` rises, at a level set by the problem count *and* by how much
those problems disagreed — not by the budget. On the 12 runs this was measured on,
the high-`n` width came in at 51–65% of the same run's `n = 1` width. That range
describes those 12 runs; it is not a ratio the estimator enforces. Either way, a
wider interval at low `n` is a property of the estimator, **not** evidence that the
model is less stable there.

Across **tasks**, widths are not comparable at all, at any `n`: `n_problems` moves
them (roughly `1/√m`) and so does the score level, which is why a 198-problem
±5.36 beside a 30-problem ±6.68 says nothing about which model is steadier. Only
the same task at the same `n` and the same problem count puts two widths on one
scale.

### Width is set by the problem count *and* the spread between problems

The half-width *scales with* `z·√Var` — Wilson on `m_eff = p(1−p)/Var` rather than
that Wald form — with `Var = m·s²/D²` over `m` problems, denominator `D` and
between-problem spread `s²` — and **nothing puts a floor under `s²`**. The
problem count decides how much a given spread is worth; it does not on its own
bound the width from below. Two cases on 30 problems:

- Near-saturated — 29 problems at 1.0 and one at 1 − 1/64 — reports **±0.139 pp**.
- A flat 0.5 with one problem at 0.25 reports **±1.605 pp**. The widest 30-problem
  set at that score level is the `n = 1` boolean one, 15 right and 15 wrong, at
  **±16.85 pp** — so this is under a tenth of it, on the same problem count.

So the ±6.68 pp the table lands on is what that run's 30 problems disagreed by, not
a bound 30 problems cannot beat. As between-problem dispersion vanishes the
interval narrows without limit, which makes the reading rule the opposite of
reassuring: **a very narrow interval on a small set says the problems behaved
alike, not that the set is large.** For size, read `n_problems`; the width alone
never reports it.

That is exactly why the two have to be read together — the same ±6.68 pp means one
thing on a 30-problem set and another on a 500-problem one — and why an interval
quoted without its problem count is unreadable rather than merely incomplete.

Read it with `denominator_policy` too, because the two together are what say
*which* problems `n_problems` counted. The interval covers the same population
`score` does, so under `requested` the failed samples are in it, entering as
deterministic zeros: they pull the centre down and contribute no variance of their
own, which narrows the interval rather than widening it.

Under `judged`, whether `n_problems` holds steady depends on whether the split was
repeated, because that decides where the population is read from:

- **Not repeated** (`mmlu_0shot_gen`, `mmlu_pro_0shot_gen`,
  `openbookqa_kshot_gen`, as configured by default) — there is nothing to collapse,
  so the population is the declared denominator, which under `judged` is
  `len(finals)`. It shrinks one-for-one with `fails`: an MMLU run with 3 failures
  reports 3 fewer problems than the split holds. This is a property of the run, not
  of the task: `repeat` is a dataset transform any config can ask for, and one of
  these configured with it takes the branch below instead, at which point
  `n_problems` stops tracking `fails`.
- **Repeated** (`gpqa_diamond_0shot_gen`, at its default `n_repeats=4`) — the
  population is the count of distinct problems in the **whole split**, so it does
  not move with run health at all. A question whose every copy failed keeps its
  slot, and `n_problems` is 198 however the run went.

So `n_problems` is not comparable across `denominator_policy` values, and on an
unrepeated split it is not comparable across two runs of the same task with
different `fails` either. Read `fails` beside it.

### Copies of one problem are one problem

`n_problems` is **not** the sample count whenever a task repeats its split.
`gpqa_diamond_0shot_gen` evaluates every question four times with different answer
orderings (simple-evals' `n_repeats`), so 198 questions arrive as 792 samples.
Those four are four draws on one problem: the interval collapses them and
`n_problems` counts questions. Reading 792 independent problems there would narrow
the interval by up to √4 — measured on a real 198-question run (792 samples), by
1.54×: ±3.48 pp reported where ±5.36 pp is true, with no other key in the report
to disagree. Short of √4 because the copies are only partly correlated: gpqa
permutes the answer choices per copy, which leaves an effective 334 problems
rather than 198. Collapsing leaves `score` bit-for-bit unchanged; only the width
moves.

**Nor is it the rollout count.** Some headlines are averaged over rollouts rather
than over samples — `hle_0shot_gen`, `scitarc_0shot_gen`, `aa_lcr_0shot_gen`,
`browsecomp_0shot_gen`, `complex_constraints_0shot_gen`,
`advanced_if_0shot_gen`, whose denominator is `(finals + fails) × n`. Their
`denominator_policy` describes that population, but `n_problems` still counts
**problems**: at `n = 4` a 500-question set reports 500, not 2000. One field
cannot carry two nouns and still be comparable against a task that reports
questions, so the interval is clustered on the problem there too. This does not
move the interval — the problem count cancels out of both the centre and the
variance — it only makes the reported population mean the same thing everywhere.

### Never infer a paired comparison from two per-run intervals

Two runs' intervals overlapping is not evidence that the runs agree. On one real
pair of runs the per-run intervals read ±10.5 pp and ±11.7 pp and overlap
heavily, while the **paired per-problem delta** between those very same two runs
reads ±1.6–2.2 pp and separates them cleanly. The two answer different questions:
a per-run interval asks whether the number would hold on another problem set,
where a paired delta asks whether these two runs differ on the problems both
scored. Overlapping bars routinely hide a real, consistent regression, so do not
read them as a null result. No report key carries the paired delta — it needs both
runs, per problem, and lives outside a single run's `report.json`.

### An interval names the axis it is clustered on

A task publishing rates on two different axes must say which axis each of its
intervals belongs to, which is what `ci95_units` records. Reusing a problem-level
population for a rate on another unit narrows the interval by the same √times an
uncollapsed repeat does, wearing a different name. Four reports are two-unit
today:

- **UGMathBench.** `eacc` is a rate per *problem* and `aacc` a rate per
  *version*, so an interval on one is not an interval on the other. Its
  `score_ci95` is EAcc's, declared over `n_problems`, and `cacc` is the other
  per-problem rate over the same count; `aacc` and this task's copies of the six
  sampling keys are per version and declare `n_versions` — the requested versions,
  failures included, which is what AAcc divides by and which the older
  `n_versions_judged` does not count. It is the only task in the tree whose
  sampling block does not sit on problems.
- **SysBench.** `csr_macro`, `isr` and `ungradeable_rate` are means over turns and
  declare `n_turns`; `ssr` is a mean over sessions and declares `n_sessions`.
- **`t_eval_before_calling`.** Every axis is a mean over judged samples
  (`n_graded`); the `*_parsed` triple narrows to the samples whose reply parsed
  (`n_parsed`). Those two differ exactly when a reply does not parse, which is why
  this report has no single default and declares per metric.
- **GSM+.** The headline is over `n_problems`; `score_wo_critical_thinking` is a
  co-headline over its own subset and declares
  `n_problems_wo_critical_thinking`.

The per-category keys (`score_<category>` on MMLU and MMLU-Pro) carry **no**
interval today. Each would need its own population count, and one `n_problems`
cannot carry 57 of them.

#### The non-problem units are not collapsed under `repeat`

`Task.problem_groups` collapses the copies `Dataset.repeat` makes, and every
interval over `n_problems` goes through it. **The units above do not.** `repeat`
is a transform any config can ask for, so a run repeating one of these splits
quotes those intervals over samples rather than problems, narrowing them by up to
√times — the defect "Copies of one problem are one problem" describes for the
headline. Latent, not active: no config here repeats these splits, and the one
repeated by default (`gpqa_diamond_0shot_gen`) clusters on problems and collapses
correctly.

Each case is a schema or semantics decision rather than wiring, which is why it
is recorded and not fixed:

- **`t_eval_before_calling`** — `n_graded` / `n_parsed` count *samples* and are
  published metrics. A collapsed interval must declare a *problem* count, so it
  needs new keys: writing one under `n_graded` would move a published number
  (12 → 3 on a 3×4 run).
- **SysBench** — blocked before the interval. `session_id` comes from the data, so
  two copies of a session fuse into one `prefixes` entry while `turns` keeps both:
  `ssr` already collapses by accident, the per-turn rates double-count.
- **UGMathBench** — `problem_groups` returns `None` deliberately (EAcc's
  per-problem reduction is nonlinear). The *version* axis needs its own grouping.

### `hle_0shot_gen`'s `confidence_interval` is a different estimator

It is a Wald half-width pooled over **attempts** — `(finals + fails) × n` of them
— and therefore *unclustered*: the rollouts of one problem enter as independent
trials, which understates the width, and understates it more the more the model
varies per problem. It is kept because it is HLE's published convention, and it
sits beside `score_ci95` exactly as `accuracy` sits beside `pass@1`: two answers
to two different questions, not two candidates for one. When the question is
whether the number would hold on another problem set, read the clustered one.

## Sampling metrics

These appear **only when the task ran with `n > 1`**. At `n = 1` there is no draw
to describe. Set the budget as a **task** argument:

```yaml
tasks:
  aime_2025_0shot_gen:
    args:
      n: 4   # sampling budget: rollouts drawn per problem
      k: 4   # the k in pass@k -- the metric's parameter. k <= n is enforced.
```

`n` on the *model* config is overridden call-time by every task that takes its
own budget, and `k > n` is rejected at construction rather than silently
reporting a `pass@k` of `0.0`.

| Key | Meaning | Appears when |
| --- | --- | --- |
| `pass@1` | Unbiased single-draw success rate, `c/n`. **Not** the first rollout's verdict. | always |
| `avg@n` | Mean verdict over the whole draw. | always |
| `pass@k` | Solved **at least once** in `k` draws. | `k > 1` |
| `pass^k` | **All** `k` draws correct. | `k > 1` |
| `maj@k` | Is the modal **answer** correct? | see below |
| `self_consistency` | Share of the draw that agreed on that modal answer. | the task votes on answers |
| `n`, `k` | The budget the numbers above were measured at. | always |
| `n_short` | Samples that came back with fewer than `n` rollouts. | always |

Rates are percentages (0–100) over the task's declared denominator; `n`, `k` and
`n_short` are counts.

Each of the six rates carries **its own** 95% interval — `pass@1_ci95`,
`avg@n_ci95`, `pass@k_ci95`, `pass^k_ci95`, `maj@k_ci95`,
`self_consistency_ci95` — clustered on problems and declared over the block's one
`n_problems` (UGMathBench is the exception: it builds these per *version*, so its
copies declare `n_versions`). All six are exact means over their unit of a
per-unit value, so
none of them borrows another's: `pass@k` is not a rescaled `pass@1`, and one
interval for a block of six would make five of them read as measured when only one
was. None is ever there for a column that is not — each is derived from its
metric's own key — but any of them can be **absent while its metric is present**,
when that metric's per-problem values had no spread: an all-identical draw
publishes `self_consistency` and no `self_consistency_ci95`, and a run where every
sample failed publishes all six rates and no interval at all. On a task whose
headline is `pass@1`, `score_ci95` and `pass@1_ci95` hold the same two bounds —
one estimate, published under the name a leaderboard reads and under the name of
the column it came from.

**`avg@n` is spelled `@n`, not `@k`, on purpose:** it takes no `k` and does not
move with one — at `n=4, k=2` it averages four verdicts where `pass@k` estimates
over two. It coincides with `pass@1` on every boolean draw, and both are still
reported: `pass@1` *estimates* a single draw's success rate, `avg@n` *measures*
the draw that was paid for. They separate once a verdict stops being a bool.

`n_unextracted` — **rollouts** whose answer could not be recovered — is reported
at **every** budget, `n = 1` included, and by the single-draw tasks below that
have no `n` at all. It measures the parser, not the draw, and `n = 1` is where a
silently-stopped extractor survives longest: no second rollout to disagree with.

`n_truncated` — **rollouts** whose generation stopped because it ran out of
tokens rather than because the model was done — and `n_scored_rollouts`, the
rollouts scored in total, are reported by **every `gen` task**, on the same terms
as `n_unextracted`: they describe the generation rather than the draw, so neither
is confined to `n > 1`. Unlike every other key here the *runner* injects them, so
a task cannot report zero truncation by having forgotten to look.

They arrive as a **pair or not at all**. The count alone says nothing about how
much of a score it explains — `26` is a different fact at 600 rollouts than at 30
— and the rule lanes that most need it publish rates plus `fails` and no sample
total, so nothing here was divisible into it. `n_scored_rollouts` is the
**observed** draw, not `n × samples`: a short sample drew fewer rollouts than its
budget asked for. Deriving the rate, and deciding what threshold should warn,
fail or annotate a score, is left to the reader.

Both are omitted rather than zeroed in two cases: outside `gen`, where `ppl`/`clp`
infer at `max_tokens=1` and finish every call the way a truncation does, so the
count would equal the total and mean nothing; and when the reasons were never
recorded, since resuming a run written with `record_meta=False` hydrates its
finals without per-stage metadata and a `0` would claim a clean run rather than an
unmeasured one. `sieval_versions` reports that state in band as `"unknown"`; a
count has no such value.

Read `n_truncated` as a **bound on how much of a score is budget, not
capability** — a truncated rollout is scored wrong whether or not the model was
on its way to the right answer, and the fix is `max_tokens`. It is independent of
`n_unextracted`: an answer can be cut short and still parse (the truncation lands
after the boxed answer), or parse cleanly and be wrong for reasons of its own.
The two are worth reading together only in the direction where they coincide — a
rollout that is both truncated *and* unextracted is the case where raising the
budget is most likely to move the score.

### Pairs that must be read together

**`pass@k` with `pass^k`.** `pass@k` is an upper bound, so a model whose sampling
variance *grew* can score **higher** on it while being worse to ship. `pass^k` is
the opposite direction and falls when that happens:

| Draw (4 rollouts) | `pass@1` | `pass@2` | `pass^2` |
| --- | --- | --- | --- |
| 4 correct | 100 | 100 | 100 |
| 2 correct | 50 | 83.3 | 16.7 |

Same benchmark, halved reliability: `pass@k` fell 17 points, `pass^k` fell 83.

**`maj@k` with `self_consistency`.** `maj@k` is thresholded — 4/4 and 3/4
agreeing both score 1.0. `self_consistency` is continuous (`75` for the second),
so it alone moves when an answer distribution **widens without its mean
changing**. It is **correctness-blind** — a consistently *wrong* model scores 100
— so read it beside a correctness key, never instead of one.

**`self_consistency` with `n_unextracted`.** The denominator is the whole draw,
so an unparseable rollout drags `self_consistency` down. Low `self_consistency`
with `n_unextracted` near zero is the model; with a high one it is the parser.

**`n_short` with everything.** A short sample scores 0 for `pass@k` and `pass^k`
— the `n < k` guard — so a non-zero `n_short` makes **those two** a lower bound.
The rest are not floored, only noisier: `pass@1`, `avg@n`, `maj@k` and
`self_consistency` are computed over what arrived, so a short draw moves them in
either direction.

### Why `maj@k` can be missing

It needs both:

1. **The task votes on answers.** The code family (HumanEval, MBPP,
   LiveCodeBench) does not — two correct programs are not one answer.
2. **`k == n`.** A majority over a *sub-sample* of the budget has no definition:
   at `k=2, n=4` there is no answer to "which two", and picking would need a
   seed, which in the metric layer is a new source of irreproducibility.

Both are properties of the **configuration**, so `maj@k` is present or absent for
a whole run by construction — it does not appear and disappear with run health.

A draw that came back **short still votes**. The arrived count is run health, and
every other key treats that the same way: compute it, annotate it with `n_short`.
`self_consistency` clusters the very same answers with the very same normalizer,
so withholding one and not the other would be two answers to whether a single
draw is fit to cluster. A short draw does move `maj@k`; read it against
`n_short`.

`maj@k` is a **lower bound**, not an estimate: votes cluster on *strings* (after
the task's own gold canonicalizer) while the grader compares *symbolically*, so
`\dfrac{1}{2}` and `1/2` become one vote but `0.5` and `\frac{1}{2}` still split.
A real majority can be missed; a false one cannot be manufactured.
`self_consistency` uses the same clustering.

## Single-draw tasks

The MCQ tasks (`gpqa_diamond_0shot_gen`, `mmlu_0shot_gen`, `mmlu_pro_0shot_gen`,
`openbookqa_kshot_gen`) and the accuracy-headline math tasks (`gsm8k_*`,
`hendrycks_math_kshot_base_gen`, `theoremqa_kshot_base_gen`, PlatinumBench)
publish a **greedy single-draw** number upstream, so their headline keeps its
first-rollout definition even under `n > 1`, with `pass@1` beside it rather than
merged in. At `n = 1` the two coincide, which is what makes adopting a budget
non-breaking for a stored score.

The MCQ four take **no** `n`/`k` argument but still grade and record every
rollout that arrives (a model-level `n` reaches them); `report()` warns once with
the count it did not score. Having no budget of their own, the remedy that
warning names is the **model** config — there is no task-side `n` to move it to.

## Task-specific keys

A task may also emit: `accuracy`, `errors` (PlatinumBench's upstream headline
unit), `timeouts` (the code family), `exact_match` / `flexible_exact_match`
(GSM8K few-shot, lm-eval's two co-equal metrics), `empty` (TheoremQA),
`score_<category>` (MMLU, MMLU-Pro), and UGMathBench's `eacc` / `aacc` / `cacc` /
`delta`. `score_key` always names which of them the headline came from.

## Reading a report across versions

Adding a key is non-destructive: `sieval leaderboard report` reads `score` and
nothing else, so older runs stay readable and comparable on the headline. They
are **not** backfilled — metrics are computed inline at report time, so a stored
`report.json` remains a function of the run that produced it. The per-metric
interval keys and `ci95_units` are additions of this kind: a report written before
them has neither, and no number in it moved when they arrived.

Two migrations: the `pass@<k>` → `pass@k` rename, where a dashboard keyed on the
literal `pass@4` needs updating and should read the budget from `n` / `k`; and
`avg@k` → `avg@n`, which renames a column without changing its value.
