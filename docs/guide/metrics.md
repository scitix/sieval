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
| `score_ci95` | 95% interval on `score`, as `[lo, hi]`, clustered on problems |
| `n_problems` | Distinct problems the headline was averaged over |

The first four are on every report. The interval pair is on every task that
computes one, and adoption is still task by task — so a report may carry no
interval at all, which is not the same as a zero-width one. It is never zeroed to
stand in for a missing one.

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
check fails a report that writes `score_ci95` without `n_problems`, or the
reverse.

One task is the documented exception, and only to the `score` half:
`t_eval_before_calling_0shot_gen` publishes one rate per axis and no headline, so
it emits neither `score` nor `score_key` — there is nothing for the latter to
name. It still declares `denominator_policy`.

## The interval on the headline

`score_ci95` is a 95% interval on `score`; `n_problems` is the population it was
clustered over. They arrive as a **pair or not at all** — an interval whose
population is unknown cannot be read, and a population with no interval beside it
is a count nothing asked for — and both are omitted, never zeroed, when there is
nothing to estimate: fewer than two problems, or no spread between them.

It is a Wilson interval on an effective sample size, so it stays inside 0–100 and
is asymmetric near a bound; at a `score` of exactly 0 or 100 it falls back to the
exact one-sided Clopper-Pearson limit over problems, which is where a reader most
needs a bound. Nothing in it is random, so two readers of the same run compute
the same interval, and a resumed run computes the same one as a fresh one.

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

It converges to a floor set by the problem count, not by the budget — across 12
runs that floor was 51–65% of the same run's `n = 1` width. So a wider interval
at low `n` is a property of the estimator, **not** evidence that the model is
less stable there.

Across **tasks**, widths are not comparable at all, at any `n`: `n_problems` moves
them (roughly `1/√m`) and so does the score level, which is why a 198-problem
±5.36 beside a 30-problem ±6.68 says nothing about which model is steadier. Only
the same task at the same `n` and the same problem count puts two widths on one
scale.

### The floor is `n_problems`, not `n`

A 30-problem set carries roughly ±7 pp of irreducible width. More sampling cannot
shrink it; only more problems can. That is why `n_problems` has to be read beside
the interval — the same ±7 pp is the floor on a 30-problem set and an alarm on a
500-problem one — and it is why an interval quoted without its problem count is
unreadable rather than merely incomplete.

Read it with `denominator_policy` too, because the two together are what say
*which* problems `n_problems` counted. The interval covers the same population
`score` does, so under `requested` the failed samples are in it, entering as
deterministic zeros: they pull the centre down and contribute no variance of their
own, which narrows the interval rather than widening it.

Under `judged`, whether `n_problems` holds steady depends on whether the split was
repeated, because that decides where the population is read from:

- **Not repeated** (`mmlu_0shot_gen`, `mmlu_pro_0shot_gen`,
  `openbookqa_kshot_gen`) — there is nothing to collapse, so the population is the
  declared denominator, which under `judged` is `len(finals)`. It shrinks
  one-for-one with `fails`: an MMLU run with 3 failures reports 3 fewer problems
  than the split holds.
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
intervals belongs to. UGMathBench is the case in this tree: `aacc` is a rate per
*version* and `eacc` a rate per *problem*, so an interval on one is not an
interval on the other, and each owes its own population. Reusing a problem-level
population for a version-level rate narrows the interval by the same √times an
uncollapsed repeat does, wearing a different name.

The per-category keys (`score_<category>` on MMLU and MMLU-Pro) carry **no**
interval today. Each would need its own population count, and one `n_problems`
cannot carry 57 of them.

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
`report.json` remains a function of the run that produced it.

Two migrations: the `pass@<k>` → `pass@k` rename, where a dashboard keyed on the
literal `pass@4` needs updating and should read the budget from `n` / `k`; and
`avg@k` → `avg@n`, which renames a column without changing its value.
