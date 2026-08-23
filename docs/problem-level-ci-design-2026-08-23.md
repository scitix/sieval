# Problem-level confidence interval on the headline

Design for the first slice of RFC #94 (scitix/sieval): a deterministic, closed-form
95% interval on `score`, reported at every sampling budget, clustered at the
**problem** rather than the sample.

Status: design, awaiting review. Not implemented.
Date: 2026-08-23.

## 1. Scope

In scope:

- One estimator in `sieval/core/tasks/metrics.py`, computed at the seam
  `aggregate()` already occupies.
- Clustering of dataset-`repeat` pseudo-duplicates, stamped at `repeat()` time
  (`sieval/core/datasets/dataset.py`) and read back via a new `Task.problem_groups`
  (`sieval/core/tasks/task.py`).
- Two new report keys: `score_ci95` and `n_problems`.
- One argument added to each of the 24 `report()` methods that call
  `sampling_report` (§4).
- One `problem_groups` override in `ugmathbench_0shot_gen_fixed`, whose clustering
  is by `problem_id` rather than by a repeat column (§4).
- `docs/guide/metrics.md`, `scripts/check_preflight.py`, tests.

Out of scope, deliberately:

- **The rollout-only interval.** Measured (§7) as 2.0x too narrow against real
  re-runs, so it cannot carry the reading people would give it. Defer with the
  paired delta, which is what actually answers "did this run move".
- **The paired delta** (RFC #94 §D). Reuses this estimator on per-problem
  differences; a separate change with its own verb.
- **Per-subject intervals** (`score_<category>_ci95`) — the *first* follow-up.
  Same metric, partitioned population, and the more valuable of the two
  extensions below: eight tasks publish `score_<category>` (`agieval`,
  `gsm_plus`, `iheval`, `mmlu`, `mmlu_kshot_clp`, `mmlu_pro`, `mmmlu_kshot_clp`,
  `ruler`) and models get ranked on those tables. Width scales as `1/√m`, so a
  57-way MMLU split makes each cell roughly `√57 ≈ 7.5x` wider than the headline
  — arithmetic, not a measurement; no run in `outputs/` carries these keys, so no
  width is quoted. Deferred only because those eight compute their categories by
  hand rather than through `sampling_report`, making it real work rather than a
  second call. The estimator is reusable unchanged.
- **Per-key intervals** (`pass@k_ci95`, `pass^k_ci95`, `maj@k_ci95`, …) — the
  second follow-up, and not the same thing as the above: same population,
  *different metric per key*. `pass@k` and `pass^k` have different dispersions
  across problems even though both come off the same draws, so each needs its own
  interval. RFC #94 §C proposes these; mechanically it is one estimator call per
  key once the estimator exists, so the only cost is key-count growth in
  `report.json`. Ship the headline first and extend on demand.
- **Retrofitting stored runs.** Computed inline at report time, matching #74 G.
- **A pass/fail gate, and any anomaly rule.** A rule rotates `rules_hash` and
  marks every stored `anomalies.json` stale fleet-wide.

## 2. What this interval is, and what it is not

Three different questions get called "the CI". This ships exactly one:

| Question | Estimand | This design |
| --- | --- | --- |
| Would this number hold on another problem set from the same distribution? | between-problem variance | **yes** |
| Would a re-run of this exact config move the score? | between-invocation variance | no — see §7 |
| Is run A different from run B on the same problems? | paired per-problem delta | no — follow-up |

At `n = 1` the first is not separable from rollout noise: a single draw per problem
gives `x_i ∈ {0,1}`, so the interval necessarily carries both. That is the honest
total for a single run and is what a paper reports, but it means the interval
**narrows with `n`** — measured on real runs at `m = 30`, `n = 1 → 64` shrinks it
from ±11.65 to ±6.68 pp, most of it by `n = 8` (±7.36). It converges to a limit set
by the problem count and by `σ²_between`, not by the budget:

```text
s²(n) = σ²_between + E[p(1-p)] / n
```

Read that limit off the formula and not off the measured runs: it is non-zero only
while `σ²_between` is, so a set whose problems all behave alike — saturated, or
uniformly hard — converges toward zero width at any `m`. The problem count sets
what a given `σ²_between` is worth; it puts no floor under the width by itself.

Consequence for the guide: **two runs at different `n` have differently-wide
intervals for estimator reasons, not model-stability reasons.** Width is not
comparable across budgets.

## 3. Estimator

Wilson on an effective sample size, `z = 1.96`:

```text
p̂     = Σ xᵢ / D            # m per-problem values; D = the declared denominator
Var̂   = m · s² / D²         # s² = variance of the per-problem values
m_eff = p̂(1 - p̂) / Var̂
centre = (p̂ + z²/2m_eff) / (1 + z²/m_eff)
half   = z/(1 + z²/m_eff) · √( p̂(1-p̂)/m_eff + z²/(4 m_eff²) )
ci     = [max(0, centre - half), min(1, centre + half)]
```

Two divisors that must not be conflated:

- `D` vs `m`. Under `denominator_policy: judged` they coincide and `Var̂ = s²/m`.
  Under `requested` the `D - m` failed samples are **fixed zeros** carrying no
  variance (§6), so the estimator is `Σ xᵢ / D` over `m` random terms and its
  variance is `m·s²/D²` — smaller than `s²/m`, while `p̂` is pulled down by the
  same zeros. Writing `s²/m` here would overstate the width on any run with
  failures.
- `s²` uses the **population** divisor `m`, not `m - 1`. With the sample divisor,
  `m_eff = m - 1` on boolean input and the reduction to plain Wilson is off by
  0.34 pp at `1/30`; with the population divisor it is exact.

Three properties this buys:

- **Bounded.** A Wald half-width puts the lower bound below zero exactly where
  saturated and very hard sets live: a real `aime_2025` run at 1/30 gives
  `3.33 ± 6.42 → [-3.09, 9.75]`. Wilson gives `[0.59, 16.67]`.
- **Degenerate-correct.** With `xᵢ ∈ {0,1}` and no clustering this reduces to the
  plain Wilson interval — verified on real data to 0.004 pp (`math_500`, 391/500)
  and 0.017 pp (`gpqa`, 101/198), with the population divisor making it exact.
- **Deterministic.** No resampling, no seed. RFC #94 §F's three-way choice does
  not arise: the aggregation across problems is a plain mean, so the CLT applies
  to the per-problem values directly. §A's premise — that `pass@k`'s
  nonlinearity in `c` forces a bootstrap — is about nonlinearity *within* a
  problem and does not reach the aggregation.

`m_eff` is a variance-matching device, not a count, and can exceed `m` when the
per-problem values are less dispersed than Bernoulli (measured: 334 for `gpqa`'s
198 collapsed problems). It is **not** reported.

`n_problems` reports the **declared** problem population — the denominator of the
estimand, passed in by the task and reported as given. It is not `m`: `m` is the
count of groups actually observed, and it is `m` that sets the width. The two
coincide on a clean run and diverge when every copy of some problem failed, which
drops that problem out of `m` while `n_problems` keeps its slot. The declared count
is also inert in the arithmetic — it scales the units by `G/D` and divides by `G`,
so it cancels out of both `p` and the variance.

## 4. Clustering repeats and pseudo-repeats

`Dataset.repeat(times)` concatenates copies of a split. Treating the copies as
independent problems inflates `m` by `times` and narrows the interval by `√times`.
Measured on a real `gpqa_diamond` run (198 questions × 4 copies): ±3.48 pp as 792
independent, **±5.36 pp** correctly collapsed to 198 — 54% wider.

**Nothing downstream can reconstruct the grouping.** Two candidate keys both fail:

- *Row content.* `gpqa_diamond_0shot_gen` draws one choice-permutation per
  `sample_id` (`self._permutations[ctx.sample_id]`, faithful to simple-evals), so
  the four copies of one question carry four different prompts by design. A
  content hash splits them.
- *Position.* Copy-major layout makes `sample_id mod n_rows` correct only while
  rows stay in emission order; `shuffle` permutes them and the arithmetic then
  returns a confident wrong answer. PR #114 rejected this for `repeat_index` for
  exactly this reason.

So the group identity is stamped where it is known, next to the existing
`repeat_index`:

```python
REPEAT_GROUP_COLUMN = "repeat_group"        # 0-based ORIGINAL row index

new_dict[split] = (
    original.repeat(times)
    .add_column(REPEAT_INDEX_COLUMN, [i for i in range(times) for _ in range(n_rows)])
    .add_column(REPEAT_GROUP_COLUMN, [j for _ in range(times) for j in range(n_rows)])
)
```

It travels with the row through `shuffle`, exactly as `repeat_index` does, and
inherits `repeat()`'s existing refusal to stamp a split that already carries the
column.

### Read it from the dataset, not from the context

`repeat_index` does **not** round-trip: `TaskContext.serialize` writes it,
`loader._parse_and_hydrate` never reads it back, and `runner._ensure_raw_sample`
backfills only `if not ctx.is_terminal()`. On a resumed run the samples already
`FINAL` — precisely the ones `report()` aggregates — carry `repeat_index = None`.
A dedup keyed on the context would therefore **silently disable itself on every
resume**, narrowing the interval by `√times` with no error and no warning.

The grouping is a property of the *dataset*, not of the run, so it is read from
the dataset:

```python
# sieval/core/tasks/task.py
def problem_groups(self, finals) -> list[Hashable]:
    """One clustering key per final, read off the live dataset row.

    `sample_id` indexes the post-transform test set -- the same relation
    `_ensure_raw_sample` relies on -- so this resolves identically on a fresh
    and a resumed run without touching the resume path.
    """
```

Absence of the column means the split was not repeated, and each sample is its
own problem. Unlike an archived record — where absence is ambiguous between "not
repeated" and "written before the field existed" — the live dataset always
carries the column if the transform ran, so the fallback is unambiguous.

This deliberately does **not** widen `_ensure_raw_sample`'s backfill. `--resume`
is a product contract; a change there needs its own justification, and reading
the dataset achieves the same result without one.

Cost: 24 `report()` methods each gain one argument, `groups=self.problem_groups(finals)`.
Mechanical, and it makes the dependency explicit rather than implicit — the
alternative that needs no task edits is the context route, which is the one that
fails silently on resume.

### The grouping is task-owned, so core and a task cannot disagree

`problem_groups` is an **overridable method on `Task`**, not a core function.
The base implementation reads `repeat_group`; a task whose clustering comes from
somewhere else overrides it. There is exactly one grouping key per task and the
task owns it, so the two mechanisms cannot both fire and produce a
double-collapse.

`ugmathbench_0shot_gen_fixed` is the case that forces this. One sample is one
*(problem, version)* pair — three randomized versions per problem, 15,183
versions over 5,061 problems — and the clustering is inherent to the data:
the task never calls `Dataset.repeat`, and its `report()` already groups by
`problem_id` off the judgement's `extra` (with an `_identify(final)` fallback for
a version that cannot name its problem). So it overrides `problem_groups` to
return that same `problem_id`, and core's collapse operates on the task's own
key.

The composition that looks like a conflict resolves itself: repeating a
UGMathBench split gives copies that **share `problem_id`**, so grouping by
`problem_id` absorbs the repeat with no special case. A task never needs to
combine the two keys.

**The rule that must not be left implicit is which axis the interval is over.**
UGMathBench's AAcc is per *version*; its EAcc — the headline — is per *problem*.
`score_ci95` is clustered at the problem, matching the headline. An AAcc interval,
if ever added, must not reuse the version-level `m`: that is the same
`√times` narrowing as an uncollapsed repeat, wearing a different name. Any task
publishing metrics on two different axes states, per metric, which axis its
interval belongs to — or omits the interval.

## 5. Report keys

Additive; `cli/leaderboard/scanner.py` still reads `report["score"]` only.

| Key | Shape | Meaning |
| --- | --- | --- |
| `score_ci95` | `[lo, hi]` | 95% interval on `score`, same unit, problem-clustered |
| `n_problems` | count | Declared problem population behind the headline (post-collapse) |

`[lo, hi]` rather than a half-width: the Wilson interval is asymmetric, and the
asymmetry is largest exactly where it matters. `n_problems` is not cosmetic — the
interval's width scales with `m` and no current key carries it (`n` is the budget,
`n_scored_rollouts / n` misestimates under short draws, and only three tasks
report an `n_graded` that counts attempts rather than problems). An interval
without `m` is unreadable.

**Not gated on `n > 1`.** The interval is defined at every budget and is *widest*
at `n = 1`, which is where a reader most needs it. Gating it would withhold it
from the default configuration — the same argument `metrics.health_metrics`
already makes for `n_unextracted`.

`hle_0shot_gen`'s existing `confidence_interval` stays: it is HLE's published
Wald half-width over `(finals + fails) × n` **attempts**, i.e. unclustered, and a
stored column. The two sit side by side as `accuracy` does beside `pass@1`, and
the guide says plainly that they are different estimators of different things.

## 6. Omission and fallback rules

Following `metrics.py`'s standing rule — a key is omitted, never zeroed, when it
cannot be computed:

- `m < 2` — no dispersion to estimate.
- `p̂ ∈ {0, 1}` — the Wilson interval is defined, but `m_eff` is not; fall back to
  the exact one-sided Clopper–Pearson bound rather than omit, since a 0.0 or
  100.0 headline is exactly when a reader needs the bound.
- Any per-problem value missing for some problems — `aggregate` already drops
  such keys rather than averaging them; the interval follows.

Failed samples under `denominator_policy: requested` are **fixed zeros**, not
sampled outcomes: they enter `Σ xᵢ` and `D` but contribute no variance
(RFC #94 §E). Under `judged` they are absent from both.

## 7. Why the rollout interval is excluded

Measured on 12 real runs of the same checkpoint (`qwen3-a3b-sft-v02-dpo-rl-199step`,
`n = 64`, `m = 30`, three `max_tokens` settings × two repeats):

- **Within a run** the closed-form rollout SE is exactly calibrated: split-half
  over 12 runs, `mean z² = 1.05`.
- **Across runs** it is 2.0x too narrow: 6 identical-config replicate pairs,
  `mean z² = 4.18`; 4 of 6 re-runs fell outside their nominal 95%.
- The excess is systematic, not sampling: 5 of 6 pairs moved the same direction,
  mean −1.70 pp between 2026-02-24 and 2026-02-25.

Rollout sampling noise is therefore fully explained by the closed form, and the
remaining factor of two is between-invocation drift that no single-run estimator
can see. Shipping a rollout interval labelled "how much a re-run would move"
would be wrong in the direction that manufactures false regressions.

The drift itself is **not pursued here**: it is plausibly the serving engine
differing between the two days rather than anything the eval side controls, and
run-to-run movement within roughly 3 pp is treated as acceptable. Recorded because
it bounds what a rollout interval could ever claim, not as an open action. One
caveat for whoever uses that 3 pp as a threshold: the six measured pairs average
−1.70 pp but the largest is −3.54 pp, so a single pair can sit just outside it.

## 8. Also needs updating

- `docs/guide/metrics.md` — a section for the two keys, the estimand table from
  §2, the "width is not comparable across budgets" rule, the `hle`
  `confidence_interval` distinction, and the pairs rule extended: an interval
  must be read with its `denominator_policy` and its `n_problems`, and a paired
  delta must never be inferred from two per-run intervals. Plus the axis rule from
  §4: an interval names the axis it is clustered on, and a task publishing metrics
  on two axes (UGMathBench's per-version AAcc beside its per-problem EAcc) says
  which axis each interval belongs to.
- `scripts/check_preflight.py` — `check_report_declarations` is where a
  requirement would be enforced if these become mandatory. Not proposed yet.
- `.claude/rules/` — nothing; this adds no new protocol.

## 9. Tests

- `tests/unit/core/tasks/test_metrics.py` — Wilson reduction to plain Wilson on
  boolean input (exact); bounds respected at `p̂` near 0 and 1; `m < 2` omitted;
  determinism (same input, same output, no RNG touched).
- `tests/unit/core/datasets/test_dataset.py` — `repeat` stamps both columns;
  copy-major alignment; refusal on a pre-stamped split; the column survives
  `shuffle`.
- `tests/unit/core/tasks/test_task.py` — `problem_groups` collapses repeats,
  falls back to per-sample on an unrepeated split, and returns the same grouping
  after `shuffle`.
- `tests/unit/tasks/test_ugmathbench_0shot_gen_fixed.py` — the override groups by
  `problem_id`, so `n_problems` is the problem count and not the version count;
  and a repeated UGMathBench split still yields the problem count, since the
  copies share `problem_id`. This is the assertion that pins the axis rule.
- One end-to-end assertion that a repeated split yields `n_problems` equal to the
  pre-repeat row count, since that is the number the silent-failure mode gets
  wrong.

## 10. Measured evidence

Reproduced by `ci_probe.py` (throwaway) against `outputs/` in the primary
checkout. Numbers quoted above:

| Measurement | Value |
| --- | --- |
| Problem-level CI, `aime_2024`, `m=30`, `n=64` | ±6.68 – 8.86 pp |
| Rollout-level CI, same runs | ±1.10 – 1.57 pp |
| CI vs budget, one run | `n=1` ±11.65 → `n=8` ±7.36 → `n=64` ±6.68 |
| Floor as a share of the `n=1` width | 51 – 65% |
| Within-run calibration, 12 runs | `mean z² = 1.05` |
| Across-run calibration, 6 pairs | `mean z² = 4.18` (σ 2.0x under) |
| Systematic drift, 6 pairs | −1.70 pp, 5/6 same sign |
| `gpqa` repeat collapse, 792 → 198 | ±3.48 → ±5.36 pp |
| Paired delta vs independent problem-level bars | ±1.6 – 2.2 vs ±10.5 – 11.7 pp |
| `max_tokens` 8192 → unbounded, `aime_2024` | +7.8 pp |

The last two are the follow-up's motivation, not this change's.
