# Report Metrics

Every evaluation writes a `report.json` per task. This page documents the keys
it can contain: what each one means, when it appears, and which pairs have to be
read together.

Two rules govern the whole file:

- **A key never spells the value of `k`.** The column is `pass@k`, not `pass@4`,
  so a leaderboard column keeps its identity when the sampling budget changes.
  The budget is reported once, as the `n` and `k` fields.
- **A key is omitted, never zeroed, when it cannot be computed.** A `0.0` that
  means "not measurable" is indistinguishable from one that means "measured, and
  it was zero". If a key you expect is missing, something below explains why.

## Always present

| Key | Meaning |
| --- | --- |
| `score` | The headline. What a leaderboard ranks on. |
| `score_key` | Which other key `score` was copied from — `pass@1`, `accuracy`, `exact_match`, … A reader never has to guess what the headline measures. |
| `fails` | Samples that failed the pipeline (an error, not a wrong answer). |
| `denominator_policy` | Which population the headline is averaged over. See below. |

### `denominator_policy`

Reports split two ways, and the split is **upstream-convention-driven rather
than accidental** — unifying it would change `score` for eight tasks and break
comparability with every stored number. So the convention is declared instead:

| Value | Denominator | Meaning |
| --- | --- | --- |
| `requested` | `finals + fails` | A pipeline failure counts as **wrong**. DeepSeek's full-set accuracy convention; used by the math-competition family, the code family, PlatinumBench, GSM8K 0-shot, Hendrycks MATH and UGMathBench. |
| `judged` | `finals` | Failures are **excluded** from the denominator. Used by GSM8K few-shot, TheoremQA, and the MCQ tasks. |

Two runs of different tasks are only directly comparable when this field agrees.
When `fails` is 0 the two policies coincide.

## Sampling metrics

These appear **only when the task ran with `n > 1`**. At `n = 1` there is no draw
to describe: `avg@k` would restate `pass@1`, `maj@k` would restate the single
verdict, and the health counters would be constants.

Set the budget as a **task** argument, not on the model:

```yaml
tasks:
  aime_2025_0shot_gen:
    args:
      n: 4   # sampling budget: rollouts drawn per problem
      k: 4   # the k in pass@k -- the metric's parameter. k <= n is enforced.
```

`n` on the model config is overridden call-time by every task that takes its own
budget, and `k > n` is rejected at construction rather than silently reporting a
`pass@k` of `0.0`.

| Key | Meaning | Appears when |
| --- | --- | --- |
| `pass@1` | Unbiased single-draw success rate, `c/n`. **Not** "the first rollout's verdict". | always |
| `avg@k` | Mean verdict over the draw. Numerically equal to `pass@1`. | always |
| `pass@k` | Solved **at least once** in `k` draws. | `k > 1` |
| `pass^k` | **All** `k` draws correct. | `k > 1` |
| `maj@k` | Is the modal **answer** correct? | `k == n`, the draw is complete, and the task votes on answers |
| `self_consistency` | Share of the draw that agreed on that modal answer. | the task votes on answers |
| `n`, `k` | The budget the numbers above were measured at. | always |
| `n_short` | Samples that came back with fewer than `n` rollouts. | always |
| `n_unextracted` | **Rollouts** whose answer could not be recovered. | always |

All rates are percentages (0–100), averaged over the task's declared
denominator. `n`, `k`, `n_short` and `n_unextracted` are counts.

### Pairs that must be read together

**`pass@k` with `pass^k`.** `pass@k` is an upper bound, so a model whose sampling
variance *grew* can score **higher** on it while being worse to ship. `pass^k` is
the opposite direction and falls when that happens. Neither is meaningful alone:

| Draw (4 rollouts) | `pass@1` | `pass@2` | `pass^2` |
| --- | --- | --- | --- |
| 4 correct | 100 | 100 | 100 |
| 2 correct | 50 | 83.3 | 16.7 |

Same benchmark, halved reliability: `pass@k` fell by 17 points, `pass^k` by 83.

**`maj@k` with `self_consistency`.** `maj@k` is thresholded — a model answering
4/4 the same way and one answering 3/4 both score 1.0 if the modal answer is
right. `self_consistency` is continuous (`0.75` for the second), so it is the
only key that moves when a converted or requantized model's answer distribution
**widens without its mean changing**. That is the delivery defect the family
exists to catch.

`self_consistency` is **correctness-blind**: a consistently *wrong* model scores
100. Read it beside a correctness key, never instead of one.

**`self_consistency` with `n_unextracted`.** `self_consistency`'s denominator is
the whole draw, so a rollout whose answer could not be parsed drags it down. That
conflates model instability with extractor failure — `n_unextracted` is what
separates them. A low `self_consistency` with `n_unextracted` near zero is the
model; the same figure with a high `n_unextracted` is the parser.

**`n_short` with everything.** A sample that came back short scores 0 for
`pass@k` and `pass^k` and biases every sampling metric downward. A non-zero
`n_short` means the numbers above are lower bounds, not measurements.

### Why `maj@k` can be missing

It needs the whole draw, and three conditions:

1. **The task votes on answers.** The code family (HumanEval, MBPP,
   LiveCodeBench) passes none — two correct programs are not one answer, and a
   majority over programs needs behavioural clustering that is out of scope.
2. **`k == n`.** Sub-sampling `k < n` would need an unbiased estimator or a seed;
   a seed in the metric layer is a new source of irreproducibility.
3. **The draw arrived complete.** A truncated draw is whatever finished first,
   not a random subset, so it is precisely the one not to vote on.

If *any* sample in the run fails a condition, the key is dropped for the **whole
run** rather than averaged over the samples that had it — otherwise a deliberate
omission would reappear as the `0.0` it was avoiding.

### `maj@k` is a lower bound

Votes are clustered on **strings**, after the canonicalizer the task already
applies to its golds, while the grader compares **symbolically**. So
`\dfrac{1}{2}` and `1/2` become one vote, but `0.5` and `\frac{1}{2}` still
split. The bias is therefore downward — a real majority can be missed, a false
one cannot be manufactured — which makes the reported figure a floor rather than
an estimate. The same clustering backs `self_consistency`, so the two never
disagree about what counts as one answer.

## Single-draw tasks

The MCQ tasks (`gpqa_diamond_0shot_gen`, `mmlu_0shot_gen`, `mmlu_pro_0shot_gen`,
`openbookqa_kshot_gen`) and the accuracy-headline math tasks
(`gsm8k_*`, `hendrycks_math_kshot_base_gen`, `theoremqa_kshot_base_gen`,
PlatinumBench) publish a **greedy single-draw** number upstream. Their headline
therefore keeps its first-rollout definition even under `n > 1`, and `pass@1`
(`c/n`, the better estimator of the same quantity) is reported *beside* it rather
than merged into it. At `n = 1` the two coincide, which is what makes adopting a
sampling budget non-breaking for a stored score.

The MCQ four take **no** `n`/`k` argument at all. They still grade and record
every rollout the model returns — a model-level `n` reaches them — but the
headline scores the first alone, and `report()` warns once with the count it did
not score.

## Task-specific keys

Beyond the above, a task may emit keys of its own: `accuracy`, `errors`
(PlatinumBench's upstream headline unit), `timeouts` (the code family),
`exact_match` / `flexible_exact_match` (GSM8K few-shot, lm-eval's two co-equal
metrics), `empty` (TheoremQA), `score_<category>` (MMLU, MMLU-Pro), and
UGMathBench's `eacc` / `aacc` / `cacc` / `delta`. `score_key` always names which
of them the headline came from.

## Reading a report across versions

Adding a key is non-destructive: `sieval leaderboard report` reads `score` and
nothing else, so a run recorded before these keys existed stays readable and
stays comparable on the headline. Old runs are **not** backfilled — the sampling
metrics are computed inline at report time, so a stored `report.json` remains a
function of the run that produced it.

The one exception in the other direction is the `pass@<k>` → `pass@k` rename: a
dashboard keyed on the literal `pass@4` needs updating to `pass@k`, and should
take the budget from the `n` / `k` fields.
