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

`denominator_policy` is `requested` (`finals + fails`, so a pipeline failure
counts as **wrong**) or `judged` (`finals` only, failures excluded). The split is
upstream-convention-driven, and unifying it would change `score` for eight tasks
— so it is declared rather than resolved. Two tasks are directly comparable only
when this field agrees; when `fails` is 0 the two coincide.

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
| `avg@k` | Mean verdict over the draw. Numerically equal to `pass@1`. | always |
| `pass@k` | Solved **at least once** in `k` draws. | `k > 1` |
| `pass^k` | **All** `k` draws correct. | `k > 1` |
| `maj@k` | Is the modal **answer** correct? | see below |
| `self_consistency` | Share of the draw that agreed on that modal answer. | the task votes on answers |
| `n`, `k` | The budget the numbers above were measured at. | always |
| `n_short` | Samples that came back with fewer than `n` rollouts. | always |
| `n_unextracted` | **Rollouts** whose answer could not be recovered. | always |

Rates are percentages (0–100) over the task's declared denominator; `n`, `k`,
`n_short` and `n_unextracted` are counts.

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

**`n_short` with everything.** A short sample scores 0 for `pass@k` and `pass^k`,
so a non-zero `n_short` makes every figure above a lower bound.

### Why `maj@k` can be missing

It needs all three:

1. **The task votes on answers.** The code family (HumanEval, MBPP,
   LiveCodeBench) does not — two correct programs are not one answer.
2. **`k == n`.** Sub-sampling would need an unbiased estimator or a seed, and a
   seed in the metric layer is a new source of irreproducibility.
3. **The draw arrived complete.** A truncated draw is whatever finished first,
   not a random subset.

If *any* sample fails one, the key is dropped for the **whole run** — averaging
over the samples that had it would turn a deliberate omission back into the `0.0`
it was avoiding.

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

The MCQ four take **no** `n`/`k` argument, but still grade and record every
rollout that arrives (a model-level `n` reaches them); `report()` warns once with
the count it did not score.

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

The one migration is the `pass@<k>` → `pass@k` rename: a dashboard keyed on the
literal `pass@4` needs updating, and should read the budget from `n` / `k`.
