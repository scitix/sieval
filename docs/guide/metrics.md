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

Both keys are enforced rather than conventional:
`scripts/check_preflight.py --check check_report_declarations` fails a task whose
`report()` omits either, whose policy is a word other than the two above, or
whose `score_key` names a column the report does not contain — nothing reads
`score_key` at run time, so that last one would otherwise go unnoticed.

One task is the documented exception, and only to the `score` half:
`t_eval_before_calling_0shot_gen` publishes one rate per axis and no headline, so
it emits neither `score` nor `score_key` — there is nothing for the latter to
name. It still declares `denominator_policy`.

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
