# Tasks — Task Implementation Guide

## Naming Conventions

File: `<task>_<N>shot_<mode>[_<variant>].py` — the mode determines `model_type`:

| Mode | `model_type` |
| --- | --- |
| `_gen.py` | `"chat"` |
| `_base_gen.py` | `"gen"` |
| `_ppl.py` | `"gen"` |
| `_clp.py` | `"gen"` |

Class: `<Benchmark><ShotType><Mode>[<Variant>]Task` — words for shot count
(`ZeroShot`, `FewShot`).

### Variants

An optional trailing segment lets two readings of one benchmark coexist as
separate registered tasks. The name is the registry key *and* the run-directory
name, so it is the only place the distinction can live.

| Variant | Means |
| --- | --- |
| *(none)* | Tracks upstream — its protocol, its grader, its defects |
| `_fixed` | Ours, diverging to repair a defect in upstream's grader or data |

- **The unqualified name always tracks upstream, bugs included**, and is never
  repurposed by a local change. It stays free even if nothing will occupy it —
  `ugmathbench`'s faithful grader cannot ship at all (upstream is GPL-3.0).
- **`_fixed` is licensed by a defect, not a preference**, and owes two things:
  every divergence enumerated in `reference_impl.notes`, and its score impact
  **quantified**. An unmeasured fork is not a fix.
- The mode is read positionally, so a variant may not spell one:
  `foo_0shot_clp_gen.py` has two readings and is rejected.
- The table is the current vocabulary, not the limit — a new variant earns a row
  when a second real case arrives. Do not coin one speculatively.

Not variants: a different **measurement regime** (that is a mode —
`arc_challenge_kshot_clp` vs `_ppl`), and a fix to **problem text or reference
answers** (a `datasets/` concern — see `sieval/datasets/CLAUDE.md`).

**Fidelity stops at execution safety.** Tracking upstream never extends to
reproducing a path that executes model output, escapes the run directory, or
cannot be bounded — grading is synchronous on one shared event loop, so an
unbounded grade stalls the session, not just the sample. There the unqualified
task carries the *hardened* behaviour, and it is the one divergence that does
**not** earn a `_fixed`: a variant exists so two readings can be compared, and
the unsafe reading is not one we will run. It still owes three things:

- **Safety, not repair** — preserve upstream everywhere safety does not object,
  including where upstream is wrong. A grader defect noticed along the way is
  still a `_fixed` owing its own number, and a large safety delta is evidence
  one got smuggled in.
- **A quantified score impact** before shipping `stable`, measured against
  upstream's actual behaviour on a stored run.
- **Evidence that no bound binds** on the pinned data — a bound that truncates a
  real comparison is a scoring change wearing a safety label.

`theoremqa_kshot_base_gen` is the worked example; its `reference_impl.notes`
carry the measurement.

### Constructor knobs: `n_shot` vs `k`

Two counts that read alike and are not interchangeable. The spelling rules are
machine-checked (`check_preflight.py --check check_task_shot_knobs`, over every
constructor under `sieval/tasks/`); what the checker cannot tell you is why:

- **`n_shot`** — the few-shot exemplar count. Store it as **`self.n_shot`**, not
  `self._n_shot`: assigning the public field shadows the class value
  `@sieval_task(n_shot=...)` seeded, and *that* is what `meta.json` records as
  the count the run used. Store it privately and the class value stands, so the
  run directory reports the declared default rather than what ran. A task with
  no knob needs no code — the seeded value is already right.
- **`k`** — the `k` in `pass@k`, and nothing else: the metric's parameter, not
  the sampling budget. That is `n`, forwarded to `agenerate(n=...)`, with
  `k <= n`. A task taking `k` must compute a `pass@k` metric, and `self.n_shot`
  may never be fed from it.

`fewshot_split` / `fewshot_seed` / `fewshot_as_multiturn` name a different noun
each and are unaffected. In prose, `k-shot` (the file-naming genre) and `top-k`
(logprobs breadth) are unrelated to either knob.

### `ppl` vs `clp`

- `ppl` — pick the answer whose full `context + candidate` has the highest
  sequence likelihood; one inference per candidate, any answer length.
- `clp` — pick the answer by the next single token's log-prob over a fixed set
  of option tokens, read from the API's `top_logprobs` in one inference.
  Single-token / labeled-choice answers only; tokenizer-sensitive.

## Key Rules

- ≥ 5 files per benchmark → subdirectory with an empty `__init__.py`. Two mechanisms make nesting work and both are load-bearing: lazy export by the top-level `tasks/__init__.py`, and name → class resolution by `get_task_class()` (`sieval/core/tasks/meta.py`), which scans subpackages once the flat path misses. With only the export half, a nested task imports fine but is unresolvable by name — a bare `KeyError` from `sieval task show`, no other symptom.
- The `__init__.py` is empty on purpose: both mechanisms key on the *module* (importing it runs `@sieval_task`, which registers the class), never on a subpackage attribute, so it needs no type surface and there stays exactly one import path per task. Contrast `sieval/datasets/`, whose subpackages do re-export, because tasks import them directly.
- **The count includes the extracted shared module**: 4 task modules plus a `_base.py` trips it. Deliberate — a benchmark only grows a shared module once its variants have logic worth reusing, so that module *is* the signal the group has become a unit, and it is what a flat layout has nowhere good to put. `arc/` (4 tasks + `_base.py`) is the reference layout.
- Keep the full task name as the filename inside the subpackage (`arc/arc_easy_kshot_ppl.py`, not `arc/easy_kshot_ppl.py`), so a grep for a registered task name still finds its file.
- Subpackage shared base module: file named `_base.py` (private module), classes inside without underscore prefix (package-internal public API, e.g. `from ._base import XxxTask`).
- General code-quality + layer rules live in `.claude/rules/tasks.md`.

## Report Declarations

Every `report()` says two things about its own headline, using the fields in
`sieval.core.tasks.metrics`:

- **`score_key`** — which key `score` was copied from (`accuracy`, `f1`,
  `pass@1`, `acc_norm`, …); a headline with no aliased twin names itself
  (`mmlu_kshot_clp`, `ruler`). It must name a key the same report writes:
  nothing reads the field at run time, so a column that is not there goes
  unnoticed. Required whenever the report emits a `score`. A task
  publishing one rate per axis and no headline omits both — picking an axis to
  crown would invent a ranking upstream does not make
  (`t_eval_before_calling_0shot_gen` is the only case).
- **`denominator_policy`** — `DENOMINATOR_REQUESTED` (`finals + fails`, so a
  pipeline failure counts as wrong) or `DENOMINATOR_JUDGED` (`finals` only,
  failures excluded). Those two words are the vocabulary; a third reads as a
  policy rather than as the typo it is.

The denominator split is upstream-convention-driven and deliberately **not**
unified — unifying it would move `score` for eight tasks — and that is exactly
what makes declaring it load-bearing: two numbers in one leaderboard column are
comparable only when the field agrees. Neither key is inferable from the values,
and both are additive, so a bare report scores correctly and stays silent about
what it measured.

A third group is checked by the same rule set, though it is measured rather than
declared: `SCORE_CI_FIELD` (`score_ci95`), `PROBLEM_COUNT_FIELD` (`n_problems`)
and `CI_UNITS_FIELD` (`ci95_units`) are emitted **together or not at all** — and
that rule applies per METRIC, so a report carrying one metric's count beside
another's interval is correct rather than half-done. What the three keys mean, why
each is unreadable without the others, and which report-level shapes look like
broken pairs and are not, are stated once in
[`docs/guide/metrics.md`](../../docs/guide/metrics.md) §"Intervals" — the reader's
contract is canonical there, so it is cited here rather than restated in different
words. What follows is only what an author does about it.

Never spell any of those keys here. Merge them from `metrics.py`:
`interval_metrics` for the headline, `metric_interval` for any other metric (it
takes the population key as `unit`), `sampling_report`, which does this for every
key of the sampling block, and `ungated_intervals`, which lifts the block's
always-published intervals out of an `n > 1` gate and trims the declaration to
what came with them.

Both emitters take `aliases` — the other key names the **same number** is
published under. Pass them on the same call, never as a second call with the same
arguments: the alias interval must *be* the headline's, and two calls are only
equal until one of them is edited. A true alias only; a metric that is a different
number gets its own `metric_interval` call over its own per-unit values, and one
that is a deterministic function of another gets nothing at all. The guide's "One
interval per metric" has that rule and its two live cases; the emitters cannot
enforce it, since they never see the values the report publishes, so the runner
does.

A metric's `unit` must be a population count — an `n_…` key the same report
writes. `metric_interval` **raises** on anything else before it builds anything,
which is the one refusal that does not wait for the runner: the population is
written UNDER that key, so save-then-raise would leave a count where the report
holds a rate. A declaration nobody can read is worth saving and refusing; a number
that is wrong is not.

Two fragments that each carry intervals are folded with `merge_metrics`, never
with `|`: a plain merge replaces `ci95_units` wholesale, so one fragment's
declarations survive and the other's intervals are left with no unit — silently,
because the intervals themselves are all still there. The preflight cannot catch
that (a per-metric interval key is built from a metric name, not a literal, so its
key set is not the runtime one); the **runner** does, at report-write time, by
saving `report.json` and then raising on any interval whose unit is missing,
unresolvable, or shared with a metric publishing a different number — plus a unit
that is not a count in a **hand-written** `ci95_units`, the one route that gets
past the emitter's own refusal.

Machine-checked over every class defining `report` under `sieval/tasks/`:
`check_preflight.py --check check_report_declarations`. A `report` that is a
single `return helper(...)` is judged on the helper, so a shared report
(`arc/_base.py`) declares once on behalf of all its leaves rather than four times.

### Metric key names follow upstream, not a house word

**A metric key is spelled the way the benchmark publishes it.**
`wikisql_0shot_gen` reports `ex_accuracy` / `lf_accuracy` because those are
`evaluate.py`'s own two JSON keys, so a number read off a report leads back to
the code that defines it.

The pull is always toward unifying: two benchmarks in one genre each measure
"did it reach the right answer" and "did the structure match", so one pair of
names looks tidier than two. Resist it on the **second** of those, which is
seldom the same measurement twice. WikiSQL's `lf_accuracy` compares condition
triples as an *unordered set*; an "exact match" elsewhere is its own upstream's
tree or string comparison, agreeing with it on most rows and not by
construction. One spelling over both asserts an equivalence nobody measured,
and it costs the pointer back to upstream — the more load-bearing property for
a port, since that pointer is what a divergence is later argued against.

Three things follow:

- **Cross-task comparability is declared, not spelled.** `score_key` names the
  headline and `denominator_policy` says what it is over; renaming a metric
  buys none of what those two buy.
- **A real correspondence goes in `reference_impl.notes`**, where it can carry
  the caveat it needs ("read `lf_accuracy` as the structural match, modulo
  condition order") — never encoded by collapsing two names into one.
- **A key that is genuinely the same count does keep one name**:
  `n_execution_errors` means the same thing in any task that executes a
  prediction. Sameness, not symmetry, is what earns that.

Nothing enforces this — the judgement is about what two upstreams mean, which
no checker can read.

## Stage-Output Protocol (opt-in)

The record vocabulary and its traps live in `.claude/rules/records.md`, scoped to
this tree *and* `sieval/core/tasks/` — the contract has two parties, and a rule
only the writing side can see is how `grader_output` gets renamed. Schema and
builder contracts are authoritative in
[`sieval/core/tasks/records.py`](../core/tasks/records.py).

## `infer_args` — Per-Task Inference Override

YAML-level `infer_args` overrides model inference parameters via `EvalSession` calling `model.with_args(**infer_args)`.

## Task Metadata: `@sieval_task`

- Every concrete Task class must be decorated with `@sieval_task(...)` from `sieval.core.tasks`; abstract base classes stay undecorated.
- Do not set `tags: ClassVar[set[str]]` manually on decorated classes — `@sieval_task` synthesizes `cls.tags` from `eval_mode` + `n_shot` for anomaly routing; the decorator's `tags=(...)` kwarg is a separate, user-facing descriptive label.
- `reference_kind` declares what form the ground truth takes: `"value"` (a letter, a number, a set of strings) or `"procedure"` (a test suite, a rubric, which the record describes in `extra` instead of faking a value for). It is a task-level constant, which is why it lives here and not on every judgement record. **Purely descriptive — nothing branches on it**: it exists so an archived run's `meta.json` can say why its judgements carry no `reference` after the installed registry has moved on. Machine-checked against the task's own `build_judgement_record` call sites: `check_preflight.py --check check_reference_kind`.
    - The criterion is **operational**: `"value"` means the task records a reference at all, not that the reference looks scalar. IFEval records the list of instruction ids its checkers run — a procedure's parameters by any reading — and is still a `"value"`, because it is recorded and compared against. Reserve `"procedure"` for a task that records nothing to compare against, which is why the check infers the kind from whether the call sites pass a literal `None` rather than from what the reference looks like. A value task that finds no gold raises instead of recording a judgement (`.claude/rules/records.md`).
    - **Spell it on every task**, `"value"` included, immediately before `reference_impl`. The parameter *defaults* to `"value"` — deliberately, so out-of-tree and plugin tasks are not forced to change and the field stays additive — but a default is only invisible until someone has to look it up. All 55 in-tree tasks declare it. The check reads the effective value either way, so this buys readability, not correctness.
- `TaskMeta.dataset` (the FK to a registered Dataset) is resolved automatically from the Task's first generic arg (its sample `TypedDict`); do not pass `dataset=` explicitly. The referenced Dataset class must already be `@sieval_dataset`-decorated (see `.claude/rules/datasets.md`) and its `source` is the authoritative origin consumed by `sieval dataset download`.
- Per-run knobs (`k`, `n`, `temperature`, `seed`) stay in runner config, not TaskMeta.
- `sieval/meta/index.json` is auto-generated; full field reference lives on the decorator docstring.
