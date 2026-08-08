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
- `TaskMeta.dataset` (the FK to a registered Dataset) is resolved automatically from the Task's first generic arg (its sample `TypedDict`); do not pass `dataset=` explicitly. The referenced Dataset class must already be `@sieval_dataset`-decorated (see `.claude/rules/datasets.md`) and its `source` is the authoritative origin consumed by `sieval dataset download`.
- Per-run knobs (`k`, `n`, `temperature`, `seed`) stay in runner config, not TaskMeta.
- `sieval/meta/index.json` is auto-generated; full field reference lives on the decorator docstring.
