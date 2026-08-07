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

### Constructor knobs: `n_shot` vs `k`

Two counts that read alike and are not interchangeable. Enforced by
`scripts/check_preflight.py --check check_task_shot_knobs`:

- **`n_shot`** — the few-shot exemplar count. The only accepted spelling (not
  `k`, `shots`, `num_shots`, `fewshot`, …). Store it as **`self.n_shot`**, the
  public field `Task` declares and `@sieval_task(n_shot=...)` seeds on the
  class: assigning it shadows the class value for that instance, and that is
  what `meta.json` records as the count the run used. Store it anywhere else
  (`self._n_shot`, …) and the class value stands, so the run directory reports
  the declared default. A task with **no** knob needs no code at all — the
  seeded class value is already right.
- **`k`** — the `k` in `pass@k`, nothing else: the metric's parameter, **not**
  the sampling budget. That is `n`, forwarded to `agenerate(n=...)`, and
  `k <= n` — `pass@k` is estimated from `n` samples per problem. A task taking
  `k` must compute a `pass@k` metric, and `self.n_shot` may never be fed from it.

The `n_shot` rules bind **every constructor under `sieval/tasks/`**, including an
undecorated shared base in a subpackage (`arc/_base.py`) — a decorated task can
inherit its `__init__`, and checking only decorated classes would leave that
knob unchecked. The `k` rule is decorated-classes-only: an undecorated base's
`pass@k` is usually computed by the subclass.

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

Per-stage record types so a sample's answer / ground truth / verdict is readable without knowing which task produced it. **Schema, serialization rules and builder contracts live in [`sieval/core/tasks/records.py`](../core/tasks/records.py)** — authoritative, and deliberately not repeated here. This section is the vocabulary. Opt-in per task; legacy shapes keep working, so migrate deliberately.

Records are named by **content**, not by the stage that emits them — a shard line reads `prediction` / `judgement`, not `postprocess` / `feedback`:

| stage | record |
| --- | --- |
| `preprocess` | `PromptRecord` |
| `infer` | `ModelOutput` — already uniform, deliberately **not** wrapped |
| `postprocess` | `PredictionRecord` |
| `feedback` | `JudgementRecord` |

Vocabulary — these denote **different layers**, keep them distinct:

- **judgement** — the verdict record, mechanism-agnostic: string-compare, math-verify, test-suite *or* LLM verdicts all produce one.
- **prediction** — the model's extracted answer. `None` means "could not extract" — never `""` or `-1`. **Read it with `.get("prediction")`, never `["prediction"]`**: a `None` value is dropped by serialization, so on resume the key is *absent* and `[]` raises `KeyError` for exactly the samples whose extraction failed — on a fresh run the same line is fine, which is why in-memory tests never see it. `extracted` is the durable companion flag. Enforced by `scripts/check_preflight.py --check check_record_key_access`; neither `ty` nor `mypy --strict` reports it.
- **reference** — ground truth; `None` when it is a *procedure* (test suite, rubric).
- **correct** / **score** — the headline verdict. `correct` is the only axis comparable across tasks.
- **metrics** — every metric measured, by name. A task with *co-equal* metrics (IFEval strict + loose, HellaSwag `acc` + `acc_norm`) records them all here and derives the headline from them; a metric parked in `extra` is hidden from any reader that doesn't already know the task.
- **grade** — an LLM autorater's categorical output (CORRECT / INCORRECT / NOT_ATTEMPTED). A judgement *contains* a grade — grade sits **below** judgement, not beside it.
- **grader** — the LLM actor (the `grader` task arg / model). Not every judgement has one; persist its whole `ModelOutput` in `extra` rather than hand-picked fields, under exactly the key **`grader_output`** (`records.GRADER_OUTPUT_KEY`). The name is load-bearing, not cosmetic: the runner reads it back to route grader spend into `profile.json`, since a `feedback` stage returning a bare record has no `ModelOutput` for the profiler to derive a call from. Spell it differently and the grader's tokens are on disk but missing from the profile.
- **judge** — HLE only, upstream's synonym with its own `parse_judge` contract; do not introduce it in new tasks.
- **extra** — mechanism detail, *not* metrics: a grader's `ModelOutput`, a code runner's failure message, per-constraint results.

Constructors are `build_*` (matching `build_model_call_meta` / `build_stage_meta`); structural sniffs are `is_*`. Records are returned **bare**, never wrapped in `TaskStageOutput`.

## `infer_args` — Per-Task Inference Override

YAML-level `infer_args` overrides model inference parameters via `EvalSession` calling `model.with_args(**infer_args)`.

## Task Metadata: `@sieval_task`

- Every concrete Task class must be decorated with `@sieval_task(...)` from `sieval.core.tasks`; abstract base classes stay undecorated.
- Do not set `tags: ClassVar[set[str]]` manually on decorated classes — `@sieval_task` synthesizes `cls.tags` from `eval_mode` + `n_shot` for anomaly routing; the decorator's `tags=(...)` kwarg is a separate, user-facing descriptive label.
- `TaskMeta.dataset` (the FK to a registered Dataset) is resolved automatically from the Task's first generic arg (its sample `TypedDict`); do not pass `dataset=` explicitly. The referenced Dataset class must already be `@sieval_dataset`-decorated (see `.claude/rules/datasets.md`) and its `source` is the authoritative origin consumed by `sieval dataset download`.
- Per-run knobs (`k`, `n`, `temperature`, `seed`) stay in runner config, not TaskMeta.
- `sieval/meta/index.json` is auto-generated; full field reference lives on the decorator docstring.

## Data Flow — Async & Concurrency

- Understand the framework's staged execution data flow before implementing.
- All intermediate state must flow through the framework's persistence layer (record/shard storage) — do NOT use external files, temp caches, or module-level mutable state to pass data between stages.
- Never introduce shared mutable state without proper locking.
