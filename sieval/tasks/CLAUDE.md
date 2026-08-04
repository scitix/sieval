# Tasks — Task Implementation Guide

## Naming Conventions

File: `<task>_<N>shot_<mode>.py` — suffix determines `model_type`:

| Suffix | `model_type` |
| --- | --- |
| `_gen.py` | `"chat"` |
| `_base_gen.py` | `"gen"` |
| `_ppl.py` | `"gen"` |
| `_clp.py` | `"gen"` |

Class: `<Benchmark><ShotType><Mode>Task` — words for shot count (`ZeroShot`, `FewShot`).

### `ppl` vs `clp`

- `ppl` — pick the answer whose full `context + candidate` has the highest
  sequence likelihood; one inference per candidate, any answer length.
- `clp` — pick the answer by the next single token's log-prob over a fixed set
  of option tokens, read from the API's `top_logprobs` in one inference.
  Single-token / labeled-choice answers only; tokenizer-sensitive.

## Key Rules

- ≥ 5 task files per benchmark → subdirectory with an empty `__init__.py` (lazy loading is handled by the top-level `tasks/__init__.py`). Empty on purpose: nothing imports a task class by name — the registry resolves it through the top-level package — so the subpackage needs no type surface of its own, and leaving it empty keeps exactly one import path per task. Contrast `sieval/datasets/`, whose subpackages do re-export, because tasks import them directly.
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
- **prediction** — the model's extracted answer. `None` means "could not extract" — never `""` or `-1`.
- **reference** — ground truth; `None` when it is a *procedure* (test suite, rubric).
- **correct** / **score** — the headline verdict. `correct` is the only axis comparable across tasks.
- **metrics** — every metric measured, by name. A task with *co-equal* metrics (IFEval strict + loose, HellaSwag `acc` + `acc_norm`) records them all here and derives the headline from them; a metric parked in `extra` is hidden from any reader that doesn't already know the task.
- **grade** — an LLM autorater's categorical output (CORRECT / INCORRECT / NOT_ATTEMPTED). A judgement *contains* a grade — grade sits **below** judgement, not beside it.
- **grader** — the LLM actor (the `grader` task arg / model). Not every judgement has one; persist its whole `ModelOutput` in `extra` rather than hand-picked fields.
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
