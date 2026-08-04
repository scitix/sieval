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

Per-stage record types so a sample's answer / ground truth / verdict is readable without knowing which task produced it. **Authoritative type + builder definitions:** [`sieval/core/tasks/records.py`](../core/tasks/records.py) — this section is the vocabulary and conventions, not the schema. Opt-in per task; legacy return shapes keep working, so migrate deliberately, not en masse.

Records are named by **content**, not by the stage that emits them — a shard line reads `prediction` / `judgement`, not `postprocess` / `feedback`:

| stage (`*_result` field) | record | holds |
| --- | --- | --- |
| `preprocess` | `PromptRecord` | `prompt` + `reference` (GT known at build time) |
| `infer` | `ModelOutput` | already uniform — deliberately **not** wrapped |
| `postprocess` | `PredictionRecord` | `rollouts[]` of extracted answers |
| `feedback` | `JudgementRecord` | `rollouts[]` of verdicts + `n_rollouts` / `n_correct` |

Vocabulary — these denote **different layers**, keep them distinct:

- **judgement** — the verdict record (feedback output), mechanism-agnostic: a string-compare, math-verify, test-suite, *or* LLM verdict all produce one.
- **prediction** — the model's extracted answer (postprocess output). Use `None` for "could not extract" (never `""` / `-1`); read `extracted`, since a `None` prediction is absent on disk.
- **reference** — ground truth. `None` when it is a *procedure* (test suite, rubric), described in `extra`.
- **correct** (bool) — the headline binary verdict. **score** (float, optional) — genuine partial credit only; never mirror `n_correct / n_rollouts`.
- **grade** — an LLM autorater's categorical output (e.g. CORRECT / INCORRECT / NOT_ATTEMPTED). A judgement *contains* a grade for LLM tasks — grade sits **below** judgement, not beside it.
- **grader** — the LLM actor (the `grader` task arg / model). Not every judgement has one.
- **judge** — HLE only, upstream's synonym for grader/grade with its own `parse_judge` contract; do not introduce it in new tasks.
- **extra** — mechanism-specific detail (a grader's full `ModelOutput`, a code runner's failure message, per-constraint results). Named for the mechanism, not for a "grader". Store what the mechanism actually reported; do not derive a taxonomy from another service's free text at write time — a stored classification looks authoritative and decays silently when the upstream wording drifts.

Conventions:

- Constructors are `build_*` (`build_prompt_record`, `build_prediction_record`, `build_rollout_judgement`, `build_judgement_record`) — matching `build_model_call_meta` / `build_stage_meta`. Structural sniffs are `is_*` (`is_prediction_record`, `is_judgement_record`).
- Return records **bare** — never wrapped in `TaskStageOutput` (the runner keeps the box, breaking the flat-dict shape). A grader is a model, so persist its whole `ModelOutput` in `extra` as a plain dict via `obj_to_dict(out, add_type=False)` rather than hand-picking fields.

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
