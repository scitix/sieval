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

### `ppl` vs `clp` (per [OpenCompass](https://opencompass.readthedocs.io/zh-cn/latest/get_started/faq.html#ppl-gen))

- **`ppl`** = sequence-likelihood selection: concatenate each candidate
  continuation with the context and compare full-sequence perplexity
  (`n` inferences; supports multi-token answers).
- **`clp`** = next-token conditional log-prob over a fixed set of option
  tokens in a single inference (single-token / labelled-choice answers;
  tokenizer-sensitive). Reads the option tokens from the API's
  `top_logprobs` rather than a local tokenizer.

## Key Rules

- ≥ 5 task files per benchmark → subdirectory with an empty `__init__.py` (lazy loading is handled by the top-level `tasks/__init__.py`).
- Subpackage shared base module: file named `_base.py` (private module), classes inside without underscore prefix (package-internal public API, e.g. `from ._base import XxxTask`).
- General code-quality + layer rules live in `.claude/rules/tasks.md`.

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
