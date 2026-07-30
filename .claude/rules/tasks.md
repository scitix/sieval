---
paths:
  - "sieval/tasks/**/*.py"
---

# Task Implementation Rules

## Naming & Model Type

- File naming must follow `<task>_<N>shot_<mode>.py` pattern (authoritative table in `sieval/tasks/CLAUDE.md`):
    - `_gen.py` → `model_type = "chat"`
    - `_base_gen.py` → `model_type = "gen"` (base model, uses GenModel)
    - `_ppl.py` → `model_type = "gen"` (perplexity, uses GenModel)
    - `_clp.py` → `model_type = "gen"` (conditional next-token log-prob, uses GenModel)
- Class naming: `<Benchmark><ShotType><Mode>Task` — words for shot count (`ZeroShot`, `FewShot`)
- `ppl` vs `clp` distinction: see `sieval/tasks/CLAUDE.md`.

## Checklist for New Benchmarks

- Add benchmark-specific dependencies to `pyproject.toml` optional dependency groups (e.g., `[project.optional-dependencies.benchmark_name]`)
- New datasets must be downloadable via `sieval dataset download <name>` — verify the `source` field (`hf:` / `url:` / `local:`) resolves.
- Record the upstream **repeat/sampling protocol** in `reference_impl.notes` when it differs from this task's default `n` (matharena's runner default `--n 4`; simple-evals' `n_repeats`), and say how to match it. Nothing enforces this — it needs upstream knowledge.
- The PostToolUse hooks in `.claude/settings.json` regenerate stubs + `meta/index.json` on a Write/Edit under `sieval/tasks/` or `sieval/datasets/`, in the checkout that owns the file. They do not fire for `sed -i`, `git apply`, or a rebase, so `--check` both sync scripts before committing: CI fails on a stale `meta/index.json`, and stub drift has no enforcer at all.
- ruff and ty run from the same hooks but only report — run the full `ruff check` / `ty check` before committing.
- Run `python scripts/check_preflight.py --check check_tasks` to verify naming, tags, and imports

## Code Quality

- Use `strict=True` in `zip()` when lengths are guaranteed to match
- Must not modify `core/` — check `sieval/core/utils/` for existing helpers first

## Tags — Anomaly Detection

- New tasks: use `@sieval_task(...)` — `cls.tags` is synthesized from `eval_mode` + `n_shot`; do not set manually. See `sieval/tasks/CLAUDE.md` §"Task Metadata: `@sieval_task`".
- Legacy (unmigrated) tasks still declare `tags: ClassVar[set[str]]` with vocabulary `gen / ppl / base / zero_shot / few_shot / llm_judge`.

## Data Flow — Async & Concurrency

- Understand the framework's staged execution data flow before implementing.
- All intermediate state must flow through the framework's persistence layer (record/shard storage) — do NOT use external files, temp caches, or module-level mutable state to pass data between stages.
- Never introduce shared mutable state without proper locking.
