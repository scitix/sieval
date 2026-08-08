# Datasets — Dataset Implementation Guide

## Naming

- Dataset: `XxxDataset` / Sample: `XxxDatasetSample`

## Key Rules

- Inherit from `Dataset[SampleType]` where `SampleType` is a `TypedDict`
- Every concrete Dataset class must be decorated with `@sieval_dataset(...)` from `sieval.core.datasets`
- The sample `TypedDict` is the reverse-lookup key for `@sieval_task`; it must be globally unique across registered Datasets. `name` is also globally unique across Datasets and Tasks.
- `source` must use scheme `hf:` / `url:` / `local:` and is the authoritative origin consumed by `sieval dataset download`.
- `deps_group` here is **loader-side** deps; evaluator-side deps stay on the Task.
- `hf:` sources are revision-pinned; `url:` sources carry per-file `checksums` (sha256).
  Regenerate the meta index (`scripts/sync_meta_index.py`) after editing either.

## Corrected variants

A dataset ships upstream's rows as they are. Repairing a genuinely broken row is
a `datasets/` concern rather than a task one: a separate registered
`<name>_fixed` dataset over the **same pinned revision**, applying a patch table
rather than forking the data — a patch table shrinks to empty when upstream
fixes the row, which is the only exit condition a local fix can have. The
unqualified name keeps tracking upstream, as it does for tasks
(`sieval/tasks/CLAUDE.md`).

No such dataset exists yet. The first one settles the details — do not design
the patch-table format in advance.

## Subpackages

A multi-module benchmark gets a subdirectory; `datasets/__init__.py` lazy-loads it.

- Registered as `subpkg.module_stem`, so the registry never needs the subpackage's
  `__init__.py` to re-export. Re-export only what other packages import directly.
- Discovery skips `_*.py`, and the duplicate-export guard applies *within* a
  subpackage — two modules exporting one name is an error, not last-one-wins.
- No generated `__init__.pyi` — the hand-written `__init__.py` is the type surface, and
  a stub would shadow it rather than supplement it. Only `datasets/__init__.py` itself
  gets one, because only it has a runtime export map to mirror.
