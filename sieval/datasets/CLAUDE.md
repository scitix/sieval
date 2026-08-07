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

A dataset ships upstream's rows as they are. Repairing a genuinely broken row
(problem text replaced by an error message, an answer the format cannot express)
means a **separate** registered dataset `<name>_fixed`, paired with a `_fixed`
task variant; the unqualified name keeps tracking upstream.

- **Apply a patch table, never fork the data.** Same pinned `source` revision,
  edits carried as data — `id`, field, old → new, why. A patch table shrinks to
  empty when upstream fixes the row, which is the only exit condition a local fix
  can have; a forked copy has none.
- The table must **fail loudly** when a targeted row no longer matches its
  recorded `old`: upstream re-cut the data and the patch is now guessing.
- A fixed dataset needs its own sample `TypedDict` — that is `@sieval_task`'s
  reverse-lookup key and must be globally unique. An empty subclass suffices.

## Subpackages

A multi-module benchmark gets a subdirectory; `datasets/__init__.py` lazy-loads it.

- Registered as `subpkg.module_stem`, so the registry never needs the subpackage's
  `__init__.py` to re-export. Re-export only what other packages import directly.
- Discovery skips `_*.py`, and the duplicate-export guard applies *within* a
  subpackage — two modules exporting one name is an error, not last-one-wins.
- No generated `__init__.pyi` — the hand-written `__init__.py` is the type surface, and
  a stub would shadow it rather than supplement it. Only `datasets/__init__.py` itself
  gets one, because only it has a runtime export map to mirror.
