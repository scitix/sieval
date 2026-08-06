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

A dataset ships upstream's rows as they are. When a row is genuinely broken —
problem text replaced by an error message, a reference answer the answer format
cannot express — the repair is a **separate** registered dataset (`<name>_fixed`),
paired with a `_fixed` task variant (see `sieval/tasks/CLAUDE.md`). The
unqualified name keeps tracking upstream.

- **Apply an explicit patch table, never fork a copy of the data.** Keep the
  same pinned `source` revision and carry the edits as data — `id`, field,
  old → new, and why. The diff is then reviewable, what we changed stays legible
  forever, and when upstream fixes the row the entry is deleted. A patch table
  that shrinks to empty is the only exit condition a local fix can have; a forked
  copy has none.
- The patch table must **fail loudly** when a row it targets no longer matches
  the recorded `old` value: upstream re-cut the data and the patch is now
  guessing. Silent no-ops are how a fix survives past the bug it fixed.
- A fixed dataset needs its own sample `TypedDict` — it is the reverse-lookup key
  for `@sieval_task` and must be globally unique. An empty subclass of the
  upstream one is enough to get a distinct key.

## Subpackages

A multi-module benchmark gets a subdirectory; `datasets/__init__.py` lazy-loads it.

- Registered as `subpkg.module_stem`, so the registry never needs the subpackage's
  `__init__.py` to re-export. Re-export only what other packages import directly.
- Discovery skips `_*.py`, and the duplicate-export guard applies *within* a
  subpackage — two modules exporting one name is an error, not last-one-wins.
- No generated `__init__.pyi` — the hand-written `__init__.py` is the type surface, and
  a stub would shadow it rather than supplement it. Only `datasets/__init__.py` itself
  gets one, because only it has a runtime export map to mirror.
