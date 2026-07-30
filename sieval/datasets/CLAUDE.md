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

## Subpackages

Mirrors `sieval/tasks/CLAUDE.md`: a benchmark that needs several loader modules gets
a subdirectory, and the top-level `datasets/__init__.py` handles lazy loading.

- The top-level registry attributes a subpackage export to `subpkg.module_stem`, so
  it resolves straight to the defining module and does **not** require the
  subpackage's `__init__.py` to re-export it. Re-export only what other packages
  import directly (e.g. helpers a Task needs) — not to satisfy the registry.
- One consequence worth knowing: because the recorded module is per-file, the
  duplicate-export guard fires *within* a subpackage too. Two modules in one
  subpackage exporting the same name is an error, not last-one-wins.
- Private modules (`_*.py`) are skipped by discovery, so per-family internals stay
  invisible to the registry. A subpackage may still re-export from them for other
  packages to import (`datasets/ruler/__init__.py` does, for helpers a Task needs).
- A dataset subpackage gets **no** generated `__init__.pyi`, unlike a task
  subpackage — its hand-written `__init__.py` is its own type surface, and a
  generated stub would shadow it and hide anything sourced from a private module.
  `sync_package_stubs.py` has the details; do not add the symmetric loop.
