---
paths:
  - "sieval/datasets/**/*.py"
---

# Dataset Loader Rules

- All dataset loaders must use the appropriate `ensure_*` helper from `sieval.core.utils.hf` (`ensure_dataset_dict`, `ensure_dataset`, `ensure_dataset_list`) — do not hand-roll `isinstance(dataset, DatasetDict)` checks
- After filtering, check for empty datasets to catch typos or schema mismatches
- New datasets must be downloadable via `sieval dataset download <name>`; the `source` field is the authoritative origin.
- Verify with `python scripts/check_preflight.py --check check_datasets` after adding new datasets

## Upstream Sample Shape

There is no shared cross-benchmark sample schema — every Task binds 1:1 to its own sample
`TypedDict`, so nothing downstream gains from two loaders agreeing on a field name or dtype.

- **Keep upstream field names.** Rename only when the upstream name is unusable as-is (not a
  valid identifier, e.g. `Short Answer`, or it collides) — never for uniformity with a sibling.
- **Cast a column's dtype only when the pinned revision requires it.** Measure the snapshot
  instead of copying a sibling; a cast matching the dtype upstream already ships is a no-op that
  reads as a guarantee. CSV/JSON content-inferred dtypes are the load-bearing case.
- Say **which case it is** in a comment, including "no cast needed, upstream ships this dtype" —
  otherwise restoring uniformity looks like a cleanup.

## Dataset Metadata: `@sieval_dataset`

- Every concrete `Dataset[TSample]` subclass must be decorated with `@sieval_dataset(...)` from `sieval.core.datasets`.
- The sample `TypedDict` is the reverse-lookup key for `@sieval_task`; it must be globally unique across registered Datasets. `name` is also globally unique across Datasets and Tasks.
- `source` must use scheme `hf:` / `url:` / `local:` and is consumed by `sieval dataset download` to stage data into `$SIEVAL_DATA_DIR`.
- `deps_group` here is **loader-side** deps; evaluator-side deps stay on the Task.
- `hf:` sources MUST be revision-pinned (`hf:org/name@<sha>`); `url:` sources MUST declare
  per-file `checksums` (sha256). `check_datasets` enforces both.
- Re-run `scripts/sync_meta_index.py` after editing `source`/`checksums` (the runtime reads
  the generated `index.json`).
