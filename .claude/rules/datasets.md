---
paths:
  - "sieval/datasets/**/*.py"
---

# Dataset Loader Rules

- All dataset loaders must use the appropriate `ensure_*` helper from `sieval.core.utils.hf` (`ensure_dataset_dict`, `ensure_dataset`, `ensure_dataset_list`) — do not hand-roll `isinstance(dataset, DatasetDict)` checks
- After filtering, check for empty datasets to catch typos or schema mismatches
- New datasets must be downloadable via `sieval dataset download <name>`; the `source` field is the authoritative origin.
- Verify with `python scripts/check_preflight.py --check check_datasets` after adding new datasets

## Dataset Metadata: `@sieval_dataset`

- Every concrete `Dataset[TSample]` subclass must be decorated with `@sieval_dataset(...)` from `sieval.core.datasets`.
- The sample `TypedDict` is the reverse-lookup key for `@sieval_task`; it must be globally unique across registered Datasets. `name` is also globally unique across Datasets and Tasks.
- `source` must use scheme `hf:` / `url:` / `local:` and is consumed by `sieval dataset download` to stage data into `$SIEVAL_DATA_DIR`.
- `deps_group` here is **loader-side** deps; evaluator-side deps stay on the Task.
- `hf:` sources MUST be revision-pinned (`hf:org/name@<sha>`); `url:` sources MUST declare
  per-file `checksums` (sha256). `check_datasets` enforces both.
- Re-run `scripts/sync_meta_index.py` after editing `source`/`checksums` (the runtime reads
  the generated `index.json`).
