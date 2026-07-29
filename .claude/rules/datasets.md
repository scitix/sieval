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
`TypedDict`, so nothing downstream benefits from two loaders agreeing on a field name or a
dtype. Both rules below therefore start from "what does the pinned revision actually ship",
not "what does the sibling loader do".

- **Keep upstream field names.** Rename only when the upstream name is unusable as-is (not a
  valid identifier, e.g. `Short Answer`; or it collides). Never rename for uniformity with a
  sibling loader — a rename is a permanent divergence from the upstream card that every future
  reader has to reconcile.
- **Cast a column's dtype only when the pinned revision requires it.** Measure the pinned
  snapshot (HF datasets-server, or just load it) instead of copying a sibling: a cast that
  matches the dtype upstream already ships is a no-op that reads as a guarantee. Content-inferred
  dtypes (CSV/JSON parsing) and mismatched dtypes are the load-bearing cases.
- Either way, say **which case it is** in a comment — including when the answer is "no cast
  needed, upstream already ships this dtype". Otherwise the next edit that restores uniformity
  looks like a cleanup.

## Dataset Metadata: `@sieval_dataset`

- Every concrete `Dataset[TSample]` subclass must be decorated with `@sieval_dataset(...)` from `sieval.core.datasets`.
- The sample `TypedDict` is the reverse-lookup key for `@sieval_task`; it must be globally unique across registered Datasets. `name` is also globally unique across Datasets and Tasks.
- `source` must use scheme `hf:` / `url:` / `local:` and is consumed by `sieval dataset download` to stage data into `$SIEVAL_DATA_DIR`.
- `deps_group` here is **loader-side** deps; evaluator-side deps stay on the Task.
- `hf:` sources MUST be revision-pinned (`hf:org/name@<sha>`); `url:` sources MUST declare
  per-file `checksums` (sha256). `check_datasets` enforces both.
- Re-run `scripts/sync_meta_index.py` after editing `source`/`checksums` (the runtime reads
  the generated `index.json`).
