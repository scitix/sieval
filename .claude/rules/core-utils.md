---
paths:
  - "sieval/**/*.py"
---

# Reuse Existing Utilities

Before implementing new helpers, check `sieval/core/utils/` for existing ones:

- `hf.py`: `ensure_dataset_dict()` — HuggingFace dataset validation
- `concurrency.py`: `CompositeLimiter` — rate limiting
- `meta.py`: `build_stage_meta()`, `build_model_call_meta()` — metadata construction
- `serialization.py`: `@sieval_record` — persistence decorator

Flag hand-rolled implementations that duplicate these utilities.

## Prefer Existing Libraries

If a dependency already in `pyproject.toml` provides the functionality, use it — do not reimplement.
