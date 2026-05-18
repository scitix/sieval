---
paths:
  - "sieval/infer/**/*.py"
---

# Infer Layer Rules

- Must NOT import from `sieval.tasks`, `sieval.datasets`, or `sieval.probe` — infer depends only on `sieval.core`
- Backends are stateless: `config` is optional (only for `create()`/`command()`). `status()`/`delete()` only need the `InferHandle`
