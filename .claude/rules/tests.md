---
paths:
  - "tests/**/*.py"
---

# Test Rules

- Async tests: `@pytest.mark.anyio` (NOT `@pytest.mark.asyncio`)
- Do NOT run `tests/tasks/` in CI — real API calls
- Assertions must have discriminating power
- Only adjust a test if the original expectation was wrong
- `tests/unit/` **directory** structure must mirror `sieval/` — e.g. `sieval/core/runners/foo.py` → `tests/unit/core/runners/test_foo.py`. Within a directory, three file layouts are allowed; anything else needs a reason:
    - **`test_<module>.py`** — one file per source module. The default.
    - **`<module>/test_<topic>.py`** — a directory splitting one large module by topic (`loader.py` → `tests/unit/core/tasks/loader/`).
    - **`test_<subject>_family.py`** — one contract asserted once over sibling modules that are clones of each other. Only when a per-module file *cannot* assert it: the point is that the same fix cannot land in one module and drift in the others.
- See `tests/README.md` for details
