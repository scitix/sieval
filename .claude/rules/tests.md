---
paths:
  - "tests/**/*.py"
---

# Test Rules

- Async tests: `@pytest.mark.anyio` (NOT `@pytest.mark.asyncio`)
- Do NOT run `tests/tasks/` in CI — real API calls
- Assertions must have discriminating power
- Only adjust a test if the original expectation was wrong
- `tests/unit/` directory structure must mirror `sieval/` — e.g. `sieval/core/runners/foo.py` → `tests/unit/core/runners/test_foo.py`
- See `tests/README.md` for details
