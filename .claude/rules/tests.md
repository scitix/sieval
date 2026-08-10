---
paths:
  - "tests/**/*.py"
---

# Test Rules

- Async tests: `@pytest.mark.anyio` (NOT `@pytest.mark.asyncio`)
- Do NOT run `tests/tasks/` in CI — real API calls
- Assertions must have discriminating power
- Only adjust a test if the original expectation was wrong
- `tests/unit/` **directory** structure mirrors `sieval/`. Files within it: `test_<module>.py` (default), `<module>/test_<topic>.py` (one big module split), `test_<subject>_family.py` (one contract over clone modules), `test_<concern>.py` (no single owner)
- See `tests/README.md` for details
