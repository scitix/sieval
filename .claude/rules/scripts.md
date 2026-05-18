---
paths:
  - "scripts/*.py"
---

# Script Rules

- Scripts with non-trivial logic should have tests in `tests/unit/scripts/test_<name>.py`
- Preflight / enforcement scripts (`scripts/check_*.py`) additionally trigger `.claude/rules/engineering-infra.md` — walk that checklist too
