---
paths:
  - "pyproject.toml"
  - "pdm.lock"
---

# Dependency Management

- Always use `pdm lock --update-reuse` — never bare `pdm lock`
- Adding a new optional group: first insert the group name into the `groups` list in `pdm.lock` `[metadata]`, then run `pdm lock --update-reuse`
- Verify the diff before committing: only the new group and its dependencies should change; existing package versions must not drift
- New third-party imports in `sieval/tasks/` or `sieval/datasets/` must be covered by a dependency group — verify with `python scripts/check_preflight.py --check check_dep_coverage`
