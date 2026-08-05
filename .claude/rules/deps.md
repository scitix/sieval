---
paths:
  - "pyproject.toml"
  - "pdm.lock"
---

# Dependency Management

- Always use `pdm lock --update-reuse` — never bare `pdm lock`
- Adding a new optional group: first insert the group name into the `groups` list in `pdm.lock` `[metadata]`, then run `pdm lock --update-reuse`
- Verify the diff before committing: only the new group and its dependencies should change; existing package versions must not drift — enforced by `python scripts/check_preflight.py --check check_deps` (pre-commit hook `lock-drift`), which compares against `HEAD` while the lock change is uncommitted and against the merge base with `main` afterwards
- A bump that is genuinely needed is declared, not resolved into existence: raise the specifier in `pyproject.toml` so the reason is in the diff. The gate treats a changed specifier the locked version violates, a newly requested extra, and a changed `requires-python` as justification; nothing else
- New third-party imports in `sieval/tasks/` or `sieval/datasets/` must be covered by a dependency group — verify with `python scripts/check_preflight.py --check check_dep_coverage`
