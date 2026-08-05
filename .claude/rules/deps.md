---
paths:
  - "pyproject.toml"
  - "pdm.lock"
---

# Dependency Management

- Always use `pdm lock --update-reuse` — never bare `pdm lock`
- Adding a new optional group: first insert the group name into the `groups` list in `pdm.lock` `[metadata]`, then run `pdm lock --update-reuse`
- Verify the diff before committing: only the new group and its dependencies should change; existing versions must not drift — enforced by `python scripts/check_preflight.py --check check_deps` (pre-commit hook `lock-drift`), baselined on `HEAD` while the lock change is uncommitted and on the merge base with `main` afterwards
- Declare bumps, don't resolve them into existence: raise the specifier in `pyproject.toml` so the reason is in the diff. The gate asks one question per moved package — could a `--update-reuse` re-lock have kept the old pin? A move is justified only when the old version violates a specifier imposed by something that did **not** itself move (a raised declaration, or a new dependency's own requirement); being reachable from a new dependency is not enough. A changed `requires-python` re-resolves everything, so it excuses what it cannot attribute and lists those moves for manual review instead of passing silently
- New third-party imports in `sieval/tasks/` or `sieval/datasets/` must be covered by a dependency group — verify with `python scripts/check_preflight.py --check check_dep_coverage`
