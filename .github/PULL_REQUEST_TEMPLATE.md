<!-- markdownlint-disable MD041 -->

## Type

<!-- Choose ONE, delete the rest -->

- feature — new benchmark, task, or capability
- fix — bug fix or alignment correction
- refactor — code restructuring, no behavior change
- docs — documentation only
- chore — CI, tooling, dependencies, config

## Summary

<!--
  What changed and WHY (not just what files were touched).
  Bullet points preferred. Keep it under 10 lines.
-->

-

## Related Issues

<!-- Link issues: Closes #123, Refs #456. Delete section if none. -->

## Test Plan

### Automated

- [ ] Lint/format clean (`ruff check && ruff format --check`)
- [ ] Type check clean (`ty check` or `mypy --strict`)
- [ ] Unit tests pass (`pdm run pytest`) <!-- delete if benchmark-only PR -->

### Manual

<!-- Describe manual verification steps. For benchmark PRs, include:
     - Model(s) tested
     - Expected vs actual scores
     - Diff from reference (target: <3%)
     Delete section if no manual steps needed. -->

- [ ]

## Checklist

### Required (all PRs)

- [ ] PR title follows conventional format (`type(scope): description`)
- [ ] No internal paths, credentials, or personal info in committed files
- [ ] AI-generated code has `AI-Generated Code - <model> (<provider>)` in module docstring
- [ ] No new upper-layer dependencies added to `core/`
- [ ] Deleted code verified — no remaining call sites depend on it

<!-- Keep sections below that apply to your PR, delete the rest -->

### If: New or Modified Benchmark

- [ ] Reference paper/repo linked in Summary
- [ ] Score comparison table included (model, expected, actual, diff)
- [ ] Dataset loading tested (`sieval dataset download <name>` succeeds)
- [ ] Task registered in package-level `__init__.py`

### If: community/ Changes

- [ ] Upstream diff documented (what differs and why)
- [ ] License attribution preserved

### If: Breaking Change

- [ ] Described what breaks and migration path in Summary
- [ ] Existing tests updated to reflect new behavior

### If: New Dependency

- [ ] Added to correct PDM dependency group
- [ ] Justified in Summary (why this package, no lighter alternative?)
