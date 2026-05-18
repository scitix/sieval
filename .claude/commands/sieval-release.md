---
description: Release a new version of sieval (update changelog, tag, push)
---

# SiEval Release

## Input

$ARGUMENTS

Expects a semantic version number, e.g. `0.3.4`. If not provided, ask the user.

## Process

### 1. Pre-flight Checks

First, run the full preflight scan:

```bash
python scripts/check_preflight.py --level deep --format text
```

If any `[FAIL]` results appear, stop and report them. Do not proceed with the release until all FAILs are resolved.

Then run in parallel and abort if any fail:

```bash
git status --porcelain                 # must be clean
git branch --show-current              # must be main

git fetch origin main
# local main must match remote
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ]

python -m pytest tests/unit tests/integration tests/acceptance --tb=short -q
ruff check .
ty check

gh pr list --state open --limit 20     # list for user review
```

Also collect changes since last tag:

```bash
PREV_TAG=$(git describe --tags --abbrev=0)
git log $PREV_TAG..HEAD --format="%h %s" --no-merges
git diff $PREV_TAG..HEAD --stat
```

If working tree is dirty, not on main, tests/lint fail — stop and report.
If there are open PRs — list them and ask the user whether to include or defer.

### 2. ReferenceImpl Sanity Check

Confirms every registered `reference_impl.url` is still reachable (pinned-SHA
URLs are cryptographically immutable, so `HTTP 200 ⇒ alignment preserved`) and
lets the releaser eyeball the prose `notes` field for any stale descriptions.

1. The preflight step above (`--level deep`) already checks HTTP reachability
   of every registered `reference_impl.url`. Any 4xx on such a URL is a release
   blocker — resolve before proceeding (force-push → GC, repo deleted, repo
   privatized are the typical causes).

2. Render the checklist for manual review:

   ```bash
   python -c "
   from sieval.core.tasks.meta import import_all_tasks, iter_task_metas
   import_all_tasks()
   for m in iter_task_metas():
       if m.reference_impl:
           r = m.reference_impl
           print(f'- [ ] {m.name}')
           print(f'      source: {r.source}')
           print(f'      url:    {r.url}')
           print(f'      notes:  {r.notes}')
           print()
   "
   ```

3. Eyeball each checklist item: does `notes` still accurately describe what we
   vendored/aligned? If a note references a sieval-local symbol that has since
   been renamed, update the note in the task file. If the vendored behavior has
   diverged from the pinned upstream, either realign our implementation or bump
   the pinned SHA and re-vendor.

4. Commit any note / SHA fixes before tagging the release.

### 3. Determine Version

- Parse target version from `$ARGUMENTS`
- Validate: must be `MAJOR.MINOR.PATCH`, higher than `$PREV_TAG`

### 4. Update Files

Update version in:

- `Dockerfile`: `sieval-X.Y.Z-py3-none-any.whl` (COPY and pip install lines)

Insert a new section in CHANGELOG.md **above** the previous version entry:

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- feat commits

### Fixed
- fix commits

### Changed
- refactor/chore/perf commits

### Docs
- docs commits (omit if empty)
```

Rules:

- Today's date
- PR numbers where available (e.g. `(#26)`)
- Human-readable descriptions, not raw commit messages
- Omit empty categories
- Append compare link: `[X.Y.Z]: https://github.com/scitix/sieval/compare/v{prev}...vX.Y.Z`

### 5. Commit, Tag, Push

```bash
git add CHANGELOG.md Dockerfile
git commit -m "chore: release X.Y.Z"
git tag vX.Y.Z
git push origin main
git push origin vX.Y.Z
```

Report final state: version, tag, commit hash.

## Notes

- Do NOT publish to PyPI unless the user explicitly asks
- All pre-commit hooks must pass; do not skip them
- If any step fails, stop and report — do not attempt to recover automatically
