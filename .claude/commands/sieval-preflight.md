---
description: Run preflight checks on the sieval codebase (link validation, dependency consistency, task/dataset registration, version alignment)
---

# SiEval Preflight Checks

## Input

$ARGUMENTS

Accepts: `deep`, `quick` (default), or a specific check name (e.g., `check_links`, `check_deps`, `check_dep_coverage`, `check_tasks`, `check_datasets`, `check_examples`, `check_imports`, `check_meta_index_sync`, `check_version`).

## Process

### 1. Parse Arguments

- No argument or `quick` → `--level quick`
- `deep` → `--level deep`
- Any other value → `--check <value>` (single check mode)

### 2. Run Mechanical Checks

```bash
python scripts/check_preflight.py --level <level> --format text
# or for single check:
python scripts/check_preflight.py --check <name> --format text
```

If the script exits with code 1, there are FAILs. Report all output to the user.

### 3. Semantic Checks

After the script completes, perform these checks that require judgment:

- Read `README.md`, `CONTRIBUTING.md`, `docs/guide/` — verify CLI examples reference real `sieval` subcommands, flag obviously wrong flags or syntax
- Skim each layer's `CLAUDE.md` — check constraints still match code reality

In `deep` mode, also:

- Read `CONTRIBUTING.md` end-to-end for outdated setup instructions or stale references
- Spot-check docstrings in recently changed files (`git diff --name-only HEAD~10`) for accuracy

### 4. Report

1. **Script output** — all PASS/FAIL/WARN/SKIP lines
2. **Semantic findings** — doc accuracy issues
3. **Summary** — total counts by status
4. **Recommendations** — for each FAIL/WARN, suggest specific fix

## Notes

- FAIL = preflight did not pass. If part of a release, abort.
- WARN = informational, user decides.
- `deep` HTTP link checks may take 1-2 minutes.
