---
paths:
  - "CLAUDE.md"
  - "sieval/**/CLAUDE.md"
  - "CONTRIBUTING.md"
  - ".claude/rules/*.md"
  - "scripts/check_*.py"
  - ".pre-commit-config.yaml"
---

# Engineering Infrastructure Coherence

Policy / enforcement / hook wiring / user-visible names are a coupled chain.
Touching one link without checking the others leaves dead rules or silent
breakage. Before committing, walk whichever section below applies.

## Touched a CLAUDE.md rule or `.claude/rules/*.md`

- Is there a `scripts/check_*.py` that enforces it? If not, intentional?
- Does `.pre-commit-config.yaml` feed the right files to that enforcer?
- Does `scripts/check_preflight.py` wrap it (`CHECKS` entry + wrapper)?
- Does `CONTRIBUTING.md` mirror the user-facing parts (layer deps, toolchain)?

## Touched a `scripts/check_*.py` enforcer

- Pre-commit `files:` still matches the script's scope; hook name still describes it
- `check_preflight.py` wrapper key, `CHECKS` entry, message strings
- Mirror test in `tests/unit/scripts/test_<name>.py`
- CLAUDE.md / `.claude/rules/` principle still matches actual behavior

## Touched `.pre-commit-config.yaml`

- Compute the actual file set: hook `files:` × global `exclude:`, not the regex alone
- Script's internal scope (`in_sieval`, `in_scripts`, …) must match what the hook feeds — otherwise enforcement is tested but never runs

## Renamed a user-visible name (preflight `check_*`, CLI flag, YAML field, env var)

- `project_pending_changelog` memory — append an entry (`sieval-release` reads + clears)
- Hard-coded string assertions in `tests/`
- Non-archive `docs/`

## Grep of last resort

```sh
rg "<old-name-or-phrase>" scripts/ tests/ docs/ .pre-commit-config.yaml CONTRIBUTING.md
```

Every hit is a coupled site. Update in the same commit or justify the staleness.
