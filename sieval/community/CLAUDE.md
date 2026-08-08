# Community — Third-Party Evaluation Adaptations

## Purpose

This directory contains local adaptations of third-party evaluation tools (e.g. livecodebench, instruction_following_eval, simple_evals). These are wrappers around upstream implementations, not original code — with the narrow exception below.

## Requirements

* **Interface compatibility** — do not break callers. The rest of the codebase depends on stable entry points.
* **Upstream alignment** — match the official implementation as closely as possible.
* When modifying, document what differs from upstream and why.

## Not Required

* No mandatory test coverage.
* No mandatory internal code style enforcement (but keep it readable).
* License attribution must be preserved where required by upstream.

## First-Party Modules

`_sympy_guards.py` is original code, not a wrapper: it holds the execution guards
the `deepseek_math` and `ugmathbench` graders share. It lives here because both
hand model output to sympy under the same threat model, so a new escape route has
to close in both or one is left open — the coupling is to these two graders, not
to anything upstream.

The package-wide `ruff` / `mypy` exclusions (`pyproject.toml`) and the
`pre-commit` exclusion exist to keep *vendored* code byte-identical to upstream,
and they cover this file too — which is the wrong default for the module holding
a security boundary. Until those exclusions are narrowed to the vendored paths,
keep first-party modules here lint-clean and formatted by hand:

```bash
ruff check --config 'exclude=["vendor"]' sieval/community/_sympy_guards.py
ruff format --check --config 'exclude=["vendor"]' sieval/community/_sympy_guards.py
```

Do not add new original code here without the same coupling argument — a helper
with one caller belongs in that caller's module, and one shared by non-community
callers belongs in `sieval/core/utils/`.
