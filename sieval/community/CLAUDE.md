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

`_sympy_guards.py` is original code, not a wrapper: the execution guards
`deepseek_math` and `ugmathbench` share. Holding it *outside* both serves
upstream alignment rather than working against it — the vendored files keep only
a small annotated divergence each, instead of carrying a copy of the guards
inline where it would swamp a diff against upstream. Do not add more original
code here without the same argument; a helper with one caller belongs in that
caller's module.

The package-wide `ruff` / `mypy` / `pre-commit` exclusions exist to keep vendored
code byte-identical and cover this file too, which is the wrong default for a
security boundary. Until they are narrowed, lint it by hand:
`ruff check --config 'exclude=["vendor"]' sieval/community/_sympy_guards.py`.
