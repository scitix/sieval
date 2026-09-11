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

`bfcl_v3/_safe_eval.py` is the second, and with exactly **one** caller it shows
what "the same argument" means. That caller is a *vendored* file
(`bfcl_v3/parser.py`, whose `ast.BinOp` branch upstream resolves with `eval` over
model output), so inlining the guard would both enlarge a diff that exists to be
compared against upstream and put a security boundary inside a file that cannot
be linted. "One caller" is the rule for ordinary helpers; a boundary that has to
stay separately auditable is not one.

The package-wide `ruff` / `mypy` / `pre-commit` exclusions exist to keep vendored
code byte-identical and cover both files too, which is the wrong default for a
security boundary. Until they are narrowed, lint them by hand:

```
ruff check --config 'exclude=["vendor"]' sieval/community/_sympy_guards.py
ruff check --config 'exclude=["vendor"]' sieval/community/bfcl_v3/_safe_eval.py
```
