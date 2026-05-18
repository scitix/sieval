# Community — Third-Party Evaluation Adaptations

## Purpose

This directory contains local adaptations of third-party evaluation tools (e.g. livecodebench, instruction_following_eval, simple_evals). These are wrappers around upstream implementations, not original code.

## Requirements

* **Interface compatibility** — do not break callers. The rest of the codebase depends on stable entry points.
* **Upstream alignment** — match the official implementation as closely as possible.
* When modifying, document what differs from upstream and why.

## Not Required

* No mandatory test coverage.
* No mandatory internal code style enforcement (but keep it readable).
* License attribution must be preserved where required by upstream.
