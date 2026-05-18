---
paths:
  - "sieval/**/*.py"
  - "tests/**/*.py"
---

# Code Review Calibration

- Calibrate severity — do NOT flag valid idioms as bugs
- Do NOT fix bugs or refactor unless explicitly asked
- When reviewing, grep for similar functionality in `sieval/core/utils/` and existing files before suggesting new helpers
