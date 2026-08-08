---
paths:
  - "sieval/tasks/**/*.py"
  - "sieval/core/tasks/**/*.py"
---

# Stage-Output Record Protocol (opt-in)

Per-stage record types so a sample's answer / ground truth / verdict is readable without knowing which task produced it. **Schema, serialization rules and builder contracts live in `sieval/core/tasks/records.py`** — authoritative, and deliberately not repeated here. This file is the vocabulary and the traps. Opt-in per task; legacy shapes keep working, so migrate deliberately.

Scoped to `sieval/tasks/` *and* `sieval/core/tasks/` because the contract has two
parties: tasks write these records, and the runner and profiler read them back.
A rule visible to only one side is how `grader_output` gets renamed.

Records are named by **content**, not by the stage that emits them — a shard line reads `prediction` / `judgement`, not `postprocess` / `feedback`:

| stage | record |
| --- | --- |
| `preprocess` | `PromptRecord` |
| `infer` | `ModelOutput` — already uniform, deliberately **not** wrapped |
| `postprocess` | `PredictionRecord` |
| `feedback` | `JudgementRecord` |

Vocabulary — these denote **different layers**, keep them distinct:

- **judgement** — the verdict record, mechanism-agnostic: string-compare, math-verify, test-suite *or* LLM verdicts all produce one.
- **prediction** — the model's extracted answer. `None` means "could not extract" — never `""` or `-1`. **Read it with `.get("prediction")`, never `["prediction"]`**: a `None` value is dropped by serialization, so on resume the key is *absent* and `[]` raises `KeyError` for exactly the samples whose extraction failed — on a fresh run the same line is fine, which is why in-memory tests never see it. `extracted` is the durable companion flag. Enforced by `scripts/check_preflight.py --check check_record_key_access`; neither `ty` nor `mypy --strict` reports it.
- **reference** — ground truth; `None` when it is a *procedure* (test suite, rubric).
- **correct** / **score** — the headline verdict. `correct` is the only axis comparable across tasks.
- **metrics** — every metric measured, by name. A task with *co-equal* metrics (IFEval strict + loose, HellaSwag `acc` + `acc_norm`) records them all here and derives the headline from them; a metric parked in `extra` is hidden from any reader that doesn't already know the task.
- **grade** — an LLM autorater's categorical output (CORRECT / INCORRECT / NOT_ATTEMPTED). A judgement *contains* a grade — grade sits **below** judgement, not beside it.
- **grader** — the LLM actor (the `grader` task arg / model). Not every judgement has one; persist its whole `ModelOutput` in `extra` rather than hand-picked fields, under exactly the key **`grader_output`** (`records.GRADER_OUTPUT_KEY`). The name is load-bearing, not cosmetic: the runner reads it back to route grader spend into `profile.json`, since a `feedback` stage returning a bare record has no `ModelOutput` for the profiler to derive a call from. Spell it differently and the grader's tokens are on disk but missing from the profile.
- **judge** — HLE only, upstream's synonym with its own `parse_judge` contract; do not introduce it in new tasks.
- **extra** — mechanism detail, *not* metrics: a grader's `ModelOutput`, a code runner's failure message, per-constraint results.

Constructors are `build_*` (matching `build_model_call_meta` / `build_stage_meta`); structural sniffs are `is_*`. Records are returned **bare**, never wrapped in `TaskStageOutput`.
