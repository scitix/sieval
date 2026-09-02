---
paths:
  - "sieval/tasks/**/*.py"
---

# Task Implementation Rules

## Naming & Model Type

- File naming must follow `<task>_<N>shot_<mode>[_<variant>].py` pattern (authoritative table in `sieval/tasks/CLAUDE.md`):
    - `_gen.py` → `model_type = "chat"`
    - `_base_gen.py` → `model_type = "gen"` (base model, uses GenModel)
    - `_ppl.py` → `model_type = "gen"` (perplexity, uses GenModel)
    - `_clp.py` → `model_type = "gen"` (conditional next-token log-prob, uses GenModel)
- Class naming: `<Benchmark><ShotType><Mode>[<Variant>]Task` — words for shot count (`ZeroShot`, `FewShot`)
- `ppl` vs `clp` distinction: see `sieval/tasks/CLAUDE.md`.

### Variants

An optional trailing segment lets two readings of one benchmark coexist as
separate registered tasks (full rationale in `sieval/tasks/CLAUDE.md`).

- The unqualified name means **what upstream measures, bugs included**. Never
  repurpose it for a local change.
- `_fixed` requires a **defect** in upstream's data or grader, not a preference,
  and owes both: every divergence in `reference_impl.notes`, and a **quantified**
  score impact.
- A variant may not spell a mode — `..._clp_gen.py` is rejected.
- A different **measurement regime** is a mode, not a variant.
- A fix to **problem text or reference answers** is a `datasets/` concern: a
  dataset variant applying a patch table over the same pinned revision, never a
  forked copy. See `sieval/datasets/CLAUDE.md`.
- Do not coin a new variant name speculatively.
- **Fidelity stops at execution safety.** Never reproduce a path that executes
  model output, escapes the run directory, or cannot be bounded — the
  **unqualified** task carries the hardened behaviour and needs no `_fixed`. It
  still owes a quantified score impact, upstream preserved everywhere safety
  does not object, and evidence no bound binds. See `sieval/tasks/CLAUDE.md`.

## Checklist for New Benchmarks

- Add benchmark-specific dependencies to `pyproject.toml` optional dependency groups (e.g., `[project.optional-dependencies.benchmark_name]`)
- New datasets must be downloadable via `sieval dataset download <name>` — verify the `source` field (`hf:` / `url:` / `local:`) resolves.
- Record the upstream **repeat/sampling protocol** in `reference_impl.notes` when it differs from this task's default `n` (matharena's runner default `--n 4`; simple-evals' `n_repeats`), and say how to match it. Nothing enforces this — it needs upstream knowledge.
- The PostToolUse hooks in `.claude/settings.json` regenerate stubs + `meta/index.json` on a Write/Edit under `sieval/tasks/` or `sieval/datasets/`, in the checkout that owns the file. They do not fire for `sed -i`, `git apply`, or a rebase, so `--check` both sync scripts before committing: CI fails on a stale `meta/index.json`, and stub drift has no enforcer at all.
- ruff and ty run from the same hooks but only report — run the full `ruff check` / `ty check` before committing.
- Declare `reference_kind` on `@sieval_task` **explicitly, on every task** — `"procedure"` when the ground truth is a test suite or a rubric, `"value"` otherwise. The decorator defaults to `"value"`, so an out-of-tree task is never forced to change; in-tree, spell it anyway, so the file says what it is instead of leaving the reader to know the default. Place it immediately before `reference_impl`. Enforced against the `build_judgement_record` call sites by `check_preflight.py --check check_reference_kind` (which checks the *effective* value, so an omission is verified rather than assumed); rationale in `sieval/tasks/CLAUDE.md` §"Task Metadata: `@sieval_task`". It is recorded, never branched on — a **missing** gold is not a `reference_kind` matter but a raise (see `.claude/rules/records.md`).
- `report()` must declare `score_key` (whenever it emits a `score`) and `denominator_policy` (`DENOMINATOR_REQUESTED` / `DENOMINATOR_JUDGED`) from `sieval.core.tasks.metrics` — on **every** return path, empty-run guards included, and `score_key` must name a key that report actually writes (a merged `metrics.py` key counts). Intervals bring four rules of their own:
    - **An interval and its population count are a pair** — a report writing `score_ci95` writes the count that interval is clustered on, and the reverse. That count is `n_problems` only when the headline is clustered on problems: a macro over strata owes `n_subjects` / `n_subsets` instead and must **not** write `n_problems` (`agieval` / `c_eval` / `cmmlu`).
    - **Every `<metric>_ci95` gets a `ci95_units` entry**, naming a population **count** (`n_…`) the same report writes — never another metric key.
    - **Emit through `metrics.py`** (`interval_metrics` / `metric_interval` / `sampling_report`), and fold two interval-bearing fragments with `merge_metrics`, not `|` — a plain merge replaces the whole `ci95_units` map.
    - **One number under two key names** rides along as `aliases=(...)` on the same call, never as a second call. A metric that is a **deterministic function** of another gets no interval at all.

    Enforced by `check_preflight.py --check check_report_declarations`, which sees only that the map exists; completeness is checked at report-write time instead. The reason for that split, and the rest of the rationale, is in `sieval/tasks/CLAUDE.md` §"Report Declarations".
- **Spell a metric key the way the benchmark publishes it** (`wikisql_0shot_gen`'s `ex_accuracy` / `lf_accuracy` are `evaluate.py`'s own keys), rather than borrowing a sibling's word for a measurement that only looks like the same one. State a real correspondence in `reference_impl.notes`; do not encode it by renaming. A key that is genuinely the same count across tasks (`n_execution_errors`) does keep one name. Nothing enforces this — it turns on what two upstreams mean; rationale in `sieval/tasks/CLAUDE.md` §"Metric key names follow upstream, not a house word".
- Run `python scripts/check_preflight.py --check check_tasks` to verify naming, tags, and imports

## Code Quality

- Use `strict=True` in `zip()` when lengths are guaranteed to match
- Must not modify `core/` — check `sieval/core/utils/` for existing helpers first
- **A grading call site catches `TimeoutError`, never `Exception`.** A grade that
  could not be computed *in time* is a wrong answer — the prediction is a shape
  the grader cannot bound, which is the model's problem — so it is swallowed and
  scored `False`. Every *other* exception propagates: a grader that is **broken
  rather than slow** (a dead worker, an optional dependency absent from the
  environment) must not be indistinguishable from a model that answered wrongly.
  Swallowing it produces a low score on a run whose `fails` is 0 and whose logs
  are gone as soon as the run is. Propagating costs nothing and buys the signal:
  `feedback` raising goes straight to `to_failed(reason="exception::<class>")` —
  no re-inference, no budget burned — and under `DENOMINATOR_REQUESTED` a fail is
  already charged as wrong, so **the headline does not move**. This is why no
  `n_grade_errors` metric is needed: `fails` plus the recorded reason already is
  one. Enforced tree-wide by an AST survey of every `run_cpu_bound` call site in
  `tests/unit/tasks/test_grading_call_site_convention.py` — a hand-kept list of
  members is what let this drift in the first place, so the check reads the
  source rather than a registry and covers tasks whose `feedback` lives in a
  shared base. The *behaviour* behind it (a timeout still scores wrong, every
  other class propagates, and the headline does not move either way) is asserted
  over the pass@k math family in `tests/unit/tasks/test_math_pass_at_k_family.py`.
    - The one thing that *does* move is a published interval: it is estimated
      over the units that came back while scaled to the requested denominator, so
      dropping one shifts the bound (in either direction), and a survivor set that
      is uniformly right or uniformly wrong drops `<metric>_ci95` / `n_problems`
      entirely — `wilson_interval` needs `0 < p < 1`. Pre-existing
      `_clustered_interval` semantics, not a new one.
    - A task whose report declares `DENOMINATOR_JUDGED` is the exception and owes
      its own measurement first: there a fail is *excluded* from the denominator,
      so moving a sample into `fails` does change the score
      (`theoremqa_kshot_base_gen` is the only such member *with a grading call
      site* today — 19 tasks declare the policy, it is the only one that grades
      through `run_cpu_bound`).
    - **The rule reaches only what the call site can see.** It buys the signal
      when the grader lets errors through; a grader that catches `Exception`
      *itself* and returns a verdict anyway defeats it from below, and narrowing
      the `except` cannot tell. `imo_answer_bench_0shot_gen` is the live case:
      its vendored `verify_math_answer` falls back to string equality, so a
      broken LaTeX backend scores every expression answer wrong with `fails`
      still 0. Upstream fidelity means that fallback stays, so the task probes
      the grader once per run instead (`_ensure_grader_healthy`) and raises on a
      definite negative. When vendoring a grader, check whether it swallows
      before relying on this rule; the twelve siblings are safe only because
      `_math_verify.verify_answer` has no handler of its own.
    - Propagating is the right default even though `exception::<class>` is
      **retriable** (`ERROR_REASONS_NON_RETRIABLE`, `core/tasks/consts.py`), so a
      deterministic breakage is re-graded once per resume until `max_retries`
      runs out. Do not reach for `NonRetriableSampleError` to avoid that: it
      declares the outcome fixed in the *sample's own input*, which an
      environment fault is not, and mislabelling a transient failure permanently
      fails a sample a retry would have recovered. The waste is bounded — under
      the default `record_each_stage=True` a feedback failure rolls back to
      `POSTPROCESSED`, so it re-grades without re-inferring.

## Tags — Anomaly Detection

- New tasks: use `@sieval_task(...)` — `cls.tags` is synthesized from `eval_mode` + `n_shot`; do not set manually. See `sieval/tasks/CLAUDE.md` §"Task Metadata: `@sieval_task`".
- Legacy (unmigrated) tasks still declare `tags: ClassVar[set[str]]` with vocabulary `gen / ppl / base / zero_shot / few_shot / llm_judge`.

## Data Flow — Async & Concurrency

- Understand the framework's staged execution data flow before implementing.
- All intermediate state must flow through the framework's persistence layer (record/shard storage) — do NOT use external files, temp caches, or module-level mutable state to pass data between stages.
- Never introduce shared mutable state without proper locking.
