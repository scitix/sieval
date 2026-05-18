# SiEval Test Suite

**~1200 tests** | **95%+ core coverage** | pytest + anyio

---

## Structure

**Convention:** `tests/unit/` mirrors `sieval/` — e.g. `sieval/core/runners/foo.py` → `tests/unit/core/runners/test_foo.py`. Scripts with non-trivial logic (`scripts/*.py`) go in `tests/unit/scripts/`. Non-`sieval/` top-level artifact dirs get their own sibling under `tests/` that mirrors them directly — e.g. `leaderboards/alignment/<tr-slug>/<stage>.md` → `tests/leaderboards/test_*.py`.

```text
tests/
├── conftest.py                  # Shared mock infrastructure (all layers)
├── unit/                        # Unit tests — mirrors sieval/ structure
│   ├── conftest.py
│   ├── cli/                     # sieval/cli/
│   │   ├── eval/                 # sieval/cli/eval/
│   │   │   └── test_session.py   # EvalSession (YAML config → eval) pure functions & E2E
│   │   ├── infer/                # sieval/cli/infer/
│   │   │   ├── test_display.py   # Infer display formatting
│   │   │   ├── test_lifecycle.py # Infer handle I/O, launch, cleanup
│   │   │   └── test_resolve.py   # YAML-mode recipe auto-resolution (decision matrix)
│   │   ├── test_main.py          # CLI entry point (command registration)
│   │   ├── test_run.py           # CLI run command (orchestration)
│   │   ├── test_eval.py          # CLI eval command (integration)
│   │   ├── test_output.py        # CLI output helpers
│   │   └── test_validation.py    # eval --dry-run config pre-validation
│   ├── core/                    # sieval/core/
│   │   ├── models/               # ChatModel, GenModel, model derivation
│   │   │   ├── test_chat_model.py
│   │   │   ├── test_gen_model.py
│   │   │   ├── test_model.py
│   │   │   └── test_model_derivation.py
│   │   ├── runners/
│   │   │   ├── test_runner.py            # TaskRunner E2E (mock Task → report)
│   │   │   └── test_multi_runner.py      # MultiTaskRunner behavior
│   │   ├── tasks/
│   │   │   ├── loader/           # TaskLoader sub-package
│   │   │   │   ├── conftest.py
│   │   │   │   ├── test_parsing.py
│   │   │   │   ├── test_manifest.py
│   │   │   │   ├── test_retries.py
│   │   │   │   ├── test_corruption.py
│   │   │   │   ├── test_integration.py
│   │   │   │   └── test_cross_stage.py
│   │   │   ├── test_saver.py
│   │   │   ├── test_context.py
│   │   │   ├── test_profiler.py
│   │   │   ├── test_progress.py
│   │   │   ├── test_concurrency.py
│   │   │   ├── test_anomaly.py
│   │   │   ├── test_consts.py
│   │   │   └── test_task.py
│   │   ├── utils/
│   │   │   ├── test_concurrency.py
│   │   │   ├── test_hf.py
│   │   │   ├── test_logging.py
│   │   │   ├── test_meta.py
│   │   │   ├── test_ppl.py
│   │   │   ├── test_serialization.py
│   │   │   └── test_texts.py
│   │   └── test_datasets.py
│   ├── infer/                   # sieval/infer/
│   │   ├── test_config.py       # InferConfig, InferHandle, InferStatus
│   │   ├── test_deployer.py     # LocalDeployer launch orchestration
│   │   ├── test_introspect.py   # Checkpoint introspection and GPU detection
│   │   ├── test_recipes.py      # Recipe loading and merging
│   │   ├── test_translator.py   # Config translation
│   │   ├── test_basic_env.py    # Basic environment checks
│   │   ├── test_process.py      # Process management
│   │   └── topology/            # DeploymentPlan, resolver, validator
│   │       ├── test_models.py
│   │       ├── test_resolver.py
│   │       └── test_validator.py
│   ├── scripts/                 # scripts/*.py with non-trivial logic
│   │   ├── test_check_layer_imports.py
│   │   └── test_check_preflight.py
│   └── test_lazy_exports.py
├── integration/                 # Integration tests — TaskRunner + mock infra
│   ├── resume/                  # Resume sub-package (basic + advanced scenarios)
│   │   ├── conftest.py
│   │   ├── test_basic.py        # Partial completion, failed retry
│   │   └── test_advanced.py     # Cross-stage, iteration bounds, max_retries
│   ├── test_runner_edge_cases.py  # Fast resume, early-exit hydration, progress dump
│   ├── test_metadata_flow.py      # Implicit/explicit metadata, disk persistence
│   ├── test_multi_task.py
│   ├── test_lifecycle.py
│   ├── test_single_turn_eval.py
│   ├── test_pass_at_k.py
│   └── test_llm_judge.py
├── acceptance/                  # Release gates — must pass before any release
│   ├── alignment/               # Task implementation alignment records (YAML, per sieval task)
│   │   └── README.md            # Record schema + filling guide
│   └── performance/
│       ├── baselines.json       # Regression baselines (_tolerance controls allowed degradation)
│       └── test_performance_acceptance.py  # 6 scenarios + regression detection
├── performance/                 # Diagnostic benchmarks — run on-demand, informational
│   ├── test_concurrency_scaling.py
│   ├── test_dataset_loading.py
│   ├── test_io_overhead.py
│   ├── test_memory_usage.py
│   ├── test_pipeline_throughput.py
│   ├── test_resume_speed.py
│   └── test_serialization.py
└── leaderboards/                # Mirrors repo-root leaderboards/ — static artifact schema
```

---

## Running Tests

```bash
# All configured tests (default excludes stress via pytest addopts)
python -m pytest -v

# Unit + integration (with coverage, ≥95% required)
python -m pytest tests/unit/ tests/integration/ --cov --cov-fail-under=95 -v

# Unit + integration (quick, no coverage)
python -m pytest tests/unit/ tests/integration/ -q

# Acceptance tests (release gate — no coverage tracer, it skews latency)
python -m pytest tests/acceptance/ -v -s

# Acceptance tests + write benchmark_summary.json to a custom directory
SIEVAL_BENCHMARK_ARTIFACT_DIR=./outputs/benchmarks \
python -m pytest tests/acceptance/ -v -s

# Performance diagnostic benchmarks (default excludes stress)
python -m pytest tests/performance/ -v

# Exclude stress tests
python -m pytest tests/performance/ -m "not stress" -v

# Run only stress tests (intentional profiling)
python -m pytest tests/performance/ -m stress -v

# Single file
python -m pytest tests/integration/resume/test_advanced.py -v

# Single class or method
python -m pytest tests/integration/resume/test_basic.py::TestResumePartialCompletion -v
```

## Mock Infrastructure (`tests/conftest.py`)

All shared test infrastructure lives here — available to every test layer without any explicit import.

### Unit / Integration mocks

| Class | Description |
| --- | --- |
| `MockDataset(samples)` | Dataset from a list of dicts |
| `MockChatModel(answers={...})` | Deterministic chat model |
| `MockGenModel(logprob_scores={...})` | Deterministic gen model (alogprobs) |
| `MockJudgeModel(verdict="yes")` | LLM-as-judge mock |
| `MockCountingChatModel(answers={...})` | `MockChatModel` that counts `_agenerate_impl` calls |
| `MockAlwaysFailModel()` | Always raises an exception |
| `MockFailingChatModel(fail_count=1)` | Fails N times then succeeds |
| `MockSelectiveFailModel(fail_samples={...})` | Fails on first call for specific prompts |
| `make_config(tmp_path, **overrides)` | `TaskRunnerConfig` for unit/integration tests |

### Performance / Acceptance infrastructure

| Class / Function | Description |
| --- | --- |
| `LatencyMockChatModel(latency_s, output_size, latency_jitter)` | Configurable-latency mock for benchmarks |
| `BenchmarkTask` / `MultiIterBenchmarkTask` | Standard 4-stage tasks for benchmarks |
| `IOProfile` | I/O pattern configuration |
| `PerfTimer` / `MemoryTracker` | Timing and memory measurement utilities |
| `make_large_dataset(n, payload_size)` | Generate large in-memory dataset |
| `make_perf_config(tmp_path, **overrides)` | `TaskRunnerConfig` for performance tests |
| `write_completed_samples(root, n_completed)` | Write FINAL contexts to disk for resume tests |

---

## Writing Tests

### Unit Tests

Test a single module in isolation. Use `TaskSaver`/`TaskLoader` directly with `tmp_path`.

```python
@pytest.mark.anyio
async def test_something(self, tmp_path):
    root = tmp_path / "test_run"
    ctx = TaskContext(sample_id=0, raw_sample={"q": "test"}, stage=TaskStage.FINAL)
    saver = TaskSaver(root_dir=root, ...)
    # ...
```

### Integration Tests

Test end-to-end flows through `TaskRunner`.

```python
@pytest.mark.anyio
async def test_something(self, tmp_path):
    dataset = MockDataset([{"question": "Q1", "answer": "A1"}])
    model = MockChatModel(answers={"Q1": "A1"})
    task = MyTask(dataset=dataset, model=model, name="test")
    config = make_config(tmp_path)

    runner = TaskRunner(task, config)
    report = await runner.arun()

    assert report["accuracy"] == 1.0
```

### Discriminating Power

Assertions must detect the actual feature being tested. For disk persistence tests, always use a fresh `TaskLoader` instance rather than reading from `runner._contexts` (in-memory):

```python
# GOOD: loads from disk
loader = TaskLoader(task=task, root_dir=runner.root_dir)
contexts = await loader.load_initial_state()
await loader.hydrate(contexts, set(), include_stages={TaskStage.FINAL})
assert contexts[0].infer_result is not None

# BAD: reads from memory (passes even if persistence is broken)
assert runner._contexts[0].infer_result is not None
```

### Async Tests

All async tests use `@pytest.mark.anyio`. Do **not** use `@pytest.mark.asyncio`.

```python
@pytest.mark.anyio
async def test_my_async_test(self, tmp_path):
    ...
```

---

## Performance Regression Tracking

`tests/acceptance/performance/baselines.json` holds minimum acceptable values per scenario.
`TestRegressionDetection` unit-tests the `_check_regressions()` logic in isolation.

The acceptance test (`TestBenchmarkSummary.test_benchmark_scenarios`) fails if any
scenario degrades beyond the configured tolerance (`_tolerance` in `baselines.json`,
e.g. `0.9` allows up to 10% degradation).

By default benchmark artifacts are written to pytest's `tmp_path`; set
`SIEVAL_BENCHMARK_ARTIFACT_DIR` to keep them in a stable output directory.

To update baselines after a genuine performance improvement, edit `baselines.json` directly.

---

## Mutation Testing

Mutation tests verify that unit tests have real discriminating power — they catch bugs, not just run code.

**Tool:** [mutmut](https://mutmut.readthedocs.io/) (`mutmut>=3.5.0` in the `test` dependency group)

**Scope** (configured in `pyproject.toml`):

| Config key | Value |
| --- | --- |
| `paths_to_mutate` | `sieval/core` |
| `tests_dir` | `tests/unit` |
| `exclude` | `sieval/core/**/__init__.py` |
| `also_copy` | `sieval/community`, `sieval/datasets`, `sieval/tasks`, `sieval/infer`, `sieval/probe`, `sieval/cli`, `sieval/__main__.py` |

Mutations are applied only to `sieval/core`; the rest of the package is copied into the sandbox so imports resolve correctly.

```bash
# Run all mutations (slow — runs the full unit suite per mutant)
mutmut run

# Show results summary
mutmut results

# Show surviving mutants (the ones your tests missed)
mutmut show

# Show a specific surviving mutant by ID
mutmut show <id>

# Apply a surviving mutant to disk for manual inspection
mutmut apply <id>

# Restore original source after applying a mutant
mutmut restore

# HTML report (written to html/ by default)
mutmut html
```

The mutation score (killed / (killed + survived)) must stay **≥70%**. Surviving mutants indicate gaps in test coverage — add assertions that kill them. Do **not** weaken tests to make mutants "pass".

---

## Test Quality Rules

- Assertions must have **discriminating power** — if the test passes whether or not the feature works, it is useless
- When a test fails, investigate the code first; only adjust the test if the original expectation was wrong
- Snapshot/dependency tests: verify that dependency loading actually fills in earlier stages from disk, not from in-memory state already computed during the run
