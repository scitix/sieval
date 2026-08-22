# Spider 1.0 & Spider 2.0-lite — Text-to-SQL for sieval

**Date:** 2026-08-22
**Status:** Design — awaiting review
**Scope:** Two new benchmarks, two datasets, two tasks, two vendored graders, one new
capability class (bounded SQL execution grading).

## 1. Why

sieval has no text-to-SQL task and no SQL-execution grading substrate. The
benchmark-coverage sheet lists LiveSQLBench but ships nothing in the family.
Spider 1.0 is the field's reference text-to-SQL set (2018, 1034 dev questions,
still the most-reported comparison point); Spider 2.0 is its enterprise-scale
successor (2024, 547 questions over real warehouses). Together they give the
delivery pipeline a structured-query capability axis it currently cannot measure
at all.

This is not a bounded change: nothing in `sieval/` executes SQL today, and the
two benchmarks stage ~930 MB of databases between them.

## 2. Decisions taken (confirmed with the requester)

| Decision | Choice | Consequence |
| --- | --- | --- |
| Spider 2.0 slice | **All 547**, BigQuery/Snowflake behind optional credentials | Full leaderboard parity when credentials exist; 412/547 fail without them |
| Spider 1.0 data | **sha256-pinned HF mirror** of `spider_data.zip` | Enables execution accuracy; third-party mirror, checksum-pinned |
| Spider 1.0 prompt | **Rajkumar et al. 2022** (`CREATE TABLE` + 3 example rows) | Comparable to published LLM-era Spider numbers |

## 3. Out of scope (explicit)

- **Test-suite accuracy** (`taoyds/test-suite-sql-eval`) — Spider's official metric
  since 2020, but a separate repo plus separately distilled databases. v1 ships
  Exact Set Match + Execution Accuracy. Named here so its absence is a decision,
  not an oversight.
- **`spider_0shot_base_gen`** — Rajkumar et al. used completion-style Codex prompts
  ending in a bare `SELECT`. A chat task cannot end mid-token (see §7.1). The
  faithful completion variant is a natural follow-up, not v1.
- **Spider 2.0-Snow** (547, Snowflake-only) — `spider2-lite`'s `sf_*` instances
  already exercise the Snowflake path.
- **Spider 2.0-DBT** (68, agentic) — already adapted under `harbor/`.
- **Spider 2.0 "oracle tables" mode** — upstream requires it be declared as such;
  it measures a different task.
- **Spider 1.0 train split** — the dataset exposes it; no task consumes it.

## 4. Verified upstream facts

Every number below was measured against the pinned artifacts, not read off a README.

### Spider 1.0

- Grader: `taoyds/spider` @ `b7b5b8c890cd30e35427348bb9eb8c6d1350ca7c`, **Apache-2.0**
  (vendorable). `evaluation.py` (868 lines) + `process_sql.py` (562 lines).
- Data: `spider_data.zip`, **205,800,266 bytes**,
  `sha256:00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b`,
  from `HAL-9001/spider-databases` @ `4a01bbac6520cd35b216db9e1724e5e1ada60aa4`
  (CC-BY-SA-4.0). 2625 members, 744 `.sqlite` files.
- **Authenticity confirmed**: `dev.json` has exactly 1034 rows; 20 distinct dev
  `db_id`s, all 20 with their `.sqlite` present; row 0 is the canonical
  *"How many singers do we have?" / `SELECT count(*) FROM singer`*.
- The official databases are Google-Drive-only, which is why a mirror is used.

### Spider 2.0-lite

- Grader + resources: `xlang-ai/Spider2` @ `cafb867313aab4e674652054198f383cf4018943`,
  **MIT** (vendorable).
- Backend split, derived from `instance_id` prefixes rather than the README's
  approximate table:

  | Backend | Prefixes | Count |
  | --- | --- | --- |
  | BigQuery | `bq` (180), `ga` (25) | **205** |
  | Snowflake | `sf_bq` (189), `sf` (18) | **207** |
  | Local SQLite | `local` (135) | **135** |
  | | | **547** |

- Rows (`spider2-lite.jsonl`) carry only `instance_id`, `db`, `question`,
  `external_knowledge`. The HF copy (`xlangai/spider2-lite`) is **stale** relative
  to GitHub — question text differs — so the pinned GitHub revision is the source.
- Grading config `evaluation_suite/gold/spider2lite_eval.jsonl`: 547 rows of
  `instance_id` / `condition_cols` / `ignore_order` / `toks`; 37 also carry `temporal`.
- Gold artifacts: ≥1000 `exec_result` CSVs, 256 gold SQL files, 69 external-knowledge
  documents. Too many to enumerate as individual `url:` sources — hence one archive.
- Archive: `356,977,771 bytes`,
  `sha256:1e3cbb6a0eb13d9a397a8a786d9cd9b06ba54df124b6a193dd52f2949580276b`.
  Uncompressed `spider2-lite/` alone is 909 MB (resource 649 MB + evaluation_suite 259 MB).
- Local databases: `xlangai/spider2-localdb` @ `c700cf8fe4064c745274870ba1c295f33610736a`,
  `sqlite.zip`, **573,418,120 bytes**,
  `sha256:d97fc0f3f6bc3f1a83548ad2b32c646bfc67433c23a0c499d3450796196b2776`.
  **Trap:** the archive interleaves `__MACOSX/._*.sqlite` resource forks with the
  ~40 real databases; extraction must filter them or half the "databases" are 4 KB
  AppleDouble stubs.
- Grading semantics (read from `evaluate_utils.py`, not inferred): execution-result
  comparison over pandas frames; absolute numeric tolerance `1e-2`; per-instance
  `condition_cols` selects which gold columns are checked; per-instance
  `ignore_order`; multi-gold via `compare_multi_pandas_table` (any gold matching
  scores 1). Column matching is directional — every gold column must find *some*
  matching predicted column, so extra predicted columns are tolerated.

### Loader constraint (measured)

`load_dataset("json", data_files="dev.json")` **fails outright**:

```text
ArrowInvalid: cannot mix list and non-list, non-null values
  — Conversion failed for column sql
```

Spider's `sql` field is a recursive parse tree whose `except`/`intersect`/`union`
slots are sometimes `null` and sometimes nested objects, which has no stable Arrow
schema. Dropping that one column yields a clean 1034-row, 6-column dataset
(`db_id`, `query`, `query_toks`, `query_toks_no_value`, `question`, `question_toks`).
Nothing is lost: the vendored `process_sql.get_sql` regenerates the parse from the
`query` string at grade time, which is what the grader does anyway.

## 5. Datasets

### `sieval/datasets/spider.py` — `SpiderDataset` / `SpiderDatasetSample`

- `source`: single `url:` to the pinned mirror archive; `checksums` as in §4.
- `license="CC-BY-SA-4.0"`, `deps_group=None` (stdlib `json`/`zipfile` only).
- `load()` extracts once into `<staged>/spider_data/`, guarded by a marker file so
  concurrent loads neither race nor re-extract 206 MB. The `url:` handler downloads
  but does not extract; `cmmlu.py` and `aa_lcr.py` are the in-repo zip precedents.
- Builds splits via `Dataset.from_list` over the projected rows (`sql` dropped, §4),
  not `load_dataset("json", ...)`, which cannot type the file.
- Exposes `db_dir` and `tables_json_path` properties for the task, mirroring
  `SciCodeDataset.h5_path`. Both `None` when the dataset was built from a preloaded
  dict (tests).
- Sample keeps upstream field names verbatim, per the datasets rule.

### `sieval/datasets/spider2_lite.py` — `Spider2LiteDataset` / `Spider2LiteDatasetSample`

- `source`: a 2-tuple — the pinned GitHub archive and the pinned `sqlite.zip`.
  Multi-source datasets are supported (`scicode.py` mixes pinned GitHub raw URLs
  with an HF mirror); basenames differ, so the staged-basename guard is satisfied.
- `load()` extracts only `Spider2-<sha>/spider2-lite/**` (909 MB, not the full 1.9 GB
  repo) plus `sqlite.zip` with `__MACOSX/` filtered, both marker-guarded.
- Rows are upstream's four fields, unmodified. The grading config
  (`condition_cols` / `ignore_order`) is **gold, not sample data**, and is read by the
  task from a `gold_dir` property rather than merged into the row schema — keeping
  the dataset a faithful mirror of `spider2-lite.jsonl`.
- Properties: `localdb_dir`, `gold_dir`, `documents_dir`, `db_schema_dir`.
- `license="MIT"`, `deps_group="spider2"` — the loader itself is stdlib, but the
  staged archives are inseparable from the grader's backends.

## 6. Vendored graders (`community/`)

`community/` mirrors upstream as closely as possible; every divergence is documented.

### `sieval/community/spider/`

`evaluation.py` + `process_sql.py`, byte-identical except for one unavoidable
change: upstream's `from process_sql import ...` is a flat top-level import that
cannot resolve inside a package, so it becomes `from .process_sql import ...`.
This is the established shape — `instruction_following_eval` carries the same
rewrite (`from . import instructions`). Apache-2.0 attribution preserved.

### `sieval/community/spider2/`

The comparison half of `evaluate_utils.py`, kept byte-identical **including** its
top-level `google.cloud.bigquery` and `snowflake.connector` imports. Deferring
those imports was considered and rejected: it is a real deviation from a mirror,
and it buys nothing, because the chosen scope (all 547) needs both client
libraries installed regardless. What stays optional is **credentials**, not
packages. MIT attribution preserved.

## 7. Tasks

### 7.1 `sieval/tasks/spider_0shot_gen.py` — `SpiderZeroShotGenTask`

- `eval_mode=GEN`, `n_shot=0`, `model_type="chat"`, `reference_kind="value"`
  (gold SQL is recorded and compared against).
- **Prompt** — Rajkumar et al. 2022 "Create Table + Select 3": a `CREATE TABLE`
  block per table in the sample's `db_id` (columns, types, primary and foreign
  keys), each followed by a commented 3-row `SELECT * LIMIT 3` sample, then the
  question. DDL and sample rows are read from the SQLite file itself.
- **Documented divergence**: upstream's prompt terminates in a bare `SELECT` for a
  completion model. A chat task cannot end mid-token, so the task asks for a fenced
  `sql` block instead. This goes in `reference_impl.notes` — it is the single reason
  a chat-mode number is not bit-comparable to the paper's Codex figures, and it is
  why the completion-faithful `_base_gen` variant is named in §3 rather than
  silently skipped.
- **Postprocess**: extract from a ```sql fence; fall back to the first statement.
  Empty extraction normalises to `None` so `extracted` reports the miss.
- **Feedback**: Exact Set Match via the vendored `Evaluator` (upstream's
  values-disabled default) plus Execution Accuracy via the hardened executor (§8).
- **Report**: `exact_match`, `execution_accuracy`; `score_key="execution_accuracy"`;
  `denominator_policy=DENOMINATOR_REQUESTED` (upstream divides by all examples, so a
  pipeline failure counts as wrong). Declared on every return path, empty-run guards
  included.

### 7.2 `sieval/tasks/spider2_lite_0shot_gen.py` — `Spider2LiteZeroShotGenTask`

- Same decorator shape; `deps_group="spider2"`.
- **Prompt**: upstream's single-call baseline — question, the DB schema for
  `sample["db"]`, and the external-knowledge document when `external_knowledge` is
  set. Dialect is stated per backend (BigQuery / Snowflake / SQLite), since the
  three expect different SQL.
- **Feedback**: route by `instance_id` prefix (`local` → SQLite, `bq`/`ga` →
  BigQuery, `sf_bq`/`sf` → Snowflake), execute, and compare the result frame against
  the gold CSVs with the vendored `compare_multi_pandas_table`, passing that
  instance's `condition_cols` and `ignore_order`.
- **Missing credentials raise** — they are never silently skipped. A run without
  BigQuery/Snowflake credentials fails 412 samples, and under
  `DENOMINATOR_REQUESTED` that caps the headline near 24.7%. This is the honest
  reading and follows "explicit over implicit", but it makes the headline
  unreadable on its own, so the report **must** also publish a per-backend
  breakdown (`execution_accuracy_local` / `_bigquery` / `_snowflake`) plus the
  attempted count per backend. The task docstring states this plainly.
- **Report**: `score_key="execution_accuracy"`, `denominator_policy=DENOMINATOR_REQUESTED`.

## 8. Execution safety — the load-bearing part

`sieval/tasks/CLAUDE.md` §"Fidelity stops at execution safety" governs here.
Upstream's `eval_exec_match` does this:

```python
conn = sqlite3.connect(db)          # read-write
cursor.execute(p_str)               # model-generated SQL
except:                             # bare
```

Read-write, unbounded, no timeout, bare `except`. A model can emit `DROP TABLE`,
`ATTACH DATABASE`, a `PRAGMA`, or a runaway cartesian join, and grading runs
synchronously on one shared event loop — so an unbounded query stalls the session,
not just the sample. Reproducing that path is out of the question.

The **unqualified** task therefore carries the hardened behaviour (this is the one
divergence that explicitly does *not* earn a `_fixed`):

- SQLite opened read-only via URI (`file:<path>?mode=ro&immutable=1`), so writes
  fail at the connection layer rather than being pattern-matched out of the SQL.
- Execution bounded by the house helper in `sieval/core/utils/offload.py`, which
  runs the call in a worker thread and raises `TimeoutError` on expiry — the event
  loop stays live.
- A row-count cap on fetch, so a pathological result set cannot exhaust memory.

Per the rule, shipping this `stable` owes three things, and the plan must produce
them rather than assert them:

1. **Safety, not repair** — upstream preserved everywhere safety does not object,
   including where upstream is wrong. A grader defect found along the way is a
   separate `_fixed` owing its own number; a large safety delta is evidence one got
   smuggled in.
2. **A quantified score impact** against upstream's actual behaviour on a stored run.
3. **Evidence no bound binds** on the pinned data — every gold query completes
   inside the timeout and under the row cap. A bound that truncates a real
   comparison is a scoring change wearing a safety label.

`theoremqa_kshot_base_gen` is the worked precedent.

## 9. Dependencies

Spider 1.0 declares **no extra at all** — loader and grader are stdlib
(`json`, `zipfile`, `sqlite3`), and an empty optional-dependency group would be a
declaration that means nothing. Both its dataset and task carry `deps_group=None`.

Spider 2.0 adds one group:

```toml
spider2 = [
  "pandas>=2.0",
  "duckdb>=1.0",
  "google-cloud-bigquery>=3.0",
  "snowflake-connector-python>=3.0",
]
```

`deps_group` preflight scans only `tasks/` + `datasets/` and is **blind to
`community/` imports**, so the vendored `evaluate_utils.py` imports must be
reconciled against this group by hand — a green preflight will not catch a gap here.

## 10. Testing

- `tests/unit/datasets/test_spider.py`, `test_spider2_lite.py` — loader shape,
  the `sql`-column projection, extraction idempotence, `__MACOSX/` filtering.
- `tests/unit/tasks/test_spider_0shot_gen.py`, `test_spider2_lite_0shot_gen.py` —
  prompt construction, SQL extraction (fenced / bare / unextractable), EM and EX
  against a small fixture database built in-test, report declarations.
- **Safety tests are not optional and must prove by deletion**: a `DROP TABLE`
  prediction is rejected by the read-only connection, and a deliberately slow query
  trips the timeout. Both must fail if the guard is removed — a test that passes
  with the guard deleted has tested nothing.
- Fixtures are built in-test; the 206 MB / 573 MB archives are never staged in CI.
- Golden tests call the production path, never a reimplementation of it.

## 11. Risks

| Risk | Assessment |
| --- | --- |
| **codeload zip byte-stability** | GitHub's generated archives are not contractually byte-stable, and a sha256 pin turns any regeneration into a hard failure. Accepted for now — it is the only single artifact carrying ≥1000 gold files. Recovery is a re-pin plus a CHANGELOG note; the durable fix is mirroring the `spider2-lite` subset ourselves. **This is the weakest point in the design.** |
| **Third-party mirror (Spider 1.0)** | Checksum-pinned and content-verified against the official row set (§4). Provenance noted in the dataset docstring. |
| **Footprint** | ~930 MB download, ~2.5 GB on disk after extraction. Inherent to "all 547". |
| **Snowflake availability** | Upstream recorded an evaluation-account suspension on 2026-08-12, and a password/MFA policy change affecting Python credentials. The Snowflake path may be unrunnable through no fault of ours; the per-backend breakdown makes that visible instead of silently depressing the headline. |
| **Prompt is not upstream-pinned** | Spider 1.0 predates LLM prompting; §7.1's divergence is recorded in `reference_impl.notes`. |

## 12. Sequencing

One plan, two phases, in this order — Spider 1.0 first even though Spider 2.0 is
the headline ask:

1. **Spider 1.0** — dataset, vendored grader, hardened executor, task, tests. This
   phase is where the execution-safety work in §8 is built and its three
   obligations discharged, against a single backend with no credentials in play.
2. **Spider 2.0-lite** — dataset, vendored comparison, three-backend routing, task,
   tests. Inherits a proven bounded-execution pattern instead of inventing one
   alongside cloud-credential handling.

The phases share no code by construction (§8's executor is per-benchmark, since
"do not abstract ahead of time — extract on coupling" argues against a shared SQL
layer before two real callers exist). If phase 2 reveals genuine coupling, that is
the moment to extract, not now.

## 13. Verification gate

`check_preflight.py` — `check_tasks`, `check_datasets`, `check_reference_kind`,
`check_report_declarations`, `check_task_shot_knobs`; `sync_meta_index.py --check`
and `sync_package_stubs.py --check` (hooks miss `sed`/`git apply`/rebase paths, and
stub drift has no CI enforcer); full `ruff check` and `ty check`; `pytest`.
