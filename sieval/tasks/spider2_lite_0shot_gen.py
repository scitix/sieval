"""Spider 2.0-lite — 0-shot generative text-to-SQL over real warehouses.

Spider 2.0 (Lei et al., ICLR 2025) is Spider's enterprise-scale successor. The
**lite** setting is its single-call text-to-SQL reading: 547 questions, no agent
loop, against schemas with hundreds of columns and questions that routinely need
an external document to answer. Scoring is execution-result comparison against
gold result frames upstream ships for all 547.

**Three engines, and the ``instance_id`` prefix is what picks one**: ``bq``/``ga``
BigQuery (205), ``sf_bq``/``sf`` Snowflake (207), ``local`` SQLite (135). Only
the 135 local questions run with no credentials.

**A credential-less run caps near 24.7%, by design.** ``denominator_policy`` is
``DENOMINATOR_REQUESTED``, so the 412 cloud questions count as wrong when they
cannot be asked. That is the honest reading — a benchmark you cannot run is not
a benchmark you passed — but it makes the headline unreadable on its own, so the
report always publishes a **per-backend breakdown** (``execution_accuracy_local``
/ ``_bigquery`` / ``_snowflake``, each with its own ``n_*`` and
``n_missing_credentials_*``). Read those before reading ``score``.

**Comparison is upstream's, from the right copy.** The repo ships two
``compare_pandas_table`` implementations; the live one in ``evaluate.py`` carries
the 2025-10-29 accuracy fix (NaN normalised to 0, an early break, an empty-gold
guard) and the one in ``evaluate_utils.py`` does not. This task uses the former,
vendored byte-identical, with each instance's own ``condition_cols`` and
``ignore_order`` from upstream's ``spider2lite_eval.jsonl``, and it branches on
upstream's own ``is_single``.

Note what that flag actually is, because the name misleads: ``resolve_gold_paths``
returns ``is_single=True`` only when an **exact** ``<instance_id>.csv`` exists,
which is true of **3 of the 547**. It is a filename shape, not a gold count —
104 instances have exactly one gold file and still take the
``compare_multi_pandas_table`` path. The 1,544 gold CSVs do mean most questions
accept more than one answer shape (440 of 547 have two or more), but that is a
separate fact from which branch runs.

**The prediction reaches the comparison through a CSV.** Upstream writes the
result frame out and reads it back before comparing, and that round trip is part
of the metric rather than plumbing: it is what re-infers dtypes, so a SQLite
column typed ``text`` holding ``"3"`` arrives as an integer and a BigQuery
``DATE`` arrives as the string the gold CSV also holds. Comparing the frame the
driver returned instead flips real verdicts in both directions.

**Snowflake is sieval's own, because upstream lite no longer has one.**
``evaluate_single_sql_instance`` routes BigQuery and SQLite and sends every other
prefix to "Unsupported instance id prefix" — 207 of 547 unscoreable by upstream
even though gold ships for them. A Snowflake number here therefore has **no
upstream lite counterpart**; the comparable published setting is Spider 2.0-Snow,
same questions, same warehouse. See ``sieval.tasks._spider2_backends``.

**Execution safety.** The local engine runs model SQL, so it carries the same
hardening as Spider 1.0 — read-only immutable connection, ATTACH/DETACH denied,
progress-handler deadline, row cap — from the same module, ``_sqlite_exec``,
rather than a second copy. Upstream instead copies the whole database into memory
per query; that stops writes but not runaway queries, and the largest local
database is 372 MB, which a concurrent grader cannot copy per call. The remote
engines are bounded (timeout + row cap) rather than hardened: the query runs on
someone else's server under their permissions.

**The prompt is bounded too, for a different reason.** The cloud-schema block
comes from the shipped resource tree, and that tree is not uniform. The 412
cloud questions run against 128 distinct databases, and rendering one of them
whole can be enormous: ``ga360`` produces 2,894,566 characters across its daily
partition tables, ``fec`` 13.5 M, and 56 of the 412 questions sit on a database
whose ``DDL.csv`` alone exceeds a megabyte. That is not a low score, it is a
request no endpoint accepts — so the block stops at :data:`MAX_SCHEMA_CHARS`, on
a statement boundary, and says how many tables it left out. 83 of the 412 cloud
questions are truncated; no local question is, since the largest local schema is
about 7.5 kB.

Target: there is **no comparable published number**, and that is a property of
the leaderboard rather than of this run. Upstream publishes one aggregate over
all 547 with no per-engine breakdown, so the BigQuery- or SQLite-only score a
credential-limited host can produce has nothing to be compared against. The
board is also almost entirely agentic (schema linking, multi-turn, execution
feedback, topping 76%); the nearest single-call entries are prompting frameworks
at 1.5–5.7%. A 0-shot single call is a different setting from all of them. This
is why the task ships ``experimental``: a faithful port with no reachable
anchor, not a port awaiting one run.

Measured, on the real staged archives:

* **All 135 local questions run end to end** — prompt, execution, comparison,
  report — with 0 pipeline failures, so ``n_local`` is 135 rather than a subset.
* **24 of 24 verdicts agree with upstream's own evaluator.** Every local
  instance for which upstream ships gold SQL was graded by both
  ``evaluate_single_sql_instance`` and this task's path; they agree on all of
  them, the 8 that score 0 included (upstream's ``gold/sql`` is stale against
  its own ``gold/exec_result``, and both harnesses call those wrong identically).
* **All 128 distinct cloud databases render a schema block** within
  :data:`MAX_SCHEMA_CHARS`, so no cloud question fails at prompt time.

Not measured: whether a *model* can answer these questions, and the BigQuery and
Snowflake engines, which need credentials this environment does not have. The
runs above script the model — upstream's gold SQL where it ships one — so they
exercise every stage except inference quality.

References:

* Paper: <https://arxiv.org/abs/2411.07763>
* Harness: <https://github.com/xlang-ai/Spider2>

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import csv
import json
import os
from functools import cache
from pathlib import Path
from typing import override

from loguru import logger

from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    JudgementRecord,
    PredictionRecord,
    PromptRecord,
    ReferenceImpl,
    RolloutJudgement,
    Task,
    build_judgement_record,
    build_prediction_record,
    build_prompt_record,
    build_rollout_judgement,
    sieval_task,
)
from sieval.core.tasks.metrics import (
    DENOMINATOR_FIELD,
    DENOMINATOR_REQUESTED,
    SCORE_KEY_FIELD,
    health_metrics,
)
from sieval.core.utils.offload import run_cpu_bound
from sieval.datasets import Spider2LiteDatasetSample
from sieval.datasets.spider2_lite import backend_for

from ._spider2_backends import caller_timeout, execute, missing_credentials
from ._sqlite_exec import open_readonly

#: Engines, in report order. Local first: it is the subset most runs can score.
BACKENDS = ("local", "bigquery", "snowflake")
#: `backend_for` returns the engine; the report groups under these names.
_BACKEND_LABEL = {"sqlite": "local", "bigquery": "bigquery", "snowflake": "snowflake"}

_DIALECT = {
    "sqlite": "SQLite",
    "bigquery": "BigQuery Standard SQL",
    "snowflake": "Snowflake SQL",
}

#: Character budget for one prompt's schema block. Bounds a *first-party* choice
#: rather than diverging from an upstream one: upstream lite's baselines build
#: their own prompt, so nothing published depends on this rendering. It exists
#: because the shipped schema tree is not uniform — ``ga360`` renders 2.9 M
#: characters (~724k tokens) across its daily partition tables — and an
#: unbounded block is a request that fails rather than a question that scores.
#: 200k characters is roughly 50k tokens, leaving a 128k-context model room for
#: the external-knowledge document and its own answer.
MAX_SCHEMA_CHARS = 200_000

#: Field cap for reading a shipped ``DDL.csv``. ``csv`` defaults to 131,072
#: characters per field and one database exceeds it: ``bigquery`` /
#: ``pancancer_atlas_2`` holds a single ``ddl`` value longer than that, so
#: ``DictReader`` raises ``_csv.Error: field larger than field limit`` and its
#: two questions (``bq151``, ``bq161``) die in ``preprocess`` rather than
#: scoring. Not ``sys.maxsize``: the cap is a C long and a value it cannot hold
#: raises ``OverflowError``, so this is the largest signed 32-bit value, which
#: every platform accepts and no shipped field comes near.
_CSV_FIELD_LIMIT = 2**31 - 1


@sieval_task(
    name="spider2_lite_0shot_gen",
    display_name="Spider 2.0-lite (0-shot, generative)",
    description="Enterprise text-to-SQL over BigQuery, Snowflake and SQLite; 547 questions.",  # noqa: E501
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "text-to-sql", "code-exec", "enterprise"),
    model_type="chat",
    status="experimental",
    deps_group="spider2",
    reference_kind="value",
    reference_impl=ReferenceImpl(
        source="xlang-ai/Spider2",
        url="https://github.com/xlang-ai/Spider2/tree/cafb867313aab4e674652054198f383cf4018943/spider2-lite/",
        notes=(
            "Comparison vendored byte-identical from evaluation_suite/evaluate.py "
            "(NOT evaluate_utils.py — the repo ships two compare_pandas_table "
            "copies and only evaluate.py's carries the 2025-10-29 fix: NaN "
            "normalised to 0, early break, empty-gold guard). Per-instance "
            "condition_cols/ignore_order from spider2lite_eval.jsonl, and "
            "upstream's is_single branch over 1,544 gold CSVs. is_single is a "
            "filename shape, not a gold count: resolve_gold_paths returns True "
            "only when an exact <instance_id>.csv exists, which is 3 of 547, so "
            "544 instances — including 104 that have exactly one gold file — go "
            "to compare_multi_pandas_table. The "
            "prediction is written to CSV and read back before comparing, as "
            "upstream does — that round trip re-infers dtypes and is part of the "
            "metric. Three of the 1,544 golds are header-only (local275_a, "
            "sf001_b, and sf_bq411_b, which holds upstream's own error string "
            "rather than a result — a corrupt gold, though its _a/_c siblings "
            "are valid); none is a bq/ga instance, which is the only prefix "
            "whose upstream path scores an empty result 0 without comparing, so "
            "not implementing that guard cannot change a verdict on this data. "
            "It does cost one free point on the first-party Snowflake path: an "
            "empty result matches sf_bq411_b's zero rows, so sf_bq411 scores 1 "
            "for a query returning nothing. 1 of 207, kept rather than "
            "special-cased because dropping an empty gold would itself be a "
            "divergence in the comparison. "
            "SQL extraction is upstream's extract_sql_query. DIVERGENCE — "
            "Snowflake execution is first-party: upstream's current lite "
            "evaluator routes only bq/ga and "
            "local and rejects every other prefix as 'Unsupported instance id "
            "prefix', leaving 207 of 547 unscoreable despite gold shipping for "
            "all 547; the stale evaluate_utils.get_snowflake_sql_result was not "
            "vendored because its sibling comparison is superseded. Snowflake "
            "numbers have no upstream lite counterpart — compare against Spider "
            "2.0-Snow. DIVERGENCE — the local engine is hardened for execution "
            "safety (read-only immutable connection, ATTACH/DETACH denied, "
            "progress-handler deadline, 100k-row cap) where upstream copies the "
            "whole database into memory per query; that stops writes but not "
            "runaway queries, and the largest local database is 372 MB. Remote "
            "engines are bounded (90s timeout + row cap), not hardened. "
            "Local-schema prompts introspect the .sqlite file; cloud-schema "
            "prompts read upstream's resource/databases tree, truncated on a "
            "statement boundary at MAX_SCHEMA_CHARS — the prompt is first-party "
            "either way (upstream lite's baselines build their own), and ga360 "
            "renders 2.9 M characters unbounded. Upstream runs one "
            "deterministic pass per question; n=1 is the protocol."
        ),
    ),
)
class Spider2LiteZeroShotGenTask(
    Task[
        Spider2LiteDatasetSample,
        PromptRecord,
        ModelOutput,
        PredictionRecord,
        JudgementRecord,
        dict[str, float | str],
    ]
):
    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        n: int = 1,
        localdb_dir: str | None = None,
        gold_dir: str | None = None,
        eval_config_path: str | None = None,
        documents_dir: str | None = None,
        db_schema_dir: str | None = None,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._n = n
        self._overrides = {
            "localdb_dir": localdb_dir,
            "gold_dir": gold_dir,
            "eval_config_path": eval_config_path,
            "documents_dir": documents_dir,
            "db_schema_dir": db_schema_dir,
        }

    def _staged(self, attribute: str) -> str:
        resolved = self._overrides.get(attribute) or getattr(
            self.dataset, attribute, None
        )
        if not resolved or not os.path.exists(resolved):
            raise ValueError(
                f"Spider 2.0-lite needs {attribute!r} but it did not resolve to "
                f"an existing path (got {resolved!r}). Stage the data with "
                "'sieval dataset download spider2_lite', or pass it to the task."
            )
        return resolved

    # -- prompt ----------------------------------------------------------

    def _schema_text(self, backend: str, db: str) -> str:
        """Schema block for *db*, from the engine that can describe it.

        SQLite is introspected from the database itself — authoritative, and it
        does not depend on the resource tree's layout. BigQuery and Snowflake
        cannot be introspected without credentials, so their schemas come from
        the DDL and table JSON upstream ships.
        """
        if backend == "sqlite":
            return _sqlite_schema(
                os.path.join(self._staged("localdb_dir"), f"{db}.sqlite")
            )
        return _resource_schema(self._staged("db_schema_dir"), backend, db)

    def _external_knowledge(self, filename: str | None) -> str:
        if not filename:
            return ""
        path = Path(self._staged("documents_dir"), filename)
        if not path.is_file():
            logger.warning("Spider 2.0-lite external knowledge missing: {}", filename)
            return ""
        return f"\n\nExternal knowledge:\n\n{path.read_text(encoding='utf-8')}"

    @override
    async def preprocess(self, raw, ctx):
        backend = backend_for(raw["instance_id"])
        prompt = (
            f"You are a {_DIALECT[backend]} expert.\n\n"
            f"Database schema:\n\n{self._schema_text(backend, raw['db'])}"
            f"{self._external_knowledge(raw['external_knowledge'])}\n\n"
            f"Question: {raw['question']}\n\n"
            f"Write one {_DIALECT[backend]} query that answers the question. "
            "Return only the query, in a ```sql code block."
        )
        return build_prompt_record(
            [{"role": "user", "content": prompt}],
            # The gold is a set of result frames on disk, not a value; the
            # instance id is what identifies them.
            reference=raw["instance_id"],
        )

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre["prompt"], n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        # Imported here, not at module scope: `sieval.community.spider2` reaches
        # google.cloud on the way in, so registering this task would need the
        # whole optional group -- and importing it has side effects the package
        # wrapper has to undo, which is not something `sieval task list` should
        # be paying for. See sieval/community/spider2/__init__.py.
        from sieval.community.spider2 import extract_sql_query

        # Upstream's own extractor: the fenced block if present, else the whole
        # reply. It never returns nothing, so a miss is an unrunnable query
        # rather than an unextracted one.
        return build_prediction_record([extract_sql_query(text) for text in inf.texts])

    # -- grading ---------------------------------------------------------

    @override
    async def feedback(self, post, ctx):
        raw = ctx.raw_sample
        instance_id = raw["instance_id"]
        backend = backend_for(instance_id)
        db_path = (
            os.path.join(self._staged("localdb_dir"), f"{raw['db']}.sqlite")
            if backend == "sqlite"
            else None
        )
        standard = _eval_standard(self._staged("eval_config_path")).get(instance_id, {})
        gold_dir = self._staged("gold_dir")

        # Asked once per sample, before anything is offloaded. The distinction
        # between "the host is unconfigured" and "the model wrote bad SQL" is
        # the whole point of the per-backend breakdown, and it cannot be drawn
        # on the far side of a worker process: telling one exception class from
        # another there means catching broadly at the call site below, which is
        # exactly what `.claude/rules/tasks.md` forbids. So it is drawn here,
        # where it is a plain environment read and no query has run yet.
        unreachable = missing_credentials(backend)

        rollouts: list[RolloutJudgement] = []
        for rollout in post["rollouts"]:
            prediction = rollout.get("prediction") or ""
            if unreachable:
                correct, error, missing = False, unreachable, True
            else:
                correct, error = await self._grade_one(
                    backend, prediction, db_path, instance_id, gold_dir, standard
                )
                missing = False
            rollouts.append(
                build_rollout_judgement(
                    rollout["index"],
                    correct,
                    metrics={"execution": correct},
                    extra={
                        "backend": _BACKEND_LABEL[backend],
                        "error": error,
                        "missing_credentials": missing,
                    },
                )
            )
        return True, build_judgement_record(instance_id, rollouts)

    async def _grade_one(
        self,
        backend: str,
        prediction: str,
        db_path: str | None,
        instance_id: str,
        gold_dir: str,
        standard: dict,
    ) -> tuple[bool, str | None]:
        """Verdict and, when there is one, the reason it is not a real answer.

        Query, gold load and comparison go over in **one** dispatch. They are
        one unit of work per rollout -- the comparison is meaningless without
        the frame -- and splitting them would pay two process hand-offs for one
        row while leaving the gold's ``read_csv`` on the shared event loop,
        where a 100k-row result stalls every other task in the session.
        """
        if not prediction.strip():
            return False, "empty prediction"
        budget = caller_timeout(backend)
        try:
            return await run_cpu_bound(
                _grade_sync,
                backend,
                prediction,
                db_path,
                instance_id,
                gold_dir,
                standard.get("condition_cols"),
                bool(standard.get("ignore_order", False)),
                timeout=budget,
            )
        except TimeoutError:
            # A grade that could not be computed *in time* is a wrong answer:
            # the prediction is a query neither the engine's own deadline nor
            # this budget could see the end of. Every other exception -- a dead
            # worker, `pandas` absent, a gold that is not there -- propagates,
            # so it lands in `fails` as `exception::<class>` rather than looking
            # like a model that answered badly.
            return False, f"TimeoutError after {budget}s"

    # -- report ----------------------------------------------------------

    @override
    async def report(self, finals, fails):
        n_correct = 0
        n_errors = 0
        n_missing = 0
        per_backend = {name: [0, 0, 0] for name in BACKENDS}
        for final in finals:
            for rollout in (final.feedback_result or {}).get("rollouts", []):
                extra = rollout.get("extra") or {}
                correct = bool(rollout["correct"])
                missing = bool(extra.get("missing_credentials"))
                n_correct += correct
                n_missing += missing
                # A credential miss carries a reason in `error` too, and it is
                # not an execution error: nothing was executed. Counting it as
                # one makes `n_execution_errors` read as 412 broken queries on
                # the very run where it should read 0.
                if extra.get("error") and not missing:
                    n_errors += 1
                bucket = per_backend.get(extra.get("backend") or "")
                if bucket is not None:
                    bucket[0] += correct
                    bucket[1] += 1
                    bucket[2] += missing

        total = (len(finals) + len(fails)) * self._n
        rate = (lambda c: round(100 * c / total, 2)) if total else (lambda c: 0.0)
        metrics: dict[str, float | str] = {
            "score": rate(n_correct),
            "execution_accuracy": rate(n_correct),
            "n": float(total),
            "fails": float(len(fails)),
            "n_execution_errors": float(n_errors),
            "n_missing_credentials": float(n_missing),
            SCORE_KEY_FIELD: "execution_accuracy",
            DENOMINATOR_FIELD: DENOMINATOR_REQUESTED,
        }
        # Per-backend rates are over rollouts actually graded on that engine.
        # Without them the headline is unreadable: a run with no cloud
        # credentials tops out near 24.7% and looks like a bad model.
        for name, (correct, seen, missing) in per_backend.items():
            metrics[f"execution_accuracy_{name}"] = (
                round(100 * correct / seen, 2) if seen else 0.0
            )
            metrics[f"n_{name}"] = float(seen)
            metrics[f"n_missing_credentials_{name}"] = float(missing)
        return metrics | health_metrics(finals)


# --- module-level helpers (picklable for run_cpu_bound) ---------------------


def _grade_sync(
    backend: str,
    sql: str,
    db_path: str | None,
    instance_id: str,
    gold_dir: str,
    condition_cols: list | None,
    ignore_order: bool,
) -> tuple[bool, str | None]:
    """Run *sql*, compare it against gold, and say why if it is not right.

    Module-level and picklable by name so ``run_cpu_bound`` can ship it to a
    worker process. Everything the vendored evaluator needs is imported here
    rather than at module scope, and deliberately **outside** the two ``try``
    blocks: an absent ``pandas`` is a broken environment, not a wrong answer,
    and must reach the caller as an exception.
    """
    import io

    import pandas as pd

    from sieval.community.spider2 import (
        compare_multi_pandas_table,
        compare_pandas_table,
        load_gold_csv,
        resolve_gold_paths,
    )

    # Resolved before the query runs, because a gold that is not there is our
    # bug: upstream scores that sample 0, which would report the model wrong for
    # a staging fault of ours. Unreachable on the pinned data -- gold ships for
    # all 547 instances -- and `.claude/rules/records.md` is what decides it.
    gold_paths, is_single = resolve_gold_paths(instance_id, gold_dir)
    if not gold_paths:
        raise ValueError(
            f"No Spider 2.0-lite gold result for {instance_id!r} under "
            f"{gold_dir!r}; re-stage with 'sieval dataset download "
            "spider2_lite --force'."
        )

    try:
        frame = execute(backend, sql, db_path=db_path)
    except Exception as exc:
        # Upstream's behaviour for a prediction that will not run: score 0 and
        # keep the engine's message, which is what separates "wrong answer" from
        # "could not ask".
        return False, f"{type(exc).__name__}: {exc}"

    try:
        # Upstream writes the result out and reads it back before comparing.
        # That round trip is part of the metric, not plumbing: `read_csv`
        # re-infers dtypes, so a SQLite `text` column holding "3" arrives as an
        # int64 and a BigQuery DATE arrives as the string the gold also holds.
        # Comparing the driver's own frame flips verdicts in both directions.
        buffer = io.StringIO()
        frame.to_csv(buffer, index=False)
        buffer.seek(0)
        predicted = pd.read_csv(buffer)
        # Upstream's own branch. One gold is compared directly; several are a
        # disjunction over answer shapes, and `compare_multi_pandas_table`
        # broadcasts `condition_cols` only when it is given more than one.
        if is_single:
            score = compare_pandas_table(
                predicted,
                load_gold_csv(str(gold_paths[0])),
                condition_cols,
                ignore_order,
            )
        else:
            golds = [load_gold_csv(str(path)) for path in gold_paths]
            score = compare_multi_pandas_table(
                predicted, golds, condition_cols, ignore_order
            )
    except Exception as exc:
        # Also upstream's: a comparison that raises is a 0 with the reason kept
        # ("Python Script Error" there). Changing it to a failure would change
        # the published score, so it stays a verdict -- prefixed, so the report
        # can tell it from a query that would not run.
        return False, f"compare: {type(exc).__name__}: {exc}"
    return bool(score), None


@cache
def _eval_standard(path: str) -> dict:
    """Upstream's per-instance comparison rules, parsed once per process."""
    rules = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        rules[item["instance_id"]] = item
    return rules


def _fit(blocks: list[str], budget: int) -> str:
    """Join *blocks*, dropping whole statements once *budget* is spent.

    Whole statements, never a cut one: half a ``CREATE TABLE`` reads as a schema
    that really does end there, which is a worse prompt than a shorter one. The
    first block always goes in, so a single oversized statement still gives the
    model something to work from. Both callers build *blocks* in sorted order, so
    which tables survive is a property of the database rather than of the
    filesystem — the same reason the test-suite walk sorts.
    """
    kept: list[str] = []
    used = 0
    for index, block in enumerate(blocks):
        if kept and used + len(block) > budget:
            kept.append(
                f"-- [{len(blocks) - index} of {len(blocks)} tables omitted: "
                f"schema exceeds this prompt's {budget}-character budget]"
            )
            break
        kept.append(block)
        used += len(block) + 2
    return "\n\n".join(kept)


@cache
def _sqlite_schema(db_path: str, budget: int = MAX_SCHEMA_CHARS) -> str:
    """CREATE TABLE statements for every user table in *db_path*.

    Cached: the 135 local questions cover far fewer databases, so a run
    re-introspects the same file many times over otherwise.
    """
    conn = open_readonly(db_path)
    try:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    finally:
        conn.close()
    return _fit([row[0].strip() for row in rows if row[0]], budget)


@cache
def _resource_schema(
    schema_root: str, backend: str, db: str, budget: int = MAX_SCHEMA_CHARS
) -> str:
    """Schema for a cloud database, from upstream's shipped resource tree.

    Upstream lays these out as ``<engine>/<db>/<schema>/`` holding a ``DDL.csv``
    (``table_name,ddl``) and one JSON per table. The DDL is preferred because it
    is the engine's own text.

    The JSON branch is a **fallback that the pinned data never reaches**: all
    128 distinct cloud databases ship a ``DDL.csv``, so no question renders a
    schema from table JSON. It stays because the tree is upstream's to reshape
    and a database arriving without DDL should degrade to a worse prompt rather
    than to a failed sample — but it is untested against real data, and a
    verdict has never depended on it.

    Cached, and not only to save the walk: 412 questions share 128 databases,
    and the largest of them is a 26 MB tree that would otherwise be re-read and
    re-rendered once per question.
    """
    root = Path(schema_root, backend, db)
    if not root.is_dir():
        raise ValueError(f"No shipped schema for {backend} database {db!r}")
    blocks: list[str] = []
    # `csv`'s field cap is process-global, so it is raised for the read and put
    # back. See `_CSV_FIELD_LIMIT` for the database that needs it.
    previous_limit = csv.field_size_limit(_CSV_FIELD_LIMIT)
    try:
        for ddl_path in sorted(root.rglob("DDL.csv")):
            with ddl_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    statement = (row.get("ddl") or "").strip()
                    if statement:
                        blocks.append(statement)
    finally:
        csv.field_size_limit(previous_limit)
    if blocks:
        return _fit(blocks, budget)
    for json_path in sorted(root.rglob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        columns = payload.get("column_names") or []
        types = payload.get("column_types") or []
        rendered = ", ".join(
            f"{name} {kind}"
            for name, kind in zip(columns, types + [""] * len(columns), strict=False)
        )
        blocks.append(f"-- {json_path.stem}\n({rendered})")
    if not blocks:
        raise ValueError(f"Shipped schema for {db!r} held no DDL or table JSON")
    return _fit(blocks, budget)
