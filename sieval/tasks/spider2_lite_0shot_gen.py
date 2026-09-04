"""Spider 2.0-lite — 0-shot generative text-to-SQL over real warehouses.

Spider 2.0 (Lei et al., ICLR 2025) is Spider's enterprise-scale successor. The
**lite** setting is its single-call reading: 547 questions over warehouse
schemas, scored by execution-result comparison against gold frames upstream
ships for all 547. The ``instance_id`` prefix picks the engine — ``bq``/``ga``
BigQuery (205), ``sf_bq``/``sf`` Snowflake (207), ``local`` SQLite (135).

**Default selection is the 135 SQLite questions, not all 547**
(:data:`~sieval.datasets.spider2_lite.DEFAULT_ENGINES`). ``missing_credentials``
can only be asked once a sample exists — after its prompt is built and inferred
— so an unconfigured host running everything pays for the 412 largest prompts in
the benchmark (~6.84 M prompt tokens against ~158 k for the local subset) to be
told it cannot ask.

**Read the denominator.** ``DENOMINATOR_REQUESTED`` means the rate is over what
was asked for, so a default run's ``score`` is over 135 and is *not* upstream's
``correct/547``; the report writes an ``engines`` field so the two cannot be
confused. Asking for all three without credentials caps near 24.7%, which is
honest but unreadable alone, so the report always publishes a per-backend
breakdown (``execution_accuracy_local`` / ``_bigquery`` / ``_snowflake``, each
with its own ``n_*`` and ``n_missing_credentials_*``). Read those first.

**Comparison is upstream's, from the right copy.** The repo ships two
``compare_pandas_table`` implementations; only ``evaluate.py``'s carries the
2025-10-29 accuracy fix (NaN normalised to 0, an early break, an empty-gold
guard). Vendored byte-identical, with each instance's ``condition_cols`` and
``ignore_order`` from ``spider2lite_eval.jsonl``.

``is_single`` misleads: ``resolve_gold_paths`` returns it ``True`` only when an
exact ``<instance_id>.csv`` exists, which is **3 of 547**. It is a filename
shape, not a gold count — 104 instances have one gold file and still take the
``compare_multi_pandas_table`` path. (Separately, the 1,544 golds do mean 440 of
547 accept more than one answer shape.)

**The prediction reaches the comparison through a CSV**, as upstream does, and
that round trip is part of the metric: ``read_csv`` re-infers dtypes, so a
SQLite ``text`` column holding ``"3"`` arrives as an integer and a BigQuery
``DATE`` as the string the gold also holds. Comparing the driver's own frame
flips verdicts in both directions.

**DIVERGENCE — Snowflake execution is first-party.**
``evaluate_single_sql_instance`` routes only BigQuery and SQLite, sending every
other prefix to "Unsupported instance id prefix" — 207 of 547 unscoreable by
upstream despite gold shipping for them. A Snowflake number here has no upstream
lite counterpart; compare against Spider 2.0-Snow. See ``_spider2_backends``.

**DIVERGENCE — the local engine is hardened**, from the shared ``_sqlite_exec``
rather than a second copy: read-only immutable connection, ATTACH/DETACH denied,
progress-handler deadline, row cap. Upstream instead copies the whole database
into memory per query, which stops writes but not runaway queries — and the
largest local database is 372 MB, which a concurrent grader cannot copy per
call. Remote engines are bounded (timeout + row cap), not hardened: the query
runs on someone else's server.

**The prompt is bounded too.** The 412 cloud questions run against 128 distinct
databases and the shipped tree is not uniform: ``ga360`` renders 2,894,566
characters, ``fec`` 13.5 M, and 56 of the 412 sit on a database whose
``DDL.csv`` alone exceeds a megabyte. Unbounded that is a request no endpoint
accepts, so the block stops at :data:`MAX_SCHEMA_CHARS` on a statement boundary
and says how many tables it dropped. 85 of the 412 are truncated; no local
question is, the 30 local schemas running 956 to 8,679 characters.

Target: **no comparable published number exists**, which is a property of the
leaderboard. It publishes one aggregate over 547 with no per-engine breakdown,
and it is almost entirely agentic (topping 76%) while the nearest single-call
entries are prompting frameworks at 1.5–5.7%. Hence ``experimental``: a faithful
port with no reachable anchor, not one awaiting a run.

Measured on the real staged archives: all **135** local questions run end to end
with 0 pipeline failures; **24 of 24** verdicts agree with upstream's own
``evaluate_single_sql_instance``, the 8 scoring 0 included (its ``gold/sql`` is
stale against its own ``gold/exec_result`` and both call those wrong
identically); all **128** cloud databases render within the budget; and a real
run over the default selection scored ``execution_accuracy`` 48.89 (66/135) with
0 fails and 0 unextracted. Not measured: the BigQuery and Snowflake engines,
which need credentials this environment does not have.

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
from sieval.datasets.spider2_lite import ALL_ENGINES, backend_for

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
#: characters and ``bigquery/pancancer_atlas_2`` holds one ``ddl`` value longer
#: (147,830), so ``DictReader`` raised ``_csv.Error`` and its two questions
#: (``bq151``, ``bq161``) died in ``preprocess``. Not ``sys.maxsize``: the cap is
#: a C long, so this is the largest value every platform accepts.
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
            "instead of a result — corrupt, though its _a/_c siblings are "
            "valid); none is bq/ga, the only prefix whose upstream path scores "
            "an empty result 0 without comparing, so omitting that guard cannot "
            "change a verdict here. It does cost one free point on the "
            "first-party Snowflake path: an empty result matches sf_bq411_b's "
            "zero rows, so sf_bq411 scores 1 for a query returning nothing. "
            "1 of 207, kept because dropping an empty gold would itself be a "
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

        # Asked here, not in the worker: telling "host unconfigured" from "model
        # wrote bad SQL" across a process boundary would mean catching broadly
        # at the grading call site, which `.claude/rules/tasks.md` forbids.
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
            # What the rate is over. A default run's 135 and upstream's
            # `correct/547` are different measurements; `n` alone cannot say
            # which one this report holds.
            "engines": ",".join(_requested_engines(self.dataset)),
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


def _requested_engines(dataset) -> tuple[str, ...]:
    """Engines the dataset was loaded for, or all of them if it carries none.

    A dataset built straight from an ``HFDatasetDict`` (a test, or a caller
    slicing rows itself) made no selection, so naming them all beats asserting
    a subset nobody asked for.
    """
    engines = getattr(dataset, "engines", None)
    return tuple(engines) if engines else ALL_ENGINES


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
    (``table_name,ddl``) and one JSON per table; the DDL is preferred as the
    engine's own text.

    The JSON branch is a **fallback the pinned data never reaches** — all 128
    cloud databases ship a ``DDL.csv`` — kept so a database arriving without one
    degrades to a worse prompt rather than a failed sample. Untested against
    real data; no verdict has depended on it.

    Cached: 412 questions share 128 databases, the largest a 26 MB tree.
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
