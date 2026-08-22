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
``ignore_order`` from upstream's ``spider2lite_eval.jsonl`` and multi-gold via
``compare_multi_pandas_table`` (1,545 gold CSVs over 547 instances, so most
questions accept more than one correct answer shape).

**Snowflake is sieval's own, because upstream lite no longer has one.**
``evaluate_single_sql_instance`` routes BigQuery and SQLite and sends every other
prefix to "Unsupported instance id prefix" — 207 of 547 unscoreable by upstream
even though gold ships for them. A Snowflake number here therefore has **no
upstream lite counterpart**; the comparable published setting is Spider 2.0-Snow,
same questions, same warehouse. See ``sieval.tasks._spider2_backends``.

**Execution safety.** The local engine runs model SQL, so it carries the same
hardening as Spider 1.0 — read-only immutable connection, ATTACH/DETACH denied,
progress-handler deadline, row cap. Upstream instead copies the whole database
into memory per query; that stops writes but not runaway queries, and the largest
local database is 372 MB, which a concurrent grader cannot copy per call. The
remote engines are bounded (timeout + row cap) rather than hardened: the query
runs on someone else's server under their permissions.

Target: upstream's Spider 2.0-lite leaderboard, for the BigQuery and SQLite
subsets it scores; Spider 2.0-Snow for the Snowflake subset.

Measured: **not yet**, on either axis — no alignment run, and (unlike Spider 1.0)
no end-to-end pass over the real archives either, because the host volume had no
room to extract them. The loader and comparison are validated against the real
archives' *contents* read in place, and the task's stages against synthetic
fixtures. Treat every number this task produces as unverified until someone runs
it on staged data.

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

import pandas as pd
from loguru import logger

from sieval.community.spider2 import (
    compare_multi_pandas_table,
    extract_sql_query,
    load_gold_csv,
    resolve_gold_paths,
)
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
from sieval.core.utils.offload import GRADE_TIMEOUT, run_cpu_bound
from sieval.datasets import Spider2LiteDatasetSample
from sieval.datasets.spider2_lite import backend_for

from ._spider2_backends import MissingCredentials, execute, open_readonly

#: Engines, in report order. Local first: it is the subset most runs can score.
BACKENDS = ("local", "bigquery", "snowflake")
#: `backend_for` returns the engine; the report groups under these names.
_BACKEND_LABEL = {"sqlite": "local", "bigquery": "bigquery", "snowflake": "snowflake"}

_DIALECT = {
    "sqlite": "SQLite",
    "bigquery": "BigQuery Standard SQL",
    "snowflake": "Snowflake SQL",
}


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
            "condition_cols/ignore_order from spider2lite_eval.jsonl, multi-gold "
            "via compare_multi_pandas_table over 1,545 gold CSVs. SQL extraction "
            "is upstream's extract_sql_query. DIVERGENCE — Snowflake execution is "
            "first-party: upstream's current lite evaluator routes only bq/ga and "
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
            "prompts read upstream's resource/databases tree. Upstream runs one "
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

        rollouts: list[RolloutJudgement] = []
        for rollout in post["rollouts"]:
            prediction = rollout.get("prediction") or ""
            correct, error, missing = await self._grade_one(
                backend, prediction, db_path, instance_id, gold_dir, standard
            )
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
    ) -> tuple[bool, str | None, bool]:
        if not prediction.strip():
            return False, "empty prediction", False
        try:
            frame = await run_cpu_bound(
                _execute_sync,
                backend,
                prediction,
                db_path,
                timeout=GRADE_TIMEOUT,
            )
        except MissingCredentials as exc:
            # Loud and distinguishable: an unconfigured host is not a bad model.
            return False, str(exc), True
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}", False
        try:
            gold_paths, _ = resolve_gold_paths(instance_id, gold_dir)
            if not gold_paths:
                raise ValueError(f"no gold result for {instance_id}")
            golds = [load_gold_csv(str(path)) for path in gold_paths]
            score = compare_multi_pandas_table(
                frame,
                golds,
                standard.get("condition_cols"),
                standard.get("ignore_order", False),
            )
        except Exception as exc:
            logger.warning(
                "Spider 2.0-lite comparison failed for {}: {}", instance_id, exc
            )
            return False, f"compare: {type(exc).__name__}: {exc}", False
        return bool(score), None, False

    # -- report ----------------------------------------------------------

    @override
    async def report(self, finals, fails):
        n_correct = 0
        n_errors = 0
        per_backend = {name: [0, 0, 0] for name in BACKENDS}
        for final in finals:
            for rollout in (final.feedback_result or {}).get("rollouts", []):
                extra = rollout.get("extra") or {}
                correct = bool(rollout["correct"])
                n_correct += correct
                if extra.get("error"):
                    n_errors += 1
                bucket = per_backend.get(extra.get("backend") or "")
                if bucket is not None:
                    bucket[0] += correct
                    bucket[1] += 1
                    bucket[2] += bool(extra.get("missing_credentials"))

        total = (len(finals) + len(fails)) * self._n
        rate = (lambda c: round(100 * c / total, 2)) if total else (lambda c: 0.0)
        metrics: dict[str, float | str] = {
            "score": rate(n_correct),
            "execution_accuracy": rate(n_correct),
            "n": float(total),
            "fails": float(len(fails)),
            "n_execution_errors": float(n_errors),
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


def _execute_sync(backend: str, sql: str, db_path: str | None) -> pd.DataFrame:
    """Thin picklable wrapper so the query runs off the event loop."""
    return execute(backend, sql, db_path=db_path)


@cache
def _eval_standard(path: str) -> dict:
    """Upstream's per-instance comparison rules, parsed once per process."""
    rules = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        rules[item["instance_id"]] = item
    return rules


def _sqlite_schema(db_path: str) -> str:
    """CREATE TABLE statements for every user table in *db_path*."""
    conn = open_readonly(db_path)
    try:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    finally:
        conn.close()
    return "\n\n".join(row[0].strip() for row in rows if row[0])


def _resource_schema(schema_root: str, backend: str, db: str) -> str:
    """Schema for a cloud database, from upstream's shipped resource tree.

    Upstream lays these out as ``<engine>/<db>/<schema>/`` holding a ``DDL.csv``
    (``table_name,ddl``) and one JSON per table. The DDL is preferred because it
    is the engine's own text; the JSON files are the fallback for databases that
    ship no DDL.csv.
    """
    root = Path(schema_root, backend, db)
    if not root.is_dir():
        raise ValueError(f"No shipped schema for {backend} database {db!r}")
    blocks: list[str] = []
    for ddl_path in sorted(root.rglob("DDL.csv")):
        with ddl_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                statement = (row.get("ddl") or "").strip()
                if statement:
                    blocks.append(statement)
    if blocks:
        return "\n\n".join(blocks)
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
    return "\n\n".join(blocks)
