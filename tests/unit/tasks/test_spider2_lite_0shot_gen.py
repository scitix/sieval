"""Unit tests for spider2_lite_0shot_gen.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import csv
import json
import sqlite3
from pathlib import Path

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import Request, Response
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
from sieval.core.utils.offload import GRADE_TIMEOUT
from sieval.datasets.spider2_lite import Spider2LiteDataset
from sieval.tasks._spider2_backends import caller_timeout, execute
from sieval.tasks.spider2_lite_0shot_gen import (
    Spider2LiteZeroShotGenTask,
    _fit,
    _resource_schema,
    _sqlite_schema,
)
from tests.conftest import HandlerTransport


class _ScriptedChatModel(ChatModel):
    def __init__(self, reply: str, model: str = "mock"):
        self._reply = reply
        super().__init__(model=model, api_key="fake")

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_chat")

    async def _stub_arun(self, req: Request) -> Response:
        return Response(
            texts=(self._reply,) * req.sampling.n,
            finish_reasons=("stop",) * req.sampling.n,
        )


_ROW = {
    "instance_id": "local001",
    "db": "tiny",
    "question": "How many rows?",
    "external_knowledge": None,
    "temporal": None,
}


@pytest.fixture
def staged(tmp_path):
    """A miniature staged tree: one local database, gold, config, documents."""
    localdb = tmp_path / "localdb"
    localdb.mkdir()
    conn = sqlite3.connect(localdb / "tiny.sqlite")
    conn.execute("CREATE TABLE t (a int)")
    conn.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (3,)])
    conn.commit()
    conn.close()

    gold = tmp_path / "gold"
    gold.mkdir()
    (gold / "local001.csv").write_text("n\n3\n")
    # `local002` has no base CSV, so upstream's `resolve_gold_paths` reports it
    # as a multi-gold instance -- 1,544 golds over 547 questions, so this is the
    # common shape rather than the exotic one. Only the `_b` answer is right.
    (gold / "local002_a.csv").write_text("n\n99\n")
    (gold / "local002_b.csv").write_text("n\n3\n")

    config = tmp_path / "eval.jsonl"
    config.write_text(
        "\n".join(
            json.dumps(
                {"instance_id": name, "condition_cols": [], "ignore_order": False}
            )
            for name in ("local001", "local002")
        )
    )

    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "notes.md").write_text("some external knowledge")

    schemas = tmp_path / "databases"
    (schemas / "bigquery" / "warehouse" / "s").mkdir(parents=True)
    (schemas / "bigquery" / "warehouse" / "s" / "DDL.csv").write_text(
        "table_name,ddl\nt,CREATE TABLE `w.t` (a INT64)\n"
    )
    return {
        "localdb_dir": str(localdb),
        "gold_dir": str(gold),
        "eval_config_path": str(config),
        "documents_dir": str(documents),
        "db_schema_dir": str(schemas),
    }


def _task(staged, reply="```sql\nSELECT count(*) AS n FROM t\n```", rows=None):
    dataset = Spider2LiteDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list(rows or [_ROW])})
    )
    return Spider2LiteZeroShotGenTask(
        dataset, _ScriptedChatModel(reply=reply), **staged
    )


def _require_grader() -> None:
    """Skip unless the ``spider2`` extra is installed.

    Only the grading half needs it: `postprocess` and `feedback` reach the
    vendored evaluator, which imports google-cloud-bigquery at module scope
    whichever engine the sample runs on. Placed at this chokepoint rather than
    on the module so the prompt, report and offload-budget tests keep running
    where the extra is absent — CI installs eight dependency groups and this is
    not one of them.

    Gated on the third-party name rather than on `sieval.community.spider2`, so
    a genuine ImportError from the package wrapper fails loudly instead of
    quietly skipping the tests that would have caught it.
    """
    pytest.importorskip("google.cloud.bigquery", reason="requires the `spider2` extra")


async def _judge(task, raw=_ROW):
    """Run the whole pipeline over one sample; return prediction and judgement."""
    _require_grader()
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    post = await task.postprocess(
        await task.infer(await task.preprocess(raw, ctx), ctx), ctx
    )
    _, judgement = await task.feedback(post, ctx)
    return post, judgement


# --- prompt -----------------------------------------------------------------


@pytest.mark.anyio
async def test_local_prompt_introspects_the_database(staged):
    task = _task(staged)
    pre = await task.preprocess(_ROW, TaskContext(sample_id=0, raw_sample=_ROW))
    content = pre["prompt"][0]["content"]
    assert "CREATE TABLE t" in content
    assert "SQLite" in content
    assert "How many rows?" in content
    assert pre["reference"] == "local001"


@pytest.mark.anyio
async def test_cloud_prompt_reads_the_shipped_ddl(staged):
    row = dict(_ROW, instance_id="bq001", db="warehouse")
    task = _task(staged, rows=[row])
    pre = await task.preprocess(row, TaskContext(sample_id=0, raw_sample=row))
    content = pre["prompt"][0]["content"]
    assert "CREATE TABLE `w.t`" in content
    assert "BigQuery" in content


@pytest.mark.anyio
async def test_external_knowledge_is_inlined_when_present(staged):
    row = dict(_ROW, external_knowledge="notes.md")
    task = _task(staged, rows=[row])
    pre = await task.preprocess(row, TaskContext(sample_id=0, raw_sample=row))
    assert "some external knowledge" in pre["prompt"][0]["content"]


@pytest.mark.anyio
async def test_absent_external_knowledge_adds_nothing(staged):
    task = _task(staged)
    pre = await task.preprocess(_ROW, TaskContext(sample_id=0, raw_sample=_ROW))
    assert "External knowledge" not in pre["prompt"][0]["content"]


# --- the schema block is bounded --------------------------------------------


def test_the_schema_block_drops_whole_statements_and_says_how_many():
    blocks = [f"CREATE TABLE {name} (x int)" for name in ("a", "b", "c")]
    fitted = _fit(blocks, 30)
    assert "CREATE TABLE a" in fitted
    assert "CREATE TABLE b" not in fitted
    assert "2 of 3 tables omitted" in fitted


def test_a_statement_is_never_cut_in_half():
    """Half a `CREATE TABLE` reads as a schema that really does end there.

    A truncated prompt should make the model answer over fewer tables, not
    over a table it thinks has three columns when it has thirty.
    """
    blocks = ["CREATE TABLE a (x int)", "CREATE TABLE b (y int)"]
    kept = _fit(blocks, 30).splitlines()[0]
    assert kept == "CREATE TABLE a (x int)"


def test_one_oversized_statement_still_gives_the_model_something():
    """The first block goes in whatever it costs — otherwise a database whose
    first table exceeds the budget renders as a prompt with no schema at all."""
    huge = "CREATE TABLE wide (" + ", ".join(f"c{i} int" for i in range(500)) + ")"
    assert _fit([huge], 100) == huge


def test_a_local_schema_over_budget_is_truncated_on_a_statement_boundary(tmp_path):
    """The same bound, reached through the introspection the prompt uses."""
    path = tmp_path / "many.sqlite"
    conn = sqlite3.connect(path)
    for name in ("a", "b", "c"):
        conn.execute(f"CREATE TABLE {name} (x int)")
    conn.commit()
    conn.close()
    rendered = _sqlite_schema(str(path), budget=30)
    assert rendered.startswith("CREATE TABLE a")
    assert "2 of 3 tables omitted" in rendered
    # Sorted by name, so which tables survive is a property of the database
    # rather than of the order SQLite happened to store them in.
    assert "CREATE TABLE c" not in rendered


def test_a_ddl_field_over_the_csv_default_limit_is_readable(tmp_path):
    """`bigquery/pancancer_atlas_2`'s shape, at the size that used to crash.

    `csv` caps a field at 131,072 characters by default, and that database
    holds one longer, so `DictReader` raised `_csv.Error` out of `preprocess`
    and its two questions (bq151, bq161) failed instead of scoring. The cap is
    process-global, so the read raises it and puts it back.
    """
    root = tmp_path / "bigquery" / "wide"
    root.mkdir(parents=True)
    columns = ", ".join(f"c{i} STRING" for i in range(20_000))
    statement = f"CREATE TABLE huge ({columns})"
    assert len(statement) > 131_072, "fixture must exceed csv's default cap"
    (root / "DDL.csv").write_text(
        f'table_name,ddl\nhuge,"{statement}"\n', encoding="utf-8"
    )

    rendered = _resource_schema(str(tmp_path), "bigquery", "wide", 10**9)

    assert rendered == statement
    # And the global cap is what it was before the read.
    assert csv.field_size_limit() == 131_072


# --- end to end over the local engine ---------------------------------------


@pytest.mark.anyio
async def test_full_pipeline_scores_a_correct_answer(staged):
    task = _task(staged)
    post, judgement = await _judge(task)
    assert post["rollouts"][0]["prediction"] == "SELECT count(*) AS n FROM t"
    rollout = judgement["rollouts"][0]
    assert rollout["correct"] is True
    assert rollout["extra"]["backend"] == "local"
    assert rollout["extra"]["missing_credentials"] is False


@pytest.mark.anyio
async def test_wrong_answer_scores_zero(staged):
    task = _task(staged, reply="```sql\nSELECT 99 AS n\n```")
    _, judgement = await _judge(task)
    assert judgement["rollouts"][0]["correct"] is False


@pytest.mark.anyio
async def test_unrunnable_sql_is_recorded_as_an_error(staged):
    task = _task(staged, reply="```sql\nSELECT * FROM no_such_table\n```")
    _, judgement = await _judge(task)
    rollout = judgement["rollouts"][0]
    assert rollout["correct"] is False
    assert rollout["extra"]["error"] is not None
    assert rollout["extra"]["missing_credentials"] is False


@pytest.mark.anyio
async def test_missing_cloud_credentials_are_flagged_not_silently_wrong(
    staged, monkeypatch
):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    row = dict(_ROW, instance_id="bq001", db="warehouse")
    task = _task(staged, rows=[row])
    _, judgement = await _judge(task, row)
    rollout = judgement["rollouts"][0]
    assert rollout["correct"] is False
    assert rollout["extra"]["missing_credentials"] is True
    assert rollout["extra"]["backend"] == "bigquery"


# --- the comparison: upstream's two branches and the CSV round trip ----------


@pytest.mark.anyio
async def test_a_text_typed_number_is_scored_as_the_number_it_spells(staged):
    """The CSV round trip, seen from the outside.

    `count(*)` cast to `text` is the same answer as `count(*)`, and upstream
    scores it right because it writes the result out and reads it back before
    comparing. Drop the round trip and this verdict flips — see the next test,
    which is that same query compared without it.
    """
    task = _task(staged, reply="```sql\nSELECT CAST(count(*) AS TEXT) AS n FROM t\n```")
    _, judgement = await _judge(task)
    assert judgement["rollouts"][0]["correct"] is True


def test_comparing_the_driver_frame_directly_would_call_that_answer_wrong(staged):
    """The counterfactual, so the test above cannot pass for another reason.

    Upstream's comparison takes `a != b` for any pair that is not two numbers,
    so the string `"3"` the SQLite driver returns never matches the int64 the
    gold CSV parses to. This is what makes `to_csv`/`read_csv` part of the
    metric rather than plumbing: it is the step that re-infers the dtype.
    """
    _require_grader()
    from sieval.community.spider2 import compare_pandas_table, load_gold_csv

    frame = execute(
        "sqlite",
        "SELECT CAST(count(*) AS TEXT) AS n FROM t",
        db_path=str(Path(staged["localdb_dir"], "tiny.sqlite")),
    )
    gold = load_gold_csv(str(Path(staged["gold_dir"], "local001.csv")))
    assert frame["n"].tolist() == ["3"]
    assert compare_pandas_table(frame, gold) == 0


@pytest.mark.anyio
async def test_several_golds_are_a_disjunction_over_answer_shapes(staged):
    """`local002` ships `_a` and `_b`; agreeing with either one is correct.

    Pinned because upstream's `is_single` flag is easy to drop on the way
    through: comparing against `gold_paths[0]` alone would score this answer
    wrong, and would do it silently on the majority of the 547 questions.
    """
    row = dict(_ROW, instance_id="local002")
    task = _task(staged, rows=[row])
    _, judgement = await _judge(task, row)
    assert judgement["rollouts"][0]["correct"] is True


@pytest.mark.anyio
async def test_a_missing_gold_raises_rather_than_scoring_the_model_wrong(staged):
    """Gold ships for all 547, so an absent one is a staging fault of ours.

    `.claude/rules/records.md`: a value-reference task that finds no gold
    fails the sample instead of recording a verdict it had nothing to compare
    against. Scoring 0 — which is what upstream does — would report the model
    wrong for an unpacked archive.
    """
    row = dict(_ROW, instance_id="local999")
    task = _task(staged, rows=[row])
    with pytest.raises(ValueError, match="No Spider 2.0-lite gold result"):
        await _judge(task, row)


# --- the grade is offloaded with the right budget ----------------------------


@pytest.mark.anyio
async def test_the_grade_is_offloaded_with_the_backend_s_own_budget(
    staged, monkeypatch
):
    """`run_cpu_bound`'s 30 s default is shorter than either engine bound.

    A process pool cannot interrupt a running call, so a caller that gives up
    first turns every in-engine deadline into decoration and reports a timeout
    where the engine would have reported a bad query. The call site therefore
    passes `caller_timeout(backend)`, and this pins it — including that the
    offload happens at all, which an assertion about the budget alone would not.
    """
    seen: list[float | None] = []

    async def _spy(_func, *_args, timeout=None):
        seen.append(timeout)
        return True, None

    monkeypatch.setattr(
        "sieval.tasks.spider2_lite_0shot_gen.run_cpu_bound", _spy, raising=True
    )
    task = _task(staged)
    ctx = TaskContext(sample_id=0, raw_sample=_ROW)
    await task.feedback(build_prediction_record(["SELECT count(*) AS n FROM t"]), ctx)
    assert seen == [caller_timeout("sqlite")]
    # Named rather than implied: this is the value the call site gets by
    # omitting `timeout=`, and it is shorter than the engine's own deadline.
    assert GRADE_TIMEOUT not in seen


@pytest.mark.anyio
async def test_an_unextracted_answer_is_not_offloaded_at_all(staged, monkeypatch):
    """Nothing to run, so no worker is paid for and no gold is read."""

    async def _spy(*_args, **_kwargs):
        raise AssertionError("a blank prediction must not reach a worker")

    monkeypatch.setattr(
        "sieval.tasks.spider2_lite_0shot_gen.run_cpu_bound", _spy, raising=True
    )
    task = _task(staged)
    ctx = TaskContext(sample_id=0, raw_sample=_ROW)
    _, judgement = await task.feedback(build_prediction_record(["   "]), ctx)
    rollout = judgement["rollouts"][0]
    assert rollout["correct"] is False
    assert rollout["extra"]["error"] == "empty prediction"


# --- report -----------------------------------------------------------------


def _final(sample_id, *, correct, backend, missing=False, error=None):
    return TaskContext(
        sample_id=sample_id,
        feedback_result=build_judgement_record(
            f"id{sample_id}",
            [
                build_rollout_judgement(
                    0,
                    correct,
                    metrics={"execution": correct},
                    extra={
                        "backend": backend,
                        "error": error,
                        "missing_credentials": missing,
                    },
                )
            ],
        ),
        postprocess_result=build_prediction_record(["SELECT 1"]),
    )


@pytest.mark.anyio
async def test_report_declares_score_key_and_denominator(staged):
    metrics = await _task(staged).report([], [])
    assert metrics["score_key"] == "execution_accuracy"
    assert metrics["denominator_policy"] == "requested"


@pytest.mark.anyio
async def test_per_backend_breakdown_makes_a_credential_less_run_readable(staged):
    """Headline 33% but local 100% — the breakdown is what says why."""
    finals = [
        _final(0, correct=True, backend="local"),
        _final(1, correct=False, backend="bigquery", missing=True),
        _final(2, correct=False, backend="snowflake", missing=True),
    ]
    metrics = await _task(staged).report(finals, [])
    assert metrics["execution_accuracy"] == pytest.approx(33.33)
    assert metrics["execution_accuracy_local"] == 100.0
    assert metrics["n_local"] == 1.0
    assert metrics["n_missing_credentials_bigquery"] == 1.0
    assert metrics["n_missing_credentials_snowflake"] == 1.0


@pytest.mark.anyio
async def test_a_pipeline_failure_counts_as_wrong(staged):
    finals = [_final(0, correct=True, backend="local")]
    metrics = await _task(staged).report(finals, [TaskContext(sample_id=1)])
    assert metrics["execution_accuracy"] == 50.0
    assert metrics["fails"] == 1.0


@pytest.mark.anyio
async def test_execution_errors_are_counted_apart_from_wrong_answers(staged):
    finals = [
        _final(0, correct=False, backend="local", error="no such table"),
        _final(1, correct=False, backend="local"),
    ]
    metrics = await _task(staged).report(finals, [])
    assert metrics["n_execution_errors"] == 1.0


@pytest.mark.anyio
async def test_a_credential_miss_is_not_an_execution_error(staged):
    """It carries a reason in `error` too, and nothing was executed.

    Counting it as one makes `n_execution_errors` read as 412 broken queries on
    the very run — no cloud credentials — where it should read 0, and that is
    the run most people have.
    """
    finals = [
        _final(
            0,
            correct=False,
            backend="bigquery",
            missing=True,
            error="GOOGLE_APPLICATION_CREDENTIALS is not set",
        ),
        _final(1, correct=False, backend="local", error="no such table"),
    ]
    metrics = await _task(staged).report(finals, [])
    assert metrics["n_execution_errors"] == 1.0
    assert metrics["n_missing_credentials"] == 1.0


@pytest.mark.anyio
async def test_every_backend_appears_even_when_unused(staged):
    """A missing engine must read as zero-of-zero, not vanish."""
    metrics = await _task(staged).report([_final(0, correct=True, backend="local")], [])
    for name in ("local", "bigquery", "snowflake"):
        assert f"execution_accuracy_{name}" in metrics
        assert f"n_{name}" in metrics


def test_unstaged_path_is_a_loud_stop(staged):
    task = _task(staged)
    task._overrides["gold_dir"] = str(Path("/nonexistent/gold"))
    with pytest.raises(ValueError, match="gold_dir"):
        task._staged("gold_dir")
