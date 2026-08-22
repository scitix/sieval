"""Unit tests for spider2_lite_0shot_gen.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

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
from sieval.datasets.spider2_lite import Spider2LiteDataset
from sieval.tasks.spider2_lite_0shot_gen import Spider2LiteZeroShotGenTask
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

    config = tmp_path / "eval.jsonl"
    config.write_text(
        json.dumps(
            {"instance_id": "local001", "condition_cols": [], "ignore_order": False}
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


# --- end to end over the local engine ---------------------------------------


@pytest.mark.anyio
async def test_full_pipeline_scores_a_correct_answer(staged):
    task = _task(staged)
    ctx = TaskContext(sample_id=0, raw_sample=_ROW)
    pre = await task.preprocess(_ROW, ctx)
    inf = await task.infer(pre, ctx)
    post = await task.postprocess(inf, ctx)
    assert post["rollouts"][0]["prediction"] == "SELECT count(*) AS n FROM t"

    finalize, judgement = await task.feedback(post, ctx)
    assert finalize is True
    rollout = judgement["rollouts"][0]
    assert rollout["correct"] is True
    assert rollout["extra"]["backend"] == "local"
    assert rollout["extra"]["missing_credentials"] is False


@pytest.mark.anyio
async def test_wrong_answer_scores_zero(staged):
    task = _task(staged, reply="```sql\nSELECT 99 AS n\n```")
    ctx = TaskContext(sample_id=0, raw_sample=_ROW)
    post = await task.postprocess(
        await task.infer(await task.preprocess(_ROW, ctx), ctx), ctx
    )
    _, judgement = await task.feedback(post, ctx)
    assert judgement["rollouts"][0]["correct"] is False


@pytest.mark.anyio
async def test_unrunnable_sql_is_recorded_as_an_error(staged):
    task = _task(staged, reply="```sql\nSELECT * FROM no_such_table\n```")
    ctx = TaskContext(sample_id=0, raw_sample=_ROW)
    post = await task.postprocess(
        await task.infer(await task.preprocess(_ROW, ctx), ctx), ctx
    )
    _, judgement = await task.feedback(post, ctx)
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
    ctx = TaskContext(sample_id=0, raw_sample=row)
    post = await task.postprocess(
        await task.infer(await task.preprocess(row, ctx), ctx), ctx
    )
    _, judgement = await task.feedback(post, ctx)
    rollout = judgement["rollouts"][0]
    assert rollout["correct"] is False
    assert rollout["extra"]["missing_credentials"] is True
    assert rollout["extra"]["backend"] == "bigquery"


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
