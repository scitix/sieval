"""Unit tests for spider_0shot_gen.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import sqlite3

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
from sieval.datasets.spider import SpiderDataset
from sieval.tasks.spider_0shot_gen import SpiderZeroShotGenTask, extract_sql
from tests.conftest import HandlerTransport


class _ScriptedChatModel(ChatModel):
    """ChatModel returning a fixed reply, recording calls."""

    def __init__(self, reply: str, model: str = "mock"):
        self._reply = reply
        self.calls: list[str] = []
        super().__init__(model=model, api_key="fake")

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_chat")

    async def _stub_arun(self, req: Request) -> Response:
        self.calls.append(str(req.input))
        return Response(
            texts=(self._reply,) * req.sampling.n,
            finish_reasons=("stop",) * req.sampling.n,
        )


# --- SQL extraction ---------------------------------------------------------


def test_extract_prefers_a_fenced_block():
    assert extract_sql("blah\n```sql\nSELECT 1\n```\ntrailing") == "SELECT 1"


def test_extract_accepts_an_unlabelled_fence():
    assert extract_sql("```\nSELECT 2\n```") == "SELECT 2"


def test_extract_falls_back_to_a_bare_statement():
    assert extract_sql("Here you go: SELECT 3 FROM t") == "SELECT 3 FROM t"


def test_extract_takes_the_last_fence_when_several_are_present():
    """Models routinely show working, then give the answer last."""
    reply = "```sql\nSELECT 1\n```\nactually, better:\n```sql\nSELECT 2\n```"
    assert extract_sql(reply) == "SELECT 2"


def test_extract_handles_a_with_clause():
    extracted = extract_sql("```sql\nWITH c AS (SELECT 1) SELECT * FROM c\n```")
    assert extracted is not None
    assert extracted.startswith("WITH")


def test_extract_strips_a_trailing_semicolon():
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"


def test_extract_returns_none_when_nothing_looks_like_sql():
    assert extract_sql("I cannot answer that.") is None


def test_extract_returns_none_for_an_empty_reply():
    assert extract_sql("") is None


# --- report -----------------------------------------------------------------


def _empty_dataset() -> SpiderDataset:
    return SpiderDataset(_hf_dict=HFDatasetDict({"test": HFDataset.from_list([])}))


def _task(**kwargs) -> SpiderZeroShotGenTask:
    return SpiderZeroShotGenTask(
        _empty_dataset(), _ScriptedChatModel(reply="```sql\nSELECT 1\n```"), **kwargs
    )


def _final(
    sample_id: int, *, execution: bool, exact: bool, hardness: str, error=None
) -> TaskContext:
    return TaskContext(
        sample_id=sample_id,
        feedback_result=build_judgement_record(
            "SELECT 1",
            [
                build_rollout_judgement(
                    0,
                    execution,
                    metrics={"execution": execution, "exact_match": exact},
                    extra={"hardness": hardness, "error": error},
                )
            ],
        ),
        postprocess_result=build_prediction_record(["SELECT 1"]),
    )


@pytest.mark.anyio
async def test_report_declares_score_key_and_denominator():
    metrics = await _task().report([], [])
    assert metrics["score_key"] == "execution_accuracy"
    assert metrics["denominator_policy"] == "requested"


@pytest.mark.anyio
async def test_empty_run_still_declares_both_fields():
    """The empty-run guard is a return path too."""
    metrics = await _task().report([], [])
    assert metrics["execution_accuracy"] == 0.0
    assert metrics["score"] == 0.0


@pytest.mark.anyio
async def test_score_is_copied_from_the_key_it_names():
    finals = [
        _final(0, execution=True, exact=True, hardness="easy"),
        _final(1, execution=False, exact=False, hardness="hard"),
    ]
    metrics = await _task().report(finals, [])
    assert metrics["score"] == metrics["execution_accuracy"] == 50.0


@pytest.mark.anyio
async def test_exact_match_is_reported_separately_from_execution():
    """The two metrics are co-equal and must not be conflated."""
    finals = [_final(0, execution=True, exact=False, hardness="medium")]
    metrics = await _task().report(finals, [])
    assert metrics["execution_accuracy"] == 100.0
    assert metrics["exact_match"] == 0.0


@pytest.mark.anyio
async def test_a_pipeline_failure_counts_as_wrong():
    """DENOMINATOR_REQUESTED: fails sit in the denominator, not outside it."""
    finals = [_final(0, execution=True, exact=True, hardness="easy")]
    metrics = await _task().report(finals, [TaskContext(sample_id=1)])
    assert metrics["execution_accuracy"] == 50.0
    assert metrics["fails"] == 1.0
    assert metrics["n"] == 2.0


@pytest.mark.anyio
async def test_per_hardness_rates_carry_their_own_counts():
    finals = [
        _final(0, execution=True, exact=True, hardness="easy"),
        _final(1, execution=False, exact=False, hardness="easy"),
        _final(2, execution=True, exact=True, hardness="extra"),
    ]
    metrics = await _task().report(finals, [])
    assert metrics["execution_accuracy_easy"] == 50.0
    assert metrics["n_easy"] == 2.0
    assert metrics["execution_accuracy_extra"] == 100.0
    assert metrics["n_extra"] == 1.0
    # A bucket nothing landed in reports zero over zero, not a crash.
    assert metrics["execution_accuracy_hard"] == 0.0
    assert metrics["n_hard"] == 0.0


@pytest.mark.anyio
async def test_unexecutable_predictions_are_counted():
    """Separates 'wrong answer' from 'no answer', which the headline cannot."""
    finals = [
        _final(0, execution=False, exact=False, hardness="easy", error="syntax error"),
        _final(1, execution=False, exact=False, hardness="easy"),
    ]
    metrics = await _task().report(finals, [])
    assert metrics["n_unexecutable"] == 1.0


# --- staged-path resolution -------------------------------------------------


def test_missing_db_dir_is_a_loud_stop():
    """A run that cannot find its databases must not grade zeros silently."""
    with pytest.raises(ValueError, match="db_dir"):
        _ = _task().db_dir


@pytest.mark.anyio
async def test_full_pipeline_scores_a_correct_answer(tmp_path):
    """preprocess -> infer -> postprocess -> feedback -> report, end to end.

    Runs the real stages against a real (tiny) SQLite database rather than
    stubbing the grader, so a break anywhere in the wiring shows up here.
    """
    db_dir = tmp_path / "database"
    (db_dir / "concert_singer").mkdir(parents=True)
    target = db_dir / "concert_singer" / "concert_singer.sqlite"
    conn = sqlite3.connect(target)
    conn.execute("CREATE TABLE singer (id int, name text)")
    conn.executemany("INSERT INTO singer VALUES (?, ?)", [(1, "Joe"), (2, "Ann")])
    conn.commit()
    conn.close()
    tables = tmp_path / "tables.json"
    tables.write_text(
        '[{"db_id": "concert_singer", "table_names_original": ["singer"], '
        '"table_names": ["singer"], '
        '"column_names_original": [[-1, "*"], [0, "id"], [0, "name"]], '
        '"column_names": [[-1, "*"], [0, "id"], [0, "name"]], '
        '"column_types": ["text", "number", "text"], '
        '"foreign_keys": [], "primary_keys": [1]}]'
    )

    row = {
        "db_id": "concert_singer",
        "query": "SELECT count(*) FROM singer",
        "query_toks": [],
        "query_toks_no_value": [],
        "question": "How many singers do we have?",
        "question_toks": [],
    }
    dataset = SpiderDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([row])})
    )
    model = _ScriptedChatModel(reply="```sql\nSELECT count(*) FROM singer\n```")
    task = SpiderZeroShotGenTask(
        dataset, model, db_dir=str(db_dir), tables_json_path=str(tables)
    )

    ctx = TaskContext(sample_id=0, raw_sample=row)
    pre = await task.preprocess(row, ctx)
    assert "CREATE TABLE singer" in pre["prompt"][0]["content"]
    assert pre["reference"] == "SELECT count(*) FROM singer"

    inf = await task.infer(pre, ctx)
    post = await task.postprocess(inf, ctx)
    assert post["rollouts"][0]["prediction"] == "SELECT count(*) FROM singer"

    finalize, judgement = await task.feedback(post, ctx)
    assert finalize is True
    rollout = judgement["rollouts"][0]
    assert rollout["correct"] is True
    assert rollout["metrics"]["exact_match"] is True
    assert rollout["extra"]["hardness"] == "easy"

    scored = TaskContext(
        sample_id=0,
        raw_sample=row,
        feedback_result=judgement,
        postprocess_result=post,
    )
    metrics = await task.report([scored], [])
    assert metrics["execution_accuracy"] == 100.0
    assert metrics["exact_match"] == 100.0
    assert metrics["n_easy"] == 1.0


def test_db_path_is_built_from_the_staged_dir(tmp_path):
    db_dir = tmp_path / "database"
    (db_dir / "concert_singer").mkdir(parents=True)
    target = db_dir / "concert_singer" / "concert_singer.sqlite"
    sqlite3.connect(target).close()
    tables = tmp_path / "tables.json"
    tables.write_text("[]")

    task = _task(db_dir=str(db_dir), tables_json_path=str(tables))
    assert task._db_path("concert_singer") == str(target)
