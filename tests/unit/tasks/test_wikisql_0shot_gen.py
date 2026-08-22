"""WikiSQL 0-shot task: prompt contract, extractor, verdicts, report.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import json
from dataclasses import dataclass

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import Request, Response
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import TaskContext
from sieval.core.tasks.meta import get_task_meta
from sieval.core.tasks.metrics import DENOMINATOR_REQUESTED
from sieval.datasets.wikisql import WikiSQLDataset
from sieval.tasks.wikisql_0shot_gen import (
    WikiSQLZeroShotGenTask,
    build_prompt,
    extract_logical_form,
    render_schema,
)


class _StubChatModel(ChatModel):
    def __init__(self, reply: str = "{}"):
        super().__init__(model="mock-chat", api_key="fake")
        self._reply = reply

    def _build_default_transport(self):
        from tests.conftest import HandlerTransport

        return HandlerTransport(self._stub_arun, "openai_chat")

    async def _stub_arun(self, req: Request) -> Response:
        return Response(texts=(self._reply,) * req.sampling.n)


@dataclass
class _Out:
    texts: list[str]


@dataclass
class _Final:
    postprocess_result: dict
    feedback_result: dict


_HEADER = ["Player", "Nationality", "Points"]
_TYPES = ["text", "text", "real"]
_ROWS = [
    ["Terrence Ross", "United States", 12],
    ["Jose Calderon", "Spain", 8],
]
#: gold: SELECT col1 WHERE col0 = 'Terrence Ross'  ->  ["united states"]
_GOLD = {"sel": 1, "agg": 0, "conds": [[0, 0, "Terrence Ross"]]}


def _sample(gold=None) -> dict:
    return {
        "phase": 1,
        "table_id": "1-234-5",
        "question": "What is Terrence Ross' nationality?",
        "sql_json": json.dumps(gold if gold is not None else _GOLD),
        "header": _HEADER,
        "types": _TYPES,
        "rows_json": json.dumps(_ROWS),
        "page_title": "Roster",
        "section_title": None,
        "caption": None,
        "name": None,
        "page_id": None,
    }


def _task(reply: str = "{}") -> WikiSQLZeroShotGenTask:
    ds = WikiSQLDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([_sample()])})
    )
    return WikiSQLZeroShotGenTask(ds, _StubChatModel(reply))


async def _judge(reply: str, gold=None, n: int = 1):
    """Run postprocess + feedback on one reply, returning the judgement."""
    raw = _sample(gold)
    task = _task(reply)
    task._n = n
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    post = await task.postprocess(_Out([reply] * n), ctx)
    _, judgement = await task.feedback(post, ctx)
    return post, judgement


# --- task metadata ---------------------------------------------------------


def test_task_metadata_declares_what_the_repo_checks():
    meta = get_task_meta(WikiSQLZeroShotGenTask)
    assert meta.name == "wikisql_0shot_gen"
    assert meta.dataset == "wikisql"
    assert meta.model_type == "chat"
    assert meta.n_shot == 0
    # No published anchor exists for a 0-shot chat protocol -- only the grader
    # is anchored, so the task ships experimental rather than stable.
    assert meta.status == "experimental"
    assert meta.reference_kind == "value"
    assert meta.deps_group == "wikisql"


# --- the prompt ------------------------------------------------------------


def test_schema_is_rendered_with_indices_because_predictions_use_them():
    rendered = render_schema(_HEADER, _TYPES)
    assert rendered.splitlines() == [
        "    0: Player [text]",
        "    1: Nationality [text]",
        "    2: Points [real]",
    ]


def test_prompt_carries_no_table_content():
    """Upstream's leaderboard rule, and the easiest thing to regress.

    "your models only use the table schema and question during inference. That
    is they do *not* use the table content" -- so a cell value appearing in the
    prompt is a protocol violation, not a formatting detail. The rows travel on
    the sample because the *grader* needs them.

    The question deliberately shares no text with any cell: WikiSQL questions
    normally *do* quote their condition values (upstream's check.py asserts it),
    so asserting over a realistic question would fail on the question's own
    words rather than on leaked content.
    """
    prompt = build_prompt("Which entry is listed first?", _HEADER, _TYPES)
    for row in _ROWS:
        for cell in row:
            assert str(cell) not in prompt, cell
    # Column names and types DO belong in the prompt -- that is the schema.
    assert "Nationality" in prompt
    assert "[real]" in prompt


def test_prompt_teaches_both_index_encodings():
    prompt = build_prompt("q", _HEADER, _TYPES)
    for token in ("MAX", "MIN", "COUNT", "SUM", "AVG"):
        assert token in prompt
    assert "0 for =, 1 for >, 2 for <" in prompt
    # `OP` is upstream's unusable fourth operator: in the scorer, not the prompt.
    assert "3 for OP" not in prompt


@pytest.mark.anyio
async def test_preprocess_sends_one_user_turn_and_records_the_gold():
    raw = _sample()
    task = _task()
    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))
    assert [m["role"] for m in pre["prompt"]] == ["user"]
    assert raw["question"] in pre["prompt"][0]["content"]
    assert pre["reference"] == raw["sql_json"]
    assert pre["extra"]["table_id"] == "1-234-5"


# --- the extractor ---------------------------------------------------------


def test_prompt_format_spec_is_not_itself_extractable():
    """Otherwise a reply quoting the instructions would score as an answer.

    This is the failure mode where a task grades its own prompt: the spec uses
    `<int>` placeholders precisely so it cannot parse as JSON.
    """
    assert extract_logical_form(build_prompt("q", _HEADER, _TYPES)) is None


@pytest.mark.parametrize(
    ("label", "reply", "expected"),
    [
        (
            "bare",
            '{"sel": 1, "agg": 0, "conds": []}',
            {"sel": 1, "agg": 0, "conds": []},
        ),
        (
            "fenced",
            '```json\n{"sel": 1, "agg": 0, "conds": []}\n```',
            {"sel": 1, "agg": 0, "conds": []},
        ),
        (
            "prose then answer",
            'The column is 1.\n{"sel": 1, "agg": 0, "conds": []}',
            {"sel": 1, "agg": 0, "conds": []},
        ),
        (
            "draft then correction takes the last",
            '{"sel": 9, "agg": 0, "conds": []}\nActually:\n'
            '{"sel": 1, "agg": 0, "conds": []}',
            {"sel": 1, "agg": 0, "conds": []},
        ),
        (
            "wrapped answer takes the inner object",
            '{"answer": {"sel": 1, "agg": 0, "conds": []}}',
            {"sel": 1, "agg": 0, "conds": []},
        ),
        ("missing a required key", '{"sel": 1, "agg": 0}', None),
        ("sql text instead of a form", "SELECT Nationality FROM t", None),
        ("nothing at all", "", None),
        ("null, as upstream's own error lines carry", "null", None),
    ],
)
def test_extractor(label, reply, expected):
    assert extract_logical_form(reply) == expected, label


# --- verdicts --------------------------------------------------------------


@pytest.mark.anyio
async def test_exact_gold_scores_both_metrics():
    _, judgement = await _judge(json.dumps(_GOLD))
    rollout = judgement["rollouts"][0]
    assert rollout["metrics"] == {"ex": True, "lf": True}
    assert rollout["correct"] is True
    assert judgement["reference"] == json.dumps(_GOLD)


@pytest.mark.anyio
async def test_condition_order_does_not_matter_to_either_metric():
    gold = {"sel": 2, "agg": 0, "conds": [[0, 0, "Jose Calderon"], [1, 0, "Spain"]]}
    reply = json.dumps(
        {"sel": 2, "agg": 0, "conds": [[1, 0, "Spain"], [0, 0, "Jose Calderon"]]}
    )
    _, judgement = await _judge(reply, gold=gold)
    assert judgement["rollouts"][0]["metrics"] == {"ex": True, "lf": True}


@pytest.mark.anyio
async def test_a_different_form_reaching_the_same_rows_is_ex_only():
    """Exactly the gap between the two published columns.

    Selecting nationality by name and by points returns the same cell, so
    execution accuracy credits it and logical-form accuracy does not.
    """
    reply = json.dumps({"sel": 1, "agg": 0, "conds": [[2, 0, 12]]})
    _, judgement = await _judge(reply)
    assert judgement["rollouts"][0]["metrics"] == {"ex": True, "lf": False}
    assert judgement["rollouts"][0]["correct"] is True


@pytest.mark.anyio
async def test_a_wrong_query_scores_neither():
    reply = json.dumps({"sel": 1, "agg": 0, "conds": [[0, 0, "Jose Calderon"]]})
    _, judgement = await _judge(reply)
    assert judgement["rollouts"][0]["metrics"] == {"ex": False, "lf": False}


@pytest.mark.anyio
async def test_unextracted_reply_scores_zero_and_carries_no_execution_error():
    """`extracted` on the prediction record is the durable flag, not a duplicate.

    `n_unextracted` is counted from it by `health_metrics`, so the judgement
    deliberately does not restate it.
    """
    post, judgement = await _judge("I cannot answer that.")
    assert post["rollouts"][0].get("prediction") is None
    assert post["rollouts"][0]["extracted"] is False
    rollout = judgement["rollouts"][0]
    assert rollout["metrics"] == {"ex": False, "lf": False}
    assert rollout["score"] == 0.0
    assert "extra" not in rollout


@pytest.mark.anyio
async def test_unexecutable_prediction_is_scored_wrong_and_the_reason_kept():
    """Upstream's `except Exception` branch, with the reason retained.

    A column index past the table's width is the common real case -- 78 of
    upstream's own example predictions do this.
    """
    reply = json.dumps({"sel": 99, "agg": 0, "conds": []})
    post, judgement = await _judge(reply)
    # It extracted fine -- the failure is downstream of extraction.
    assert post["rollouts"][0]["extracted"] is True
    rollout = judgement["rollouts"][0]
    assert rollout["metrics"] == {"ex": False, "lf": False}
    assert "sel" in rollout["extra"]["execution_error"]


@pytest.mark.anyio
async def test_a_broken_gold_fails_the_sample_rather_than_scoring_it():
    """Upstream's check.py asserts every gold executes, so this is bad data.

    Scoring against a gold we could not compute would be worse than failing.
    """
    from sieval.core.tasks import NonRetriableSampleError

    with pytest.raises(NonRetriableSampleError, match="gold query failed"):
        await _judge(json.dumps(_GOLD), gold={"sel": 99, "agg": 0, "conds": []})


@pytest.mark.anyio
async def test_every_rollout_is_judged_when_n_is_greater_than_one():
    _, judgement = await _judge(json.dumps(_GOLD), n=3)
    assert judgement["n_rollouts"] == 3
    assert judgement["n_correct"] == 3
    assert [r["index"] for r in judgement["rollouts"]] == [0, 1, 2]


# --- the report ------------------------------------------------------------


@pytest.mark.anyio
async def test_report_declares_its_headline_and_denominator():
    post, judgement = await _judge(json.dumps(_GOLD))
    task = _task()
    report = await task.report(
        [_Final(postprocess_result=post, feedback_result=judgement)], fails=[]
    )
    assert report["score"] == report["ex_accuracy"] == 100.0
    assert report["lf_accuracy"] == 100.0
    assert report["score_key"] == "ex_accuracy"
    assert report["denominator_policy"] == DENOMINATOR_REQUESTED
    assert report["n"] == 1.0
    assert report["n_unextracted"] == 0.0
    assert report["n_execution_errors"] == 0.0


@pytest.mark.anyio
async def test_a_pipeline_failure_counts_against_the_denominator():
    """Upstream's denominator is every line of the prediction file."""
    post, judgement = await _judge(json.dumps(_GOLD))
    task = _task()
    report = await task.report(
        [_Final(postprocess_result=post, feedback_result=judgement)],
        fails=[object()],
    )
    assert report["n"] == 2.0
    assert report["fails"] == 1.0
    assert report["ex_accuracy"] == 50.0


@pytest.mark.anyio
async def test_report_separates_unextracted_from_unexecutable():
    """Both score zero; without the pair they are indistinguishable."""
    task = _task()
    finals = []
    for reply in ("I cannot answer.", json.dumps({"sel": 99, "agg": 0, "conds": []})):
        post, judgement = await _judge(reply)
        finals.append(_Final(postprocess_result=post, feedback_result=judgement))
    report = await task.report(finals, fails=[])
    assert report["ex_accuracy"] == 0.0
    assert report["n_unextracted"] == 1.0
    assert report["n_execution_errors"] == 1.0


@pytest.mark.anyio
async def test_empty_run_reports_zeroes_and_still_declares():
    task = _task()
    report = await task.report([], fails=[])
    assert report["score"] == 0.0
    assert report["score_key"] == "ex_accuracy"
    assert report["denominator_policy"] == DENOMINATOR_REQUESTED
