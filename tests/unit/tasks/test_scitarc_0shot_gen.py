"""Unit tests for the SciTaRC 0-shot generative task.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from unittest.mock import patch

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

import sieval.datasets.scitarc as scitarc_module
from sieval.community.scitarc import create_language_prompt
from sieval.core.models import Request, Response
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
from sieval.datasets.scitarc import SciTaRCDataset
from sieval.tasks.scitarc_0shot_gen import SciTaRCZeroShotGenTask
from tests.conftest import HandlerTransport

TABLES = [["\\begin{table}\n", "NLLB 64.71\n", "\\end{table}\n"]]


class _ScriptedChatModel(ChatModel):
    """ChatModel returning a fixed reply, recording calls and last kwargs."""

    def __init__(self, reply: str, model: str = "mock"):
        self._reply = reply
        self.calls: list[str] = []
        self.last_kwargs: dict[str, object] = {}
        super().__init__(model=model, api_key="fake")

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_chat")

    async def _stub_arun(self, req: Request) -> Response:
        self.last_kwargs = {"n": req.sampling.n}
        self.calls.append(str(req.input))
        # `finish_reasons` is set because the whole ModelOutput is persisted:
        # it is what separates a reasoning grader that spent its budget
        # thinking from an empty API response.
        return Response(
            texts=(self._reply,) * req.sampling.n,
            finish_reasons=("stop",) * req.sampling.n,
        )


def _row() -> dict:
    return {
        "paper": "2401.06769",
        "relevant_tables": TABLES,
        "tables": TABLES,
        "fulltext": "\\documentclass{article}",
        "question": "Which model is best?",
        "answer": "NLLB-200-1.3B. 64.71",
        "plan": "SELECT all models\nRETURN best",
    }


def _dataset(rows: list[dict] | None = None) -> SciTaRCDataset:
    hf = HFDatasetDict({"test": HFDataset.from_list(rows or [_row()])})
    with patch.object(scitarc_module, "load_dataset", return_value=hf):
        return SciTaRCDataset("jhu-clsp/SciTaRC")


def _task(
    grader_reply: str = '{"reasoning": "match", "score": 1.0}',
    *,
    candidate_reply: str = "Answer: NLLB-200-1.3B. 64.71",
    n: int = 1,
):
    dataset = _dataset()
    model = _ScriptedChatModel(reply=candidate_reply, model="candidate")
    grader = _ScriptedChatModel(reply=grader_reply, model="grader-70b")
    return SciTaRCZeroShotGenTask(dataset, model, grader=grader, n=n), model, grader


# --- grader is mandatory; exact match alone is not the headline ---


def test_build_grader_requires_config():
    with pytest.raises(ValueError, match="requires an LLM grader"):
        SciTaRCZeroShotGenTask._build_grader(None)


def test_build_grader_accepts_mapping_and_model():
    built = SciTaRCZeroShotGenTask._build_grader(
        {"model": "llama-3.3-70b-instruct", "api_key": "fake"}
    )
    assert isinstance(built, ChatModel)
    existing = _ScriptedChatModel(reply="x")
    assert SciTaRCZeroShotGenTask._build_grader(existing) is existing


def test_constructor_accepts_composed_grader_role():
    base, _, grader = _task()
    task = SciTaRCZeroShotGenTask(
        base.dataset, base.model, models_by_role={"grader": grader}
    )
    assert task._grader is grader


# --- preprocess: ONE user turn, upstream's string unchanged ---


@pytest.mark.anyio
async def test_preprocess_sends_a_single_user_turn():
    task, _, _ = _task()
    pre = await task.preprocess(_row(), TaskContext(sample_id=0))

    # Byte-equal to the vendored builder: upstream bakes the persona into the
    # same string as the table block, so a separate system message would change
    # the rendered prompt. Asserting against the builder (not a literal) means
    # this test also fails if the task starts reshaping the prompt.
    assert pre["prompt"] == [
        {
            "role": "user",
            "content": create_language_prompt("Which model is best?", TABLES),
        }
    ]
    assert len(pre["prompt"]) == 1
    # The gold reaches disk from preprocess; raw_sample is never serialized.
    assert pre["reference"] == "NLLB-200-1.3B. 64.71"
    assert pre["extra"]["paper"] == "2401.06769"


@pytest.mark.anyio
async def test_preprocess_uses_relevant_tables_not_every_table():
    """Upstream's ``generate.py`` prompts with ``relevant_tables`` only."""
    row = _row()
    row["tables"] = [["\\begin{table}\nUNRELATED\n\\end{table}\n"]]
    task, _, _ = _task()
    pre = await task.preprocess(row, TaskContext(sample_id=0))
    assert "UNRELATED" not in pre["prompt"][0]["content"]
    assert "NLLB 64.71" in pre["prompt"][0]["content"]


# --- infer forwards ONLY n (decode params are model-layer) ---


@pytest.mark.anyio
async def test_infer_forwards_only_n():
    task, model, _ = _task()
    await task.infer(
        {"prompt": [{"role": "user", "content": "q"}]}, TaskContext(sample_id=0)
    )
    assert model.last_kwargs == {"n": 1}


# --- postprocess: extraction, and blank -> None ---


@pytest.mark.anyio
async def test_postprocess_extracts_after_the_answer_marker():
    task, _, _ = _task()
    out = Response(texts=("reasoning...\nAnswer: 64.71",), finish_reasons=("stop",))
    post = await task.postprocess(out, TaskContext(sample_id=0))
    assert post["rollouts"][0]["prediction"] == "64.71"
    assert post["rollouts"][0]["extracted"] is True


@pytest.mark.anyio
async def test_postprocess_blank_extraction_is_none_not_empty_string():
    """``Answer:`` with nothing after it is a failed extraction, not an answer.

    The reply is non-empty, so a check on the raw text would call this
    extracted; the check has to be on the extracted value.
    """
    task, _, _ = _task()
    out = Response(texts=("I thought about it.\nAnswer:   ",), finish_reasons=("stop",))
    post = await task.postprocess(out, TaskContext(sample_id=0))
    assert post["rollouts"][0]["prediction"] is None
    assert post["rollouts"][0]["extracted"] is False


# --- feedback ---


@pytest.mark.anyio
async def test_feedback_binarises_the_ternary_and_records_provenance():
    task, _, grader = _task()
    ctx = TaskContext(sample_id=0, raw_sample=_row())
    finalize, judgement = await task.feedback(
        build_prediction_record(["NLLB-200-1.3B. 64.71"]), ctx
    )

    assert finalize is True
    fb = judgement["rollouts"][0]
    assert fb["correct"] is True
    assert fb["score"] == 1.0
    assert fb["metrics"]["exact_match"] is True
    assert fb["extra"]["grader_parsed"] is True
    assert fb["extra"]["grader_skipped"] is False
    assert fb["extra"]["grader_reasoning"] == "match"
    # The grader's WHOLE ModelOutput, not a hand-picked reply field.
    assert fb["extra"]["grader_output"]["texts"] == [grader._reply]
    assert fb["extra"]["grader_output"]["finish_reasons"] == ["stop"]
    assert judgement["reference"] == "NLLB-200-1.3B. 64.71"


@pytest.mark.anyio
async def test_feedback_partial_credit_is_not_correct():
    """0.5 is on upstream's scale but not in its accuracy numerator."""
    task, _, _ = _task(grader_reply='{"reasoning": "missing part", "score": 0.5}')
    ctx = TaskContext(sample_id=0, raw_sample=_row())
    _, judgement = await task.feedback(build_prediction_record(["NLLB-200"]), ctx)
    fb = judgement["rollouts"][0]
    assert fb["score"] == 0.5
    assert fb["correct"] is False


@pytest.mark.anyio
async def test_feedback_unparsed_grader_reply_is_flagged():
    task, _, _ = _task(grader_reply="I refuse to evaluate.")
    ctx = TaskContext(sample_id=0, raw_sample=_row())
    _, judgement = await task.feedback(build_prediction_record(["something"]), ctx)
    fb = judgement["rollouts"][0]
    assert fb["correct"] is False
    assert fb["extra"]["grader_parsed"] is False
    # The reply is the evidence separating format drift from a matcher gap.
    assert fb["extra"]["grader_output"]["texts"] == ["I refuse to evaluate."]


@pytest.mark.anyio
async def test_feedback_skips_the_grader_on_a_blank_answer():
    """Upstream's ``if item['prediction'].strip()`` filter: no call, score 0.0.

    Asserting the call count, not just the verdict — a grader that gets asked
    about an empty answer still usually returns 0.0, so the verdict alone
    cannot tell the two apart, and the difference is real money at 371 rows.
    """
    task, _, grader = _task()
    ctx = TaskContext(sample_id=0, raw_sample=_row())
    _, judgement = await task.feedback(build_prediction_record([None]), ctx)

    assert grader.calls == []
    fb = judgement["rollouts"][0]
    assert fb["correct"] is False
    assert fb["score"] == 0.0
    assert fb["extra"]["grader_skipped"] is True
    assert fb["metrics"]["exact_match"] is False


@pytest.mark.anyio
async def test_feedback_exact_match_is_case_sensitive():
    task, _, _ = _task(grader_reply='{"reasoning": "same", "score": 1.0}')
    ctx = TaskContext(sample_id=0, raw_sample=_row())
    _, judgement = await task.feedback(
        build_prediction_record(["nllb-200-1.3b. 64.71"]), ctx
    )
    fb = judgement["rollouts"][0]
    # The grader forgives case; upstream's EM does not. Both columns are
    # reported, so they must be able to disagree on the same rollout.
    assert fb["correct"] is True
    assert fb["metrics"]["exact_match"] is False


@pytest.mark.anyio
async def test_feedback_grades_every_rollout():
    task, _, grader = _task(n=2)
    ctx = TaskContext(sample_id=0, raw_sample=_row())
    _, judgement = await task.feedback(
        build_prediction_record(["NLLB-200-1.3B. 64.71", "wrong"]), ctx
    )
    assert len(grader.calls) == 2
    assert judgement["n_rollouts"] == 2


# --- report ---


def _final(
    sample_id: int,
    *,
    correct: bool,
    score: float,
    em: bool,
    skipped: bool = False,
    parsed: bool = True,
) -> TaskContext:
    return TaskContext(
        sample_id=sample_id,
        feedback_result=build_judgement_record(
            "gold",
            [
                build_rollout_judgement(
                    0,
                    correct,
                    score=score,
                    metrics={"exact_match": em},
                    extra={"grader_skipped": skipped, "grader_parsed": parsed},
                )
            ],
        ),
        postprocess_result=build_prediction_record([None if skipped else "x"]),
    )


@pytest.mark.anyio
async def test_report_counts_fails_in_the_denominator():
    task, _, _ = _task()
    finals = [
        _final(0, correct=True, score=1.0, em=True),
        _final(1, correct=False, score=0.0, em=False),
    ]
    fails = [TaskContext(sample_id=9)]
    report = await task.report(finals, fails)

    # n = (2 finals + 1 fail) * 1 = 3; 1 correct => 33.33. A `len(finals)`
    # denominator would give 50.0, so this discriminates.
    assert report["n"] == 3
    assert report["fails"] == 1
    assert report["accuracy"] == pytest.approx(33.33, abs=1e-2)
    assert report["score"] == report["accuracy"]
    assert report["exact_match"] == pytest.approx(33.33, abs=1e-2)
    assert report["score_key"] == "accuracy"
    assert report["denominator_policy"] == "requested"


@pytest.mark.anyio
async def test_report_separates_partial_from_wrong():
    task, _, _ = _task()
    finals = [
        _final(0, correct=False, score=0.5, em=False),
        _final(1, correct=False, score=0.0, em=False),
    ]
    report = await task.report(finals, [])
    assert report["accuracy"] == 0.0
    assert report["partial"] == 50.0


@pytest.mark.anyio
async def test_report_separates_unparsed_grader_from_skipped_grader():
    """Three ways to score 0.0, and the report must not conflate them.

    A wrong answer, a grader that returned nothing readable, and an answer the
    grader was never shown all land at 0.0. ``n_grader_unparsed`` counts only
    the middle one; the last is ``n_unextracted``, and ``n_graded`` excludes it.
    """
    task, _, _ = _task()
    finals = [
        _final(0, correct=False, score=0.0, em=False),
        _final(1, correct=False, score=0.0, em=False, parsed=False),
        _final(2, correct=False, score=0.0, em=False, skipped=True, parsed=False),
    ]
    report = await task.report(finals, [])

    assert report["n_graded"] == 2
    assert report["n_grader_unparsed"] == 1
    assert report["n_unextracted"] == 1


@pytest.mark.anyio
async def test_report_on_an_empty_run_does_not_divide_by_zero():
    task, _, _ = _task()
    report = await task.report([], [])
    assert report["n"] == 0
    assert report["accuracy"] == 0.0
    assert report["score_key"] == "accuracy"
    assert report["denominator_policy"] == "requested"


@pytest.mark.anyio
async def test_report_weights_the_denominator_by_n():
    task, _, _ = _task(n=2)
    finals = [_final(0, correct=True, score=1.0, em=True)]
    fails = [TaskContext(sample_id=9)]
    report = await task.report(finals, fails)
    # (1 final + 1 fail) * n=2 = 4 requested attempts, 1 correct => 25.0.
    assert report["n"] == 4
    assert report["accuracy"] == 25.0
