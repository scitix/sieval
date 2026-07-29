"""Unit tests for the AA-LCR 0-shot generative task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.community.aa_lcr import aggregate_metrics, parse_grade
from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import TaskContext
from sieval.datasets.aa_lcr import AALCRDataset, AALCRDatasetSample
from sieval.tasks.aa_lcr_0shot_gen import (
    AALCRZeroShotGenTask,
    GradeFeedback,
)


class _ScriptedChatModel(ChatModel):
    """ChatModel returning a fixed reply, recording the last agenerate kwargs."""

    def __init__(self, reply: str, model: str = "mock"):
        super().__init__(model=model, api_key="fake")
        self._reply = reply
        self.last_kwargs: dict[str, object] = {}

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = prompt
        self.last_kwargs = dict(kwargs)
        return ModelOutput(model=self.meta(), texts=[self._reply])

    async def _alogprobs_impl(
        self, prompt, *, max_tokens=1, logprobs=5, echo=True, temperature=0.0, **kwargs
    ) -> ModelOutput:
        _ = (prompt, max_tokens, logprobs, echo, temperature, kwargs)
        return ModelOutput(model=self.meta(), texts=[""])


def _sample() -> AALCRDatasetSample:
    return {
        "question_id": 7,
        "document_category": "Academia",
        "document_set_id": "ac_markets",
        "question": "What is the trend?",
        "answer": "Rising",
        "documents": ["doc one", "doc two"],
        "data_source_filenames": "one.txt;two.txt",
        "input_tokens": 1234,
    }


def _task(answer_reply: str = "Rising", grader_reply: str = "CORRECT"):
    dataset = AALCRDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    model = _ScriptedChatModel(reply=answer_reply, model="candidate")
    grader = _ScriptedChatModel(reply=grader_reply, model="qwen3-235b")
    task = AALCRZeroShotGenTask(dataset, model, grader=grader)
    return task, grader


# --- grader is mandatory; no deterministic fallback ---


def test_build_grader_requires_config():
    with pytest.raises(ValueError, match="requires an LLM grader"):
        AALCRZeroShotGenTask._build_grader(None)


def test_build_grader_accepts_mapping_and_model():
    built = AALCRZeroShotGenTask._build_grader(
        {"model": "qwen3-235b", "api_key": "fake"}
    )
    assert isinstance(built, ChatModel)
    existing = _ScriptedChatModel(reply="CORRECT")
    assert AALCRZeroShotGenTask._build_grader(existing) is existing


# --- preprocess: documents + question assembled into one user turn ---


@pytest.mark.anyio
async def test_preprocess_builds_document_prompt():
    task, _ = _task()
    messages = await task.preprocess(
        _sample(), TaskContext(sample_id=0, raw_sample=_sample())
    )
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    # Documents wrapped + ordered, and the question inlined.
    assert "BEGIN DOCUMENT 1:\ndoc one\nEND DOCUMENT 1" in content
    assert "BEGIN DOCUMENT 2:\ndoc two\nEND DOCUMENT 2" in content
    assert "What is the trend?" in content


# --- infer forwards n to the candidate model ---


@pytest.mark.anyio
async def test_infer_forwards_n():
    dataset = AALCRDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    model = _ScriptedChatModel(reply="x", model="candidate")
    grader = _ScriptedChatModel(reply="CORRECT", model="grader")
    task = AALCRZeroShotGenTask(dataset, model, grader=grader, n=3)
    await task.infer([{"role": "user", "content": "q"}], TaskContext(sample_id=0))
    assert model.last_kwargs.get("n") == 3


# --- feedback: grades each answer via the grader, records provenance ---


@pytest.mark.anyio
async def test_feedback_grades_and_records_provenance():
    task, _ = _task(grader_reply="CORRECT")
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    finalize, feedbacks = await task.feedback(["Rising"], ctx)

    assert finalize is True
    assert len(feedbacks) == 1
    fb: GradeFeedback = feedbacks[0]
    assert fb["grade"] == "CORRECT"
    assert fb["gold"] == "Rising"
    assert fb["predicted"] == "Rising"
    assert fb["grader_model"] == "qwen3-235b"
    assert fb["question_id"] == 7


@pytest.mark.anyio
async def test_feedback_unrecognized_grader_reply_is_incorrect():
    task, _ = _task(grader_reply="")
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    _, feedbacks = await task.feedback(["some answer"], ctx)
    assert feedbacks[0]["grade"] == "INCORRECT"


@pytest.mark.anyio
@pytest.mark.parametrize("empty", ["", "   ", "\n\n"])
async def test_feedback_empty_answer_is_incorrect_without_grading(empty: str):
    # A truncated/empty answer must be INCORRECT and must NOT reach the grader —
    # even a grader that would say CORRECT cannot flip an empty answer (this is
    # the fix for empty truncated outputs being spuriously graded CORRECT).
    task, grader = _task(grader_reply="CORRECT")  # grader would say CORRECT
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    _, feedbacks = await task.feedback([empty], ctx)
    assert feedbacks[0]["grade"] == "INCORRECT"
    assert feedbacks[0]["predicted"] == empty
    # Grader was bypassed: its last_kwargs stays empty (never called).
    assert grader.last_kwargs == {}


# --- report: accuracy over graded + failed samples ---


@pytest.mark.anyio
async def test_report_accuracy_matches_hand_computation():
    task, _ = _task()
    grades = ["CORRECT", "CORRECT", "CORRECT", "INCORRECT"]
    finals = [
        TaskContext(
            sample_id=i,
            feedback_result=[
                {
                    "grade": g,
                    "gold": "",
                    "predicted": "",
                    "grader_model": "m",
                    "question_id": i,
                }
            ],
        )
        for i, g in enumerate(grades)
    ]
    report = await task.report(finals, fails=[])

    # 3 correct / 4 graded = 75%.
    assert report["n_graded"] == 4
    assert report["fails"] == 0
    assert report["accuracy"] == pytest.approx(75.0)
    assert report["correct"] == pytest.approx(75.0)
    assert report["incorrect"] == pytest.approx(25.0)
    assert report["score"] == report["accuracy"]


@pytest.mark.anyio
async def test_report_counts_fails_as_incorrect():
    # Failed samples must dilute accuracy (full-set metric), not be excluded.
    task, _ = _task()  # n=1
    finals = [
        TaskContext(
            sample_id=i,
            feedback_result=[
                {
                    "grade": g,
                    "gold": "",
                    "predicted": "",
                    "grader_model": "m",
                    "question_id": i,
                }
            ],
        )
        for i, g in enumerate(["CORRECT", "CORRECT"])
    ]
    fails = [TaskContext(sample_id=10), TaskContext(sample_id=11)]
    report = await task.report(finals, fails)

    # 2 correct + 2 fails-as-INCORRECT => 4 units, accuracy 50%.
    # Excluding fails would give 100%, so this assertion discriminates.
    assert report["n_graded"] == 2
    assert report["fails"] == 2
    assert report["accuracy"] == pytest.approx(50.0)
    assert report["incorrect"] == pytest.approx(50.0)


@pytest.mark.anyio
async def test_report_fails_weighted_by_n():
    # Each failed sample stands in for its n requested attempts.
    dataset = AALCRDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    model = _ScriptedChatModel(reply="x", model="candidate")
    grader = _ScriptedChatModel(reply="CORRECT", model="grader")
    task = AALCRZeroShotGenTask(dataset, model, grader=grader, n=2)
    finals = [
        TaskContext(
            sample_id=0,
            feedback_result=[
                {
                    "grade": "CORRECT",
                    "gold": "",
                    "predicted": "",
                    "grader_model": "m",
                    "question_id": 0,
                },
                {
                    "grade": "CORRECT",
                    "gold": "",
                    "predicted": "",
                    "grader_model": "m",
                    "question_id": 0,
                },
            ],
        )
    ]
    fails = [TaskContext(sample_id=1)]
    report = await task.report(finals, fails)

    # 2 correct attempts + n*1 = 2 INCORRECT => 4 units, accuracy 50%.
    # A per-sample (unweighted) count would give 3 units and accuracy 66.7.
    assert report["n_graded"] == 2
    assert report["accuracy"] == pytest.approx(50.0)


def test_report_empty_is_zero():
    # aggregate_metrics is the pure kernel report() delegates to.
    m = aggregate_metrics([])
    assert m["accuracy"] == 0.0
    assert parse_grade("nonsense") == "INCORRECT"
