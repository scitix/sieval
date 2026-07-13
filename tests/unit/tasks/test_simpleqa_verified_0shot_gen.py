"""Unit tests for the SimpleQA Verified 0-shot generative task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.community.simpleqa_verified import aggregate_metrics, parse_grade
from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import TaskContext
from sieval.datasets.simpleqa_verified import (
    SimpleQAVerifiedDataset,
    SimpleQAVerifiedDatasetSample,
)
from sieval.tasks.simpleqa_verified_0shot_gen import (
    GradeFeedback,
    SimpleQAVerifiedZeroShotGenTask,
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


def _sample() -> SimpleQAVerifiedDatasetSample:
    return {
        "original_index": 0,
        "problem": "Who wrote Hamlet?",
        "answer": "William Shakespeare",
        "topic": "Art",
        "answer_type": "Person",
        "multi_step": False,
        "requires_reasoning": False,
        "urls": "[]",
    }


def _task(answer_reply: str = "William Shakespeare", grader_reply: str = "A"):
    dataset = SimpleQAVerifiedDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    model = _ScriptedChatModel(reply=answer_reply, model="candidate")
    grader = _ScriptedChatModel(reply=grader_reply, model="grader-4.1")
    task = SimpleQAVerifiedZeroShotGenTask(dataset, model, grader=grader)
    return task, grader


# --- grader is mandatory; no deterministic fallback ---


def test_build_grader_requires_config():
    with pytest.raises(ValueError, match="requires an LLM grader"):
        SimpleQAVerifiedZeroShotGenTask._build_grader(None)


def test_build_grader_accepts_mapping_and_model():
    built = SimpleQAVerifiedZeroShotGenTask._build_grader(
        {"model": "gpt-4.1", "api_key": "fake"}
    )
    assert isinstance(built, ChatModel)
    existing = _ScriptedChatModel(reply="A")
    assert SimpleQAVerifiedZeroShotGenTask._build_grader(existing) is existing


# --- preprocess: bare problem as a single user turn (no template) ---


@pytest.mark.anyio
async def test_preprocess_single_user_turn():
    task, _ = _task()
    messages = await task.preprocess(
        _sample(), TaskContext(sample_id=0, raw_sample=_sample())
    )
    assert messages == [{"role": "user", "content": "Who wrote Hamlet?"}]


# --- infer forwards n to the candidate model ---


@pytest.mark.anyio
async def test_infer_forwards_n():
    dataset = SimpleQAVerifiedDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    model = _ScriptedChatModel(reply="x", model="candidate")
    grader = _ScriptedChatModel(reply="A", model="grader")
    task = SimpleQAVerifiedZeroShotGenTask(dataset, model, grader=grader, n=3)
    await task.infer([{"role": "user", "content": "q"}], TaskContext(sample_id=0))
    assert model.last_kwargs.get("n") == 3


# --- feedback: grades each answer via the grader, records provenance ---


@pytest.mark.anyio
async def test_feedback_grades_and_records_provenance():
    task, _ = _task(grader_reply="A")
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    finalize, feedbacks = await task.feedback(["William Shakespeare"], ctx)

    assert finalize is True
    assert len(feedbacks) == 1
    fb: GradeFeedback = feedbacks[0]
    assert fb["grade"] == "CORRECT"
    assert fb["gold"] == "William Shakespeare"
    assert fb["predicted"] == "William Shakespeare"
    assert fb["grader_model"] == "grader-4.1"


@pytest.mark.anyio
async def test_feedback_empty_grader_reply_is_not_attempted():
    task, _ = _task(grader_reply="")
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    _, feedbacks = await task.feedback(["some answer"], ctx)
    assert feedbacks[0]["grade"] == "NOT_ATTEMPTED"


# --- report: F1 aggregation matches simple-evals ---


@pytest.mark.anyio
async def test_report_f1_matches_hand_computation():
    task, _ = _task()
    grades = ["CORRECT", "CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
    finals = [
        TaskContext(
            sample_id=i,
            feedback_result=[
                {"grade": g, "gold": "", "predicted": "", "grader_model": "m"}
            ],
        )
        for i, g in enumerate(grades)
    ]
    report = await task.report(finals, fails=[])

    # correct=0.5, incorrect=0.25 -> acc_given_attempted=0.5/0.75=0.6667
    # f1 = 2*0.6667*0.5 / (0.6667+0.5) = 0.5714
    assert report["n_graded"] == 4
    assert report["fails"] == 0
    assert report["correct"] == pytest.approx(50.0)
    assert report["accuracy_given_attempted"] == pytest.approx(66.6667, abs=1e-3)
    assert report["f1"] == pytest.approx(57.1429, abs=1e-3)
    assert report["score"] == report["f1"]


def test_report_empty_is_zero():
    # aggregate_metrics is the pure kernel report() delegates to.
    m = aggregate_metrics([])
    assert m["f1"] == 0.0
    assert parse_grade("C") == "NOT_ATTEMPTED"
