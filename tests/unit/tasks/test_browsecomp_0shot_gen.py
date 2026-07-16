"""Unit tests for the BrowseComp 0-shot generative task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.community.browsecomp import aggregate_metrics, parse_grade
from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import TaskContext
from sieval.datasets.browsecomp import (
    BrowseCompDataset,
    BrowseCompDatasetSample,
)
from sieval.tasks.browsecomp_0shot_gen import (
    BrowseCompZeroShotGenTask,
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


def _sample() -> BrowseCompDatasetSample:
    return {
        "original_index": 0,
        "problem": "Who wrote Hamlet?",
        "answer": "William Shakespeare",
        "problem_topic": "Art",
    }


def _task(
    answer_reply: str = "Exact Answer: William Shakespeare",
    grader_reply: str = "correct: yes",
):
    dataset = BrowseCompDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    model = _ScriptedChatModel(reply=answer_reply, model="candidate")
    grader = _ScriptedChatModel(reply=grader_reply, model="grader-4.1")
    task = BrowseCompZeroShotGenTask(dataset, model, grader=grader)
    return task, grader


# --- grader is mandatory; no deterministic fallback ---


def test_build_grader_requires_config():
    with pytest.raises(ValueError, match="requires an LLM grader"):
        BrowseCompZeroShotGenTask._build_grader(None)


def test_build_grader_accepts_mapping_and_model():
    built = BrowseCompZeroShotGenTask._build_grader(
        {"model": "gpt-4.1", "api_key": "fake"}
    )
    assert isinstance(built, ChatModel)
    existing = _ScriptedChatModel(reply="correct: no")
    assert BrowseCompZeroShotGenTask._build_grader(existing) is existing


# --- preprocess: wraps the question in the BrowseComp QUERY_TEMPLATE ---


@pytest.mark.anyio
async def test_preprocess_wraps_query_template():
    task, _ = _task()
    messages = await task.preprocess(
        _sample(), TaskContext(sample_id=0, raw_sample=_sample())
    )
    assert len(messages) == 1 and messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert content.startswith("Who wrote Hamlet?")
    assert "Exact Answer:" in content and "Confidence:" in content


# --- infer forwards n to the candidate model ---


@pytest.mark.anyio
async def test_infer_forwards_n():
    dataset = BrowseCompDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    model = _ScriptedChatModel(reply="x", model="candidate")
    grader = _ScriptedChatModel(reply="correct: yes", model="grader")
    task = BrowseCompZeroShotGenTask(dataset, model, grader=grader, n=3)
    await task.infer([{"role": "user", "content": "q"}], TaskContext(sample_id=0))
    assert model.last_kwargs.get("n") == 3


# --- feedback: yes/no grading + confidence + provenance ---


@pytest.mark.anyio
async def test_feedback_grades_yes_and_records_provenance():
    task, _ = _task(grader_reply="reasoning: matches\ncorrect: yes\nconfidence: 90")
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    finalize, feedbacks = await task.feedback(
        ["Exact Answer: William Shakespeare"], ctx
    )

    assert finalize is True
    assert len(feedbacks) == 1
    fb: GradeFeedback = feedbacks[0]
    assert fb["grade"] == "CORRECT"
    assert fb["confidence"] == 90
    assert fb["gold"] == "William Shakespeare"
    assert fb["predicted"] == "Exact Answer: William Shakespeare"
    assert fb["grader_model"] == "grader-4.1"


@pytest.mark.anyio
async def test_feedback_defaults_to_incorrect_without_verdict():
    # No recognizable "correct: yes|no" -> default INCORRECT (matches upstream).
    task, _ = _task(grader_reply="the grader rambled without a verdict")
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    _, feedbacks = await task.feedback(["some answer"], ctx)
    assert feedbacks[0]["grade"] == "INCORRECT"


# --- report: accuracy over the full requested set ---


@pytest.mark.anyio
async def test_report_accuracy_matches_hand_computation():
    task, _ = _task()
    grades = ["CORRECT", "INCORRECT", "INCORRECT"]
    finals = [
        TaskContext(
            sample_id=i,
            feedback_result=[
                {
                    "grade": g,
                    "confidence": 100,
                    "gold": "",
                    "predicted": "",
                    "grader_model": "m",
                }
            ],
        )
        for i, g in enumerate(grades)
    ]
    report = await task.report(finals, fails=[])
    assert report["n_graded"] == 3
    assert report["fails"] == 0
    assert report["accuracy"] == pytest.approx(33.3333, abs=1e-3)
    assert report["correct"] == pytest.approx(33.3333, abs=1e-3)
    assert report["score"] == report["accuracy"]


@pytest.mark.anyio
async def test_report_counts_fails_as_incorrect():
    # Failed samples dilute accuracy (full-set metric), not excluded.
    task, _ = _task()  # n=1
    finals = [
        TaskContext(
            sample_id=0,
            feedback_result=[
                {
                    "grade": "CORRECT",
                    "confidence": 100,
                    "gold": "",
                    "predicted": "",
                    "grader_model": "m",
                }
            ],
        )
    ]
    fails = [
        TaskContext(sample_id=10),
        TaskContext(sample_id=11),
        TaskContext(sample_id=12),
    ]
    report = await task.report(finals, fails)
    # 1 correct + 3 fails-as-INCORRECT => 4 units, accuracy 25%.
    # Excluding fails would give 100%, so this discriminates.
    assert report["n_graded"] == 1
    assert report["fails"] == 3
    assert report["accuracy"] == pytest.approx(25.0)


@pytest.mark.anyio
async def test_report_fails_weighted_by_n():
    dataset = BrowseCompDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    model = _ScriptedChatModel(reply="x", model="candidate")
    grader = _ScriptedChatModel(reply="correct: yes", model="grader")
    task = BrowseCompZeroShotGenTask(dataset, model, grader=grader, n=2)
    finals = [
        TaskContext(
            sample_id=0,
            feedback_result=[
                {
                    "grade": "CORRECT",
                    "confidence": 100,
                    "gold": "",
                    "predicted": "",
                    "grader_model": "m",
                },
                {
                    "grade": "CORRECT",
                    "confidence": 100,
                    "gold": "",
                    "predicted": "",
                    "grader_model": "m",
                },
            ],
        )
    ]
    fails = [TaskContext(sample_id=1)]
    report = await task.report(finals, fails)
    # 2 correct attempts + n*1 = 2 INCORRECT => 4 units, accuracy 50%.
    assert report["n_graded"] == 2
    assert report["accuracy"] == pytest.approx(50.0)


def test_aggregate_and_parse_kernels():
    assert aggregate_metrics([])["accuracy"] == 0.0
    assert parse_grade("correct: yes") == "CORRECT"
    assert parse_grade("no verdict here") == "INCORRECT"
