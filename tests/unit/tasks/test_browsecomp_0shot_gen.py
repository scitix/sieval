"""Unit tests for the BrowseComp 0-shot generative task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.community.browsecomp import aggregate_metrics, parse_grade
from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
from sieval.datasets.browsecomp import (
    BrowseCompDataset,
    BrowseCompDatasetSample,
)
from sieval.tasks.browsecomp_0shot_gen import (
    BrowseCompZeroShotGenTask,
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
        # `finish_reasons` is set because the judge-family migration exists to
        # persist the grader's WHOLE ModelOutput -- #51's flat `grader_reply`
        # dropped everything but the text.
        return ModelOutput(
            model=self.meta(), texts=[self._reply], finish_reasons=["stop"]
        )

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
    pre = await task.preprocess(
        _sample(), TaskContext(sample_id=0, raw_sample=_sample())
    )
    messages = pre["prompt"]
    assert len(messages) == 1 and messages[0]["role"] == "user"
    # The gold reaches disk from preprocess; raw_sample is never serialized.
    assert pre["reference"] == "William Shakespeare"
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
    await task.infer(
        {"prompt": [{"role": "user", "content": "q"}]}, TaskContext(sample_id=0)
    )
    assert model.last_kwargs.get("n") == 3


# --- feedback: yes/no grading + confidence + provenance ---


@pytest.mark.anyio
async def test_feedback_grades_yes_and_records_provenance():
    reply = "reasoning: matches\ncorrect: yes\nconfidence: 90"
    task, _ = _task(grader_reply=reply)
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    finalize, judgement = await task.feedback(
        build_prediction_record(["Exact Answer: William Shakespeare"]), ctx
    )

    assert finalize is True
    assert judgement["n_rollouts"] == 1
    fb = judgement["rollouts"][0]
    assert fb["extra"]["grade"] == "CORRECT"
    assert fb["correct"] is True
    assert fb["extra"]["confidence"] == 90
    assert judgement["reference"] == "William Shakespeare"
    assert fb["extra"]["grader_output"]["model"]["model"] == "grader-4.1"
    # Multi-line on purpose, with reasoning the parse discards: storing only the
    # matched fields, or only on parse failure, fails this assertion.
    assert fb["extra"]["grader_output"]["texts"] == [reply]
    # #51's limit closes: the whole ModelOutput is kept, so finish_reasons -- what
    # separates a reasoning grader that spent its budget from an empty response --
    # is on disk rather than discarded with the rest of the output.
    assert fb["extra"]["grader_output"]["finish_reasons"] == ["stop"]


@pytest.mark.anyio
async def test_feedback_defaults_to_incorrect_without_verdict():
    # No recognizable "correct: yes|no" -> default INCORRECT (matches upstream).
    reply = "the grader rambled without a verdict"
    task, _ = _task(grader_reply=reply)
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    _, judgement = await task.feedback(build_prediction_record(["some answer"]), ctx)
    assert judgement["rollouts"][0]["extra"]["grade"] == "INCORRECT"
    assert judgement["rollouts"][0]["correct"] is False
    # The INCORRECT default is indistinguishable from a real negative in the
    # grade alone; the reply separates format drift from a wrong answer.
    assert judgement["rollouts"][0]["extra"]["grader_output"]["texts"] == [reply]


# --- report: accuracy over the full requested set ---


@pytest.mark.anyio
async def test_report_accuracy_matches_hand_computation():
    task, _ = _task()
    grades = ["CORRECT", "INCORRECT", "INCORRECT"]
    finals = [
        TaskContext(
            sample_id=i,
            feedback_result=build_judgement_record(
                "",
                [
                    build_rollout_judgement(
                        0, g == "CORRECT", extra={"grade": g, "confidence": 100}
                    )
                ],
            ),
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
            feedback_result=build_judgement_record(
                "",
                [
                    build_rollout_judgement(
                        0, True, extra={"grade": "CORRECT", "confidence": 100}
                    )
                ],
            ),
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
            feedback_result=build_judgement_record(
                "",
                [
                    build_rollout_judgement(
                        i, True, extra={"grade": "CORRECT", "confidence": 100}
                    )
                    for i in range(2)
                ],
            ),
        )
    ]
    fails = [TaskContext(sample_id=1)]
    report = await task.report(finals, fails)
    # 2 correct attempts + n*1 = 2 INCORRECT => 4 units, accuracy 50%.
    assert report["n_graded"] == 2
    assert report["accuracy"] == pytest.approx(50.0)


@pytest.mark.anyio
async def test_report_separates_an_empty_response_from_a_wrong_answer():
    # BrowseComp has no NOT_ATTEMPTED bucket, so a blank response and a wrong
    # answer both score INCORRECT and both land in `n_graded`. Only
    # `n_unextracted` separates "the model returned nothing" from "the model got
    # it wrong", and those want different responses from whoever reads the run.
    task, _ = _task()
    finals = [
        TaskContext(
            sample_id=i,
            postprocess_result=build_prediction_record([prediction]),
            feedback_result=build_judgement_record(
                "",
                [
                    build_rollout_judgement(
                        0, False, extra={"grade": "INCORRECT", "confidence": 100}
                    )
                ],
            ),
        )
        for i, prediction in enumerate(["Christopher Marlowe", None])
    ]
    report = await task.report(finals, fails=[])

    assert report["n_graded"] == 2
    assert report["incorrect"] == pytest.approx(100.0)
    assert report["n_unextracted"] == 1


def test_aggregate_and_parse_kernels():
    assert aggregate_metrics([])["accuracy"] == 0.0
    assert parse_grade("correct: yes") == "CORRECT"
    assert parse_grade("no verdict here") == "INCORRECT"
