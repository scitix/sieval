"""Unit tests for the AA-LCR 0-shot generative task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.community.aa_lcr import aggregate_metrics, parse_grade
from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
from sieval.datasets.aa_lcr import AALCRDataset, AALCRDatasetSample
from sieval.tasks.aa_lcr_0shot_gen import (
    AALCRZeroShotGenTask,
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
    pre = await task.preprocess(
        _sample(), TaskContext(sample_id=0, raw_sample=_sample())
    )
    messages = pre["prompt"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    # The gold + question id reach disk from preprocess; raw_sample is never
    # serialized, so before the migration neither was recoverable from a row.
    assert pre["reference"] == "Rising"
    assert pre["extra"]["question_id"] == 7
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
    await task.infer(
        {"prompt": [{"role": "user", "content": "q"}]}, TaskContext(sample_id=0)
    )
    assert model.last_kwargs.get("n") == 3


# --- feedback: grades each answer via the grader, records provenance ---


@pytest.mark.anyio
async def test_feedback_grades_and_records_provenance():
    reply = "The candidate matches the official answer.\nCORRECT"
    task, _ = _task(grader_reply=reply)
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    finalize, judgement = await task.feedback(build_prediction_record(["Rising"]), ctx)

    assert finalize is True
    assert judgement["n_rollouts"] == 1
    fb = judgement["rollouts"][0]
    assert fb["extra"]["grade"] == "CORRECT"
    assert fb["correct"] is True
    assert judgement["reference"] == "Rising"
    assert judgement["extra"]["question_id"] == 7
    assert fb["extra"]["grader_output"]["model"]["model"] == "qwen3-235b"
    # Multi-line on purpose, with reasoning the parse discards: storing only the
    # matched verdict, or only on parse failure, fails this assertion.
    assert fb["extra"]["grader_output"]["texts"] == [reply]
    # #51's limit closes: the whole ModelOutput is kept, so finish_reasons is on
    # disk instead of being discarded with the rest of the grader's output.
    assert fb["extra"]["grader_output"]["finish_reasons"] == ["stop"]


@pytest.mark.anyio
async def test_feedback_empty_grader_reply_is_incorrect():
    task, _ = _task(grader_reply="")
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    _, judgement = await task.feedback(build_prediction_record(["some answer"]), ctx)
    fb = judgement["rollouts"][0]
    assert fb["extra"]["grade"] == "INCORRECT"
    # The checker WAS called and returned nothing; the grader output is present
    # and empty, which is what distinguishes this from the short-circuit below.
    assert fb["extra"]["grader_output"]["texts"] == [""]


@pytest.mark.anyio
async def test_feedback_persists_reply_behind_incorrect_default():
    # `parse_grade` sends anything unreadable to INCORRECT, so the grade alone
    # cannot tell format drift from a genuinely wrong answer. The reply can.
    reply = "I am unable to compare these two answers."
    task, _ = _task(grader_reply=reply)
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    _, judgement = await task.feedback(build_prediction_record(["some answer"]), ctx)
    fb = judgement["rollouts"][0]
    assert fb["extra"]["grade"] == "INCORRECT"
    assert fb["extra"]["grader_output"]["texts"] == [reply]


@pytest.mark.anyio
@pytest.mark.parametrize("empty", ["", "   ", "\n\n"])
async def test_feedback_empty_answer_is_incorrect_without_grading(empty: str):
    # A truncated/empty answer must be INCORRECT and must NOT reach the grader —
    # even a grader that would say CORRECT cannot flip an empty answer (this is
    # the fix for empty truncated outputs being spuriously graded CORRECT).
    task, grader = _task(grader_reply="CORRECT")  # grader would say CORRECT
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    post = build_prediction_record([empty if empty.strip() else None])
    _, judgement = await task.feedback(post, ctx)
    fb = judgement["rollouts"][0]
    assert fb["extra"]["grade"] == "INCORRECT"
    assert fb["correct"] is False
    # The prediction rollout records the miss independently of the verdict.
    assert post["rollouts"][0]["extracted"] is False
    # Grader was bypassed: its last_kwargs stays empty (never called).
    assert grader.last_kwargs == {}
    # No call, so grader_output is ABSENT -- not an empty string. That is the
    # improvement over #51's flat grader_reply, which used "" for both "never
    # called" and "the checker returned nothing" (asserted distinct above).
    assert "grader_output" not in fb["extra"]


@pytest.mark.anyio
async def test_feedback_short_circuit_does_not_inherit_prior_reply():
    # With n>1 the attempts share one loop, so the short-circuit must rebind
    # `reply`; leaving the graded attempt's in scope would attribute a real
    # checker verdict to an ungraded answer.
    reply = "The candidate matches the official answer.\nCORRECT"
    task, _ = _task(grader_reply=reply)
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    _, judgement = await task.feedback(build_prediction_record(["Rising", None]), ctx)
    rollouts = judgement["rollouts"]

    assert [r["extra"]["grade"] for r in rollouts] == ["CORRECT", "INCORRECT"]
    assert rollouts[0]["extra"]["grader_output"]["texts"] == [reply]
    # The ungraded attempt carries no grader output at all, so a real verdict
    # cannot be attributed to it by a stale binding.
    assert "grader_output" not in rollouts[1]["extra"]


# --- report: accuracy over graded + failed samples ---


@pytest.mark.anyio
async def test_report_accuracy_matches_hand_computation():
    task, _ = _task()
    grades = ["CORRECT", "CORRECT", "CORRECT", "INCORRECT"]
    finals = [
        TaskContext(
            sample_id=i,
            feedback_result=build_judgement_record(
                "",
                [build_rollout_judgement(0, g == "CORRECT", extra={"grade": g})],
                extra={"question_id": i},
            ),
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
            feedback_result=build_judgement_record(
                "",
                [build_rollout_judgement(0, g == "CORRECT", extra={"grade": g})],
                extra={"question_id": i},
            ),
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
            feedback_result=build_judgement_record(
                "",
                [
                    build_rollout_judgement(i, True, extra={"grade": "CORRECT"})
                    for i in range(2)
                ],
                extra={"question_id": 0},
            ),
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
