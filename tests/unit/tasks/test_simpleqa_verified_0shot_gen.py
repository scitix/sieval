"""Unit tests for the SimpleQA Verified 0-shot generative task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.community.simpleqa_verified import aggregate_metrics, parse_grade
from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import (
    RolloutJudgement,
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
from sieval.datasets.simpleqa_verified import (
    SimpleQAVerifiedDataset,
    SimpleQAVerifiedDatasetSample,
)
from sieval.tasks.simpleqa_verified_0shot_gen import SimpleQAVerifiedZeroShotGenTask


def _graded(*grades: str):
    """A JudgementRecord holding one rollout per grade, as feedback() would build.

    report() reads only ``extra["grade"]``, so the grader_output the real path
    also writes is omitted here — these fixtures exercise aggregation, not the
    record shape (which its own tests above cover).
    """
    return build_judgement_record(
        "",
        [
            build_rollout_judgement(index, grade == "CORRECT", extra={"grade": grade})
            for index, grade in enumerate(grades)
        ],
    )


class _ScriptedChatModel(ChatModel):
    """ChatModel returning a fixed reply, recording the last agenerate kwargs."""

    def __init__(
        self,
        reply: str,
        model: str = "mock",
        finish_reason: str | None = None,
        reasoning: str | None = None,
    ):
        super().__init__(model=model, api_key="fake")
        self._reply = reply
        self._finish_reason = finish_reason
        self._reasoning = reasoning
        self.last_kwargs: dict[str, object] = {}

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = prompt
        self.last_kwargs = dict(kwargs)
        return ModelOutput(
            model=self.meta(),
            texts=[self._reply],
            finish_reasons=[self._finish_reason] if self._finish_reason else None,
            reasoning_texts=[self._reasoning] if self._reasoning else None,
            usage={"input_tokens": 40, "output_tokens": 3, "total_tokens": 43},
        )

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


def _task(
    answer_reply: str = "William Shakespeare",
    grader_reply: str = "A",
    grader_finish_reason: str | None = None,
    grader_reasoning: str | None = None,
):
    dataset = SimpleQAVerifiedDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    model = _ScriptedChatModel(reply=answer_reply, model="candidate")
    grader = _ScriptedChatModel(
        reply=grader_reply,
        model="grader-4.1",
        finish_reason=grader_finish_reason,
        reasoning=grader_reasoning,
    )
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
    record = await task.preprocess(
        _sample(), TaskContext(sample_id=0, raw_sample=_sample())
    )
    assert record["prompt"] == [{"role": "user", "content": "Who wrote Hamlet?"}]
    # The gold lands in the prompt record too, so the sample's ground truth is on
    # disk from the first stage onward.
    assert record["reference"] == "William Shakespeare"


# --- infer forwards n to the candidate model ---


@pytest.mark.anyio
async def test_infer_forwards_n():
    dataset = SimpleQAVerifiedDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    model = _ScriptedChatModel(reply="x", model="candidate")
    grader = _ScriptedChatModel(reply="A", model="grader")
    task = SimpleQAVerifiedZeroShotGenTask(dataset, model, grader=grader, n=3)
    await task.infer(
        {"prompt": [{"role": "user", "content": "q"}]}, TaskContext(sample_id=0)
    )
    assert model.last_kwargs.get("n") == 3


# --- feedback: grades each answer via the grader, records provenance ---


@pytest.mark.anyio
async def test_feedback_grades_and_records_provenance():
    # Multi-line on purpose, with prose the A/B/C parse discards: storing only
    # the matched letter, or only on parse failure, fails this assertion.
    reply = "The predicted answer matches the gold target.\nA"
    task, _ = _task(grader_reply=reply)
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    finalize, judgement = await task.feedback(
        build_prediction_record(["William Shakespeare"]), ctx
    )

    assert finalize is True
    assert judgement["reference"] == "William Shakespeare"
    assert judgement["n_rollouts"] == 1
    assert judgement["n_correct"] == 1

    rollout: RolloutJudgement = judgement["rollouts"][0]
    assert rollout["correct"] is True
    assert rollout["extra"]["grade"] == "CORRECT"
    # The grader's full output is persisted, not hand-picked fields: the reply is
    # there verbatim (storing only the matched letter fails this), and so is the
    # grader's cost and model id, so a run's grading spend is on disk.
    grader_output = rollout["extra"]["grader_output"]
    assert grader_output["texts"] == [reply]
    assert grader_output["model"]["model"] == "grader-4.1"
    assert grader_output["usage"]["total_tokens"] == 43


@pytest.mark.anyio
async def test_feedback_empty_grader_reply_is_not_attempted():
    task, _ = _task(grader_reply="")
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    _, judgement = await task.feedback(build_prediction_record(["some answer"]), ctx)
    rollout = judgement["rollouts"][0]
    assert rollout["extra"]["grade"] == "NOT_ATTEMPTED"
    assert rollout["correct"] is False
    # An empty reply is exactly the case the raw text disambiguates.
    assert rollout["extra"]["grader_output"]["texts"] == [""]


@pytest.mark.anyio
async def test_feedback_persists_reply_behind_not_attempted_default():
    # `parse_grade` defaults a non-matching reply to NOT_ATTEMPTED, which the F1
    # weights very differently from INCORRECT. The grade alone cannot say whether
    # the candidate abstained or the grader drifted off format; the reply can.
    reply = "I cannot classify this response into the given options."
    task, _ = _task(grader_reply=reply)
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    _, judgement = await task.feedback(build_prediction_record(["some answer"]), ctx)
    rollout = judgement["rollouts"][0]
    assert rollout["extra"]["grade"] == "NOT_ATTEMPTED"
    assert rollout["extra"]["grader_output"]["texts"] == [reply]


@pytest.mark.anyio
async def test_finish_reason_separates_a_thinking_grader_from_an_empty_response():
    # The reply is response content only, so a reasoning autorater that spends its
    # whole budget thinking records an empty one — identical on disk to an empty
    # API response. The grader's `finish_reasons`, captured in `grader_output`, is
    # the only thing that tells them apart. Without it both cases look like a real
    # abstention, which the F1 weights very differently from an incorrect answer.
    truncated, _ = _task(grader_reply="", grader_finish_reason="length")
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    _, judgement = await truncated.feedback(
        build_prediction_record(["some answer"]), ctx
    )
    thinking = judgement["rollouts"][0]["extra"]

    empty, _ = _task(grader_reply="", grader_finish_reason="stop")
    _, judgement = await empty.feedback(build_prediction_record(["some answer"]), ctx)
    no_content = judgement["rollouts"][0]["extra"]

    # Indistinguishable on the reply and the grade alone...
    assert thinking["grader_output"]["texts"] == no_content["grader_output"]["texts"]
    assert thinking["grade"] == no_content["grade"] == "NOT_ATTEMPTED"
    # ...and separable only via the recorded finish reason.
    assert thinking["grader_output"]["finish_reasons"] == ["length"]
    assert no_content["grader_output"]["finish_reasons"] == ["stop"]


@pytest.mark.anyio
async def test_grader_reasoning_is_persisted_not_dropped():
    # The whole point of storing the grader's full ModelOutput rather than a
    # hand-picked reply: a reasoning autorater's reasoning survives to disk. A
    # scheme that stored only texts + a call-meta projection would lose it.
    task, _ = _task(grader_reply="A", grader_reasoning="It matches the gold target.")
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    _, judgement = await task.feedback(
        build_prediction_record(["William Shakespeare"]), ctx
    )
    grader_output = judgement["rollouts"][0]["extra"]["grader_output"]
    assert grader_output["reasoning_texts"] == ["It matches the gold target."]


@pytest.mark.anyio
async def test_feedback_grades_a_blank_answer_without_crashing():
    # postprocess maps a blank answer to None so `extracted` stays meaningful; the
    # grader must still see a string and return a verdict for it.
    task, _ = _task(grader_reply="C")
    ctx = TaskContext(sample_id=0, raw_sample=_sample())
    post = build_prediction_record([None])
    assert post["rollouts"][0]["extracted"] is False

    _, judgement = await task.feedback(post, ctx)
    assert judgement["rollouts"][0]["extra"]["grade"] == "NOT_ATTEMPTED"


# --- report: F1 aggregation matches simple-evals ---


@pytest.mark.anyio
async def test_report_f1_matches_hand_computation():
    task, _ = _task()
    grades = ["CORRECT", "CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
    finals = [
        TaskContext(sample_id=i, feedback_result=_graded(g))
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


@pytest.mark.anyio
async def test_report_counts_fails_as_not_attempted():
    # Failed samples must dilute the F1 (full-set metric), not be excluded.
    task, _ = _task()  # n=1
    finals = [
        TaskContext(sample_id=i, feedback_result=_graded(g))
        for i, g in enumerate(["CORRECT", "INCORRECT"])
    ]
    fails = [TaskContext(sample_id=10), TaskContext(sample_id=11)]
    report = await task.report(finals, fails)

    # 2 graded (1 correct, 1 incorrect) + 2 fails-as-NOT_ATTEMPTED => 4 units.
    # correct=0.25, incorrect=0.25 -> acc_given_attempted=0.5 -> f1=0.3333.
    # Excluding fails would give correct=0.5, f1=50.0, so this discriminates.
    assert report["n_graded"] == 2
    assert report["fails"] == 2
    assert report["correct"] == pytest.approx(25.0)
    assert report["not_attempted"] == pytest.approx(50.0)
    assert report["f1"] == pytest.approx(33.3333, abs=1e-3)


@pytest.mark.anyio
async def test_report_fails_weighted_by_n():
    # Each failed sample stands in for its n requested attempts.
    dataset = SimpleQAVerifiedDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    model = _ScriptedChatModel(reply="x", model="candidate")
    grader = _ScriptedChatModel(reply="A", model="grader")
    task = SimpleQAVerifiedZeroShotGenTask(dataset, model, grader=grader, n=2)
    finals = [TaskContext(sample_id=0, feedback_result=_graded("CORRECT", "CORRECT"))]
    fails = [TaskContext(sample_id=1)]
    report = await task.report(finals, fails)

    # 2 correct attempts + n*1 = 2 NOT_ATTEMPTED => 4 units, correct rate 0.5.
    # A per-sample (unweighted) count would give 3 units and correct=66.7.
    assert report["n_graded"] == 2
    assert report["correct"] == pytest.approx(50.0)
    assert report["not_attempted"] == pytest.approx(50.0)


@pytest.mark.anyio
async def test_report_separates_an_empty_response_from_a_hedged_one():
    # NOT_ATTEMPTED is the autorater's reading of an answer, and a blank response
    # -- which it never saw a claim in -- grades the same way. The rate cannot
    # say which happened; `n_unextracted` can, and "the model returned nothing"
    # is a delivery-side fact, not a factuality result.
    task, _ = _task()
    finals = [
        TaskContext(
            sample_id=i,
            postprocess_result=build_prediction_record([prediction]),
            feedback_result=_graded("NOT_ATTEMPTED"),
        )
        for i, prediction in enumerate(["I am not sure.", None])
    ]
    report = await task.report(finals, fails=[])

    assert report["n_graded"] == 2
    assert report["not_attempted"] == pytest.approx(100.0)
    assert report["n_unextracted"] == 1


def test_report_empty_is_zero():
    # aggregate_metrics is the pure kernel report() delegates to.
    m = aggregate_metrics([])
    assert m["f1"] == 0.0
    assert parse_grade("C") == "NOT_ATTEMPTED"
