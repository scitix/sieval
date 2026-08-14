"""Unit tests for the HLE 0-shot generative task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from unittest.mock import patch

import numpy as np
import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

import sieval.datasets.hle as hle_module
from sieval.community.hle import (
    JUDGE_PROMPT,
    SYSTEM_PROMPT,
    aggregate_metrics,
    calib_err,
    parse_judge,
)
from sieval.core.models import Request, Response
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import (
    TaskContext,
    build_judgement_record,
    build_prediction_record,
    build_rollout_judgement,
)
from sieval.datasets.hle import HLEDataset
from sieval.tasks.hle_0shot_gen import HLEZeroShotGenTask
from tests.conftest import HandlerTransport


class _ScriptedChatModel(ChatModel):
    """ChatModel returning a fixed reply, recording the last agenerate kwargs."""

    def __init__(self, reply: str, model: str = "mock"):
        self._reply = reply
        self.last_kwargs: dict[str, object] = {}
        super().__init__(model=model, api_key="fake")

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(self._stub_arun, "openai_chat")

    async def _stub_arun(self, req: Request) -> Response:
        self.last_kwargs = {"n": req.sampling.n}
        # `finish_reasons` is set because the judge-family migration exists to
        # persist the grader's WHOLE ModelOutput: it is the field that separates a
        # reasoning judge which spent its budget thinking from an empty API
        # response, and #51's flat `grader_reply` never carried it.
        return Response(
            texts=(self._reply,) * req.sampling.n,
            finish_reasons=("stop",) * req.sampling.n,
        )


def _row(image: str = "") -> dict:
    # Mirrors the pinned revision's schema; `HLEDataset.load` asserts the
    # auxiliary Image columns are present before disabling their decoding.
    return {
        "id": "q1",
        "question": "What is 2 + 2?",
        "image": image,
        "image_preview": None,
        "answer": "4",
        "answer_type": "exactMatch",
        "author_name": "author",
        "rationale": "",
        "rationale_image": None,
        "raw_subject": "Math",
        "category": "Math",
    }


def _dataset(rows: list[dict], *, text_only: bool = True) -> HLEDataset:
    """Build through the real ``load`` path so ``text_only`` is recorded on it."""
    hf = HFDatasetDict({"test": HFDataset.from_list(rows)})
    with patch.object(hle_module, "load_dataset", return_value=hf):
        return HLEDataset("cais/hle", text_only=text_only)


def _task(
    grader_reply: str = "correct: yes\nconfidence: 90",
    *,
    rows: list[dict] | None = None,
    text_only: bool = True,
    n: int = 1,
):
    dataset = _dataset(rows if rows is not None else [_row()], text_only=text_only)
    model = _ScriptedChatModel(reply="Answer: 4", model="candidate")
    grader = _ScriptedChatModel(reply=grader_reply, model="judge-5.2")
    task = HLEZeroShotGenTask(dataset, model, grader=grader, n=n)
    return task, model, grader


# --- grader is mandatory; no deterministic fallback ---


def test_build_grader_requires_config():
    with pytest.raises(ValueError, match="requires an LLM judge"):
        HLEZeroShotGenTask._build_grader(None)


def test_build_grader_accepts_mapping_and_model():
    built = HLEZeroShotGenTask._build_grader({"model": "gpt-5.2", "api_key": "fake"})
    assert isinstance(built, ChatModel)
    existing = _ScriptedChatModel(reply="x")
    assert HLEZeroShotGenTask._build_grader(existing) is existing


def test_constructor_accepts_composed_grader_model():
    base, _, grader = _task()
    task = HLEZeroShotGenTask(
        base.dataset,
        base.model,
        models_by_role={"grader": grader},
    )
    assert task._grader is grader


def test_constructor_rejects_missing_composed_grader_role():
    base, _, _ = _task()
    with pytest.raises(ValueError, match="missing the 'grader'"):
        HLEZeroShotGenTask(base.dataset, base.model, models_by_role={})


def test_constructor_rejects_ambiguous_grader_sources():
    base, _, grader = _task()
    with pytest.raises(ValueError, match="cannot both be supplied"):
        HLEZeroShotGenTask(
            base.dataset,
            base.model,
            grader=grader,
            models_by_role={"grader": grader},
        )


# --- subset is the dataset's decision; the task only grades what it is handed ---


def test_task_does_not_refilter_a_text_only_dataset():
    # The task has no `text_only` arg: the image question was already dropped at
    # load time, and the task must neither re-filter nor restore it.
    task, _, _ = _task(rows=[_row(), _row(image="data:image/png;base64,AAAA")])
    assert task.dataset.test_set is not None
    assert len(task.dataset.test_set) == 1
    assert task.dataset.test_set[0]["image"] == ""


def test_task_keeps_image_questions_from_a_full_set_dataset():
    task, _, _ = _task(
        rows=[_row(), _row(image="data:image/png;base64,AAAA")], text_only=False
    )
    assert task.dataset.test_set is not None
    assert len(task.dataset.test_set) == 2


# --- preprocess: HLE system prompt + user content blocks (mirrors format_message) ---


@pytest.mark.anyio
async def test_preprocess_text_only_message():
    task, _, _ = _task()
    pre = await task.preprocess(_row(), TaskContext(sample_id=0))
    assert pre["prompt"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": "What is 2 + 2?"}]},
    ]
    # The gold reaches disk from preprocess; raw_sample is never serialized.
    assert pre["reference"] == "4"
    assert pre["extra"]["has_image"] is False


@pytest.mark.anyio
async def test_preprocess_attaches_image_block():
    task, _, _ = _task(text_only=False)
    raw = _row(image="data:image/png;base64,AAAA")
    pre = await task.preprocess(raw, TaskContext(sample_id=0))
    assert pre["extra"]["has_image"] is True
    user_content = pre["prompt"][1]["content"]
    assert user_content[0] == {"type": "text", "text": "What is 2 + 2?"}
    assert user_content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAAA"},
    }


# --- infer forwards ONLY n (no decode-param injection) ---


@pytest.mark.anyio
async def test_infer_forwards_only_n():
    task, model, _ = _task(n=1)
    await task.infer(
        {"prompt": [{"role": "user", "content": "q"}]}, TaskContext(sample_id=0)
    )
    # Decode params (temperature/top_p/max_tokens) must come from the model
    # layer, never from the task — infer passes n and nothing else.
    assert model.last_kwargs == {"n": 1}


# --- feedback: parse judge correct/confidence, record provenance ---


@pytest.mark.anyio
async def test_feedback_parses_correct_and_confidence():
    reply = "reasoning: matches\ncorrect: yes\nconfidence: 90"
    task, _, _ = _task(grader_reply=reply)
    ctx = TaskContext(sample_id=0, raw_sample=_row())
    finalize, judgement = await task.feedback(
        build_prediction_record(["Answer: 4"]), ctx
    )

    assert finalize is True
    fb = judgement["rollouts"][0]
    assert fb["correct"] is True
    assert fb["extra"]["confidence"] == 90
    assert fb["extra"]["grader_parsed"] is True
    assert judgement["reference"] == "4"
    # The judge's WHOLE ModelOutput, not a hand-picked reply field. Multi-line on
    # purpose, with reasoning the parse discards: storing only the matched fields,
    # or only on parse failure, fails this assertion.
    grader_output = fb["extra"]["grader_output"]
    assert grader_output["texts"] == [reply]
    assert grader_output["model"]["model"] == "judge-5.2"
    # #51's documented limit closes here: finish_reasons is what separates a
    # reasoning judge that spent its budget thinking from an empty API response,
    # and the flat grader_reply never carried it.
    assert grader_output["finish_reasons"] == ["stop"]


@pytest.mark.anyio
async def test_feedback_unparseable_reply_flagged_not_graded():
    reply = "the judge rambled without the fields"
    task, _, _ = _task(grader_reply=reply)
    ctx = TaskContext(sample_id=0, raw_sample=_row())
    _, judgement = await task.feedback(build_prediction_record(["whatever"]), ctx)
    fb = judgement["rollouts"][0]
    # Unparseable -> flagged so report() drops it from grading (not a verdict).
    assert fb["extra"]["grader_parsed"] is False
    assert fb["correct"] is False
    assert fb["extra"]["confidence"] == 100
    # The motivating case: `n_grader_unparsed` alone cannot separate format drift
    # from an error body from a matcher gap. The reply is the evidence.
    assert fb["extra"]["grader_output"]["texts"] == [reply]


# --- report: accuracy over the full requested set (fails in denominator) ---


def _finals(grades: list[tuple[bool, int]]) -> list[TaskContext]:
    return [
        TaskContext(
            sample_id=i,
            feedback_result=build_judgement_record(
                "",
                [
                    build_rollout_judgement(
                        0, c, extra={"confidence": conf, "grader_parsed": True}
                    )
                ],
            ),
        )
        for i, (c, conf) in enumerate(grades)
    ]


@pytest.mark.anyio
async def test_report_accuracy_and_counts_fails_in_denominator():
    task, _, _ = _task()  # n=1
    finals = _finals([(True, 90), (False, 40)])
    fails = [TaskContext(sample_id=10)]
    report = await task.report(finals, fails)

    # n = (2 finals + 1 fail) * 1 = 3; 1 correct => 33.33%.
    # The old len(finals)=2 denominator would give 50.0, so this discriminates.
    assert report["n"] == 3
    assert report["n_graded"] == 2
    assert report["fails"] == 1
    assert report["n_grader_unparsed"] == 0
    assert report["subset"] == "text_only"  # dataset loaded the text-only subset
    assert report["accuracy"] == pytest.approx(33.33, abs=1e-2)
    assert report["score"] == report["accuracy"]


@pytest.mark.anyio
async def test_report_fails_weighted_by_n():
    task, _, _ = _task(n=2)
    # One finalized sample carrying its n=2 judged attempts (both correct).
    finals = [
        TaskContext(
            sample_id=0,
            feedback_result=build_judgement_record(
                "",
                [
                    build_rollout_judgement(
                        i, True, extra={"confidence": 90, "grader_parsed": True}
                    )
                    for i in range(2)
                ],
            ),
        )
    ]
    fails = [TaskContext(sample_id=5)]
    report = await task.report(finals, fails)
    # n = (1 final + 1 fail) * 2 = 4; 2 correct => 50.0%.
    # An unweighted (n=1) denominator would give 3 and 66.67%.
    assert report["n"] == 4
    assert report["n_graded"] == 2
    assert report["accuracy"] == pytest.approx(50.0)


@pytest.mark.anyio
async def test_report_drops_unparsed_judge_from_grading():
    # Unparseable judge replies stay in `n` (counted incorrect) but must not
    # enter the grading/calibration arrays or they would inflate metrics.
    task, _, _ = _task()  # n=1

    def judgement(correct, confidence, grader_parsed):
        return build_judgement_record(
            "",
            [
                build_rollout_judgement(
                    0,
                    correct,
                    extra={"confidence": confidence, "grader_parsed": grader_parsed},
                )
            ],
        )

    finals = [
        TaskContext(sample_id=0, feedback_result=judgement(True, 90, True)),
        TaskContext(sample_id=1, feedback_result=judgement(False, 100, False)),
    ]
    report = await task.report(finals, [])
    assert report["n_grader_unparsed"] == 1
    assert report["n_graded"] == 1  # only the parsed reply is graded
    assert report["n"] == 2  # both stay in the denominator
    # 1 correct / 2 => 50.0; the dropped record counts as incorrect via `n`.
    assert report["accuracy"] == pytest.approx(50.0)


@pytest.mark.anyio
async def test_report_empty_is_zero():
    task, _, _ = _task()
    report = await task.report([], [])
    assert report["n"] == 0
    assert report["accuracy"] == 0.0
    assert report["calibration_error"] is None
    assert report["n_grader_unparsed"] == 0


@pytest.mark.anyio
async def test_report_separates_an_unparsed_judge_from_an_empty_response():
    # Two counts for two different actors: `n_grader_unparsed` is the GRADER failing
    # to answer, `n_unextracted` is the candidate producing nothing to grade.
    # Both end up incorrect, and with only the first the report cannot say which
    # side of the pipeline went quiet.
    task, _, _ = _task()  # n=1
    finals = [
        TaskContext(
            sample_id=0,
            postprocess_result=build_prediction_record(["Answer: 5"]),
            feedback_result=build_judgement_record(
                "",
                [
                    build_rollout_judgement(
                        0, False, extra={"confidence": 90, "grader_parsed": True}
                    )
                ],
            ),
        ),
        TaskContext(
            sample_id=1,
            postprocess_result=build_prediction_record([None]),
            feedback_result=build_judgement_record(
                "",
                [
                    build_rollout_judgement(
                        0, False, extra={"confidence": 100, "grader_parsed": True}
                    )
                ],
            ),
        ),
    ]
    report = await task.report(finals, [])

    assert report["n_grader_unparsed"] == 0
    assert report["n_unextracted"] == 1
    assert report["accuracy"] == pytest.approx(0.0)


@pytest.mark.anyio
async def test_report_subset_reflects_full_set():
    # `subset` must distinguish a full-set run from the text-only default.
    task, _, _ = _task(text_only=False)
    report = await task.report([], [])
    assert report["subset"] == "full"


# --- prompt fidelity: byte-for-byte pins on the vendored HLE prompts ---
# These lock the reproduction invariant so any drift from upstream
# (centerforaisafety/hle @ 26dca2e) fails loudly. `test_preprocess_*` above
# compare against the constants by reference and cannot catch such drift.


def test_system_prompt_pinned():
    assert SYSTEM_PROMPT == (
        "Your response should be in the following format:\n"
        "Explanation: {your explanation for your answer choice}\n"
        "Answer: {your chosen answer}\n"
        "Confidence: {your confidence score between 0% and 100% for your answer}"
    )


def test_judge_prompt_pinned():
    # Upstream ships a duplicated-word typo and pipe-escaped percent signs;
    # both are preserved verbatim.
    assert "i.e. if there if there is any inconsistency" in JUDGE_PROMPT
    assert r"confidence score between 0|\%| and 100|\%| from [response]" in JUDGE_PROMPT
    assert "extracted_final_answer:" in JUDGE_PROMPT
    for field in ("{question}", "{response}", "{correct_answer}"):
        assert field in JUDGE_PROMPT


# --- metric kernel: parse_judge, calib_err, aggregate_metrics ---


def test_parse_judge_last_field_wins():
    # Returns (correct, confidence, parsed); parsed is True when `correct` matched.
    assert parse_judge("correct: yes\nconfidence: 85") == (True, 85, True)
    assert parse_judge("correct: no") == (False, 100, True)
    # reasoning may mention "correct:"; the trailing field value wins.
    assert parse_judge("reasoning: correct: no\ncorrect: yes\nconfidence: 30") == (
        True,
        30,
        True,
    )
    # `\b` anchor: "incorrect: yes" must NOT be read as the `correct` field.
    # Without the anchor the substring "correct: yes" would match -> parsed True;
    # with no real verdict field the parse is not graded (parsed False).
    assert parse_judge("extracted_final_answer: 42 is incorrect: yes") == (
        False,
        100,
        False,
    )


def test_parse_judge_unparseable_flags_not_parsed():
    # No `correct` field -> parsed False so report() drops it from grading.
    assert parse_judge("the judge rambled without the fields") == (False, 100, False)


def test_parse_judge_tolerates_markdown_bold_and_quotes():
    # Markdown-bold and JSON-shaped (quoted) replies must still parse.
    assert parse_judge("**correct:** yes\n**confidence:** 70") == (True, 70, True)
    assert parse_judge("correct: **no**") == (False, 100, True)
    assert parse_judge('{"correct": "yes", "confidence": "90"}') == (True, 90, True)


def test_calib_err_matches_hand_computation():
    # beta=2 forces two bins over four samples so the first bin is scored
    # (upstream excludes the final bin via range(len(bins) - 1)).
    confidence = np.array([0.1, 0.2, 0.9, 0.95])
    correct = np.array([0, 0, 1, 1])
    # bin[0] conf mean 0.15, correct mean 0 -> diff 0.15;
    # cerr = sqrt(2/4 * 0.15**2) = 0.106066...
    assert calib_err(confidence, correct, p="2", beta=2) == pytest.approx(
        0.106066, abs=1e-5
    )


def test_aggregate_metrics_accuracy_ci_and_calibration_guard():
    # 1 correct of n=4 -> 25.0%; Wald half-width = 1.96*sqrt(25*75/4) = 42.44.
    m = aggregate_metrics([True, False], [100, 50], n=4)
    assert m["accuracy"] == pytest.approx(25.0)
    assert m["confidence_interval"] == pytest.approx(42.44, abs=1e-2)
    # Fewer than BETA graded records -> not computable, None (not a real 0.0).
    assert m["calibration_error"] is None


def test_aggregate_metrics_zero_n():
    assert aggregate_metrics([], [], n=0) == {
        "accuracy": 0.0,
        "confidence_interval": 0.0,
        "calibration_error": None,
    }
