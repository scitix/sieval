"""Invalid labels, evidence and typed answers fail before they can be scored.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

import base64
import json

import pytest
from sieval_decisions import (
    ChoiceAnswer,
    ChoiceOption,
    ChoiceProbability,
    ChoiceQuestion,
    ContextInput,
    DecisionParams,
    DecisionRequest,
    DecisionResponse,
    DecisionResult,
    ImageEvidence,
    JsonPart,
    PredicateAnswer,
    PredicateQuestion,
    ScoreAnswer,
    ScoreLegendEntry,
    ScoreProbability,
    ScoreQuestion,
    TextEvidence,
    validate_decision_result,
)
from sieval_decisions.dataset import DecisionJSONLDataset
from sieval_decisions.samples import request_from_sample


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PredicateQuestion("q", "?", true_guidance=1),  # ty: ignore[invalid-argument-type]
        lambda: ScoreQuestion("q", "?", (JsonPart.from_value(None), "high")),
        lambda: ChoiceQuestion("q", "?", (ChoiceOption("a"),)),
        lambda: ScoreQuestion("q", "?", ("low",)),
        lambda: DecisionParams(()),
        lambda: DecisionParams((PredicateQuestion("q", "?"),), require_probabilities=1),  # ty: ignore[invalid-argument-type]
        lambda: ChoiceAnswer(
            "q", "a", (ChoiceProbability("a", 0.5), ChoiceProbability("a", 0.5))
        ),
        lambda: ScoreProbability(True, 0.5),
        lambda: ScoreProbability(-1, 0.5),
        lambda: DecisionResult((PredicateAnswer("q", 0.1), PredicateAnswer("q", 0.2))),
        lambda: ContextInput(()),
        lambda: JsonPart(1),  # ty: ignore[invalid-argument-type]
        lambda: JsonPart.from_value({1: "invalid"}),  # ty: ignore[invalid-argument-type]
        lambda: JsonPart.from_value({"invalid": object()}),
        lambda: DecisionRequest("text", DecisionParams((PredicateQuestion("q", "?"),))),  # ty: ignore[invalid-argument-type]
        lambda: DecisionRequest(ContextInput((TextEvidence("q"),)), "params"),  # ty: ignore[invalid-argument-type]
        lambda: DecisionResponse("answers", "m"),  # ty: ignore[invalid-argument-type]
        lambda: DecisionResponse(DecisionResult(()), "m", usage={}),  # ty: ignore[invalid-argument-type]
        lambda: DecisionResponse(DecisionResult(()), "m", raw={}),  # ty: ignore[invalid-argument-type]
        lambda: ContextInput([TextEvidence("q")]),  # ty: ignore[invalid-argument-type]
    ],
)
def test_invalid_record_construction_is_rejected(factory):
    with pytest.raises((ValueError, TypeError)):
        factory()


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"url": "https://example.com/a", "data": "YWJj", "media_type": "image/png"},
        {"url": "file:///tmp/image"},
        {"url": "https://user:secret@example.com/image"},
        {"data": "YWJj"},
        {"data": "not base64", "media_type": "image/png"},
    ],
)
def test_invalid_image_is_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        ImageEvidence(**kwargs)


@pytest.mark.parametrize(
    "question, answer",
    [
        (PredicateQuestion("q", "?"), ChoiceAnswer("q", "a")),
        (
            ChoiceQuestion("q", "?", (ChoiceOption("a"), ChoiceOption("b"))),
            PredicateAnswer("q", 0.5),
        ),
        (ScoreQuestion("q", "?", ("low", "high")), PredicateAnswer("q", 0.5)),
        (
            ScoreQuestion("q", "?", ("low", "high")),
            ScoreAnswer("q", 0.5, legend=(ScoreLegendEntry(2, "outside"),)),
        ),
        (
            ScoreQuestion("q", "?", ("low", "high")),
            ScoreAnswer("q", 0.5, probabilities=(ScoreProbability(2, 1),)),
        ),
    ],
)
def test_answer_must_match_its_question(question, answer):
    with pytest.raises(ValueError):
        validate_decision_result(DecisionParams((question,)), DecisionResult((answer,)))


@pytest.mark.parametrize(
    "input_value",
    [
        [{"type": "unknown"}],
        [{"type": "json"}],
        [{"type": "text", "text": ""}],
    ],
)
def test_invalid_sample_evidence_is_rejected(input_value):
    with pytest.raises((ValueError, TypeError)):
        request_from_sample(
            {
                "input": input_value,
                "questions": [{"id": "q", "type": "predicate", "instructions": "?"}],
                "reference": {"q": True},
            }
        )


def test_text_and_image_sample_parts_are_supported():
    request = request_from_sample(
        {
            "input": [
                {"type": "text", "text": "photo"},
                {
                    "type": "image",
                    "data": base64.b64encode(b"image").decode(),
                    "media_type": "image/png",
                },
            ],
            "questions": [
                {
                    "id": "q",
                    "type": "predicate",
                    "instructions": "?",
                    "true_guidance": {"meaning": "yes"},
                    "false_guidance": None,
                }
            ],
            "reference": {"q": True},
        }
    )
    assert isinstance(request.context.parts[1], ImageEvidence)
    question = request.decisions.questions[0]
    assert isinstance(question, PredicateQuestion)
    assert isinstance(question.false_guidance, JsonPart)
    assert question.false_guidance.value is None


@pytest.mark.parametrize("kind, reference", [("choice", "c"), ("score", 3)])
def test_out_of_range_labels_fail_before_http(kind, reference):
    question = {
        "id": "q",
        "type": kind,
        "instructions": "?",
        "options": [{"value": "a"}, {"value": "b"}],
        "levels": ["low", "high"],
    }
    with pytest.raises(ValueError):
        request_from_sample(
            {"input": "q", "questions": [question], "reference": {"q": reference}}
        )


@pytest.mark.parametrize(
    "contents",
    [
        "",
        "\n",
        "{invalid json}",
        json.dumps({"input": "q", "questions": [], "reference": {}}),
    ],
)
def test_invalid_jsonl_reports_input_error(tmp_path, contents):
    path = tmp_path / "samples.jsonl"
    path.write_text(contents)
    with pytest.raises(ValueError):
        DecisionJSONLDataset(str(path))
