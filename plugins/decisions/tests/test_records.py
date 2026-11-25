"""Typed decision validation and persistence preserve native values.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

import json
from types import MappingProxyType

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
    JsonPart,
    PredicateAnswer,
    PredicateQuestion,
    RefusalAnswer,
    ScoreAnswer,
    ScoreProbability,
    ScoreQuestion,
    TextEvidence,
    validate_decision_result,
)
from sieval_decisions.dataset import DecisionJSONLDataset
from sieval_decisions.samples import request_from_sample

from sieval.core.utils.serialization import dict_to_obj, obj_to_dict


def test_json_evidence_and_guidance_preserve_null_and_detach_input():
    original = {"filter": None, "nested": [{"a": 1}]}
    part = JsonPart.from_value(MappingProxyType(original))
    original["nested"][0]["a"] = 2
    decoded = dict_to_obj(obj_to_dict(part, True), {})
    assert isinstance(decoded, JsonPart)
    assert decoded.value == {"filter": None, "nested": [{"a": 1}]}
    value = decoded.value
    value["nested"][0]["a"] = 3
    assert decoded.value["nested"][0]["a"] == 1
    assert JsonPart.from_value(None).value is None
    with pytest.raises(ValueError, match="null"):
        ContextInput((JsonPart.from_value(None),))


@pytest.mark.parametrize(
    "encoded", ['{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}', "1"]
)
def test_invalid_json_is_not_silently_normalized(encoded):
    with pytest.raises((ValueError, TypeError)):
        JsonPart(encoded)


@pytest.mark.parametrize("value", [True, -0.1, 1.1, float("nan"), float("inf")])
def test_predicate_probabilities_require_finite_numbers_in_range(value):
    with pytest.raises((ValueError, TypeError)):
        PredicateAnswer("q", value)


def test_question_ids_options_and_order_are_explicit():
    with pytest.raises(ValueError, match="unique"):
        DecisionParams((PredicateQuestion("q", "yes?"), PredicateQuestion("q", "no?")))
    with pytest.raises(ValueError, match="unique"):
        ChoiceQuestion("q", "which?", (ChoiceOption("a"), ChoiceOption("a")))
    params = DecisionParams(
        (PredicateQuestion("a", "yes?"), PredicateQuestion("b", "yes?"))
    )
    with pytest.raises(ValueError, match="identity and order"):
        validate_decision_result(
            params,
            DecisionResult((PredicateAnswer("b", 0.4), PredicateAnswer("a", 0.6))),
        )


def test_required_probabilities_and_confidence_apply_to_successes_only():
    question = ChoiceQuestion("q", "which?", (ChoiceOption("a"), ChoiceOption("b")))
    params = DecisionParams(
        (question,), require_probabilities=True, require_confidence=True
    )
    validate_decision_result(params, DecisionResult((RefusalAnswer("q"),)))
    with pytest.raises(ValueError, match="confidence"):
        validate_decision_result(params, DecisionResult((ChoiceAnswer("q", "a"),)))
    with pytest.raises(ValueError, match="probabilities"):
        validate_decision_result(
            params, DecisionResult((ChoiceAnswer("q", "a", confidence=0.6),))
        )
    with pytest.raises(ValueError, match="complete"):
        validate_decision_result(
            params,
            DecisionResult(
                (ChoiceAnswer("q", "a", (ChoiceProbability("a", 1),), 0.6),)
            ),
        )


def test_score_bounds_and_duplicate_levels_are_validated():
    params = DecisionParams((ScoreQuestion("q", "rate?", ("low", "high")),))
    with pytest.raises(ValueError, match="outside"):
        validate_decision_result(params, DecisionResult((ScoreAnswer("q", 1.1),)))
    with pytest.raises(ValueError, match="unique"):
        ScoreAnswer("q", 0.5, (ScoreProbability(0, 0.5), ScoreProbability(0, 0.5)))


def test_typed_request_and_response_roundtrip():
    request = DecisionRequest(
        ContextInput((TextEvidence("q"),)),
        DecisionParams((PredicateQuestion("q", "yes?"),)),
    )
    response = DecisionResponse(
        DecisionResult((PredicateAnswer("q", 0.8),)),
        "m",
        JsonPart.from_value({"input_tokens": 0}),
        JsonPart.from_value({"filter": None}),
    )
    assert dict_to_obj(obj_to_dict(request, True), {}) == request
    assert dict_to_obj(obj_to_dict(response, True), {}) == response


def test_jsonl_loader_preserves_mixed_labels_and_nulls(tmp_path):
    path = tmp_path / "data.jsonl"
    samples = [
        {
            "input": "yes",
            "questions": [{"type": "predicate", "id": "q", "instructions": "yes?"}],
            "reference": {"q": True},
        },
        {
            "input": [{"type": "json", "data": {"optional": None}}],
            "questions": [
                {
                    "type": "choice",
                    "id": "q",
                    "instructions": "which?",
                    "options": [{"value": "a"}, {"value": "b"}],
                }
            ],
            "reference": {"q": "a"},
        },
    ]
    path.write_text("\n".join(json.dumps(sample) for sample in samples))
    dataset = DecisionJSONLDataset(str(path))
    split = dataset.test_set
    assert split is not None
    assert len(split) == 2
    assert json.loads(split[0]["record_json"])["reference"]["q"] is True
    part = request_from_sample(split[1]).context.parts[0]
    assert isinstance(part, JsonPart)
    assert part.value == {"optional": None}


@pytest.mark.parametrize("reference", [{}, {"q": "true"}, {"other": True}])
def test_invalid_reference_is_rejected_before_inference(reference):
    with pytest.raises(ValueError):
        request_from_sample(
            {
                "input": "q",
                "questions": [{"id": "q", "type": "predicate", "instructions": "yes?"}],
                "reference": reference,
            }
        )
