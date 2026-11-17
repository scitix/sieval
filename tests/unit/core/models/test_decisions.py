"""Decision question and answer contracts.

AI-Generated Code - GPT-6 (OpenAI)
"""

from typing import cast

import pytest

from sieval.core.models import (
    ChoiceAnswer,
    ChoiceOption,
    ChoiceProbability,
    ChoiceQuestion,
    DecisionParams,
    DecisionResult,
    JsonPart,
    PredicateAnswer,
    PredicateQuestion,
    RefusalAnswer,
    ScoreAnswer,
    ScoreLegendEntry,
    ScoreProbability,
    ScoreQuestion,
)
from sieval.core.models.decisions import validate_decision_result
from sieval.core.utils.serialization import (
    dict_to_obj,
    global_type_registry,
    obj_to_dict,
)


def questions():
    return (
        PredicateQuestion("defect", "Is the product defective?"),
        ChoiceQuestion(
            "department",
            "Which department?",
            (ChoiceOption("support", "Product support"), ChoiceOption("billing")),
        ),
        ScoreQuestion("urgency", "How urgent?", ("Routine", "Soon", "Today")),
    )


def test_multiple_question_types_and_partial_refusal_roundtrip():
    params = DecisionParams(questions())
    result = DecisionResult(
        (
            PredicateAnswer("defect", 0.9),
            ChoiceAnswer(
                "department",
                "support",
                (ChoiceProbability("support", 0.8), ChoiceProbability("billing", 0.2)),
                confidence=0.6,
            ),
            RefusalAnswer("urgency"),
        )
    )
    validate_decision_result(params, result)
    restored = dict_to_obj(obj_to_dict(result, True), global_type_registry)
    assert restored == result
    assert isinstance(restored.answers[1].probabilities[0], ChoiceProbability)
    assert restored.answers[1].confidence == 0.6


def test_question_guidance_preserves_null_distinct_from_absent():
    option = ChoiceOption("support", JsonPart.from_value(None))
    assert option.guidance != ChoiceOption("support").guidance
    restored = dict_to_obj(obj_to_dict(option, True), global_type_registry)
    assert restored.guidance.value is None


@pytest.mark.parametrize(
    "value", [{"a": None, "b": [None, {"c": None}]}, [None, {"x": 2}]]
)
def test_json_material_roundtrip_preserves_nested_nulls(value):
    part = JsonPart.from_value(value)
    restored = dict_to_obj(obj_to_dict(part, True), global_type_registry)
    assert restored.value == value
    detached = cast(dict[str, object] | list[object], part.value)
    if isinstance(detached, dict):
        detached["a"] = "changed"
    elif isinstance(detached, list):
        detached.append("changed")
    else:
        pytest.fail("JSON material must decode to an object or array")
    assert part.value == value


@pytest.mark.parametrize("encoded", ['{"x": 1, "x": 2}', '{"x": NaN}', '"text"', "42"])
def test_json_material_rejects_ambiguous_or_unsupported_values(encoded):
    with pytest.raises((TypeError, ValueError)):
        JsonPart(encoded)


@pytest.mark.parametrize("value", [True, False, 3])
def test_choice_values_are_strings_and_are_not_coerced(value):
    with pytest.raises(TypeError):
        ChoiceOption(value)


def test_question_ids_and_option_ids_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        DecisionParams((questions()[0], questions()[0]))
    with pytest.raises(ValueError, match="unique"):
        ChoiceQuestion("q", "Pick", (ChoiceOption("a"), ChoiceOption("a")))


def test_request_requires_at_least_one_question():
    with pytest.raises(ValueError, match="non-empty"):
        DecisionParams(())


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_probabilities_are_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        PredicateAnswer("q", value)


def test_answers_must_match_question_identity_type_and_order():
    params = DecisionParams(questions()[:2])
    for answers in (
        (PredicateAnswer("defect", 0.9),),
        (PredicateAnswer("wrong", 0.9), ChoiceAnswer("department", "support")),
        (ChoiceAnswer("department", "support"), PredicateAnswer("defect", 0.9)),
        (ChoiceAnswer("defect", "support"), ChoiceAnswer("department", "support")),
    ):
        with pytest.raises(ValueError):
            validate_decision_result(params, DecisionResult(answers))


def test_missing_distribution_is_optional_unless_requested():
    params = DecisionParams((questions()[1],))
    result = DecisionResult((ChoiceAnswer("department", "support"),))
    validate_decision_result(params, result)
    with pytest.raises(ValueError, match="probabilities"):
        validate_decision_result(
            DecisionParams(params.questions, require_probabilities=True), result
        )


def test_requested_distribution_must_cover_all_options_without_filling_gaps():
    params = DecisionParams((questions()[1],), require_probabilities=True)
    result = DecisionResult(
        (ChoiceAnswer("department", "support", (ChoiceProbability("support", 0.8),)),)
    )
    with pytest.raises(ValueError, match="complete"):
        validate_decision_result(params, result)
    answer = result.answers[0]
    assert isinstance(answer, ChoiceAnswer)
    assert answer.probabilities == (ChoiceProbability("support", 0.8),)


def test_choice_and_probability_keys_must_belong_to_the_question():
    params = DecisionParams((questions()[1],))
    for answer in (
        ChoiceAnswer("department", "unknown"),
        ChoiceAnswer("department", "support", (ChoiceProbability("unknown", 0.2),)),
    ):
        with pytest.raises(ValueError, match="option"):
            validate_decision_result(params, DecisionResult((answer,)))


def test_score_keeps_fractional_native_value_and_distribution():
    params = DecisionParams((questions()[2],), require_probabilities=True)
    answer = ScoreAnswer(
        "urgency",
        1.8,
        (
            ScoreProbability(0, 0.05),
            ScoreProbability(1, 0.1),
            ScoreProbability(2, 0.85),
        ),
        confidence=0.7,
    )
    validate_decision_result(params, DecisionResult((answer,)))
    assert answer.score == 1.8
    assert answer.confidence == 0.7
    with pytest.raises(ValueError, match="rubric"):
        validate_decision_result(params, DecisionResult((ScoreAnswer("urgency", 3),)))


def test_score_legend_preserves_native_guidance_and_validates_level_bounds():
    params = DecisionParams((questions()[2],))
    legend = (
        ScoreLegendEntry(2, JsonPart.from_value({"action": "today", "note": None})),
    )
    result = DecisionResult((ScoreAnswer("urgency", 1.8, legend=legend),))
    validate_decision_result(params, result)
    assert dict_to_obj(obj_to_dict(result, True), global_type_registry) == result
    with pytest.raises(ValueError, match="legend"):
        validate_decision_result(
            params,
            DecisionResult(
                (ScoreAnswer("urgency", 1.8, legend=(ScoreLegendEntry(3, "outside"),)),)
            ),
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ScoreQuestion("q", "Rate", ("one",)),
        lambda: ScoreQuestion("q", "Rate", ("one", JsonPart.from_value(None))),
        lambda: ScoreProbability(True, 0.5),
        lambda: ScoreProbability(-1, 0.5),
        lambda: ScoreLegendEntry(0, JsonPart.from_value(None)),
        lambda: ScoreAnswer("q", float("nan")),
        lambda: ChoiceAnswer(
            "q", "a", (ChoiceProbability("a", 0.5), ChoiceProbability("a", 0.5))
        ),
        lambda: DecisionResult((RefusalAnswer("q"), RefusalAnswer("q"))),
        lambda: DecisionParams(questions(), require_confidence=cast(bool, 1)),
    ],
)
def test_invalid_decision_records_fail_at_construction(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_refusal_does_not_require_fabricated_probability_or_confidence():
    params = DecisionParams(
        (questions()[1],), require_probabilities=True, require_confidence=True
    )
    result = DecisionResult((RefusalAnswer("department"),))
    validate_decision_result(params, result)


def test_confidence_requirement_is_enforced_without_recomputing_it():
    params = DecisionParams((questions()[0],), require_confidence=True)
    with pytest.raises(ValueError, match="confidence"):
        validate_decision_result(
            params, DecisionResult((PredicateAnswer("defect", 0.9),))
        )
