"""Parse a labelled, provider-neutral decision evaluation sample.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

import json
from collections.abc import Mapping

from ._validation import array_value, json_copy, nonempty_string, number, object_value
from .client import DecisionRequest
from .content import ContextInput, ImageEvidence, JsonPart, TextEvidence
from .decisions import (
    ChoiceOption,
    ChoiceQuestion,
    DecisionGuidance,
    DecisionParams,
    DecisionQuestion,
    PredicateQuestion,
    ScoreQuestion,
)


def guidance(value: object) -> DecisionGuidance:
    return value if isinstance(value, str) else JsonPart.from_value(json_copy(value))


def question_from_mapping(data: Mapping[str, object]) -> DecisionQuestion:
    question_id = nonempty_string(data.get("id"), "question id")
    instructions = nonempty_string(data.get("instructions"), "instructions")
    kind = data.get("type")
    if kind == "predicate":
        return PredicateQuestion(
            question_id,
            instructions,
            None if "true_guidance" not in data else guidance(data["true_guidance"]),
            None if "false_guidance" not in data else guidance(data["false_guidance"]),
        )
    if kind == "choice":
        choices = array_value(data.get("options"), "options")
        options = []
        for value in choices:
            item = object_value(value, "option")
            options.append(
                ChoiceOption(
                    nonempty_string(item.get("value"), "option value"),
                    None if "guidance" not in item else guidance(item["guidance"]),
                )
            )
        return ChoiceQuestion(question_id, instructions, tuple(options))
    if kind == "score":
        return ScoreQuestion(
            question_id,
            instructions,
            tuple(guidance(v) for v in array_value(data.get("levels"), "levels")),
        )
    raise ValueError(f"unsupported decision question type {kind!r}")


def context_from_value(value: object) -> ContextInput:
    if isinstance(value, str):
        return ContextInput((TextEvidence(value),))
    parts = []
    for item in array_value(value, "input"):
        part = object_value(item, "input part")
        kind = part.get("type")
        if kind == "text":
            parts.append(TextEvidence(nonempty_string(part.get("text"), "text")))
        elif kind == "json":
            if "data" not in part:
                raise ValueError("JSON input requires data")
            parts.append(JsonPart.from_value(json_copy(part["data"])))
        elif kind == "image":
            url = part.get("url")
            data = part.get("data")
            media_type = part.get("media_type")
            parts.append(
                ImageEvidence(
                    url=None if url is None else nonempty_string(url, "image url"),
                    data=None if data is None else nonempty_string(data, "image data"),
                    media_type=None
                    if media_type is None
                    else nonempty_string(media_type, "image media_type"),
                )
            )
        else:
            raise ValueError(f"unsupported evidence type {kind!r}")
    return ContextInput(tuple(parts))


def sample_mapping(raw: Mapping[str, object]) -> Mapping[str, object]:
    """The JSONL loader preserves each source row without Arrow coercion."""
    if "record_json" in raw:
        encoded = nonempty_string(raw["record_json"], "record_json")
        return object_value(json.loads(encoded), "sample")
    return raw


def request_from_sample(
    raw: Mapping[str, object],
    *,
    require_probabilities: bool = True,
    require_confidence: bool = False,
) -> DecisionRequest:
    sample = sample_mapping(raw)
    questions = tuple(
        question_from_mapping(object_value(value, "question"))
        for value in array_value(sample.get("questions"), "questions")
    )
    request = DecisionRequest(
        context_from_value(sample.get("input")),
        DecisionParams(questions, require_probabilities, require_confidence),
    )
    reference = object_value(sample.get("reference"), "reference")
    if set(reference) != {question.question_id for question in questions}:
        raise ValueError("reference must contain exactly the question IDs")
    for question in questions:
        value = reference[question.question_id]
        if isinstance(question, PredicateQuestion):
            if not isinstance(value, bool):
                raise ValueError("predicate reference must be a boolean")
        elif isinstance(question, ChoiceQuestion):
            if not isinstance(value, str) or value not in {
                o.value for o in question.options
            }:
                raise ValueError("choice reference must be one of the options")
        elif not 0 <= number(value, "score reference") <= len(question.levels) - 1:
            raise ValueError("score reference must be inside the rubric")
    return request
