"""Question and answer translation shared by matching provider wire contracts.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

import math
from collections.abc import Mapping

from sieval.core.types import JSONValue

from .._validation import (
    json_copy,
    nonempty_string,
    number,
    object_value,
    optional_probability,
    probability,
)
from ..content import JsonPart
from ..decisions import (
    ChoiceAnswer,
    ChoiceProbability,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionGuidance,
    DecisionQuestion,
    DecisionResult,
    PredicateAnswer,
    PredicateQuestion,
    RefusalAnswer,
    ScoreAnswer,
    ScoreLegendEntry,
    ScoreProbability,
    ScoreQuestion,
)


def guidance_value(value: DecisionGuidance) -> JSONValue:
    return value.value if isinstance(value, JsonPart) else value


def mapped_question(question: DecisionQuestion) -> dict[str, JSONValue]:
    """Map the shared question to OpenRouter's typed question object."""
    result: dict[str, JSONValue] = {"instructions": question.instructions}
    if isinstance(question, PredicateQuestion):
        result["type"] = "noul"
        criteria: dict[str, JSONValue] = {}
        if question.true_guidance is not None:
            criteria["true"] = guidance_value(question.true_guidance)
        if question.false_guidance is not None:
            criteria["false"] = guidance_value(question.false_guidance)
        if criteria:
            result["criteria"] = criteria
    elif isinstance(question, ChoiceQuestion):
        result["type"] = "choice"
        result["criteria"] = {
            option.value: (
                "" if option.guidance is None else guidance_value(option.guidance)
            )
            for option in question.options
        }
    else:
        result["type"] = "score"
        result["criteria"] = [guidance_value(level) for level in question.levels]
    return result


def mapped_answer(
    data: Mapping[str, object], question_id: str, *, allow_refusal: bool
) -> DecisionAnswer:
    answer_type = data.get("type")
    if answer_type == "refusal" and allow_refusal:
        return RefusalAnswer(question_id)
    confidence = optional_probability(data.get("confidence"), "confidence")
    if answer_type == "noul":
        return PredicateAnswer(
            question_id, probability(data.get("noul"), "noul"), confidence
        )
    if answer_type == "choice":
        distribution = data.get("probabilities")
        probabilities = None
        if distribution is not None:
            mapping = object_value(distribution, "choice probabilities")
            probabilities = tuple(
                ChoiceProbability(key, probability(value, "probability"))
                for key, value in mapping.items()
            )
        return ChoiceAnswer(
            question_id,
            nonempty_string(data.get("choice"), "choice"),
            probabilities,
            confidence,
        )
    if answer_type == "score":
        probabilities = None
        distribution = data.get("probabilities")
        if distribution is not None:
            mapping = object_value(distribution, "score probabilities")
            probabilities = tuple(
                ScoreProbability(level_index(key), probability(value, "probability"))
                for key, value in mapping.items()
            )
        legend = None
        if data.get("legend") is not None:
            mapping = object_value(data["legend"], "score legend")
            legend = tuple(
                ScoreLegendEntry(
                    level_index(key),
                    value
                    if isinstance(value, str)
                    else JsonPart.from_value(json_copy(value)),
                )
                for key, value in mapping.items()
            )
        return ScoreAnswer(
            question_id,
            number(data.get("score"), "score"),
            probabilities,
            confidence,
            legend,
        )
    raise ValueError(f"unsupported answer type {answer_type!r}")


def level_index(value: object) -> int:
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise ValueError("score probability keys must be non-negative integer strings")
    result = int(value)
    if str(result) != value:
        raise ValueError("score probability keys must use canonical integer spelling")
    return result


def named_decision_result(
    raw: object,
    questions: tuple[DecisionQuestion, ...],
    *,
    allow_refusal: bool,
) -> DecisionResult:
    """Read map-based answers, restore public IDs, and validate native results."""
    answers = object_value(raw, "answers")
    ids = tuple(q.question_id for q in questions)
    if set(answers) != set(ids):
        raise ValueError("answer IDs must exactly match question IDs")
    parsed = []
    for wire_id, question in zip(ids, questions, strict=True):
        answer = mapped_answer(
            object_value(answers[wire_id], "answer"),
            question.question_id,
            allow_refusal=allow_refusal,
        )
        validate_distribution(question, answer)
        parsed.append(answer)
    return DecisionResult(tuple(parsed))


def validate_distribution(question: DecisionQuestion, answer: DecisionAnswer) -> None:
    """Check complete distributions within provider precision, preserving values.

    OpenRouter may display probabilities to two decimal places.
    Reported values are preserved without normalization.
    """
    if isinstance(answer, RefusalAnswer | PredicateAnswer):
        return
    if answer.probabilities is None:
        return
    if isinstance(question, ChoiceQuestion) and isinstance(answer, ChoiceAnswer):
        expected = {option.value for option in question.options}
        if answer.choice not in expected:
            raise ValueError("returned choice is not a question option")
        actual = {entry.value for entry in answer.probabilities}
    elif isinstance(question, ScoreQuestion) and isinstance(answer, ScoreAnswer):
        expected = set(range(len(question.levels)))
        actual = {entry.level for entry in answer.probabilities}
    else:
        return  # The common question/result validator reports the type mismatch.
    if actual != expected:
        raise ValueError("reported probabilities must cover every option or level")
    half_unit = 0.005
    lower_mass = math.fsum(
        max(0, entry.probability - half_unit) for entry in answer.probabilities
    )
    upper_mass = math.fsum(
        min(1, entry.probability + half_unit) for entry in answer.probabilities
    )
    if not lower_mass <= 1 + 1e-6 or not upper_mass >= 1 - 1e-6:
        raise ValueError("reported probabilities must sum to one")
    if isinstance(answer, ChoiceAnswer):
        selected = next(
            entry.probability
            for entry in answer.probabilities
            if entry.value == answer.choice
        )
        if selected + 2 * half_unit + 1e-6 < max(
            entry.probability for entry in answer.probabilities
        ):
            raise ValueError("returned choice must have maximal probability")
    elif isinstance(answer, ScoreAnswer):
        expected_score = math.fsum(
            entry.level * entry.probability for entry in answer.probabilities
        )
        tolerance = (
            half_unit * (1 + sum(entry.level for entry in answer.probabilities)) + 1e-6
        )
        if not math.isclose(answer.score, expected_score, rel_tol=0, abs_tol=tolerance):
            raise ValueError(
                "reported score must match its probability-weighted levels"
            )


def normalize_usage(raw: object) -> dict[str, JSONValue] | None:
    if raw is None:
        return None
    usage = object_value(raw, "usage")
    result: dict[str, JSONValue] = {}
    fields = {
        "input_tokens": "input_tokens",
        "output_tokens": "output_tokens",
        "total_tokens": "total_tokens",
    }
    for target, source in fields.items():
        if source in usage:
            value = usage[source]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"usage.{source} must be a non-negative integer")
            result[target] = value
    for container, source, target in (
        ("input_tokens_details", "cached_tokens", "cached_tokens"),
        ("input_tokens_details", "cache_write_tokens", "cache_write_tokens"),
        ("output_tokens_details", "reasoning_tokens", "reasoning_tokens"),
    ):
        if container in usage:
            details = object_value(usage[container], f"usage.{container}")
            if source in details:
                value = details[source]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        f"usage.{container}.{source} must be a non-negative integer"
                    )
                result[target] = value
    return result
