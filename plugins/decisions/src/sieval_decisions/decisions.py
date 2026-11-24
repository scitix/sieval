"""Provider-neutral typed questions and persisted decision answers.

Question identifiers correlate answers; their spelling is not model input.
Choice submission order is retained without promising provider scoring order.
Native probabilities, scores and confidence are never inferred or normalized.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

from dataclasses import dataclass
from typing import Literal

from sieval.core.utils.serialization import sieval_record

from ._validation import (
    nonempty_string,
    number,
    probability,
    record_tuple,
)
from .content import JsonPart

type DecisionQuestionType = Literal["predicate", "choice", "score"]
type DecisionGuidance = str | JsonPart


def _guidance(value: object, name: str, *, allow_null: bool = True) -> None:
    if not isinstance(value, str | JsonPart):
        raise TypeError(f"{name} must be text or JsonPart")
    if isinstance(value, JsonPart) and value.value is None and not allow_null:
        raise ValueError(f"{name} must not be null")


def _question(question_id: str, instructions: str) -> None:
    nonempty_string(question_id, "question_id")
    nonempty_string(instructions, "instructions")


@sieval_record
@dataclass(frozen=True)
class PredicateQuestion:
    question_id: str
    instructions: str
    true_guidance: DecisionGuidance | None = None
    false_guidance: DecisionGuidance | None = None

    def __post_init__(self) -> None:
        _question(self.question_id, self.instructions)
        for name in ("true_guidance", "false_guidance"):
            value = getattr(self, name)
            if value is not None:
                _guidance(value, name)


@sieval_record
@dataclass(frozen=True)
class ChoiceOption:
    value: str
    guidance: DecisionGuidance | None = None

    def __post_init__(self) -> None:
        nonempty_string(self.value, "choice option value")
        if self.guidance is not None:
            _guidance(self.guidance, "choice option guidance")


@sieval_record
@dataclass(frozen=True)
class ChoiceQuestion:
    question_id: str
    instructions: str
    options: tuple[ChoiceOption, ...]

    def __post_init__(self) -> None:
        _question(self.question_id, self.instructions)
        record_tuple(self.options, ChoiceOption, "options")
        if len(self.options) < 2:
            raise ValueError("choice questions require at least two options")
        if len({item.value for item in self.options}) != len(self.options):
            raise ValueError("choice option values must be unique")


@sieval_record
@dataclass(frozen=True)
class ScoreQuestion:
    question_id: str
    instructions: str
    levels: tuple[DecisionGuidance, ...]

    def __post_init__(self) -> None:
        _question(self.question_id, self.instructions)
        record_tuple(self.levels, (str, JsonPart), "levels")
        if len(self.levels) < 2:
            raise ValueError("score questions require at least two ordered levels")
        for level in self.levels:
            _guidance(level, "score level", allow_null=False)


type DecisionQuestion = PredicateQuestion | ChoiceQuestion | ScoreQuestion


@sieval_record
@dataclass(frozen=True)
class DecisionParams:
    questions: tuple[DecisionQuestion, ...]
    require_probabilities: bool = False
    require_confidence: bool = False

    def __post_init__(self) -> None:
        record_tuple(
            self.questions,
            (PredicateQuestion, ChoiceQuestion, ScoreQuestion),
            "questions",
        )
        if not self.questions:
            raise ValueError("questions must be non-empty")
        if len({item.question_id for item in self.questions}) != len(self.questions):
            raise ValueError("question ids must be unique")
        for name in ("require_probabilities", "require_confidence"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")


def _answer(question_id: str, confidence: float | None) -> None:
    nonempty_string(question_id, "question_id")
    if confidence is not None:
        probability(confidence, "confidence")


@sieval_record
@dataclass(frozen=True)
class PredicateAnswer:
    question_id: str
    probability: float
    confidence: float | None = None

    def __post_init__(self) -> None:
        _answer(self.question_id, self.confidence)
        probability(self.probability, "probability")


@sieval_record
@dataclass(frozen=True)
class ChoiceProbability:
    value: str
    probability: float

    def __post_init__(self) -> None:
        nonempty_string(self.value, "choice probability value")
        probability(self.probability, "probability")


@sieval_record
@dataclass(frozen=True)
class ChoiceAnswer:
    question_id: str
    choice: str
    probabilities: tuple[ChoiceProbability, ...] | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        _answer(self.question_id, self.confidence)
        nonempty_string(self.choice, "choice")
        if self.probabilities is not None:
            record_tuple(self.probabilities, ChoiceProbability, "probabilities")
            if len({item.value for item in self.probabilities}) != len(
                self.probabilities
            ):
                raise ValueError("choice probability values must be unique")


def _level(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("level must be an integer")
    if value < 0:
        raise ValueError("level must be non-negative")


@sieval_record
@dataclass(frozen=True)
class ScoreProbability:
    level: int
    probability: float

    def __post_init__(self) -> None:
        _level(self.level)
        probability(self.probability, "probability")


@sieval_record
@dataclass(frozen=True)
class ScoreLegendEntry:
    level: int
    guidance: DecisionGuidance

    def __post_init__(self) -> None:
        _level(self.level)
        _guidance(self.guidance, "legend guidance", allow_null=False)


@sieval_record
@dataclass(frozen=True)
class ScoreAnswer:
    question_id: str
    score: float
    probabilities: tuple[ScoreProbability, ...] | None = None
    confidence: float | None = None
    legend: tuple[ScoreLegendEntry, ...] | None = None

    def __post_init__(self) -> None:
        _answer(self.question_id, self.confidence)
        number(self.score, "score")
        for name, item_type in (
            ("probabilities", ScoreProbability),
            ("legend", ScoreLegendEntry),
        ):
            entries = getattr(self, name)
            if entries is not None:
                record_tuple(entries, item_type, name)
                if len({item.level for item in entries}) != len(entries):
                    raise ValueError(f"{name} levels must be unique")


@sieval_record
@dataclass(frozen=True)
class RefusalAnswer:
    question_id: str

    def __post_init__(self) -> None:
        nonempty_string(self.question_id, "question_id")


type DecisionAnswer = PredicateAnswer | ChoiceAnswer | ScoreAnswer | RefusalAnswer


@sieval_record
@dataclass(frozen=True)
class DecisionResult:
    answers: tuple[DecisionAnswer, ...]

    def __post_init__(self) -> None:
        record_tuple(
            self.answers,
            (PredicateAnswer, ChoiceAnswer, ScoreAnswer, RefusalAnswer),
            "answers",
        )
        if len({item.question_id for item in self.answers}) != len(self.answers):
            raise ValueError("answer question ids must be unique")


def validate_decision_result(params: DecisionParams, result: DecisionResult) -> None:
    """Validate successful answers without changing native result values.

    Probability mass and score reconstruction are provider-specific: reported
    precision differs. The common contract validates identities, bounds and
    completeness; it does not renormalize or reconstruct native outputs.
    """

    if not isinstance(result, DecisionResult):
        raise TypeError("decisions channel must be a DecisionResult")
    if tuple(item.question_id for item in result.answers) != tuple(
        item.question_id for item in params.questions
    ):
        raise ValueError("decision answers must match question identity and order")
    for question, answer in zip(params.questions, result.answers, strict=True):
        if isinstance(answer, RefusalAnswer):
            continue
        if params.require_confidence and answer.confidence is None:
            raise ValueError(f"question {question.question_id!r} requires confidence")
        if isinstance(question, PredicateQuestion):
            if not isinstance(answer, PredicateAnswer):
                raise ValueError("predicate question requires a predicate answer")
            continue
        if isinstance(question, ChoiceQuestion):
            if not isinstance(answer, ChoiceAnswer):
                raise ValueError("choice question requires a choice answer")
            expected = {item.value for item in question.options}
            if answer.choice not in expected:
                raise ValueError("returned choice is not a question option")
            actual = (
                None
                if answer.probabilities is None
                else {item.value for item in answer.probabilities}
            )
        else:
            if not isinstance(answer, ScoreAnswer):
                raise ValueError("score question requires a score answer")
            expected = set(range(len(question.levels)))
            if not 0 <= answer.score <= len(question.levels) - 1:
                raise ValueError("returned score is outside the question rubric")
            if (
                answer.legend is not None
                and not {item.level for item in answer.legend} <= expected
            ):
                raise ValueError("legend contains a level outside the question rubric")
            actual = (
                None
                if answer.probabilities is None
                else {item.level for item in answer.probabilities}
            )
        if actual is not None and not actual <= expected:
            raise ValueError("probabilities contain an unknown option or rubric level")
        if params.require_probabilities:
            if actual is None:
                raise ValueError(
                    f"question {question.question_id!r} requires probabilities"
                )
            if actual != expected:
                raise ValueError("required probabilities must be complete")
