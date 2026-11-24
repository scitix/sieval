"""Reusable Decision API clients and typed questions for SiEval plugins.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

from .client import (
    DecisionProtocolError,
    DecisionRequest,
    DecisionResponse,
    DecisionsClient,
    UnsupportedDecisionInput,
)
from .content import ContextInput, ImageEvidence, JsonPart, TextEvidence
from .decisions import (
    ChoiceAnswer,
    ChoiceOption,
    ChoiceProbability,
    ChoiceQuestion,
    DecisionParams,
    DecisionResult,
    PredicateAnswer,
    PredicateQuestion,
    RefusalAnswer,
    ScoreAnswer,
    ScoreLegendEntry,
    ScoreProbability,
    ScoreQuestion,
    validate_decision_result,
)
from .factory import create_decisions_client
from .providers import OpenRouterDecisionsClient

__all__ = [
    "ChoiceAnswer",
    "ChoiceOption",
    "ChoiceProbability",
    "ChoiceQuestion",
    "ContextInput",
    "DecisionParams",
    "DecisionProtocolError",
    "DecisionRequest",
    "DecisionResponse",
    "DecisionResult",
    "DecisionsClient",
    "ImageEvidence",
    "JsonPart",
    "OpenRouterDecisionsClient",
    "PredicateAnswer",
    "PredicateQuestion",
    "RefusalAnswer",
    "ScoreAnswer",
    "ScoreLegendEntry",
    "ScoreProbability",
    "ScoreQuestion",
    "TextEvidence",
    "UnsupportedDecisionInput",
    "validate_decision_result",
    "create_decisions_client",
]
