"""Labelled Decision API evaluation independent of provider wire formats.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

import math
from collections.abc import Mapping

from sieval.core.datasets import Dataset
from sieval.core.models import Model
from sieval.core.tasks import Task
from sieval.core.tasks.context import TaskContext
from sieval.core.types import JSONValue

from ._validation import array_value, number, object_value, probability
from .client import DecisionRequest, DecisionResponse, DecisionsClient
from .decisions import ChoiceAnswer, PredicateAnswer, RefusalAnswer, ScoreAnswer
from .factory import create_decisions_client
from .samples import request_from_sample, sample_mapping


class DecisionJSONLTask(Task):
    """Evaluate predicate, choice and score questions with one shared client."""

    requires_model = False

    def __init__(
        self,
        dataset: Dataset,
        model: Model | None,
        name: str | None = None,
        *,
        provider: str,
        model_name: str,
        base_url: str | None = None,
        api_key_env: str | None = None,
        revision: str | None = None,
        options: Mapping[str, JSONValue] | None = None,
        timeout: float = 60,
        client_max_retries: int = 2,
        predicate_threshold: float = 0.5,
        score_tolerance: float = 0.5,
        require_probabilities: bool = True,
        require_confidence: bool = False,
    ) -> None:
        super().__init__(dataset, model, name)
        self._predicate_threshold = probability(
            predicate_threshold, "predicate_threshold"
        )
        self._score_tolerance = number(score_tolerance, "score_tolerance")
        if score_tolerance < 0:
            raise ValueError("score_tolerance must be non-negative")
        if not isinstance(require_probabilities, bool):
            raise TypeError("require_probabilities must be a boolean")
        if not isinstance(require_confidence, bool):
            raise TypeError("require_confidence must be a boolean")
        self._require_probabilities = require_probabilities
        self._require_confidence = require_confidence
        self._client = create_decisions_client(
            provider,
            model_name,
            base_url=base_url,
            api_key_env=api_key_env,
            revision=revision,
            options=options,
            timeout=timeout,
            max_retries=client_max_retries,
        )

    @property
    def subject(self) -> DecisionsClient:
        return self._client

    def inference_args(self) -> Mapping[str, JSONValue]:
        return {
            "require_probabilities": self._require_probabilities,
            "require_confidence": self._require_confidence,
        }

    async def preprocess(
        self, raw: Mapping[str, object], ctx: TaskContext
    ) -> DecisionRequest:
        return request_from_sample(
            raw,
            require_probabilities=self._require_probabilities,
            require_confidence=self._require_confidence,
        )

    async def infer(self, pre: DecisionRequest, ctx: TaskContext) -> DecisionResponse:
        return await self._client.decide(pre)

    async def postprocess(
        self, inf: DecisionResponse, ctx: TaskContext
    ) -> DecisionResponse:
        # Keep probabilities and refusal information available to the grader.
        return inf

    async def feedback(
        self, post: DecisionResponse, ctx: TaskContext
    ) -> tuple[bool, dict[str, JSONValue]]:
        raw = object_value(ctx.raw_sample, "raw sample")
        reference = object_value(sample_mapping(raw).get("reference"), "reference")
        outcomes: list[JSONValue] = []
        for answer in post.result.answers:
            expected = reference[answer.question_id]
            outcome: dict[str, JSONValue] = {
                "question_id": answer.question_id,
                "refused": False,
            }
            if isinstance(answer, RefusalAnswer):
                outcome.update(correct=False, refused=True)
            elif isinstance(answer, PredicateAnswer):
                if not isinstance(expected, bool):
                    raise ValueError("predicate reference must be a boolean")
                outcome["correct"] = (
                    answer.probability >= self._predicate_threshold
                ) == expected
                outcome["predicate_brier"] = (answer.probability - float(expected)) ** 2
            elif isinstance(answer, ChoiceAnswer):
                outcome["correct"] = answer.choice == expected
                if answer.probabilities is not None:
                    outcome["choice_brier"] = math.fsum(
                        (p.probability - (p.value == expected)) ** 2
                        for p in answer.probabilities
                    )
            elif isinstance(answer, ScoreAnswer):
                error = abs(answer.score - number(expected, "reference score"))
                outcome["correct"] = error <= self._score_tolerance
                outcome["absolute_error"] = error
            outcomes.append(outcome)
        return True, {"questions": outcomes}

    async def report(
        self, finals: list[TaskContext], fails: list[TaskContext]
    ) -> dict[str, JSONValue]:
        outcomes = []
        for ctx in finals:
            result = object_value(ctx.feedback_result, "feedback")
            for item in array_value(result["questions"], "question outcomes"):
                outcomes.append(object_value(item, "question outcome"))
        answered = [item for item in outcomes if not item["refused"]]
        result: dict[str, JSONValue] = {
            "n_samples": len(finals) + len(fails),
            "n_failed": len(fails),
            "n_questions": len(outcomes),
            "n_refused": len(outcomes) - len(answered),
        }
        if outcomes:
            result["accuracy"] = sum(
                item["correct"] is True for item in outcomes
            ) / len(outcomes)
            result["refusal_rate"] = (len(outcomes) - len(answered)) / len(outcomes)
        if answered:
            result["answered_accuracy"] = sum(
                item["correct"] is True for item in answered
            ) / len(answered)
        for source, target in (
            ("predicate_brier", "predicate_brier_score"),
            ("choice_brier", "choice_brier_score"),
            ("absolute_error", "mean_absolute_error"),
        ):
            values = [
                number(item[source], source) for item in outcomes if source in item
            ]
            if values:
                result[target] = math.fsum(values) / len(values)
        return result

    async def shutdown(self) -> None:
        await self._client.aclose()
