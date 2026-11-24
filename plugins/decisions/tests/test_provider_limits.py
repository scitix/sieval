"""Provider-specific limits and semantic differences fail explicitly.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

import pytest
from sieval_decisions import (
    ChoiceOption,
    ChoiceQuestion,
    ContextInput,
    DecisionParams,
    DecisionRequest,
    OpenRouterDecisionsClient,
    PredicateQuestion,
    ScoreQuestion,
    TextEvidence,
)


def make_request(question, context=None, **params):
    return DecisionRequest(
        context or ContextInput((TextEvidence("context"),)),
        DecisionParams((question,), **params),
    )


@pytest.mark.parametrize("implementation", [OpenRouterDecisionsClient])
def test_predicate_outcome_criteria_preserve_provider_field_names(implementation):
    question = PredicateQuestion("q", "Eligible?", "Meets rules", "Does not")
    body = implementation("m").build_request(make_request(question))
    sent = body["questions"]["q"]
    assert sent["criteria"] == {"true": "Meets rules", "false": "Does not"}


@pytest.mark.parametrize("key", ["01", "-1", "1.0", "١"])
def test_score_distribution_keys_are_canonical_integers(key):
    request = make_request(ScoreQuestion("q", "rate?", ("low", "high")))
    with pytest.raises(ValueError, match="integer"):
        OpenRouterDecisionsClient("m").parse_response(
            {
                "model": "m",
                "answers": {
                    "q": {
                        "type": "score",
                        "score": 0.5,
                        "probabilities": {key: 0.5, "0": 0.5},
                    }
                },
            },
            request,
        )


def test_rounded_distribution_is_preserved_without_renormalizing():
    question = ChoiceQuestion(
        "q", "which?", tuple(ChoiceOption(key) for key in ("a", "b", "c"))
    )
    request = make_request(question)
    data = {
        "model": "m",
        "answers": {
            "q": {
                "type": "choice",
                "choice": "a",
                "probabilities": {"a": 0.33, "b": 0.33, "c": 0.33},
            }
        },
    }
    answer = (
        OpenRouterDecisionsClient("m").parse_response(data, request).result.answers[0]
    )
    from sieval_decisions import ChoiceAnswer

    assert isinstance(answer, ChoiceAnswer)
    assert answer.probabilities is not None
    assert sum(p.probability for p in answer.probabilities) == pytest.approx(0.99)


@pytest.mark.parametrize(
    "answer",
    [
        {"type": "choice", "choice": "a", "probabilities": {"a": 0.1, "b": 0.9}},
        {"type": "score", "score": 0.1, "probabilities": {"0": 0.1, "1": 0.9}},
    ],
)
def test_value_must_agree_with_its_probability_distribution(answer):
    question = (
        ChoiceQuestion("q", "which?", (ChoiceOption("a"), ChoiceOption("b")))
        if answer["type"] == "choice"
        else ScoreQuestion("q", "rate?", ("low", "high"))
    )
    with pytest.raises(ValueError):
        OpenRouterDecisionsClient("m").parse_response(
            {"model": "m", "answers": {"q": answer}}, make_request(question)
        )
