"""Native question, response, authentication and usage contract tests.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

import base64
import json

import httpx
import pytest
from sieval_decisions import (
    ChoiceAnswer,
    ChoiceOption,
    ChoiceQuestion,
    ContextInput,
    DecisionParams,
    DecisionProtocolError,
    DecisionRequest,
    ImageEvidence,
    JsonPart,
    OpenRouterDecisionsClient,
    PredicateAnswer,
    PredicateQuestion,
    RefusalAnswer,
    ScoreAnswer,
    ScoreQuestion,
    TextEvidence,
)

PROVIDERS = [
    OpenRouterDecisionsClient,
]


def request():
    return DecisionRequest(
        ContextInput((TextEvidence("Customer was charged twice."),)),
        DecisionParams(
            (
                PredicateQuestion("damage.check", "Is there damage?"),
                ChoiceQuestion(
                    "department",
                    "Which department?",
                    (
                        ChoiceOption("billing", "Payments"),
                        ChoiceOption("support", "Other"),
                    ),
                ),
                ScoreQuestion("priority", "How urgent?", ("routine", "urgent")),
            ),
            require_probabilities=True,
        ),
    )


def payload():
    ids = ("damage.check", "department", "priority")
    result = {
        "answers": {
            ids[0]: {"type": "noul", "noul": 0.1, "confidence": 0.9},
            ids[1]: {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.9,
                "probabilities": {"billing": 0.9, "support": 0.1},
            },
            ids[2]: {
                "type": "score",
                "score": 0.75,
                "confidence": 0.7,
                "probabilities": {"0": 0.25, "1": 0.75},
                "legend": {"0": "routine", "1": "urgent"},
            },
        },
        "usage": {"input_tokens": 7, "output_tokens": 0, "total_tokens": 7},
    }
    result["model"] = "snapshot-model"
    return result


@pytest.mark.anyio
@pytest.mark.parametrize("implementation", PROVIDERS)
async def test_native_request_and_response_share_one_public_contract(
    implementation, monkeypatch
):
    observed = []
    monkeypatch.setenv(implementation.default_api_key_env, "test-token")

    def respond(sent):
        observed.append(sent)
        return httpx.Response(200, json=payload())

    client = implementation(
        "test-model",
        base_url="https://provider.invalid/v1",
        transport=httpx.MockTransport(respond),
    )
    try:
        result = await client.decide(request())
    finally:
        await client.aclose()
    assert [answer.question_id for answer in result.result.answers] == [
        "damage.check",
        "department",
        "priority",
    ]
    assert isinstance(result.result.answers[0], PredicateAnswer)
    assert isinstance(result.result.answers[1], ChoiceAnswer)
    assert isinstance(result.result.answers[2], ScoreAnswer)
    assert result.result.answers[0].probability == 0.1
    assert result.result.answers[1].choice == "billing"
    assert result.result.answers[2].score == 0.75
    assert result.usage.value["output_tokens"] == 0
    [sent] = observed
    body = json.loads(sent.content)
    assert "reference" not in body
    assert sent.headers["Authorization"] == "Bearer test-token"
    assert sent.url.path == "/alpha/decisions"
    assert body["questions"]["damage.check"]["type"] == "noul"


@pytest.mark.parametrize("implementation", PROVIDERS)
@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "unknown",
        "wrong_type",
        "missing_distribution",
        "partial",
        "mass",
        "nan",
    ],
)
def test_invalid_answers_are_rejected(implementation, bad):
    client = implementation("m", base_url="https://example.invalid/v1")
    data = payload()
    answer = data["answers"]["department"]
    if bad == "missing":
        data["answers"].pop("priority")
    elif bad == "unknown":
        answer["choice"] = "unknown"
    elif bad == "wrong_type":
        answer["type"] = "noul"
    elif bad == "missing_distribution":
        del answer["probabilities"]
    elif bad == "partial":
        answer["probabilities"].pop("support")
    elif bad == "mass":
        answer["probabilities"]["billing"] = 0.5
    else:
        answer["probabilities"]["billing"] = float("nan")
    from sieval_decisions import validate_decision_result

    with pytest.raises((ValueError, TypeError, KeyError)):
        parsed = client.parse_response(data, request())
        validate_decision_result(request().decisions, parsed.result)


@pytest.mark.parametrize(
    "implementation",
    [OpenRouterDecisionsClient],
)
def test_per_question_refusal_does_not_replace_other_answers(implementation):
    client = implementation("m", base_url="https://example.invalid/v1")
    data = payload()
    data["answers"]["damage.check"] = {"type": "refusal"}
    parsed = client.parse_response(data, request())
    assert isinstance(parsed.result.answers[0], RefusalAnswer)
    assert parsed.result.answers[1].choice == "billing"


@pytest.mark.parametrize("implementation", PROVIDERS)
def test_json_and_image_evidence_is_explicitly_adapted(implementation):
    client = implementation("m", base_url="https://example.invalid/v1")
    original = request()
    evidence = ContextInput(
        (
            JsonPart.from_value({"filter": None}),
            ImageEvidence(
                data=base64.b64encode(b"image").decode(), media_type="image/png"
            ),
        )
    )
    body = client.build_request(DecisionRequest(evidence, original.decisions))
    assert body["state"][0] == {"filter": None}
    assert body["state"][1]["type"] == "image_url"


@pytest.mark.anyio
async def test_malformed_response_raises_protocol_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    client = OpenRouterDecisionsClient(
        "m",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"answers": []})
        ),
    )
    try:
        with pytest.raises(DecisionProtocolError):
            await client.decide(request())
    finally:
        await client.aclose()
