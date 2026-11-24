"""Subject identity and transport settings remain explicit and credential-free.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

import json

import httpx
import pytest
from sieval_decisions import (
    ContextInput,
    DecisionParams,
    DecisionRequest,
    OpenRouterDecisionsClient,
    PredicateAnswer,
    PredicateQuestion,
    TextEvidence,
    create_decisions_client,
)


def test_request_identity_tracks_endpoint_options_and_revision(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-value")
    first = OpenRouterDecisionsClient(
        "m", revision="v1", options={"provider": {"only": ["TypeSafe"]}}
    )
    same = OpenRouterDecisionsClient(
        "m",
        revision="v1",
        options={"provider": {"only": ["TypeSafe"]}},
        timeout=10,
        max_retries=0,
        api_key_env="ANOTHER_KEY",
    )
    assert first.call_key() == same.call_key()
    assert first.request_args() == same.request_args()
    assert "secret-value" not in json.dumps(first.subject_meta())
    assert "secret-value" not in json.dumps(first.request_args())
    assert first.call_key() != OpenRouterDecisionsClient("m", revision="v2").call_key()
    assert (
        first.request_args()
        != OpenRouterDecisionsClient(
            "m", base_url="https://another.invalid/api"
        ).request_args()
    )


@pytest.mark.parametrize(
    "options",
    [{"questions": {}}, {"model": "other"}, {"api_key": "secret"}, {"stream": True}],
)
def test_options_cannot_override_protected_payload_fields(options):
    with pytest.raises(ValueError, match="unsupported"):
        OpenRouterDecisionsClient("m", options=options)


@pytest.mark.parametrize(
    "url",
    [
        "https://user:secret@example.com/v1",
        "https://example.com/v1?api_key=secret",
        "https://example.com/v1#secret",
        "file:///tmp/api",
    ],
)
def test_base_url_rejects_credentials_and_non_http_schemes(url):
    with pytest.raises(ValueError):
        OpenRouterDecisionsClient("m", base_url=url)


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown decisions provider"):
        create_decisions_client("typo", "m")


@pytest.mark.anyio
async def test_transport_retries_use_one_logical_decision_call(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    attempts = []

    def respond(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(503, headers={"retry-after-ms": "1"})
        return httpx.Response(
            200,
            json={
                "model": "m",
                "answers": {"q": {"type": "noul", "noul": 0.8}},
            },
        )

    client = OpenRouterDecisionsClient(
        "m", max_retries=1, transport=httpx.MockTransport(respond)
    )
    request = DecisionRequest(
        ContextInput((TextEvidence("q"),)),
        DecisionParams((PredicateQuestion("q", "yes?"),)),
    )
    try:
        response = await client.decide(request)
    finally:
        await client.aclose()
    answer = response.result.answers[0]
    assert isinstance(answer, PredicateAnswer)
    assert answer.probability == 0.8
    assert len(attempts) == 2


@pytest.mark.anyio
async def test_http_redirect_is_not_followed(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    attempts = []

    def respond(request):
        attempts.append(request)
        return httpx.Response(
            307, headers={"location": "https://another.invalid/decisions"}
        )

    client = OpenRouterDecisionsClient("m", transport=httpx.MockTransport(respond))
    request = DecisionRequest(
        ContextInput((TextEvidence("q"),)),
        DecisionParams((PredicateQuestion("q", "yes?"),)),
    )
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.decide(request)
    finally:
        await client.aclose()
    assert len(attempts) == 1
