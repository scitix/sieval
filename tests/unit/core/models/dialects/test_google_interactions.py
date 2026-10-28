"""Tests for the native Google Gemini ``interactions`` dialect.

The response fixtures under ``fixtures/google_interactions/`` are replies
recorded from the live API (``signature`` replaced by a placeholder), so lifting
is checked against what Google actually sends, not against our own reading of
the reference page.

AI-Generated Code - Claude Sonnet 5.5 (Anthropic)
"""

import copy
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import httpx
import pytest

from sieval.core.models.capabilities import (
    CAPABILITY_KEYS,
    Capability,
    HostedToolsOptions,
    MultimodalInputOptions,
    ReasoningOptions,
    Supported,
    Unsupported,
)
from sieval.core.models.connection_factory import AsyncHTTPJSONConnection
from sieval.core.models.dialect import (
    DialectError,
    Guarantee,
    OutputContractError,
    PreparedRequest,
    Rejected,
    RequestAudit,
    RequestAuditError,
    active_request_leaves,
)
from sieval.core.models.dialect_registry import DIALECT_SPECS
from sieval.core.models.dialects import google_interactions
from sieval.core.models.dialects.google_interactions import (
    CAPABILITY_DECISIONS,
    OUTPUT_CONTRACT,
    GoogleInteractionsDialect,
)
from sieval.core.models.ir import (
    ChatInput,
    ChatMessage,
    ChatRole,
    ChoiceLogprobs,
    Citation,
    CompletionInput,
    DialectOptions,
    HostedToolSpec,
    ImagePart,
    OpaqueContinuation,
    ReasoningOutput,
    ReasoningParams,
    Request,
    Response,
    SamplingParams,
    SchedulingParams,
    ScoringParams,
    SessionParams,
    StructuredOutputParams,
    TextPart,
    ToolCallPart,
    ToolParams,
    ToolResultPart,
    UsageStats,
)
from sieval.core.models.model import _LEGACY_CAPABILITY_MAP
from sieval.core.tasks.run_meta import digest, request_definition
from sieval.core.utils.meta import is_truncated_finish_reason
from sieval.core.utils.serialization import (
    dict_to_obj,
    global_type_registry,
    obj_to_dict,
)

FIXTURES = Path(__file__).parent / "fixtures" / "google_interactions"
MODEL = "gemini-3.8-flash"
OLD_MODEL = "gemini-2.5-flash"
ROOT = "https://generativelanguage.googleapis.com/"
_DROP = object()


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _say(role: ChatRole, text: str) -> ChatMessage:
    return ChatMessage(role, (TextPart(text),))


def _chat(*messages: ChatMessage) -> ChatInput:
    return ChatInput(messages or (_say("user", "Hello"),))


def _request(**changes: object) -> Request:
    values: dict[str, object] = {"input": _chat()}
    values.update(changes)
    return Request(**cast(Any, values))


def _options(**values: object) -> DialectOptions:
    return DialectOptions("google_interactions", cast(Any, values))


def _model_output(*texts: str) -> dict[str, Any]:
    return {
        "type": "model_output",
        "content": [{"type": "text", "text": text} for text in texts],
    }


def _thought(summary: list[str] | None = None) -> dict[str, Any]:
    step: dict[str, Any] = {"type": "thought", "signature": "sig-redacted"}
    if summary is not None:
        step["summary"] = [{"type": "text", "text": text} for text in summary]
    return step


def _reply(**changes: object) -> dict[str, Any]:
    """A minimal completed Interaction; ``_DROP`` removes a key."""

    reply: dict[str, Any] = {
        "status": "completed",
        "object": "interaction",
        "model": MODEL,
        "steps": [_model_output("Hi")],
        "usage": {
            "total_input_tokens": 4,
            "total_output_tokens": 2,
            "total_thought_tokens": 1,
            "total_cached_tokens": 0,
            "total_tokens": 7,
        },
    }
    reply.update(changes)
    return {key: value for key, value in reply.items() if value is not _DROP}


def _ok(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=_reply())


def _no_io(_: httpx.Request) -> httpx.Response:
    raise AssertionError("request rejection must happen before HTTP I/O")


def _connection(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    base_url: str = ROOT,
    credential: str | None = "google-secret",
) -> AsyncHTTPJSONConnection:
    return AsyncHTTPJSONConnection(
        httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(handler)),
        credential,
    )


def _dialect(
    handler: Callable[[httpx.Request], httpx.Response] = _no_io,
    *,
    model: str = MODEL,
    base_url: str = ROOT,
) -> GoogleInteractionsDialect:
    return GoogleInteractionsDialect(_connection(handler, base_url=base_url), model)


async def _run(
    handler: Callable[[httpx.Request], httpx.Response],
    request: Request | None = None,
    *,
    model: str = MODEL,
    base_url: str = ROOT,
    credential: str | None = "google-secret",
) -> Response:
    connection = _connection(handler, base_url=base_url, credential=credential)
    try:
        return await GoogleInteractionsDialect(connection, model).arun(
            request or _request()
        )
    finally:
        await connection.aclose()


def _json_handler(payload: object, status: int = 200):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handler


def _audit(req: Request) -> RequestAudit:
    audit = RequestAudit(active_request_leaves(req))
    _dialect().validate_request(req, audit, cast(Any, SimpleNamespace()))
    return audit


def _edit(*path: str, value: object = _DROP) -> Callable[[dict[str, Any]], None]:
    def mutate(body: dict[str, Any]) -> None:
        target = body
        for key in path[:-1]:
            target = target[key]
        if value is _DROP:
            del target[path[-1]]
        else:
            target[path[-1]] = value

    return mutate


def _tamper_prepared(
    monkeypatch: pytest.MonkeyPatch, mutate: Callable[[dict[str, Any]], None]
) -> None:
    """Rewrite the prepared body after lowering, as a buggy lowering would."""

    real = google_interactions.PreparedRequest

    def build(
        operation: str, body: Mapping[str, Any], context: object = None
    ) -> PreparedRequest:
        edited = copy.deepcopy(dict(body))
        mutate(edited)
        return real(operation, edited, context)

    monkeypatch.setattr(google_interactions, "PreparedRequest", build)


_IMAGE_INPUT = _chat(
    ChatMessage(
        "user",
        (
            TextPart("see"),
            ImagePart(data="aGk=", media_type="image/png", detail="high"),
        ),
    )
)
_TOOL_INPUT = _chat(
    _say("user", "go"),
    ChatMessage("assistant", (ToolCallPart("c1", "f", {"a": 1}),)),
    ChatMessage("tool", (ToolResultPart("c1", "ok", is_error=True),)),
)
_STRUCTURED = StructuredOutputParams(
    format="json_schema", schema={"type": "object"}, name="n", strict=True
)
_FUNCTION_TOOLS = ToolParams(
    functions=({"name": "f", "description": "d", "parameters": {"type": "object"}},),
    choice="auto",
    parallel=True,
)
_OPAQUE = SessionParams(
    opaque_continuation=OpaqueContinuation("google_interactions", "x")
)

#: One request per leaf the dialect refuses to lower.
_REJECTED_REQUESTS: dict[str, Request] = {
    "scoring.input_scoring": _request(scoring=ScoringParams(input_scoring=True)),
    "scoring.sampled_logprobs": _request(scoring=ScoringParams(sampled_logprobs=True)),
    # top_logprobs without sampled_logprobs fails the shared request invariants
    # before any dialect sees it, so the leaf is only reachable together.
    "scoring.top_logprobs": _request(
        scoring=ScoringParams(sampled_logprobs=True, top_logprobs=2)
    ),
    "sampling.n": _request(sampling=SamplingParams(n=2)),
    "sampling.top_k": _request(sampling=SamplingParams(top_k=40)),
    "sampling.frequency_penalty": _request(
        sampling=SamplingParams(frequency_penalty=0.1)
    ),
    "sampling.presence_penalty": _request(
        sampling=SamplingParams(presence_penalty=0.1)
    ),
    "reasoning.budget_tokens": _request(reasoning=ReasoningParams(budget_tokens=128)),
    "tools.parallel": _request(tools=_FUNCTION_TOOLS),
    "structured_output.name": _request(structured_output=_STRUCTURED),
    "structured_output.strict": _request(structured_output=_STRUCTURED),
    "input.modality.image.detail": _request(input=_IMAGE_INPUT),
}

#: Requests refused for their value or shape, with the reason's wording.
_INVALID_REQUESTS = [
    pytest.param(_request(session=_OPAQUE), "opaque.*JSON", id="invalid-opaque-json"),
    pytest.param(
        _request(input=_chat(_say("developer", "d"), _say("user", "u"))),
        "developer message authority",
        id="developer-message",
    ),
    pytest.param(
        _request(
            input=_chat(
                _say("user", "go"),
                ChatMessage("assistant", (ToolCallPart("c1", "f", {"a": 1}),)),
                ChatMessage("tool", (ToolResultPart("c1", "ok"), TextPart("more"))),
                _say("user", "next"),
            )
        ),
        "exactly one ToolResultPart",
        id="tool-message-with-extra-content",
    ),
    pytest.param(
        _request(
            input=_chat(_say("system", "a"), _say("system", "b"), _say("user", "u"))
        ),
        "exactly one system message",
        id="two-system-messages",
    ),
    pytest.param(
        _request(
            input=_chat(
                ChatMessage("system", (TextPart("a"), TextPart("b"))), _say("user", "u")
            )
        ),
        "exactly one system message",
        id="split-system-message",
    ),
    pytest.param(
        _request(input=_chat(_say("user", "u"), _say("system", "s"))),
        "must precede conversation turns",
        id="late-system-message",
    ),
    pytest.param(
        _request(input=_chat(_say("system", "s"))),
        "at least one conversation turn",
        id="no-conversation-turn",
    ),
    pytest.param(
        _request(input=_chat(_say("user", "u"), _say("assistant", "a"))),
        "assistant-prefill",
        id="assistant-prefill",
    ),
    pytest.param(
        _request(input=_chat(ChatMessage("user", ()))),
        "no content to lower",
        id="empty-turn",
    ),
    pytest.param(
        _request(input=_chat(ChatMessage("user", (TextPart("u"),), name="alice"))),
        "no ChatMessage.name",
        id="named-message",
    ),
    pytest.param(
        Request(CompletionInput("complete me")),
        "requires ChatInput",
        id="completion-input",
    ),
    pytest.param(
        _request(sampling=SamplingParams(max_tokens=0)),
        "max_tokens must be positive",
        id="zero-max-tokens",
    ),
    pytest.param(
        _request(reasoning=ReasoningParams(effort="xhigh")),
        "unsupported Google thinking level",
        id="unknown-thinking-level",
    ),
    pytest.param(
        _request(reasoning=ReasoningParams(summary="detailed")),
        "only summary",
        id="unsupported-summary-mode",
    ),
    *(
        pytest.param(
            _request(dialect_options=_options(**{key: value})),
            match,
            id=f"option-{key}",
        )
        for key, value, match in (
            ("model", "x", "provider-neutral field"),
            ("input", [], "provider-neutral field"),
            ("stream", True, "provider-neutral field"),
            ("previous_interaction_id", "p", "provider-neutral field"),
            ("agent", "deep-research", "bound to a model"),
            ("agent_config", {}, "bound to a model"),
            ("background", True, "synchronous"),
            ("store", "yes", "must be a boolean"),
        )
    ),
    *(
        pytest.param(
            _request(dialect_options=_options(service_tier=tier)),
            "queued interaction",
            id=f"service-tier-{tier}",
        )
        for tier in ("flex", "deferred", 3)
    ),
    pytest.param(
        _request(dialect_options=_options(**{"": 1})),
        "must be non-empty",
        id="empty-option-key",
    ),
    pytest.param(
        _request(dialect_options=DialectOptions("google_interactions", {1: "x"})),  # ty: ignore[invalid-argument-type]
        "must be strings",
        id="non-string-option-key",
    ),
]


class TestCapabilities:
    def test_per_choice_logprobs_are_explicitly_absent(self) -> None:
        req = _request()
        response = Response(texts=("Hi",), choice_logprobs=(ChoiceLogprobs(tokens=()),))
        assert OUTPUT_CONTRACT.rules["choice_logprobs"].guarantee is Guarantee.NEVER
        with pytest.raises(OutputContractError, match="choice_logprobs.*absent"):
            OUTPUT_CONTRACT.validate(cast(Any, SimpleNamespace()), req, response)

    def test_decision_row_and_registry_are_complete(self) -> None:
        assert set(CAPABILITY_DECISIONS) == set(CAPABILITY_KEYS)
        supported = {
            key
            for key, decision in CAPABILITY_DECISIONS.items()
            if isinstance(decision, Supported)
        }
        assert supported == {
            "reasoning",
            "function_tools",
            "hosted_tools",
            "structured_output",
            "multimodal_input",
            "opaque_continuation",
        }

        spec = DIALECT_SPECS["google_interactions"]
        assert spec.implementation_status.value == "active"
        assert spec.request_seed_support.value == "supported"
        assert spec.connection_family == "async_http_json"
        assert spec.input_kinds == ("chat",)
        assert spec.input_modalities == (
            "image",
            "text",
            "tool_call",
            "tool_result",
        )

    def test_an_incomplete_decision_row_fails_at_import(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            google_interactions, "CAPABILITY_KEYS", (*CAPABILITY_KEYS, "extra")
        )
        with pytest.raises(AssertionError, match="capability row is incomplete"):
            google_interactions._decisions()

    def test_stateful_session_is_unavailable_without_prompt_retention(self) -> None:
        decision = CAPABILITY_DECISIONS["stateful_session"]
        assert isinstance(decision, Unsupported)
        assert "store=false" in decision.reason
        assert OUTPUT_CONTRACT.rules["session_id"].guarantee is Guarantee.NEVER

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            (ReasoningOptions(budget_tokens=128), "token-budget"),
            (ReasoningOptions(summary="detailed"), "'none' or 'auto'"),
            (ReasoningOptions(effort="xhigh"), "thinking level"),
        ],
    )
    def test_reasoning_config_is_provider_validated(
        self, options: ReasoningOptions, message: str
    ) -> None:
        decision = cast(Supported, CAPABILITY_DECISIONS["reasoning"])
        with pytest.raises(ValueError, match=message):
            decision.binding.validate_config(options)

    @pytest.mark.parametrize("effort", ["minimal", "low", "medium", "HIGH", None])
    @pytest.mark.parametrize("summary", ["none", "auto", None])
    def test_reasoning_config_accepts_the_supported_domain(
        self, effort: str | None, summary: str | None
    ) -> None:
        decision = cast(Supported, CAPABILITY_DECISIONS["reasoning"])
        decision.binding.validate_config(
            ReasoningOptions(effort=effort, summary=summary)
        )

    def test_output_contract_matches_the_declared_capabilities(self) -> None:
        rules = OUTPUT_CONTRACT.rules
        assert rules["session_id"].guarantee is Guarantee.NEVER
        assert rules["reasoning"].guarantee is Guarantee.PRESENT_OR_ERROR
        assert rules["structured_output"].guarantee is Guarantee.PRESENT_OR_ERROR
        assert rules["tool_calls"].guarantee is Guarantee.BEST_EFFORT
        assert rules["server_tool_uses"].guarantee is Guarantee.BEST_EFFORT
        assert rules["citations"].guarantee is Guarantee.BEST_EFFORT
        for channel in (
            "logprobs",
            "top_logprobs",
            "input_scoring",
            "grounding",
        ):
            assert rules[channel].guarantee is Guarantee.NEVER
        assert OUTPUT_CONTRACT.empty_outcome_finish_reasons == {"incomplete"}


class TestConstruction:
    def test_requires_an_http_json_connection(self) -> None:
        with pytest.raises(TypeError, match="AsyncHTTPJSONConnection"):
            GoogleInteractionsDialect(object(), MODEL)

    def test_requires_a_model_id(self) -> None:
        with pytest.raises(ValueError, match="requested_model_id"):
            GoogleInteractionsDialect(_connection(_no_io), "")

    def test_ambiguous_api_base_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Google api_base"):
            _dialect(base_url="https://example.test/v2")

    def test_capabilities_view_is_the_class_constant(self) -> None:
        assert _dialect().capabilities is GoogleInteractionsDialect.CAPABILITIES

    def test_legacy_capability_view_matches_supported_decisions(self) -> None:
        expected = frozenset(
            capability
            for capability_key, decision in CAPABILITY_DECISIONS.items()
            if isinstance(decision, Supported)
            for capability in _LEGACY_CAPABILITY_MAP.get(capability_key, ())
        ) | {Capability.Chat}
        assert _dialect().capabilities == expected


class TestRequestLowering:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("base_url", "expected_path"),
        [
            (ROOT, "/v1beta/interactions"),
            (f"{ROOT}v1beta/", "/v1beta/interactions"),
            (f"{ROOT}v1", "/v1/interactions"),
            ("https://gateway.example/google/v1/", "/google/v1/interactions"),
            ("https://gateway.example/v2/google/", "/v2/google/v1beta/interactions"),
        ],
    )
    async def test_final_url_header_and_wire_body(
        self, base_url: str, expected_path: str
    ) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["request"] = request
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_reply())

        req = _request(
            input=_chat(_say("system", "Be concise"), _say("user", "Hello")),
            sampling=SamplingParams(
                temperature=0.2,
                top_p=0.9,
                max_tokens=64,
                stop=("END",),
                seed=7,
            ),
        )
        result = await _run(handler, req, base_url=base_url, model=OLD_MODEL)

        request = cast(httpx.Request, seen["request"])
        assert request.method == "POST"
        assert request.url.path == expected_path
        assert request.headers["x-goog-api-key"] == "google-secret"
        assert seen["body"] == {
            "model": OLD_MODEL,
            "store": False,
            "input": [
                {"type": "user_input", "content": [{"type": "text", "text": "Hello"}]}
            ],
            "system_instruction": "Be concise",
            "generation_config": {
                "temperature": 0.2,
                "top_p": 0.9,
                "max_output_tokens": 64,
                "stop_sequences": ["END"],
                "seed": 7,
            },
        }
        assert seen["body"]["store"] is False
        assert "stream" not in seen["body"]
        assert result.request_params == {
            "envelope": "http",
            "method": "POST",
            "path": expected_path,
            "body": {
                "generation_config": seen["body"]["generation_config"],
                "store": False,
            },
        }
        assert "google-secret" not in json.dumps(result.request_params)

    @pytest.mark.anyio
    async def test_missing_credential_does_not_emit_auth_header(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "x-goog-api-key" not in request.headers
            return httpx.Response(200, json=_reply())

        await _run(handler, credential=None)

    @pytest.mark.anyio
    async def test_history_maps_roles_to_step_types(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_reply())

        req = _request(
            input=_chat(
                _say("user", "My number is 41."),
                _say("assistant", "OK, noted."),
                ChatMessage("user", (TextPart("What is"), TextPart("my number + 1?"))),
            )
        )
        await _run(handler, req)

        assert seen["body"]["input"] == [
            {
                "type": "user_input",
                "content": [{"type": "text", "text": "My number is 41."}],
            },
            {
                "type": "model_output",
                "content": [{"type": "text", "text": "OK, noted."}],
            },
            {
                "type": "user_input",
                "content": [
                    {"type": "text", "text": "What is"},
                    {"type": "text", "text": "my number + 1?"},
                ],
            },
        ]
        assert "system_instruction" not in seen["body"]

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("reasoning", "expected"),
        [
            (
                ReasoningParams(effort="LOW", summary="auto"),
                {"thinking_level": "low", "thinking_summaries": "auto"},
            ),
            (ReasoningParams(summary="none"), {"thinking_summaries": "none"}),
            (ReasoningParams(effort="minimal"), {"thinking_level": "minimal"}),
        ],
    )
    async def test_reasoning_controls_use_lowercase_wire_values(
        self, reasoning: ReasoningParams, expected: dict[str, str]
    ) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200, json=_reply(steps=[_thought(["thinking"]), _model_output("Hi")])
            )

        await _run(handler, _request(reasoning=reasoning))
        assert seen["body"]["generation_config"] == expected

    @pytest.mark.anyio
    async def test_previous_response_id_requires_stateless_history_replay(self) -> None:
        req = _request(session=SessionParams(previous_response_id="prev-1"))
        with pytest.raises(RequestAuditError, match="resend the full conversation"):
            _audit(req).raise_rejections()
        with pytest.raises(
            DialectError, match="unavailable capability.*stateful_session"
        ):
            await _run(_no_io, req)

    @pytest.mark.anyio
    async def test_dialect_options_reach_the_wire_and_the_record(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_reply(id="stored-1"))

        result = await _run(
            handler,
            _request(
                dialect_options=_options(
                    labels={"run": "x"}, store=False, service_tier="priority"
                )
            ),
        )

        assert seen["body"]["labels"] == {"run": "x"}
        assert seen["body"]["store"] is False
        assert seen["body"]["service_tier"] == "priority"
        assert result.request_params is not None
        recorded = cast(Mapping[str, Any], result.request_params["body"])
        assert recorded["labels"] == {"run": "x"}
        assert recorded["store"] is False
        assert result.session_id is None

    @pytest.mark.anyio
    async def test_unstored_request_never_reports_a_session_id(self) -> None:
        # Even a reply that carries an id: it is not a reusable chain handle.
        result = await _run(
            _json_handler(_reply(id="ghost")),
            _request(dialect_options=_options(store=False)),
        )
        assert result.session_id is None

    @pytest.mark.anyio
    async def test_empty_dialect_options_have_no_wire_effect(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_reply())

        await _run(handler, _request(dialect_options=_options()))
        assert set(seen["body"]) == {"model", "input", "store"}


class TestRequestRejection:
    def test_every_rejected_leaf_has_a_request_here(self) -> None:
        assert set(_REJECTED_REQUESTS) == set(google_interactions._REJECTED_PATHS)

    @pytest.mark.parametrize("path", sorted(_REJECTED_REQUESTS))
    def test_unlowerable_leaf_is_rejected_by_the_audit(self, path: str) -> None:
        req = _REJECTED_REQUESTS[path]
        assert path in active_request_leaves(req)

        audit = _audit(req)

        assert isinstance(audit.decisions[path], Rejected)
        with pytest.raises(RequestAuditError, match=path.replace(".", r"\.")):
            audit.raise_rejections()

    @pytest.mark.anyio
    @pytest.mark.parametrize("path", sorted(_REJECTED_REQUESTS))
    async def test_unlowerable_leaf_never_reaches_the_network(self, path: str) -> None:
        # Leaves owned by an unsupported capability fail the runtime-plan check
        # first; the rest fail the audit. Either way, before any I/O.
        with pytest.raises(DialectError):
            await _run(_no_io, _REJECTED_REQUESTS[path])

    @pytest.mark.anyio
    @pytest.mark.parametrize(("req", "message"), _INVALID_REQUESTS)
    async def test_invalid_request_is_rejected_before_io(
        self, req: Request, message: str
    ) -> None:
        with pytest.raises(DialectError, match=message):
            await _run(_no_io, req)

    def test_prepare_refuses_an_audit_with_rejections(self) -> None:
        req = _request(sampling=SamplingParams(n=2))
        with pytest.raises(RequestAuditError, match="sampling.n"):
            _dialect().prepare(req, _audit(req))

    def test_prepare_refuses_an_audit_for_a_different_request(self) -> None:
        other = _request(sampling=SamplingParams(seed=1))
        audit = RequestAudit(active_request_leaves(_request()))
        with pytest.raises(RequestAuditError, match="audit does not match"):
            _dialect().prepare(other, audit)

    def test_prepare_requires_chat_input(self) -> None:
        req = Request(CompletionInput("x"))
        audit = RequestAudit(active_request_leaves(req))
        with pytest.raises(DialectError, match="requires ChatInput"):
            _dialect().prepare(req, audit)

    @pytest.mark.parametrize(
        ("req", "message"),
        [
            pytest.param(
                _request(input=_chat(_say("developer", "d"), _say("user", "u"))),
                "developer message authority",
                id="chat-shape",
            ),
            pytest.param(
                _request(input=_chat(ChatMessage("user", (TextPart("u"),), name="a"))),
                "no ChatMessage.name",
                id="named-message",
            ),
            pytest.param(
                _request(
                    input=_chat(
                        _say("user", "go"),
                        ChatMessage("assistant", (ToolCallPart("c1", "f", {}),)),
                        ChatMessage(
                            "tool", (ToolResultPart("c1", "ok"), TextPart("more"))
                        ),
                        _say("user", "next"),
                    )
                ),
                "exactly one ToolResultPart",
                id="tool-message-with-extra-content",
            ),
            pytest.param(
                _request(
                    input=_chat(
                        _say("user", "go"),
                        ChatMessage("assistant", (ToolResultPart("c1", "ok"),)),
                        _say("user", "next"),
                    )
                ),
                "answers no earlier tool call",
                id="tool-result-without-its-call",
            ),
            pytest.param(
                _request(
                    dialect_options=DialectOptions("google_interactions", {1: "x"})  # ty: ignore[invalid-argument-type]
                ),
                "must be strings",
                id="non-string-option-key",
            ),
        ],
    )
    def test_prepare_refuses_what_lowering_cannot_represent_on_its_own(
        self, req: Request, message: str
    ) -> None:
        # A caller may skip ``validate_request``; lowering must not emit a
        # lossy body just because no earlier step recorded a rejection.
        audit = RequestAudit(active_request_leaves(req))
        with pytest.raises(DialectError, match=message):
            _dialect().prepare(req, audit)


class TestFinalWireEvidence:
    @pytest.mark.anyio
    async def test_input_evidence_catches_a_lowering_that_changes_the_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def corrupt(
            _chat: ChatInput, _opaque: str | None = None
        ) -> tuple[str | None, list[Any]]:
            return (
                None,
                [
                    {
                        "type": "user_input",
                        "content": [{"type": "text", "text": "tampered"}],
                    }
                ],
            )

        monkeypatch.setattr(google_interactions, "_input_to_wire", corrupt)
        with pytest.raises(RequestAuditError, match="text no longer matches"):
            await _run(_no_io)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("req", "mutate", "message"),
        [
            pytest.param(
                _request(sampling=SamplingParams(max_tokens=64)),
                _edit("generation_config", "max_output_tokens", value=999),
                r"changed for 'sampling\.max_tokens'",
                id="sampling-value-changed",
            ),
            pytest.param(
                _request(sampling=SamplingParams(seed=7)),
                _edit("generation_config", "seed"),
                "is missing",
                id="sampling-value-dropped",
            ),
            pytest.param(
                _request(sampling=SamplingParams(stop=("A",))),
                _edit("generation_config", "stop_sequences", value=["B"]),
                r"sampling\.stop",
                id="stop-sequences-changed",
            ),
            pytest.param(
                _request(reasoning=ReasoningParams(effort="low")),
                _edit("generation_config", "thinking_level", value="high"),
                "thinking level changed",
                id="thinking-level-changed",
            ),
            pytest.param(
                _request(reasoning=ReasoningParams(summary="auto")),
                _edit("generation_config", "thinking_summaries", value="none"),
                r"reasoning\.summary",
                id="thinking-summaries-changed",
            ),
            pytest.param(
                _request(),
                _edit("previous_interaction_id", value="other"),
                "previous_interaction_id",
                id="chain-id-injected",
            ),
            pytest.param(
                _request(dialect_options=_options(labels={"a": "b"})),
                _edit("labels", value={"a": "c"}),
                "dialect option 'labels' changed",
                id="dialect-option-changed",
            ),
            pytest.param(
                _request(),
                _edit("model", value="other-model"),
                "model does not match the bound model",
                id="model-changed",
            ),
            pytest.param(
                _request(),
                _edit("stream", value=False),
                "stream must be true or absent",
                id="stream-injected",
            ),
            pytest.param(
                _request(input=_chat(_say("system", "s"), _say("user", "u"))),
                _edit("system_instruction", value="other"),
                "system_instruction no longer matches",
                id="system-changed",
            ),
            pytest.param(
                _request(),
                _edit("system_instruction", value="smuggled"),
                "system_instruction no longer matches",
                id="system-injected",
            ),
            pytest.param(
                _request(),
                _edit("input", value=[]),
                "input steps no longer match",
                id="input-dropped",
            ),
        ],
    )
    async def test_audit_catches_a_prepared_body_that_drifts_from_the_request(
        self,
        monkeypatch: pytest.MonkeyPatch,
        req: Request,
        mutate: Callable[[dict[str, Any]], None],
        message: str,
    ) -> None:
        _tamper_prepared(monkeypatch, mutate)
        with pytest.raises(RequestAuditError, match=message):
            await _run(_no_io, req)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("final_body", "message"),
        [
            pytest.param([], "must be an object", id="non-object-body"),
            pytest.param(
                {"model": "other", "input": []},
                "final Interactions request: model does not match",
                id="model-drifted-in-serialization",
            ),
        ],
    )
    async def test_execute_checks_the_serialized_body_not_only_the_prepared_one(
        self, monkeypatch: pytest.MonkeyPatch, final_body: object, message: str
    ) -> None:
        real = google_interactions.strict_json_loads

        def serialized(payload: str, path: str) -> object:
            value = real(payload, path)
            return final_body if path == "final Interactions request body" else value

        monkeypatch.setattr(google_interactions, "strict_json_loads", serialized)
        with pytest.raises(DialectError, match=message):
            await _run(_no_io)

    @pytest.mark.anyio
    async def test_audit_catches_a_prepared_operation_for_another_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = google_interactions.PreparedRequest
        monkeypatch.setattr(
            google_interactions,
            "PreparedRequest",
            lambda operation, body, context=None: real(
                "v1/interactions", body, context
            ),
        )
        with pytest.raises(
            RequestAuditError, match="does not match 'v1beta/interactions'"
        ):
            await _run(_no_io)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("prepared", "message"),
        [
            pytest.param(
                PreparedRequest("v1beta/interactions", {"model": MODEL, "input": []}),
                "invalid prepared context",
                id="foreign-context",
            ),
            pytest.param(
                PreparedRequest(
                    "v1/interactions",
                    {"model": MODEL, "input": []},
                    google_interactions._InteractionsContext(_request()),
                ),
                "wrong prepared operation",
                id="other-version",
            ),
            pytest.param(
                PreparedRequest(
                    "v1beta/interactions",
                    {"model": "other", "input": []},
                    google_interactions._InteractionsContext(_request()),
                ),
                "model does not match",
                id="other-model",
            ),
            pytest.param(
                PreparedRequest(
                    "v1beta/interactions",
                    {"model": MODEL, "input": [], "stream": "yes"},
                    google_interactions._InteractionsContext(_request()),
                ),
                "stream must be true or absent",
                id="non-boolean-stream",
            ),
            pytest.param(
                PreparedRequest(
                    "v1beta/interactions",
                    {"model": MODEL, "input": [], "store": "yes"},
                    google_interactions._InteractionsContext(_request()),
                ),
                "store must be false",
                id="non-boolean-store",
            ),
        ],
    )
    async def test_execute_refuses_a_prepared_request_it_would_not_have_built(
        self, prepared: PreparedRequest, message: str
    ) -> None:
        with pytest.raises(DialectError, match=message):
            await _dialect().execute(prepared)


class TestInputVerifier:
    _GOOD: dict[str, Any] = {
        "input": [{"type": "user_input", "content": [{"type": "text", "text": "U"}]}],
        "system_instruction": "S",
    }

    def _verifier(self) -> google_interactions._InputVerifier:
        return google_interactions._InputVerifier.from_request(
            _chat(_say("system", "S"), _say("user", "U"))
        )

    def test_accepts_the_matching_body(self) -> None:
        assert self._verifier().verify(self._GOOD) is None

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"input": "U"}, "not a step array"),
            ({"input": []}, "no longer match"),
            (
                {"input": [{"type": "user_input", "content": [], "id": "x"}]},
                "unexpected shape",
            ),
            ({"input": [{"type": 5, "content": []}]}, "unexpected shape"),
            (
                {"input": [{"type": "user_input", "content": "U"}]},
                "content no longer matches",
            ),
            (
                {"input": [{"type": "user_input", "content": ["U"]}]},
                "is not a content block",
            ),
            (
                {
                    "input": [
                        {
                            "type": "user_input",
                            "content": [
                                {"type": "text", "text": "U", "annotations": []}
                            ],
                        }
                    ]
                },
                "has unexpected field",
            ),
            (
                {
                    "input": [
                        {
                            "type": "user_input",
                            "content": [{"type": "image", "text": "U"}],
                        }
                    ]
                },
                "changed type",
            ),
            (
                {
                    "input": [
                        {
                            "type": "user_input",
                            "content": [{"type": "text", "text": 5}],
                        }
                    ]
                },
                "text no longer matches",
            ),
            (
                {
                    "input": [
                        {
                            "type": "user_input",
                            "content": [{"type": "text", "text": "other"}],
                        }
                    ]
                },
                "no longer match",
            ),
            (
                {
                    "input": [
                        {
                            "type": "model_output",
                            "content": [{"type": "text", "text": "U"}],
                        }
                    ]
                },
                "unexpected shape",
            ),
            ({"input": []}, "no longer match"),
            ({"system_instruction": "other"}, "system_instruction no longer"),
            ({"system_instruction": None}, "system_instruction no longer"),
        ],
    )
    def test_rejects_each_divergence(
        self, changes: dict[str, Any], message: str
    ) -> None:
        assert message in (self._verifier().verify({**self._GOOD, **changes}) or "")


class TestRouteVerifier:
    ROUTE = google_interactions._RouteVerifier(MODEL, "v1beta")
    ENDPOINT = httpx.URL(ROOT)
    URL = f"{ROOT}v1beta/interactions"

    def test_body_must_name_the_bound_model_and_omit_stream(self) -> None:
        assert self.ROUTE.verify_body({"model": MODEL, "store": False}) is None
        assert "model" in (self.ROUTE.verify_body({"model": "other"}) or "")
        assert "model" in (self.ROUTE.verify_body({}) or "")
        # Even ``stream: false``: a synchronous request must not carry the key.
        assert "stream" in (
            self.ROUTE.verify_body({"model": MODEL, "stream": False}) or ""
        )

    def test_prepared_operation_must_be_the_bound_version(self) -> None:
        good = PreparedRequest("v1beta/interactions", {"model": MODEL, "store": False})
        assert self.ROUTE.verify_prepared(good) is None
        other = PreparedRequest("v1/interactions", {"model": MODEL})
        assert "does not match" in (self.ROUTE.verify_prepared(other) or "")
        wrong_body = PreparedRequest("v1beta/interactions", {"model": "other"})
        assert "model" in (self.ROUTE.verify_prepared(wrong_body) or "")

    def test_http_target_must_be_the_post_to_the_bound_route(self) -> None:
        self.ROUTE.verify_http(httpx.Request("POST", self.URL), self.ENDPOINT)
        for request in (
            httpx.Request("GET", self.URL),
            httpx.Request("POST", f"{ROOT}v1/interactions"),
            httpx.Request("POST", f"{self.URL}?alt=sse"),
        ):
            with pytest.raises(DialectError, match="does not match the bound"):
                self.ROUTE.verify_http(request, self.ENDPOINT)


class TestOptionVerifiers:
    def test_thinking_level_is_compared_case_insensitively(self) -> None:
        verifier = google_interactions._ThinkingLevelVerifier("LOW")
        assert verifier.verify({"generation_config": {"thinking_level": "low"}}) is None
        assert verifier.verify({"generation_config": {"thinking_level": "high"}})
        assert verifier.verify({}) == "thinking level changed"

    def test_root_option_must_survive_unchanged(self) -> None:
        verifier = google_interactions._RootOptionVerifier("labels", {"a": "b"})
        assert verifier.verify({"labels": {"a": "b"}}) is None
        assert verifier.verify({"labels": {"a": "c"}}) is not None
        assert verifier.verify({}) is not None

    @pytest.mark.parametrize("expected", [False, True])
    def test_boolean_root_options_do_not_accept_equivalent_integers(
        self, expected: bool
    ) -> None:
        verifier = google_interactions._RootOptionVerifier("store", expected)
        assert verifier.verify({"store": expected}) is None
        assert verifier.verify({"store": int(expected)}) is not None
        assert verifier.verify({"store": not expected}) is not None

    def test_root_option_is_detached_from_the_caller_value(self) -> None:
        value: dict[str, Any] = {"a": [1]}
        verifier = google_interactions._RootOptionVerifier("labels", value)
        value["a"].append(2)
        assert verifier.verify({"labels": {"a": [1]}}) is None


class TestResponseLifting:
    @pytest.mark.anyio
    async def test_completed_text_recording(self) -> None:
        result = await _run(_json_handler(_fixture("completed_text")))

        assert result.texts == ("pong",)
        assert result.finish_reasons == ("completed",)
        assert result.reasoning is not None
        assert result.reasoning[0] is not None
        assert result.reasoning[0].text is None
        assert result.reasoning[0].opaque_roundtrip is not None
        assert result.session_id is None
        assert result.response_model == MODEL
        assert result.usage == UsageStats(
            input_tokens=8,
            output_tokens=103,
            total_tokens=111,
            reasoning_tokens=103,
            cached_tokens=0,
        )
        assert not is_truncated_finish_reason("completed", None)

    @pytest.mark.anyio
    async def test_incomplete_recording_has_no_steps_and_counts_as_truncated(
        self,
    ) -> None:
        result = await _run(_json_handler(_fixture("incomplete_no_steps")))

        assert result.texts == ("",)
        assert result.finish_reasons == ("incomplete",)
        assert result.usage == UsageStats(
            input_tokens=14,
            output_tokens=5,
            total_tokens=19,
            reasoning_tokens=5,
            cached_tokens=0,
        )
        assert is_truncated_finish_reason(result.finish_reasons[0], None)

    @pytest.mark.anyio
    async def test_incomplete_reply_may_leave_requested_channels_empty(self) -> None:
        result = await _run(
            _json_handler(_fixture("incomplete_no_steps")),
            _request(
                reasoning=ReasoningParams(summary="auto"),
            ),
        )
        assert result.reasoning == (None,)
        assert result.session_id is None

    @pytest.mark.anyio
    async def test_stop_sequence_recording_is_completed_with_empty_text(self) -> None:
        result = await _run(_json_handler(_fixture("stop_sequence_no_steps")))
        assert result.texts == ("",)
        assert result.finish_reasons == ("completed",)

    @pytest.mark.anyio
    async def test_response_id_is_not_exposed_as_a_reusable_session(self) -> None:
        recording = _fixture("stored_with_id")
        result = await _run(_json_handler(recording))

        assert recording["id"]
        assert result.session_id is None
        assert result.response_model == "gemini-3.6-flash"
        assert result.texts == (
            "Got it! I've made a note that your number is 41. "
            "What would you like to do next?",
        )

    @pytest.mark.anyio
    async def test_thought_summary_recording(self) -> None:
        recording = _fixture("thought_summary")
        summary = recording["steps"][0]["summary"][0]["text"]
        result = await _run(
            _json_handler(recording),
            _request(reasoning=ReasoningParams(summary="auto")),
        )

        assert result.texts == ("391",)
        assert result.reasoning is not None
        assert result.reasoning[0] is not None
        assert result.reasoning[0].text == summary
        assert summary.startswith("**My Calculation of 17 x 23**")

    @pytest.mark.anyio
    async def test_a_summary_the_provider_sent_is_not_dropped(self) -> None:
        result = await _run(_json_handler(_fixture("thought_summary")))
        assert result.reasoning is not None
        assert result.reasoning[0] is not None
        assert result.reasoning[0].text

    @pytest.mark.anyio
    async def test_requested_summary_may_be_omitted(self) -> None:
        result = await _run(
            _json_handler(_fixture("completed_text")),
            _request(reasoning=ReasoningParams(summary="auto")),
        )
        assert result.reasoning is not None
        assert result.reasoning[0] is not None
        assert result.reasoning[0].text is None
        assert result.reasoning[0].opaque_roundtrip is not None

    @pytest.mark.anyio
    async def test_summary_none_preserves_opaque_without_visible_reasoning(
        self,
    ) -> None:
        result = await _run(
            _json_handler(_fixture("completed_text")),
            _request(reasoning=ReasoningParams(summary="none")),
        )
        assert result.reasoning is not None
        assert result.reasoning[0] is not None
        assert result.reasoning[0].text is None
        assert result.reasoning[0].opaque_roundtrip is not None

    @pytest.mark.anyio
    async def test_output_text_concatenates_in_step_and_content_order(self) -> None:
        reply = _reply(steps=[_thought(), _model_output("a", "b"), _model_output("c")])
        result = await _run(_json_handler(reply))
        assert result.texts == ("abc",)

    @pytest.mark.anyio
    async def test_empty_annotation_list_is_accepted(self) -> None:
        step = _model_output("x")
        step["content"][0]["annotations"] = []
        result = await _run(_json_handler(_reply(steps=[step])))
        assert result.texts == ("x",)

    @pytest.mark.anyio
    async def test_stateless_reply_does_not_require_an_interaction_id(self) -> None:
        result = await _run(_json_handler(_reply(id=_DROP)))
        assert result.texts == ("Hi",)
        assert result.session_id is None

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            pytest.param(
                {"status": _DROP}, "status must be a non-empty", id="no-status"
            ),
            pytest.param(
                {"status": ""}, "status must be a non-empty", id="empty-status"
            ),
            pytest.param({"object": "response"}, "is not 'interaction'", id="object"),
            pytest.param({"status": "failed"}, "reply failed$", id="failed"),
            pytest.param(
                {"status": "failed", "errors": [{"code": "x", "message": "boom"}]},
                "failed: boom",
                id="failed-with-message",
            ),
            pytest.param(
                {"status": "failed", "errors": [{"code": "x"}]},
                "reply failed$",
                id="failed-without-message",
            ),
            pytest.param(
                {"status": "requires_action"},
                "requires an action",
                id="requires-action",
            ),
            pytest.param({"status": "in_progress"}, "not terminal", id="in-progress"),
            pytest.param({"status": "queued"}, "not terminal", id="queued"),
            pytest.param({"status": "cancelled"}, "was cancelled", id="cancelled"),
            pytest.param({"status": "mystery"}, "unknown status", id="unknown-status"),
            pytest.param({"steps": {}}, "steps must be an array", id="steps-object"),
            pytest.param(
                {"steps": ["x"]}, r"steps\[0\] must be an object", id="step-str"
            ),
            pytest.param(
                {"steps": [{"type": "function_call", "name": "f"}]},
                "must be a non-empty string",
                id="function-call-without-id",
            ),
            pytest.param(
                {"steps": [{"type": "function_result", "call_id": "c1", "result": 1}]},
                "unsupported type 'function_result'",
                id="function-result-in-a-reply",
            ),
            pytest.param(
                {"steps": [{"type": "google_search_call", "id": "s"}]},
                "unrequested google_search call",
                id="hosted-call-unrequested",
            ),
            pytest.param(
                {"steps": [{"type": "mcp_server_tool_call", "id": "s"}]},
                "unsupported type 'mcp_server_tool_call'",
                id="unsupported-step",
            ),
            pytest.param(
                {"steps": [{**_model_output("x"), "id": "s"}]},
                r"unsupported field\(s\)",
                id="model-output-extra",
            ),
            pytest.param(
                {"steps": [{"type": "model_output", "content": "x"}]},
                "content must be an array",
                id="content-str",
            ),
            pytest.param(
                {
                    "steps": [
                        {
                            "type": "model_output",
                            "content": [{"type": "image", "data": "aGk="}],
                        }
                    ]
                },
                "unsupported type 'image'",
                id="image-output",
            ),
            pytest.param(
                {
                    "steps": [
                        {
                            "type": "model_output",
                            "content": [{"type": "text", "text": "x", "bogus": 1}],
                        }
                    ]
                },
                r"unsupported field\(s\)",
                id="content-extra",
            ),
            pytest.param(
                {
                    "steps": [
                        {
                            "type": "model_output",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "x",
                                    "annotations": [{"type": "file_citation"}],
                                }
                            ],
                        }
                    ]
                },
                "citation type 'file_citation'",
                id="annotation-without-url",
            ),
            pytest.param(
                {
                    "steps": [
                        {
                            "type": "model_output",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "x",
                                    "annotations": [
                                        {
                                            "type": "url_citation",
                                            "url": "u",
                                            "start_index": -1,
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                },
                "start_index must be a non-negative integer",
                id="annotation-negative-span",
            ),
            pytest.param(
                {
                    "steps": [
                        {
                            "type": "model_output",
                            "content": [{"type": "text", "text": 5}],
                        }
                    ]
                },
                "text must be a string",
                id="text-int",
            ),
            pytest.param(
                {"steps": [{**_thought(), "bogus": 1}]},
                r"unsupported field\(s\)",
                id="thought-extra",
            ),
            pytest.param(
                {"steps": [{"type": "thought", "signature": 5}]},
                "signature must be a string",
                id="signature-int",
            ),
            pytest.param(
                {"steps": [{"type": "thought", "summary": "x"}]},
                "summary must be an array",
                id="summary-str",
            ),
            pytest.param(
                {
                    "steps": [
                        {"type": "thought", "summary": [{"type": "image", "text": "x"}]}
                    ]
                },
                "not a plain text summary",
                id="summary-image",
            ),
            pytest.param(
                {
                    "steps": [
                        {
                            "type": "thought",
                            "summary": [{"type": "text", "text": "x", "extra": 1}],
                        }
                    ]
                },
                "not a plain text summary",
                id="summary-extra",
            ),
            pytest.param(
                {
                    "steps": [
                        {"type": "thought", "summary": [{"type": "text", "text": 5}]}
                    ]
                },
                "text must be a string",
                id="summary-text-int",
            ),
            pytest.param({"id": ""}, "id must be a non-empty string", id="empty-id"),
            pytest.param({"id": 5}, "id must be a non-empty string", id="int-id"),
            pytest.param({"model": ""}, "model must be a non-empty", id="empty-model"),
            pytest.param({"usage": []}, "usage must be an object", id="usage-list"),
            pytest.param(
                {"usage": {"total_input_tokens": -1}},
                "total_input_tokens must be a non-negative",
                id="usage-negative",
            ),
            pytest.param(
                {"usage": {"total_output_tokens": True}},
                "total_output_tokens must be a non-negative",
                id="usage-bool",
            ),
            pytest.param(
                {"usage": {"total_thought_tokens": 1.5}},
                "total_thought_tokens must be a non-negative",
                id="usage-float",
            ),
        ],
    )
    async def test_malformed_or_non_terminal_reply_fails_loudly(
        self, changes: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(OutputContractError, match=message):
            await _run(_json_handler(_reply(**changes)))

    @pytest.mark.anyio
    async def test_reply_must_be_an_object(self) -> None:
        with pytest.raises(OutputContractError, match="response must be an object"):
            await _run(_json_handler([_reply()]))

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "body",
        [
            b'{"status":"completed","status":"completed"}',
            b'{"status":"completed","usage":{"total_input_tokens":NaN}}',
            b"not json",
            b"",
        ],
    )
    async def test_reply_must_be_strict_json(self, body: bytes) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body)

        with pytest.raises(OutputContractError, match="not strict JSON"):
            await _run(handler)

    @pytest.mark.anyio
    @pytest.mark.parametrize("status", [400, 404, 429, 503])
    async def test_http_errors_surface_as_status_errors(self, status: int) -> None:
        error = {"error": {"message": "nope", "code": "invalid_request"}}
        with pytest.raises(httpx.HTTPStatusError) as raised:
            await _run(_json_handler(error, status))
        assert raised.value.response.status_code == status


class TestUsage:
    def test_absent_usage_is_none(self) -> None:
        assert google_interactions._usage(None) is None

    def test_missing_counts_read_as_zero_and_breakdowns_stay_unreported(self) -> None:
        assert google_interactions._usage({}) == UsageStats()

    @pytest.mark.parametrize(
        "name",
        [
            "total_input_tokens",
            "total_output_tokens",
            "total_tool_use_tokens",
            "total_thought_tokens",
            "total_cached_tokens",
            "total_tokens",
        ],
    )
    def test_a_provided_count_cannot_be_null(self, name: str) -> None:
        with pytest.raises(OutputContractError, match=rf"usage\.{name}"):
            google_interactions._usage({name: None})

    def test_thoughts_fold_into_output_so_reasoning_is_a_breakdown(self) -> None:
        usage = google_interactions._usage(
            {
                "total_input_tokens": 10,
                "total_output_tokens": 2,
                "total_thought_tokens": 3,
                "total_cached_tokens": 4,
                "total_tokens": 15,
            }
        )
        assert usage == UsageStats(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            reasoning_tokens=3,
            cached_tokens=4,
        )

    def test_a_total_that_does_not_decompose_is_recorded_not_rejected(self) -> None:
        usage = google_interactions._usage(
            {
                "total_input_tokens": 10,
                "total_output_tokens": 2,
                "total_thought_tokens": 3,
                "total_tokens": 99,
            }
        )
        assert usage is not None
        assert usage.total_tokens == 15
        assert usage.reported_total_tokens == 99

    def test_tool_use_tokens_fold_into_input_as_in_google_genai(self) -> None:
        # Assumption carried over from generateContent, pinned here so it cannot
        # change silently: no recorded reply has non-zero tool-use tokens yet.
        usage = google_interactions._usage(
            {"total_input_tokens": 10, "total_tool_use_tokens": 4}
        )
        assert usage is not None
        assert usage.input_tokens == 14


def _sse(*events: object) -> bytes:
    lines: list[str] = []
    for event in events:
        if isinstance(event, str):
            lines.extend((event, ""))
            continue
        lines.extend(
            (
                f"event: {cast(Mapping[str, Any], event)['event_type']}",
                "data: " + json.dumps(event, separators=(",", ":")),
                "",
            )
        )
    return "\n".join(lines).encode()


def _recorded_stream(name: str) -> bytes:
    return (FIXTURES / f"{name}.sse").read_bytes()


def _stream(events: bytes, status: int = 200):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=events)

    return handler


def _start(index: int, step: dict[str, Any]) -> dict[str, Any]:
    return {"event_type": "step.start", "index": index, "step": step}


def _delta(index: int, delta: dict[str, Any]) -> dict[str, Any]:
    return {"event_type": "step.delta", "index": index, "delta": delta}


def _stop(index: int) -> dict[str, Any]:
    return {"event_type": "step.stop", "index": index}


def _created() -> dict[str, Any]:
    return {
        "event_type": "interaction.created",
        "interaction": {
            "object": "interaction",
            "model": MODEL,
            "status": "in_progress",
        },
    }


def _completed(**changes: object) -> dict[str, Any]:
    interaction: dict[str, Any] = {
        "object": "interaction",
        "model": MODEL,
        "status": "completed",
        "usage": {"total_input_tokens": 3, "total_output_tokens": 1},
    }
    interaction.update(changes)
    return {"event_type": "interaction.completed", "interaction": interaction}


_DONE = "data: [DONE]"

_FUNCTION_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


def _function_request(**changes: object) -> Request:
    values: dict[str, object] = {"tools": ToolParams(functions=(_FUNCTION_TOOL,))}
    values.update(changes)
    return _request(**values)


class TestFunctionTools:
    """Function tools: declared flat, called back as `function_call` steps."""

    @pytest.mark.anyio
    async def test_declaration_is_flat_with_the_ir_json_schema(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_reply())

        await _run(handler, _function_request())

        assert seen["body"]["tools"] == [_FUNCTION_TOOL]

    @pytest.mark.anyio
    async def test_openai_shaped_declaration_is_normalized(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_reply())

        await _run(
            handler,
            _request(
                tools=ToolParams(
                    functions=(
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get the current weather.",
                            },
                        },
                    )
                )
            ),
        )

        assert seen["body"]["tools"] == [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the current weather.",
            }
        ]

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("choice", "wire"),
        [
            ("auto", "auto"),
            ("required", "any"),
            ("none", "none"),
            (
                {"type": "function", "name": "get_weather"},
                {"allowed_tools": {"mode": "any", "tools": ["get_weather"]}},
            ),
        ],
    )
    async def test_tool_choice_lowers_into_generation_config(
        self, choice: object, wire: object
    ) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_reply())

        await _run(
            handler,
            _request(
                tools=ToolParams(functions=(_FUNCTION_TOOL,), choice=cast(Any, choice))
            ),
        )

        assert seen["body"]["generation_config"]["tool_choice"] == wire

    @pytest.mark.anyio
    async def test_function_call_lifts_to_tool_calls(self) -> None:
        reply = _reply(
            status="requires_action",
            steps=[
                _thought(),
                {
                    "type": "function_call",
                    "id": "call_1",
                    "name": "get_weather",
                    "arguments": {"city": "Paris"},
                },
            ],
        )
        result = await _run(_json_handler(reply), _function_request())

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].call_id == "call_1"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"city": "Paris"}
        assert result.finish_reasons == ("requires_action",)

    @pytest.mark.anyio
    async def test_replayed_call_and_result_round_trip(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_reply())

        req = Request(
            input=_chat(
                _say("user", "What is the weather in Paris?"),
                ChatMessage(
                    "assistant",
                    (ToolCallPart("call_1", "get_weather", {"city": "Paris"}),),
                ),
                ChatMessage("tool", (ToolResultPart("call_1", {"temp_c": 18}),)),
                _say("user", "And tomorrow?"),
            ),
            tools=ToolParams(functions=(_FUNCTION_TOOL,)),
        )
        await _run(handler, req)

        assert seen["body"]["input"] == [
            {
                "type": "user_input",
                "content": [{"type": "text", "text": "What is the weather in Paris?"}],
            },
            # A call is a step of its own, not a content block of model_output.
            {
                "type": "function_call",
                "id": "call_1",
                "name": "get_weather",
                "arguments": {"city": "Paris"},
            },
            {
                "type": "function_result",
                "call_id": "call_1",
                "result": {"temp_c": 18},
                "name": "get_weather",
            },
            {
                "type": "user_input",
                "content": [{"type": "text", "text": "And tomorrow?"}],
            },
        ]

    @pytest.mark.anyio
    async def test_assistant_text_around_a_call_keeps_its_order(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_reply())

        req = Request(
            input=_chat(
                _say("user", "go"),
                ChatMessage(
                    "assistant",
                    (
                        TextPart("Checking."),
                        ToolCallPart("a", "get_weather", {"city": "Paris"}),
                        ToolCallPart("b", "get_weather", {"city": "Rome"}),
                        TextPart("Done."),
                    ),
                ),
                ChatMessage("tool", (ToolResultPart("a", "18C"),)),
                ChatMessage("tool", (ToolResultPart("b", "21C"),)),
            ),
            tools=ToolParams(functions=(_FUNCTION_TOOL,)),
        )
        await _run(handler, req)

        assert [
            (step["type"], step.get("id") or step.get("call_id"))
            for step in seen["body"]["input"]
        ] == [
            ("user_input", None),
            ("model_output", None),
            ("function_call", "a"),
            ("function_call", "b"),
            ("model_output", None),
            ("function_result", "a"),
            ("function_result", "b"),
        ]
        assert seen["body"]["input"][1]["content"] == [
            {"type": "text", "text": "Checking."}
        ]
        assert seen["body"]["input"][4]["content"] == [
            {"type": "text", "text": "Done."}
        ]

    @pytest.mark.anyio
    async def test_a_tool_result_may_end_the_input(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_reply())

        stateless = Request(
            input=_chat(
                _say("user", "Weather in Paris?"),
                ChatMessage("assistant", (ToolCallPart("c1", "get_weather", {}),)),
                ChatMessage("tool", (ToolResultPart("c1", {"temp_c": 18}),)),
            ),
            tools=ToolParams(functions=(_FUNCTION_TOOL,)),
        )
        await _run(handler, stateless)
        assert seen["body"]["input"][-1] == {
            "type": "function_result",
            "call_id": "c1",
            "result": {"temp_c": 18},
            "name": "get_weather",
        }

        orphan = Request(
            input=_chat(ChatMessage("tool", (ToolResultPart("c1", "18C"),))),
            tools=ToolParams(functions=(_FUNCTION_TOOL,)),
        )
        with pytest.raises(RequestAuditError, match="earlier tool call.*replay"):
            await _run(_no_io, orphan)

    @pytest.mark.parametrize(
        ("result", "ok"),
        [
            ("text", True),
            ({"temp_c": 18}, True),
            ([{"type": "text", "text": "18C"}], True),
            ([{"type": "image", "data": "aGk=", "mime_type": "image/png"}], True),
            ([{"type": "image", "uri": "https://example.com/a.png"}], True),
            ([], True),
            (7, False),
            (1.5, False),
            (False, False),
            (None, False),
            ([1, 2], False),
            (["x"], False),
            ([{"type": "audio", "data": "x"}], False),
            ([{"type": "text"}], False),
            ([{"type": "text", "text": "x", "junk": 1}], False),
            (
                [{"type": "image", "data": "x", "uri": "y", "mime_type": "image/png"}],
                False,
            ),
            ([{"type": "image", "mime_type": "image/png"}], False),
            ([{"type": "image", "data": "x", "mime_type": "application/pdf"}], False),
        ],
    )
    def test_tool_result_values_the_wire_can_carry(
        self, result: object, ok: bool
    ) -> None:
        error = google_interactions._tool_result_error(cast(Any, result))
        assert (error is None) is ok
        if not ok:
            req = Request(
                input=_chat(
                    _say("user", "go"),
                    ChatMessage("assistant", (ToolCallPart("c1", "f", {}),)),
                    ChatMessage("tool", (ToolResultPart("c1", cast(Any, result)),)),
                ),
                tools=ToolParams(functions=(_FUNCTION_TOOL,)),
            )
            audit = _audit(req)
            assert isinstance(audit.decisions["input.chat"], Rejected)

    @pytest.mark.anyio
    async def test_parallel_calls_arrive_on_both_transports(self) -> None:
        two = [
            {
                "type": "function_call",
                "id": "a",
                "name": "get_weather",
                "arguments": {"city": "Paris"},
            },
            {
                "type": "function_call",
                "id": "b",
                "name": "get_weather",
                "arguments": {"city": "Rome"},
            },
        ]
        plain = await _run(
            _json_handler(_reply(status="requires_action", steps=two)),
            _function_request(),
        )
        events: list[object] = [_created()]
        for index, (call_id, city) in enumerate((("a", "Paris"), ("b", "Rome"))):
            events += [
                _start(
                    index,
                    {
                        "type": "function_call",
                        "id": call_id,
                        "name": "get_weather",
                        "arguments": {},
                    },
                ),
                _delta(
                    index,
                    {
                        "type": "arguments_delta",
                        "arguments": json.dumps({"city": city}),
                    },
                ),
                _stop(index),
            ]
        events += [_completed(status="requires_action"), _DONE]
        streamed = await _run(
            _stream(_sse(*events)),
            _function_request(scheduling=SchedulingParams(stream=True)),
        )

        for result in (plain, streamed):
            assert result.tool_calls is not None
            assert [
                (c.call_id, cast(Mapping[str, Any], c.arguments)["city"])
                for c in result.tool_calls
            ] == [
                ("a", "Paris"),
                ("b", "Rome"),
            ]

    @pytest.mark.anyio
    async def test_a_repeated_call_id_is_an_error(self) -> None:
        call = {
            "type": "function_call",
            "id": "a",
            "name": "get_weather",
            "arguments": {"city": "Paris"},
        }
        with pytest.raises(OutputContractError, match="id 'a' twice"):
            await _run(
                _json_handler(_reply(status="requires_action", steps=[call, call])),
                _function_request(),
            )

    @pytest.mark.anyio
    async def test_error_result_sets_is_error(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_reply())

        req = Request(
            input=_chat(
                _say("user", "go"),
                ChatMessage("assistant", (ToolCallPart("c1", "get_weather", {}),)),
                ChatMessage("tool", (ToolResultPart("c1", "no data", is_error=True),)),
                _say("user", "then?"),
            ),
            tools=ToolParams(functions=(_FUNCTION_TOOL,)),
        )
        await _run(handler, req)

        assert seen["body"]["input"][2] == {
            "type": "function_result",
            "call_id": "c1",
            "result": "no data",
            "name": "get_weather",
            "is_error": True,
        }

    @pytest.mark.anyio
    async def test_call_outside_the_declared_set_fails(self) -> None:
        reply = _reply(
            status="requires_action",
            steps=[
                {"type": "function_call", "id": "c1", "name": "other", "arguments": {}}
            ],
        )
        with pytest.raises(OutputContractError, match="outside the request's allowed"):
            await _run(_json_handler(reply), _function_request())

    @pytest.mark.anyio
    async def test_choice_none_forbids_any_call(self) -> None:
        reply = _reply(
            status="requires_action",
            steps=[
                {
                    "type": "function_call",
                    "id": "c1",
                    "name": "get_weather",
                    "arguments": {},
                }
            ],
        )
        with pytest.raises(OutputContractError, match="outside the request's allowed"):
            await _run(
                _json_handler(reply),
                _request(tools=ToolParams(functions=(_FUNCTION_TOOL,), choice="none")),
            )

    @pytest.mark.anyio
    async def test_live_provider_call_violating_choice_none_is_rejected(self) -> None:
        declaration = {
            "name": "get_code",
            "description": "Retrieve the four-digit code.",
            "parameters": {"type": "object"},
        }
        with pytest.raises(OutputContractError, match="outside the request's allowed"):
            await _run(
                _json_handler(_fixture("function_choice_none_provider_violation")),
                _request(tools=ToolParams(functions=(declaration,), choice="none")),
            )

    @pytest.mark.anyio
    async def test_function_call_without_function_tools_fails(self) -> None:
        reply = _reply(
            status="requires_action",
            steps=[
                {
                    "type": "function_call",
                    "id": "c1",
                    "name": "get_weather",
                    "arguments": {},
                }
            ],
        )
        with pytest.raises(OutputContractError, match="requires an action"):
            await _run(_json_handler(reply))

    @pytest.mark.parametrize(
        ("function", "message"),
        [
            ({}, "function name"),
            ({"name": "bad name"}, "function name"),
            ({"name": "f"}, "non-empty description"),
            ({"name": "f", "description": "  "}, "non-empty description"),
            ({"name": "f", "description": "d", "parameters": []}, "parameters must be"),
            (
                {"name": "f", "description": "d", "parameters": {"type": "string"}},
                "explicit type: object",
            ),
            (
                {"name": "f", "description": "d", "extra": 1},
                "unsupported Interactions function declaration field",
            ),
        ],
    )
    def test_declaration_is_validated(
        self, function: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(DialectError, match=message):
            google_interactions._function_source(function)

    def test_unsupported_tool_choice_is_rejected(self) -> None:
        with pytest.raises(DialectError, match="unsupported Interactions tool choice"):
            google_interactions._tool_choice_wire("sometimes")
        with pytest.raises(DialectError, match="must be a string or mapping"):
            google_interactions._tool_choice_wire(5)
        with pytest.raises(DialectError, match="invalid Interactions"):
            google_interactions._tool_choice_wire({"type": "other"})
        with pytest.raises(DialectError, match="needs a name"):
            google_interactions._tool_choice_wire({"type": "function", "name": ""})


class TestHostedTools:
    HANDLER_REPLY = _reply(
        steps=[
            _thought(),
            {
                "type": "google_search_call",
                "id": "s1",
                "arguments": {"queries": ["nobel"]},
            },
            {
                "type": "google_search_result",
                "call_id": "s1",
                "result": [{"search_suggestions": "snippet"}],
            },
            _model_output("answer"),
        ]
    )

    @pytest.mark.anyio
    async def test_declaration_is_a_typed_tool(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_reply())

        await _run(
            handler,
            _request(
                tools=ToolParams(
                    hosted=(
                        HostedToolSpec(
                            "google_search", {"search_types": ["web_search"]}
                        ),
                        HostedToolSpec("url_context"),
                    )
                )
            ),
        )

        assert seen["body"]["tools"] == [
            {"type": "google_search", "search_types": ["web_search"]},
            {"type": "url_context"},
        ]

    @pytest.mark.anyio
    async def test_call_and_result_lift_to_server_tool_use(self) -> None:
        result = await _run(
            _json_handler(self.HANDLER_REPLY),
            _request(tools=ToolParams(hosted=(HostedToolSpec("google_search"),))),
        )

        assert result.server_tool_uses is not None
        assert len(result.server_tool_uses) == 1
        use = result.server_tool_uses[0]
        assert use.tool_type == "google_search"
        assert use.tool_use_id == "s1"
        assert use.input == {"queries": ["nobel"]}
        assert use.result == [{"search_suggestions": "snippet"}]

    @pytest.mark.anyio
    async def test_unrequested_hosted_call_fails(self) -> None:
        with pytest.raises(OutputContractError, match="unrequested google_search call"):
            await _run(_json_handler(self.HANDLER_REPLY))

    @pytest.mark.anyio
    async def test_url_context_result_becomes_a_citation(self) -> None:
        reply = _reply(
            steps=[
                {
                    "type": "url_context_call",
                    "id": "u1",
                    "arguments": {"urls": ["https://example.com"]},
                },
                {
                    "type": "url_context_result",
                    "call_id": "u1",
                    "result": [{"url": "https://example.com", "status": "success"}],
                },
                _model_output("summary"),
            ]
        )
        result = await _run(
            _json_handler(reply),
            _request(tools=ToolParams(hosted=(HostedToolSpec("url_context"),))),
        )

        assert result.citations is not None
        assert [citation.url for citation in result.citations] == [
            "https://example.com"
        ]

    @pytest.mark.anyio
    async def test_result_without_its_call_fails(self) -> None:
        reply = _reply(
            steps=[
                {
                    "type": "google_search_result",
                    "call_id": "s1",
                    "result": [{"search_suggestions": "x"}],
                },
            ]
        )
        with pytest.raises(OutputContractError, match="which no earlier step called"):
            await _run(
                _json_handler(reply),
                _request(tools=ToolParams(hosted=(HostedToolSpec("google_search"),))),
            )

    @pytest.mark.parametrize(
        ("kind", "config", "message"),
        [
            ("google_search", [], "config must be an object"),
            ("url_context", {"x": 1}, "url_context takes no config"),
            ("google_maps", {}, "unsupported Interactions hosted tool"),
            ("google_search", {"other": 1}, "unsupported field"),
            ("google_search", {"search_types": "web_search"}, "must be an array"),
            ("google_search", {"search_types": []}, "non-empty array"),
            ("google_search", {"search_types": ["tv"]}, "non-empty array"),
            (
                "google_search",
                {"search_types": ["web_search", "web_search"]},
                "must not repeat",
            ),
        ],
    )
    def test_config_is_validated(self, kind: str, config: object, message: str) -> None:
        error = google_interactions._hosted_tool_config_error(kind, config)
        assert error is not None and message in error

    @pytest.mark.parametrize(
        ("kind", "config"),
        [
            ("google_search", {}),
            ("google_search", {"search_types": ["image_search"]}),
            ("url_context", {}),
        ],
    )
    def test_valid_config_passes(self, kind: str, config: object) -> None:
        assert google_interactions._hosted_tool_config_error(kind, config) is None


class TestStructuredOutputAndImages:
    SCHEMA = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    @pytest.mark.anyio
    async def test_response_format_carries_the_schema(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200, json=_reply(steps=[_model_output('{"answer": "4"}')])
            )

        result = await _run(
            handler,
            _request(
                structured_output=StructuredOutputParams(
                    format="json_schema", schema=self.SCHEMA
                )
            ),
        )

        assert seen["body"]["response_format"] == {
            "type": "text",
            "mime_type": "application/json",
            "schema": self.SCHEMA,
        }
        assert result.structured_output is not None
        assert result.structured_output.value == {"answer": "4"}

    @pytest.mark.anyio
    async def test_invalid_json_output_fails(self) -> None:
        with pytest.raises(OutputContractError, match="not valid JSON"):
            await _run(
                _json_handler(_reply(steps=[_model_output("not json")])),
                _request(
                    structured_output=StructuredOutputParams(
                        format="json_schema", schema=self.SCHEMA
                    )
                ),
            )

    @pytest.mark.anyio
    async def test_json_object_format_is_rejected(self) -> None:
        with pytest.raises(DialectError, match="only carries a JSON schema"):
            await _run(
                _no_io,
                _request(
                    structured_output=StructuredOutputParams(format="json_object")
                ),
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("part", "wire"),
        [
            (
                ImagePart(data="aGk=", media_type="image/png"),
                {"type": "image", "data": "aGk=", "mime_type": "image/png"},
            ),
            (
                ImagePart(url="https://example.com/a.png", media_type="image/png"),
                {
                    "type": "image",
                    "uri": "https://example.com/a.png",
                    "mime_type": "image/png",
                },
            ),
            (
                ImagePart(url="https://example.com/a.png"),
                {"type": "image", "uri": "https://example.com/a.png"},
            ),
        ],
    )
    async def test_image_parts_lower_to_content_blocks(
        self, part: ImagePart, wire: dict[str, Any]
    ) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_reply())

        await _run(
            handler,
            _request(input=_chat(ChatMessage("user", (TextPart("see"), part)))),
        )
        assert seen["body"]["input"][0]["content"] == [
            {"type": "text", "text": "see"},
            wire,
        ]

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("part", "message"),
        [
            (
                ImagePart(data="aGk=", media_type="application/pdf"),
                "unsupported Interactions image media type",
            ),
            (ImagePart(data="aGk="), "require data and an explicit MIME type"),
            (
                ImagePart(data="", media_type="image/png"),
                "require data and an explicit MIME type",
            ),
            (ImagePart(url="", media_type="image/png"), "must not be empty"),
            (
                ImagePart(data="aGk=", media_type="image/png", detail="high"),
                "no per-image detail",
            ),
        ],
    )
    async def test_invalid_image_parts_are_rejected(
        self, part: ImagePart, message: str
    ) -> None:
        with pytest.raises(DialectError, match=message):
            await _run(_no_io, _request(input=_chat(ChatMessage("user", (part,)))))


class TestStreaming:
    @pytest.mark.anyio
    async def test_request_asks_for_stream_and_reads_recorded_events(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                content=_recorded_stream("stream_text"),
                headers={"content-type": "text/event-stream"},
            )

        result = await _run(
            handler,
            _request(scheduling=SchedulingParams(stream=True)),
        )

        assert seen["body"]["stream"] is True
        # The recording streamed a thought signature, never a summary, and two
        # text deltas reassembling to "391".
        assert result.texts == ("391",)
        assert result.finish_reasons == ("completed",)
        assert result.finish_reasons == ("completed",)

    @pytest.mark.anyio
    async def test_text_deltas_concatenate_in_order(self) -> None:
        events = _sse(
            _created(),
            _start(0, {"type": "model_output"}),
            _delta(0, {"type": "text", "text": "He"}),
            _delta(0, {"type": "text", "text": "llo"}),
            _stop(0),
            _completed(),
            _DONE,
        )
        result = await _run(
            _stream(events), _request(scheduling=SchedulingParams(stream=True))
        )
        assert result.texts == ("Hello",)

    @pytest.mark.anyio
    async def test_function_arguments_accumulate_across_deltas(self) -> None:
        events = _sse(
            _created(),
            _start(
                0,
                {
                    "type": "function_call",
                    "id": "c1",
                    "name": "get_weather",
                    "arguments": {},
                },
            ),
            _delta(0, {"type": "arguments_delta", "arguments": '{"city":'}),
            _delta(0, {"type": "arguments_delta", "arguments": ' "Paris"}'}),
            _stop(0),
            _completed(status="requires_action"),
            _DONE,
        )
        result = await _run(
            _stream(events),
            _function_request(scheduling=SchedulingParams(stream=True)),
        )

        assert result.tool_calls is not None
        assert result.tool_calls[0].arguments == {"city": "Paris"}

    @pytest.mark.anyio
    async def test_recorded_function_call_stream(self) -> None:
        result = await _run(
            _stream(_recorded_stream("stream_function_call")),
            _function_request(scheduling=SchedulingParams(stream=True)),
        )

        assert result.finish_reasons == ("requires_action",)
        assert result.tool_calls is not None
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"city": "Paris"}

    @pytest.mark.anyio
    async def test_recorded_incomplete_stream_carries_no_steps(self) -> None:
        result = await _run(
            _stream(_recorded_stream("stream_max_tokens")),
            _request(scheduling=SchedulingParams(stream=True)),
        )

        assert result.texts == ("",)
        assert result.finish_reasons == ("incomplete",)

    @pytest.mark.anyio
    async def test_thought_summary_deltas_assemble(self) -> None:
        events = _sse(
            _created(),
            _start(0, {"type": "thought"}),
            _delta(
                0,
                {
                    "type": "thought_summary",
                    "content": {"type": "text", "text": "part one"},
                },
            ),
            _delta(
                0,
                {
                    "type": "thought_summary",
                    "content": {"type": "text", "text": "part two"},
                },
            ),
            _delta(0, {"type": "thought_signature", "signature": "s"}),
            _stop(0),
            _completed(),
            _DONE,
        )
        result = await _run(
            _stream(events),
            _request(
                scheduling=SchedulingParams(stream=True),
                reasoning=ReasoningParams(summary="auto"),
            ),
        )

        assert result.reasoning is not None
        assert result.reasoning[0] is not None
        assert result.reasoning[0].text == "part onepart two"

    @pytest.mark.parametrize(
        "name",
        [
            "stream_thought_summary_completed",
            "stream_thought_summary_incomplete",
            "stream_signature_only_completed",
            "stream_multiple_functions_signature_only",
            "stream_url_context_signature_only",
            "stream_google_search_signature_only",
        ],
    )
    def test_live_sse_capture_preserves_final_frame_separator(self, name: str) -> None:
        assert (FIXTURES / f"{name}.sse").read_bytes().endswith(b"\n\n")

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("name", "hosted_kind", "function_count", "status"),
        [
            ("stream_signature_only_completed", None, 0, "completed"),
            ("stream_multiple_functions_signature_only", None, 2, "requires_action"),
            ("stream_url_context_signature_only", "url_context", 0, "completed"),
            ("stream_google_search_signature_only", "google_search", 0, "completed"),
        ],
    )
    async def test_live_signature_only_thought_preserves_opaque(
        self,
        name: str,
        hosted_kind: str | None,
        function_count: int,
        status: str,
    ) -> None:
        functions = (
            (
                {
                    "name": "lookup_value",
                    "description": "Look up a label's value.",
                    "parameters": {"type": "object"},
                },
            )
            if function_count
            else ()
        )
        hosted = (HostedToolSpec(hosted_kind),) if hosted_kind else ()
        result = await _run(
            _stream(_recorded_stream(name)),
            _request(
                reasoning=ReasoningParams(summary="auto"),
                scheduling=SchedulingParams(stream=True),
                tools=ToolParams(functions=functions, hosted=hosted),
            ),
        )
        assert result.finish_reasons == (status,)
        assert result.reasoning is not None
        assert result.reasoning[0] is not None
        assert result.reasoning[0].text is None
        assert result.reasoning[0].opaque_roundtrip is not None
        history = json.loads(result.reasoning[0].opaque_roundtrip)["steps"]
        assert any(
            step["type"] == "thought" and step.get("signature") == "sig-redacted"
            for step in history
        )
        assert len(result.tool_calls or ()) == function_count
        if hosted_kind:
            assert result.server_tool_uses
            assert all(
                tool.tool_type == hosted_kind and tool.result is not None
                for tool in result.server_tool_uses
            )
            assert result.citations
            assert result.texts[0]

    @pytest.mark.anyio
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        ("name", "status", "summary_chars"),
        [
            ("stream_thought_summary_completed", "completed", 968),
            ("stream_thought_summary_incomplete", "incomplete", 721),
        ],
    )
    async def test_recorded_thought_summary_text_is_preserved_verbatim(
        self, stream: bool, name: str, status: str, summary_chars: int
    ) -> None:
        recording = _recorded_stream(name)
        events = [
            json.loads(line.removeprefix("data: "))
            for line in recording.decode().splitlines()
            if line.startswith("data: {")
        ]
        summary_parts = [
            event["delta"]["content"]["text"]
            for event in events
            if event["event_type"] == "step.delta"
            and event["delta"]["type"] == "thought_summary"
        ]
        text = "".join(
            event["delta"]["text"]
            for event in events
            if event["event_type"] == "step.delta" and event["delta"]["type"] == "text"
        )
        summary = "".join(summary_parts)
        handler = (
            _stream(recording)
            if stream
            else _json_handler(
                _reply(
                    status=status, steps=[_thought(summary_parts), _model_output(text)]
                )
            )
        )

        result = await _run(
            handler,
            _request(
                scheduling=SchedulingParams(stream=stream),
                reasoning=ReasoningParams(summary="auto"),
            ),
        )

        assert len(summary_parts) == 2
        assert all(part.endswith("\n\n\n") for part in summary_parts)
        assert len(summary) == summary_chars
        assert result.texts == (text,)
        assert result.finish_reasons == (status,)
        assert result.reasoning is not None
        assert result.reasoning[0] is not None
        assert result.reasoning[0].text == summary

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("events", "message"),
        [
            pytest.param(
                [_DONE], "ended without interaction.completed", id="done-only"
            ),
            pytest.param(
                [_created(), _completed(status="failed")], "reply failed", id="failed"
            ),
            pytest.param(
                [_created(), {"event_type": "error", "error": {"message": "boom"}}],
                "reported an error: boom",
                id="error-event",
            ),
            pytest.param(
                [_created(), _start(0, {"type": "model_output"}), _completed(), _DONE],
                "unfinished step",
                id="step-without-stop",
            ),
            pytest.param(
                [
                    _created(),
                    _delta(0, {"type": "text", "text": "x"}),
                    _completed(),
                    _DONE,
                ],
                "delta for unknown step",
                id="delta-before-start",
            ),
            pytest.param(
                [
                    _created(),
                    _start(0, {"type": "model_output"}),
                    _stop(0),
                    _start(0, {"type": "model_output"}),
                    _completed(),
                    _DONE,
                ],
                "restarted step",
                id="restart",
            ),
            pytest.param(
                [_created(), _stop(3), _completed(), _DONE],
                "stopped unknown step",
                id="stop-without-start",
            ),
            pytest.param(
                [
                    _created(),
                    _start(0, {"type": "model_output"}),
                    _delta(0, {"type": "image", "data": "x"}),
                    _stop(0),
                    _completed(),
                    _DONE,
                ],
                "unsupported delta 'image'",
                id="unsupported-delta",
            ),
            pytest.param(
                [_completed(), _DONE, _completed()],
                "after its terminal marker",
                id="event-after-done",
            ),
            pytest.param(
                [
                    _created(),
                    _completed(),
                    _start(0, {"type": "model_output"}),
                    _DONE,
                ],
                "after interaction.completed",
                id="step-after-completed",
            ),
            pytest.param(
                [_created(), {"event_type": "mystery"}],
                "unsupported event_type 'mystery'",
                id="unsupported-event",
            ),
            pytest.param(
                [
                    {
                        "event_type": "interaction.created",
                        "interaction": {"status": "completed"},
                    }
                ],
                "status other than in_progress",
                id="created-terminal",
            ),
            pytest.param(
                [
                    _created(),
                    {"event_type": "interaction.status_update", "status": "completed"},
                ],
                "terminal status",
                id="status-update-terminal",
            ),
            pytest.param(
                [_created(), _completed(status="queued"), _DONE],
                "not terminal",
                id="queued",
            ),
        ],
    )
    async def test_stream_failures_are_loud(
        self, events: list[object], message: str
    ) -> None:
        with pytest.raises(OutputContractError, match=message):
            await _run(
                _stream(_sse(*events)),
                _request(scheduling=SchedulingParams(stream=True)),
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "content",
        [
            b"",
            b"event: x\ndata: not json\n",
            b"event: x\ndata: {}\n",
            b'event: x\ndata: {"event_type": "step.start"}\n',
            b'event: x\ndata: {"index": -1, "step": {}, "event_type": "step.start"}\n',
            b'event: x\ndata: {"event_type": "step.start", "index": true}\n',
            b"garbage line\n",
            b"weird: field\ndata: {}\n",
        ],
    )
    async def test_malformed_streams_fail(self, content: bytes) -> None:
        with pytest.raises(OutputContractError):
            await _run(
                _stream(content),
                _request(scheduling=SchedulingParams(stream=True)),
            )

    @pytest.mark.anyio
    async def test_stream_http_error_is_raised(self) -> None:
        events = _sse(_created())
        with pytest.raises(httpx.HTTPStatusError):
            await _run(
                _stream(events, status=500),
                _request(scheduling=SchedulingParams(stream=True)),
            )

    @pytest.mark.anyio
    async def test_commented_lines_and_event_ids_are_tolerated(self) -> None:
        body = (
            b": keep-alive\n"
            b"id: 7\n"
            b"event: interaction.created\n"
            b'data: {"event_type":"interaction.created","interaction":{}}\n'
            b"\n"
            b"retry: 10\n"
            b"event: interaction.completed\n"
            b'data: {"event_type":"interaction.completed",\n'
            b'data: "interaction":{"status":"completed"}}\n'
            b"\n"
            b"data: [DONE]\n"
        )
        result = await _run(
            _stream(body), _request(scheduling=SchedulingParams(stream=True))
        )
        assert result.finish_reasons == ("completed",)

    @pytest.mark.anyio
    async def test_multiline_data_is_one_event(self) -> None:
        body = (
            b"event: interaction.completed\n"
            b'data: {"event_type":"interaction.completed",\n'
            b'data:  "interaction":{"status":"completed"}}\n'
            b"\n"
            b"data: [DONE]\n"
        )
        result = await _run(
            _stream(body), _request(scheduling=SchedulingParams(stream=True))
        )
        assert result.finish_reasons == ("completed",)


class TestBudgetExceeded:
    @pytest.mark.anyio
    async def test_budget_exceeded_is_a_named_status_error(self) -> None:
        with pytest.raises(OutputContractError, match="budget_exceeded"):
            await _run(_json_handler(_reply(status="budget_exceeded")))

    @pytest.mark.anyio
    async def test_budget_exceeded_on_a_stream_is_a_named_status_error(self) -> None:
        events = _sse(_created(), _completed(status="budget_exceeded"), _DONE)
        with pytest.raises(OutputContractError, match="budget_exceeded"):
            await _run(
                _stream(events), _request(scheduling=SchedulingParams(stream=True))
            )


class TestVerifiersAndHelpers:
    def test_functions_verifier_detects_changed_declarations(self) -> None:
        verifier = google_interactions._FunctionsVerifier((_FUNCTION_TOOL,))
        good = {"tools": [_FUNCTION_TOOL]}
        assert verifier.verify(good) is None
        assert verifier.verify({"tools": []}) == "function tools changed"
        assert verifier.verify({"tools": "x"}) == "function tools changed"
        changed = {"tools": [{**_FUNCTION_TOOL, "description": "other"}]}
        assert verifier.verify(changed) == "function tools changed"

    def test_functions_verifier_normalizes_openai_shape(self) -> None:
        verifier = google_interactions._FunctionsVerifier(
            ({"type": "function", "function": {"name": "f", "description": "d"}},)
        )
        assert (
            verifier.verify(
                {"tools": [{"type": "function", "name": "f", "description": "d"}]}
            )
            is None
        )

    def test_hosted_verifier(self) -> None:
        verifier = google_interactions._HostedToolsVerifier(
            (HostedToolSpec("google_search"),)
        )
        assert verifier.verify({"tools": [{"type": "google_search"}]}) is None
        assert (
            verifier.verify({"tools": [{"type": "url_context"}]})
            == "hosted tools changed"
        )
        assert verifier.verify({}) == "hosted tools changed"

    def test_stream_verifier(self) -> None:
        verifier = google_interactions._StreamVerifier(MODEL)
        assert verifier.verify({"model": MODEL, "stream": True}) is None
        assert verifier.verify({"model": MODEL}) == "stream flag changed"
        assert (
            verifier.verify({"model": "other", "stream": True})
            == "model does not match the bound model"
        )

    def test_annotation_requires_url_citation(self) -> None:
        with pytest.raises(OutputContractError, match="file_citation"):
            google_interactions._annotation({"type": "file_citation"}, "a")
        assert google_interactions._annotation({}, "a") is None
        citation = google_interactions._annotation(
            {"type": "url_citation", "url": "u", "title": "t"}, "a"
        )
        assert citation is not None and citation.url == "u" and citation.title == "t"
        with pytest.raises(OutputContractError, match="title must be a string"):
            google_interactions._annotation(
                {"type": "url_citation", "url": "u", "title": 5}, "a"
            )

    def test_structured_output_none_without_format_or_on_incomplete(self) -> None:
        assert (
            google_interactions._structured_output(_request(), "completed", "{}")
            is None
        )
        structured = _request(
            structured_output=StructuredOutputParams(
                format="json_schema", schema={"type": "object"}
            )
        )
        assert (
            google_interactions._structured_output(structured, "incomplete", "{}")
            is None
        )


class TestCoverageBranches:
    """Error and edge branches not reached by the behavioural tests above."""

    def test_capability_config_domains(self) -> None:
        hosted = cast(Supported, CAPABILITY_DECISIONS["hosted_tools"])
        with pytest.raises(ValueError, match="only google_search and url_context"):
            hosted.binding.validate_config(HostedToolsOptions(("google_maps",)))
        # ``MultimodalInputOptions`` admits only "image"; the dialect's own
        # wider-domain guard is exercised on a record built past the type check.
        wider = MultimodalInputOptions(("image",))
        object.__setattr__(wider, "modalities", ("image", "video"))
        with pytest.raises(ValueError, match="does not support modalities"):
            google_interactions._validate_multimodal_config(wider)

    def test_output_validators_reject_wrong_shapes(self) -> None:
        with pytest.raises(OutputContractError, match="invalid shape"):
            google_interactions._tuple_validator(Citation, "citations")("not a tuple")

    @pytest.mark.parametrize(
        ("function", "message"),
        [
            (
                {"type": "function", "function": {"name": "f"}, "extra": 1},
                "invalid OpenAI-shaped",
            ),
            (
                {
                    "name": "f",
                    "description": "d",
                    "parameters": {"type": "object"},
                    "junk": 1,
                },
                "unsupported Interactions function declaration field",
            ),
        ],
    )
    def test_function_source_rejections(
        self, function: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(DialectError, match=message):
            google_interactions._function_source(function)

    def test_tool_choice_accepts_a_nested_function_object(self) -> None:
        assert google_interactions._tool_choice_wire(
            {"type": "function", "function": {"name": "f"}}
        ) == {"allowed_tools": {"mode": "any", "tools": ["f"]}}
        assert google_interactions._declared_function_names(
            _request(
                tools=ToolParams(
                    functions=(_FUNCTION_TOOL,),
                    choice=cast(Any, {"type": "function", "function": {"name": "f"}}),
                )
            )
        ) == frozenset({"f"})

    def test_hosted_tool_helpers_reject_unknown_kinds(self) -> None:
        with pytest.raises(DialectError, match="unsupported Interactions hosted tool"):
            google_interactions._hosted_tool(SimpleNamespace(kind=5, config={}))
        with pytest.raises(DialectError, match="unsupported field"):
            google_interactions._hosted_tool(
                SimpleNamespace(kind="google_search", config={"time": 1})
            )

    def test_image_rejects_detail_and_non_string_text(self) -> None:
        with pytest.raises(DialectError, match="no per-image detail"):
            google_interactions._image_content(
                ImagePart(data="aGk=", media_type="image/png", detail="high")
            )
        with pytest.raises(DialectError, match="must carry a string"):
            google_interactions._part_to_content(TextPart(cast(Any, 5)))

    def test_message_step_rejections(self) -> None:
        with pytest.raises(DialectError, match="exactly one ToolResultPart"):
            google_interactions._message_to_steps(
                ChatMessage("tool", (ToolResultPart("c1", "x"), TextPart("y")))
            )
        with pytest.raises(DialectError, match="cannot be empty"):
            google_interactions._message_to_steps(ChatMessage("user", ()))
        with pytest.raises(DialectError, match="non-empty id and name"):
            google_interactions._message_to_steps(
                ChatMessage("assistant", (ToolCallPart("", "f", {}),))
            )
        with pytest.raises(DialectError, match="object of arguments"):
            google_interactions._message_to_steps(
                ChatMessage("assistant", (ToolCallPart("c", "f", cast(Any, "x")),))
            )

    def test_input_verifier_reports_tool_result_drift(self) -> None:
        verifier = google_interactions._InputVerifier.from_request(
            _chat(
                _say("user", "go"),
                ChatMessage("assistant", (ToolCallPart("c1", "f", {"a": 1}),)),
                ChatMessage("tool", (ToolResultPart("c1", "ok"),)),
                _say("user", "next"),
            )
        )
        good: dict[str, Any] = {
            "input": [
                {"type": "user_input", "content": [{"type": "text", "text": "go"}]},
                {
                    "type": "function_call",
                    "id": "c1",
                    "name": "f",
                    "arguments": {"a": 1},
                },
                {
                    "type": "function_result",
                    "call_id": "c1",
                    "name": "f",
                    "result": "ok",
                },
                {"type": "user_input", "content": [{"type": "text", "text": "next"}]},
            ]
        }
        assert verifier.verify(good) is None

        def result_changed(**edits: Any) -> str | None:
            body = copy.deepcopy(good)
            body["input"][2].update(edits)
            return verifier.verify(body)

        assert result_changed(call_id="other") == "input[2] answers a different call"
        assert (
            result_changed(result="no")
            == "input[2] result no longer matches the tool result"
        )
        # The IR result was not an error: an absent or false flag is faithful,
        # anything else (true, 5) is not.
        assert result_changed(is_error=False) is None
        assert (
            result_changed(is_error=True) == "input[2] carries an unexpected is_error"
        )
        assert result_changed(is_error=5) == "input[2] carries an unexpected is_error"

        def call_changed(**edits: Any) -> str | None:
            body = copy.deepcopy(good)
            body["input"][1].update(edits)
            return verifier.verify(body)

        for field_name, value in (
            ("arguments", {"a": 2}),
            ("name", "g"),
            ("id", "c2"),
        ):
            assert call_changed(**{field_name: value}) == (
                "input[1] function call no longer matches the provider-neutral input"
            )
        assert call_changed(signature="x") == "input[1] has an unexpected shape"

    def test_input_verifier_pins_a_true_is_error_flag(self) -> None:
        # The audit blind spot: a result the IR marked as an error must keep its
        # flag; losing it would present a failure to the model as a success.
        verifier = google_interactions._InputVerifier.from_request(
            _chat(
                _say("user", "go"),
                ChatMessage("assistant", (ToolCallPart("c1", "f", {}),)),
                ChatMessage("tool", (ToolResultPart("c1", "boom", is_error=True),)),
            )
        )
        body: dict[str, Any] = {
            "input": [
                {"type": "user_input", "content": [{"type": "text", "text": "go"}]},
                {"type": "function_call", "id": "c1", "name": "f", "arguments": {}},
                {
                    "type": "function_result",
                    "call_id": "c1",
                    "result": "boom",
                    "name": "f",
                    "is_error": True,
                },
            ]
        }
        assert verifier.verify(body) is None
        del body["input"][2]["is_error"]
        assert verifier.verify(body) == "input[2] lost its is_error flag"
        body["input"][2]["is_error"] = False
        assert verifier.verify(body) == "input[2] lost its is_error flag"

    def test_functions_verifier_rejects_unserializable_and_missing_tools(self) -> None:
        verifier = google_interactions._FunctionsVerifier((_FUNCTION_TOOL,))
        with pytest.raises(TypeError, match="must be objects"):
            google_interactions._FunctionsVerifier((cast(Any, "x"),))
        # No tools at all is a change; only a non-function tool alone is fine.
        assert verifier.verify({}) == "function tools changed"
        assert verifier.verify({"tools": [{"type": "google_search"}]}) == (
            "function tools changed"
        )

    def test_hosted_verifier_rejects_invalid_configs(self) -> None:
        bad = google_interactions._HostedToolsVerifier(
            (SimpleNamespace(kind=5, config={}),)
        )
        assert bad.verify({"tools": []}) == "hosted tool config is invalid"
        none_config = google_interactions._HostedToolsVerifier(
            (SimpleNamespace(kind="google_search", config=None),)
        )
        assert none_config.verify({"tools": []}) == "hosted tool config is invalid"

    def test_structured_format_verifier_reports_each_drift(self) -> None:
        verifier = google_interactions._StructuredFormatVerifier()
        assert (
            verifier.verify(
                {"response_format": {"type": "text", "mime_type": "application/json"}}
            )
            is None
        )
        assert verifier.verify({}) == "structured output response_format changed"
        assert verifier.verify({"response_format": {"type": "image"}}) == (
            "structured output type changed"
        )
        assert (
            verifier.verify(
                {"response_format": {"type": "text", "mime_type": "text/plain"}}
            )
            == "structured output MIME type changed"
        )

    @pytest.mark.parametrize(
        ("tools", "message"),
        [
            (
                ToolParams(functions=(cast(Any, {"name": "f"}),)),
                "non-empty description",
            ),
            (
                ToolParams(
                    functions=(_FUNCTION_TOOL,),
                    choice=cast(Any, {"type": "function", "name": ""}),
                ),
                "needs a name",
            ),
            (
                ToolParams(hosted=(HostedToolSpec("google_search", {"oops": 1}),)),
                "unsupported field",
            ),
        ],
    )
    def test_validate_request_rejects_bad_tool_configs(
        self, tools: ToolParams, message: str
    ) -> None:
        req = _request(tools=tools)
        audit = _audit(req)
        assert any(
            isinstance(decision, Rejected) and message in decision.reason
            for decision in audit.decisions.values()
        )

    def test_structured_schema_is_required(self) -> None:
        # ``json_schema`` without a schema fails the options type itself, so the
        # dialect guard is exercised with a directly built options record.
        forged = object.__new__(StructuredOutputParams)
        object.__setattr__(forged, "format", "json_schema")
        object.__setattr__(forged, "schema", None)
        object.__setattr__(forged, "name", None)
        object.__setattr__(forged, "strict", None)
        req = _request(structured_output=forged)
        audit = RequestAudit(active_request_leaves(req))
        with pytest.raises(DialectError, match="requires a schema"):
            _dialect().prepare(req, audit)

    @pytest.mark.anyio
    async def test_execute_rejects_a_non_boolean_stream(self) -> None:
        prepared = PreparedRequest(
            "v1beta/interactions",
            {"model": MODEL, "input": [], "stream": "yes"},
            google_interactions._InteractionsContext(_request()),
        )
        with pytest.raises(DialectError, match="stream must be true or absent"):
            await _dialect().execute(prepared)

    @pytest.mark.anyio
    async def test_function_arguments_must_decode_to_an_object(self) -> None:
        events = _sse(
            _created(),
            _start(
                0,
                {
                    "type": "function_call",
                    "id": "c1",
                    "name": "get_weather",
                    "arguments": {},
                },
            ),
            _delta(0, {"type": "arguments_delta", "arguments": "[1, 2]"}),
            _stop(0),
            _completed(status="requires_action"),
            _DONE,
        )
        with pytest.raises(OutputContractError, match="must decode to an object"):
            await _run(
                _stream(events),
                _function_request(scheduling=SchedulingParams(stream=True)),
            )

    @pytest.mark.anyio
    async def test_unparsable_function_arguments_fail(self) -> None:
        events = _sse(
            _created(),
            _start(
                0,
                {
                    "type": "function_call",
                    "id": "c1",
                    "name": "get_weather",
                    "arguments": {},
                },
            ),
            _delta(0, {"type": "arguments_delta", "arguments": "{"}),
            _stop(0),
            _completed(status="requires_action"),
            _DONE,
        )
        with pytest.raises(OutputContractError, match="not strict JSON"):
            await _run(
                _stream(events),
                _function_request(scheduling=SchedulingParams(stream=True)),
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "steps",
        [
            [
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": "x"}],
                    "error": {"code": 1},
                }
            ],
            [{"type": "function_call", "id": "c1", "name": "", "arguments": {}}],
            [
                {
                    "type": "function_call",
                    "id": "c1",
                    "name": "get_weather",
                    "arguments": "x",
                }
            ],
            [{"type": "function_result", "call_id": "", "result": 1}],
            [
                {
                    "type": "function_result",
                    "call_id": "c1",
                    "result": 1,
                    "is_error": "yes",
                }
            ],
            [{"type": "google_search_call", "id": "", "arguments": {}}],
            [{"type": "google_search_call", "id": "s1", "arguments": "x"}],
            [{"type": "google_search_result", "call_id": "", "result": []}],
            [
                {
                    "type": "google_search_result",
                    "call_id": "s1",
                    "result": [],
                    "is_error": "x",
                }
            ],
            [
                {
                    "type": "google_search_result",
                    "call_id": "s1",
                    "result": {"not": "array"},
                }
            ],
        ],
    )
    async def test_step_field_rejections(self, steps: list[dict[str, Any]]) -> None:
        with pytest.raises(OutputContractError):
            await _run(
                _json_handler(_reply(status="requires_action", steps=steps)),
                _function_request(
                    tools=ToolParams(
                        functions=(_FUNCTION_TOOL,),
                        hosted=(HostedToolSpec("google_search"),),
                    )
                ),
            )

    @pytest.mark.anyio
    async def test_hosted_error_result_is_recorded_and_skips_citations(self) -> None:
        reply = _reply(
            steps=[
                {
                    "type": "url_context_call",
                    "id": "u1",
                    "arguments": {"urls": ["https://example.com"]},
                },
                {
                    "type": "url_context_result",
                    "call_id": "u1",
                    "is_error": True,
                    "result": [{"url": "not a citation"}],
                },
                _model_output("x"),
            ]
        )
        result = await _run(
            _json_handler(reply),
            _request(tools=ToolParams(hosted=(HostedToolSpec("url_context"),))),
        )
        assert result.citations is None
        assert result.server_tool_uses is not None
        assert result.server_tool_uses[0].result == [{"url": "not a citation"}]

    @pytest.mark.anyio
    async def test_frozen_function_call_arguments_still_serialize(self) -> None:
        reply = _reply(
            status="requires_action",
            steps=[
                {
                    "type": "function_call",
                    "id": "c1",
                    "name": "get_weather",
                    "arguments": {"city": "Paris"},
                }
            ],
        )
        result = await _run(_json_handler(reply), _function_request())
        json.dumps(result.tool_calls[0].arguments)  # type: ignore[union-attr]

    def test_stream_state_rejects_an_unknown_event(self) -> None:
        state = google_interactions._StreamState()
        with pytest.raises(OutputContractError, match="unexpected event 'created'"):
            state.consume({"event_type": "created"})

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("events", "message"),
        [
            (
                [
                    _created(),
                    _start(0, {"type": "model_output"}),
                    _delta(0, {"type": "text", "text": 5}),
                    _stop(0),
                    _completed(),
                    _DONE,
                ],
                "text delta must carry a string",
            ),
            (
                [
                    _created(),
                    _start(
                        0, {"type": "function_call", "id": "c1", "name": "get_weather"}
                    ),
                    _delta(0, {"type": "arguments_delta", "arguments": 5}),
                    _stop(0),
                    _completed(status="requires_action"),
                    _DONE,
                ],
                "arguments delta must be a string",
            ),
            (
                [
                    _created(),
                    _start(0, {"type": "thought"}),
                    _delta(0, {"type": "thought_signature", "signature": 5}),
                    _stop(0),
                    _completed(),
                    _DONE,
                ],
                "signature delta must be a string",
            ),
            (
                [
                    _created(),
                    _start(0, {"type": "thought", "summary": {}}),
                    _delta(
                        0,
                        {
                            "type": "thought_summary",
                            "content": {"type": "text", "text": "x"},
                        },
                    ),
                    _stop(0),
                    _completed(),
                    _DONE,
                ],
                "thought summary is not an array",
            ),
            (
                [_created(), _completed(status=""), _DONE],
                "has no status",
            ),
            (
                [_created(), _completed(status="mystery"), _DONE],
                "unknown status",
            ),
        ],
    )
    async def test_stream_delta_and_status_branches(
        self, events: list[object], message: str
    ) -> None:
        with pytest.raises(OutputContractError, match=message):
            await _run(
                _stream(_sse(*events)),
                _function_request(scheduling=SchedulingParams(stream=True)),
            )


class TestSamplingControlsByModel:
    """Protocol-supported sampling is forwarded without model-family tables."""

    CONTROLS = (
        SamplingParams(temperature=0.2),
        SamplingParams(top_p=0.9),
    )

    @pytest.mark.anyio
    @pytest.mark.parametrize("sampling", CONTROLS)
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        "model",
        [
            "gemini-2.5-flash",
            "gemini-3-flash-preview",
            "gemini-3.1-pro-preview",
            "gemini-3.8-flash",
            "models/gemini-3.8-flash",
            "gemini-4-argon",
            "gemini-10-pro",
            "gemini-flash-latest",
            "gateway-alias",
        ],
    )
    async def test_controls_reach_the_wire_and_record_for_every_model(
        self, model: str, sampling: SamplingParams, stream: bool
    ) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            if stream:
                return httpx.Response(
                    200,
                    content=_recorded_stream("stream_text"),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(200, json=_reply())

        result = await _run(
            handler,
            _request(sampling=sampling, scheduling=SchedulingParams(stream=stream)),
            model=model,
        )

        key, value = next(
            (name, getattr(sampling, name))
            for name in ("temperature", "top_p")
            if getattr(sampling, name) is not None
        )
        assert seen["body"]["generation_config"] == {key: value}
        assert result.request_params is not None
        recorded_body = result.request_params["body"]
        assert isinstance(recorded_body, dict)
        assert recorded_body["generation_config"] == {key: value}

    @pytest.mark.anyio
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("model", [OLD_MODEL, MODEL, "gateway-alias"])
    async def test_top_k_is_rejected_by_protocol_before_io(
        self, model: str, stream: bool
    ) -> None:
        with pytest.raises(DialectError, match=r"sampling.top_k"):
            await _run(
                _no_io,
                _request(
                    sampling=SamplingParams(top_k=40),
                    scheduling=SchedulingParams(stream=stream),
                ),
                model=model,
            )

    @pytest.mark.anyio
    async def test_other_controls_are_unaffected_on_new_models(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_reply())

        await _run(
            handler,
            _request(sampling=SamplingParams(max_tokens=64, stop=("END",), seed=7)),
            model="gemini-3.8-flash",
        )

        assert seen["body"]["generation_config"] == {
            "max_output_tokens": 64,
            "stop_sequences": ["END"],
            "seed": 7,
        }


class TestRequestIdentityAndDiagnostics:
    @pytest.mark.anyio
    @pytest.mark.parametrize("stream", [False, True])
    async def test_actual_request_digest_ignores_gateway_and_tracks_sampling(
        self, stream: bool
    ) -> None:
        definitions = []
        for base_url, temperature in [
            (ROOT, 0.2),
            ("https://gateway.example/google/v1beta/", 0.2),
            (ROOT, 0.3),
        ]:

            def handler(_request: httpx.Request) -> httpx.Response:
                if stream:
                    return httpx.Response(
                        200,
                        content=_recorded_stream("stream_text"),
                        headers={"content-type": "text/event-stream"},
                    )
                return httpx.Response(200, json=_reply())

            result = await _run(
                handler,
                _request(
                    sampling=SamplingParams(temperature=temperature),
                    scheduling=SchedulingParams(stream=stream),
                ),
                base_url=base_url,
                model=OLD_MODEL,
            )
            assert result.request_params is not None
            definitions.append(request_definition(result.request_params))

        assert definitions[0] == {
            "body": {"generation_config": {"temperature": 0.2}, "store": False},
            "stream": stream,
        }
        assert digest(definitions[0]) == digest(definitions[1])
        assert digest(definitions[0]) != digest(definitions[2])

    @pytest.mark.anyio
    @pytest.mark.parametrize("stream", [False, True])
    async def test_provider_reason_is_in_the_status_exception(
        self, stream: bool
    ) -> None:
        reason = "Thinking level is unsupported: THINKING_LEVEL_MINIMAL"
        with pytest.raises(httpx.HTTPStatusError) as raised:
            await _run(
                _json_handler({"error": {"code": 400, "message": reason}}, 400),
                _request(
                    reasoning=ReasoningParams(effort="minimal"),
                    scheduling=SchedulingParams(stream=stream),
                ),
            )
        assert raised.value.response.status_code == 400
        assert f"Google: {reason}" in str(raised.value)

    @pytest.mark.anyio
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        ("content", "message"),
        [
            (b"upstream unavailable", "Google: upstream unavailable"),
            (b'{"error": "plain"}', 'Google: {"error": "plain"}'),
            (("x" * 600).encode(), "Google: " + "x" * 500 + "..."),
            (b"", None),
        ],
    )
    async def test_proxy_error_diagnostics_are_bounded(
        self, content: bytes, message: str | None, stream: bool
    ) -> None:
        with pytest.raises(httpx.HTTPStatusError) as raised:
            await _run(
                lambda _: httpx.Response(502, content=content),
                _request(scheduling=SchedulingParams(stream=stream)),
            )
        if message is None:
            assert "Google:" not in str(raised.value)
        else:
            assert str(raised.value).endswith(message)


class TestStreamedHostedToolsAndAnnotations:
    """Deltas that carry whole values or citations, not just text."""

    ANNOTATION = {
        "type": "url_citation",
        "url": "https://example.com",
        "title": "Example",
        "start_index": 0,
        "end_index": 5,
    }
    SEARCH = ToolParams(hosted=(HostedToolSpec("google_search"),))
    CONTEXT = ToolParams(hosted=(HostedToolSpec("url_context"),))

    def _streaming(self, **changes: object) -> Request:
        return _request(scheduling=SchedulingParams(stream=True), **changes)

    @pytest.mark.anyio
    async def test_annotation_delta_attaches_a_citation_to_the_text(self) -> None:
        events = _sse(
            _created(),
            _start(0, {"type": "model_output"}),
            _delta(0, {"type": "text", "text": "Paris"}),
            _delta(
                0, {"type": "text_annotation_delta", "annotations": [self.ANNOTATION]}
            ),
            _stop(0),
            _completed(),
            _DONE,
        )
        result = await _run(_stream(events), self._streaming())

        assert result.texts == ("Paris",)
        assert result.citations is not None
        assert [(c.url, c.title) for c in result.citations] == [
            ("https://example.com", "Example")
        ]

    @pytest.mark.anyio
    async def test_annotation_deltas_accumulate(self) -> None:
        second = {**self.ANNOTATION, "url": "https://example.org"}
        events = _sse(
            _created(),
            _start(0, {"type": "model_output"}),
            _delta(0, {"type": "text", "text": "Paris"}),
            _delta(
                0, {"type": "text_annotation_delta", "annotations": [self.ANNOTATION]}
            ),
            _delta(0, {"type": "text_annotation_delta", "annotations": [second]}),
            _stop(0),
            _completed(),
            _DONE,
        )
        result = await _run(_stream(events), self._streaming())

        assert result.citations is not None
        assert [c.url for c in result.citations] == [
            "https://example.com",
            "https://example.org",
        ]

    @pytest.mark.anyio
    async def test_a_streamed_annotation_gets_the_same_checks_as_a_buffered_one(
        self,
    ) -> None:
        bad = {**self.ANNOTATION, "start_index": -1}
        events = _sse(
            _created(),
            _start(0, {"type": "model_output"}),
            _delta(0, {"type": "text", "text": "Paris"}),
            _delta(0, {"type": "text_annotation_delta", "annotations": [bad]}),
            _stop(0),
            _completed(),
            _DONE,
        )
        with pytest.raises(OutputContractError, match="start_index must be"):
            await _run(_stream(events), self._streaming())

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("blocks", "annotations", "message"),
        [
            pytest.param(
                [("text", "x")],
                "not a list",
                "must carry an annotations array",
                id="annotations-not-an-array",
            ),
            pytest.param(
                [],
                [{"type": "url_citation", "url": "u"}],
                "before any text",
                id="annotation-before-any-text",
            ),
        ],
    )
    async def test_malformed_annotation_deltas_fail(
        self, blocks: list[tuple[str, str]], annotations: object, message: str
    ) -> None:
        events: list[object] = [_created(), _start(0, {"type": "model_output"})]
        events += [_delta(0, {"type": kind, "text": text}) for kind, text in blocks]
        events += [
            _delta(0, {"type": "text_annotation_delta", "annotations": annotations}),
            _stop(0),
            _completed(),
            _DONE,
        ]
        with pytest.raises(OutputContractError, match=message):
            await _run(_stream(_sse(*events)), self._streaming())

    @pytest.mark.anyio
    async def test_annotations_already_on_the_started_block_must_be_an_array(
        self,
    ) -> None:
        events = _sse(
            _created(),
            _start(
                0,
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": "x", "annotations": "oops"}],
                },
            ),
            _delta(
                0, {"type": "text_annotation_delta", "annotations": [self.ANNOTATION]}
            ),
            _stop(0),
            _completed(),
            _DONE,
        )
        with pytest.raises(OutputContractError, match="annotations are not an array"):
            await _run(_stream(events), self._streaming())

    @pytest.mark.anyio
    async def test_google_search_call_and_result_deltas_fill_the_started_steps(
        self,
    ) -> None:
        events = _sse(
            _created(),
            _start(0, {"type": "google_search_call", "id": "s1", "arguments": {}}),
            _delta(
                0,
                {
                    "type": "google_search_call",
                    "arguments": {"queries": ["nobel prize"]},
                    "signature": "sig",
                },
            ),
            _stop(0),
            _start(1, {"type": "google_search_result", "call_id": "s1", "result": []}),
            _delta(
                1,
                {
                    "type": "google_search_result",
                    "result": [{"search_suggestions": "snippet"}],
                    "is_error": False,
                },
            ),
            _stop(1),
            _start(2, {"type": "model_output"}),
            _delta(2, {"type": "text", "text": "answer"}),
            _stop(2),
            _completed(),
            _DONE,
        )
        result = await _run(_stream(events), self._streaming(tools=self.SEARCH))

        assert result.texts == ("answer",)
        assert result.server_tool_uses is not None
        use = result.server_tool_uses[0]
        assert (use.tool_type, use.tool_use_id) == ("google_search", "s1")
        assert use.input == {"queries": ["nobel prize"]}
        assert use.result == [{"search_suggestions": "snippet"}]

    @pytest.mark.anyio
    async def test_url_context_deltas_yield_a_citation(self) -> None:
        events = _sse(
            _created(),
            _start(0, {"type": "url_context_call", "id": "u1", "arguments": {}}),
            _delta(
                0,
                {
                    "type": "url_context_call",
                    "arguments": {"urls": ["https://example.com"]},
                },
            ),
            _stop(0),
            _start(1, {"type": "url_context_result", "call_id": "u1", "result": []}),
            _delta(
                1,
                {
                    "type": "url_context_result",
                    "result": [{"url": "https://example.com", "status": "success"}],
                },
            ),
            _stop(1),
            _completed(),
            _DONE,
        )
        result = await _run(_stream(events), self._streaming(tools=self.CONTEXT))

        assert result.citations is not None
        assert [c.url for c in result.citations] == ["https://example.com"]
        assert result.server_tool_uses is not None
        assert result.server_tool_uses[0].tool_type == "url_context"

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("delta_type", "extra"),
        [
            ("google_search_call", {"result": []}),
            ("google_search_result", {"arguments": {}}),
            ("url_context_call", {"result": []}),
            ("url_context_result", {"arguments": {}}),
        ],
    )
    async def test_a_whole_value_delta_may_not_set_unrelated_fields(
        self, delta_type: str, extra: dict[str, object]
    ) -> None:
        events = _sse(
            _created(),
            _start(0, {"type": delta_type}),
            _delta(0, {"type": delta_type, **extra}),
            _stop(0),
            _completed(),
            _DONE,
        )
        with pytest.raises(OutputContractError, match="unsupported field"):
            await _run(
                _stream(events),
                self._streaming(
                    tools=ToolParams(
                        hosted=(
                            HostedToolSpec("google_search"),
                            HostedToolSpec("url_context"),
                        )
                    )
                ),
            )

    @pytest.mark.anyio
    async def test_a_function_result_delta_is_not_a_model_output(self) -> None:
        # The model never answers its own call, so this delta has no place in a
        # reply and is refused like any other unsupported type.
        events = _sse(
            _created(),
            _start(0, {"type": "function_result", "call_id": "c1"}),
            _delta(0, {"type": "function_result", "result": "x"}),
            _stop(0),
            _completed(),
            _DONE,
        )
        with pytest.raises(OutputContractError, match="unsupported delta"):
            await _run(
                _stream(events),
                self._streaming(tools=ToolParams(functions=(_FUNCTION_TOOL,))),
            )


class TestRemainingGuards:
    """Small reachable branches left over after the behavioural suites."""

    @pytest.mark.anyio
    async def test_an_annotation_without_a_type_is_ignored(self) -> None:
        # ``Annotation`` is a union; a block that names no kind carries no URL
        # to cite, so it contributes nothing rather than failing the call.
        reply = _reply(
            steps=[
                {
                    "type": "model_output",
                    "content": [
                        {"type": "text", "text": "x", "annotations": [{}]},
                    ],
                }
            ]
        )
        result = await _run(_json_handler(reply))

        assert result.texts == ("x",)
        assert result.citations is None

    @pytest.mark.anyio
    async def test_an_annotation_entry_that_is_not_an_object_fails(self) -> None:
        reply = _reply(
            steps=[
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": "x", "annotations": ["u"]}],
                }
            ]
        )
        with pytest.raises(
            OutputContractError, match=r"annotations\[0\] must be an object"
        ):
            await _run(_json_handler(reply))

    @pytest.mark.anyio
    async def test_arguments_already_present_where_a_delta_arrives_fails(self) -> None:
        # ``step.start`` normally seeds ``arguments`` as ``{}``; a non-empty
        # object means the delta would overwrite real arguments, not extend them.
        events = _sse(
            _created(),
            _start(
                0,
                {
                    "type": "function_call",
                    "id": "c1",
                    "name": "get_weather",
                    "arguments": {"city": "Paris"},
                },
            ),
            _delta(0, {"type": "arguments_delta", "arguments": '{"city":"Rome"}'}),
            _stop(0),
            _completed(status="requires_action"),
            _DONE,
        )
        with pytest.raises(OutputContractError, match="accumulated function arguments"):
            await _run(
                _stream(events),
                _function_request(scheduling=SchedulingParams(stream=True)),
            )

    @pytest.mark.anyio
    async def test_a_streamed_reply_that_requires_action_without_tools_fails(
        self,
    ) -> None:
        events = _sse(_created(), _completed(status="requires_action"), _DONE)
        with pytest.raises(OutputContractError, match="requires an action"):
            await _run(
                _stream(events), _request(scheduling=SchedulingParams(stream=True))
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("step", "message"),
        [
            pytest.param(
                {
                    "type": "function_call",
                    "id": "",
                    "name": "get_weather",
                    "arguments": {},
                },
                "id must be a non-empty string",
                id="empty-call-id",
            ),
            pytest.param(
                {"type": "function_call", "id": "c", "name": "", "arguments": {}},
                "name must be a non-empty string",
                id="empty-call-name",
            ),
            pytest.param(
                # A string is treated as accumulated JSON text and fails to
                # decode first; a number reaches the type check itself.
                {
                    "type": "function_call",
                    "id": "c",
                    "name": "get_weather",
                    "arguments": 5,
                },
                "arguments must be an object",
                id="call-arguments-not-an-object",
            ),
            pytest.param(
                {"type": "google_search_call", "id": "", "arguments": {}},
                "id must be a non-empty string",
                id="empty-hosted-id",
            ),
            pytest.param(
                {"type": "google_search_call", "id": "s", "arguments": 5},
                "arguments must be an object",
                id="hosted-arguments-not-an-object",
            ),
        ],
    )
    async def test_call_step_field_rejections(
        self, step: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(OutputContractError, match=message):
            await _run(
                _json_handler(_reply(status="requires_action", steps=[step])),
                _function_request(
                    tools=ToolParams(
                        functions=(_FUNCTION_TOOL,),
                        hosted=(HostedToolSpec("google_search"),),
                    )
                ),
            )


class TestStreamTerminalEvidence:
    """The terminal snapshot is evidence; the deltas are another; they must agree."""

    def _streaming(self, **changes: object) -> Request:
        return _request(scheduling=SchedulingParams(stream=True), **changes)

    @pytest.mark.anyio
    @pytest.mark.parametrize("summary", [None, "auto"])
    @pytest.mark.parametrize(
        ("streamed_thought", "streamed_summary", "snapshot_summary", "field"),
        [
            (True, None, "terminal summary", "summaries"),
            (True, "stream summary", "terminal summary", "summaries"),
            (True, "stream summary", None, "summaries"),
            (False, None, "terminal summary", "summaries"),
            (False, None, None, "saw_thought"),
            (True, "same summary", "same summary", None),
        ],
    )
    async def test_terminal_thought_evidence_matches_the_stream(
        self,
        summary: Literal["auto"] | None,
        streamed_thought: bool,
        streamed_summary: str | None,
        snapshot_summary: str | None,
        field: str | None,
    ) -> None:
        events: list[object] = [_created()]
        if streamed_thought:
            events.append(_start(0, {"type": "thought"}))
            if streamed_summary is not None:
                events.append(
                    _delta(
                        0,
                        {
                            "type": "thought_summary",
                            "content": {"type": "text", "text": streamed_summary},
                        },
                    )
                )
            events.append(_stop(0))
        index = 1 if streamed_thought else 0
        events.extend(
            [
                _start(index, {"type": "model_output"}),
                _delta(index, {"type": "text", "text": "ok"}),
                _stop(index),
                _completed(
                    steps=[
                        _thought([snapshot_summary] if snapshot_summary else None),
                        _model_output("ok"),
                    ]
                ),
                _DONE,
            ]
        )
        request = self._streaming(reasoning=ReasoningParams(summary=summary))
        if field is not None:
            with pytest.raises(OutputContractError, match=f"disagree.*{field!r}"):
                await _run(_stream(_sse(*events)), request)
        else:
            result = await _run(_stream(_sse(*events)), request)
            assert result.reasoning is not None
            assert result.reasoning[0] is not None
            assert result.reasoning[0].text == "same summary"

    @pytest.mark.anyio
    async def test_snapshot_only_thought_summary_is_preserved(self) -> None:
        events = _sse(
            _created(),
            _completed(steps=[_thought(["summary"]), _model_output("ok")]),
            _DONE,
        )
        result = await _run(
            _stream(events), self._streaming(reasoning=ReasoningParams(summary="auto"))
        )
        assert result.reasoning is not None
        assert result.reasoning[0] is not None
        assert result.reasoning[0].text == "summary"

    @pytest.mark.anyio
    async def test_a_snapshot_without_deltas_is_the_whole_reply(self) -> None:
        events = _sse(
            _created(),
            _completed(
                steps=[
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": "VISIBLE"}],
                    }
                ]
            ),
            _DONE,
        )
        result = await _run(_stream(events), self._streaming())
        assert result.texts == ("VISIBLE",)

    @pytest.mark.anyio
    @pytest.mark.parametrize("include_snapshot", [False, True])
    @pytest.mark.parametrize(
        ("delta_texts", "snapshot_texts"),
        [
            (("Hel", "lo"), ("Hel", "lo")),
            (("Hel", "lo"), ("Hello",)),
            (("Hello",), ("", "Hello", "")),
            (
                ("\n  分析", "\t🙂e\u0301\r\n结束  "),
                ("\n", "  分析\t🙂e\u0301\r\n", "结束  "),
            ),
        ],
    )
    async def test_terminal_text_projection_is_independent_of_content_segmentation(
        self,
        include_snapshot: bool,
        delta_texts: tuple[str, ...],
        snapshot_texts: tuple[str, ...],
    ) -> None:
        events: list[object] = [_created(), _start(0, {"type": "model_output"})]
        events.extend(_delta(0, {"type": "text", "text": text}) for text in delta_texts)
        events.append(_stop(0))
        terminal = (
            _completed(steps=[_model_output(*snapshot_texts)])
            if include_snapshot
            else _completed()
        )
        events.extend((terminal, _DONE))

        result = await _run(_stream(_sse(*events)), self._streaming())

        assert result.texts == ("".join(delta_texts),)
        assert result.texts == ("".join(snapshot_texts),)

    @pytest.mark.anyio
    @pytest.mark.parametrize("include_snapshot", [False, True])
    @pytest.mark.parametrize(
        ("streamed_summary", "snapshot_summary"),
        [
            (("first\nsecond",), ("first\n", "second")),
            (("first\n", "second"), ("first\nsecond",)),
            (("  分析\t🙂\n", "结论  "), ("  分析\t🙂\n结论  ",)),
            (("Hello ", "world"), ("Hello world",)),
            (("", "summary", ""), ("summary",)),
            (
                (
                    "I",
                    " ",
                    "computed",
                    " ",
                    "17×23",
                    ".",
                    " ",
                    "Result",
                    ":",
                    " ",
                    "391",
                    ".",
                ),
                ("I computed 17×23. Result: 391.",),
            ),
        ],
    )
    async def test_terminal_summary_projection_is_independent_of_content_segmentation(
        self,
        include_snapshot: bool,
        streamed_summary: tuple[str, ...],
        snapshot_summary: tuple[str, ...],
    ) -> None:
        events: list[object] = [_created(), _start(0, {"type": "thought"})]
        events.extend(
            _delta(
                0,
                {
                    "type": "thought_summary",
                    "content": {"type": "text", "text": text},
                },
            )
            for text in streamed_summary
        )
        events.extend(
            (
                _stop(0),
                _start(1, {"type": "model_output"}),
                _delta(1, {"type": "text", "text": "ok"}),
                _stop(1),
            )
        )
        terminal = (
            _completed(steps=[_thought(list(snapshot_summary)), _model_output("ok")])
            if include_snapshot
            else _completed()
        )
        events.extend((terminal, _DONE))

        result = await _run(
            _stream(_sse(*events)),
            self._streaming(reasoning=ReasoningParams(summary="auto")),
        )

        assert result.texts == ("ok",)
        assert result.reasoning is not None
        assert result.reasoning[0] is not None
        assert result.reasoning[0].text == "".join(streamed_summary)
        assert result.reasoning[0].text == "".join(snapshot_summary)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("channel", "streamed_parts", "snapshot_parts"),
        [
            ("texts", ("Hel", "lo"), ("Hallo",)),
            ("texts", ("Hello",), ("Hello ",)),
            ("texts", ("\nHello",), ("Hello",)),
            ("texts", ("e\u0301",), ("é",)),
            ("summaries", ("Hello ", "world"), ("Hello \nworld",)),
            ("summaries", ("first", "second"), ("first\nsecond",)),
            ("summaries", ("first", "second"), ("second", "first")),
            ("summaries", ("first\nsecond",), ("first\r\nsecond",)),
            ("summaries", ("summary",), (" summary",)),
        ],
    )
    async def test_terminal_projection_rejects_changed_content_without_normalization(
        self,
        channel: Literal["texts", "summaries"],
        streamed_parts: tuple[str, ...],
        snapshot_parts: tuple[str, ...],
    ) -> None:
        is_summary = channel == "summaries"
        events: list[object] = [
            _created(),
            _start(0, {"type": "thought" if is_summary else "model_output"}),
        ]
        for text in streamed_parts:
            delta = (
                {"type": "thought_summary", "content": {"type": "text", "text": text}}
                if is_summary
                else {"type": "text", "text": text}
            )
            events.append(_delta(0, delta))
        events.append(_stop(0))
        if is_summary:
            events.extend(
                (
                    _start(1, {"type": "model_output"}),
                    _delta(1, {"type": "text", "text": "ok"}),
                    _stop(1),
                )
            )
            snapshot_steps = [_thought(list(snapshot_parts)), _model_output("ok")]
        else:
            snapshot_steps = [_model_output(*snapshot_parts)]
        events.extend((_completed(steps=snapshot_steps), _DONE))

        with pytest.raises(OutputContractError, match=f"disagree.*{channel!r}"):
            await _run(
                _stream(_sse(*events)),
                self._streaming(
                    reasoning=ReasoningParams(summary="auto" if is_summary else None)
                ),
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("delta_text", "snapshot_text", "field"),
        [
            ("OLD", "NEW", "texts"),
            ("same", "same", None),
        ],
    )
    async def test_a_snapshot_that_contradicts_the_deltas_fails(
        self, delta_text: str, snapshot_text: str, field: str | None
    ) -> None:
        events = _sse(
            _created(),
            _start(0, {"type": "model_output"}),
            _delta(0, {"type": "text", "text": delta_text}),
            _stop(0),
            _completed(
                steps=[
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": snapshot_text}],
                    }
                ]
            ),
            _DONE,
        )
        if field is None:
            result = await _run(_stream(events), self._streaming())
            assert result.texts == (delta_text,)
        else:
            with pytest.raises(OutputContractError, match=f"disagree.*{field!r}"):
                await _run(_stream(events), self._streaming())

    @pytest.mark.anyio
    async def test_a_snapshot_that_disagrees_on_tool_calls_fails(self) -> None:
        events = _sse(
            _created(),
            _start(
                0,
                {
                    "type": "function_call",
                    "id": "a",
                    "name": "get_weather",
                    "arguments": {},
                },
            ),
            _delta(0, {"type": "arguments_delta", "arguments": '{"city":"Paris"}'}),
            _stop(0),
            _completed(
                status="requires_action",
                steps=[
                    {
                        "type": "function_call",
                        "id": "b",
                        "name": "get_weather",
                        "arguments": {"city": "Rome"},
                    }
                ],
            ),
            _DONE,
        )
        with pytest.raises(OutputContractError, match="disagree.*'tool_calls'"):
            await _run(
                _stream(events),
                self._streaming(tools=ToolParams(functions=(_FUNCTION_TOOL,))),
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "snapshot_url",
        ["https://example.test/source", "https://example.test/changed", None],
    )
    async def test_resegmented_text_keeps_citation_comparison_strict(
        self, snapshot_url: str | None
    ) -> None:
        url = "https://example.test/source"
        snapshot = _model_output("Par", "is")
        if snapshot_url is not None:
            snapshot["content"][1]["annotations"] = [
                {"type": "url_citation", "url": snapshot_url}
            ]
        events = _sse(
            _created(),
            _start(0, {"type": "model_output"}),
            _delta(0, {"type": "text", "text": "Par"}),
            _delta(0, {"type": "text", "text": "is"}),
            _delta(
                0,
                {
                    "type": "text_annotation_delta",
                    "annotations": [{"type": "url_citation", "url": url}],
                },
            ),
            _stop(0),
            _completed(steps=[snapshot]),
            _DONE,
        )

        if snapshot_url == url:
            result = await _run(_stream(events), self._streaming())
            assert result.texts == ("Paris",)
            assert result.citations is not None
            assert [citation.url for citation in result.citations] == [url]
        else:
            with pytest.raises(OutputContractError, match="disagree.*'citations'"):
                await _run(_stream(events), self._streaming())

    @pytest.mark.anyio
    async def test_a_snapshot_that_disagrees_on_hosted_results_fails(self) -> None:
        events = _sse(
            _created(),
            _start(0, {"type": "google_search_call", "id": "s1", "arguments": {}}),
            _delta(0, {"type": "google_search_call", "arguments": {"queries": ["q"]}}),
            _stop(0),
            _start(1, {"type": "google_search_result", "call_id": "s1"}),
            _delta(
                1,
                {
                    "type": "google_search_result",
                    "result": [{"search_suggestions": "delta"}],
                },
            ),
            _stop(1),
            _completed(
                steps=[
                    {
                        "type": "google_search_call",
                        "id": "s1",
                        "arguments": {"queries": ["q"]},
                    },
                    {
                        "type": "google_search_result",
                        "call_id": "s1",
                        "result": [{"search_suggestions": "snapshot"}],
                    },
                ]
            ),
            _DONE,
        )
        with pytest.raises(OutputContractError, match="disagree.*'server_results'"):
            await _run(
                _stream(events),
                self._streaming(
                    tools=ToolParams(hosted=(HostedToolSpec("google_search"),))
                ),
            )

    @pytest.mark.anyio
    async def test_terminal_steps_must_be_an_array(self) -> None:
        events = _sse(_created(), _completed(steps="x"), _DONE)
        with pytest.raises(OutputContractError, match="steps must be an array"):
            await _run(_stream(events), self._streaming())

    @pytest.mark.anyio
    async def test_two_terminal_events_fail(self) -> None:
        events = _sse(
            _created(),
            _completed(usage={"total_input_tokens": 5, "total_output_tokens": 5}),
            _completed(usage={"total_input_tokens": 99, "total_output_tokens": 99}),
            _DONE,
        )
        with pytest.raises(OutputContractError, match="completed twice"):
            await _run(_stream(events), self._streaming())

    @pytest.mark.anyio
    async def test_step_stop_usage_is_kept_when_the_terminal_omits_it(self) -> None:
        events = _sse(
            _created(),
            _start(0, {"type": "model_output"}),
            _delta(0, {"type": "text", "text": "hi"}),
            {
                **_stop(0),
                "usage": {
                    "total_input_tokens": 7,
                    "total_output_tokens": 3,
                    "total_thought_tokens": 1,
                },
            },
            # A terminal event that omits ``usage`` entirely (the helper adds a
            # default one, so this event is built by hand).
            {
                "event_type": "interaction.completed",
                "interaction": {
                    "object": "interaction",
                    "model": MODEL,
                    "status": "completed",
                },
            },
            _DONE,
        )
        result = await _run(_stream(events), self._streaming())

        assert result.usage == UsageStats(
            input_tokens=7,
            output_tokens=4,
            total_tokens=11,
            reasoning_tokens=1,
        )

    @pytest.mark.anyio
    async def test_the_terminal_usage_wins_over_step_stop(self) -> None:
        events = _sse(
            _created(),
            _start(0, {"type": "model_output"}),
            {
                **_stop(0),
                "usage": {"total_input_tokens": 1, "total_output_tokens": 1},
            },
            _completed(usage={"total_input_tokens": 9, "total_output_tokens": 9}),
            _DONE,
        )
        result = await _run(_stream(events), self._streaming())

        assert result.usage is not None
        assert result.usage.input_tokens == 9

    @pytest.mark.anyio
    async def test_steps_are_reported_in_index_order_not_stop_order(self) -> None:
        events = _sse(
            _created(),
            _start(0, {"type": "model_output"}),
            _delta(0, {"type": "text", "text": "first"}),
            _start(1, {"type": "model_output"}),
            _delta(1, {"type": "text", "text": "second"}),
            _stop(1),
            _stop(0),
            _completed(),
            _DONE,
        )
        result = await _run(_stream(events), self._streaming())
        assert result.texts == ("firstsecond",)


class TestStreamDeltaUsage:
    @staticmethod
    async def run_events(
        *events: dict[str, Any], terminal_usage: dict[str, int] | None = None
    ) -> Response:
        terminal = _completed()
        if terminal_usage is None:
            terminal["interaction"].pop("usage")
        else:
            terminal["interaction"]["usage"] = terminal_usage
        return await _run(
            _stream(_sse(_created(), *events, terminal, _DONE)),
            _request(scheduling=SchedulingParams(stream=True)),
        )

    @staticmethod
    def delta(usage: dict[str, Any], index: int = 0) -> dict[str, Any]:
        return {
            **_delta(index, {"type": "text", "text": "hi"}),
            "metadata": {"total_usage": usage},
        }

    @pytest.mark.anyio
    @pytest.mark.parametrize("reported_total", [12, 99])
    async def test_delta_usage_survives_without_stop_or_terminal_usage(
        self, reported_total: int
    ) -> None:
        result = await self.run_events(
            _start(0, {"type": "model_output"}),
            self.delta(
                {
                    "total_input_tokens": 7,
                    "total_output_tokens": 3,
                    "total_tool_use_tokens": 1,
                    "total_thought_tokens": 1,
                    "total_cached_tokens": 2,
                    "total_tokens": reported_total,
                }
            ),
            _stop(0),
        )
        assert result.texts == ("hi",)
        assert result.usage == UsageStats(
            input_tokens=8,
            output_tokens=4,
            total_tokens=12,
            reasoning_tokens=1,
            cached_tokens=2,
            reported_total_tokens=None if reported_total == 12 else reported_total,
        )

    @pytest.mark.anyio
    async def test_latest_cumulative_delta_replaces_instead_of_adds(self) -> None:
        result = await self.run_events(
            _start(0, {"type": "model_output"}),
            self.delta({"total_input_tokens": 7, "total_output_tokens": 2}),
            self.delta({"total_input_tokens": 7, "total_output_tokens": 3}),
            _stop(0),
        )
        assert result.usage == UsageStats(
            input_tokens=7, output_tokens=3, total_tokens=10
        )

    @pytest.mark.anyio
    @pytest.mark.parametrize("explicit_zeroes", [False, True])
    async def test_empty_or_zero_counts_are_valid(self, explicit_zeroes: bool) -> None:
        usage = (
            {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tool_use_tokens": 0,
                "total_thought_tokens": 0,
                "total_cached_tokens": 0,
                "total_tokens": 0,
            }
            if explicit_zeroes
            else {}
        )
        result = await self.run_events(
            _start(0, {"type": "model_output"}),
            self.delta(usage),
            _stop(0),
        )
        expected = (
            UsageStats(reasoning_tokens=0, cached_tokens=0)
            if explicit_zeroes
            else UsageStats()
        )
        assert result.usage == expected

    @pytest.mark.anyio
    @pytest.mark.parametrize("metadata", [_DROP, {}])
    async def test_missing_or_empty_metadata_keeps_previous_usage(
        self, metadata: Any
    ) -> None:
        delta = _delta(0, {"type": "text", "text": "more"})
        if metadata is not _DROP:
            delta["metadata"] = metadata
        result = await self.run_events(
            _start(0, {"type": "model_output"}),
            self.delta({"total_input_tokens": 7, "total_output_tokens": 3}),
            delta,
            _stop(0),
        )
        assert result.usage == UsageStats(
            input_tokens=7, output_tokens=3, total_tokens=10
        )

    @pytest.mark.anyio
    @pytest.mark.parametrize("terminal_present", [False, True])
    async def test_stop_and_terminal_usage_keep_their_precedence(
        self, terminal_present: bool
    ) -> None:
        result = await self.run_events(
            _start(0, {"type": "model_output"}),
            self.delta({"total_input_tokens": 1, "total_output_tokens": 1}),
            {
                **_stop(0),
                "usage": {"total_input_tokens": 7, "total_output_tokens": 3},
            },
            terminal_usage=(
                {"total_input_tokens": 9, "total_output_tokens": 4}
                if terminal_present
                else None
            ),
        )
        expected = (
            UsageStats(input_tokens=9, output_tokens=4, total_tokens=13)
            if terminal_present
            else UsageStats(input_tokens=7, output_tokens=3, total_tokens=10)
        )
        assert result.usage == expected

    @pytest.mark.anyio
    async def test_delta_after_another_steps_stop_is_the_latest_usage(self) -> None:
        result = await self.run_events(
            _start(0, {"type": "model_output"}),
            _start(1, {"type": "model_output"}),
            {
                **_stop(0),
                "usage": {"total_input_tokens": 7, "total_output_tokens": 2},
            },
            self.delta({"total_input_tokens": 7, "total_output_tokens": 3}, index=1),
            _stop(1),
        )
        assert result.usage == UsageStats(
            input_tokens=7, output_tokens=3, total_tokens=10
        )

    @pytest.mark.anyio
    async def test_step_usage_is_not_a_cumulative_fallback(self) -> None:
        result = await self.run_events(
            _start(0, {"type": "model_output"}),
            self.delta({"total_input_tokens": 7, "total_output_tokens": 3}),
            {**_stop(0), "step_usage": {"total_output_tokens": 1}},
        )
        assert result.usage == UsageStats(
            input_tokens=7, output_tokens=3, total_tokens=10
        )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "metadata",
        [
            None,
            True,
            0,
            "",
            [],
            {"total_usage": None},
            {"total_usage": True},
            {"total_usage": 0},
            {"total_usage": ""},
            {"total_usage": []},
        ],
    )
    async def test_invalid_metadata_is_not_hidden_by_terminal_usage(
        self, metadata: Any
    ) -> None:
        with pytest.raises(OutputContractError, match=r"step\.delta\.metadata"):
            await self.run_events(
                _start(0, {"type": "model_output"}),
                {**_delta(0, {"type": "text", "text": "hi"}), "metadata": metadata},
                _stop(0),
                terminal_usage={"total_input_tokens": 7, "total_output_tokens": 3},
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "name",
        [
            "total_input_tokens",
            "total_output_tokens",
            "total_tool_use_tokens",
            "total_thought_tokens",
            "total_cached_tokens",
            "total_tokens",
        ],
    )
    @pytest.mark.parametrize("value", [None, True, -1, 1.5, "1", [], {}])
    async def test_invalid_counts_fail_even_with_a_valid_terminal(
        self, name: str, value: Any
    ) -> None:
        with pytest.raises(
            OutputContractError,
            match=rf"step\.delta\.metadata\.total_usage: usage\.{name}",
        ):
            await self.run_events(
                _start(0, {"type": "model_output"}),
                self.delta({name: value}),
                _stop(0),
                terminal_usage={"total_input_tokens": 7, "total_output_tokens": 3},
            )


class TestHostedResultEvidence:
    """Failures and duplicate/ill-typed results must survive or be refused."""

    TOOLS = ToolParams(
        hosted=(HostedToolSpec("google_search"), HostedToolSpec("url_context"))
    )

    def _reply(self, *, is_error: bool) -> dict[str, Any]:
        return _reply(
            steps=[
                {
                    "type": "google_search_call",
                    "id": "s1",
                    "arguments": {"queries": ["q"]},
                },
                {
                    "type": "google_search_result",
                    "call_id": "s1",
                    "is_error": is_error,
                    "result": [{"search_suggestions": "x"}],
                },
                _model_output("answer"),
            ]
        )

    @pytest.mark.anyio
    async def test_a_failure_and_a_success_do_not_persist_identically(self) -> None:
        failed = await _run(
            _json_handler(self._reply(is_error=True)), _request(tools=self.TOOLS)
        )
        passed = await _run(
            _json_handler(self._reply(is_error=False)), _request(tools=self.TOOLS)
        )

        assert (
            failed.server_tool_uses is not None and passed.server_tool_uses is not None
        )
        assert failed.server_tool_uses[0].error_code is not None
        assert passed.server_tool_uses[0].error_code is None
        assert failed.server_tool_uses[0].result == passed.server_tool_uses[0].result
        assert json.dumps(
            failed.server_tool_uses[0].__dict__, sort_keys=True, default=str
        ) != (
            json.dumps(passed.server_tool_uses[0].__dict__, sort_keys=True, default=str)
        )

    @pytest.mark.anyio
    async def test_a_result_must_match_the_kind_of_its_call(self) -> None:
        reply = _reply(
            steps=[
                {
                    "type": "google_search_call",
                    "id": "s1",
                    "arguments": {"queries": ["q"]},
                },
                {
                    "type": "url_context_result",
                    "call_id": "s1",
                    "result": [{"url": "https://x.test"}],
                },
                _model_output("answer"),
            ]
        )
        with pytest.raises(
            OutputContractError, match="is a url_context result for a google_search"
        ):
            await _run(_json_handler(reply), _request(tools=self.TOOLS))

    @pytest.mark.anyio
    async def test_two_results_for_one_call_fail(self) -> None:
        reply = _reply(
            steps=[
                {
                    "type": "google_search_call",
                    "id": "s1",
                    "arguments": {"queries": ["q"]},
                },
                {
                    "type": "google_search_result",
                    "call_id": "s1",
                    "result": [{"search_suggestions": "a"}],
                },
                {
                    "type": "google_search_result",
                    "call_id": "s1",
                    "result": [{"search_suggestions": "b"}],
                },
                _model_output("answer"),
            ]
        )
        with pytest.raises(OutputContractError, match="second google_search result"):
            await _run(_json_handler(reply), _request(tools=self.TOOLS))

    @pytest.mark.anyio
    async def test_two_calls_with_one_id_fail(self) -> None:
        reply = _reply(
            steps=[
                {
                    "type": "google_search_call",
                    "id": "s1",
                    "arguments": {"queries": ["q"]},
                },
                {
                    "type": "google_search_call",
                    "id": "s1",
                    "arguments": {"queries": ["r"]},
                },
                _model_output("answer"),
            ]
        )
        with pytest.raises(OutputContractError, match="hosted tool call id 's1' twice"):
            await _run(_json_handler(reply), _request(tools=self.TOOLS))


class TestStatelessRetentionContract:
    @pytest.mark.anyio
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("options", [None, _options(), _options(store=False)])
    async def test_store_false_is_sent_and_recorded(
        self, stream: bool, options: DialectOptions | None
    ) -> None:
        sent: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            if stream:
                return httpx.Response(200, content=_recorded_stream("stream_text"))
            return httpx.Response(200, json=_reply(id="not-a-session"))

        result = await _run(
            handler,
            _request(
                dialect_options=options, scheduling=SchedulingParams(stream=stream)
            ),
        )
        assert sent[0]["store"] is False
        assert result.request_params is not None
        body = result.request_params["body"]
        assert isinstance(body, dict)
        assert body["store"] is False
        assert result.session_id is None

    @pytest.mark.anyio
    @pytest.mark.parametrize("value", [True, None, 0, 1, "false", [], {}])
    async def test_store_cannot_be_enabled_or_forged(self, value: object) -> None:
        with pytest.raises(RequestAuditError, match="store"):
            await _run(_no_io, _request(dialect_options=_options(store=value)))

    @pytest.mark.anyio
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        "options", [None, _options(store=False), _options(store=True)]
    )
    @pytest.mark.parametrize("previous_id", ["", "previous"])
    async def test_chaining_is_rejected_for_all_storage_options(
        self, stream: bool, options: DialectOptions | None, previous_id: str
    ) -> None:
        req = _request(
            dialect_options=options,
            session=SessionParams(previous_response_id=previous_id),
            scheduling=SchedulingParams(stream=stream),
        )
        message = "stateless" if previous_id else "must not be empty"
        with pytest.raises(RequestAuditError, match=message):
            _audit(req).raise_rejections()
        with pytest.raises(
            DialectError, match="unavailable capability.*stateful_session"
        ):
            await _run(_no_io, req)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("key", "value"),
        [
            *(("store", value) for value in (_DROP, True, None, 0, 1, "false")),
            ("previous_interaction_id", "previous"),
            ("background", True),
            ("background", False),
        ],
    )
    @pytest.mark.parametrize("stage", ["audit", "final_http"])
    async def test_stateless_retention_is_pinned_at_both_request_boundaries(
        self, monkeypatch: pytest.MonkeyPatch, key: str, value: object, stage: str
    ) -> None:
        def change(body: dict[str, Any]) -> None:
            if value is _DROP:
                body.pop(key, None)
            else:
                body[key] = value

        if stage == "audit":
            _tamper_prepared(monkeypatch, change)
            with pytest.raises(RequestAuditError, match=key):
                await _run(_no_io)
        else:
            connection = _connection(_no_io)
            dialect = GoogleInteractionsDialect(connection, MODEL)
            request = _request()
            audit = _audit(request)
            prepared = dialect.prepare(request, audit)
            build_request = connection.client.build_request

            def tampered_request(*args: Any, **kwargs: Any) -> httpx.Request:
                body = copy.deepcopy(kwargs["json"])
                change(body)
                kwargs["json"] = body
                return build_request(*args, **kwargs)

            monkeypatch.setattr(connection.client, "build_request", tampered_request)
            try:
                with pytest.raises(DialectError, match=key):
                    await dialect.execute(prepared)
            finally:
                await connection.aclose()


class TestToolResultContentFields:
    """Every sub-field of a content block, not just its main payload."""

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            (
                [{"type": "text", "text": "ok", "annotations": 7}],
                "annotations must be an array",
            ),
            (
                [{"type": "text", "text": "ok", "annotations": [7]}],
                "annotations[0] must be an object",
            ),
            (
                [
                    {
                        "type": "text",
                        "text": "ok",
                        "annotations": [
                            {"type": "url_citation", "url": "u", "start_index": -1}
                        ],
                    }
                ],
                "start_index must be a non-negative integer",
            ),
            (
                [
                    {
                        "type": "text",
                        "text": "ok",
                        "annotations": [{"type": "file_citation"}],
                    }
                ],
                "citation type 'file_citation'",
            ),
            (
                [{"type": "image", "uri": "https://x.test/i", "resolution": "garbage"}],
                "resolution must be one of",
            ),
            (
                [{"type": "image", "data": "aGk=", "resolution": "ultra_high"}],
                None,
            ),
            (
                [
                    {
                        "type": "text",
                        "text": "ok",
                        "annotations": [{"type": "url_citation", "url": "u"}],
                    }
                ],
                None,
            ),
        ],
    )
    def test_content_subfields_are_checked(
        self, value: object, message: str | None
    ) -> None:
        error = google_interactions._tool_result_error(cast(Any, value))
        assert (error is None) is (message is None)
        if message is not None:
            assert error is not None and message in error

    @pytest.mark.anyio
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        ("item", "message"),
        [
            pytest.param(
                {"type": "text", "text": "ok", "annotations": None},
                "function result[0].annotations must be an array",
                id="annotations-null",
            ),
            pytest.param(
                {"type": "text", "text": "ok", "annotations": [None]},
                "function result[0].annotations[0] must be an object",
                id="annotation-null",
            ),
            pytest.param(
                {"type": "text", "text": "ok", "annotations": [{}]},
                "function result[0].annotations[0].type is required",
                id="annotation-missing-type",
            ),
            *(
                pytest.param(
                    {
                        "type": "text",
                        "text": "ok",
                        "annotations": [{"type": value}],
                    },
                    "function result[0].annotations[0] has citation type",
                    id=f"annotation-type-{value!r}",
                )
                for value in (None, 7, [], {}, "file_citation")
            ),
            *(
                pytest.param(
                    {
                        "type": "text",
                        "text": "ok",
                        "annotations": [{"type": "url_citation", field: value}],
                    },
                    f"function result[0].annotations[0].{field} must be a string",
                    id=f"annotation-{field}-{value!r}",
                )
                for field in ("url", "title")
                for value in (None, 7, False, [], {})
            ),
            *(
                pytest.param(
                    {
                        "type": "text",
                        "text": "ok",
                        "annotations": [{"type": "url_citation", field: value}],
                    },
                    f"function result[0].annotations[0].{field} must be "
                    "a non-negative integer",
                    id=f"annotation-{field}-{value!r}",
                )
                for field in ("start_index", "end_index")
                for value in (None, False, -1, 0.5, "0", [], {})
            ),
            *(
                pytest.param(
                    {"type": "image", opposite: source, field: value},
                    f"function result[0].{field} must be a string",
                    id=f"image-{field}-{value!r}",
                )
                for field, opposite, source in (
                    ("data", "uri", "https://example.com/i.png"),
                    ("uri", "data", "aGk="),
                )
                for value in (None, 0, False, [], {})
            ),
            *(
                pytest.param(
                    {"type": "image", "uri": "https://example.com/i.png", field: value},
                    f"function result[0].{field}",
                    id=f"image-{field}-{value!r}",
                )
                for field in ("mime_type", "resolution")
                for value in (None, 7, False, [], {}, "garbage")
            ),
        ],
    )
    async def test_invalid_content_is_rejected_before_http(
        self, item: dict[str, Any], message: str, stream: bool
    ) -> None:
        sent: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(request)
            raise AssertionError("invalid tool results must not reach HTTP")

        tool_result = ToolResultPart("c1", [item])
        messages = [
            _say("user", "go"),
            ChatMessage("assistant", (ToolCallPart("c1", "get_weather", {}),)),
        ]
        messages.append(ChatMessage("tool", (tool_result,)))
        request = _request(
            input=_chat(*messages),
            tools=ToolParams(functions=(_FUNCTION_TOOL,)),
            scheduling=SchedulingParams(stream=stream),
        )
        with pytest.raises(RequestAuditError) as raised:
            await _run(handler, request)
        assert message in str(raised.value)
        assert sent == []
        with pytest.raises(DialectError) as lowering_error:
            google_interactions._function_result_step(tool_result)
        assert message in str(lowering_error.value)

    @pytest.mark.anyio
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param([{"type": "text", "text": "ok"}], id="annotations-omitted"),
            pytest.param(
                [{"type": "text", "text": "ok", "annotations": []}],
                id="annotations-empty",
            ),
            pytest.param(
                [
                    {
                        "type": "text",
                        "text": "ok",
                        "annotations": [{"type": "url_citation"}],
                    }
                ],
                id="annotation-optional-fields-omitted",
            ),
            pytest.param(
                [
                    {
                        "type": "text",
                        "text": "ok",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "",
                                "title": "",
                                "start_index": 0,
                                "end_index": 0,
                            }
                        ],
                    }
                ],
                id="annotation-empty-strings-and-zero-indices",
            ),
            pytest.param(
                [
                    {
                        "type": "text",
                        "text": "ok",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://example.com",
                                "title": "source",
                                "start_index": 0,
                                "end_index": 2,
                            }
                        ],
                    }
                ],
                id="annotation-all-fields",
            ),
            pytest.param([{"type": "image", "data": "aGk="}], id="image-data-only"),
            pytest.param(
                [{"type": "image", "uri": "https://example.com/i.png"}],
                id="image-uri-only",
            ),
            *(
                pytest.param(
                    [
                        {
                            "type": "image",
                            "uri": "https://example.com/i.png",
                            "mime_type": "image/png",
                            "resolution": resolution,
                        }
                    ],
                    id=f"image-resolution-{resolution}",
                )
                for resolution in ("low", "medium", "high", "ultra_high")
            ),
            pytest.param(
                [{"type": "image", "uri": "https://example.com/i.png", "data": ""}],
                id="image-unused-empty-string-source",
            ),
            pytest.param(
                {"annotations": None, "data": 0, "nested": [1, None, False]},
                id="arbitrary-json-object",
            ),
            pytest.param([], id="empty-content-array"),
        ],
    )
    async def test_valid_content_is_forwarded_without_changes(
        self, value: Any, stream: bool
    ) -> None:
        original = copy.deepcopy(value)
        sent: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            if stream:
                return httpx.Response(
                    200,
                    content=_sse(
                        _created(),
                        _start(0, {"type": "model_output"}),
                        _delta(0, {"type": "text", "text": "ok"}),
                        _stop(0),
                        _completed(id="next"),
                        _DONE,
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(
                200, json=_reply(id="next", steps=[_model_output("ok")])
            )

        messages = [
            _say("user", "go"),
            ChatMessage("assistant", (ToolCallPart("c1", "get_weather", {}),)),
        ]
        messages.append(ChatMessage("tool", (ToolResultPart("c1", value),)))
        request = _request(
            input=_chat(*messages),
            tools=ToolParams(functions=(_FUNCTION_TOOL,)),
            scheduling=SchedulingParams(stream=stream),
        )
        response = await _run(handler, request)
        assert response.texts == ("ok",)
        assert len(sent) == 1
        assert sent[0]["input"][-1] == {
            "type": "function_result",
            "call_id": "c1",
            "result": original,
            "name": "get_weather",
        }
        assert value == original
        assert "previous_interaction_id" not in sent[0]
        assert sent[0]["store"] is False
        assert response.session_id is None
        assert sent[0].get("stream", False) is stream

    def test_input_annotations_do_not_use_response_citation_converters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail_if_called(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("response citation converters must not validate input")

        monkeypatch.setattr(google_interactions, "_annotation", fail_if_called)
        monkeypatch.setattr(google_interactions, "_check_span", fail_if_called)
        result = [
            {
                "type": "text",
                "text": "ok",
                "annotations": [{"type": "url_citation", "start_index": 0}],
            }
        ]
        assert google_interactions._tool_result_error(result) is None


class TestOpaqueContinuation:
    @staticmethod
    def reasoning(response: Response) -> ReasoningOutput:
        assert response.reasoning is not None
        reasoning = response.reasoning[0]
        assert reasoning is not None
        return reasoning

    @classmethod
    def opaque(cls, response: Response) -> str:
        opaque = cls.reasoning(response).opaque_roundtrip
        assert isinstance(opaque, str)
        return opaque

    @staticmethod
    def session(steps: list[dict[str, Any]], **fields: Any) -> SessionParams:
        envelope = {"version": 1, "steps": steps, **fields}
        return SessionParams(
            opaque_continuation=OpaqueContinuation(
                "google_interactions", json.dumps(envelope)
            )
        )

    @staticmethod
    def history() -> list[dict[str, Any]]:
        return [
            {"type": "user_input", "content": [{"type": "text", "text": "first"}]},
            _thought(["one\n\n\n", "two"]),
            _model_output("answer"),
        ]

    @pytest.mark.anyio
    async def test_current_call_cannot_reuse_a_historical_id(self) -> None:
        history = [{"type": "function_call", "id": "c1", "name": "f", "arguments": {}}]
        with pytest.raises(DialectError, match="repeats a call"):
            await _run(
                _no_io,
                _request(
                    session=self.session(history),
                    input=_chat(
                        ChatMessage("assistant", (ToolCallPart("c1", "f", {}),)),
                        _say("user", "next"),
                    ),
                ),
            )

    @pytest.mark.anyio
    async def test_response_cannot_emit_an_unreplayable_duplicate_id(self) -> None:
        history = [{"type": "function_call", "id": "c1", "name": "f", "arguments": {}}]
        output = [
            {
                "type": "function_call",
                "id": "c1",
                "name": "get_weather",
                "arguments": {},
            }
        ]
        with pytest.raises(
            OutputContractError, match="invalid opaque history.*repeats"
        ):
            await _run(
                _json_handler(_reply(status="requires_action", steps=output)),
                _request(
                    session=self.session(history),
                    tools=ToolParams(functions=(_FUNCTION_TOOL,)),
                ),
            )

    @pytest.mark.anyio
    async def test_opaque_survives_typed_disk_round_trip(self, tmp_path: Path) -> None:
        response = await _run(
            _json_handler(
                _reply(steps=[_thought(["summary"]), _model_output("answer")])
            )
        )
        target = tmp_path / "response.json"
        target.write_text(json.dumps(obj_to_dict(response, add_type=True)))
        restored = dict_to_obj(json.loads(target.read_text()), global_type_registry)
        assert isinstance(restored, Response)
        assert self.opaque(restored) == self.opaque(response)
        assert restored.reasoning == response.reasoning

    @pytest.mark.anyio
    @pytest.mark.parametrize("field", ["arguments", "is_error"])
    async def test_verifier_preserves_numeric_and_boolean_types(
        self, monkeypatch: pytest.MonkeyPatch, field: str
    ) -> None:
        history = [
            {
                "type": "function_call",
                "id": "c1",
                "name": "f",
                "arguments": {"value": 1},
            },
            {
                "type": "function_result",
                "call_id": "c1",
                "result": "failed",
                "is_error": True,
            },
        ]

        def mutate(body: dict[str, Any]) -> None:
            if field == "arguments":
                body["input"][0]["arguments"]["value"] = True
            else:
                body["input"][1]["is_error"] = 1

        _tamper_prepared(monkeypatch, mutate)
        with pytest.raises(DialectError, match="opaque history"):
            await _run(_no_io, _request(session=self.session(history)))

    @pytest.mark.anyio
    async def test_signed_response_emits_exact_history(self) -> None:
        output = [_thought(["one\n\n\n", "two"]), _model_output("answer")]
        response = await _run(
            _json_handler(_reply(steps=output)),
            _request(input=_chat(_say("system", "rules"), _say("user", "first"))),
        )
        reasoning = self.reasoning(response)
        assert reasoning.text == "one\n\n\ntwo"
        assert json.loads(self.opaque(response)) == {
            "version": 1,
            "system_instruction": "rules",
            "steps": [
                {"type": "user_input", "content": [{"type": "text", "text": "first"}]},
                *output,
            ],
        }
        assert response.session_id is None

    @pytest.mark.anyio
    async def test_replay_preserves_history_and_system_without_new_signature(
        self,
    ) -> None:
        history = self.history()
        sent = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            return httpx.Response(200, json=_reply())

        response = await _run(
            handler,
            _request(
                input=_chat(_say("user", "next")),
                session=self.session(history, system_instruction="rules"),
            ),
        )
        assert sent[0]["input"] == [
            *history,
            {"type": "user_input", "content": [{"type": "text", "text": "next"}]},
        ]
        assert sent[0]["system_instruction"] == "rules"
        assert sent[0]["store"] is False
        assert "previous_interaction_id" not in sent[0]
        reasoning = self.reasoning(response)
        assert reasoning.text is None
        assert json.loads(self.opaque(response))["steps"] == [
            *sent[0]["input"],
            _model_output("Hi"),
        ]

    @pytest.mark.anyio
    async def test_current_system_must_match_prior_authority(self) -> None:
        with pytest.raises(DialectError, match="system_instruction.*conflict"):
            await _run(
                _no_io,
                _request(
                    input=_chat(_say("system", "different"), _say("user", "next")),
                    session=self.session(self.history(), system_instruction="rules"),
                ),
            )

    @pytest.mark.anyio
    async def test_tool_result_can_answer_a_call_in_opaque_history(self) -> None:
        history = [
            {"type": "user_input", "content": [{"type": "text", "text": "weather"}]},
            _thought(),
            {"type": "function_call", "id": "c1", "name": "old_tool", "arguments": {}},
        ]
        sent = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            return httpx.Response(200, json=_reply())

        await _run(
            handler,
            _request(
                input=_chat(ChatMessage("tool", (ToolResultPart("c1", "sunny"),))),
                session=self.session(history),
            ),
        )
        assert sent[0]["input"] == [
            *history,
            {
                "type": "function_result",
                "call_id": "c1",
                "name": "old_tool",
                "result": "sunny",
            },
        ]

    @pytest.mark.anyio
    @pytest.mark.parametrize("replacement", [None, "wrong_function"])
    async def test_result_name_cannot_be_removed_or_changed(
        self, monkeypatch: pytest.MonkeyPatch, replacement: str | None
    ) -> None:
        history = [
            {"type": "function_call", "id": "c1", "name": "old_tool", "arguments": {}}
        ]

        def mutate(body: dict[str, Any]) -> None:
            if replacement is None:
                body["input"][-1].pop("name")
            else:
                body["input"][-1]["name"] = replacement

        _tamper_prepared(monkeypatch, mutate)
        with pytest.raises(DialectError, match="result name"):
            await _run(
                _no_io,
                _request(
                    input=_chat(ChatMessage("tool", (ToolResultPart("c1", "sunny"),))),
                    session=self.session(history),
                ),
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize("answered", [False, True])
    async def test_unknown_or_already_answered_historical_call_fails(
        self, answered: bool
    ) -> None:
        history = self.history()
        if answered:
            history.extend(
                [
                    {"type": "function_call", "id": "c1", "name": "f", "arguments": {}},
                    {"type": "function_result", "call_id": "c1", "result": "done"},
                ]
            )
        with pytest.raises(DialectError, match="earlier tool call|more than once"):
            await _run(
                _no_io,
                _request(
                    input=_chat(ChatMessage("tool", (ToolResultPart("c1", "again"),))),
                    session=self.session(history),
                ),
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "envelope",
        [
            {"version": True, "steps": []},
            {"version": 2, "steps": []},
            {"version": 1, "steps": None},
            {"version": 1, "steps": [1]},
            {"version": 1, "steps": [{"type": "unknown"}]},
            {"version": 1, "steps": [{"type": "thought", "signature": None}]},
            {"version": 1, "steps": [{"type": "thought", "summary": None}]},
            {
                "version": 1,
                "steps": [
                    {"type": "function_call", "id": "x", "name": "f", "arguments": 1}
                ],
            },
            {
                "version": 1,
                "steps": [
                    {"type": "function_result", "call_id": "x", "result": "orphan"}
                ],
            },
            {"version": 1, "steps": [], "system_instruction": None},
            {"version": 1, "steps": [], "extra": True},
        ],
    )
    async def test_malformed_history_fails_before_io(self, envelope: Any) -> None:
        session = SessionParams(
            opaque_continuation=OpaqueContinuation(
                "google_interactions", json.dumps(envelope)
            )
        )
        with pytest.raises(DialectError, match="opaque"):
            await _run(_no_io, _request(session=session))

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "value",
        [
            "not JSON",
            '{"version":1,"version":1,"steps":[]}',
            '{"version":1,"steps":[],"system_instruction":NaN}',
        ],
    )
    async def test_history_requires_strict_json(self, value: str) -> None:
        session = SessionParams(
            opaque_continuation=OpaqueContinuation("google_interactions", value)
        )
        with pytest.raises(DialectError, match="opaque.*JSON"):
            await _run(_no_io, _request(session=session))

    @pytest.mark.anyio
    @pytest.mark.parametrize("field", ["signature", "summary", "input", "current"])
    async def test_prepared_history_and_current_turn_are_verified(
        self, monkeypatch: pytest.MonkeyPatch, field: str
    ) -> None:
        def mutate(body: dict[str, Any]) -> None:
            if field == "signature":
                body["input"][1]["signature"] = "changed"
            elif field == "summary":
                body["input"][1]["summary"].reverse()
            elif field == "input":
                body["input"][0]["content"][0]["text"] = "changed"
            else:
                body["input"][-1]["content"][0]["text"] = "changed"

        _tamper_prepared(monkeypatch, mutate)
        with pytest.raises(DialectError, match="opaque history|provider-neutral input"):
            await _run(_no_io, _request(session=self.session(self.history())))

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "name",
        ["stream_thought_summary_completed", "stream_thought_summary_incomplete"],
    )
    async def test_stream_keeps_summary_item_boundaries_in_opaque(
        self, name: str
    ) -> None:
        events = _recorded_stream(name)
        summary = [
            event["delta"]["content"]
            for line in events.decode().splitlines()
            if line.startswith("data: {")
            for event in [json.loads(line[6:])]
            if event.get("delta", {}).get("type") == "thought_summary"
        ]
        response = await _run(
            _stream(events), _request(scheduling=SchedulingParams(stream=True))
        )
        reasoning = self.reasoning(response)
        history = json.loads(self.opaque(response))["steps"]
        thought = next(step for step in history if step["type"] == "thought")
        assert thought["summary"] == summary
        assert reasoning.text == "".join(item["text"] for item in summary)
        assert thought["signature"] == "sig-redacted"

    @pytest.mark.anyio
    async def test_snapshot_only_retains_native_steps(self) -> None:
        output = [_thought(["summary"]), _model_output("answer")]
        response = await _run(
            _stream(_sse(_created(), _completed(steps=output), _DONE)),
            _request(scheduling=SchedulingParams(stream=True)),
        )
        assert json.loads(self.opaque(response))["steps"][1:] == output

    @pytest.mark.anyio
    async def test_conflicting_terminal_signatures_fail(self) -> None:
        events = _sse(
            _created(),
            _start(0, {"type": "thought", "signature": "first"}),
            _stop(0),
            _completed(steps=[{"type": "thought", "signature": "different"}]),
            _DONE,
        )
        with pytest.raises(OutputContractError, match="signatures"):
            await _run(
                _stream(events), _request(scheduling=SchedulingParams(stream=True))
            )

    @pytest.mark.anyio
    async def test_hosted_signatures_survive_replay_without_current_hosted_tools(
        self,
    ) -> None:
        output = [
            {
                "type": "google_search_call",
                "id": "s1",
                "arguments": {},
                "signature": "call-signature",
            },
            {
                "type": "google_search_result",
                "call_id": "s1",
                "result": [],
                "signature": "result-signature",
            },
            _model_output("answer"),
        ]
        first = await _run(
            _json_handler(_reply(steps=output)),
            _request(tools=ToolParams(hosted=(HostedToolSpec("google_search"),))),
        )
        opaque = self.opaque(first)
        sent = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            return httpx.Response(200, json=_reply())

        await _run(
            handler,
            _request(
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation(
                        "google_interactions", opaque
                    )
                )
            ),
        )
        assert sent[0]["input"][1:4] == output
