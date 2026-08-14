"""OpenAI Chat Completions dialect for the provider-neutral model IR.

The dialect owns wire translation and response validation, but not the SDK
client lifetime.  A bound instance borrows the connection owned by its
ConnectionPool and is only executed while that pool has admitted the request.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from sieval.core.models.capabilities import (
    CAPABILITY_KEYS,
    Capability,
    DialectCapabilityBinding,
    DialectCapabilityDecision,
    MultimodalInputOptions,
    ReasoningOptions,
    Supported,
    Unsupported,
)
from sieval.core.models.deployment import BINDING_RESOURCE_KEYS
from sieval.core.models.dialect import (
    Guarantee,
    OutputContract,
    OutputContractError,
    OutputRule,
    PassthroughObservation,
    PreparedRequest,
    RequestAudit,
    RuntimePlanView,
    active_request_leaves,
    validate_reasoning,
    validate_request_invariants,
    validate_runtime_binding_plan,
    validate_structured_output,
    validate_tool_calls,
    validate_top_logprobs,
)
from sieval.core.models.ir import (
    ChatInput,
    ChatMessage,
    FunctionToolCall,
    ImagePart,
    ReasoningOutput,
    Request,
    Response,
    StructuredOutput,
    TextPart,
    TokenLogprob,
    ToolCallPart,
    ToolResultPart,
    TopKEntry,
    UsageStats,
)

from ._usage import usage_stats

_OPENAI_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)
_PREFILL_UNSUPPORTED_REASON = "Chat Completions has no assistant-prefill field"


def _validate_reasoning_config(options: object) -> None:
    assert isinstance(options, ReasoningOptions)
    if options.budget_tokens is not None:
        raise ValueError("openai_chat does not support reasoning budget_tokens")
    if options.effort is not None and options.effort not in _OPENAI_REASONING_EFFORTS:
        raise ValueError(f"openai_chat does not support effort {options.effort!r}")
    if options.summary not in {None, "none"}:
        raise ValueError(
            "openai_chat supports only summary='none' as a documented no-op"
        )


def _validate_multimodal_config(options: object) -> None:
    assert isinstance(options, MultimodalInputOptions)
    unsupported = set(options.modalities) - {"image"}
    if unsupported:
        raise ValueError(f"openai_chat does not support modalities {unsupported}")


CAPABILITY_DECISIONS: Mapping[str, DialectCapabilityDecision] = MappingProxyType(
    {
        "input_scoring": Unsupported("Chat Completions cannot score input tokens"),
        "sampled_logprobs": Supported(
            DialectCapabilityBinding(
                "sampled_logprobs",
                request_leaves=("scoring.sampled_logprobs",),
                response_channels=("logprobs",),
            )
        ),
        "top_logprobs": Supported(
            DialectCapabilityBinding(
                "top_logprobs",
                request_leaves=("scoring.top_logprobs",),
                response_channels=("top_logprobs",),
            )
        ),
        "reasoning": Supported(
            DialectCapabilityBinding(
                "reasoning",
                request_leaves=(
                    "reasoning.effort",
                    "reasoning.summary",
                ),
                response_channels=("reasoning",),
                _config_validator=_validate_reasoning_config,
            )
        ),
        "function_tools": Supported(
            DialectCapabilityBinding(
                "function_tools",
                request_leaves=(
                    "tools.functions",
                    "tools.choice",
                    "tools.parallel",
                ),
                response_channels=("tool_calls",),
            )
        ),
        "hosted_tools": Unsupported(
            "PR 1 does not bind hosted tools on Chat Completions"
        ),
        "structured_output": Supported(
            DialectCapabilityBinding(
                "structured_output",
                request_leaves=(
                    "structured_output.format",
                    "structured_output.schema",
                    "structured_output.name",
                    "structured_output.strict",
                ),
                response_channels=("structured_output",),
            )
        ),
        "stateful_session": Unsupported(
            "Chat Completions has no previous-response-id continuation"
        ),
        "opaque_continuation": Unsupported(
            "Chat Completions cannot round-trip provider opaque state"
        ),
        "multimodal_input": Supported(
            DialectCapabilityBinding(
                "multimodal_input",
                request_leaves=(
                    "input.modality.image",
                    "input.modality.image.media_type",
                ),
                _config_validator=_validate_multimodal_config,
            )
        ),
        "prefill": Unsupported(_PREFILL_UNSUPPORTED_REASON),
        "fim": Unsupported("Chat Completions has no completion suffix field"),
    }
)

assert set(CAPABILITY_DECISIONS) == set(CAPABILITY_KEYS)


def _tuple_validator(item_type: type, channel: str) -> Callable[[object], None]:
    def validate(value: object) -> None:
        if not isinstance(value, tuple) or not all(
            isinstance(item, item_type) for item in value
        ):
            raise OutputContractError(f"{channel} channel has invalid shape")

    return validate


OUTPUT_CONTRACT = OutputContract(
    {
        "reasoning": OutputRule(Guarantee.PRESENT_OR_ERROR, validate_reasoning),
        "tool_calls": OutputRule(Guarantee.BEST_EFFORT, validate_tool_calls),
        "server_tool_uses": OutputRule(Guarantee.NEVER),
        "structured_output": OutputRule(
            Guarantee.PRESENT_OR_ERROR, validate_structured_output
        ),
        "logprobs": OutputRule(
            Guarantee.PRESENT_OR_ERROR,
            _tuple_validator(TokenLogprob, "logprobs"),
        ),
        "top_logprobs": OutputRule(Guarantee.PRESENT_OR_ERROR, validate_top_logprobs),
        "input_scoring": OutputRule(Guarantee.NEVER),
        "citations": OutputRule(Guarantee.NEVER),
        "grounding": OutputRule(Guarantee.NEVER),
        "session_id": OutputRule(Guarantee.NEVER),
        "usage": OutputRule(Guarantee.BEST_EFFORT),
    }
)


_CANONICAL_WIRE_KEYS = BINDING_RESOURCE_KEYS | frozenset(
    {
        "model",
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stop",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "n",
        "logprobs",
        "top_logprobs",
        "echo",
        "score_input",
        "return_logprobs",
        "response_format",
        "tools",
        "functions",
        "tool_choice",
        "function_call",
        "parallel_tool_calls",
        "server_tools",
        "web_search_options",
        "reasoning",
        "reasoning_effort",
        "previous_response_id",
        "session_id",
        "opaque_continuation",
        "suffix",
        "stream",
        "stream_options",
        "extra_body",
        # Chat Completions receives this extension through ``extra_body``, but
        # it still has a provider-neutral SamplingParams owner.  Accepting it
        # again through DialectOptions would create two competing sources.
        "top_k",
    }
)


@dataclass(frozen=True)
class _ChatContext:
    messages: list[dict[str, Any]]
    request: Request


@dataclass(frozen=True)
class _LegacyPlan:
    dialect_id: str = "openai_chat"
    available_capabilities: frozenset[str] = frozenset(
        key
        for key, value in CAPABILITY_DECISIONS.items()
        if isinstance(value, Supported)
    )
    capability_minimums: Mapping[str, Mapping[str, object]] = MappingProxyType({})
    required_output_channels: frozenset[str] = frozenset()


def _chat_usage_stats(raw: Any) -> UsageStats | None:
    """Build usage from the reply's prompt and completion counts.

    Only these two counts are contractual. ``total_tokens`` is computed from
    them and the optional detail breakdown is best-effort -- see
    :func:`._usage.usage_stats`.
    """
    if raw is None:
        return None
    names = ("prompt_tokens", "completion_tokens")
    values: list[int] = []
    for name in names:
        value = getattr(raw, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise OutputContractError(
                f"chat usage.{name} must be a non-negative integer"
            )
        values.append(value)
    return usage_stats(raw, values[0], values[1])


def _optional_response_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise OutputContractError(f"chat {path} must be a string or None")
    return value


def _choice_sequence(raw: object) -> Sequence[object]:
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise OutputContractError("chat response choices must be a sequence")
    return raw


def _choice_index(choice: object, n: int) -> int:
    index = getattr(choice, "index", None)
    if isinstance(index, bool) or not isinstance(index, int):
        raise OutputContractError("chat choice index must be an integer")
    if not 0 <= index < n:
        raise OutputContractError(
            f"chat choice index {index} is outside the requested range [0, {n})"
        )
    return index


def _validate_choice_coverage(seen: set[int], n: int, *, streaming: bool) -> None:
    missing = sorted(set(range(n)) - seen)
    if missing:
        mode = "stream" if streaming else "response"
        raise OutputContractError(f"chat {mode} omitted choice indexes {missing}")


def _reasoning_text(part: Any) -> str:
    value = getattr(part, "reasoning", None)
    if value:
        return str(value)
    value = getattr(part, "reasoning_content", None)
    return str(value) if value else ""


def _finite_logprob(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise OutputContractError(f"{path} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise OutputContractError(f"{path} must be finite")
    return normalized


def _content_to_ir(
    content: object,
) -> tuple[tuple[TokenLogprob, ...], tuple[tuple[TopKEntry, ...], ...]]:
    if not isinstance(content, Sequence) or isinstance(content, str | bytes):
        raise OutputContractError("chat logprobs.content must be a sequence")

    sampled: list[TokenLogprob] = []
    alternatives: list[tuple[TopKEntry, ...]] = []
    for position, item in enumerate(content):
        token = getattr(item, "token", None)
        if not isinstance(token, str):
            raise OutputContractError(
                f"chat logprobs.content[{position}].token must be a string"
            )
        logprob = _finite_logprob(
            getattr(item, "logprob", None),
            f"chat logprobs.content[{position}].logprob",
        )
        sampled.append(TokenLogprob(token=token, logprob=logprob))

        raw_top = getattr(item, "top_logprobs", None)
        if raw_top is None:
            raw_top = ()
        if not isinstance(raw_top, Sequence) or isinstance(raw_top, str | bytes):
            raise OutputContractError(
                f"chat logprobs.content[{position}].top_logprobs must be a sequence"
            )
        parsed_top: list[TopKEntry] = []
        for rank, entry in enumerate(raw_top):
            top_token = getattr(entry, "token", None)
            if not isinstance(top_token, str):
                raise OutputContractError(
                    "chat logprobs.content"
                    f"[{position}].top_logprobs[{rank}].token must be a string"
                )
            top_logprob = _finite_logprob(
                getattr(entry, "logprob", None),
                f"chat logprobs.content[{position}].top_logprobs[{rank}].logprob",
            )
            parsed_top.append(TopKEntry(token=top_token, logprob=top_logprob))
        alternatives.append(tuple(parsed_top))
    return tuple(sampled), tuple(alternatives)


def _tool_call_to_ir(raw: Any) -> FunctionToolCall:
    function = getattr(raw, "function", None)
    if function is None and isinstance(raw, Mapping):
        function = raw.get("function")
    call_id = getattr(raw, "id", None)
    if call_id is None and isinstance(raw, Mapping):
        call_id = raw.get("id")
    name = getattr(function, "name", None)
    arguments = getattr(function, "arguments", None)
    if isinstance(function, Mapping):
        name = function.get("name")
        arguments = function.get("arguments")
    if not isinstance(call_id, str) or not isinstance(name, str):
        raise OutputContractError("tool call is missing id or function name")
    if arguments is None:
        arguments = ""
    return FunctionToolCall(call_id, name, arguments)


def _part_to_wire(part: object) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ImagePart):
        if part.url is not None:
            url = part.url
        else:
            media = part.media_type or "application/octet-stream"
            url = f"data:{media};base64,{part.data}"
        image_url: dict[str, Any] = {"url": url}
        if part.detail is not None:
            image_url["detail"] = part.detail
        return {"type": "image_url", "image_url": image_url}
    raise TypeError(f"content part {type(part).__name__} is not inline content")


def _message_to_wire(message: ChatMessage) -> dict[str, Any]:
    wire: dict[str, Any] = {"role": message.role}
    if message.name is not None:
        wire["name"] = message.name
    inline = [
        part for part in message.content if isinstance(part, TextPart | ImagePart)
    ]
    calls = [part for part in message.content if isinstance(part, ToolCallPart)]
    results = [part for part in message.content if isinstance(part, ToolResultPart)]
    if results:
        if len(results) != 1 or calls or len(inline) > 0:
            raise ValueError(
                "a tool-result message must contain exactly one result part"
            )
        result = results[0]
        wire["tool_call_id"] = result.call_id
        wire["content"] = (
            result.result
            if isinstance(result.result, str)
            else json.dumps(result.result, sort_keys=True, separators=(",", ":"))
        )
        return wire
    if calls:
        wire["tool_calls"] = [
            {
                "id": part.call_id,
                "type": "function",
                "function": {
                    "name": part.name,
                    "arguments": (
                        part.arguments
                        if isinstance(part.arguments, str)
                        else json.dumps(
                            part.arguments, sort_keys=True, separators=(",", ":")
                        )
                    ),
                },
            }
            for part in calls
        ]
    if not inline:
        wire["content"] = None
    elif all(isinstance(part, TextPart) for part in inline):
        text_parts = cast(list[TextPart], inline)
        wire["content"] = "".join(part.text for part in text_parts)
    else:
        wire["content"] = [_part_to_wire(part) for part in inline]
    return wire


def _response_format(req: Request) -> dict[str, Any] | None:
    params = req.structured_output
    if params.format is None:
        return None
    if params.format == "json_object":
        return {"type": "json_object"}
    schema: dict[str, Any] = {
        "name": params.name or "response",
        "schema": dict(params.schema or {}),
    }
    if params.strict is not None:
        schema["strict"] = params.strict
    return {"type": "json_schema", "json_schema": schema}


class OpenAIChatDialect:
    dialect_id = "openai_chat"
    connection_family = "openai_sdk"
    capability_decisions = CAPABILITY_DECISIONS
    output_contract = OUTPUT_CONTRACT
    CAPABILITIES = frozenset(
        {
            Capability.Chat,
            Capability.FunctionCalling,
            Capability.Reasoning,
            Capability.ReasoningEffort,
            Capability.SampledLogprobs,
            Capability.StructuredOutput,
            Capability.TopKLogprobs,
        }
    )

    def __init__(self, client: Any, requested_model_id: str):
        self._client = client
        self._requested_model_id = requested_model_id

    @property
    def capabilities(self) -> frozenset[Capability]:
        """Deprecated enum view retained with the transport import alias."""

        return self.CAPABILITIES

    def validate_request(
        self, req: Request, audit: RequestAudit, plan: RuntimePlanView
    ) -> None:
        del plan
        for path in audit.active:
            if path == "dialect_options":
                audit.noop(
                    path,
                    "an empty matching dialect-options block has no effect",
                )
            elif path == "input.completion" or path == "input.completion.suffix":
                audit.rejected(path, "openai_chat requires ChatInput")
            elif path == "scoring.input_scoring":
                audit.rejected(path, "Chat Completions cannot score input tokens")
            elif path == "reasoning.budget_tokens":
                audit.rejected(path, "openai_chat has no reasoning token-budget field")
            elif (
                path == "reasoning.effort"
                and req.reasoning.effort not in _OPENAI_REASONING_EFFORTS
            ):
                audit.rejected(
                    path,
                    f"unsupported OpenAI reasoning effort {req.reasoning.effort!r}",
                )
            elif path == "reasoning.summary" and req.reasoning.summary != "none":
                audit.rejected(
                    path,
                    "openai_chat supports only summary='none' as a documented no-op",
                )
            elif path == "tools.hosted":
                audit.rejected(path, "hosted tools are not active in PR 1")
            elif (
                path == "input.modality.image.media_type"
                and isinstance(req.input, ChatInput)
                and any(
                    isinstance(part, ImagePart)
                    and part.url is not None
                    and part.media_type is not None
                    for message in req.input.messages
                    for part in message.content
                )
            ):
                audit.rejected(
                    path,
                    "Chat Completions has no media-type field for URL-backed images",
                )
            elif path == "input.modality.tool_result.is_error":
                audit.rejected(
                    path,
                    "Chat Completions cannot transmit the tool-result error state",
                )
            elif path.startswith("session."):
                audit.rejected(path, "Chat Completions has no session-state field")
            elif path == "structured_output.format" and (
                req.structured_output.format not in {"json_object", "json_schema"}
            ):
                audit.rejected(path, "unsupported Chat Completions response format")
            elif (
                path
                in {
                    "structured_output.schema",
                    "structured_output.name",
                    "structured_output.strict",
                }
                and req.structured_output.format != "json_schema"
            ):
                audit.rejected(
                    path,
                    "schema, name, and strict require format='json_schema'",
                )
            elif path.startswith("dialect_options."):
                key = path.removeprefix("dialect_options.")
                if key in {"prefill", "prefix"}:
                    audit.rejected(path, _PREFILL_UNSUPPORTED_REASON)
                elif key in _CANONICAL_WIRE_KEYS:
                    audit.rejected(
                        path,
                        f"{key!r} must use its canonical provider-neutral field",
                    )

    def prepare(self, req: Request, audit: RequestAudit) -> PreparedRequest:
        if not isinstance(req.input, ChatInput):
            raise TypeError("openai_chat requires ChatInput")
        messages = [_message_to_wire(message) for message in req.input.messages]
        params: dict[str, Any] = {}
        consumed: set[str] = set()
        passthrough: dict[str, PassthroughObservation] = {}

        def consume(path: str) -> None:
            if path in audit.active and path not in audit.decisions:
                audit.consumed(path)
                consumed.add(path)

        for path in (
            "input.chat",
            "input.modality.text",
            "input.modality.image",
            "input.modality.image.media_type",
            "input.modality.tool_call",
            "input.modality.tool_result",
        ):
            consume(path)

        sampling = req.sampling
        sampling_wire = {
            "max_tokens": sampling.max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "seed": sampling.seed,
            "frequency_penalty": sampling.frequency_penalty,
            "presence_penalty": sampling.presence_penalty,
        }
        for name, value in sampling_wire.items():
            path = f"sampling.{name}"
            if value is not None:
                params[name] = value
                consume(path)
        if sampling.stop is not None:
            params["stop"] = list(sampling.stop)
            consume("sampling.stop")
        if sampling.n != 1:
            params["n"] = sampling.n
            consume("sampling.n")

        extra_body: dict[str, Any] = {}
        if sampling.top_k is not None:
            extra_body["top_k"] = sampling.top_k
            consume("sampling.top_k")
        if req.scoring.sampled_logprobs:
            params["logprobs"] = True
            consume("scoring.sampled_logprobs")
        if req.scoring.top_logprobs > 0:
            params["top_logprobs"] = req.scoring.top_logprobs
            consume("scoring.top_logprobs")

        reasoning = req.reasoning
        if reasoning.effort is not None:
            params["reasoning_effort"] = reasoning.effort
            consume("reasoning.effort")
        if reasoning.summary == "none":
            # Current Chat Completions compatibility servers expose visible
            # reasoning independently of the request; summary is a documented
            # request no-op until Responses API owns a wire control.
            audit.noop(
                "reasoning.summary",
                "Chat Completions returns its visible reasoning channel unchanged",
            )

        if req.tools.functions:
            params["tools"] = [dict(item) for item in req.tools.functions]
            consume("tools.functions")
        if req.tools.choice is not None:
            params["tool_choice"] = req.tools.choice
            consume("tools.choice")
        if req.tools.parallel is not None:
            params["parallel_tool_calls"] = req.tools.parallel
            consume("tools.parallel")

        response_format = _response_format(req)
        if response_format is not None:
            params["response_format"] = response_format
        for item in ("format", "schema", "name", "strict"):
            consume(f"structured_output.{item}")

        if req.scheduling.stream:
            params["stream"] = True
            params["stream_options"] = {"include_usage": True}
            consume("scheduling.stream")
        else:
            params["stream"] = False

        if req.dialect_options is not None:
            for key, option_value in req.dialect_options.values.items():
                path = f"dialect_options.{key}"
                if path in audit.decisions:
                    continue
                extra_body[key] = option_value
                audit.passthrough(path, "extra_body")
                passthrough[path] = PassthroughObservation("extra_body", option_value)
        if extra_body:
            params["extra_body"] = extra_body

        return PreparedRequest(
            operation="chat.completions.create",
            body=params,
            consumed_paths=frozenset(consumed),
            passthrough=passthrough,
            context=_ChatContext(messages, req),
        )

    def _structured_output(
        self, req: Request, texts: list[str]
    ) -> StructuredOutput | None:
        if req.structured_output.format is None:
            return None
        try:
            return StructuredOutput(json.loads(texts[0]))
        except (IndexError, TypeError, json.JSONDecodeError) as exc:
            raise OutputContractError("structured output is not valid JSON") from exc

    def _lift(self, raw: Any, req: Request, params: dict[str, Any]) -> Response:
        n = req.sampling.n
        texts = [""] * n
        finish_reasons = [""] * n
        reasoning: list[ReasoningOutput | None] = [None] * n
        tool_calls: tuple[FunctionToolCall, ...] | None = None
        logprobs: tuple[TokenLogprob, ...] | None = None
        top_logprobs: tuple[tuple[TopKEntry, ...], ...] | None = None
        seen: set[int] = set()
        for raw_choice in _choice_sequence(getattr(raw, "choices", None)):
            choice = cast(Any, raw_choice)
            index = _choice_index(choice, n)
            if index in seen:
                raise OutputContractError(
                    f"chat response duplicated choice index {index}"
                )
            seen.add(index)
            message = choice.message
            texts[index] = message.content or ""
            finish_reasons[index] = choice.finish_reason or ""
            text = _reasoning_text(message)
            if text:
                reasoning[index] = ReasoningOutput(text=text)
            if index == 0:
                raw_calls = getattr(message, "tool_calls", None)
                if raw_calls:
                    tool_calls = tuple(_tool_call_to_ir(call) for call in raw_calls)
                raw_logprobs = getattr(choice, "logprobs", None)
                if raw_logprobs is not None:
                    logprobs, top_logprobs = _content_to_ir(
                        getattr(raw_logprobs, "content", None)
                    )
        _validate_choice_coverage(seen, n, streaming=False)
        return Response(
            texts=tuple(texts),
            reasoning=tuple(reasoning) if any(reasoning) else None,
            finish_reasons=tuple(finish_reasons),
            tool_calls=tool_calls,
            structured_output=self._structured_output(req, texts),
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            usage=_chat_usage_stats(getattr(raw, "usage", None)),
            request_params=params,
            response_model=_optional_response_string(
                getattr(raw, "model", None), "model"
            ),
            system_fingerprint=_optional_response_string(
                getattr(raw, "system_fingerprint", None), "system_fingerprint"
            ),
        )

    async def _lift_stream(
        self, stream: Any, req: Request, params: dict[str, Any]
    ) -> Response:
        n = req.sampling.n
        texts = [""] * n
        finish_reasons = [""] * n
        reasoning_texts = [""] * n
        tool_fragments: dict[int, dict[str, str]] = {}
        usage: UsageStats | None = None
        saw_logprobs = False
        logprobs: list[TokenLogprob] = []
        top_logprobs: list[tuple[TopKEntry, ...]] = []
        response_model: str | None = None
        system_fingerprint: str | None = None
        seen: set[int] = set()

        async for chunk in stream:
            chunk_model = _optional_response_string(
                getattr(chunk, "model", None), "model"
            )
            chunk_fingerprint = _optional_response_string(
                getattr(chunk, "system_fingerprint", None), "system_fingerprint"
            )
            if response_model is None:
                response_model = chunk_model
            if system_fingerprint is None:
                system_fingerprint = chunk_fingerprint
            for raw_choice in _choice_sequence(getattr(chunk, "choices", None)):
                choice = cast(Any, raw_choice)
                choice_index = _choice_index(choice, n)
                seen.add(choice_index)
                finish_reasons[choice_index] = choice.finish_reason or ""
                delta = choice.delta
                if delta is not None:
                    texts[choice_index] += getattr(delta, "content", None) or ""
                    reasoning_texts[choice_index] += _reasoning_text(delta)
                    if choice_index == 0:
                        for call in getattr(delta, "tool_calls", None) or []:
                            tool_index = getattr(call, "index", 0)
                            if (
                                isinstance(tool_index, bool)
                                or not isinstance(tool_index, int)
                                or tool_index < 0
                            ):
                                raise OutputContractError(
                                    "streamed tool-call index must be a non-negative "
                                    "integer"
                                )
                            target = tool_fragments.setdefault(
                                tool_index,
                                {"id": "", "name": "", "arguments": ""},
                            )
                            target["id"] += getattr(call, "id", None) or ""
                            function = getattr(call, "function", None)
                            if function is not None:
                                target["name"] += getattr(function, "name", None) or ""
                                target["arguments"] += (
                                    getattr(function, "arguments", None) or ""
                                )
                if choice_index == 0:
                    raw_logprobs = getattr(choice, "logprobs", None)
                    if raw_logprobs is not None:
                        saw_logprobs = True
                        chunk_logprobs, chunk_top = _content_to_ir(
                            getattr(raw_logprobs, "content", None)
                        )
                        logprobs.extend(chunk_logprobs)
                        top_logprobs.extend(chunk_top)
            if getattr(chunk, "usage", None) is not None:
                usage = _chat_usage_stats(chunk.usage)

        _validate_choice_coverage(seen, n, streaming=True)

        calls_list: list[FunctionToolCall] = []
        for _, value in sorted(tool_fragments.items()):
            if not value["id"] or not value["name"]:
                raise OutputContractError(
                    "streamed tool call is missing id or function name"
                )
            calls_list.append(
                FunctionToolCall(value["id"], value["name"], value["arguments"])
            )
        calls = tuple(calls_list)
        reasoning = tuple(
            ReasoningOutput(text=value) if value else None for value in reasoning_texts
        )
        return Response(
            texts=tuple(texts),
            reasoning=reasoning if any(reasoning) else None,
            finish_reasons=tuple(finish_reasons),
            tool_calls=calls or None,
            structured_output=self._structured_output(req, texts),
            logprobs=tuple(logprobs) if saw_logprobs else None,
            top_logprobs=tuple(top_logprobs) if saw_logprobs else None,
            usage=usage,
            request_params=params,
            response_model=response_model,
            system_fingerprint=system_fingerprint,
        )

    async def execute(self, prepared: PreparedRequest) -> Response:
        if not isinstance(prepared.context, _ChatContext):
            raise TypeError("openai_chat received an invalid prepared context")
        context = prepared.context
        params = dict(prepared.body)
        raw = await self._client.chat.completions.create(
            model=self._requested_model_id,
            messages=context.messages,
            **params,
        )
        if params["stream"]:
            return await self._lift_stream(raw, context.request, params)
        return self._lift(raw, context.request, params)

    async def arun(self, req: Request) -> Response:
        """One-cycle direct compatibility path; Model uses the split contract."""

        plan = cast(RuntimePlanView, _LegacyPlan())
        validate_request_invariants(req)
        validate_runtime_binding_plan(plan, req)
        audit = RequestAudit(active_request_leaves(req))
        self.validate_request(req, audit, plan)
        audit.raise_rejections()
        prepared = self.prepare(req, audit)
        audit.finish(prepared)
        response = await self.execute(prepared)
        self.output_contract.validate(plan, req, response)
        return response
