"""Anthropic Messages dialect for the provider-neutral model IR.

The adapter speaks the native ``POST /v1/messages`` JSON/SSE protocol through
the provider-neutral HTTP connection family.  It owns Anthropic authentication,
top-level system extraction, content-block translation, extended-thinking
round trips, and strict terminal-response lifting; it does not own the HTTP
client lifetime.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import NoReturn, cast

import httpx

from sieval.core.models._shared import copy_json_value
from sieval.core.models.capabilities import (
    CAPABILITY_KEYS,
    Capability,
    CapabilityKey,
    DialectCapabilityBinding,
    DialectCapabilityDecision,
    MultimodalInputOptions,
    ReasoningOptions,
    StructuredOutputOptions,
    Supported,
    Unsupported,
)
from sieval.core.models.connection_factory import AsyncHTTPJSONConnection
from sieval.core.models.deployment import BINDING_RESOURCE_KEYS
from sieval.core.models.dialect import (
    DialectError,
    Guarantee,
    OutputContract,
    OutputContractError,
    OutputRule,
    PreparedRequest,
    RequestAudit,
    RuntimePlanView,
    WireEvidence,
    WireObservation,
    active_request_leaves,
    validate_reasoning,
    validate_request_invariants,
    validate_runtime_binding_plan,
    validate_structured_output,
    validate_tool_calls,
)
from sieval.core.models.ir import (
    ChatInput,
    FunctionToolCall,
    ImagePart,
    OpaqueContinuation,
    ReasoningOutput,
    Request,
    Response,
    StructuredOutput,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    UsageStats,
)
from sieval.core.types import JSONValue

_ANTHROPIC_VERSION = "2023-06-01"
_MESSAGES_PATH = "messages"
_OPAQUE_CONTINUATION_VERSION = 2
_IMAGE_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})
_REASONING_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
_STOP_REASONS = frozenset(
    {
        "end_turn",
        "max_tokens",
        "stop_sequence",
        "tool_use",
        "refusal",
        "model_context_window_exceeded",
    }
)
_IR_OWNED_BODY_KEYS = BINDING_RESOURCE_KEYS | frozenset(
    {
        "model",
        "messages",
        "system",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "stream",
        "thinking",
        "output_config",
        "tools",
        "tool_choice",
    }
)
_MESSAGE_RESPONSE_KEYS = frozenset(
    {
        "id",
        "type",
        "role",
        "content",
        "model",
        "stop_reason",
        "stop_sequence",
        "usage",
        "container",
        "stop_details",
    }
)
_STREAM_EVENT_KEYS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "ping": frozenset({"type"}),
        "error": frozenset({"type", "error"}),
        "message_start": frozenset({"type", "message"}),
        "content_block_start": frozenset({"type", "index", "content_block"}),
        "content_block_delta": frozenset({"type", "index", "delta"}),
        "content_block_stop": frozenset({"type", "index"}),
        "message_delta": frozenset({"type", "delta", "usage"}),
        "message_stop": frozenset({"type"}),
    }
)
_MESSAGE_DELTA_KEYS = frozenset(
    {"stop_reason", "stop_sequence", "container", "stop_details"}
)
#: ``recommended_model``, ``fallback_credit_token`` and
#: ``fallback_has_prefill_claim`` are omitted: each needs an ``anthropic-beta``
#: header, which this dialect does not send, so all three are unreachable and
#: rejected as unknown.  Adding header passthrough means mapping them here.
_REFUSAL_DETAIL_KEYS = frozenset({"type", "category", "explanation"})


def _validate_reasoning_config(options: object) -> None:
    assert isinstance(options, ReasoningOptions)
    if options.effort is not None and options.effort not in _REASONING_EFFORTS:
        raise ValueError(
            f"anthropic_messages does not support effort {options.effort!r}"
        )
    if options.budget_tokens is not None and options.budget_tokens < 1024:
        raise ValueError(
            "anthropic_messages reasoning budget_tokens must be at least 1024"
        )
    if options.summary not in {None, "none", "auto"}:
        raise ValueError(
            "anthropic_messages supports only summary='none' or summary='auto'"
        )


def _validate_structured_output_config(options: object) -> None:
    assert isinstance(options, StructuredOutputOptions)
    unsupported = set(options.formats) - {"json_schema"}
    if unsupported:
        raise ValueError(
            "anthropic_messages supports only json_schema structured output"
        )


def _validate_multimodal_config(options: object) -> None:
    assert isinstance(options, MultimodalInputOptions)
    unsupported = set(options.modalities) - {"image"}
    if unsupported:
        raise ValueError(
            f"anthropic_messages does not support modalities {unsupported}"
        )


CAPABILITY_DECISIONS: Mapping[CapabilityKey, DialectCapabilityDecision] = (
    MappingProxyType(
        {
            "input_scoring": Unsupported(
                "Anthropic Messages cannot score input tokens"
            ),
            "sampled_logprobs": Unsupported(
                "Anthropic Messages does not return sampled token logprobs"
            ),
            "top_logprobs": Unsupported(
                "Anthropic Messages does not return alternative token logprobs"
            ),
            "reasoning": Supported(
                DialectCapabilityBinding(
                    "reasoning",
                    request_leaves=(
                        "reasoning.effort",
                        "reasoning.budget_tokens",
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
                "anthropic_messages does not support Anthropic server-hosted tools"
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
                    _config_validator=_validate_structured_output_config,
                )
            ),
            "stateful_session": Unsupported(
                "Anthropic Messages has no previous-response-id continuation"
            ),
            "opaque_continuation": Supported(
                DialectCapabilityBinding(
                    "opaque_continuation",
                    request_leaves=("session.opaque_continuation",),
                    response_channels=("reasoning",),
                )
            ),
            "multimodal_input": Supported(
                DialectCapabilityBinding(
                    "multimodal_input",
                    request_leaves=(
                        "input.modality.image",
                        "input.modality.image.media_type",
                        "input.modality.image.detail",
                    ),
                    _config_validator=_validate_multimodal_config,
                )
            ),
            # Anthropic's final assistant message is a native prefill.  The
            # semantic is carried by ChatInput rather than a separate Request leaf.
            "prefill": Supported(DialectCapabilityBinding("prefill")),
            "fim": Unsupported("Anthropic Messages has no completion suffix"),
        }
    )
)

assert set(CAPABILITY_DECISIONS) == set(CAPABILITY_KEYS)


OUTPUT_CONTRACT = OutputContract(
    {
        "reasoning": OutputRule(Guarantee.PRESENT_OR_ERROR, validate_reasoning),
        "tool_calls": OutputRule(Guarantee.BEST_EFFORT, validate_tool_calls),
        "server_tool_uses": OutputRule(Guarantee.NEVER),
        "structured_output": OutputRule(
            Guarantee.PRESENT_OR_ERROR, validate_structured_output
        ),
        "logprobs": OutputRule(Guarantee.NEVER),
        "top_logprobs": OutputRule(Guarantee.NEVER),
        "input_scoring": OutputRule(Guarantee.NEVER),
        "citations": OutputRule(Guarantee.NEVER),
        "grounding": OutputRule(Guarantee.NEVER),
        "session_id": OutputRule(Guarantee.NEVER),
        "usage": OutputRule(Guarantee.BEST_EFFORT),
    }
)


@dataclass(frozen=True)
class _AnthropicContext:
    request: Request


@dataclass(frozen=True)
class _ExecutionContext:
    request: Request
    system: tuple[Mapping[str, JSONValue], ...]
    messages: tuple[Mapping[str, JSONValue], ...]
    request_params: Mapping[str, JSONValue]


@dataclass(frozen=True)
class _DecodedContinuation:
    system: tuple[dict[str, JSONValue], ...]
    messages: tuple[dict[str, JSONValue], ...]
    tools: tuple[dict[str, JSONValue], ...] | None
    open_turn: bool
    thinking: Mapping[str, JSONValue] | None
    effort: str | None


@dataclass(frozen=True)
class _LegacyPlan:
    dialect_id: str = "anthropic_messages"
    available_capabilities: frozenset[str] = frozenset(
        key
        for key, decision in CAPABILITY_DECISIONS.items()
        if isinstance(decision, Supported)
    )
    capability_minimums: Mapping[str, Mapping[str, JSONValue]] = MappingProxyType({})
    required_output_channels: frozenset[str] = frozenset()


def _canonical_json(value: object, path: str) -> str:
    return json.dumps(
        copy_json_value(value, path),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _json_mapping(value: object, path: str) -> dict[str, JSONValue]:
    copied = copy_json_value(value, path)
    if not isinstance(copied, dict):
        raise OutputContractError(f"{path} must be an object")
    return cast(dict[str, JSONValue], copied)


def _reject_unknown_fields(
    value: Mapping[str, object], allowed: frozenset[str], path: str
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise OutputContractError(
            f"{path} has unsupported field(s): " + ", ".join(sorted(unknown))
        )


def _required_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise OutputContractError(f"{path} must be a non-empty string")
    return value


def _optional_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, path)


def _validate_stop_details(stop_reason: str, raw: object) -> None:
    if stop_reason != "refusal":
        if raw is not None:
            raise OutputContractError(
                "response.stop_details must be null unless stop_reason is 'refusal'"
            )
        return

    details = _json_mapping(raw, "response.stop_details")
    _reject_unknown_fields(details, _REFUSAL_DETAIL_KEYS, "response.stop_details")
    if details.get("type") != "refusal":
        raise OutputContractError("response.stop_details.type must be 'refusal'")
    for field in ("category", "explanation"):
        value = details.get(field)
        if value is not None and not isinstance(value, str):
            raise OutputContractError(
                f"response.stop_details.{field} must be a string or null"
            )


def _sequence(value: object, path: str) -> Sequence[object]:
    if not isinstance(value, list | tuple):
        raise OutputContractError(f"{path} must be a list")
    return cast(Sequence[object], value)


def _nonnegative_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OutputContractError(f"{path} must be a non-negative integer")
    return value


def _json_text(value: JSONValue) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        copy_json_value(value, "tool result"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _reject_nonstandard_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _unique_json_object(
    pairs: list[tuple[str, JSONValue]],
) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _strict_json_loads(value: str | bytes) -> JSONValue:
    return cast(
        JSONValue,
        json.loads(
            value,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_nonstandard_json_constant,
        ),
    )


def _image_rejection(part: ImagePart) -> tuple[str, str] | None:
    if part.url == "" or part.data == "":
        return (
            "input.modality.image",
            "Anthropic image URL/data payloads must not be empty",
        )
    if part.detail is not None:
        return (
            "input.modality.image.detail",
            "Anthropic Messages has no provider-neutral image detail control",
        )
    if part.url is not None and part.media_type is not None:
        return (
            "input.modality.image.media_type",
            "Anthropic URL image sources do not carry a media_type",
        )
    if part.data is not None and part.media_type is None:
        return (
            "input.modality.image",
            "Anthropic base64 image sources require media_type",
        )
    if part.data is not None and part.media_type not in _IMAGE_MEDIA_TYPES:
        return (
            "input.modality.image.media_type",
            f"Anthropic base64 images require one of {sorted(_IMAGE_MEDIA_TYPES)}",
        )
    return None


def _image_to_wire(part: ImagePart) -> dict[str, JSONValue]:
    if part.url is not None:
        return {
            "type": "image",
            "source": {"type": "url", "url": part.url},
        }
    assert part.data is not None
    assert part.media_type is not None
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": part.media_type,
            "data": part.data,
        },
    }


def _part_to_wire(part: object) -> dict[str, JSONValue]:
    if isinstance(part, TextPart):
        if not isinstance(part.text, str):
            raise DialectError("Anthropic text content must be a string")
        return {"type": "text", "text": part.text}
    if isinstance(part, ImagePart):
        return _image_to_wire(part)
    if isinstance(part, ToolCallPart):
        if not isinstance(part.arguments, Mapping):
            raise DialectError("Anthropic tool_use input must be a JSON object")
        return {
            "type": "tool_use",
            "id": part.call_id,
            "name": part.name,
            "input": copy_json_value(part.arguments, "tool call arguments"),
        }
    if isinstance(part, ToolResultPart):
        return {
            "type": "tool_result",
            "tool_use_id": part.call_id,
            "content": _json_text(part.result),
            "is_error": part.is_error,
        }
    raise DialectError(f"unsupported Anthropic message part {type(part).__name__}")


def _lower_chat_input(
    input_: ChatInput,
) -> tuple[list[dict[str, JSONValue]], list[dict[str, JSONValue]]]:
    system: list[dict[str, JSONValue]] = []
    messages: list[dict[str, JSONValue]] = []
    for message in input_.messages:
        if message.role == "system":
            for part in message.content:
                if not isinstance(part, TextPart):
                    raise DialectError(
                        "Anthropic top-level system accepts text content only"
                    )
                system.append(_part_to_wire(part))
            continue
        role = "user" if message.role == "tool" else message.role
        messages.append(
            {
                "role": role,
                "content": [_part_to_wire(part) for part in message.content],
            }
        )
    return system, messages


def _function_tool_to_wire(raw: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    data = dict(raw)
    if data.get("type") != "function":
        raise DialectError("function tool declarations require type='function'")
    nested = data.get("function")
    if nested is not None:
        if set(data) != {"type", "function"} or not isinstance(nested, Mapping):
            raise DialectError("invalid Chat-shaped function tool declaration")
        source = dict(nested)
    else:
        source = data
        source.pop("type", None)

    allowed = {"name", "description", "parameters", "strict"}
    unknown = set(source) - allowed
    if unknown:
        raise DialectError(
            "unsupported function tool field(s): " + ", ".join(sorted(unknown))
        )
    name = source.get("name")
    if not isinstance(name, str) or not name:
        raise DialectError("function tool name must be a non-empty string")
    parameters = source.get("parameters")
    if not isinstance(parameters, Mapping):
        raise DialectError("function tool parameters must be a JSON object")
    result: dict[str, JSONValue] = {
        "name": name,
        "input_schema": copy_json_value(parameters, "function tool parameters"),
    }
    if "description" in source:
        description = source["description"]
        if not isinstance(description, str):
            raise DialectError("function tool description must be a string")
        result["description"] = description
    if "strict" in source:
        strict = source["strict"]
        if not isinstance(strict, bool):
            raise DialectError("function tool strict must be a boolean")
        result["strict"] = strict
    return result


def _tool_choice_to_wire(raw: JSONValue) -> dict[str, JSONValue]:
    if isinstance(raw, str):
        choices = {"auto": "auto", "required": "any", "none": "none"}
        try:
            return {"type": choices[raw]}
        except KeyError as exc:
            raise DialectError(f"unsupported Anthropic tool choice {raw!r}") from exc
    if not isinstance(raw, Mapping):
        raise DialectError("Anthropic tool choice must be a string or mapping")
    data = dict(raw)
    if data.get("type") != "function":
        raise DialectError("only function-specific mapped tool choices are supported")
    nested = data.get("function")
    if nested is not None:
        if set(data) != {"type", "function"} or not isinstance(nested, Mapping):
            raise DialectError("invalid Chat-shaped function tool choice")
        if set(nested) != {"name"}:
            raise DialectError("function tool choice requires only a name")
        name = nested.get("name")
    else:
        if set(data) != {"type", "name"}:
            raise DialectError("invalid function tool choice")
        name = data.get("name")
    if not isinstance(name, str) or not name:
        raise DialectError("function tool choice requires a non-empty name")
    return {"type": "tool", "name": name}


def _tool_choice(
    req: Request,
) -> dict[str, JSONValue] | None:
    choice = (
        _tool_choice_to_wire(req.tools.choice) if req.tools.choice is not None else None
    )
    if req.tools.parallel is not None:
        if choice is None:
            choice = {"type": "auto"}
        choice["disable_parallel_tool_use"] = not req.tools.parallel
    return choice


def _thinking_config(req: Request) -> dict[str, JSONValue] | None:
    # ``display`` lives only on the ``thinking`` object, so "no visible
    # summary" cannot be said without also saying "think": ``summary='none'``
    # enables thinking on a model where it defaults off, and those tokens bill
    # as output. Leaving ``summary`` unset is what sends no thinking at all.
    # (On Responses ``summary='none'`` is a no-op -- the same field, a
    # different meaning.)
    thinking: dict[str, JSONValue] | None = None
    if req.reasoning.budget_tokens is not None:
        thinking = {
            "type": "enabled",
            "budget_tokens": req.reasoning.budget_tokens,
        }
    if req.reasoning.summary is not None:
        if thinking is None:
            thinking = {"type": "adaptive"}
        thinking["display"] = (
            "summarized" if req.reasoning.summary == "auto" else "omitted"
        )
    return thinking


def _has_assistant_prefill(input_: ChatInput) -> bool:
    return input_.messages[-1].role == "assistant"


def _validate_continuation_thinking(value: object) -> Mapping[str, JSONValue] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise DialectError("Anthropic continuation thinking state must be an object")
    thinking = cast(Mapping[str, JSONValue], value)
    thinking_type = thinking.get("type")
    if thinking_type == "enabled":
        allowed = {"type", "budget_tokens", "display"}
        if set(thinking) - allowed:
            raise DialectError("Anthropic continuation thinking state is malformed")
        budget = thinking.get("budget_tokens")
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1024:
            raise DialectError("Anthropic continuation thinking budget is invalid")
    elif thinking_type == "adaptive":
        if set(thinking) - {"type", "display"}:
            raise DialectError("Anthropic continuation thinking state is malformed")
    else:
        raise DialectError("Anthropic continuation thinking type is invalid")
    if thinking.get("display") not in {None, "summarized", "omitted"}:
        raise DialectError("Anthropic continuation thinking display is invalid")
    return thinking


def _decode_continuation(
    value: str,
) -> _DecodedContinuation:
    try:
        raw = _strict_json_loads(value)
    except (UnicodeDecodeError, ValueError) as exc:
        raise DialectError("Anthropic opaque continuation is not valid JSON") from exc
    expected_keys = {
        "version",
        "system",
        "messages",
        "tools",
        "open_turn",
        "reasoning",
    }
    if not isinstance(raw, dict) or set(raw) != expected_keys:
        raise DialectError("Anthropic opaque continuation has an invalid envelope")
    if raw["version"] != _OPAQUE_CONTINUATION_VERSION:
        raise DialectError("Anthropic opaque continuation version is unsupported")
    raw_system = raw["system"]
    raw_messages = raw["messages"]
    if not isinstance(raw_system, list) or not isinstance(raw_messages, list):
        raise DialectError("Anthropic opaque continuation history must be lists")
    open_turn = raw["open_turn"]
    if not isinstance(open_turn, bool):
        raise DialectError("Anthropic continuation open_turn must be a boolean")
    raw_reasoning = raw["reasoning"]
    if not isinstance(raw_reasoning, dict) or set(raw_reasoning) != {
        "thinking",
        "effort",
    }:
        raise DialectError("Anthropic continuation reasoning state is malformed")
    effort = raw_reasoning["effort"]
    if effort is not None and (
        not isinstance(effort, str) or effort not in _REASONING_EFFORTS
    ):
        raise DialectError("Anthropic continuation effort is invalid")
    thinking = _validate_continuation_thinking(raw_reasoning["thinking"])
    try:
        system = [
            _json_mapping(item, f"opaque.system[{index}]")
            for index, item in enumerate(raw_system)
        ]
        messages = [
            _json_mapping(item, f"opaque.messages[{index}]")
            for index, item in enumerate(raw_messages)
        ]
    except OutputContractError as exc:
        raise DialectError("Anthropic opaque continuation is malformed") from exc
    _validate_replay_system(system)
    _validate_replay_messages(messages)
    tool_history_rejection, pending_tool_calls = _tool_history_state(messages)
    if tool_history_rejection is not None:
        raise DialectError(tool_history_rejection)
    if open_turn != bool(pending_tool_calls):
        raise DialectError(
            "Anthropic continuation open_turn disagrees with its terminal "
            "tool-use state"
        )
    raw_tools = raw["tools"]
    if raw_tools is None:
        tools = None
    elif isinstance(raw_tools, list) and raw_tools:
        try:
            tools = [
                _json_mapping(item, f"opaque.tools[{index}]")
                for index, item in enumerate(raw_tools)
            ]
        except OutputContractError as exc:
            raise DialectError("Anthropic continuation tools are malformed") from exc
        _validate_replay_tools(tools)
    else:
        raise DialectError("Anthropic continuation tools must be null or non-empty")
    has_signed_thinking = any(
        block.get("type") in {"thinking", "redacted_thinking"}
        for message in messages
        if message.get("role") == "assistant"
        for block in cast(list[Mapping[str, JSONValue]], message.get("content", []))
    )
    if not has_signed_thinking and (
        not open_turn or (thinking is None and effort is None)
    ):
        raise DialectError(
            "Anthropic opaque continuation contains no replayable thinking/tool state"
        )
    return _DecodedContinuation(
        system=tuple(system),
        messages=tuple(messages),
        tools=tuple(tools) if tools is not None else None,
        open_turn=open_turn,
        thinking=thinking,
        effort=effort,
    )


def _continuation_reasoning_rejection(req: Request) -> str | None:
    continuation = req.session.opaque_continuation
    if continuation is None:
        return None
    decoded = _decode_continuation(continuation.value)
    if not decoded.open_turn:
        return None
    current_thinking = _thinking_config(req)
    if _canonical_json(
        current_thinking, "current thinking configuration"
    ) != _canonical_json(decoded.thinking, "continued thinking configuration"):
        return (
            "an open Anthropic tool turn must reuse the exact thinking mode, "
            "budget, and display from the prior request"
        )
    if req.reasoning.effort != decoded.effort:
        return (
            "an open Anthropic tool turn must reuse the exact reasoning effort "
            "from the prior request"
        )
    return None


def _continuation_tools_mismatch(
    continuation: OpaqueContinuation,
    body: Mapping[str, JSONValue],
) -> str | None:
    decoded = _decode_continuation(continuation.value)
    if not decoded.open_turn:
        return None
    expected: JSONValue = (
        [dict(tool) for tool in decoded.tools] if decoded.tools is not None else None
    )
    if _canonical_json(body.get("tools"), "body.tools") != _canonical_json(
        expected, "continued tools"
    ):
        return (
            "body.tools must exactly match the tool definitions from the "
            "continued Anthropic thinking history"
        )
    return None


def _validate_replay_system(system: Sequence[Mapping[str, JSONValue]]) -> None:
    for index, block in enumerate(system):
        if set(block) != {"type", "text"} or block.get("type") != "text":
            raise DialectError(
                f"opaque.system[{index}] must be an Anthropic text block"
            )
        if not isinstance(block.get("text"), str):
            raise DialectError(f"opaque.system[{index}].text must be a string")


def _validate_replay_messages(messages: Sequence[Mapping[str, JSONValue]]) -> None:
    for index, message in enumerate(messages):
        if set(message) != {"role", "content"}:
            raise DialectError(f"opaque.messages[{index}] has unsupported fields")
        if message.get("role") not in {"user", "assistant"}:
            raise DialectError(f"opaque.messages[{index}].role is invalid")
        content = message.get("content")
        if not isinstance(content, list) or not content:
            raise DialectError(f"opaque.messages[{index}].content must be non-empty")
        for block_index, block in enumerate(content):
            if not isinstance(block, Mapping):
                raise DialectError(
                    f"opaque.messages[{index}].content[{block_index}] must be an object"
                )
            _validate_history_block(
                block,
                f"opaque.messages[{index}].content[{block_index}]",
                role=cast(str, message["role"]),
            )


def _tool_history_state(
    messages: Sequence[Mapping[str, JSONValue]],
) -> tuple[str | None, frozenset[str]]:
    """Validate Anthropic's immediate, one-result-per-call tool protocol."""

    seen_call_ids: set[str] = set()
    pending_call_ids: frozenset[str] = frozenset()
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            return f"messages[{message_index}].content must be a list", frozenset()

        call_ids: list[str] = []
        result_ids: list[str] = []
        saw_non_result = False
        for block_index, block in enumerate(content):
            if not isinstance(block, Mapping):
                return (
                    f"messages[{message_index}].content[{block_index}] must be an "
                    "object",
                    frozenset(),
                )
            block_type = block.get("type")
            if block_type == "tool_result":
                if saw_non_result:
                    return (
                        f"messages[{message_index}] tool_result blocks must precede "
                        "all non-tool-result content",
                        frozenset(),
                    )
                tool_use_id = block.get("tool_use_id")
                if isinstance(tool_use_id, str):
                    result_ids.append(tool_use_id)
            else:
                saw_non_result = True
            if block_type == "tool_use":
                call_id = block.get("id")
                if isinstance(call_id, str):
                    call_ids.append(call_id)

        if result_ids:
            if message.get("role") != "user":
                return (
                    f"messages[{message_index}] tool results require user role",
                    frozenset(),
                )
            if len(result_ids) != len(set(result_ids)):
                return (
                    f"messages[{message_index}] duplicates a tool_result id",
                    frozenset(),
                )
            if not pending_call_ids:
                return (
                    f"messages[{message_index}] tool results do not immediately "
                    "follow tool uses",
                    frozenset(),
                )
            actual_results = frozenset(result_ids)
            if actual_results != pending_call_ids:
                return (
                    f"messages[{message_index}] tool_result ids must exactly match "
                    "the preceding assistant tool_use ids",
                    frozenset(),
                )
            pending_call_ids = frozenset()
        elif pending_call_ids:
            return (
                f"messages[{message_index}] must immediately return every preceding "
                "assistant tool result",
                frozenset(),
            )

        if call_ids:
            if message.get("role") != "assistant":
                return (
                    f"messages[{message_index}] tool uses require assistant role",
                    frozenset(),
                )
            if len(call_ids) != len(set(call_ids)) or any(
                call_id in seen_call_ids for call_id in call_ids
            ):
                return (
                    f"messages[{message_index}] duplicates a tool_use id",
                    frozenset(),
                )
            seen_call_ids.update(call_ids)
            pending_call_ids = frozenset(call_ids)

    return None, pending_call_ids


def _validate_replay_tools(tools: Sequence[Mapping[str, JSONValue]]) -> None:
    for index, tool in enumerate(tools):
        path = f"opaque.tools[{index}]"
        allowed = {"name", "description", "input_schema", "strict"}
        if set(tool) - allowed:
            raise DialectError(f"{path} has unsupported fields")
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise DialectError(f"{path}.name must be a non-empty string")
        if not isinstance(tool.get("input_schema"), Mapping):
            raise DialectError(f"{path}.input_schema must be an object")
        if "description" in tool and not isinstance(tool["description"], str):
            raise DialectError(f"{path}.description must be a string")
        if "strict" in tool and not isinstance(tool["strict"], bool):
            raise DialectError(f"{path}.strict must be a boolean")


def _validate_history_block(
    block: Mapping[str, JSONValue], path: str, *, role: str
) -> None:
    block_type = block.get("type")
    if block_type == "text":
        if set(block) - {"type", "text", "citations"} or not isinstance(
            block.get("text"), str
        ):
            raise DialectError(f"{path} is not a canonical text block")
        citations = block.get("citations")
        if "citations" in block and citations is not None and citations != []:
            raise DialectError(f"{path}.citations cannot be replayed losslessly")
        return
    if block_type == "image":
        if role != "user":
            raise DialectError(f"{path} image block requires user role")
        _validate_history_image(block, path)
        return
    if block_type == "tool_use":
        if role != "assistant":
            raise DialectError(f"{path} tool_use block requires assistant role")
        allowed = {"type", "id", "name", "input"}
        if set(block) != allowed:
            raise DialectError(f"{path} is not a canonical tool_use block")
        _required_history_string(block.get("id"), f"{path}.id")
        _required_history_string(block.get("name"), f"{path}.name")
        if not isinstance(block.get("input"), Mapping):
            raise DialectError(f"{path}.input must be an object")
        return
    if block_type == "tool_result":
        if role != "user":
            raise DialectError(f"{path} tool_result block requires user role")
        allowed = {"type", "tool_use_id", "content", "is_error"}
        if set(block) != allowed:
            raise DialectError(f"{path} is not a canonical tool_result block")
        _required_history_string(block.get("tool_use_id"), f"{path}.tool_use_id")
        if not isinstance(block.get("content"), str):
            raise DialectError(f"{path}.content must be a string")
        if not isinstance(block.get("is_error"), bool):
            raise DialectError(f"{path}.is_error must be a boolean")
        return
    if block_type == "thinking":
        if role != "assistant" or set(block) != {
            "type",
            "thinking",
            "signature",
        }:
            raise DialectError(f"{path} is not a canonical thinking block")
        if not isinstance(block.get("thinking"), str):
            raise DialectError(f"{path}.thinking must be a string")
        _required_history_string(block.get("signature"), f"{path}.signature")
        return
    if block_type == "redacted_thinking":
        if role != "assistant" or set(block) != {"type", "data"}:
            raise DialectError(f"{path} is not a canonical redacted-thinking block")
        _required_history_string(block.get("data"), f"{path}.data")
        return
    raise DialectError(f"{path} has unsupported Anthropic block type {block_type!r}")


def _validate_history_image(block: Mapping[str, JSONValue], path: str) -> None:
    if set(block) != {"type", "source"} or not isinstance(block.get("source"), Mapping):
        raise DialectError(f"{path} is not a canonical image block")
    source = cast(Mapping[str, JSONValue], block["source"])
    source_type = source.get("type")
    if source_type == "url":
        if set(source) != {"type", "url"}:
            raise DialectError(f"{path}.source is not a canonical URL source")
        _required_history_string(source.get("url"), f"{path}.source.url")
        return
    if source_type == "base64":
        if set(source) != {"type", "media_type", "data"}:
            raise DialectError(f"{path}.source is not a canonical base64 source")
        if source.get("media_type") not in _IMAGE_MEDIA_TYPES:
            raise DialectError(f"{path}.source.media_type is unsupported")
        _required_history_string(source.get("data"), f"{path}.source.data")
        return
    raise DialectError(f"{path}.source.type is unsupported")


def _required_history_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise DialectError(f"{path} must be a non-empty string")
    return value


def _verification_image(part: ImagePart) -> dict[str, JSONValue]:
    if part.url is not None:
        return {"type": "image", "source": {"type": "url", "url": part.url}}
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": cast(str, part.media_type),
            "data": cast(str, part.data),
        },
    }


def _verification_part(part: object) -> dict[str, JSONValue]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ImagePart):
        return _verification_image(part)
    if isinstance(part, ToolCallPart):
        return {
            "type": "tool_use",
            "id": part.call_id,
            "name": part.name,
            "input": copy_json_value(part.arguments, "verified tool call"),
        }
    if isinstance(part, ToolResultPart):
        content = (
            part.result
            if isinstance(part.result, str)
            else json.dumps(
                copy_json_value(part.result, "verified tool result"),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return {
            "type": "tool_result",
            "tool_use_id": part.call_id,
            "content": content,
            "is_error": part.is_error,
        }
    raise AssertionError(f"unclassified verification part {type(part).__name__}")


def _verification_current_input(
    input_: ChatInput,
) -> tuple[list[dict[str, JSONValue]], list[dict[str, JSONValue]]]:
    system: list[dict[str, JSONValue]] = []
    messages: list[dict[str, JSONValue]] = []
    for message in input_.messages:
        if message.role == "system":
            for part in message.content:
                assert isinstance(part, TextPart)
                system.append({"type": "text", "text": part.text})
            continue
        messages.append(
            {
                "role": "user" if message.role == "tool" else message.role,
                "content": [_verification_part(part) for part in message.content],
            }
        )
    return system, messages


def _verification_function_tool(
    raw: Mapping[str, JSONValue], index: int
) -> dict[str, JSONValue]:
    """Independently derive one expected tool wire shape for audit evidence."""

    data = dict(raw)
    nested = data.get("function")
    if nested is not None:
        if data.get("type") != "function" or not isinstance(nested, Mapping):
            raise AssertionError(f"validated tool {index} changed shape")
        source = dict(nested)
    else:
        source = {key: value for key, value in data.items() if key != "type"}
    name = source.get("name")
    parameters = source.get("parameters")
    if not isinstance(name, str) or not isinstance(parameters, Mapping):
        raise AssertionError(f"validated tool {index} changed shape")
    result: dict[str, JSONValue] = {
        "name": name,
        "input_schema": copy_json_value(
            parameters, f"verified function tool {index} parameters"
        ),
    }
    if "description" in source:
        result["description"] = cast(str, source["description"])
    if "strict" in source:
        result["strict"] = cast(bool, source["strict"])
    return result


def _verification_tool_choice(raw: JSONValue) -> dict[str, JSONValue]:
    """Independently derive Anthropic tool choice from provider-neutral input."""

    if isinstance(raw, str):
        return {
            "type": {
                "auto": "auto",
                "required": "any",
                "none": "none",
            }[raw]
        }
    assert isinstance(raw, Mapping)
    raw_mapping = cast(Mapping[str, object], raw)
    nested = raw_mapping.get("function")
    name = (
        cast(Mapping[str, object], nested).get("name")
        if isinstance(nested, Mapping)
        else raw_mapping.get("name")
    )
    assert isinstance(name, str)
    return {"type": "tool", "name": name}


@dataclass(frozen=True)
class _AnthropicInputVerifier:
    source: ChatInput
    continuation: OpaqueContinuation | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", deepcopy(self.source))
        object.__setattr__(self, "continuation", deepcopy(self.continuation))

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        current_system, current_messages = _verification_current_input(self.source)
        history_system: list[dict[str, JSONValue]] = []
        history_messages: list[dict[str, JSONValue]] = []
        if self.continuation is not None:
            decoded = _decode_continuation(self.continuation.value)
            history_system = list(decoded.system)
            history_messages = list(decoded.messages)
        expected_system = [*history_system, *current_system]
        expected_messages = [*history_messages, *current_messages]
        actual_system = body.get("system")
        if expected_system:
            if actual_system is None:
                return "body.system is missing"
            if _canonical_json(actual_system, "body.system") != _canonical_json(
                expected_system, "expected system"
            ):
                return "body.system changed"
        elif actual_system is not None:
            return "body.system was added"
        actual_messages = body.get("messages")
        if _canonical_json(actual_messages, "body.messages") != _canonical_json(
            expected_messages, "expected messages"
        ):
            return "body.messages changed"
        if self.continuation is not None:
            mismatch = _continuation_tools_mismatch(self.continuation, body)
            if mismatch is not None:
                return mismatch
        return None


@dataclass(frozen=True)
class _AnthropicToolsVerifier:
    source: tuple[Mapping[str, JSONValue], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", deepcopy(self.source))

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        actual = body.get("tools")
        if not isinstance(actual, list) or len(actual) != len(self.source):
            return "body.tools count changed"
        for index, (source, wire) in enumerate(zip(self.source, actual, strict=True)):
            expected = _verification_function_tool(source, index)
            if _canonical_json(wire, f"body.tools[{index}]") != _canonical_json(
                expected, f"expected tools[{index}]"
            ):
                return f"body.tools[{index}] changed"
        return None


@dataclass(frozen=True)
class _AnthropicToolChoiceVerifier:
    choice: JSONValue
    parallel: bool | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "choice", deepcopy(self.choice))

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        expected: dict[str, JSONValue] | None = None
        if self.choice is not None:
            expected = _verification_tool_choice(self.choice)
        if self.parallel is not None:
            if expected is None:
                expected = {"type": "auto"}
            expected["disable_parallel_tool_use"] = not self.parallel
        if _canonical_json(body.get("tool_choice"), "body.tool_choice") != (
            _canonical_json(expected, "expected tool choice")
        ):
            return "body.tool_choice changed"
        return None


@dataclass(frozen=True)
class _ReasoningSummaryVerifier:
    summary: str
    has_budget: bool

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        thinking = body.get("thinking")
        if not isinstance(thinking, Mapping):
            return "body.thinking is missing"
        thinking_mapping = cast(Mapping[str, object], thinking)
        expected = "summarized" if self.summary == "auto" else "omitted"
        if thinking_mapping.get("display") != expected:
            return "body.thinking.display changed"
        expected_type = "enabled" if self.has_budget else "adaptive"
        if thinking_mapping.get("type") != expected_type:
            return "body.thinking.type changed"
        return None


def _message_rejection(input_: ChatInput) -> str | None:
    saw_conversation = False
    for index, message in enumerate(input_.messages):
        if not message.content:
            return f"messages[{index}] has empty content"
        for part_index, part in enumerate(message.content):
            if isinstance(part, TextPart) and not isinstance(part.text, str):
                return f"messages[{index}].content[{part_index}].text must be a string"
        if message.name is not None:
            return "Anthropic Messages has no message name field"
        if message.role == "developer":
            return "Anthropic Messages has no developer-role authority level"
        if message.role == "system":
            if saw_conversation:
                return "Anthropic system messages must precede conversation turns"
            if not all(isinstance(part, TextPart) for part in message.content):
                return "Anthropic top-level system accepts text content only"
            continue
        saw_conversation = True
        if message.role == "tool" and not all(
            isinstance(part, ToolResultPart) for part in message.content
        ):
            return "Anthropic tool-role messages require only tool results"
        if message.role == "user" and any(
            isinstance(part, ToolCallPart) for part in message.content
        ):
            return "Anthropic tool_use blocks require assistant role"
        if message.role == "assistant" and any(
            isinstance(part, ImagePart | ToolResultPart) for part in message.content
        ):
            return "Anthropic assistant messages cannot contain images or tool results"
        for part in message.content:
            if isinstance(part, ToolCallPart):
                if not part.call_id or not part.name:
                    return "Anthropic tool_use requires non-empty id and name"
                if not isinstance(part.arguments, Mapping):
                    return "Anthropic tool_use input must be a JSON object"
            if isinstance(part, ToolResultPart) and not part.call_id:
                return "Anthropic tool_result requires a non-empty tool_use_id"
    if not saw_conversation:
        return "Anthropic Messages requires at least one conversational message"
    return None


def _tool_history_rejection(req: Request) -> str | None:
    if not isinstance(req.input, ChatInput):
        return None
    history_messages: list[dict[str, JSONValue]] = []
    continuation = req.session.opaque_continuation
    if continuation is not None:
        history_messages.extend(_decode_continuation(continuation.value).messages)
    _, current_messages = _lower_chat_input(req.input)
    rejection, pending_call_ids = _tool_history_state(
        [*history_messages, *current_messages]
    )
    if rejection is not None:
        return rejection
    if pending_call_ids:
        return "Anthropic request leaves tool_use blocks without immediate results"
    return None


def _has_system(input_: ChatInput) -> bool:
    return any(message.role == "system" for message in input_.messages)


def _validate_function_tools(req: Request) -> str | None:
    if (req.tools.choice is not None or req.tools.parallel is not None) and not (
        req.tools.functions
    ):
        return "tool choice and parallel controls require function tools"
    try:
        for tool in req.tools.functions:
            _function_tool_to_wire(tool)
        if req.tools.choice is not None:
            choice = _tool_choice_to_wire(req.tools.choice)
            if choice.get("type") == "none" and req.tools.parallel is not None:
                return "parallel tool control is meaningless with tool_choice='none'"
    except DialectError as exc:
        return str(exc)
    return None


def _structured_output(req: Request, text: str) -> StructuredOutput | None:
    if req.structured_output.format is None:
        return None
    try:
        return StructuredOutput(_strict_json_loads(text))
    except (TypeError, ValueError) as exc:
        raise OutputContractError("structured output is not valid JSON") from exc


def _usage_stats(raw: object) -> UsageStats:
    usage = _json_mapping(raw, "response.usage")
    allowed = {
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "cache_creation",
        "server_tool_use",
        "service_tier",
        "inference_geo",
        "output_tokens_details",
    }
    unknown = set(usage) - allowed
    if unknown:
        # Unknown accounting fields can change totals.  Refuse to pretend the
        # persisted usage record is complete until they have an explicit mapping.
        raise OutputContractError(
            "response.usage has unsupported field(s): " + ", ".join(sorted(unknown))
        )
    uncached = _nonnegative_int(
        usage.get("input_tokens"), "response.usage.input_tokens"
    )
    output = _nonnegative_int(
        usage.get("output_tokens"), "response.usage.output_tokens"
    )
    reported_cache_creation = usage.get("cache_creation_input_tokens")
    reported_cache_read = usage.get("cache_read_input_tokens")
    cache_creation = 0 if reported_cache_creation is None else reported_cache_creation
    cache_read = 0 if reported_cache_read is None else reported_cache_read
    cache_creation_count = _nonnegative_int(
        cache_creation, "response.usage.cache_creation_input_tokens"
    )
    cache_read_count = _nonnegative_int(
        cache_read, "response.usage.cache_read_input_tokens"
    )
    cache_creation_details = usage.get("cache_creation")
    if cache_creation_details is not None:
        details = _json_mapping(cache_creation_details, "response.usage.cache_creation")
        expected_detail_keys = {
            "ephemeral_1h_input_tokens",
            "ephemeral_5m_input_tokens",
        }
        if set(details) != expected_detail_keys:
            raise OutputContractError(
                "response.usage.cache_creation has unsupported fields"
            )
        cache_detail_total = sum(
            _nonnegative_int(details[key], f"response.usage.cache_creation.{key}")
            for key in sorted(expected_detail_keys)
        )
        if reported_cache_creation is None:
            cache_creation_count = cache_detail_total
        elif cache_detail_total != cache_creation_count:
            raise OutputContractError(
                "response.usage cache-creation breakdown disagrees with its total"
            )
    server_tool_usage = usage.get("server_tool_use")
    if server_tool_usage is not None:
        server_details = _json_mapping(
            server_tool_usage, "response.usage.server_tool_use"
        )
        expected_server_keys = {"web_fetch_requests", "web_search_requests"}
        if set(server_details) != expected_server_keys:
            raise OutputContractError(
                "response.usage.server_tool_use has unsupported fields"
            )
        server_counts = [
            _nonnegative_int(
                server_details[key], f"response.usage.server_tool_use.{key}"
            )
            for key in sorted(expected_server_keys)
        ]
        if any(server_counts):
            raise OutputContractError(
                "Anthropic response reports server-tool usage that has no "
                "shared-IR mapping"
            )
    service_tier = usage.get("service_tier")
    if service_tier is not None and service_tier not in {
        "standard",
        "priority",
        "batch",
    }:
        raise OutputContractError("response.usage.service_tier is invalid")
    inference_geo = usage.get("inference_geo")
    if inference_geo is not None and not isinstance(inference_geo, str):
        raise OutputContractError("response.usage.inference_geo must be a string")
    reasoning_tokens: int | None = None
    raw_output_details = usage.get("output_tokens_details")
    if raw_output_details is not None:
        details_mapping = _json_mapping(
            raw_output_details, "response.usage.output_tokens_details"
        )
        if set(details_mapping) != {"thinking_tokens"}:
            raise OutputContractError(
                "response.usage.output_tokens_details has unsupported fields"
            )
        reasoning_tokens = _nonnegative_int(
            details_mapping.get("thinking_tokens"),
            "response.usage.output_tokens_details.thinking_tokens",
        )
        if reasoning_tokens > output:
            raise OutputContractError(
                "response.usage thinking_tokens exceeds inclusive output_tokens"
            )
    input_total = uncached + cache_creation_count + cache_read_count
    return UsageStats(
        input_tokens=input_total,
        output_tokens=output,
        total_tokens=input_total + output,
        reasoning_tokens=reasoning_tokens,
        cached_tokens=cache_read_count if reported_cache_read is not None else None,
    )


def _validated_content_block(
    raw: object, index: int
) -> tuple[dict[str, JSONValue], str | None, FunctionToolCall | None, bool]:
    path = f"response.content[{index}]"
    block = _json_mapping(raw, path)
    block_type = block.get("type")
    if block_type == "text":
        allowed = {"type", "text", "citations"}
        if set(block) - allowed:
            raise OutputContractError(f"{path} has unsupported text fields")
        text = block.get("text")
        if not isinstance(text, str):
            raise OutputContractError(f"{path}.text must be a string")
        citations = block.get("citations")
        if citations is not None and citations != []:
            raise OutputContractError(
                "Anthropic citations have no lossless provider-neutral mapping"
            )
        canonical: dict[str, JSONValue] = {"type": "text", "text": text}
        if "citations" in block:
            canonical["citations"] = copy_json_value(citations, f"{path}.citations")
        return canonical, text, None, False
    if block_type == "thinking":
        if set(block) != {"type", "thinking", "signature"}:
            raise OutputContractError(f"{path} has unsupported thinking fields")
        thinking = block.get("thinking")
        if not isinstance(thinking, str):
            raise OutputContractError(f"{path}.thinking must be a string")
        signature = _required_string(block.get("signature"), f"{path}.signature")
        return (
            {"type": "thinking", "thinking": thinking, "signature": signature},
            thinking,
            None,
            True,
        )
    if block_type == "redacted_thinking":
        if set(block) != {"type", "data"}:
            raise OutputContractError(
                f"{path} has unsupported redacted-thinking fields"
            )
        data = _required_string(block.get("data"), f"{path}.data")
        return {"type": "redacted_thinking", "data": data}, None, None, True
    if block_type == "tool_use":
        allowed = {"type", "id", "name", "input", "caller", "toolset_name"}
        if set(block) - allowed:
            raise OutputContractError(f"{path} has unsupported tool_use fields")
        if "caller" in block or "toolset_name" in block:
            raise OutputContractError(
                "Anthropic delegated/toolset tool calls have no shared-IR mapping"
            )
        call_id = _required_string(block.get("id"), f"{path}.id")
        name = _required_string(block.get("name"), f"{path}.name")
        arguments = block.get("input")
        if not isinstance(arguments, Mapping):
            raise OutputContractError(f"{path}.input must be an object")
        copied_arguments = copy_json_value(arguments, f"{path}.input")
        canonical = {
            "type": "tool_use",
            "id": call_id,
            "name": name,
            "input": copied_arguments,
        }
        return (
            canonical,
            None,
            FunctionToolCall(call_id, name, copied_arguments),
            False,
        )
    raise OutputContractError(
        f"Anthropic content block type {block_type!r} has no shared-IR mapping"
    )


def _opaque_history_bundle(
    context: _ExecutionContext,
    output_content: Sequence[Mapping[str, JSONValue]],
    *,
    saw_opaque_reasoning: bool,
    stop_reason: str,
) -> str | None:
    raw_thinking = context.request_params.get("thinking")
    thinking: JSONValue = None
    if raw_thinking is not None:
        thinking = copy_json_value(raw_thinking, "request.thinking")
        try:
            _validate_continuation_thinking(thinking)
        except DialectError as exc:
            raise OutputContractError(
                "final Anthropic request had invalid thinking state"
            ) from exc
    effort: JSONValue = None
    raw_output_config = context.request_params.get("output_config")
    if raw_output_config is not None:
        output_config = _json_mapping(raw_output_config, "request.output_config")
        effort = output_config.get("effort")
        if effort is not None and (
            not isinstance(effort, str) or effort not in _REASONING_EFFORTS
        ):
            raise OutputContractError("final Anthropic request had invalid effort")
    raw_tools = context.request_params.get("tools")
    tools: JSONValue = None
    if raw_tools is not None:
        if not isinstance(raw_tools, list) or not raw_tools:
            raise OutputContractError("final Anthropic request had invalid tools")
        tools = copy_json_value(raw_tools, "request.tools")
        try:
            _validate_replay_tools(
                [
                    _json_mapping(tool, f"request.tools[{index}]")
                    for index, tool in enumerate(raw_tools)
                ]
            )
        except (DialectError, OutputContractError) as exc:
            raise OutputContractError(
                "final Anthropic request had invalid tools"
            ) from exc
    # ``pause_turn`` is intentionally rejected before this helper because PR 3
    # does not bind the server-hosted tools that produce it.
    open_turn = stop_reason == "tool_use"
    should_preserve = (
        saw_opaque_reasoning
        or context.request.session.opaque_continuation is not None
        or (open_turn and (thinking is not None or effort is not None))
    )
    if not should_preserve:
        return None
    messages: list[JSONValue] = [
        copy_json_value(message, f"request.messages[{index}]")
        for index, message in enumerate(context.messages)
    ]
    # Anthropic documents an empty ``end_turn`` response as a valid outcome and
    # instructs callers to append a new user prompt without replaying that empty
    # assistant response.  Keeping ``content: []`` here would also create an
    # opaque continuation that this dialect's own replay validator rejects.
    if output_content:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    copy_json_value(block, f"response.content[{index}]")
                    for index, block in enumerate(output_content)
                ],
            }
        )
    return json.dumps(
        {
            "version": _OPAQUE_CONTINUATION_VERSION,
            "system": [
                copy_json_value(block, f"request.system[{index}]")
                for index, block in enumerate(context.system)
            ],
            "messages": messages,
            "tools": tools,
            "open_turn": open_turn,
            "reasoning": {"thinking": thinking, "effort": effort},
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _headers(connection: AsyncHTTPJSONConnection) -> dict[str, str]:
    headers = {
        "anthropic-version": _ANTHROPIC_VERSION,
        "accept": "application/json",
    }
    if connection.credential is not None:
        headers["x-api-key"] = connection.credential
    return headers


def _execution_context(
    prepared: PreparedRequest,
    context: _AnthropicContext,
    expected_model: str,
) -> tuple[dict[str, JSONValue], bool, _ExecutionContext]:
    body = prepared.thaw_body()
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise DialectError("Anthropic prepared body has an invalid model")
    if model != expected_model:
        raise DialectError(
            "Anthropic prepared body model does not match the bound model"
        )
    stream = body.get("stream")
    if not isinstance(stream, bool):
        raise DialectError("Anthropic prepared body has an invalid stream flag")
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list):
        raise DialectError("Anthropic prepared body has invalid messages")
    raw_system = body.get("system", [])
    if not isinstance(raw_system, list):
        raise DialectError("Anthropic prepared body has invalid system content")
    messages = tuple(
        _json_mapping(item, f"prepared.messages[{index}]")
        for index, item in enumerate(raw_messages)
    )
    system = tuple(
        _json_mapping(item, f"prepared.system[{index}]")
        for index, item in enumerate(raw_system)
    )
    continuation = context.request.session.opaque_continuation
    if continuation is not None:
        mismatch = _continuation_tools_mismatch(continuation, body)
        if mismatch is not None:
            raise DialectError(mismatch)
    request_params = {
        key: copy_json_value(value, f"prepared.{key}")
        for key, value in body.items()
        if key not in {"model", "messages", "system"}
    }
    return (
        body,
        stream,
        _ExecutionContext(
            request=deepcopy(context.request),
            system=system,
            messages=messages,
            request_params=request_params,
        ),
    )


def _parse_json_response(response: httpx.Response) -> dict[str, JSONValue]:
    try:
        raw = _strict_json_loads(response.content)
    except (UnicodeDecodeError, ValueError) as exc:
        raise OutputContractError("Anthropic response is not valid JSON") from exc
    return _json_mapping(raw, "response")


async def _sse_events(response: httpx.Response) -> AsyncIterator[dict[str, JSONValue]]:
    event_name: str | None = None
    data_lines: list[str] = []

    def finish_event() -> dict[str, JSONValue] | None:
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = None
            return None
        data = "\n".join(data_lines)
        data_lines = []
        try:
            parsed = _strict_json_loads(data)
        except ValueError as exc:
            raise OutputContractError("Anthropic SSE data is not valid JSON") from exc
        event = _json_mapping(parsed, "stream event")
        payload_type = event.get("type")
        if event_name is not None and event_name != payload_type:
            raise OutputContractError(
                "Anthropic SSE event name disagrees with payload type"
            )
        event_name = None
        return event

    async for line in response.aiter_lines():
        if line == "":
            event = finish_event()
            if event is not None:
                yield event
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)
        elif field not in {"id", "retry"}:
            raise OutputContractError(f"unsupported Anthropic SSE field {field!r}")
    event = finish_event()
    if event is not None:
        yield event


def _apply_stream_delta(
    block: dict[str, JSONValue], delta: Mapping[str, JSONValue], path: str
) -> None:
    delta_type = delta.get("type")
    block_type = block.get("type")
    if delta_type == "text_delta" and block_type == "text":
        _reject_unknown_fields(delta, frozenset({"type", "text"}), path)
        text = delta.get("text")
        if not isinstance(text, str):
            raise OutputContractError(f"{path}.text must be a string")
        block["text"] = cast(str, block.get("text", "")) + text
        return
    if delta_type == "thinking_delta" and block_type == "thinking":
        _reject_unknown_fields(delta, frozenset({"type", "thinking"}), path)
        if block.get("_signature_delta_seen") is True:
            raise OutputContractError(f"{path} followed the terminal signature delta")
        thinking = delta.get("thinking")
        if not isinstance(thinking, str):
            raise OutputContractError(f"{path}.thinking must be a string")
        block["thinking"] = cast(str, block.get("thinking", "")) + thinking
        return
    if delta_type == "signature_delta" and block_type == "thinking":
        _reject_unknown_fields(delta, frozenset({"type", "signature"}), path)
        if block.get("_signature_delta_seen") is True:
            raise OutputContractError(f"{path} duplicated the signature delta")
        signature = delta.get("signature")
        if not isinstance(signature, str):
            raise OutputContractError(f"{path}.signature must be a string")
        block["signature"] = cast(str, block.get("signature", "")) + signature
        block["_signature_delta_seen"] = True
        return
    if delta_type == "input_json_delta" and block_type == "tool_use":
        _reject_unknown_fields(delta, frozenset({"type", "partial_json"}), path)
        partial = delta.get("partial_json")
        if not isinstance(partial, str):
            raise OutputContractError(f"{path}.partial_json must be a string")
        block["_partial_json"] = cast(str, block.get("_partial_json", "")) + partial
        return
    raise OutputContractError(
        f"{path} type {delta_type!r} does not match block type {block_type!r}"
    )


def _finish_stream_block(block: dict[str, JSONValue], path: str) -> None:
    block.pop("_signature_delta_seen", None)
    if block.get("type") != "tool_use":
        return
    partial = block.pop("_partial_json", "")
    if not isinstance(partial, str):
        raise OutputContractError(f"{path} tool JSON state is invalid")
    if not partial:
        existing = block.get("input")
        if isinstance(existing, Mapping):
            return
        raise OutputContractError(f"{path}.input is absent")
    existing = block.get("input")
    if not isinstance(existing, Mapping) or existing:
        raise OutputContractError(
            f"{path}.input must start as an empty object before JSON deltas"
        )
    try:
        parsed = _strict_json_loads(partial)
    except ValueError as exc:
        raise OutputContractError(f"{path}.input is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise OutputContractError(f"{path}.input must be an object")
    block["input"] = copy_json_value(parsed, f"{path}.input")


async def _terminal_stream_message(response: httpx.Response) -> dict[str, JSONValue]:
    message: dict[str, JSONValue] | None = None
    blocks: dict[int, dict[str, JSONValue]] = {}
    stopped_blocks: set[int] = set()
    saw_delta = False
    saw_stop = False
    async for event in _sse_events(response):
        event_type = event.get("type")
        if not isinstance(event_type, str) or event_type not in _STREAM_EVENT_KEYS:
            raise OutputContractError(
                f"unsupported Anthropic stream event {event_type!r}"
            )
        _reject_unknown_fields(event, _STREAM_EVENT_KEYS[event_type], "stream event")
        if saw_stop:
            raise OutputContractError(
                "Anthropic stream emitted data after message_stop"
            )
        if event_type == "ping":
            continue
        if event_type == "error":
            error = event.get("error")
            detail = ""
            if isinstance(error, Mapping):
                error_message = cast(Mapping[str, object], error).get("message")
                if isinstance(error_message, str):
                    detail = f": {error_message}"
            raise OutputContractError(f"Anthropic stream error{detail}")
        if event_type == "message_start":
            if message is not None:
                raise OutputContractError(
                    "Anthropic stream emitted two message_start events"
                )
            message = _json_mapping(event.get("message"), "message_start.message")
            initial_content = message.get("content")
            if initial_content is not None and initial_content != []:
                raise OutputContractError(
                    "Anthropic message_start content must be empty"
                )
            if message.get("stop_reason") is not None:
                raise OutputContractError(
                    "Anthropic message_start stop_reason must be null"
                )
            if message.get("stop_sequence") is not None:
                raise OutputContractError(
                    "Anthropic message_start stop_sequence must be null"
                )
            message["content"] = []
            continue
        if message is None:
            raise OutputContractError("Anthropic stream event preceded message_start")
        if event_type == "content_block_start":
            if saw_delta:
                raise OutputContractError(
                    "Anthropic stream emitted content after message_delta"
                )
            index = _nonnegative_int(event.get("index"), "content_block_start.index")
            if index in blocks:
                raise OutputContractError(
                    f"Anthropic stream duplicated content block {index}"
                )
            blocks[index] = _json_mapping(
                event.get("content_block"), f"content_block_start[{index}]"
            )
            continue
        if event_type == "content_block_delta":
            if saw_delta:
                raise OutputContractError(
                    "Anthropic stream emitted content after message_delta"
                )
            index = _nonnegative_int(event.get("index"), "content_block_delta.index")
            if index not in blocks or index in stopped_blocks:
                raise OutputContractError(
                    f"Anthropic delta targets inactive content block {index}"
                )
            delta = _json_mapping(event.get("delta"), f"content_block_delta[{index}]")
            _apply_stream_delta(blocks[index], delta, f"content_block_delta[{index}]")
            continue
        if event_type == "content_block_stop":
            if saw_delta:
                raise OutputContractError(
                    "Anthropic stream emitted content after message_delta"
                )
            index = _nonnegative_int(event.get("index"), "content_block_stop.index")
            if index not in blocks or index in stopped_blocks:
                raise OutputContractError(
                    f"Anthropic stop targets inactive content block {index}"
                )
            _finish_stream_block(blocks[index], f"content_block[{index}]")
            stopped_blocks.add(index)
            continue
        if event_type == "message_delta":
            if saw_delta:
                raise OutputContractError(
                    "Anthropic stream emitted two message_delta events"
                )
            delta = _json_mapping(event.get("delta"), "message_delta.delta")
            _reject_unknown_fields(delta, _MESSAGE_DELTA_KEYS, "message_delta.delta")
            stop_reason = delta.get("stop_reason")
            if not isinstance(stop_reason, str) or not stop_reason:
                raise OutputContractError(
                    "message_delta.delta.stop_reason must be a non-empty string"
                )
            _validate_stop_details(stop_reason, delta.get("stop_details"))
            saw_delta = True
            if stop_reason == "refusal":
                # Refusal is a terminal non-success outcome.  Deliberately
                # discard every earlier block, whether still open or already
                # closed, so streamed and non-streamed refusals both lift to an
                # empty scoreable answer and blocked partial content cannot
                # enter task scoring.  This is a normalization policy, not an
                # implication that Anthropic guarantees either stream shape.
                blocks.clear()
                stopped_blocks.clear()
            elif set(blocks) != stopped_blocks:
                raise OutputContractError(
                    "Anthropic message_delta preceded content_block_stop"
                )
            if delta.get("container") is not None:
                raise OutputContractError(
                    "Anthropic response container has no shared-IR mapping"
                )
            message["stop_reason"] = stop_reason
            message["stop_sequence"] = delta.get("stop_sequence")
            message["stop_details"] = delta.get("stop_details")
            initial_usage = _json_mapping(
                message.get("usage"), "message_start.message.usage"
            )
            final_usage = _json_mapping(event.get("usage"), "message_delta.usage")
            _nonnegative_int(
                final_usage.get("output_tokens"),
                "message_delta.usage.output_tokens",
            )
            message["usage"] = {
                **initial_usage,
                **{
                    key: value
                    for key, value in final_usage.items()
                    if value is not None
                },
            }
            continue
        if event_type == "message_stop":
            saw_stop = True
            continue
    if message is None:
        raise OutputContractError("Anthropic stream omitted message_start")
    if not saw_delta:
        raise OutputContractError("Anthropic stream omitted message_delta")
    if not saw_stop:
        raise OutputContractError("Anthropic stream omitted message_stop")
    if set(blocks) != stopped_blocks:
        raise OutputContractError("Anthropic stream left a content block unfinished")
    if set(blocks) != set(range(len(blocks))):
        raise OutputContractError(
            "Anthropic stream content block indexes are not dense"
        )
    message["content"] = [blocks[index] for index in range(len(blocks))]
    return message


class AnthropicMessagesDialect:
    """Executable native Anthropic Messages adapter."""

    dialect_id = "anthropic_messages"
    connection_family = "async_http_json"
    capability_decisions = CAPABILITY_DECISIONS
    output_contract = OUTPUT_CONTRACT
    CAPABILITIES = frozenset(
        {
            Capability.Chat,
            Capability.FunctionCalling,
            Capability.Reasoning,
            Capability.ReasoningEffort,
            Capability.StructuredOutput,
            Capability.Prefill,
        }
    )

    def __init__(
        self, connection: AsyncHTTPJSONConnection, requested_model_id: str
    ) -> None:
        if not isinstance(connection, AsyncHTTPJSONConnection):
            raise TypeError("anthropic_messages requires an AsyncHTTPJSONConnection")
        if not requested_model_id:
            raise ValueError("requested_model_id must not be empty")
        self._connection = connection
        self._requested_model_id = requested_model_id

    @property
    def capabilities(self) -> frozenset[Capability]:
        """Deprecated enum view retained during the compatibility cycle."""

        return self.CAPABILITIES

    def validate_request(
        self, req: Request, audit: RequestAudit, plan: RuntimePlanView
    ) -> None:
        audit.require_matches_request(req)
        del plan
        if req.sampling.max_tokens is None:
            raise DialectError("Anthropic Messages requires sampling.max_tokens")
        input_rejection = (
            _message_rejection(req.input) if isinstance(req.input, ChatInput) else None
        )
        image_rejection = None
        if isinstance(req.input, ChatInput):
            image_rejection = next(
                (
                    rejection
                    for message in req.input.messages
                    for part in message.content
                    if isinstance(part, ImagePart)
                    if (rejection := _image_rejection(part)) is not None
                ),
                None,
            )
        tool_rejection = _validate_function_tools(req)
        tool_history_rejection = (
            _tool_history_rejection(req)
            if input_rejection is None and image_rejection is None
            else None
        )
        thinking_config = _thinking_config(req)
        has_prefill = isinstance(req.input, ChatInput) and _has_assistant_prefill(
            req.input
        )
        forced_tool_choice = False
        if req.tools.choice is not None and tool_rejection is None:
            forced_tool_choice = _tool_choice_to_wire(req.tools.choice).get("type") in {
                "any",
                "tool",
            }
        continuation_reasoning_rejection = _continuation_reasoning_rejection(req)
        continuation_with_system = (
            req.session.opaque_continuation is not None
            and isinstance(req.input, ChatInput)
            and _has_system(req.input)
        )
        for path in audit.active_paths:
            if path == "dialect_options":
                audit.noop(path, "an empty matching options mapping has no effect")
            elif path in {"input.completion", "input.completion.suffix"}:
                audit.rejected(path, "anthropic_messages requires ChatInput")
            elif path == "input.chat" and input_rejection is not None:
                audit.rejected(path, input_rejection)
            elif path == "input.chat" and tool_history_rejection is not None:
                audit.rejected(path, tool_history_rejection)
            elif path == "input.chat" and continuation_with_system:
                audit.rejected(
                    path,
                    "opaque continuation already carries the prior system prompt; "
                    "new system messages would make authority ambiguous",
                )
            elif (
                path == "input.chat"
                and has_prefill
                and req.structured_output.format is not None
            ):
                audit.rejected(
                    path,
                    "Anthropic JSON-schema output is incompatible with assistant "
                    "prefill",
                )
            elif path == "input.chat" and has_prefill and thinking_config is not None:
                audit.rejected(
                    path,
                    "Anthropic thinking is incompatible with assistant prefill",
                )
            elif image_rejection is not None and path == image_rejection[0]:
                audit.rejected(path, image_rejection[1])
            elif path in {
                "scoring.input_scoring",
                "scoring.sampled_logprobs",
                "scoring.top_logprobs",
            }:
                audit.rejected(path, "Anthropic Messages does not provide logprobs")
            elif path == "sampling.n":
                audit.rejected(path, "Anthropic Messages returns one choice per call")
            elif path == "sampling.seed":
                audit.rejected(path, "Anthropic Messages has no per-request seed")
            elif path in {"sampling.frequency_penalty", "sampling.presence_penalty"}:
                audit.rejected(path, "Anthropic Messages has no equivalent penalty")
            elif (
                path == "sampling.max_tokens"
                and req.sampling.max_tokens is not None
                and req.sampling.max_tokens < 0
            ):
                audit.rejected(path, "Anthropic max_tokens must be non-negative")
            elif (
                path == "sampling.temperature"
                and req.sampling.temperature is not None
                and not 0 <= req.sampling.temperature <= 1
            ):
                audit.rejected(path, "Anthropic temperature must be between 0 and 1")
            elif path == "sampling.temperature" and thinking_config is not None:
                audit.rejected(
                    path,
                    "Anthropic sampling temperature is incompatible with explicit "
                    "thinking",
                )
            elif (
                path == "sampling.top_p"
                and req.sampling.top_p is not None
                and not 0 <= req.sampling.top_p <= 1
            ):
                audit.rejected(path, "Anthropic top_p must be between 0 and 1")
            elif (
                path == "sampling.top_p"
                and thinking_config is not None
                and req.sampling.top_p is not None
                and req.sampling.top_p < 0.95
            ):
                audit.rejected(
                    path,
                    "Anthropic top_p must be between 0.95 and 1 with explicit thinking",
                )
            elif (
                path == "sampling.top_k"
                and req.sampling.top_k is not None
                and req.sampling.top_k < 0
            ):
                audit.rejected(path, "Anthropic top_k must be non-negative")
            elif path == "sampling.top_k" and thinking_config is not None:
                audit.rejected(
                    path,
                    "Anthropic top_k is incompatible with explicit thinking",
                )
            elif (
                path == "sampling.stop"
                and req.sampling.stop is not None
                and any(not item for item in req.sampling.stop)
            ):
                audit.rejected(path, "Anthropic stop sequences must not be empty")
            elif (
                path == "reasoning.effort"
                and req.reasoning.effort not in _REASONING_EFFORTS
            ):
                audit.rejected(
                    path,
                    f"unsupported Anthropic reasoning effort {req.reasoning.effort!r}",
                )
            elif (
                path == "reasoning.budget_tokens"
                and req.reasoning.budget_tokens is not None
                and req.reasoning.budget_tokens < 1024
            ):
                audit.rejected(path, "Anthropic thinking budget must be at least 1024")
            elif (
                path == "sampling.max_tokens"
                and req.reasoning.budget_tokens is not None
                and req.sampling.max_tokens is not None
                and req.reasoning.budget_tokens >= req.sampling.max_tokens
            ):
                audit.rejected(
                    path,
                    "Anthropic max_tokens must exceed reasoning.budget_tokens",
                )
            elif path == "reasoning.summary" and req.reasoning.summary not in {
                "none",
                "auto",
            }:
                audit.rejected(
                    path,
                    "Anthropic Messages supports only summary='none' or 'auto'",
                )
            elif (
                path in {"tools.functions", "tools.choice", "tools.parallel"}
                and tool_rejection is not None
            ):
                audit.rejected(path, tool_rejection)
            elif (
                path == "tools.choice"
                and req.reasoning.budget_tokens is not None
                and forced_tool_choice
            ):
                audit.rejected(
                    path,
                    "manual Anthropic thinking supports only tool_choice='auto' "
                    "or 'none'",
                )
            elif path == "tools.hosted":
                audit.rejected(
                    path,
                    "Anthropic hosted tools are not supported by anthropic_messages",
                )
            elif (
                path == "structured_output.format"
                and req.structured_output.format != "json_schema"
            ):
                audit.rejected(path, "Anthropic structured output requires json_schema")
            elif path == "structured_output.name":
                audit.rejected(path, "Anthropic output_config has no schema name field")
            elif (
                path == "structured_output.strict"
                and req.structured_output.strict is False
            ):
                audit.rejected(
                    path, "Anthropic JSON-schema output is always schema constrained"
                )
            elif path == "session.previous_response_id":
                audit.rejected(
                    path, "Anthropic Messages has no previous-response-id continuation"
                )
            elif (
                path == "session.opaque_continuation"
                and continuation_reasoning_rejection is not None
            ):
                audit.rejected(path, continuation_reasoning_rejection)
            elif path.startswith("dialect_options."):
                key = path.removeprefix("dialect_options.")
                if not key:
                    audit.rejected(path, "dialect option keys must be non-empty")
                elif key in _IR_OWNED_BODY_KEYS:
                    audit.rejected(
                        path,
                        f"{key!r} must use its canonical provider-neutral field",
                    )
                else:
                    audit.rejected(
                        path,
                        "anthropic_messages does not support raw Anthropic request "
                        "passthrough",
                    )

    def prepare(self, req: Request, audit: RequestAudit) -> PreparedRequest:
        audit.require_matches_request(req)
        audit.raise_rejections()
        if not isinstance(req.input, ChatInput):
            raise DialectError("anthropic_messages requires ChatInput")
        if req.sampling.max_tokens is None:
            raise DialectError("Anthropic Messages requires sampling.max_tokens")

        active_paths = frozenset(audit.active_paths)

        def consume(path: str, *wire: WireEvidence) -> None:
            if path in active_paths and path not in audit.decisions:
                audit.consumed(path, *wire)

        continuation = req.session.opaque_continuation
        input_verifier = _AnthropicInputVerifier(req.input, continuation)
        history_system: list[dict[str, JSONValue]] = []
        history_messages: list[dict[str, JSONValue]] = []
        if continuation is not None:
            decoded = _decode_continuation(continuation.value)
            history_system = list(decoded.system)
            history_messages = list(decoded.messages)
        current_system, current_messages = _lower_chat_input(req.input)
        system = [*history_system, *current_system]
        messages = [*history_messages, *current_messages]
        for path in (
            "input.chat",
            "input.chat.message.name",
            "input.modality.text",
            "input.modality.image",
            "input.modality.image.media_type",
            "input.modality.tool_call",
            "input.modality.tool_result",
            "input.modality.tool_result.is_error",
        ):
            consume(path, input_verifier)

        body: dict[str, JSONValue] = {
            "model": self._requested_model_id,
            "max_tokens": req.sampling.max_tokens,
            "messages": cast(JSONValue, messages),
            "stream": req.scheduling.stream,
        }
        if system:
            body["system"] = cast(JSONValue, system)
        consume("sampling.max_tokens", WireObservation(("max_tokens",)))
        for name in ("temperature", "top_p", "top_k"):
            value = getattr(req.sampling, name)
            if value is not None:
                body[name] = value
                consume(f"sampling.{name}", WireObservation((name,)))
        if req.sampling.stop is not None:
            body["stop_sequences"] = list(req.sampling.stop)
            consume("sampling.stop", WireObservation(("stop_sequences",)))

        output_config: dict[str, JSONValue] = {}
        thinking = _thinking_config(req)
        if req.reasoning.budget_tokens is not None:
            consume(
                "reasoning.budget_tokens",
                WireObservation(("thinking", "budget_tokens")),
            )
        if req.reasoning.effort is not None:
            output_config["effort"] = req.reasoning.effort
            consume(
                "reasoning.effort",
                WireObservation(("output_config", "effort")),
            )
        if req.reasoning.summary == "auto":
            assert thinking is not None
            consume(
                "reasoning.summary",
                _ReasoningSummaryVerifier(
                    "auto", req.reasoning.budget_tokens is not None
                ),
            )
        elif req.reasoning.summary == "none":
            assert thinking is not None
            consume(
                "reasoning.summary",
                _ReasoningSummaryVerifier(
                    "none", req.reasoning.budget_tokens is not None
                ),
            )
        if thinking is not None:
            body["thinking"] = thinking

        if req.tools.functions:
            body["tools"] = [
                _function_tool_to_wire(tool) for tool in req.tools.functions
            ]
            consume("tools.functions", _AnthropicToolsVerifier(req.tools.functions))
        choice = _tool_choice(req)
        if choice is not None:
            body["tool_choice"] = choice
            choice_verifier = _AnthropicToolChoiceVerifier(
                req.tools.choice, req.tools.parallel
            )
            consume("tools.choice", choice_verifier)
            consume("tools.parallel", choice_verifier)

        if req.structured_output.format is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": dict(req.structured_output.schema or {}),
            }
            consume(
                "structured_output.format",
                WireObservation(("output_config", "format", "type")),
            )
            consume(
                "structured_output.schema",
                WireObservation(("output_config", "format", "schema")),
            )
            if req.structured_output.strict is True:
                audit.noop(
                    "structured_output.strict",
                    "Anthropic JSON-schema output is always schema constrained",
                )
        if output_config:
            body["output_config"] = output_config

        if continuation is not None:
            consume("session.opaque_continuation", input_verifier)
        if req.scheduling.stream:
            consume("scheduling.stream", WireObservation(("stream",)))

        return PreparedRequest(
            operation="messages.create",
            body=body,
            context=_AnthropicContext(deepcopy(req)),
        )

    def _lift(self, raw: object, context: _ExecutionContext) -> Response:
        message = _json_mapping(raw, "response")
        _reject_unknown_fields(message, _MESSAGE_RESPONSE_KEYS, "response")
        if message.get("type") != "message":
            raise OutputContractError("response.type must be 'message'")
        if message.get("role") != "assistant":
            raise OutputContractError("response.role must be 'assistant'")
        _required_string(message.get("id"), "response.id")
        response_model = _required_string(message.get("model"), "response.model")
        stop_reason = _required_string(
            message.get("stop_reason"), "response.stop_reason"
        )
        if stop_reason not in _STOP_REASONS:
            raise OutputContractError(
                f"response.stop_reason {stop_reason!r} is unsupported"
            )
        if message.get("container") is not None:
            raise OutputContractError(
                "Anthropic response container has no shared-IR mapping"
            )
        _validate_stop_details(stop_reason, message.get("stop_details"))
        stop_sequence = message.get("stop_sequence")
        if stop_reason == "stop_sequence":
            _required_string(stop_sequence, "response.stop_sequence")
        elif stop_sequence is not None:
            _optional_string(stop_sequence, "response.stop_sequence")
            raise OutputContractError(
                "response.stop_sequence must be null unless stop_reason is "
                "'stop_sequence'"
            )

        usage = _usage_stats(message.get("usage"))
        if stop_reason == "refusal":
            # Content is discarded: Anthropic documents a refusal as arriving
            # before any output or mid-stream, and says to treat partial output
            # as incomplete either way. Validated first anyway, so a shape this
            # dialect cannot parse still raises here as it would anywhere else.
            for index, raw_block in enumerate(
                _sequence(message.get("content"), "response.content")
            ):
                _validated_content_block(raw_block, index)
            return Response(
                texts=("",),
                finish_reasons=(stop_reason,),
                usage=usage,
                request_params=context.request_params,
                response_model=response_model,
            )

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[FunctionToolCall] = []
        canonical_content: list[dict[str, JSONValue]] = []
        saw_opaque_reasoning = False
        for index, raw_block in enumerate(
            _sequence(message.get("content"), "response.content")
        ):
            block, text, call, is_opaque_reasoning = _validated_content_block(
                raw_block, index
            )
            canonical_content.append(block)
            if block["type"] == "text" and text is not None:
                text_parts.append(text)
            elif block["type"] == "thinking" and text is not None:
                thinking_parts.append(text)
            if call is not None:
                tool_calls.append(call)
            saw_opaque_reasoning = saw_opaque_reasoning or is_opaque_reasoning

        if (
            context.request.reasoning.budget_tokens is not None
            and context.request.session.opaque_continuation is None
            and saw_opaque_reasoning
            and canonical_content
            and canonical_content[0].get("type")
            not in {"thinking", "redacted_thinking"}
        ):
            raise OutputContractError(
                "Anthropic manual-thinking response must begin with a thinking block"
            )

        if (
            context.request.reasoning.budget_tokens is not None
            and not saw_opaque_reasoning
            and context.request.session.opaque_continuation is None
        ):
            raise OutputContractError(
                "Anthropic reply omitted required manual-thinking state"
            )
        if stop_reason == "tool_use" and not tool_calls:
            raise OutputContractError(
                "Anthropic stop_reason='tool_use' requires a tool_use content block"
            )
        if tool_calls and stop_reason != "tool_use":
            raise OutputContractError(
                "Anthropic tool_use content requires stop_reason='tool_use'"
            )
        call_ids = [call.call_id for call in tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise OutputContractError("Anthropic reply duplicated a tool_use id")
        response_message: Mapping[str, JSONValue] = {
            "role": "assistant",
            "content": cast(JSONValue, canonical_content),
        }
        history_rejection, _ = _tool_history_state(
            (*context.messages, response_message)
        )
        if history_rejection is not None:
            raise OutputContractError(
                f"Anthropic reply produced invalid tool history: {history_rejection}"
            )
        text = "".join(text_parts)
        if (
            context.request.reasoning.summary == "auto"
            and stop_reason not in {"max_tokens", "model_context_window_exceeded"}
            and any(block.get("type") == "thinking" for block in canonical_content)
            and not any(part.strip() for part in thinking_parts)
        ):
            raise OutputContractError(
                "Anthropic reply omitted the requested visible thinking summary"
            )
        opaque = _opaque_history_bundle(
            context,
            canonical_content,
            saw_opaque_reasoning=saw_opaque_reasoning,
            stop_reason=stop_reason,
        )
        reasoning: tuple[ReasoningOutput | None, ...] | None = None
        if opaque is not None:
            reasoning = (
                ReasoningOutput(
                    text="\n".join(thinking_parts) or None,
                    opaque_roundtrip=opaque,
                    thinking_tokens=(
                        usage.reasoning_tokens
                        if usage.reasoning_tokens is not None
                        else 0
                    ),
                ),
            )
        elif context.request.reasoning.summary == "auto":
            # Adaptive thinking is optional per turn.  Keep the choice-aligned
            # channel present without inventing reasoning when Claude skips it.
            reasoning = (None,)
        return Response(
            texts=(text,),
            reasoning=reasoning,
            finish_reasons=(stop_reason,),
            tool_calls=tuple(tool_calls) or None,
            structured_output=_structured_output(context.request, text),
            usage=usage,
            request_params=context.request_params,
            response_model=response_model,
        )

    async def execute(self, prepared: PreparedRequest) -> Response:
        if prepared.operation != "messages.create":
            raise DialectError(f"unexpected Anthropic operation {prepared.operation!r}")
        context = prepared.context
        if not isinstance(context, _AnthropicContext):
            raise DialectError("prepared Anthropic request has invalid context")
        body, stream, execution = _execution_context(
            prepared,
            context,
            self._requested_model_id,
        )
        headers = _headers(self._connection)
        if not stream:
            response = await self._connection.client.post(
                _MESSAGES_PATH,
                json=body,
                headers=headers,
            )
            response.raise_for_status()
            return self._lift(_parse_json_response(response), execution)

        stream_headers = {**headers, "accept": "text/event-stream"}
        async with self._connection.client.stream(
            "POST",
            _MESSAGES_PATH,
            json=body,
            headers=stream_headers,
        ) as response:
            if not response.is_success:
                await response.aread()
            response.raise_for_status()
            terminal = await _terminal_stream_message(response)
        return self._lift(terminal, execution)

    async def arun(self, req: Request) -> Response:
        """One-cycle direct path; canonical ``Model`` uses the split contract."""

        plan = _LegacyPlan()
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
