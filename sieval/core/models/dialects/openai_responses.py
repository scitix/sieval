"""OpenAI Responses API dialect for the provider-neutral model IR.

This adapter deliberately implements only semantics the shared IR can preserve.
Unsupported request fields and output item variants fail before they can be
silently reinterpreted.  Streaming is lifted from the single terminal event's
complete ``Response`` object, so the streaming and non-streaming paths share one
response parser.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

import json
import math
from collections.abc import AsyncIterable, Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, NoReturn, cast

from sieval.core.models._shared import copy_json_value, thaw_json_mapping
from sieval.core.models.capabilities import (
    CAPABILITY_KEYS,
    Capability,
    CapabilityKey,
    DialectCapabilityBinding,
    DialectCapabilityDecision,
    HostedToolsOptions,
    MultimodalInputOptions,
    ReasoningOptions,
    Supported,
    TopLogprobsOptions,
    Unsupported,
)
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
    validate_top_logprobs,
)
from sieval.core.models.ir import (
    ChatInput,
    ChatMessage,
    Citation,
    FunctionToolCall,
    HostedToolSpec,
    ImagePart,
    ReasoningOutput,
    Request,
    Response,
    ServerToolUse,
    StructuredOutput,
    TextPart,
    TokenLogprob,
    ToolCallPart,
    ToolResultPart,
    TopKEntry,
    UsageStats,
)
from sieval.core.types import JSONValue

_OPENAI_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)
_SUPPORTED_HOSTED_TOOL_KINDS = frozenset({"web_search", "web_search_preview"})
_SUPPORTED_IMAGE_MEDIA_TYPES = frozenset(
    {"image/gif", "image/jpeg", "image/png", "image/webp"}
)
_SUPPORTED_IMAGE_DETAILS = frozenset({"auto", "high", "low", "original"})
_OPAQUE_CONTINUATION_VERSION = 2

# Keep the accepted wire vocabulary local instead of deriving it from the SDK.
# AsyncOpenAI uses lenient response construction by default, so an unknown
# provider shape can be coerced into a known SDK model instead of being rejected;
# opaque continuation replay bypasses SDK parsing entirely.  Value-identical
# sets below remain separate because they mirror independent upstream enums.
_REASONING_ITEM_KEYS = frozenset(
    {"type", "id", "summary", "content", "encrypted_content", "status"}
)
_REASONING_ITEM_STATUSES = frozenset({"in_progress", "completed", "incomplete"})
_REASONING_TEXT_PART_KEYS = frozenset({"type", "text"})
_HISTORY_ITEM_TYPES = frozenset(
    {
        "message",
        "reasoning",
        "function_call",
        "function_call_output",
        "web_search_call",
    }
)
_MESSAGE_ITEM_KEYS = frozenset({"type", "id", "role", "status", "content"})
_MESSAGE_ROLES = frozenset({"system", "developer", "user", "assistant"})
_MESSAGE_ITEM_STATUSES = frozenset({"in_progress", "completed", "incomplete"})
_TERMINAL_ITEM_STATUSES = frozenset({"completed", "incomplete"})
_FUNCTION_CALL_ITEM_KEYS = frozenset(
    {"type", "id", "call_id", "name", "arguments", "status"}
)
_FUNCTION_CALL_OUTPUT_ITEM_KEYS = frozenset(
    {"type", "id", "call_id", "output", "status"}
)
_WEB_SEARCH_ITEM_KEYS = frozenset({"type", "id", "status", "action"})
_INPUT_TEXT_PART_KEYS = frozenset({"type", "text"})
_INPUT_IMAGE_PART_KEYS = frozenset({"type", "image_url", "detail"})
_OUTPUT_TEXT_PART_KEYS = frozenset({"type", "text", "annotations", "logprobs"})
_MISSING = object()


def _validate_top_logprobs_config(options: object) -> None:
    assert isinstance(options, TopLogprobsOptions)
    if options.minimum > 20:
        raise ValueError("openai_responses supports at most 20 top logprobs")


def _validate_reasoning_config(options: object) -> None:
    assert isinstance(options, ReasoningOptions)
    if options.budget_tokens is not None:
        raise ValueError("openai_responses does not support reasoning budget_tokens")
    if options.effort is not None and options.effort not in _OPENAI_REASONING_EFFORTS:
        raise ValueError(f"openai_responses does not support effort {options.effort!r}")


def _validate_hosted_tools_config(options: object) -> None:
    assert isinstance(options, HostedToolsOptions)
    unsupported = set(options.kinds) - _SUPPORTED_HOSTED_TOOL_KINDS
    if unsupported:
        raise ValueError(
            "openai_responses currently supports only web-search hosted tools; "
            f"unsupported kinds: {sorted(unsupported)}"
        )


def _validate_multimodal_config(options: object) -> None:
    assert isinstance(options, MultimodalInputOptions)
    unsupported = set(options.modalities) - {"image"}
    if unsupported:
        raise ValueError(
            f"openai_responses does not support modalities {sorted(unsupported)}"
        )


def _decisions() -> Mapping[CapabilityKey, DialectCapabilityDecision]:
    decisions: dict[CapabilityKey, DialectCapabilityDecision] = {
        "input_scoring": Unsupported("Responses cannot score input tokens"),
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
                _config_validator=_validate_top_logprobs_config,
            )
        ),
        "reasoning": Supported(
            DialectCapabilityBinding(
                "reasoning",
                request_leaves=("reasoning.effort", "reasoning.summary"),
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
        "hosted_tools": Supported(
            DialectCapabilityBinding(
                "hosted_tools",
                request_leaves=("tools.hosted",),
                response_channels=("server_tool_uses",),
                _config_validator=_validate_hosted_tools_config,
            )
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
        "stateful_session": Supported(
            DialectCapabilityBinding(
                "stateful_session",
                request_leaves=("session.previous_response_id",),
                response_channels=("session_id",),
            )
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
                    "input.modality.image.detail",
                    "input.modality.image.media_type",
                ),
                _config_validator=_validate_multimodal_config,
            )
        ),
        "prefill": Unsupported("Responses has no assistant-prefill field"),
        "fim": Unsupported("Responses has no completion-suffix field"),
    }
    if set(decisions) != set(CAPABILITY_KEYS):
        raise AssertionError("OpenAI Responses capability row is incomplete")
    return MappingProxyType(decisions)


def _tuple_validator(item_type: type, channel: str) -> Callable[[object], None]:
    def validate(value: object) -> None:
        if not isinstance(value, tuple) or not all(
            isinstance(item, item_type) for item in value
        ):
            raise OutputContractError(f"{channel} channel has invalid shape")

    return validate


CAPABILITY_DECISIONS = _decisions()


OUTPUT_CONTRACT = OutputContract(
    {
        "reasoning": OutputRule(Guarantee.PRESENT_OR_ERROR, validate_reasoning),
        "tool_calls": OutputRule(Guarantee.BEST_EFFORT, validate_tool_calls),
        "server_tool_uses": OutputRule(
            Guarantee.BEST_EFFORT,
            _tuple_validator(ServerToolUse, "server_tool_uses"),
        ),
        "structured_output": OutputRule(
            Guarantee.PRESENT_OR_ERROR, validate_structured_output
        ),
        "logprobs": OutputRule(
            Guarantee.PRESENT_OR_ERROR,
            _tuple_validator(TokenLogprob, "logprobs"),
        ),
        "top_logprobs": OutputRule(Guarantee.PRESENT_OR_ERROR, validate_top_logprobs),
        "input_scoring": OutputRule(Guarantee.NEVER),
        "citations": OutputRule(
            Guarantee.BEST_EFFORT,
            _tuple_validator(Citation, "citations"),
        ),
        "grounding": OutputRule(Guarantee.NEVER),
        "session_id": OutputRule(Guarantee.PRESENT_OR_ERROR),
        "usage": OutputRule(Guarantee.BEST_EFFORT),
    }
)


_UNSUPPORTED_REQUEST_PATHS = frozenset(
    {
        "sampling.top_k",
        "sampling.stop",
        "sampling.seed",
        "sampling.frequency_penalty",
        "sampling.presence_penalty",
        "sampling.n",
    }
)

_IR_OWNED_BODY_KEYS = BINDING_RESOURCE_KEYS | frozenset(
    {
        "model",
        "input",
        "instructions",
        "messages",
        "prompt",
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "temperature",
        "top_p",
        "top_k",
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
        "reasoning",
        "reasoning_effort",
        "tools",
        "functions",
        "tool_choice",
        "function_call",
        "parallel_tool_calls",
        "server_tools",
        "web_search_options",
        "text",
        "response_format",
        "previous_response_id",
        "session_id",
        "opaque_continuation",
        "suffix",
        "include",
        "stream",
        "stream_options",
        "extra_body",
    }
)


@dataclass(frozen=True)
class _ResponsesContext:
    request: Request


@dataclass(frozen=True)
class _ResponsesExecutionContext:
    request: Request
    request_params: Mapping[str, JSONValue]
    input_items: tuple[Mapping[str, JSONValue], ...]
    requested_store: bool
    store_explicit: bool
    has_hidden_previous_response: bool


@dataclass(frozen=True)
class _LegacyPlan:
    dialect_id: str = "openai_responses"
    available_capabilities: frozenset[str] = frozenset(
        key
        for key, decision in CAPABILITY_DECISIONS.items()
        if isinstance(decision, Supported)
    )
    capability_minimums: Mapping[str, Mapping[str, JSONValue]] = field(
        default_factory=dict
    )
    required_output_channels: frozenset[str] = frozenset()


def _request_stores_response(req: Request) -> bool:
    options = req.dialect_options
    return options is None or options.values.get("store") is not False


def _execution_context(
    prepared: PreparedRequest,
    context: _ResponsesContext,
) -> tuple[dict[str, JSONValue], bool, _ResponsesExecutionContext]:
    body = prepared.thaw_body()
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise DialectError("prepared Responses request has an invalid model")
    stream = body.get("stream")
    if not isinstance(stream, bool):
        raise DialectError("prepared Responses request has an invalid stream flag")

    raw_input = body.get("input")
    if not isinstance(raw_input, list):
        raise DialectError("prepared Responses request has an invalid input body")
    copied_input = copy_json_value(raw_input, "Responses request input")
    if not isinstance(copied_input, list):
        raise AssertionError("copied Responses input must remain a list")
    input_items: list[Mapping[str, JSONValue]] = []
    for index, item in enumerate(copied_input):
        if not isinstance(item, dict):
            raise DialectError(
                f"prepared Responses input item {index} must be an object"
            )
        input_items.append(item)

    extra_body = body.get("extra_body")
    if extra_body is not None and not isinstance(extra_body, dict):
        raise DialectError("prepared Responses extra_body must be an object")
    extra_mapping = extra_body if isinstance(extra_body, dict) else None
    store_explicit = extra_mapping is not None and "store" in extra_mapping
    raw_store = extra_mapping.get("store", True) if extra_mapping is not None else True
    if not isinstance(raw_store, bool):
        raise DialectError("prepared Responses store must be a boolean")

    request_params = thaw_json_mapping(
        {key: value for key, value in body.items() if key not in {"model", "input"}},
        "Responses request parameters",
    )
    return (
        body,
        stream,
        _ResponsesExecutionContext(
            request=context.request,
            request_params=request_params,
            input_items=tuple(input_items),
            requested_store=raw_store,
            store_explicit=store_explicit,
            has_hidden_previous_response="previous_response_id" in body,
        ),
    )


def _get(raw: object, name: str, default: object = None) -> object:
    if isinstance(raw, Mapping):
        return cast(Mapping[object, object], raw).get(name, default)
    return getattr(raw, name, default)


def _sequence(raw: object, path: str) -> Sequence[object]:
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise OutputContractError(f"{path} must be a sequence")
    return cast(Sequence[object], raw)


def _required_string(raw: object, path: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise OutputContractError(f"{path} must be a non-empty string")
    return raw


def _string(raw: object, path: str) -> str:
    if not isinstance(raw, str):
        raise OutputContractError(f"{path} must be a string")
    return raw


def _optional_string(raw: object, path: str) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise OutputContractError(f"{path} must be a string or None")
    return raw


def _finite_logprob(raw: object, path: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise OutputContractError(f"{path} must be numeric")
    value = float(raw)
    if not math.isfinite(value):
        raise OutputContractError(f"{path} must be finite")
    return value


def _json_text(value: JSONValue) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _arguments_text(value: JSONValue) -> str:
    return value if isinstance(value, str) else _json_text(value)


def _image_rejection(part: ImagePart) -> tuple[str, str] | None:
    if part.url is not None:
        if not part.url:
            return ("input.modality.image", "Responses image URLs must not be empty")
        if part.media_type is not None:
            return (
                "input.modality.image.media_type",
                "Responses has no media-type field for URL-backed images",
            )
    else:
        if not part.data:
            return (
                "input.modality.image",
                "Responses inline images require non-empty base64 data",
            )
        if part.media_type is None:
            return (
                "input.modality.image",
                "Responses inline images require an explicit image media type",
            )
        if part.media_type not in _SUPPORTED_IMAGE_MEDIA_TYPES:
            return (
                "input.modality.image.media_type",
                f"unsupported Responses image media type {part.media_type!r}",
            )
    if part.detail is not None and part.detail not in _SUPPORTED_IMAGE_DETAILS:
        return (
            "input.modality.image.detail",
            f"unsupported Responses image detail {part.detail!r}",
        )
    return None


def _image_to_wire(part: ImagePart) -> dict[str, JSONValue]:
    rejection = _image_rejection(part)
    if rejection is not None:
        raise DialectError(rejection[1])
    if part.url is not None:
        image_url = part.url
    else:
        assert part.media_type is not None
        image_url = f"data:{part.media_type};base64,{part.data}"
    return {
        "type": "input_image",
        "image_url": image_url,
        "detail": part.detail if part.detail is not None else "auto",
    }


def _message_to_input_items(message: ChatMessage) -> list[dict[str, JSONValue]]:
    if message.name is not None:
        raise DialectError("Responses messages cannot carry ChatMessage.name")

    results = [part for part in message.content if isinstance(part, ToolResultPart)]

    if results:
        if message.role != "tool" or len(results) != 1 or len(message.content) != 1:
            raise DialectError(
                "a Responses tool-result message must contain exactly one result"
            )
        result = results[0]
        if result.is_error:
            raise DialectError(
                "Responses function_call_output cannot transmit is_error"
            )
        return [
            {
                "type": "function_call_output",
                "call_id": result.call_id,
                "output": _json_text(result.result),
            }
        ]

    if message.role == "tool":
        raise DialectError("a Responses tool message requires one result part")
    if any(isinstance(part, ToolCallPart) for part in message.content) and (
        message.role != "assistant"
    ):
        raise DialectError("historical function calls require an assistant message")

    items: list[dict[str, JSONValue]] = []
    inline_run: list[TextPart | ImagePart] = []

    def flush_inline(*, force: bool = False) -> None:
        if not inline_run and not force:
            return
        if all(isinstance(part, TextPart) for part in inline_run):
            content: JSONValue = "".join(
                cast(TextPart, part).text for part in inline_run
            )
        else:
            parts: list[JSONValue] = []
            for part in inline_run:
                if isinstance(part, TextPart):
                    parts.append({"type": "input_text", "text": part.text})
                else:
                    parts.append(_image_to_wire(part))
            content = parts
        items.append(
            {
                "type": "message",
                "role": message.role,
                "content": content,
            }
        )
        inline_run.clear()

    for part in message.content:
        if isinstance(part, TextPart | ImagePart):
            inline_run.append(part)
            continue
        if isinstance(part, ToolCallPart):
            flush_inline()
            items.append(
                {
                    "type": "function_call",
                    "call_id": part.call_id,
                    "name": part.name,
                    "arguments": _arguments_text(part.arguments),
                }
            )
            continue
        raise DialectError(f"unsupported Responses message part {type(part).__name__}")

    flush_inline(force=not message.content)
    return items


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

    result: dict[str, JSONValue] = {"type": "function", "name": name}
    for key in ("description", "parameters", "strict"):
        if key in source:
            result[key] = source[key]
    return result


def _tool_choice_to_wire(raw: JSONValue) -> JSONValue:
    if isinstance(raw, str):
        if raw not in {"none", "auto", "required"}:
            raise DialectError(f"unsupported Responses tool choice {raw!r}")
        return raw
    if not isinstance(raw, Mapping):
        raise DialectError("Responses tool choice must be a string or mapping")
    data = dict(raw)
    if data.get("type") != "function":
        raise DialectError("only function-specific mapped tool choices are supported")
    nested = data.get("function")
    if nested is not None:
        if set(data) != {"type", "function"} or not isinstance(nested, Mapping):
            raise DialectError("invalid Chat-shaped function tool choice")
        name = nested.get("name")
    else:
        if set(data) != {"type", "name"}:
            raise DialectError("invalid Responses function tool choice")
        name = data.get("name")
    if not isinstance(name, str) or not name:
        raise DialectError("function tool choice requires a non-empty name")
    return {"type": "function", "name": name}


def _hosted_tool_to_wire(spec: HostedToolSpec) -> dict[str, JSONValue]:
    if spec.kind not in _SUPPORTED_HOSTED_TOOL_KINDS:
        raise DialectError(
            f"unsupported Responses hosted tool kind {spec.kind!r}; "
            "only web search is currently losslessly lifted"
        )
    if "type" in spec.config:
        raise DialectError("hosted tool config cannot override its type")
    return {"type": spec.kind, **dict(spec.config)}


def _structured_text_config(req: Request) -> dict[str, JSONValue] | None:
    params = req.structured_output
    if params.format is None:
        return None
    if params.format == "json_object":
        return {"format": {"type": "json_object"}}
    schema: dict[str, JSONValue] = {
        "type": "json_schema",
        "name": params.name if params.name is not None else "response",
        "schema": dict(params.schema or {}),
    }
    if params.strict is not None:
        schema["strict"] = params.strict
    return {"format": schema}


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


def _decode_opaque_continuation(value: str) -> list[dict[str, JSONValue]]:
    try:
        raw = json.loads(
            value,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise DialectError("invalid Responses opaque continuation JSON") from exc
    if not isinstance(raw, dict):
        raise DialectError("invalid Responses opaque continuation envelope")
    unknown = set(raw) - {"version", "items"}
    if unknown:
        raise DialectError(
            "invalid Responses opaque continuation envelope field(s): "
            + ", ".join(sorted(unknown))
        )
    version = raw.get("version")
    if type(version) is not int or version != _OPAQUE_CONTINUATION_VERSION:
        raise DialectError("unsupported Responses opaque continuation version")
    items = raw.get("items")
    if not isinstance(items, list) or not items:
        raise DialectError("Responses opaque continuation has no history items")
    result: list[dict[str, JSONValue]] = []
    saw_reasoning = False
    for index, item in enumerate(items):
        try:
            payload = _history_item_payload(
                item,
                index,
                require_encrypted_reasoning=True,
            )
        except OutputContractError as exc:
            raise DialectError(
                f"invalid Responses opaque continuation item {index}: {exc}"
            ) from exc
        saw_reasoning |= payload["type"] == "reasoning"
        result.append(payload)
    if not saw_reasoning:
        raise DialectError("Responses opaque continuation has no reasoning item")
    return result


def _canonical_wire_value(value: object, path: str) -> str:
    return json.dumps(
        copy_json_value(value, path),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _wire_values_match(expected: object, actual: object, path: str) -> bool:
    try:
        return _canonical_wire_value(expected, path) == _canonical_wire_value(
            actual, path
        )
    except (TypeError, ValueError):
        return False


def _verification_json_text(value: JSONValue) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        copy_json_value(value, "Responses verification value"),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _verification_image(part: ImagePart) -> dict[str, JSONValue]:
    image_url = (
        part.url
        if part.url is not None
        else f"data:{part.media_type};base64,{part.data}"
    )
    return {
        "type": "input_image",
        "image_url": image_url,
        "detail": part.detail if part.detail is not None else "auto",
    }


def _verification_message_items(message: ChatMessage) -> list[dict[str, JSONValue]]:
    """Independently derive the expected wire items for audit verification.

    Do not reuse ``_message_to_input_items`` here: the verifier must detect
    lowering drift rather than reproduce it through the same implementation.
    """

    results = [part for part in message.content if isinstance(part, ToolResultPart)]
    if results:
        result = results[0]
        return [
            {
                "type": "function_call_output",
                "call_id": result.call_id,
                "output": _verification_json_text(result.result),
            }
        ]

    items: list[dict[str, JSONValue]] = []
    inline: list[TextPart | ImagePart] = []

    def flush_inline(*, force: bool = False) -> None:
        if not inline and not force:
            return
        if all(isinstance(part, TextPart) for part in inline):
            content: JSONValue = "".join(cast(TextPart, part).text for part in inline)
        else:
            content = [
                {"type": "input_text", "text": part.text}
                if isinstance(part, TextPart)
                else _verification_image(part)
                for part in inline
            ]
        items.append(
            {
                "type": "message",
                "role": message.role,
                "content": content,
            }
        )
        inline.clear()

    for part in message.content:
        if isinstance(part, TextPart | ImagePart):
            inline.append(part)
        elif isinstance(part, ToolCallPart):
            flush_inline()
            items.append(
                {
                    "type": "function_call",
                    "call_id": part.call_id,
                    "name": part.name,
                    "arguments": _verification_json_text(part.arguments),
                }
            )
    flush_inline(force=not message.content)
    return items


def _verification_continuation_items(value: str) -> list[JSONValue]:
    raw = json.loads(
        value,
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_nonstandard_json_constant,
    )
    if not isinstance(raw, dict) or raw.get("version") != _OPAQUE_CONTINUATION_VERSION:
        raise ValueError("invalid opaque continuation envelope")
    items = raw.get("items")
    if not isinstance(items, list):
        raise ValueError("invalid opaque continuation items")
    expected: list[JSONValue] = []
    for item in items:
        copied = copy_json_value(item, "opaque continuation item")
        if not isinstance(copied, dict):
            raise TypeError("opaque continuation item must be an object")
        if copied.get("type") == "reasoning":
            copied = {key: item for key, item in copied.items() if item is not None}
        expected.append(copied)
    return expected


@dataclass(frozen=True)
class _ResponsesInputVerifier:
    source: ChatInput
    opaque_continuation: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", deepcopy(self.source))

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        actual = body.get("input")
        if not isinstance(actual, list):
            return "body.input must be a list"
        try:
            expected = (
                _verification_continuation_items(self.opaque_continuation)
                if self.opaque_continuation is not None
                else []
            )
            for message in self.source.messages:
                expected.extend(_verification_message_items(message))
        except (TypeError, ValueError, json.JSONDecodeError):
            return "source input cannot be verified"
        if not _wire_values_match(expected, actual, "Responses input"):
            return "body.input changed or reordered"
        return None


def _verification_function_tool(
    raw: Mapping[str, JSONValue],
) -> dict[str, JSONValue]:
    data = dict(raw)
    nested = data.get("function")
    source = (
        dict(nested)
        if isinstance(nested, Mapping)
        else {key: value for key, value in data.items() if key != "type"}
    )
    result: dict[str, JSONValue] = {
        "type": "function",
        "name": source.get("name"),
    }
    for key in ("description", "parameters", "strict"):
        if key in source:
            result[key] = source[key]
    return result


@dataclass(frozen=True)
class _ResponsesToolsVerifier:
    functions: tuple[Mapping[str, JSONValue], ...]
    hosted: tuple[HostedToolSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "functions", deepcopy(self.functions))
        object.__setattr__(self, "hosted", deepcopy(self.hosted))

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        expected: list[JSONValue] = [
            _verification_function_tool(tool) for tool in self.functions
        ]
        expected.extend(
            {"type": tool.kind, **dict(tool.config)} for tool in self.hosted
        )
        actual = body.get("tools")
        if not _wire_values_match(expected, actual, "Responses tools"):
            return "body.tools changed or reordered"
        if self.hosted:
            includes = body.get("include")
            if not isinstance(includes, list) or (
                "web_search_call.action.sources" not in includes
            ):
                return "body.include omitted hosted-tool sources"
        return None


@dataclass(frozen=True)
class _ResponsesToolChoiceVerifier:
    source: JSONValue

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source",
            copy_json_value(self.source, "Responses tool choice"),
        )

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        source = self.source
        if isinstance(source, str):
            expected: JSONValue = source
        else:
            if not isinstance(source, dict):
                return "source tool choice is invalid"
            nested = source.get("function")
            name = (
                nested.get("name") if isinstance(nested, dict) else source.get("name")
            )
            expected = {"type": "function", "name": name}
        if not _wire_values_match(
            expected, body.get("tool_choice"), "Responses tool choice"
        ):
            return "body.tool_choice changed"
        return None


@dataclass(frozen=True)
class _ResponsesLogprobsVerifier:
    top_logprobs: int

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        actual = body.get("top_logprobs")
        if (
            not isinstance(actual, int)
            or isinstance(actual, bool)
            or actual != self.top_logprobs
        ):
            return "body.top_logprobs changed"
        includes = body.get("include")
        if not isinstance(includes, list) or (
            "message.output_text.logprobs" not in includes
        ):
            return "body.include omitted message logprobs"
        return None


def _plain_json(value: object, path: str) -> JSONValue:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        value = model_dump(mode="json", exclude_none=True)
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise OutputContractError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise OutputContractError(f"{path} contains a non-string key")
            result[key] = _plain_json(item, f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [
            _plain_json(item, f"{path}[{index}]") for index, item in enumerate(value)
        ]
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return _plain_json(
            {key: item for key, item in attributes.items() if not key.startswith("_")},
            path,
        )
    raise OutputContractError(
        f"{path} contains unsupported value {type(value).__name__}"
    )


def _validate_reasoning_text_parts(
    payload: Mapping[str, JSONValue],
    *,
    field_name: str,
    expected_type: str,
    item_index: int,
    required: bool,
) -> None:
    path = f"response.output[{item_index}].{field_name}"
    raw = payload.get(field_name)
    if raw is None:
        if required:
            raise OutputContractError(f"{path} must be a list")
        return
    if not isinstance(raw, list):
        raise OutputContractError(f"{path} must be a list")
    for part_index, part in enumerate(raw):
        part_path = f"{path}[{part_index}]"
        if not isinstance(part, dict):
            raise OutputContractError(f"{part_path} must be an object")
        unknown = set(part) - _REASONING_TEXT_PART_KEYS
        if unknown:
            raise OutputContractError(
                f"{part_path} has unsupported field(s): " + ", ".join(sorted(unknown))
            )
        if set(part) != _REASONING_TEXT_PART_KEYS:
            raise OutputContractError(f"{part_path} requires type and text")
        if part["type"] != expected_type:
            raise OutputContractError(f"{part_path}.type must be {expected_type!r}")
        if not isinstance(part["text"], str):
            raise OutputContractError(f"{part_path}.text must be a string")


def _reasoning_payload(raw: object, index: int) -> dict[str, JSONValue]:
    raw_payload = _plain_json(raw, f"response.output[{index}]")
    if not isinstance(raw_payload, dict) or raw_payload.get("type") != "reasoning":
        raise OutputContractError("reasoning output item has invalid shape")
    unknown = set(raw_payload) - _REASONING_ITEM_KEYS
    if unknown:
        raise OutputContractError(
            "reasoning output item has unsupported field(s): "
            + ", ".join(sorted(unknown))
        )
    payload = {key: value for key, value in raw_payload.items() if value is not None}
    item_id = payload.get("id")
    if not isinstance(item_id, str) or not item_id:
        raise OutputContractError("reasoning output item is missing id")
    _validate_reasoning_text_parts(
        payload,
        field_name="summary",
        expected_type="summary_text",
        item_index=index,
        required=True,
    )
    _validate_reasoning_text_parts(
        payload,
        field_name="content",
        expected_type="reasoning_text",
        item_index=index,
        required=False,
    )
    status = payload.get("status")
    if status is not None and (
        not isinstance(status, str) or status not in _REASONING_ITEM_STATUSES
    ):
        raise OutputContractError("reasoning output item has invalid status")
    encrypted = payload.get("encrypted_content")
    if encrypted is not None and not isinstance(encrypted, str):
        raise OutputContractError(
            "reasoning output item encrypted_content must be a string"
        )
    return cast(dict[str, JSONValue], payload)


def _opaque_history_bundle(
    input_items: Sequence[Mapping[str, JSONValue]],
    output_items: Sequence[object],
    *,
    has_hidden_previous_response: bool,
) -> str | None:
    """Serialize the complete stateless Responses history for the next call.

    OpenAI's stateless contract requires replaying the exact prior input and
    every output item in order.  A request chained through
    ``previous_response_id`` has hidden server-side history, so no faithful
    standalone continuation can be constructed for that response.  Output
    message logprobs remain part of that history because Responses accepts its
    output-message shape as input and manual replay must preserve items intact.
    """

    if has_hidden_previous_response:
        return None
    history = [dict(item) for item in input_items]
    saw_reasoning = any(item.get("type") == "reasoning" for item in history)
    if not saw_reasoning and not any(
        _get(item, "type") == "reasoning" for item in output_items
    ):
        return None
    for index, item in enumerate(output_items):
        payload = _history_item_payload(
            item,
            index,
            require_encrypted_reasoning=False,
        )
        if payload["type"] == "reasoning":
            encrypted = payload.get("encrypted_content")
            if not isinstance(encrypted, str) or not encrypted:
                return None
        history.append(payload)
    # Guaranteed by the returns above: the bundle holds a reasoning item with
    # encrypted_content, which _decode_opaque_continuation requires next hop.
    return json.dumps(
        {"version": _OPAQUE_CONTINUATION_VERSION, "items": history},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _visible_reasoning_text(raw: object, path: str) -> tuple[str | None, bool]:
    summary_texts: list[str] = []
    summary = _sequence(_get(raw, "summary", []), f"{path}.summary")
    for index, part in enumerate(summary):
        if _get(part, "type") != "summary_text":
            raise OutputContractError(f"{path}.summary[{index}] has unknown type")
        summary_texts.append(
            _string(_get(part, "text"), f"{path}.summary[{index}].text")
        )
    if summary_texts:
        return "\n".join(summary_texts), True

    content = _get(raw, "content")
    if content is None:
        return None, False
    content_texts: list[str] = []
    for index, part in enumerate(_sequence(content, f"{path}.content")):
        if _get(part, "type") != "reasoning_text":
            raise OutputContractError(f"{path}.content[{index}] has unknown type")
        content_texts.append(
            _string(_get(part, "text"), f"{path}.content[{index}].text")
        )
    return "\n".join(content_texts) or None, False


def _usage_detail(raw: object, container: str, name: str) -> int | None:
    details = _get(raw, container)
    value = _get(details, name) if details is not None else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _usage_stats(raw: object) -> UsageStats | None:
    if raw is None:
        return None
    values: list[int] = []
    for name in ("input_tokens", "output_tokens"):
        value = _get(raw, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise OutputContractError(
                f"Responses usage.{name} must be a non-negative integer"
            )
        values.append(value)
    computed = values[0] + values[1]
    reported = _get(raw, "total_tokens")
    reported_total = (
        reported
        if isinstance(reported, int)
        and not isinstance(reported, bool)
        and reported >= 0
        and reported != computed
        else None
    )
    # ``cache_write_tokens`` is newer than the repository's minimum OpenAI SDK
    # and has no field in the shared UsageStats record. Usage is explicitly a
    # best-effort channel, so the adapter preserves every representable count
    # without inventing a destination for that future breakdown.
    return UsageStats(
        input_tokens=values[0],
        output_tokens=values[1],
        total_tokens=computed,
        reasoning_tokens=_usage_detail(
            raw, "output_tokens_details", "reasoning_tokens"
        ),
        cached_tokens=_usage_detail(raw, "input_tokens_details", "cached_tokens"),
        reported_total_tokens=reported_total,
    )


def _finish_reason(raw: object) -> str:
    status = _get(raw, "status")
    incomplete_details = _get(raw, "incomplete_details")
    error = _get(raw, "error")
    if status == "completed":
        if error is not None:
            raise OutputContractError("completed Responses reply contains an error")
        if incomplete_details is not None:
            raise OutputContractError(
                "completed Responses reply contains incomplete_details"
            )
        return "completed"
    if status == "incomplete":
        if error is not None:
            raise OutputContractError("incomplete Responses reply contains an error")
        reason = (
            _get(incomplete_details, "reason")
            if incomplete_details is not None
            else None
        )
        return _required_string(reason, "response.incomplete_details.reason")
    if status == "failed":
        if incomplete_details is not None:
            raise OutputContractError(
                "failed Responses reply contains incomplete_details"
            )
        if error is None:
            raise OutputContractError("failed Responses reply omitted its error")
        code = _get(error, "code") if error is not None else None
        message = _get(error, "message") if error is not None else None
        raise OutputContractError(
            "Responses request failed"
            + (f" ({code})" if isinstance(code, str) and code else "")
            + (f": {message}" if isinstance(message, str) and message else "")
        )
    if status == "cancelled":
        raise OutputContractError(
            "Responses reply has unsupported terminal status 'cancelled'"
        )
    raise OutputContractError(f"Responses reply has non-terminal status {status!r}")


def _parse_logprobs(
    raw: object,
    path: str,
) -> tuple[list[TokenLogprob], list[tuple[TopKEntry, ...]]]:
    sampled: list[TokenLogprob] = []
    alternatives: list[tuple[TopKEntry, ...]] = []
    for position, item in enumerate(_sequence(raw, path)):
        token = _string(_get(item, "token"), f"{path}[{position}].token")
        logprob = _finite_logprob(_get(item, "logprob"), f"{path}[{position}].logprob")
        sampled.append(TokenLogprob(token=token, logprob=logprob))
        top: list[TopKEntry] = []
        raw_top = _get(item, "top_logprobs", [])
        for rank, entry in enumerate(
            _sequence(raw_top, f"{path}[{position}].top_logprobs")
        ):
            top.append(
                TopKEntry(
                    token=_string(
                        _get(entry, "token"),
                        f"{path}[{position}].top_logprobs[{rank}].token",
                    ),
                    logprob=_finite_logprob(
                        _get(entry, "logprob"),
                        f"{path}[{position}].top_logprobs[{rank}].logprob",
                    ),
                )
            )
        alternatives.append(tuple(top))
    return sampled, alternatives


def _url_citations(raw: object, path: str) -> list[Citation]:
    citations: list[Citation] = []
    for index, annotation in enumerate(_sequence(raw, path)):
        annotation_type = _get(annotation, "type")
        if annotation_type != "url_citation":
            raise OutputContractError(
                f"{path}[{index}] type {annotation_type!r} has no lossless IR mapping"
            )
        citations.append(
            Citation(
                url=_required_string(_get(annotation, "url"), f"{path}[{index}].url"),
                title=_optional_string(
                    _get(annotation, "title"), f"{path}[{index}].title"
                ),
            )
        )
    return citations


def _function_call(raw: object, path: str) -> FunctionToolCall:
    arguments = _get(raw, "arguments")
    if not isinstance(arguments, str):
        raise OutputContractError(f"{path}.arguments must be a string")
    return FunctionToolCall(
        call_id=_required_string(_get(raw, "call_id"), f"{path}.call_id"),
        name=_required_string(_get(raw, "name"), f"{path}.name"),
        arguments=arguments,
    )


def _validate_terminal_item_status(
    raw: object,
    path: str,
) -> None:
    status = _get(raw, "status", _MISSING)
    if status is _MISSING or status is None:
        return
    if status == "in_progress":
        raise OutputContractError(
            f"{path} remains in_progress in a terminal Responses reply"
        )
    if not isinstance(status, str) or status not in _TERMINAL_ITEM_STATUSES:
        raise OutputContractError(f"{path}.status {status!r} is not terminal")


def _web_search_use(raw: object, path: str) -> ServerToolUse:
    status = _required_string(_get(raw, "status"), f"{path}.status")
    if status in {"in_progress", "searching"}:
        raise OutputContractError(
            f"{path} remains {status} in a terminal Responses reply"
        )
    if status not in {"completed", "failed", "incomplete"}:
        raise OutputContractError(
            f"{path}.status {status!r} has no terminal shared-IR mapping"
        )
    action_value = _plain_json(_get(raw, "action"), f"{path}.action")
    if not isinstance(action_value, dict):
        raise OutputContractError(f"{path}.action must be an object")
    action = dict(action_value)
    result = action.pop("sources", None)
    return ServerToolUse(
        tool_type="web_search",
        tool_use_id=_required_string(_get(raw, "id"), f"{path}.id"),
        input=action,
        result=result,
        error_code=None if status == "completed" else status,
    )


def _reject_unknown_history_fields(
    payload: Mapping[str, JSONValue],
    allowed: frozenset[str],
    path: str,
) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise OutputContractError(
            f"{path} has unsupported field(s): " + ", ".join(sorted(unknown))
        )


def _validate_history_status(value: object, path: str) -> None:
    if not isinstance(value, str) or value not in _MESSAGE_ITEM_STATUSES:
        raise OutputContractError(f"{path} has invalid status {value!r}")


def _validate_input_content_part(
    part: object,
    path: str,
) -> None:
    if not isinstance(part, Mapping):
        raise OutputContractError(f"{path} must be an object")
    payload = cast(Mapping[str, JSONValue], part)
    part_type = payload.get("type")
    if part_type == "input_text":
        _reject_unknown_history_fields(payload, _INPUT_TEXT_PART_KEYS, path)
        _string(payload.get("text"), f"{path}.text")
        return
    if part_type == "input_image":
        _reject_unknown_history_fields(payload, _INPUT_IMAGE_PART_KEYS, path)
        _required_string(payload.get("image_url"), f"{path}.image_url")
        detail = payload.get("detail")
        if not isinstance(detail, str) or detail not in _SUPPORTED_IMAGE_DETAILS:
            raise OutputContractError(f"{path}.detail has invalid value {detail!r}")
        return
    raise OutputContractError(f"{path} has unsupported type {part_type!r}")


def _validate_output_content_part(
    part: object,
    path: str,
) -> None:
    if not isinstance(part, Mapping):
        raise OutputContractError(f"{path} must be an object")
    payload = cast(Mapping[str, JSONValue], part)
    if payload.get("type") != "output_text":
        raise OutputContractError(
            f"{path} has unsupported type {payload.get('type')!r}"
        )
    _reject_unknown_history_fields(payload, _OUTPUT_TEXT_PART_KEYS, path)
    _string(payload.get("text"), f"{path}.text")
    annotations = payload.get("annotations")
    if annotations is not None:
        _url_citations(annotations, f"{path}.annotations")
    logprobs = payload.get("logprobs")
    if logprobs is not None:
        _parse_logprobs(logprobs, f"{path}.logprobs")


def _validate_history_message(
    payload: Mapping[str, JSONValue],
    path: str,
) -> None:
    _reject_unknown_history_fields(payload, _MESSAGE_ITEM_KEYS, path)
    role = payload.get("role")
    if not isinstance(role, str) or role not in _MESSAGE_ROLES:
        raise OutputContractError(f"{path}.role has invalid value {role!r}")
    content = payload.get("content")
    is_output = "id" in payload or "status" in payload
    if is_output:
        _required_string(payload.get("id"), f"{path}.id")
        _validate_history_status(payload.get("status"), f"{path}.status")
        if role != "assistant":
            raise OutputContractError(f"{path}.role must be 'assistant'")
        for index, part in enumerate(_sequence(content, f"{path}.content")):
            _validate_output_content_part(part, f"{path}.content[{index}]")
        return
    if isinstance(content, str):
        return
    for index, part in enumerate(_sequence(content, f"{path}.content")):
        _validate_input_content_part(part, f"{path}.content[{index}]")


def _validate_history_function_call(
    payload: Mapping[str, JSONValue],
    path: str,
) -> None:
    _reject_unknown_history_fields(payload, _FUNCTION_CALL_ITEM_KEYS, path)
    _required_string(payload.get("call_id"), f"{path}.call_id")
    _required_string(payload.get("name"), f"{path}.name")
    _string(payload.get("arguments"), f"{path}.arguments")
    if "id" in payload:
        _required_string(payload.get("id"), f"{path}.id")
    if "status" in payload:
        _validate_history_status(payload.get("status"), f"{path}.status")


def _validate_history_function_call_output(
    payload: Mapping[str, JSONValue],
    path: str,
) -> None:
    _reject_unknown_history_fields(payload, _FUNCTION_CALL_OUTPUT_ITEM_KEYS, path)
    _required_string(payload.get("call_id"), f"{path}.call_id")
    _string(payload.get("output"), f"{path}.output")
    if "id" in payload:
        _required_string(payload.get("id"), f"{path}.id")
    if "status" in payload:
        _validate_history_status(payload.get("status"), f"{path}.status")


def _history_item_payload(
    raw: object,
    index: int,
    *,
    require_encrypted_reasoning: bool,
) -> dict[str, JSONValue]:
    path = f"history.items[{index}]"
    raw_payload = _plain_json(raw, path)
    if not isinstance(raw_payload, dict):
        raise OutputContractError(f"{path} must be an object")
    payload = raw_payload
    item_type = payload.get("type")
    if not isinstance(item_type, str) or item_type not in _HISTORY_ITEM_TYPES:
        raise OutputContractError(f"{path} has unsupported type {item_type!r}")
    if item_type == "reasoning":
        reasoning = _reasoning_payload(payload, index)
        if require_encrypted_reasoning:
            encrypted = reasoning.get("encrypted_content")
            if not isinstance(encrypted, str) or not encrypted:
                raise OutputContractError(
                    f"{path} reasoning item has no encrypted_content"
                )
        return reasoning
    if item_type == "message":
        _validate_history_message(payload, path)
    elif item_type == "function_call":
        _validate_history_function_call(payload, path)
    elif item_type == "function_call_output":
        _validate_history_function_call_output(payload, path)
    else:
        _reject_unknown_history_fields(payload, _WEB_SEARCH_ITEM_KEYS, path)
        _web_search_use(payload, path)
    return dict(payload)


def _structured_output(req: Request, text: str) -> StructuredOutput | None:
    if req.structured_output.format is None:
        return None
    try:
        return StructuredOutput(
            json.loads(
                text,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_nonstandard_json_constant,
            )
        )
    except (TypeError, ValueError) as exc:
        raise OutputContractError("structured output is not valid JSON") from exc


class OpenAIResponsesDialect:
    """Executable adapter for ``client.responses.create``."""

    dialect_id = "openai_responses"
    connection_family = "openai_sdk"
    capability_decisions = CAPABILITY_DECISIONS
    output_contract = OUTPUT_CONTRACT
    CAPABILITIES = frozenset(
        {
            Capability.Chat,
            Capability.FunctionCalling,
            Capability.ServerTools,
            Capability.Reasoning,
            Capability.ReasoningEffort,
            Capability.SampledLogprobs,
            Capability.StructuredOutput,
            Capability.TopKLogprobs,
        }
    )

    def __init__(self, client: Any, requested_model_id: str):
        if not requested_model_id:
            raise ValueError("requested_model_id must not be empty")
        self._client = client
        self._requested_model_id = requested_model_id

    @property
    def capabilities(self) -> frozenset[Capability]:
        """Deprecated enum view retained during the compatibility cycle."""

        return self.CAPABILITIES

    def validate_request(
        self, req: Request, audit: RequestAudit, plan: RuntimePlanView
    ) -> None:
        """Account for every active leaf and reject unsupported semantics."""

        audit.require_matches_request(req)
        del plan
        has_named_message = isinstance(req.input, ChatInput) and any(
            message.name is not None for message in req.input.messages
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
        has_both_continuations = (
            req.session.previous_response_id is not None
            and req.session.opaque_continuation is not None
        )
        response_is_stored = _request_stores_response(req)

        for path in audit.active_paths:
            if path == "input.chat" and has_named_message:
                audit.rejected(path, "Responses messages have no name field")
            elif path in {"input.completion", "input.completion.suffix"}:
                audit.rejected(path, "openai_responses requires ChatInput")
            elif path in _UNSUPPORTED_REQUEST_PATHS:
                audit.rejected(path, "Responses has no equivalent request field")
            elif path == "scoring.input_scoring":
                audit.rejected(path, "Responses cannot score input tokens")
            elif path == "scoring.top_logprobs" and req.scoring.top_logprobs > 20:
                audit.rejected(path, "Responses supports at most 20 top logprobs")
            elif path == "reasoning.budget_tokens":
                audit.rejected(path, "Responses has no reasoning token-budget field")
            elif (
                path == "reasoning.effort"
                and req.reasoning.effort not in _OPENAI_REASONING_EFFORTS
            ):
                audit.rejected(
                    path,
                    f"unsupported OpenAI reasoning effort {req.reasoning.effort!r}",
                )
            elif path == "tools.hosted" and any(
                tool.kind not in _SUPPORTED_HOSTED_TOOL_KINDS
                for tool in req.tools.hosted
            ):
                audit.rejected(
                    path,
                    "only Responses web-search hosted tools currently have a "
                    "lossless IR mapping",
                )
            elif image_rejection is not None and path == image_rejection[0]:
                audit.rejected(path, image_rejection[1])
            elif path == "input.modality.tool_result.is_error":
                audit.rejected(
                    path,
                    "Responses function_call_output cannot transmit is_error",
                )
            elif path.startswith("session.") and has_both_continuations:
                audit.rejected(
                    path,
                    "previous_response_id and opaque continuation are mutually "
                    "exclusive",
                )
            elif path == "session.previous_response_id" and not response_is_stored:
                # OpenAI permits this as a one-way continuation, but SiEval's
                # stateful_session contract requires a reusable output ID.
                audit.rejected(
                    path,
                    "store=false is stateless and cannot return a reusable "
                    "session_id; use stored response chaining or replay the "
                    "prior output items",
                )
            elif path == "structured_output.format" and (
                req.structured_output.format not in {"json_object", "json_schema"}
            ):
                audit.rejected(path, "unsupported Responses text format")
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
            elif path == "structured_output.name" and req.structured_output.name == "":
                audit.rejected(path, "Responses JSON schema names must not be empty")
            elif path.startswith("dialect_options."):
                key = path.removeprefix("dialect_options.")
                if not key:
                    # ``finish`` refuses this too, but without the request path;
                    # reject at admission like the other OpenAI dialects.
                    audit.rejected(path, "dialect option keys must be non-empty")
                elif key == "conversation":
                    audit.rejected(
                        path,
                        "Responses conversation state is not yet modeled by the "
                        "provider-neutral stateful_session contract",
                    )
                elif key == "instructions":
                    audit.rejected(
                        path,
                        "Responses instructions must be expressed through "
                        "provider-neutral system or developer messages",
                    )
                elif key == "store" and (
                    req.dialect_options is None
                    or not isinstance(req.dialect_options.values[key], bool)
                ):
                    audit.rejected(path, "Responses store must be a boolean")
                elif (
                    key == "background"
                    and req.dialect_options is not None
                    and req.dialect_options.values[key] is True
                ):
                    audit.rejected(
                        path,
                        "background Responses cannot complete in this synchronous "
                        "execution contract",
                    )
                elif key in _IR_OWNED_BODY_KEYS:
                    audit.rejected(
                        path,
                        f"{key!r} must use its canonical provider-neutral field",
                    )

    def prepare(self, req: Request, audit: RequestAudit) -> PreparedRequest:
        """Lower one audited request into Responses API keyword arguments."""

        audit.require_matches_request(req)
        audit.raise_rejections()
        if not isinstance(req.input, ChatInput):
            raise DialectError("openai_responses requires ChatInput")

        active_paths = frozenset(audit.active_paths)
        continuation = req.session.opaque_continuation
        input_verifier = _ResponsesInputVerifier(
            req.input,
            continuation.value if continuation is not None else None,
        )

        def consume(path: str, *wire: WireEvidence) -> None:
            if path in active_paths and path not in audit.decisions:
                audit.consumed(path, *wire)

        input_items: list[dict[str, JSONValue]] = []
        if continuation is not None:
            input_items.extend(_decode_opaque_continuation(continuation.value))
        for message in req.input.messages:
            input_items.extend(_message_to_input_items(message))
        canonical_input_items = [
            _history_item_payload(
                item,
                index,
                require_encrypted_reasoning=True,
            )
            for index, item in enumerate(input_items)
        ]

        for path in (
            "input.chat",
            "input.modality.text",
            "input.modality.image",
            "input.modality.image.detail",
            "input.modality.image.media_type",
            "input.modality.tool_call",
            "input.modality.tool_result",
        ):
            consume(path, input_verifier)

        body: dict[str, Any] = {
            "model": self._requested_model_id,
            "input": canonical_input_items,
            "stream": req.scheduling.stream,
        }
        sampling = req.sampling
        if sampling.max_tokens is not None:
            body["max_output_tokens"] = sampling.max_tokens
            consume(
                "sampling.max_tokens",
                WireObservation(("max_output_tokens",)),
            )
        for name in ("temperature", "top_p"):
            value = getattr(sampling, name)
            if value is not None:
                body[name] = value
                consume(f"sampling.{name}", WireObservation((name,)))

        includes: set[str] = set()
        # Plain/non-reasoning Responses endpoints may reject encrypted reasoning.
        # Request it only when reasoning state is active or already being replayed.
        if (
            continuation is not None
            or req.reasoning.effort not in {None, "none"}
            or req.reasoning.summary not in {None, "none"}
        ):
            includes.add("reasoning.encrypted_content")
        logprobs_verifier = _ResponsesLogprobsVerifier(req.scoring.top_logprobs)
        if req.scoring.sampled_logprobs:
            body["top_logprobs"] = req.scoring.top_logprobs
            includes.add("message.output_text.logprobs")
            consume("scoring.sampled_logprobs", logprobs_verifier)
        if req.scoring.top_logprobs > 0:
            body["top_logprobs"] = req.scoring.top_logprobs
            consume(
                "scoring.top_logprobs",
                WireObservation(("top_logprobs",)),
                logprobs_verifier,
            )

        reasoning: dict[str, JSONValue] = {}
        if req.reasoning.effort is not None:
            reasoning["effort"] = req.reasoning.effort
            consume(
                "reasoning.effort",
                WireObservation(("reasoning", "effort")),
            )
        if req.reasoning.summary == "none":
            audit.noop(
                "reasoning.summary",
                "omitting Responses reasoning.summary requests no visible summary",
            )
        elif req.reasoning.summary is not None:
            reasoning["summary"] = req.reasoning.summary
            consume(
                "reasoning.summary",
                WireObservation(("reasoning", "summary")),
            )
        if reasoning:
            body["reasoning"] = reasoning
        tools: list[dict[str, JSONValue]] = []
        tools_verifier = _ResponsesToolsVerifier(
            req.tools.functions,
            req.tools.hosted,
        )
        if req.tools.functions:
            tools.extend(_function_tool_to_wire(tool) for tool in req.tools.functions)
            consume("tools.functions", tools_verifier)
        if req.tools.hosted:
            tools.extend(_hosted_tool_to_wire(tool) for tool in req.tools.hosted)
            includes.add("web_search_call.action.sources")
            consume("tools.hosted", tools_verifier)
        if tools:
            body["tools"] = tools
        if req.tools.choice is not None:
            body["tool_choice"] = _tool_choice_to_wire(req.tools.choice)
            consume(
                "tools.choice",
                _ResponsesToolChoiceVerifier(req.tools.choice),
            )
        if req.tools.parallel is not None:
            body["parallel_tool_calls"] = req.tools.parallel
            consume(
                "tools.parallel",
                WireObservation(("parallel_tool_calls",)),
            )

        text_config = _structured_text_config(req)
        if text_config is not None:
            body["text"] = text_config
        if req.structured_output.format is not None:
            consume(
                "structured_output.format",
                WireObservation(("text", "format", "type")),
            )
        if req.structured_output.schema is not None:
            consume(
                "structured_output.schema",
                WireObservation(("text", "format", "schema")),
            )
        if req.structured_output.name is not None:
            consume(
                "structured_output.name",
                WireObservation(("text", "format", "name")),
            )
        if req.structured_output.strict is not None:
            consume(
                "structured_output.strict",
                WireObservation(("text", "format", "strict")),
            )

        if req.session.previous_response_id is not None:
            body["previous_response_id"] = req.session.previous_response_id
            consume(
                "session.previous_response_id",
                WireObservation(("previous_response_id",)),
            )
        if continuation is not None:
            consume("session.opaque_continuation", input_verifier)
        if req.scheduling.stream:
            consume("scheduling.stream", WireObservation(("stream",)))

        if includes:
            body["include"] = sorted(includes)

        extra_body: dict[str, JSONValue] = {}
        options = req.dialect_options
        if options is not None:
            if not options.values:
                audit.noop(
                    "dialect_options",
                    "an empty matching options mapping has no wire semantics",
                )
            for key, option_value in options.values.items():
                path = f"dialect_options.{key}"
                if path in audit.decisions:
                    continue
                extra_body[key] = option_value
                audit.passthrough(path, "extra_body")
        if extra_body:
            body["extra_body"] = extra_body

        return PreparedRequest(
            operation="responses.create",
            body=cast(Mapping[str, JSONValue], body),
            context=_ResponsesContext(request=deepcopy(req)),
        )

    def _lift(self, raw: object, context: _ResponsesExecutionContext) -> Response:
        finish_reason = _finish_reason(raw)
        response_id = _required_string(_get(raw, "id"), "response.id")
        reported_store = _get(raw, "store", _MISSING)
        # A compliant reply omits ``store``; a partially compatible endpoint may
        # echo an explicit null. Treat that as unreported, not a type violation.
        if reported_store is _MISSING or reported_store is None:
            response_is_stored = context.requested_store
        else:
            if not isinstance(reported_store, bool):
                raise OutputContractError("response.store must be a boolean")
            if context.store_explicit and reported_store is not context.requested_store:
                raise OutputContractError(
                    "response.store contradicts the requested store value"
                )
            response_is_stored = reported_store
        usage = _usage_stats(_get(raw, "usage"))
        text_parts: list[str] = []
        reasoning_payloads: list[dict[str, JSONValue]] = []
        reasoning_texts: list[str] = []
        requested_reasoning_summary = context.request.reasoning.summary not in {
            None,
            "none",
        }
        saw_reasoning_summary = False
        tool_calls: list[FunctionToolCall] = []
        server_uses: list[ServerToolUse] = []
        citations: list[Citation] = []
        sampled_logprobs: list[TokenLogprob] = []
        top_logprobs: list[tuple[TopKEntry, ...]] = []
        saw_logprobs = False
        requested_logprobs = context.request.scoring.sampled_logprobs

        output = _sequence(_get(raw, "output"), "response.output")
        for item_index, item in enumerate(output):
            path = f"response.output[{item_index}]"
            item_type = _get(item, "type")
            if item_type == "message":
                _validate_terminal_item_status(item, path)
                content = _sequence(_get(item, "content"), f"{path}.content")
                for content_index, part in enumerate(content):
                    part_path = f"{path}.content[{content_index}]"
                    part_type = _get(part, "type")
                    if part_type == "refusal":
                        raise OutputContractError(
                            "Responses refusal has no faithful shared-IR channel"
                        )
                    if part_type != "output_text":
                        raise OutputContractError(
                            f"{part_path} has unsupported type {part_type!r}"
                        )
                    part_text = _string(_get(part, "text"), f"{part_path}.text")
                    text_parts.append(part_text)
                    raw_annotations = _get(part, "annotations", [])
                    citations.extend(
                        _url_citations(raw_annotations, f"{part_path}.annotations")
                    )
                    raw_logprobs = _get(part, "logprobs")
                    if requested_logprobs and part_text and raw_logprobs is None:
                        raise OutputContractError(
                            f"{part_path}.logprobs is absent for non-empty text"
                        )
                    if raw_logprobs is not None:
                        saw_logprobs = True
                        sampled, alternatives = _parse_logprobs(
                            raw_logprobs, f"{part_path}.logprobs"
                        )
                        if requested_logprobs and part_text and not sampled:
                            raise OutputContractError(
                                f"{part_path}.logprobs is empty for non-empty text"
                            )
                        sampled_logprobs.extend(sampled)
                        top_logprobs.extend(alternatives)
            elif item_type == "function_call":
                _validate_terminal_item_status(item, path)
                tool_calls.append(_function_call(item, path))
            elif item_type == "reasoning":
                _validate_terminal_item_status(item, path)
                reasoning_payloads.append(_reasoning_payload(item, item_index))
                text, is_summary = _visible_reasoning_text(item, path)
                if is_summary and text:
                    saw_reasoning_summary = True
                    reasoning_texts.append(text)
                elif not requested_reasoning_summary and text:
                    reasoning_texts.append(text)
            elif item_type == "web_search_call":
                server_uses.append(_web_search_use(item, path))
            else:
                raise OutputContractError(
                    f"Responses output item type {item_type!r} has no IR mapping"
                )

        if requested_reasoning_summary and not saw_reasoning_summary:
            raise OutputContractError(
                "Responses reply omitted the requested visible reasoning summary"
            )

        text = "".join(text_parts)
        opaque_history = _opaque_history_bundle(
            context.input_items,
            output,
            has_hidden_previous_response=context.has_hidden_previous_response,
        )
        reasoning: tuple[ReasoningOutput | None, ...] | None = None
        if reasoning_payloads or opaque_history is not None:
            effort = _get(_get(raw, "reasoning"), "effort")
            reasoning = (
                ReasoningOutput(
                    text="\n".join(reasoning_texts) or None,
                    opaque_roundtrip=opaque_history,
                    thinking_tokens=(
                        usage.reasoning_tokens
                        if usage is not None and usage.reasoning_tokens is not None
                        else 0
                    ),
                    effort_used=_optional_string(effort, "response.reasoning.effort"),
                ),
            )

        return Response(
            texts=(text,),
            reasoning=reasoning,
            finish_reasons=(finish_reason,),
            tool_calls=tuple(tool_calls) or None,
            server_tool_uses=tuple(server_uses) or None,
            structured_output=_structured_output(context.request, text),
            logprobs=(
                tuple(sampled_logprobs) if saw_logprobs or requested_logprobs else None
            ),
            top_logprobs=(
                tuple(top_logprobs)
                if context.request.scoring.top_logprobs > 0
                else None
            ),
            citations=tuple(citations) or None,
            session_id=response_id if response_is_stored else None,
            usage=usage,
            request_params=context.request_params,
            response_model=_optional_string(_get(raw, "model"), "response.model"),
            system_fingerprint=_optional_string(
                _get(raw, "system_fingerprint"), "response.system_fingerprint"
            ),
        )

    async def _lift_stream(
        self, stream: object, context: _ResponsesExecutionContext
    ) -> Response:
        if not isinstance(stream, AsyncIterable):
            raise OutputContractError(
                "streaming Responses reply is not asynchronously iterable"
            )
        terminal: object | None = None
        terminal_type: str | None = None
        saw_terminal = False
        expected_status = {
            "response.completed": "completed",
            "response.incomplete": "incomplete",
            "response.failed": "failed",
        }
        async for event in stream:
            event_type = _get(event, "type")
            if event_type == "error":
                message = _get(event, "message")
                raise OutputContractError(
                    "Responses stream error"
                    + (f": {message}" if isinstance(message, str) else "")
                )
            if event_type not in expected_status:
                continue
            if saw_terminal:
                raise OutputContractError(
                    "Responses stream emitted two terminal events"
                )
            saw_terminal = True
            terminal = _get(event, "response")
            if terminal is None:
                raise OutputContractError(
                    f"Responses terminal event {event_type!r} omitted its response"
                )
            terminal_type = cast(str, event_type)
        if not saw_terminal or terminal is None or terminal_type is None:
            raise OutputContractError("Responses stream omitted its terminal response")
        status = _get(terminal, "status")
        if status != expected_status[terminal_type]:
            raise OutputContractError(
                f"Responses terminal event {terminal_type!r} carried status {status!r}"
            )
        return self._lift(terminal, context)

    async def execute(self, prepared: PreparedRequest) -> Response:
        """Execute through the borrowed SDK client and lift the terminal reply."""

        if prepared.operation != "responses.create":
            raise DialectError(f"unexpected Responses operation {prepared.operation!r}")
        context = prepared.context
        if not isinstance(context, _ResponsesContext):
            raise DialectError("prepared Responses request has invalid context")
        body, stream, execution_context = _execution_context(prepared, context)
        raw = await self._client.responses.create(**body)
        if stream:
            return await self._lift_stream(raw, execution_context)
        return self._lift(raw, execution_context)

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
