"""Provider-neutral request and response records for model dialects.

The request side is immutable runtime input.  The response side is a closed,
additive persisted schema: every nested response record is decorated so resume
rehydration preserves its concrete type.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal, cast

from sieval.core.types import JSONValue
from sieval.core.utils.serialization import sieval_record

from ._engine_source import EngineSource

# ---------------------------------------------------------------------------
# Typed input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompletionInput:
    """Plain completion input, optionally with a fill-in-the-middle suffix."""

    text: str
    suffix: str | None = None


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    """Provider-neutral image reference or inline payload."""

    url: str | None = None
    data: str | None = None
    media_type: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if (self.url is None) == (self.data is None):
            raise ValueError("ImagePart requires exactly one of url or data")


@dataclass(frozen=True)
class ToolCallPart:
    call_id: str
    name: str
    arguments: JSONValue


@dataclass(frozen=True)
class ToolResultPart:
    call_id: str
    result: JSONValue
    is_error: bool = False


type ContentPart = TextPart | ImagePart | ToolCallPart | ToolResultPart
type ChatRole = Literal["system", "developer", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ChatMessage:
    role: ChatRole
    content: tuple[ContentPart, ...]
    name: str | None = None


@dataclass(frozen=True)
class ChatInput:
    messages: tuple[ChatMessage, ...]

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("ChatInput.messages must not be empty")


type ModelInput = CompletionInput | ChatInput


@dataclass(frozen=True)
class DialectOptions:
    """Audited options meaningful only to one named wire dialect."""

    dialect_id: str
    values: Mapping[str, JSONValue]

    def __post_init__(self) -> None:
        if not self.dialect_id:
            raise ValueError("DialectOptions.dialect_id must not be empty")
        object.__setattr__(self, "values", dict(self.values))


def _legacy_content_part(raw: object) -> ContentPart:
    if isinstance(raw, str):
        return TextPart(raw)
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"chat content part must be a mapping, got {type(raw).__name__}"
        )
    raw_mapping = cast(Mapping[str, object], raw)

    part_type = raw_mapping.get("type")
    if part_type in ("text", "input_text", "output_text"):
        text = raw_mapping.get("text")
        if not isinstance(text, str):
            raise TypeError("text content part requires a string `text`")
        return TextPart(text)
    if part_type in ("image", "image_url", "input_image"):
        source = raw_mapping.get(
            "image_url", raw_mapping.get("source", raw_mapping.get("url"))
        )
        detail = raw_mapping.get("detail")
        if isinstance(source, Mapping):
            source_mapping = cast(Mapping[str, object], source)
            detail = source_mapping.get("detail", detail)
            source = source_mapping.get("url", source_mapping.get("data"))
        if not isinstance(source, str):
            raise TypeError("image content part requires a string URL or data payload")
        if source.startswith("data:"):
            header, separator, data = source.removeprefix("data:").partition(",")
            if not separator or not header.endswith(";base64"):
                raise ValueError("inline image data URLs must use base64 encoding")
            media_type = header.removesuffix(";base64")
            return ImagePart(
                data=data,
                media_type=media_type or None,
                detail=_str_or_none(detail),
            )
        return ImagePart(url=source, detail=_str_or_none(detail))
    if part_type in ("tool_call", "function_call"):
        call_id = raw_mapping.get("id", raw_mapping.get("call_id"))
        function = raw_mapping.get("function")
        if isinstance(function, Mapping):
            function_mapping = cast(Mapping[str, object], function)
            name = function_mapping.get("name")
            arguments = function_mapping.get("arguments")
        else:
            name = raw_mapping.get("name")
            arguments = raw_mapping.get("arguments")
        if not isinstance(call_id, str) or not isinstance(name, str):
            raise TypeError("tool-call content requires string id and name")
        return ToolCallPart(call_id, name, _json_value(arguments))
    if part_type in ("tool_result", "function_result"):
        call_id = raw_mapping.get("tool_call_id", raw_mapping.get("call_id"))
        if not isinstance(call_id, str):
            raise TypeError("tool-result content requires a string tool_call_id")
        return ToolResultPart(
            call_id,
            _json_value(raw_mapping.get("content", raw_mapping.get("result"))),
            bool(raw_mapping.get("is_error", False)),
        )
    raise ValueError(f"unsupported chat content part type: {part_type!r}")


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _json_value(value: object) -> JSONValue:
    """Validate and detach a JSON value received through a legacy mapping."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON values must not contain non-finite floats")
        return value
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, Mapping):
        copied: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            copied[key] = _json_value(item)
        return copied
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    raise TypeError(f"value must be JSON-compatible, got {type(value).__name__}")


def normalize_chat_input(
    messages: Iterable[ChatMessage | Mapping[str, object]],
) -> ChatInput:
    """Normalize legacy OpenAI-shaped message dictionaries once at the edge."""

    normalized: list[ChatMessage] = []
    for raw in messages:
        if isinstance(raw, ChatMessage):
            normalized.append(raw)
            continue
        if not isinstance(raw, Mapping):
            raise TypeError(f"chat message must be a mapping, got {type(raw).__name__}")
        role = raw.get("role")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported chat role: {role!r}")

        content = raw.get("content", "")
        parts: list[ContentPart] = []
        if isinstance(content, str):
            parts.append(TextPart(content))
        elif isinstance(content, Iterable) and not isinstance(content, Mapping | bytes):
            parts.extend(_legacy_content_part(part) for part in content)
        elif content is None:
            pass
        else:
            raise TypeError(
                "chat message content must be a string or iterable of parts"
            )

        raw_calls = raw.get("tool_calls")
        if raw_calls is not None:
            if not isinstance(raw_calls, Iterable) or isinstance(
                raw_calls, Mapping | str | bytes
            ):
                raise TypeError("tool_calls must be an iterable")
            for index, call in enumerate(raw_calls):
                if not isinstance(call, Mapping):
                    raise TypeError(
                        f"tool_calls[{index}] must be a mapping, got "
                        f"{type(call).__name__}"
                    )
                parts.append(_legacy_content_part({**dict(call), "type": "tool_call"}))
        tool_call_id = raw.get("tool_call_id")
        if role == "tool" and isinstance(tool_call_id, str):
            result: JSONValue
            if len(parts) == 1 and isinstance(parts[0], TextPart):
                result = parts[0].text
            else:
                result = str(content)
            parts = [ToolResultPart(tool_call_id, result)]

        normalized.append(
            ChatMessage(
                role=cast(ChatRole, role),
                content=tuple(parts),
                name=_str_or_none(raw.get("name")),
            )
        )
    return ChatInput(tuple(normalized))


# ---------------------------------------------------------------------------
# Request groups
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplingParams:
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    stop: tuple[str, ...] | None = None
    seed: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    n: int = 1


@dataclass(frozen=True)
class ScoringParams:
    input_scoring: bool = False
    sampled_logprobs: bool = False
    top_logprobs: int = 0


@dataclass(frozen=True)
class ReasoningParams:
    effort: str | None = None
    budget_tokens: int | None = None
    summary: Literal["none", "auto", "concise", "detailed"] | None = None

    def __post_init__(self) -> None:
        if self.effort is not None and self.budget_tokens is not None:
            raise ValueError(
                "reasoning effort and budget_tokens are mutually exclusive"
            )
        if self.budget_tokens is not None and self.budget_tokens < 1:
            raise ValueError("reasoning budget_tokens must be >= 1")


@dataclass(frozen=True)
class HostedToolSpec:
    kind: str
    config: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", dict(self.config))


@dataclass(frozen=True)
class ToolParams:
    functions: tuple[Mapping[str, JSONValue], ...] = ()
    choice: JSONValue = None
    parallel: bool | None = None
    hosted: tuple[HostedToolSpec, ...] = ()


@dataclass(frozen=True)
class StructuredOutputParams:
    format: Literal["json_object", "json_schema"] | None = None
    schema: Mapping[str, JSONValue] | None = None
    name: str | None = None
    strict: bool | None = None

    def __post_init__(self) -> None:
        if self.schema is not None:
            object.__setattr__(self, "schema", dict(self.schema))
        if self.format == "json_schema" and self.schema is None:
            raise ValueError("json_schema structured output requires a schema")


@dataclass(frozen=True)
class OpaqueContinuation:
    dialect_id: str
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.dialect_id, str) or not self.dialect_id:
            raise ValueError("OpaqueContinuation.dialect_id must not be empty")
        if not isinstance(self.value, str) or not self.value:
            raise ValueError("OpaqueContinuation.value must not be empty")


@dataclass(frozen=True)
class SessionParams:
    previous_response_id: str | None = None
    opaque_continuation: OpaqueContinuation | None = None


@dataclass(frozen=True)
class SchedulingParams:
    stream: bool = False


@dataclass(frozen=True)
class Request:
    input: ModelInput
    sampling: SamplingParams = field(default_factory=SamplingParams)
    scoring: ScoringParams = field(default_factory=ScoringParams)
    reasoning: ReasoningParams = field(default_factory=ReasoningParams)
    tools: ToolParams = field(default_factory=ToolParams)
    structured_output: StructuredOutputParams = field(
        default_factory=StructuredOutputParams
    )
    session: SessionParams = field(default_factory=SessionParams)
    scheduling: SchedulingParams = field(default_factory=SchedulingParams)
    dialect_options: DialectOptions | None = None


# ---------------------------------------------------------------------------
# Persisted response records
# ---------------------------------------------------------------------------


@sieval_record
@dataclass(frozen=True)
class TokenLogprob:
    token: str
    logprob: float | None = None
    token_id: int | None = None


@sieval_record
@dataclass(frozen=True)
class TopKEntry:
    token: str
    logprob: float
    token_id: int | None = None


@sieval_record
@dataclass(frozen=True)
class InputScoringResult:
    token_logprobs: tuple[TokenLogprob, ...]
    byte_count: int | None = None
    char_count: int | None = None


@sieval_record
@dataclass(frozen=True)
class ReasoningOutput:
    text: str | None = None
    opaque_roundtrip: str | None = None
    thinking_tokens: int = 0
    effort_used: str | None = None


@sieval_record
@dataclass(frozen=True)
class FunctionToolCall:
    call_id: str
    name: str
    arguments: JSONValue


@sieval_record
@dataclass(frozen=True)
class ServerToolUse:
    tool_type: str
    tool_use_id: str
    input: Mapping[str, JSONValue]
    result: JSONValue = None
    error_code: str | None = None


# Provider-neutral spelling used by the keyed capability plane.  The persisted
# field/class keep their PR #45 names because Response is an additive schema.
HostedToolUse = ServerToolUse
ServerToolSpec = HostedToolSpec
ServerToolType = str


@sieval_record
@dataclass(frozen=True)
class StructuredOutput:
    """Present wrapper keeps JSON null distinct from an absent channel."""

    value: JSONValue = None


@sieval_record
@dataclass(frozen=True)
class Citation:
    url: str
    title: str | None = None
    page_age: str | None = None


@sieval_record
@dataclass(frozen=True)
class GroundingChunk:
    uri: str
    title: str | None = None


@sieval_record
@dataclass(frozen=True)
class GroundingMetadata:
    chunks: tuple[GroundingChunk, ...]
    rendered_content: str | None = None


@sieval_record
@dataclass(frozen=True)
class UsageStats:
    """Token counts for one call, plus whatever breakdown the server reported.

    The trailing fields are **breakdowns of** ``input_tokens`` /
    ``output_tokens``, not addends: summing them double-counts. Each is
    ``None`` when the server did not report it, which ``0`` cannot express --
    most OpenAI-compatible servers omit the detail objects entirely, so a zero
    default would make "no cache hits" and "never said" the same number, and
    any average over a mixed fleet silently wrong.

    No subset relation is enforced. ``reasoning_tokens <= output_tokens`` is an
    OpenAI convention rather than a wire guarantee, and a server that counts
    reasoning outside its completion count is exactly the one whose reported
    total exceeds the computed one -- which ``reported_total_tokens`` records
    instead of rejecting.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    accepted_prediction_tokens: int | None = None
    rejected_prediction_tokens: int | None = None
    reported_total_tokens: int | None = None


@sieval_record
@dataclass(frozen=True)
class ModelIdentity:
    requested_model_id: str
    provider_reported_model_id: str | None = None


@sieval_record
@dataclass(frozen=True)
class CapabilityEvidence:
    declared: Mapping[str, JSONValue]
    effective: Mapping[str, JSONValue]
    plan_fingerprint: str
    verification_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "declared", dict(self.declared))
        object.__setattr__(self, "effective", dict(self.effective))


@sieval_record
@dataclass(frozen=True)
class ModelProvenance:
    dialect_id: str
    engine_id: str
    engine_source: EngineSource
    model_identity: ModelIdentity
    deployment_id: str | None = None
    engine_version: str | None = None
    deployment_fingerprint: str | None = None
    capabilities: CapabilityEvidence | None = None


_CORE = {"response_role": "core", "choice_indexed": True}
_CHOICE_CHANNEL = {"response_role": "channel", "choice_indexed": True}
_SINGULAR_CHANNEL = {"response_role": "channel", "choice_indexed": False}
_PROVENANCE = {"response_role": "provenance", "choice_indexed": False}


@sieval_record
@dataclass(frozen=True)
class Response:
    """Closed additive response schema shared by every dialect."""

    texts: tuple[str, ...] = field(metadata=_CORE)
    reasoning: tuple[ReasoningOutput | None, ...] | None = field(
        default=None, metadata=_CHOICE_CHANNEL
    )
    finish_reasons: tuple[str, ...] | None = field(default=None, metadata=_CORE)

    tool_calls: tuple[FunctionToolCall, ...] | None = field(
        default=None, metadata=_SINGULAR_CHANNEL
    )
    server_tool_uses: tuple[ServerToolUse, ...] | None = field(
        default=None, metadata=_SINGULAR_CHANNEL
    )
    structured_output: StructuredOutput | None = field(
        default=None, metadata=_SINGULAR_CHANNEL
    )
    logprobs: tuple[TokenLogprob, ...] | None = field(
        default=None, metadata=_SINGULAR_CHANNEL
    )
    top_logprobs: tuple[tuple[TopKEntry, ...], ...] | None = field(
        default=None, metadata=_SINGULAR_CHANNEL
    )
    input_scoring: InputScoringResult | None = field(
        default=None, metadata=_SINGULAR_CHANNEL
    )
    citations: tuple[Citation, ...] | None = field(
        default=None, metadata=_SINGULAR_CHANNEL
    )
    grounding: GroundingMetadata | None = field(
        default=None, metadata=_SINGULAR_CHANNEL
    )
    session_id: str | None = field(default=None, metadata=_SINGULAR_CHANNEL)
    usage: UsageStats | None = field(default=None, metadata=_SINGULAR_CHANNEL)

    request_params: Mapping[str, JSONValue] | None = field(
        default=None, metadata=_PROVENANCE
    )
    response_model: str | None = field(default=None, metadata=_PROVENANCE)
    system_fingerprint: str | None = field(default=None, metadata=_PROVENANCE)
    provenance: ModelProvenance | None = field(default=None, metadata=_PROVENANCE)

    def __post_init__(self) -> None:
        if self.request_params is not None:
            object.__setattr__(self, "request_params", dict(self.request_params))
        if self.reasoning is not None and len(self.reasoning) != len(self.texts):
            raise ValueError("Response.reasoning must align with Response.texts")
        if self.finish_reasons is not None and len(self.finish_reasons) != len(
            self.texts
        ):
            raise ValueError("Response.finish_reasons must align with Response.texts")


def response_field_contract() -> dict[str, tuple[str, bool]]:
    """Return the mechanically declared role/cardinality of every root field."""

    result: dict[str, tuple[str, bool]] = {}
    for item in Response.__dataclass_fields__.values():
        role = item.metadata.get("response_role")
        choice_indexed = item.metadata.get("choice_indexed")
        if not isinstance(role, str) or not isinstance(choice_indexed, bool):
            raise AssertionError(f"unclassified Response field: {item.name}")
        result[item.name] = (role, choice_indexed)
    return result
