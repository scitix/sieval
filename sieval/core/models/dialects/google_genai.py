"""Google Gemini native ``generateContent`` dialect for the model IR.

The adapter speaks the public REST protocol through the provider-neutral
``async_http_json`` connection.  Request leaves are translated only where the
shared IR has a lossless Google representation; provider-only controls remain
audited dialect options and ambiguous response shapes fail loudly.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import cast
from urllib.parse import quote

import httpx

from sieval.core.models._shared import copy_json_value
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
from sieval.core.models.connection_factory import AsyncHTTPJSONConnection
from sieval.core.models.deployment import BINDING_RESOURCE_KEYS
from sieval.core.models.dialect import (
    DialectError,
    Guarantee,
    OutputContract,
    OutputContractError,
    OutputRule,
    PreparedRequest,
    PreparedRequestEvidence,
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
    FINISH_SOURCE_CANDIDATE,
    FINISH_SOURCE_PROMPT_BLOCK,
    REQUEST_ENVELOPE_HTTP,
    REQUEST_ENVELOPE_KEY,
    ChatInput,
    ChatMessage,
    Citation,
    FunctionToolCall,
    GroundingChunk,
    GroundingMetadata,
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

from ._google_genai_wire import (
    GoogleWireContract,
    inspect_function_schema,
    lower_function_schema,
    resolve_wire_endpoint,
    validate_function_parameters,
    validate_wire_body,
)

_THINKING_LEVELS = frozenset({"minimal", "low", "medium", "high"})
_HOSTED_TOOL_KINDS = frozenset({"google_search", "url_context"})
_IMAGE_MEDIA_TYPES = frozenset(
    {
        "image/avif",
        "image/gif",
        "image/heic",
        "image/heif",
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
    }
)
_OPAQUE_VERSION = 1
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1
_CHAT_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})
_FUNCTION_NAME_PATTERN = re.compile(r"[A-Za-z0-9_:.\-]{1,128}")
_RFC3339_TIMESTAMP_PATTERN = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})T"
    r"(?P<time>\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?"
    r"(?P<offset>Z|[+\-]\d{2}:\d{2})"
)
# Per-model limits are left to Google's 400, not mirrored here. This pattern
# keys only the Gemini 3 call-id/signature contracts; aliases skip it.
_GEMINI_3_MODEL_PATTERN = re.compile(r"gemini-3(?:[.\-]|$)")
_STANDARD_USER_PART_FIELDS = frozenset({"text", "inlineData", "fileData"})
#: Finish reasons that may leave requested channels empty. ``CONTINUATION``
#: carries a ``continuationToken`` this dialect does not replay; like
#: ``MAX_TOKENS`` it counts as truncated.
_CHANNEL_OPTIONAL_FINISH_REASONS = frozenset(
    {
        "MAX_TOKENS",
        "CONTINUATION",
        "SAFETY",
        "RECITATION",
        "LANGUAGE",
        "OTHER",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "MALFORMED_FUNCTION_CALL",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_OTHER",
        "NO_IMAGE",
        "IMAGE_RECITATION",
        "UNEXPECTED_TOOL_CALL",
        "TOO_MANY_TOOL_CALLS",
        "MALFORMED_RESPONSE",
        "ESCALATION",
    }
)


def _requires_exact_function_call_ids(model_name: str) -> bool:
    """Return whether Google guarantees provider-issued function call IDs."""

    return _GEMINI_3_MODEL_PATTERN.match(model_name) is not None


_FATAL_FINISH_REASON_MESSAGES = MappingProxyType(
    {
        "MISSING_THOUGHT_SIGNATURE": (
            "Google rejected the request because continuation history omitted a "
            "required thought signature"
        ),
        "PUP_LIMITED_DISABLED": (
            "Google stopped generation because the account is limited or disabled "
            "by Prohibited Use Policy"
        ),
    }
)
_IR_OWNED_KEYS = BINDING_RESOURCE_KEYS | frozenset(
    {
        "model",
        "contents",
        "systemInstruction",
        "generationConfig",
        "tools",
        "toolConfig",
        "stream",
    }
)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _reject_duplicate_keys(
    pairs: list[tuple[str, JSONValue]],
) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json_loads(payload: str, path: str) -> JSONValue:
    try:
        return cast(
            JSONValue,
            json.loads(
                payload,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            ),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise OutputContractError(f"{path} is not strict JSON") from exc


def _tuple_validator(item_type: type, channel: str) -> Callable[[object], None]:
    def validate(value: object) -> None:
        if not isinstance(value, tuple) or not all(
            isinstance(item, item_type) for item in value
        ):
            raise OutputContractError(f"{channel} channel has invalid shape")

    return validate


def _validate_top_logprobs_config(options: object) -> None:
    assert isinstance(options, TopLogprobsOptions)
    if options.minimum > 20:
        raise ValueError("google_genai supports at most 20 top logprobs")


def _validate_reasoning_config(options: object) -> None:
    assert isinstance(options, ReasoningOptions)
    if options.summary not in {None, "none", "auto"}:
        raise ValueError(
            "google_genai supports only reasoning summary 'none' or 'auto'"
        )
    if options.effort is not None and options.effort.lower() not in _THINKING_LEVELS:
        raise ValueError(f"unsupported Google thinking level {options.effort!r}")


def _validate_hosted_tools_config(options: object) -> None:
    assert isinstance(options, HostedToolsOptions)
    unsupported = set(options.kinds) - _HOSTED_TOOL_KINDS
    if unsupported:
        raise ValueError(
            "google_genai supports only google_search and url_context hosted tools; "
            f"unsupported kinds: {sorted(unsupported)}"
        )


def _validate_multimodal_config(options: object) -> None:
    assert isinstance(options, MultimodalInputOptions)
    unsupported = set(options.modalities) - {"image"}
    if unsupported:
        raise ValueError(f"google_genai does not support modalities {unsupported!r}")


def _decisions() -> Mapping[CapabilityKey, DialectCapabilityDecision]:
    decisions: dict[CapabilityKey, DialectCapabilityDecision] = {
        "input_scoring": Unsupported(
            "Gemini generateContent cannot score input tokens"
        ),
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
                request_leaves=("tools.functions", "tools.choice"),
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
        "stateful_session": Unsupported(
            "Gemini responseId is not a request continuation handle"
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
                ),
                _config_validator=_validate_multimodal_config,
            )
        ),
        "prefill": Unsupported("Gemini has no independent assistant-prefill field"),
        "fim": Unsupported("Gemini generateContent has no completion suffix field"),
    }
    if set(decisions) != set(CAPABILITY_KEYS):
        raise AssertionError("Google GenAI capability row is incomplete")
    return MappingProxyType(decisions)


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
            Guarantee.BEST_EFFORT, _tuple_validator(Citation, "citations")
        ),
        "grounding": OutputRule(Guarantee.BEST_EFFORT),
        "session_id": OutputRule(Guarantee.NEVER),
        "usage": OutputRule(Guarantee.BEST_EFFORT),
    },
    empty_outcome_finish_reasons=_CHANNEL_OPTIONAL_FINISH_REASONS,
)

type _SyntheticCallKey = tuple[int, int, str]


@dataclass(frozen=True)
class _GoogleContext:
    request: Request
    synthetic_call_ids: Mapping[_SyntheticCallKey, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "synthetic_call_ids",
            MappingProxyType(dict(self.synthetic_call_ids)),
        )


@dataclass(frozen=True)
class _GoogleExecutionContext:
    request: Request
    request_params: Mapping[str, JSONValue]
    system_instruction: Mapping[str, JSONValue] | None
    input_contents: tuple[Mapping[str, JSONValue], ...]
    synthetic_call_ids: Mapping[_SyntheticCallKey, str]
    historical_function_call_ids: frozenset[str]
    historical_server_call_ids: frozenset[str]


@dataclass(frozen=True)
class _LegacyPlan:
    dialect_id: str = "google_genai"
    available_capabilities: frozenset[str] = frozenset(
        key
        for key, decision in CAPABILITY_DECISIONS.items()
        if isinstance(decision, Supported)
    )
    capability_minimums: Mapping[str, Mapping[str, JSONValue]] = field(
        default_factory=dict
    )
    required_output_channels: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _OperationVerifier:
    model_name: str
    api_version: str
    stream: bool

    def verify_prepared(self, prepared: PreparedRequest) -> str | None:
        """Derive the expected route independently from request lowering."""

        method = "streamGenerateContent" if self.stream else "generateContent"
        expected = (
            f"{self.api_version}/models/{quote(self.model_name, safe='')}:{method}"
        )
        if self.stream:
            expected = f"{expected}?alt=sse"
        if prepared.operation != expected:
            return f"operation {prepared.operation!r} does not match {expected!r}"
        return None

    def verify_http(self, request: httpx.Request, endpoint_root: httpx.URL) -> None:
        """Check the final HTTP target against the bound model and version."""

        method = "streamGenerateContent" if self.stream else "generateContent"
        expected = endpoint_root.join(
            f"{self.api_version}/models/{quote(self.model_name, safe='')}:{method}"
        )
        if self.stream:
            expected = expected.copy_with(query=b"alt=sse")
        if request.method != "POST" or request.url != expected:
            raise DialectError(
                "final Google request URL or method does not match the bound "
                f"{self.api_version} operation"
            )


@dataclass(frozen=True)
class _InputVerifier:
    expected_system: Mapping[str, JSONValue] | None
    expected_contents: tuple[JSONValue, ...]

    @classmethod
    def from_request(
        cls, request_input: ChatInput, opaque: str | None
    ) -> "_InputVerifier":
        system, contents = _verification_input_to_wire(request_input, opaque)
        return cls(system, tuple(contents))

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        if body.get("contents") != list(self.expected_contents):
            return "contents no longer match the provider-neutral input"
        if body.get("systemInstruction") != self.expected_system:
            return "systemInstruction no longer matches system/developer messages"
        return None


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


def _finite_number(raw: object, path: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise OutputContractError(f"{path} must be numeric")
    value = float(raw)
    if not math.isfinite(value):
        raise OutputContractError(f"{path} must be finite")
    return value


def _nonnegative_int(raw: object, path: str, *, default: int = 0) -> int:
    if raw is None:
        return default
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise OutputContractError(f"{path} must be a non-negative integer")
    return raw


def _optional_nonnegative_int(raw: object, path: str) -> int | None:
    if raw is None:
        return None
    return _nonnegative_int(raw, path)


def _function_declaration_source(raw: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    data = dict(raw)
    if data.get("type") == "function":
        nested = data.get("function")
        if nested is not None:
            if set(data) != {"type", "function"} or not isinstance(nested, Mapping):
                raise DialectError("invalid OpenAI-shaped function declaration")
            data = dict(nested)
        else:
            data.pop("type")
    allowed = {"name", "description", "parameters"}
    unknown = set(data) - allowed
    if unknown:
        raise DialectError(
            "unsupported Google function declaration field(s): "
            + ", ".join(sorted(unknown))
        )
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise DialectError("Google function declaration requires a name")
    if _FUNCTION_NAME_PATTERN.fullmatch(name) is None:
        raise DialectError(
            "Google function name must be at most 128 characters and contain "
            "only letters, digits, underscores, colons, dots, or dashes"
        )
    description = data.get("description")
    if description is None:
        raise DialectError(
            "Google function declaration requires a non-empty description"
        )
    if not isinstance(description, str):
        raise DialectError("Google function description must be a string")
    if not description.strip():
        raise DialectError("Google function description must not be empty")
    result: dict[str, JSONValue] = {"name": name, "description": description}
    if "parameters" in data:
        parameters = data["parameters"]
        if not isinstance(parameters, Mapping):
            raise DialectError("Google function parameters must be an object")
        validate_function_parameters(parameters)
        result["parameters"] = dict(parameters)
    return result


def _function_declaration(
    raw: Mapping[str, JSONValue], contract: GoogleWireContract
) -> dict[str, JSONValue]:
    result = _function_declaration_source(raw)
    if "parameters" in result:
        parameters = result.pop("parameters")
        if contract.function_schema_field == "parameters":
            parameters = lower_function_schema(parameters)
        result[contract.function_schema_field] = parameters
    return result


def _tool_choice(
    raw: JSONValue, *, include_server_side_invocations: bool = False
) -> dict[str, JSONValue]:
    if isinstance(raw, str):
        modes = {
            "auto": "VALIDATED" if include_server_side_invocations else "AUTO",
            "none": "NONE",
            "required": "ANY",
        }
        try:
            return {"functionCallingConfig": {"mode": modes[raw]}}
        except KeyError as exc:
            raise DialectError(f"unsupported Google tool choice {raw!r}") from exc
    if not isinstance(raw, Mapping):
        raise DialectError("Google tool choice must be a string or mapping")
    data = dict(raw)
    nested = data.get("function")
    if set(data) == {"type", "function"} and isinstance(nested, Mapping):
        name = nested.get("name")
    elif set(data) == {"type", "name"} and data.get("type") == "function":
        name = data.get("name")
    else:
        raise DialectError("invalid Google function-specific tool choice")
    if not isinstance(name, str) or not name:
        raise DialectError("Google function-specific tool choice requires a name")
    return {
        "functionCallingConfig": {
            "mode": "ANY",
            "allowedFunctionNames": [name],
        }
    }


def _rfc3339_timestamp_ns(value: str) -> int | None:
    """Return a comparable UTC nanosecond value for a Google timestamp."""

    match = _RFC3339_TIMESTAMP_PATTERN.fullmatch(value)
    if match is None:
        return None
    offset = "+00:00" if match["offset"] == "Z" else match["offset"]
    try:
        parsed = datetime.fromisoformat(
            f"{match['date']}T{match['time']}{offset}"
        ).astimezone(UTC)
    except ValueError:
        return None
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = parsed - epoch
    whole_seconds = delta.days * 86400 + delta.seconds
    fraction = (match["fraction"] or "").ljust(9, "0")
    return whole_seconds * 1_000_000_000 + int(fraction or "0")


def _hosted_tool_config_error(kind: str, config: object) -> str | None:
    if not isinstance(config, Mapping):
        return f"Google hosted tool {kind!r} config must be an object"
    data = cast(Mapping[object, object], config)
    if kind == "url_context":
        if data:
            return "Google url_context config must be an empty object"
        return None
    if kind != "google_search":
        return f"unsupported Google hosted tool {kind!r}"

    unknown = set(data) - {"timeRangeFilter", "searchTypes"}
    if unknown:
        return "Google google_search config has unsupported field(s): " + ", ".join(
            sorted(str(key) for key in unknown)
        )
    time_range = data.get("timeRangeFilter")
    if time_range is not None:
        if not isinstance(time_range, Mapping):
            return "Google google_search.timeRangeFilter must be an object"
        interval = cast(Mapping[object, object], time_range)
        interval_unknown = set(interval) - {"startTime", "endTime"}
        if interval_unknown:
            return (
                "Google google_search.timeRangeFilter has unsupported field(s): "
                + ", ".join(sorted(str(key) for key in interval_unknown))
            )
        has_start = "startTime" in interval
        has_end = "endTime" in interval
        if has_start != has_end:
            return (
                "Google google_search.timeRangeFilter requires startTime and "
                "endTime together"
            )
        parsed_times: dict[str, int] = {}
        for key in ("startTime", "endTime"):
            if key in interval and (
                not isinstance(interval[key], str) or not interval[key]
            ):
                return (
                    f"Google google_search.timeRangeFilter.{key} must be a "
                    "non-empty timestamp string"
                )
            if key in interval:
                value = cast(str, interval[key])
                parsed = _rfc3339_timestamp_ns(value)
                if parsed is None:
                    return (
                        f"Google google_search.timeRangeFilter.{key} must be an "
                        "RFC3339 timestamp with an explicit UTC offset"
                    )
                parsed_times[key] = parsed
        if (
            "startTime" in parsed_times
            and "endTime" in parsed_times
            and parsed_times["startTime"] > parsed_times["endTime"]
        ):
            return (
                "Google google_search.timeRangeFilter.startTime must not be after "
                "endTime"
            )

    search_types = data.get("searchTypes")
    if search_types is not None:
        if not isinstance(search_types, Mapping):
            return "Google google_search.searchTypes must be an object"
        search_mapping = cast(Mapping[object, object], search_types)
        search_unknown = set(search_mapping) - {"webSearch", "imageSearch"}
        if search_unknown:
            return (
                "Google google_search.searchTypes has unsupported field(s): "
                + ", ".join(sorted(str(key) for key in search_unknown))
            )
        for key in ("webSearch", "imageSearch"):
            if key in search_mapping and (
                not isinstance(search_mapping[key], Mapping)
                or bool(search_mapping[key])
            ):
                return f"Google google_search.searchTypes.{key} must be an empty object"
    return None


def _validate_final_hosted_tools(body: Mapping[str, JSONValue]) -> None:
    raw_tools = body.get("tools")
    if not isinstance(raw_tools, Sequence) or isinstance(raw_tools, str | bytes):
        return
    for index, raw_tool in enumerate(raw_tools):
        if not isinstance(raw_tool, Mapping):
            continue
        for field_name, kind in (
            ("googleSearch", "google_search"),
            ("urlContext", "url_context"),
        ):
            if field_name not in raw_tool:
                continue
            error = _hosted_tool_config_error(kind, raw_tool[field_name])
            if error is not None:
                raise DialectError(f"final Google tools[{index}]: {error}")


def _validate_final_function_call_ids(
    body: Mapping[str, JSONValue],
    model_name: str,
) -> None:
    if not _requires_exact_function_call_ids(model_name):
        return
    contents = body.get("contents")
    if not isinstance(contents, Sequence) or isinstance(contents, str | bytes):
        return
    for content_index, content in enumerate(contents):
        if not isinstance(content, Mapping):
            continue
        parts = content.get("parts")
        if not isinstance(parts, Sequence) or isinstance(parts, str | bytes):
            continue
        for part_index, part in enumerate(parts):
            if not isinstance(part, Mapping):
                continue
            for field_name in ("functionCall", "functionResponse"):
                if field_name not in part:
                    continue
                raw_call = part[field_name]
                if not isinstance(raw_call, Mapping):
                    raise DialectError(
                        f"final Google contents[{content_index}].parts[{part_index}]."
                        f"{field_name} must be an object"
                    )
                call_id = raw_call.get("id")
                if not isinstance(call_id, str) or not call_id:
                    raise DialectError(
                        f"{model_name} requires every {field_name} to contain the "
                        "exact non-empty provider call id"
                    )


def _validate_final_function_call_signatures(
    body: Mapping[str, JSONValue],
    model_name: str,
) -> None:
    """Validate Gemini 3 signatures only within Google's current turn."""

    if _GEMINI_3_MODEL_PATTERN.match(model_name) is None:
        return
    contents = body.get("contents")
    if not isinstance(contents, Sequence) or isinstance(contents, str | bytes):
        return

    current_turn_start = 0
    for content_index, raw_content in enumerate(contents):
        if not isinstance(raw_content, Mapping):
            continue
        content = cast(Mapping[str, object], raw_content)
        if content.get("role") != "user":
            continue
        parts = content.get("parts")
        if not isinstance(parts, Sequence) or isinstance(parts, str | bytes):
            continue
        if any(
            isinstance(part, Mapping) and bool(_STANDARD_USER_PART_FIELDS & part.keys())
            for part in parts
        ):
            current_turn_start = content_index

    for content_index in range(current_turn_start, len(contents)):
        raw_content = contents[content_index]
        if not isinstance(raw_content, Mapping):
            continue
        content = cast(Mapping[str, object], raw_content)
        if content.get("role") != "model":
            continue
        parts = content.get("parts")
        if not isinstance(parts, Sequence) or isinstance(parts, str | bytes):
            continue
        for part_index, part in enumerate(parts):
            if not isinstance(part, Mapping) or "functionCall" not in part:
                continue
            signature = part.get("thoughtSignature")
            if not isinstance(signature, str) or not signature:
                raise DialectError(
                    f"{model_name} current-turn functionCall at "
                    f"contents[{content_index}].parts[{part_index}] must preserve "
                    "the exact non-empty thoughtSignature; replay the returned "
                    "opaque continuation instead of reconstructing Gemini 3 "
                    "tool history"
                )
            break


def _server_tool_signature_error(part: Mapping[str, object], path: str) -> str | None:
    for field_name in ("toolCall", "toolResponse"):
        if field_name not in part:
            continue
        signature = part.get("thoughtSignature")
        if not isinstance(signature, str) or not signature:
            return (
                f"{path}.{field_name} must preserve a non-empty thoughtSignature "
                "for Gemini 3 tool-context circulation"
            )
    return None


def _validate_server_tool_history_signatures(
    body: Mapping[str, JSONValue], model_name: str
) -> None:
    """Check circulated server state, including before the current user turn.

    A subsequent request need not enable hosted tools again. The server-tool
    parts themselves identify circulated history, so omitting the current
    invocation flag must not bypass this check. Ordinary functionCall parts
    retain their separate first-call/current-turn signature contract.
    """

    if _GEMINI_3_MODEL_PATTERN.match(model_name) is None:
        return
    contents = body.get("contents")
    if not isinstance(contents, Sequence) or isinstance(contents, str | bytes):
        return
    for content_index, content in enumerate(contents):
        if not isinstance(content, Mapping):
            continue
        parts = content.get("parts")
        if not isinstance(parts, Sequence) or isinstance(parts, str | bytes):
            continue
        for part_index, part in enumerate(parts):
            if not isinstance(part, Mapping):
                continue
            error = _server_tool_signature_error(
                part, f"Google contents[{content_index}].parts[{part_index}]"
            )
            if error is not None:
                raise DialectError(error)


_PROVIDER_ERROR_MESSAGE_LIMIT = 500


def _raise_for_google_status(response: httpx.Response) -> None:
    """Raise httpx's status error with Google's ``error.message`` appended.

    Per-model limits are left to Google, so its reason is the diagnostic.
    """

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        try:
            payload = json.loads(response.text)
        except ValueError:
            payload = None
        error = payload.get("error") if isinstance(payload, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        if not isinstance(message, str) or not message:
            # A proxy in front of Google can answer with a non-JSON body.
            message = response.text.strip()
        if not message:
            raise
        if len(message) > _PROVIDER_ERROR_MESSAGE_LIMIT:
            message = message[:_PROVIDER_ERROR_MESSAGE_LIMIT] + "..."
        raise httpx.HTTPStatusError(
            f"{exc}\nGoogle: {message}",
            request=exc.request,
            response=exc.response,
        ) from None


def _validate_final_request_constraints(
    body: Mapping[str, JSONValue], model_name: str
) -> None:
    _validate_final_function_call_ids(body, model_name)
    _validate_final_function_call_signatures(body, model_name)
    _validate_server_tool_history_signatures(body, model_name)
    _validate_final_hosted_tools(body)
    tool_config = body.get("toolConfig")
    if not isinstance(tool_config, Mapping):
        return
    tool_config_mapping = cast(Mapping[str, object], tool_config)
    function_config = tool_config_mapping.get("functionCallingConfig")
    function_mode = (
        cast(Mapping[str, object], function_config).get("mode")
        if isinstance(function_config, Mapping)
        else None
    )
    if (
        tool_config_mapping.get("includeServerSideToolInvocations") is True
        and function_mode == "AUTO"
    ):
        raise DialectError(
            "Google server-side tool invocation circulation does not support "
            "function calling mode AUTO; use VALIDATED"
        )


def _allowed_function_call_names(req: Request) -> frozenset[str]:
    declared = frozenset(
        cast(str, _function_declaration_source(item)["name"])
        for item in req.tools.functions
    )
    choice = req.tools.choice
    if choice is None:
        return declared
    if isinstance(choice, str):
        return frozenset() if choice == "none" else declared
    normalized = _tool_choice(choice)
    allowed = _get(
        _get(normalized, "functionCallingConfig"),
        "allowedFunctionNames",
        [],
    )
    return frozenset(
        _required_string(name, "tool choice allowed function")
        for name in _sequence(allowed, "tool choice allowed functions")
    )


def _allowed_server_tool_types(req: Request) -> frozenset[str]:
    allowed: set[str] = set()
    for tool in req.tools.hosted:
        if tool.kind == "url_context":
            allowed.add("URL_CONTEXT")
            continue
        if tool.kind != "google_search":
            continue
        search_types = tool.config.get("searchTypes")
        if search_types is None:
            allowed.add("GOOGLE_SEARCH_WEB")
            continue
        if not isinstance(search_types, Mapping):
            continue
        search_types_mapping = cast(Mapping[str, object], search_types)
        if search_types_mapping.get("webSearch") is not None:
            allowed.add("GOOGLE_SEARCH_WEB")
        if search_types_mapping.get("imageSearch") is not None:
            allowed.add("GOOGLE_SEARCH_IMAGE")
    return frozenset(allowed)


def _hosted_tool(spec: object) -> dict[str, JSONValue]:
    kind = getattr(spec, "kind", None)
    config = getattr(spec, "config", None)
    if not isinstance(kind, str):
        raise DialectError(f"unsupported Google hosted tool {kind!r}")
    if error := _hosted_tool_config_error(kind, config):
        raise DialectError(error)
    assert isinstance(config, Mapping)
    key = "googleSearch" if kind == "google_search" else "urlContext"
    return {key: dict(config)}


def _image_to_part(part: ImagePart) -> dict[str, JSONValue]:
    if part.detail is not None:
        raise DialectError("Google image parts have no per-image detail field")
    if part.media_type is not None and part.media_type not in _IMAGE_MEDIA_TYPES:
        raise DialectError(f"unsupported Google image media type {part.media_type!r}")
    if part.url is not None:
        if not part.url:
            raise DialectError("Google image URLs must not be empty")
        data: dict[str, JSONValue] = {"fileUri": part.url}
        if part.media_type is not None:
            data["mimeType"] = part.media_type
        return {"fileData": data}
    if not part.data or not part.media_type:
        raise DialectError(
            "Google inline images require data and an explicit MIME type"
        )
    return {"inlineData": {"data": part.data, "mimeType": part.media_type}}


def _tool_result_payload(part: ToolResultPart) -> dict[str, JSONValue]:
    value = copy_json_value(part.result, "function response")
    if part.is_error:
        return {"error": value}
    if isinstance(value, dict):
        return dict(value)
    return {"result": value}


def _content_parts(
    message: ChatMessage,
    call_names: dict[str, str],
    answered_calls: set[str],
    call_wire_ids: dict[str, str | None],
    synthetic_call_ids: dict[_SyntheticCallKey, str],
    content_index: int | None,
) -> list[JSONValue]:
    parts: list[JSONValue] = []
    for part in message.content:
        if isinstance(part, TextPart):
            if message.role == "tool":
                raise DialectError("Google tool messages may contain only tool results")
            parts.append({"text": part.text})
        elif isinstance(part, ImagePart):
            if message.role == "tool":
                raise DialectError("Google tool messages may contain only tool results")
            parts.append(_image_to_part(part))
        elif isinstance(part, ToolCallPart):
            if message.role != "assistant":
                raise DialectError(
                    "Google historical function calls require assistant role"
                )
            if not part.call_id or not part.name:
                raise DialectError(
                    "Google historical function calls require id and name"
                )
            if part.call_id in call_names:
                raise DialectError(
                    f"duplicate Google function call id {part.call_id!r}"
                )
            call_names[part.call_id] = part.name
            call_wire_ids[part.call_id] = part.call_id
            parts.append(
                {
                    "functionCall": {
                        "id": part.call_id,
                        "name": part.name,
                        "args": copy_json_value(part.arguments, "function arguments"),
                    }
                }
            )
        elif isinstance(part, ToolResultPart):
            if message.role != "tool":
                raise DialectError("Google function results require tool role")
            try:
                name = call_names[part.call_id]
            except KeyError as exc:
                raise DialectError(
                    f"Google function result {part.call_id!r} has no prior call"
                ) from exc
            if part.call_id in answered_calls:
                raise DialectError(
                    f"duplicate Google function result for {part.call_id!r}"
                )
            answered_calls.add(part.call_id)
            wire_id = call_wire_ids[part.call_id]
            response: dict[str, JSONValue] = {
                "name": name,
                "response": _tool_result_payload(part),
            }
            if wire_id is not None:
                response["id"] = wire_id
            parts.append({"functionResponse": response})
            if wire_id is None:
                if content_index is None:
                    raise AssertionError(
                        "Google function responses require a conversation index"
                    )
                synthetic_call_ids[
                    (content_index, len(parts) - 1, "functionResponse")
                ] = part.call_id
        else:
            raise DialectError(f"unsupported Google content part {type(part).__name__}")
    if not parts:
        raise DialectError("Google contents must not contain an empty message")
    return parts


_OPAQUE_PART_FIELDS = frozenset(
    {
        "text",
        "inlineData",
        "fileData",
        "functionCall",
        "functionResponse",
        "toolCall",
        "toolResponse",
    }
)
_PART_AUXILIARY_FIELDS = frozenset(
    {"partMetadata", "mediaResolution", "mediaProcessing", "videoMetadata"}
)
_SYNTHETIC_CALL_FIELDS = frozenset(
    {"functionCall", "functionResponse", "toolCall", "toolResponse"}
)


def _part_auxiliary_error(part: Mapping[str, object], path: str) -> str | None:
    for field_name in ("partMetadata", "mediaResolution", "videoMetadata"):
        value = part.get(field_name)
        if value is not None and not isinstance(value, Mapping):
            return f"{path}.{field_name} must be an object"
    media_processing = part.get("mediaProcessing")
    if media_processing is not None and (
        not isinstance(media_processing, str) or not media_processing
    ):
        return f"{path}.mediaProcessing must be a non-empty string"
    return None


def _validate_opaque_contents(raw: object) -> list[dict[str, JSONValue]]:
    if not isinstance(raw, list) or not raw:
        raise DialectError("Google opaque continuation has no contents")
    copied = copy_json_value(raw, "Google opaque continuation contents")
    if not isinstance(copied, list):
        raise AssertionError("copied Google contents must remain a list")
    result: list[dict[str, JSONValue]] = []
    for content_index, content in enumerate(copied):
        if not isinstance(content, dict) or set(content) != {"role", "parts"}:
            raise DialectError(
                f"Google opaque content {content_index} has invalid shape"
            )
        if content["role"] not in {"user", "model"}:
            raise DialectError(
                f"Google opaque content {content_index} has invalid role"
            )
        parts = content["parts"]
        if not isinstance(parts, list) or not parts:
            raise DialectError(f"Google opaque content {content_index} has no parts")
        for part_index, part in enumerate(parts):
            if not isinstance(part, dict):
                raise DialectError(
                    f"Google opaque part {content_index}.{part_index} is not an object"
                )
            unknown = set(part) - (
                _OPAQUE_PART_FIELDS
                | _PART_AUXILIARY_FIELDS
                | {"thought", "thoughtSignature"}
            )
            if unknown:
                raise DialectError(
                    "Google opaque part has unsupported field(s): "
                    + ", ".join(sorted(unknown))
                )
            data_fields = set(part) & _OPAQUE_PART_FIELDS
            if len(data_fields) != 1:
                raise DialectError("Google opaque part must contain one data field")
            if "text" in part and not isinstance(part["text"], str):
                raise DialectError("Google opaque text part must contain a string")
            for key in data_fields - {"text"}:
                if not isinstance(part[key], dict):
                    raise DialectError(f"Google opaque {key} must be an object")
            if error := _part_auxiliary_error(
                part, f"Google opaque part {content_index}.{part_index}"
            ):
                raise DialectError(error)
            thought = part.get("thought")
            if thought is not None and not isinstance(thought, bool):
                raise DialectError("Google opaque thought marker must be boolean")
            signature = part.get("thoughtSignature")
            if signature is not None and (
                not isinstance(signature, str) or not signature
            ):
                raise DialectError(
                    "Google opaque thoughtSignature must be a non-empty string"
                )
        result.append(content)
    return result


def _has_thought_signature(contents: Sequence[Mapping[str, JSONValue]]) -> bool:
    for content in contents:
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for raw_part in parts:
            if not isinstance(raw_part, Mapping):
                continue
            part = cast(Mapping[str, object], raw_part)
            signature = part.get("thoughtSignature")
            if isinstance(signature, str) and signature:
                return True
    return False


def _validate_opaque_system(
    raw: object,
) -> dict[str, JSONValue] | None:
    if raw is None:
        return None
    copied = copy_json_value(raw, "Google opaque systemInstruction")
    if not isinstance(copied, dict) or set(copied) != {"parts"}:
        raise DialectError("Google opaque systemInstruction has invalid shape")
    parts = copied["parts"]
    if not isinstance(parts, list) or not parts:
        raise DialectError("Google opaque systemInstruction has no parts")
    if any(
        not isinstance(part, dict)
        or set(part) != {"text"}
        or not isinstance(part["text"], str)
        for part in parts
    ):
        raise DialectError("Google opaque systemInstruction must be text-only")
    return cast(dict[str, JSONValue], copied)


def _validate_synthetic_call_ids(
    raw: object,
    contents: Sequence[Mapping[str, JSONValue]],
) -> dict[_SyntheticCallKey, str]:
    if raw is None:
        return {}
    if not isinstance(raw, list):
        raise DialectError("Google opaque syntheticCallIds must be a list")
    result: dict[_SyntheticCallKey, str] = {}
    for index, raw_item in enumerate(raw):
        if not isinstance(raw_item, Mapping) or set(raw_item) != {
            "contentIndex",
            "partIndex",
            "field",
            "id",
        }:
            raise DialectError(
                f"Google opaque syntheticCallIds[{index}] has invalid shape"
            )
        item = cast(Mapping[str, object], raw_item)
        content_index = item["contentIndex"]
        part_index = item["partIndex"]
        field_name = item["field"]
        call_id = item["id"]
        if (
            isinstance(content_index, bool)
            or not isinstance(content_index, int)
            or content_index < 0
            or isinstance(part_index, bool)
            or not isinstance(part_index, int)
            or part_index < 0
        ):
            raise DialectError(
                f"Google opaque syntheticCallIds[{index}] indexes must be "
                "non-negative integers"
            )
        if not isinstance(field_name, str) or field_name not in _SYNTHETIC_CALL_FIELDS:
            raise DialectError(
                f"Google opaque syntheticCallIds[{index}] has invalid field"
            )
        if not isinstance(call_id, str) or not call_id:
            raise DialectError(
                f"Google opaque syntheticCallIds[{index}].id must be non-empty"
            )
        try:
            parts = contents[content_index]["parts"]
            if not isinstance(parts, list):
                raise TypeError
            part = parts[part_index]
        except (IndexError, KeyError, TypeError) as exc:
            raise DialectError(
                f"Google opaque syntheticCallIds[{index}] points outside contents"
            ) from exc
        if not isinstance(part, Mapping) or field_name not in part:
            raise DialectError(
                f"Google opaque syntheticCallIds[{index}] points to the wrong part"
            )
        part_mapping = cast(Mapping[str, object], part)
        raw_call = part_mapping[field_name]
        if not isinstance(raw_call, Mapping):
            raise DialectError(
                f"Google opaque syntheticCallIds[{index}] requires a call without id"
            )
        call_mapping = cast(Mapping[str, object], raw_call)
        raw_id = call_mapping.get("id")
        if raw_id is not None and raw_id != "":
            raise DialectError(
                f"Google opaque syntheticCallIds[{index}] requires a call without id"
            )
        key = (content_index, part_index, field_name)
        if key in result:
            raise DialectError("Google opaque syntheticCallIds contains a duplicate")
        result[key] = call_id
    return result


def _decode_opaque(
    value: str,
) -> tuple[
    dict[str, JSONValue] | None,
    list[dict[str, JSONValue]],
    dict[_SyntheticCallKey, str],
]:
    try:
        raw = json.loads(
            value,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise DialectError("invalid Google opaque continuation JSON") from exc
    if (
        not isinstance(raw, dict)
        or not {"version", "contents"}.issubset(raw)
        or set(raw) - {"version", "contents", "systemInstruction", "syntheticCallIds"}
    ):
        raise DialectError("invalid Google opaque continuation envelope")
    if raw.get("version") != _OPAQUE_VERSION:
        raise DialectError("unsupported Google opaque continuation version")
    system = _validate_opaque_system(raw.get("systemInstruction"))
    contents = _validate_opaque_contents(raw.get("contents"))
    synthetic_call_ids = _validate_synthetic_call_ids(
        raw.get("syntheticCallIds"), contents
    )
    if not _has_thought_signature(contents) and not synthetic_call_ids:
        raise DialectError(
            "Google opaque continuation has no thought signature or synthetic "
            "call correlation"
        )
    return system, contents, synthetic_call_ids


def _local_call_id(
    raw_call: Mapping[str, object],
    key: _SyntheticCallKey,
    synthetic_call_ids: Mapping[_SyntheticCallKey, str],
    path: str,
) -> tuple[str, str | None]:
    raw_call_id = raw_call.get("id")
    if raw_call_id is None or raw_call_id == "":
        try:
            return synthetic_call_ids[key], None
        except KeyError as exc:
            raise DialectError(
                f"{path} without a provider id has no synthetic correlation id"
            ) from exc
    if not isinstance(raw_call_id, str):
        raise DialectError(f"{path}.id must be a string when present")
    return raw_call_id, raw_call_id


def _call_state_from_contents(
    contents: Sequence[Mapping[str, JSONValue]],
    synthetic_call_ids: Mapping[_SyntheticCallKey, str],
) -> tuple[
    dict[str, str],
    set[str],
    dict[str, str | None],
    dict[str, str],
]:
    names: dict[str, str] = {}
    answered: set[str] = set()
    wire_ids: dict[str, str | None] = {}
    pending_calls: dict[str, str] = {}
    server_types: dict[str, str] = {}
    answered_server_calls: set[str] = set()
    for content_index, content in enumerate(contents):
        role = content.get("role")
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        turn_calls: dict[str, str] = {}
        turn_responses: set[str] = set()
        response_only_turn = True
        for part_index, part in enumerate(parts):
            if not isinstance(part, dict):
                response_only_turn = False
                continue
            if set(part) & _OPAQUE_PART_FIELDS != {"functionResponse"}:
                response_only_turn = False
            call = part.get("functionCall")
            if isinstance(call, dict):
                if role != "model":
                    raise DialectError(
                        "Google functionCall history requires a model turn"
                    )
                call_id, wire_id = _local_call_id(
                    call,
                    (content_index, part_index, "functionCall"),
                    synthetic_call_ids,
                    "Google opaque functionCall",
                )
                name = call.get("name")
                if not isinstance(name, str) or not name:
                    raise DialectError(
                        "Google opaque functionCall requires a non-empty name"
                    )
                if call_id in names:
                    raise DialectError(f"duplicate Google function call id {call_id!r}")
                names[call_id] = name
                wire_ids[call_id] = wire_id
                turn_calls[call_id] = name
            response = part.get("functionResponse")
            if isinstance(response, dict):
                if role != "user":
                    raise DialectError(
                        "Google functionResponse history requires a user turn"
                    )
                call_id, _ = _local_call_id(
                    response,
                    (content_index, part_index, "functionResponse"),
                    synthetic_call_ids,
                    "Google opaque functionResponse",
                )
                name = response.get("name")
                try:
                    expected_name = names[call_id]
                except KeyError as exc:
                    raise DialectError(
                        "Google opaque functionResponse at "
                        f"{content_index}.{part_index} has no prior functionCall"
                    ) from exc
                if name != expected_name:
                    raise DialectError(
                        f"Google opaque functionResponse name changed for {call_id!r}"
                    )
                if call_id in answered:
                    raise DialectError(
                        f"duplicate Google function result for {call_id!r}"
                    )
                answered.add(call_id)
                turn_responses.add(call_id)
            server_call = part.get("toolCall")
            if isinstance(server_call, dict):
                call_id, _ = _local_call_id(
                    server_call,
                    (content_index, part_index, "toolCall"),
                    synthetic_call_ids,
                    "Google opaque toolCall",
                )
                tool_type = server_call.get("toolType")
                if not isinstance(tool_type, str) or not tool_type:
                    raise DialectError(
                        "Google opaque toolCall requires a non-empty toolType"
                    )
                if call_id in server_types:
                    raise DialectError(
                        f"duplicate Google server tool call id {call_id!r}"
                    )
                server_types[call_id] = tool_type
            server_response = part.get("toolResponse")
            if not isinstance(server_response, dict):
                continue
            call_id, _ = _local_call_id(
                server_response,
                (content_index, part_index, "toolResponse"),
                synthetic_call_ids,
                "Google opaque toolResponse",
            )
            tool_type = server_response.get("toolType")
            if not isinstance(tool_type, str) or not tool_type:
                raise DialectError(
                    "Google opaque toolResponse requires a non-empty toolType"
                )
            try:
                expected_type = server_types[call_id]
            except KeyError as exc:
                raise DialectError(
                    "Google opaque toolResponse at "
                    f"{content_index}.{part_index} has no prior toolCall"
                ) from exc
            if tool_type != expected_type:
                raise DialectError(
                    f"Google opaque toolResponse type changed for {call_id!r}"
                )
            if call_id in answered_server_calls:
                raise DialectError(
                    f"duplicate Google server tool result for {call_id!r}"
                )
            answered_server_calls.add(call_id)

        if pending_calls:
            expected = set(pending_calls)
            missing = sorted(expected - turn_responses)
            unexpected = sorted(turn_responses - expected)
            if role != "user" or not response_only_turn or missing or unexpected:
                details: list[str] = []
                if missing:
                    details.append(
                        "unanswered function call id(s): "
                        + ", ".join(repr(call_id) for call_id in missing)
                    )
                if unexpected:
                    details.append(
                        "response id(s) outside the immediately preceding calls: "
                        + ", ".join(repr(call_id) for call_id in unexpected)
                    )
                if role != "user" or not response_only_turn:
                    details.append(
                        "the next content turn must contain only matching "
                        "functionResponse parts"
                    )
                raise DialectError(
                    "Google function responses must immediately follow and exactly "
                    "match the preceding function calls; " + "; ".join(details)
                )
            pending_calls = {}
        elif turn_responses:
            raise DialectError(
                "Google functionResponse must answer the immediately preceding "
                "model functionCall turn"
            )

        if turn_calls:
            pending_calls = turn_calls
    return names, answered, wire_ids, pending_calls


def _require_answered_function_calls(
    pending_calls: Mapping[str, str], path: str
) -> None:
    unanswered = sorted(pending_calls)
    if unanswered:
        rendered = ", ".join(repr(call_id) for call_id in unanswered)
        raise DialectError(f"{path} has unanswered function call id(s): {rendered}")


def _historical_output_call_ids(
    contents: Sequence[Mapping[str, JSONValue]],
    synthetic_call_ids: Mapping[_SyntheticCallKey, str],
) -> tuple[frozenset[str], frozenset[str]]:
    """Recover conversation-scoped call ids from the final request body."""

    function_ids: set[str] = set()
    server_ids: set[str] = set()
    for content_index, content in enumerate(contents):
        parts = content.get("parts")
        if not isinstance(parts, list):
            raise DialectError(
                f"final Google contents[{content_index}].parts must be a list"
            )
        for part_index, part in enumerate(parts):
            if not isinstance(part, Mapping):
                raise DialectError(
                    f"final Google contents[{content_index}].parts[{part_index}] "
                    "must be an object"
                )
            for field_name, ids in (
                ("functionCall", function_ids),
                ("toolCall", server_ids),
            ):
                if field_name not in part:
                    continue
                call = part[field_name]
                if not isinstance(call, Mapping):
                    raise DialectError(
                        f"final Google {field_name} history must be an object"
                    )
                call_id, _ = _local_call_id(
                    cast(Mapping[str, object], call),
                    (content_index, part_index, field_name),
                    synthetic_call_ids,
                    f"final Google {field_name} history",
                )
                if call_id in ids:
                    raise DialectError(
                        f"final Google history duplicates {field_name} id {call_id!r}"
                    )
                ids.add(call_id)
    return frozenset(function_ids), frozenset(server_ids)


def _chat_shape_error(request_input: ChatInput) -> str | None:
    """Return a lossless-mapping error for Google chat history, if any."""

    if not request_input.messages:
        return "Google chat input must contain at least one message"
    for message_index, message in enumerate(request_input.messages):
        if message.role not in _CHAT_ROLES:
            return f"Google message {message_index} has invalid role {message.role!r}"
        for part_index, part in enumerate(message.content):
            path = f"messages[{message_index}].content[{part_index}]"
            if isinstance(part, TextPart):
                if not isinstance(part.text, str):
                    return f"{path}.text must be a string"
            elif isinstance(part, ImagePart):
                if (part.url is None) == (part.data is None):
                    return f"{path} must contain exactly one of url or data"
                if part.url is not None and not isinstance(part.url, str):
                    return f"{path}.url must be a string"
                if part.data is not None and not isinstance(part.data, str):
                    return f"{path}.data must be a string"
                if part.media_type is not None and not isinstance(part.media_type, str):
                    return f"{path}.media_type must be a string"
                if part.detail is not None and not isinstance(part.detail, str):
                    return f"{path}.detail must be a string"
            elif isinstance(part, ToolCallPart):
                if not isinstance(part.call_id, str) or not part.call_id:
                    return f"{path}.call_id must be a non-empty string"
                if not isinstance(part.name, str) or not part.name:
                    return f"{path}.name must be a non-empty string"
                if not isinstance(part.arguments, Mapping):
                    return f"{path}.arguments must be an object"
            elif isinstance(part, ToolResultPart):
                if not isinstance(part.call_id, str) or not part.call_id:
                    return f"{path}.call_id must be a non-empty string"
                if not isinstance(part.is_error, bool):
                    return f"{path}.is_error must be a boolean"
            else:
                return f"{path} has unsupported type {type(part).__name__}"
    if any(message.role == "developer" for message in request_input.messages):
        return (
            "Google systemInstruction cannot preserve the distinct developer "
            "message authority"
        )
    if request_input.messages[-1].role == "assistant":
        return "google_genai does not support assistant-prefill input"
    return None


def _require_generation_turn(contents: Sequence[JSONValue]) -> None:
    """Reject a body that cannot ask Google for a new model turn."""

    if not contents or _get(contents[-1], "role") != "user":
        raise DialectError("Google contents must end with a user or tool-result turn")


def _input_to_wire(
    request_input: ChatInput, opaque: str | None
) -> tuple[
    dict[str, JSONValue] | None,
    list[JSONValue],
    dict[_SyntheticCallKey, str],
]:
    if error := _chat_shape_error(request_input):
        raise DialectError(error)
    system_parts: list[JSONValue] = []
    opaque_system, opaque_contents, opaque_call_ids = (
        _decode_opaque(opaque) if opaque is not None else (None, [], {})
    )
    contents: list[JSONValue] = list(opaque_contents)
    synthetic_call_ids = dict(opaque_call_ids)
    call_names, answered_calls, call_wire_ids, _ = _call_state_from_contents(
        opaque_contents, synthetic_call_ids
    )
    saw_conversation = False
    for message in request_input.messages:
        if message.name is not None:
            raise DialectError("Google contents have no ChatMessage.name field")
        if message.role == "system":
            if saw_conversation:
                raise DialectError(
                    "Google system/developer messages must precede conversation turns"
                )
            if any(not isinstance(part, TextPart) for part in message.content):
                raise DialectError("Google systemInstruction is text-only")
            parts = _content_parts(
                message,
                call_names,
                answered_calls,
                call_wire_ids,
                synthetic_call_ids,
                None,
            )
            system_parts.extend(parts)
            continue
        saw_conversation = True
        content_index = len(contents)
        parts = _content_parts(
            message,
            call_names,
            answered_calls,
            call_wire_ids,
            synthetic_call_ids,
            content_index,
        )
        role = "model" if message.role == "assistant" else "user"
        contents.append({"role": role, "parts": parts})
    _, _, _, pending_calls = _call_state_from_contents(
        tuple(cast(list[Mapping[str, JSONValue]], contents)),
        synthetic_call_ids,
    )
    _require_answered_function_calls(pending_calls, "Google conversation history")
    _require_generation_turn(contents)
    current_system: dict[str, JSONValue] | None = (
        {"parts": system_parts} if system_parts else None
    )
    if (
        opaque_system is not None
        and current_system is not None
        and opaque_system != current_system
    ):
        raise DialectError(
            "Google opaque continuation systemInstruction conflicts with "
            "the current request"
        )
    system = current_system if current_system is not None else opaque_system
    return system, contents, synthetic_call_ids


def _verification_input_to_wire(
    request_input: ChatInput, opaque: str | None
) -> tuple[dict[str, JSONValue] | None, list[JSONValue]]:
    """Independently derive the expected input for request-audit verification."""

    if error := _chat_shape_error(request_input):
        raise DialectError(error)
    opaque_system, opaque_contents, synthetic_call_ids = (
        _decode_opaque(opaque) if opaque is not None else (None, [], {})
    )
    contents: list[JSONValue] = [
        copy_json_value(item, "verified Google opaque content")
        for item in opaque_contents
    ]
    system_parts: list[JSONValue] = []
    call_names, answered, call_wire_ids, _ = _call_state_from_contents(
        opaque_contents, synthetic_call_ids
    )
    saw_conversation = False

    for message in request_input.messages:
        if message.name is not None:
            raise DialectError("Google contents have no ChatMessage.name field")
        is_instruction = message.role == "system"
        if is_instruction and saw_conversation:
            raise DialectError(
                "Google system/developer messages must precede conversation turns"
            )
        if not is_instruction:
            saw_conversation = True

        expected_parts: list[JSONValue] = []
        for part in message.content:
            if isinstance(part, TextPart):
                if message.role == "tool":
                    raise DialectError(
                        "Google tool messages may contain only tool results"
                    )
                expected_parts.append({"text": part.text})
                continue
            if isinstance(part, ImagePart):
                if message.role == "tool" or is_instruction:
                    raise DialectError("Google instruction/tool messages are text-only")
                if part.detail is not None:
                    raise DialectError(
                        "Google image parts have no per-image detail field"
                    )
                if (
                    part.media_type is not None
                    and part.media_type not in _IMAGE_MEDIA_TYPES
                ):
                    raise DialectError(
                        f"unsupported Google image media type {part.media_type!r}"
                    )
                if part.url is not None:
                    if not part.url:
                        raise DialectError("Google image URLs must not be empty")
                    file_data: dict[str, JSONValue] = {"fileUri": part.url}
                    if part.media_type is not None:
                        file_data["mimeType"] = part.media_type
                    expected_parts.append({"fileData": file_data})
                else:
                    if not part.data or not part.media_type:
                        raise DialectError(
                            "Google inline images require data and MIME type"
                        )
                    expected_parts.append(
                        {
                            "inlineData": {
                                "data": part.data,
                                "mimeType": part.media_type,
                            }
                        }
                    )
                continue
            if isinstance(part, ToolCallPart):
                if message.role != "assistant":
                    raise DialectError(
                        "Google historical function calls require assistant role"
                    )
                if part.call_id in call_names:
                    raise DialectError(
                        f"duplicate Google function call id {part.call_id!r}"
                    )
                call_names[part.call_id] = part.name
                call_wire_ids[part.call_id] = part.call_id
                expected_parts.append(
                    {
                        "functionCall": {
                            "id": part.call_id,
                            "name": part.name,
                            "args": copy_json_value(
                                part.arguments, "verified function arguments"
                            ),
                        }
                    }
                )
                continue
            if isinstance(part, ToolResultPart):
                if message.role != "tool":
                    raise DialectError("Google function results require tool role")
                if part.call_id not in call_names:
                    raise DialectError(
                        f"Google function result {part.call_id!r} has no prior call"
                    )
                if part.call_id in answered:
                    raise DialectError(
                        f"duplicate Google function result for {part.call_id!r}"
                    )
                answered.add(part.call_id)
                raw_result = copy_json_value(
                    part.result, "verified Google function response"
                )
                response: JSONValue
                if part.is_error:
                    response = {"error": raw_result}
                elif isinstance(raw_result, dict):
                    response = raw_result
                else:
                    response = {"result": raw_result}
                function_response: dict[str, JSONValue] = {
                    "name": call_names[part.call_id],
                    "response": response,
                }
                wire_id = call_wire_ids[part.call_id]
                if wire_id is not None:
                    function_response["id"] = wire_id
                expected_parts.append({"functionResponse": function_response})
                if wire_id is None:
                    synthetic_call_ids[
                        (len(contents), len(expected_parts) - 1, "functionResponse")
                    ] = part.call_id
                continue
            raise DialectError(f"unsupported Google content part {type(part).__name__}")

        if not expected_parts:
            raise DialectError("Google contents must not contain an empty message")
        if is_instruction:
            if any(not isinstance(part, TextPart) for part in message.content):
                raise DialectError("Google systemInstruction is text-only")
            system_parts.extend(expected_parts)
        else:
            role = "model" if message.role == "assistant" else "user"
            contents.append({"role": role, "parts": expected_parts})

    _, _, _, pending_calls = _call_state_from_contents(
        tuple(cast(list[Mapping[str, JSONValue]], contents)),
        synthetic_call_ids,
    )
    _require_answered_function_calls(
        pending_calls, "verified Google conversation history"
    )
    _require_generation_turn(contents)
    current_system: dict[str, JSONValue] | None = (
        {"parts": system_parts} if system_parts else None
    )
    if (
        opaque_system is not None
        and current_system is not None
        and opaque_system != current_system
    ):
        raise DialectError(
            "Google opaque continuation systemInstruction conflicts with "
            "the current request"
        )
    system = current_system if current_system is not None else opaque_system
    return system, contents


def _structured_output(req: Request, text: str) -> StructuredOutput | None:
    if req.structured_output.format is None:
        return None
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
        return StructuredOutput(cast(JSONValue, value))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise OutputContractError("Google structured output is not valid JSON") from exc


def _usage(raw: object) -> UsageStats | None:
    """Lift Google's counts onto the shared accounting.

    Google reports the addends of ``totalTokenCount``; ``candidatesTokenCount``
    excludes thoughts. Folding thoughts into ``output_tokens`` (and tool-use
    prompt into ``input_tokens``) keeps ``reasoning_tokens`` a breakdown of its
    parent. ``promptTokenCount`` already includes ``cachedContentTokenCount``.
    """

    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise OutputContractError("usageMetadata must be an object")
    prompt = _nonnegative_int(_get(raw, "promptTokenCount"), "usage.promptTokenCount")
    output = _nonnegative_int(
        _get(raw, "candidatesTokenCount"), "usage.candidatesTokenCount"
    )
    thoughts = _optional_nonnegative_int(
        _get(raw, "thoughtsTokenCount"), "usage.thoughtsTokenCount"
    )
    tool_use_prompt = _optional_nonnegative_int(
        _get(raw, "toolUsePromptTokenCount"), "usage.toolUsePromptTokenCount"
    )
    reported_total = _optional_nonnegative_int(
        _get(raw, "totalTokenCount"), "usage.totalTokenCount"
    )
    input_tokens = prompt + (tool_use_prompt or 0)
    output_tokens = output + (thoughts or 0)
    total = input_tokens + output_tokens
    return UsageStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total,
        reasoning_tokens=thoughts,
        cached_tokens=(
            _nonnegative_int(
                _get(raw, "cachedContentTokenCount"), "usage.cachedContentTokenCount"
            )
            if _get(raw, "cachedContentTokenCount") is not None
            else None
        ),
        reported_total_tokens=(
            reported_total
            if reported_total is not None and reported_total != total
            else None
        ),
    )


def _logprobs(
    raw: object, path: str, requested_top_k: int
) -> tuple[tuple[TokenLogprob, ...], tuple[tuple[TopKEntry, ...], ...]]:
    chosen = _sequence(_get(raw, "chosenCandidates"), f"{path}.chosenCandidates")
    raw_tops = _get(raw, "topCandidates")
    tops = _sequence(raw_tops, f"{path}.topCandidates") if raw_tops is not None else ()
    if requested_top_k > 0 and len(chosen) != len(tops):
        raise OutputContractError("Google chosen/top logprob positions are misaligned")
    if requested_top_k == 0 and tops and len(chosen) != len(tops):
        raise OutputContractError("Google chosen/top logprob positions are misaligned")
    sampled: list[TokenLogprob] = []
    alternatives: list[tuple[TopKEntry, ...]] = []
    for index, item in enumerate(chosen):
        sampled.append(
            TokenLogprob(
                _required_string(_get(item, "token"), f"{path}.chosen[{index}].token"),
                _finite_number(
                    _get(item, "logProbability"),
                    f"{path}.chosen[{index}].logProbability",
                ),
                _optional_nonnegative_int(
                    _get(item, "tokenId"), f"{path}.chosen[{index}].tokenId"
                ),
            )
        )
        if not tops:
            continue
        entries = _sequence(_get(tops[index], "candidates"), f"{path}.top[{index}]")
        if requested_top_k > 0 and len(entries) != requested_top_k:
            raise OutputContractError(
                "Google returned a different number of top logprobs than requested"
            )
        alternatives.append(
            tuple(
                TopKEntry(
                    _required_string(_get(entry, "token"), f"{path}.top.token"),
                    _finite_number(
                        _get(entry, "logProbability"), f"{path}.top.logprob"
                    ),
                    _optional_nonnegative_int(
                        _get(entry, "tokenId"), f"{path}.top.tokenId"
                    ),
                )
                for entry in entries
            )
        )
    return tuple(sampled), tuple(alternatives)


def _citations(raw: object, path: str) -> list[Citation]:
    if raw is None:
        return []
    if not isinstance(raw, Mapping):
        raise OutputContractError(f"{path} must be an object")
    result: list[Citation] = []
    for index, source in enumerate(
        _sequence(_get(raw, "citationSources", []), f"{path}.citationSources")
    ):
        uri = _get(source, "uri")
        if uri is None:
            raise OutputContractError(f"{path}[{index}].uri is absent")
        result.append(Citation(_required_string(uri, f"{path}[{index}].uri")))
    return result


def _url_context_citations(raw: object, path: str) -> list[Citation]:
    if raw is None:
        return []
    if not isinstance(raw, Mapping):
        raise OutputContractError(f"{path} must be an object")
    result: list[Citation] = []
    for index, item in enumerate(
        _sequence(_get(raw, "urlMetadata", []), f"{path}.urlMetadata")
    ):
        url = _get(item, "retrievedUrl")
        if url is None:
            raise OutputContractError(f"{path}.urlMetadata[{index}] omitted its URL")
        result.append(Citation(_required_string(url, f"{path}.urlMetadata[{index}]")))
    return result


def _grounding(raw: object, path: str) -> GroundingMetadata | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise OutputContractError(f"{path} must be an object")
    chunks: list[GroundingChunk] = []
    for index, chunk in enumerate(
        _sequence(_get(raw, "groundingChunks", []), f"{path}.groundingChunks")
    ):
        sources = [
            _get(chunk, name)
            for name in ("web", "retrievedContext", "maps", "image")
            if _get(chunk, name) is not None
        ]
        if len(sources) != 1:
            raise OutputContractError(f"{path}.groundingChunks[{index}] is ambiguous")
        source = sources[0]
        uri = (
            _get(source, "sourceUri", _get(source, "imageUri"))
            if _get(chunk, "image") is not None
            else _get(source, "uri")
        )
        if uri is None:
            raise OutputContractError(
                f"{path}.groundingChunks[{index}] has no representable URI"
            )
        chunks.append(
            GroundingChunk(
                _required_string(uri, f"{path}.groundingChunks[{index}].uri"),
                _optional_string(
                    _get(source, "title"), f"{path}.groundingChunks[{index}].title"
                ),
            )
        )
    entry = _get(raw, "searchEntryPoint")
    rendered = _optional_string(
        _get(entry, "renderedContent") if entry is not None else None,
        f"{path}.searchEntryPoint.renderedContent",
    )
    return GroundingMetadata(tuple(chunks), rendered)


def _prompt_block_reason(raw: object) -> str | None:
    feedback = _get(raw, "promptFeedback")
    if feedback is None:
        return None
    if not isinstance(feedback, Mapping):
        raise OutputContractError("response.promptFeedback must be an object")
    feedback_mapping = cast(Mapping[str, object], feedback)
    if "blockReason" not in feedback_mapping:
        return None
    reason = feedback_mapping["blockReason"]
    if not isinstance(reason, str) or not reason.strip():
        raise OutputContractError(
            "response.promptFeedback.blockReason must be a non-empty string"
        )
    return None if reason == "BLOCK_REASON_UNSPECIFIED" else reason


def _merge_stream_grounding_metadata(
    accumulated: dict[str, object] | None,
    incoming: object,
    path: str,
) -> dict[str, object]:
    """Merge Google streaming grounding metadata without losing prior evidence."""

    if not isinstance(incoming, Mapping):
        raise OutputContractError(f"{path} must be an object")
    result: dict[str, object] = {} if accumulated is None else accumulated
    for raw_name, value in incoming.items():
        if not isinstance(raw_name, str):
            raise OutputContractError(f"{path} field names must be strings")
        if value is None:
            continue
        if raw_name in {"groundingChunks", "groundingSupports"}:
            target = result.get(raw_name)
            if target is None:
                new_target: list[object] = []
                result[raw_name] = new_target
                target = new_target
            if not isinstance(target, list):
                raise AssertionError("Google grounding accumulator changed shape")
            cast(list[object], target).extend(_sequence(value, f"{path}.{raw_name}"))
            continue
        previous = result.get(raw_name)
        if previous is not None and previous != value:
            raise OutputContractError(
                f"Google stream changed {path}.{raw_name} between events"
            )
        result[raw_name] = value
    return result


def _merge_stream_text_delta(
    previous: object, current: object
) -> dict[str, object] | None:
    """Join wire text deltas only while they remain one semantic part."""

    if not isinstance(previous, Mapping) or not isinstance(current, Mapping):
        return None
    previous_mapping = cast(Mapping[str, object], previous)
    current_mapping = cast(Mapping[str, object], current)
    allowed = {"text", "thought", "thoughtSignature"}
    if set(previous_mapping) - allowed or set(current_mapping) - allowed:
        return None
    previous_text = previous_mapping.get("text")
    current_text = current_mapping.get("text")
    if not isinstance(previous_text, str) or not isinstance(current_text, str):
        return None
    if previous_mapping.get("thought") != current_mapping.get("thought"):
        return None
    previous_signature = previous_mapping.get("thoughtSignature")
    current_signature = current_mapping.get("thoughtSignature")
    # A signature closes the prior semantic part. A signature delivered with
    # the next delta belongs to the part formed by joining that delta backward.
    if previous_signature is not None:
        return None
    if current_signature is not None and (
        not isinstance(current_signature, str) or not current_signature
    ):
        return None
    merged = dict(previous_mapping)
    merged["text"] = previous_text + current_text
    if "thought" in current_mapping:
        merged["thought"] = current_mapping["thought"]
    if current_signature is not None:
        merged["thoughtSignature"] = current_signature
    return merged


def _append_stream_parts(
    accumulated: list[object], incoming: object, path: str
) -> None:
    parts = list(_sequence(incoming, path))
    if accumulated and parts:
        merged = _merge_stream_text_delta(accumulated[-1], parts[0])
        if merged is not None:
            accumulated[-1] = merged
            parts.pop(0)
    accumulated.extend(parts)


def _synthetic_call_id(
    input_contents: Sequence[Mapping[str, JSONValue]],
    candidate_index: int,
    part_index: int,
    kind: str,
    name: str,
    arguments: Mapping[str, JSONValue],
) -> str:
    """Derive a stable IR id when Google omits an optional call id."""

    payload = {
        "input_contents": list(input_contents),
        "candidate_index": candidate_index,
        "part_index": part_index,
        "kind": kind,
        "name": name,
        "arguments": dict(arguments),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    prefix = "google_call" if kind == "function" else "google_server"
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:24]}"


def _opaque_history(
    system_instruction: Mapping[str, JSONValue] | None,
    input_contents: Sequence[Mapping[str, JSONValue]],
    input_synthetic_call_ids: Mapping[_SyntheticCallKey, str],
    output_content: Mapping[str, JSONValue],
    opaque_call_ids: Mapping[int, tuple[str, str]],
) -> str | None:
    history: list[JSONValue] = [
        copy_json_value(item, "Google input history") for item in input_contents
    ]
    copied_output = copy_json_value(output_content, "Google output history")
    if not isinstance(copied_output, dict):
        raise AssertionError("copied Google output content must remain an object")
    copied_parts = copied_output.get("parts")
    if not isinstance(copied_parts, list):
        raise AssertionError("copied Google output content must retain parts")
    synthetic_call_ids = dict(input_synthetic_call_ids)
    output_content_index = len(history)
    for part_index, (field_name, call_id) in opaque_call_ids.items():
        try:
            copied_part = copied_parts[part_index]
        except IndexError as exc:
            raise AssertionError("Google function-call part index changed") from exc
        if not isinstance(copied_part, dict):
            raise AssertionError("Google function-call part changed shape")
        copied_call = copied_part.get(field_name)
        if not isinstance(copied_call, dict):
            raise AssertionError("Google call payload changed shape")
        copied_id = copied_call.get("id")
        if copied_id is not None and copied_id != "":
            raise AssertionError("synthetic Google call unexpectedly gained an id")
        synthetic_call_ids[(output_content_index, part_index, field_name)] = call_id
    history.append(copied_output)
    if (
        not _has_thought_signature(cast(Sequence[Mapping[str, JSONValue]], history))
        and not synthetic_call_ids
    ):
        return None
    envelope: dict[str, JSONValue] = {
        "version": _OPAQUE_VERSION,
        "contents": history,
    }
    if system_instruction is not None:
        envelope["systemInstruction"] = copy_json_value(
            system_instruction, "Google system history"
        )
    if synthetic_call_ids:
        envelope["syntheticCallIds"] = [
            {
                "contentIndex": content_index,
                "partIndex": part_index,
                "field": field_name,
                "id": call_id,
            }
            for (content_index, part_index, field_name), call_id in sorted(
                synthetic_call_ids.items()
            )
        ]
    return json.dumps(
        envelope,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class GoogleGenAIDialect:
    """Executable adapter for Gemini native generateContent REST endpoints."""

    dialect_id = "google_genai"
    connection_family = "async_http_json"
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

    def __init__(self, connection: object, requested_model_id: str):
        if not isinstance(connection, AsyncHTTPJSONConnection):
            raise TypeError("google_genai requires AsyncHTTPJSONConnection")
        if not requested_model_id:
            raise ValueError("requested_model_id must not be empty")
        model_name = requested_model_id.removeprefix("models/")
        if not model_name or "/" in model_name:
            raise ValueError(
                "requested_model_id must be a Gemini model id or models/<id>"
            )
        self._connection = connection
        self._requested_model_id = requested_model_id
        self._model_name = model_name
        self._wire_contract, self._endpoint_root = resolve_wire_endpoint(
            connection.client.base_url
        )

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self.CAPABILITIES

    def validate_request(
        self, req: Request, audit: RequestAudit, plan: RuntimePlanView
    ) -> None:
        audit.require_matches_request(req)
        del plan
        chat_shape_error = (
            _chat_shape_error(req.input) if isinstance(req.input, ChatInput) else None
        )
        invalid_option_paths = (
            {
                f"dialect_options.{key}"
                for key in req.dialect_options.values
                if not isinstance(key, str)
            }
            if req.dialect_options is not None
            else set()
        )
        for path in audit.active_paths:
            if path == "input.chat":
                if chat_shape_error is not None:
                    audit.rejected(path, chat_shape_error)
            elif path == "input.chat.message.name":
                audit.rejected(path, "Google contents have no ChatMessage.name field")
            elif path.startswith("input.completion"):
                audit.rejected(path, "google_genai requires ChatInput")
            elif path == "input.modality.image.detail":
                audit.rejected(path, "Google images have no per-image detail field")
            elif path == "scoring.input_scoring":
                audit.rejected(path, "Gemini cannot score input tokens")
            elif path == "sampling.temperature" and not (
                0 <= cast(float, req.sampling.temperature) <= 2
            ):
                audit.rejected(path, "Google temperature must be between 0 and 2")
            elif path == "sampling.top_p" and not (
                0 <= cast(float, req.sampling.top_p) <= 1
            ):
                audit.rejected(path, "Google top_p must be between 0 and 1")
            elif path == "sampling.top_k" and cast(int, req.sampling.top_k) < 1:
                audit.rejected(path, "Google top_k must be positive")
            elif (
                path == "sampling.top_k" and cast(int, req.sampling.top_k) > _INT32_MAX
            ):
                audit.rejected(path, "Google top_k must fit a signed 32-bit integer")
            elif (
                path == "sampling.max_tokens" and cast(int, req.sampling.max_tokens) < 1
            ):
                audit.rejected(path, "Google max_tokens must be positive")
            elif (
                path == "sampling.max_tokens"
                and cast(int, req.sampling.max_tokens) > _INT32_MAX
            ):
                audit.rejected(
                    path,
                    "Google max_tokens must fit a signed 32-bit integer",
                )
            elif path == "sampling.seed" and not (
                _INT32_MIN <= cast(int, req.sampling.seed) <= _INT32_MAX
            ):
                audit.rejected(path, "Google seed must fit a signed 32-bit integer")
            elif path == "sampling.n" and req.sampling.n > _INT32_MAX:
                audit.rejected(path, "Google n must fit a signed 32-bit integer")
            elif (
                path == "sampling.stop"
                and len(cast(tuple[str, ...], req.sampling.stop)) > 5
            ):
                audit.rejected(path, "Google supports at most five stop sequences")
            elif path == "scoring.top_logprobs" and req.scoring.top_logprobs > 20:
                audit.rejected(path, "Gemini supports at most 20 top logprobs")
            elif path == "reasoning.effort" and (
                not isinstance(req.reasoning.effort, str)
                or req.reasoning.effort.lower() not in _THINKING_LEVELS
            ):
                audit.rejected(path, "unsupported Google thinking level")
            elif path == "reasoning.summary" and req.reasoning.summary not in {
                "none",
                "auto",
            }:
                audit.rejected(path, "Google supports only summary='none' or 'auto'")
            elif (
                path == "reasoning.budget_tokens"
                and cast(int, req.reasoning.budget_tokens) > _INT32_MAX
            ):
                audit.rejected(
                    path,
                    "Google reasoning budget_tokens must fit a signed 32-bit integer",
                )
            elif path == "tools.parallel":
                audit.rejected(path, "Google has no per-request parallel tool switch")
            elif path == "tools.functions":
                try:
                    for function in req.tools.functions:
                        _function_declaration(function, self._wire_contract)
                except DialectError as exc:
                    audit.rejected(path, str(exc))
            elif path == "tools.choice" and not req.tools.functions:
                audit.rejected(path, "Google tool choice requires function tools")
            elif path == "tools.hosted":
                for tool in req.tools.hosted:
                    error = _hosted_tool_config_error(tool.kind, tool.config)
                    if error is not None:
                        audit.rejected(path, error)
                        break
            elif path == "session.previous_response_id":
                audit.rejected(path, "Gemini responseId is not a continuation handle")
            elif (
                path == "structured_output.format"
                and req.structured_output.format
                not in {
                    "json_object",
                    "json_schema",
                }
            ):
                audit.rejected(path, "unsupported Google structured output format")
            elif path == "structured_output.schema" and (
                req.structured_output.format != "json_schema"
            ):
                audit.rejected(
                    path, "Google response schema requires format='json_schema'"
                )
            elif path == "structured_output.name":
                audit.rejected(path, "Google response JSON schemas have no name field")
            elif path == "structured_output.strict":
                audit.rejected(path, "Google exposes no JSON-schema strict switch")
            elif path.startswith("dialect_options."):
                key = path.removeprefix("dialect_options.")
                if path in invalid_option_paths:
                    audit.rejected(path, "Google dialect option keys must be strings")
                elif not key:
                    audit.rejected(path, "dialect option keys must be non-empty")
                elif key in _IR_OWNED_KEYS:
                    audit.rejected(path, f"{key!r} must use its provider-neutral field")
                elif key in self._wire_contract.unavailable_fields.get("request", ()):
                    audit.rejected(
                        path,
                        f"{key!r} is unavailable in Google "
                        f"{self._wire_contract.api_version}",
                    )

    def prepare(self, req: Request, audit: RequestAudit) -> PreparedRequest:
        audit.require_matches_request(req)
        audit.raise_rejections()
        if not isinstance(req.input, ChatInput):
            raise DialectError("google_genai requires ChatInput")
        active = frozenset(audit.active_paths)

        def consume(path: str, *evidence: WireEvidence) -> None:
            if path in active and path not in audit.decisions:
                audit.consumed(path, *evidence)

        opaque = (
            req.session.opaque_continuation.value
            if req.session.opaque_continuation is not None
            else None
        )
        input_verifier = _InputVerifier.from_request(req.input, opaque)
        operation_verifier = _OperationVerifier(
            self._model_name,
            self._wire_contract.api_version,
            req.scheduling.stream,
        )
        system, contents, synthetic_call_ids = _input_to_wire(req.input, opaque)
        body: dict[str, JSONValue] = {"contents": contents}
        if system is not None:
            body["systemInstruction"] = system
        _validate_server_tool_history_signatures(body, self._model_name)
        consume(
            "input.chat", input_verifier, PreparedRequestEvidence(operation_verifier)
        )
        for path in (
            "input.modality.text",
            "input.modality.image",
            "input.modality.image.media_type",
            "input.modality.tool_call",
            "input.modality.tool_result",
            "input.modality.tool_result.is_error",
        ):
            consume(path, input_verifier)
        if opaque is not None:
            consume("session.opaque_continuation", input_verifier)

        generation: dict[str, JSONValue] = {}
        sampling = req.sampling
        fields = {
            "temperature": "temperature",
            "top_p": "topP",
            "top_k": "topK",
            "max_tokens": "maxOutputTokens",
            "stop": "stopSequences",
            "seed": "seed",
            "frequency_penalty": "frequencyPenalty",
            "presence_penalty": "presencePenalty",
            "n": "candidateCount",
        }
        for source, destination in fields.items():
            value = getattr(sampling, source)
            default = getattr(type(sampling)(), source)
            if value != default:
                generation[destination] = (
                    list(value) if isinstance(value, tuple) else value
                )
                consume(
                    f"sampling.{source}",
                    WireObservation(("generationConfig", destination)),
                )

        if req.scoring.sampled_logprobs:
            generation["responseLogprobs"] = True
            consume(
                "scoring.sampled_logprobs",
                WireObservation(("generationConfig", "responseLogprobs")),
            )
        if req.scoring.top_logprobs > 0:
            generation["logprobs"] = req.scoring.top_logprobs
            consume(
                "scoring.top_logprobs",
                WireObservation(("generationConfig", "logprobs")),
            )

        thinking: dict[str, JSONValue] = {}
        if req.reasoning.effort is not None:
            thinking["thinkingLevel"] = req.reasoning.effort.upper()
            consume("reasoning.effort", _ThinkingEffortVerifier(req.reasoning.effort))
        if req.reasoning.budget_tokens is not None:
            thinking["thinkingBudget"] = req.reasoning.budget_tokens
            consume(
                "reasoning.budget_tokens",
                WireObservation(
                    ("generationConfig", "thinkingConfig", "thinkingBudget")
                ),
            )
        if req.reasoning.summary == "none":
            audit.noop(
                "reasoning.summary", "omitting includeThoughts requests no thoughts"
            )
        elif req.reasoning.summary == "auto":
            thinking["includeThoughts"] = True
            consume(
                "reasoning.summary",
                _ReasoningSummaryVerifier(),
            )
        if thinking:
            generation["thinkingConfig"] = thinking

        if req.structured_output.format is not None:
            generation["responseMimeType"] = "application/json"
            consume(
                "structured_output.format",
                _StructuredFormatVerifier(req.structured_output.format),
            )
        if req.structured_output.schema is not None:
            generation["responseJsonSchema"] = dict(req.structured_output.schema)
            consume(
                "structured_output.schema",
                WireObservation(("generationConfig", "responseJsonSchema")),
            )
        if generation:
            body["generationConfig"] = generation

        tools: list[JSONValue] = []
        if req.tools.functions:
            tools.append(
                {
                    "functionDeclarations": [
                        _function_declaration(item, self._wire_contract)
                        for item in req.tools.functions
                    ]
                }
            )
            consume(
                "tools.functions",
                _FunctionsVerifier(req.tools.functions, self._wire_contract),
            )
        if req.tools.hosted:
            tools.extend(_hosted_tool(tool) for tool in req.tools.hosted)
            consume(
                "tools.hosted",
                _HostedToolsVerifier(
                    req.tools.hosted, has_functions=bool(req.tools.functions)
                ),
            )
        if tools:
            body["tools"] = tools
        if req.tools.choice is not None:
            tool_config = _tool_choice(
                req.tools.choice,
                include_server_side_invocations=bool(req.tools.hosted),
            )
            allowed = _get(
                _get(tool_config, "functionCallingConfig"),
                "allowedFunctionNames",
                [],
            )
            declared = {
                _get(_function_declaration_source(item), "name")
                for item in req.tools.functions
            }
            if any(name not in declared for name in _sequence(allowed, "tool choice")):
                raise DialectError("Google tool choice names an undeclared function")
            body["toolConfig"] = tool_config
            consume(
                "tools.choice",
                _ToolChoiceVerifier(
                    req.tools.choice,
                    include_server_side_invocations=bool(req.tools.hosted),
                ),
            )
        if req.tools.hosted:
            existing_tool_config = body.get("toolConfig")
            if existing_tool_config is None:
                merged_tool_config: dict[str, JSONValue] = {}
            elif isinstance(existing_tool_config, Mapping):
                merged_tool_config = dict(existing_tool_config)
            else:
                raise AssertionError("Google toolConfig must remain an object")
            merged_tool_config["includeServerSideToolInvocations"] = True
            body["toolConfig"] = merged_tool_config

        options = req.dialect_options
        if options is not None:
            if not options.values:
                audit.noop(
                    "dialect_options", "empty matching options have no wire semantics"
                )
            for key, value in options.values.items():
                if not isinstance(key, str):
                    raise DialectError("Google dialect option keys must be strings")
                path = f"dialect_options.{key}"
                if path in audit.decisions:
                    continue
                body[key] = value
                audit.consumed(path, _RootOptionVerifier(key, value))

        validate_wire_body(body, self._wire_contract)
        method = "streamGenerateContent" if req.scheduling.stream else "generateContent"
        operation = (
            f"{self._wire_contract.api_version}/models/"
            f"{quote(self._model_name, safe='')}:{method}"
        )
        if req.scheduling.stream:
            operation = f"{operation}?alt=sse"
        if req.scheduling.stream:
            consume(
                "scheduling.stream",
                PreparedRequestEvidence(operation_verifier),
            )
        return PreparedRequest(
            operation,
            body,
            _GoogleContext(deepcopy(req), synthetic_call_ids),
        )

    def _lift(self, raw: object, context: _GoogleExecutionContext) -> Response:
        req = context.request
        prior_opaque = (
            req.session.opaque_continuation.value
            if req.session.opaque_continuation is not None
            else None
        )
        block_reason = _prompt_block_reason(raw)
        raw_candidates = _get(raw, "candidates")
        candidates = (
            ()
            if raw_candidates is None and block_reason is not None
            else _sequence(raw_candidates, "response.candidates")
        )
        if block_reason is not None:
            if candidates:
                raise OutputContractError(
                    "Google response is contradictory: prompt was blocked "
                    f"with reason {block_reason}, but candidates were returned"
                )
            return Response(
                texts=("",) * req.sampling.n,
                finish_reasons=(block_reason,) * req.sampling.n,
                finish_reason_sources=(FINISH_SOURCE_PROMPT_BLOCK,) * req.sampling.n,
                usage=_usage(_get(raw, "usageMetadata")),
                request_params=context.request_params,
                response_model=_optional_string(
                    _get(raw, "modelVersion"), "response.modelVersion"
                ),
            )
        if len(candidates) != req.sampling.n:
            raise OutputContractError(
                f"expected {req.sampling.n} Google candidates, "
                f"received {len(candidates)}"
            )
        texts: list[str] = []
        finish_reasons: list[str] = []
        reasoning: list[ReasoningOutput | None] = []
        saw_thought_parts: list[bool] = []
        calls: list[FunctionToolCall] = []
        call_ids = set(context.historical_function_call_ids)
        server_call_ids = set(context.historical_server_call_ids)
        server_calls: dict[str, tuple[str, Mapping[str, JSONValue]]] = {}
        server_results: dict[str, tuple[str, JSONValue]] = {}
        citations: list[Citation] = []
        groundings: list[GroundingMetadata] = []
        sampled: list[TokenLogprob] = []
        top: list[tuple[TopKEntry, ...]] = []
        saw_logprobs = False
        allowed_function_names = _allowed_function_call_names(req)
        allowed_server_tool_types = _allowed_server_tool_types(req)
        require_server_signatures = (
            _GEMINI_3_MODEL_PATTERN.match(self._model_name) is not None
            and _get(
                _get(_get(context.request_params, "body"), "toolConfig"),
                "includeServerSideToolInvocations",
            )
            is True
        )
        for index, candidate in enumerate(candidates):
            candidate_index = _get(candidate, "index", index)
            if candidate_index != index:
                raise OutputContractError("Google candidate indexes are not ordered")
            finish_reason = _required_string(
                _get(candidate, "finishReason"), "candidate.finishReason"
            )
            if finish_reason == "FINISH_REASON_UNSPECIFIED":
                raise OutputContractError("Google candidate is not terminal")
            fatal_message = _FATAL_FINISH_REASON_MESSAGES.get(finish_reason)
            if fatal_message is not None:
                raise OutputContractError(fatal_message)
            finish_reasons.append(finish_reason)
            content = _get(candidate, "content")
            if content is None:
                content_mapping: Mapping[str, JSONValue] | None = None
                parts: Sequence[object] = ()
            elif isinstance(content, Mapping):
                content_mapping = cast(Mapping[str, JSONValue], content)
                unknown_content = set(content_mapping) - {"role", "parts"}
                if unknown_content:
                    raise OutputContractError(
                        "Google response content has unsupported field(s): "
                        + ", ".join(sorted(unknown_content))
                    )
                role = content_mapping.get("role")
                if role is not None and role != "model":
                    raise OutputContractError(
                        "Google response content role is not model"
                    )
                parts = _sequence(
                    content_mapping.get("parts"), f"candidates[{index}].parts"
                )
            else:
                raise OutputContractError("Google response content must be an object")
            text_parts: list[str] = []
            thought_text: list[str] = []
            opaque_call_ids: dict[int, tuple[str, str]] = {}
            saw_function_call = False
            for part_index, raw_part in enumerate(parts):
                if not isinstance(raw_part, Mapping):
                    raise OutputContractError("Google response part must be an object")
                part = cast(Mapping[str, JSONValue], raw_part)
                unknown_part = set(part) - (
                    _OPAQUE_PART_FIELDS
                    | _PART_AUXILIARY_FIELDS
                    | {"thought", "thoughtSignature"}
                )
                if unknown_part:
                    raise OutputContractError(
                        "Google response part has unsupported field(s): "
                        + ", ".join(sorted(unknown_part))
                    )
                if error := _part_auxiliary_error(
                    part, f"candidates[{index}].parts[{part_index}]"
                ):
                    raise OutputContractError(error)
                thought_marker = part.get("thought")
                if thought_marker is not None and not isinstance(thought_marker, bool):
                    raise OutputContractError(
                        "Google response thought marker must be boolean"
                    )
                signature = part.get("thoughtSignature")
                if signature is not None and (
                    not isinstance(signature, str) or not signature
                ):
                    raise OutputContractError(
                        "Google response thoughtSignature must be a non-empty string"
                    )
                if require_server_signatures:
                    error = _server_tool_signature_error(
                        part, f"candidates[{index}].parts[{part_index}]"
                    )
                    if error is not None:
                        raise OutputContractError(error)
                known = [
                    name
                    for name in ("text", "functionCall", "toolCall", "toolResponse")
                    if name in part
                ]
                if part.get("thought") is True:
                    if known != ["text"] or not isinstance(part.get("text"), str):
                        raise OutputContractError(
                            "Google thought part must contain exactly one text field"
                        )
                    thought_text.append(cast(str, part["text"]))
                    continue
                if len(known) != 1:
                    raise OutputContractError(
                        f"candidates[{index}].parts[{part_index}] has unsupported shape"
                    )
                if known[0] == "text":
                    text_parts.append(_string(part["text"], "response text"))
                elif known[0] == "functionCall":
                    is_first_function_call = not saw_function_call
                    saw_function_call = True
                    raw_call = part["functionCall"]
                    if not isinstance(raw_call, Mapping):
                        raise OutputContractError("functionCall must be an object")
                    call = cast(Mapping[str, object], raw_call)
                    arguments = copy_json_value(
                        _get(call, "args", {}), "functionCall.args"
                    )
                    if not isinstance(arguments, dict):
                        raise OutputContractError("functionCall.args must be an object")
                    name = _required_string(_get(call, "name"), "functionCall.name")
                    if name not in allowed_function_names:
                        if not req.tools.functions:
                            raise OutputContractError(
                                "Google returned an unrequested function call"
                            )
                        raise OutputContractError(
                            f"Google returned function {name!r} outside the "
                            "request's allowed function set"
                        )
                    raw_call_id = _get(call, "id")
                    if raw_call_id is None or raw_call_id == "":
                        if _requires_exact_function_call_ids(self._model_name):
                            raise OutputContractError(
                                f"{self._model_name} response omitted the required "
                                "functionCall.id"
                            )
                        call_id = _synthetic_call_id(
                            context.input_contents,
                            index,
                            part_index,
                            "function",
                            name,
                            arguments,
                        )
                    elif isinstance(raw_call_id, str):
                        call_id = raw_call_id
                    else:
                        raise OutputContractError(
                            "functionCall.id must be a string when present"
                        )
                    if (
                        is_first_function_call
                        and _GEMINI_3_MODEL_PATTERN.match(self._model_name) is not None
                        and signature is None
                    ):
                        raise OutputContractError(
                            f"{self._model_name} function-call response omitted "
                            "the required thoughtSignature on its first "
                            "functionCall part"
                        )
                    if call_id in call_ids:
                        raise OutputContractError(
                            f"duplicate Google function call id {call_id!r}"
                        )
                    call_ids.add(call_id)
                    if raw_call_id is None or raw_call_id == "":
                        opaque_call_ids[part_index] = ("functionCall", call_id)
                    calls.append(
                        FunctionToolCall(
                            call_id,
                            name,
                            arguments,
                        )
                    )
                elif known[0] == "toolCall":
                    raw_item = part["toolCall"]
                    if not isinstance(raw_item, Mapping):
                        raise OutputContractError("toolCall must be an object")
                    item = cast(Mapping[str, object], raw_item)
                    tool_type = _required_string(
                        _get(item, "toolType"), "toolCall.toolType"
                    )
                    raw_tool_id = _get(item, "id")
                    raw_args = copy_json_value(_get(item, "args", {}), "toolCall.args")
                    if not isinstance(raw_args, dict):
                        raise OutputContractError("toolCall.args must be an object")
                    if raw_tool_id is None or raw_tool_id == "":
                        tool_id = _synthetic_call_id(
                            context.input_contents,
                            index,
                            part_index,
                            "server",
                            tool_type,
                            raw_args,
                        )
                    elif isinstance(raw_tool_id, str):
                        tool_id = raw_tool_id
                    else:
                        raise OutputContractError(
                            "toolCall.id must be a string when present"
                        )
                    if tool_type not in allowed_server_tool_types:
                        if not req.tools.hosted:
                            raise OutputContractError(
                                "Google returned an unrequested server tool call"
                            )
                        raise OutputContractError(
                            f"Google returned server tool type {tool_type!r} "
                            "outside the request's allowed hosted-tool set"
                        )
                    if tool_id in server_call_ids:
                        raise OutputContractError(
                            f"duplicate Google server tool call id {tool_id!r}"
                        )
                    server_call_ids.add(tool_id)
                    server_calls[tool_id] = (tool_type, raw_args)
                    if raw_tool_id is None or raw_tool_id == "":
                        opaque_call_ids[part_index] = ("toolCall", tool_id)
                else:
                    if not req.tools.hosted:
                        raise OutputContractError(
                            "Google returned an unrequested server tool response"
                        )
                    raw_item = part["toolResponse"]
                    if not isinstance(raw_item, Mapping):
                        raise OutputContractError("toolResponse must be an object")
                    item = cast(Mapping[str, object], raw_item)
                    tool_type = _required_string(
                        _get(item, "toolType"), "toolResponse.toolType"
                    )
                    raw_tool_id = _get(item, "id")
                    if raw_tool_id is None or raw_tool_id == "":
                        matching = [
                            call_id
                            for call_id, (call_type, _) in server_calls.items()
                            if call_type == tool_type and call_id not in server_results
                        ]
                        if len(matching) != 1:
                            raise OutputContractError(
                                "Google toolResponse without id is ambiguous"
                            )
                        tool_id = matching[0]
                    elif isinstance(raw_tool_id, str):
                        tool_id = raw_tool_id
                    else:
                        raise OutputContractError(
                            "toolResponse.id must be a string when present"
                        )
                    if tool_id in server_results:
                        raise OutputContractError(
                            f"duplicate Google server tool response id {tool_id!r}"
                        )
                    tool_result = copy_json_value(
                        _get(item, "response"), "toolResponse.response"
                    )
                    if not isinstance(tool_result, dict):
                        raise OutputContractError(
                            "toolResponse.response must be an object"
                        )
                    server_results[tool_id] = (tool_type, tool_result)
                    if raw_tool_id is None or raw_tool_id == "":
                        opaque_call_ids[part_index] = ("toolResponse", tool_id)
            text = "".join(text_parts)
            texts.append(text)
            if content_mapping is not None and parts:
                opaque = _opaque_history(
                    context.system_instruction,
                    context.input_contents,
                    context.synthetic_call_ids,
                    content_mapping,
                    opaque_call_ids,
                )
            else:
                # No provider part means there is no new state to append.  The
                # previous signed history is still the exact continuation state,
                # including when thinking consumes the output budget before a
                # visible part is emitted.
                opaque = prior_opaque
            saw_thought_parts.append(bool(thought_text))
            reasoning.append(
                ReasoningOutput(
                    text="\n".join(thought_text) or None,
                    opaque_roundtrip=opaque,
                )
                if thought_text or opaque is not None
                else None
            )
            citations.extend(
                _citations(_get(candidate, "citationMetadata"), "citationMetadata")
            )
            citations.extend(
                _url_context_citations(
                    _get(candidate, "urlContextMetadata"), "urlContextMetadata"
                )
            )
            grounding = _grounding(
                _get(candidate, "groundingMetadata"), "groundingMetadata"
            )
            if grounding is not None:
                groundings.append(grounding)
            raw_logprobs = _get(candidate, "logprobsResult")
            if req.scoring.sampled_logprobs:
                if raw_logprobs is None:
                    if finish_reason in _CHANNEL_OPTIONAL_FINISH_REASONS:
                        continue
                    raise OutputContractError("Google reply omitted requested logprobs")
                saw_logprobs = True
                chosen, alternatives = _logprobs(
                    raw_logprobs,
                    "candidate.logprobsResult",
                    req.scoring.top_logprobs,
                )
                if text and not chosen:
                    raise OutputContractError(
                        "Google returned non-empty text without chosen logprobs"
                    )
                sampled.extend(chosen)
                top.extend(alternatives)
        if req.reasoning.summary == "auto":
            for finish_reason, reasoning_item, saw_thought in zip(
                finish_reasons, reasoning, saw_thought_parts, strict=True
            ):
                if (
                    finish_reason not in _CHANNEL_OPTIONAL_FINISH_REASONS
                    and saw_thought
                    and (
                        reasoning_item is None
                        or reasoning_item.text is None
                        or not reasoning_item.text.strip()
                    )
                ):
                    raise OutputContractError(
                        "Google reply returned an empty requested thought summary"
                    )
        if req.sampling.n > 1 and (citations or groundings):
            raise OutputContractError(
                "Google singular citation/grounding metadata cannot represent choices"
            )
        if len(groundings) > 1:
            raise OutputContractError(
                "multiple candidates returned singular grounding metadata"
            )
        text = texts[0] if len(texts) == 1 else ""
        server_uses: list[ServerToolUse] = []
        for tool_id, (tool_type, tool_input) in server_calls.items():
            result_entry = server_results.pop(tool_id, None)
            if result_entry is not None and result_entry[0] != tool_type:
                raise OutputContractError(
                    f"Google server tool type changed for {tool_id!r}"
                )
            server_uses.append(
                ServerToolUse(
                    tool_type=tool_type,
                    tool_use_id=tool_id,
                    input=tool_input,
                    result=None if result_entry is None else result_entry[1],
                )
            )
        if server_results:
            raise OutputContractError(
                "Google returned a server tool response without its call"
            )
        all_channels_optional = bool(finish_reasons) and all(
            reason in _CHANNEL_OPTIONAL_FINISH_REASONS for reason in finish_reasons
        )
        usage = _usage(_get(raw, "usageMetadata"))
        no_effect = _reasoning_without_effect(
            req,
            usage,
            any(saw_thought_parts),
            all_channels_optional,
        )
        request_params = context.request_params
        if no_effect:
            request_params = {**request_params, "no_effect": list(no_effect)}
        return Response(
            texts=tuple(texts),
            reasoning=(
                tuple(reasoning)
                if any(reasoning) or req.reasoning.summary == "auto"
                else None
            ),
            finish_reasons=tuple(finish_reasons),
            finish_reason_sources=(FINISH_SOURCE_CANDIDATE,) * len(finish_reasons),
            tool_calls=tuple(calls) or None,
            server_tool_uses=tuple(server_uses) or None,
            structured_output=(
                None if all_channels_optional else _structured_output(req, text)
            ),
            logprobs=(
                tuple(sampled)
                if req.scoring.sampled_logprobs and saw_logprobs
                else None
            ),
            top_logprobs=(
                tuple(top) if req.scoring.top_logprobs > 0 and saw_logprobs else None
            ),
            citations=tuple(citations) or None,
            grounding=groundings[0] if groundings else None,
            usage=usage,
            request_params=request_params,
            response_model=_optional_string(
                _get(raw, "modelVersion"), "response.modelVersion"
            ),
        )

    async def _lift_stream(
        self, response: httpx.Response, context: _GoogleExecutionContext
    ) -> Response:
        chunks: list[object] = []
        data_lines: list[str] = []

        def flush_event() -> None:
            if not data_lines:
                return
            payload = "\n".join(data_lines)
            data_lines.clear()
            event = _strict_json_loads(payload, "Google stream event")
            if not isinstance(event, dict) or not event:
                raise OutputContractError(
                    "Google stream event must be a non-empty response object"
                )
            chunks.append(event)

        async for line in response.aiter_lines():
            if not line:
                flush_event()
                continue
            if line.startswith(":"):
                continue
            field, separator, value = line.partition(":")
            if not separator:
                raise OutputContractError("Google stream emitted a malformed SSE line")
            if field == "data":
                data_lines.append(value.removeprefix(" "))
            elif field not in {"event", "id", "retry"}:
                raise OutputContractError(
                    f"Google stream emitted unsupported SSE field {field!r}"
                )
        flush_event()
        if not chunks:
            raise OutputContractError("Google stream emitted no response chunks")
        return self._merge_stream(chunks, context)

    def _merge_stream(
        self, chunks: list[object], context: _GoogleExecutionContext
    ) -> Response:
        # Each stream event carries candidate deltas plus cumulative/terminal
        # metadata. Merge only provider-defined additive content; usage is a
        # terminal snapshot and must come from the final event.
        by_index: dict[int, dict[str, object]] = {}
        model_version: object = None
        prompt_feedback: object = None
        for chunk_index, chunk in enumerate(chunks):
            if _get(chunk, "modelVersion") is not None:
                current_model = _get(chunk, "modelVersion")
                if model_version is not None and current_model != model_version:
                    raise OutputContractError(
                        "Google stream changed modelVersion between events"
                    )
                model_version = current_model
            current_feedback = _get(chunk, "promptFeedback")
            if current_feedback is not None:
                _prompt_block_reason(chunk)
                if prompt_feedback is not None and current_feedback != prompt_feedback:
                    raise OutputContractError(
                        "Google stream changed promptFeedback between events"
                    )
                prompt_feedback = current_feedback
            for raw_candidate in _sequence(
                _get(chunk, "candidates", []), f"stream[{chunk_index}].candidates"
            ):
                index = _get(
                    raw_candidate,
                    "index",
                    0 if context.request.sampling.n == 1 else None,
                )
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise OutputContractError(
                        "Google stream candidate index is invalid"
                    )
                state = by_index.setdefault(index, {"parts": []})
                if "finishReason" in state:
                    raise OutputContractError(
                        f"Google stream emitted candidate {index} after its "
                        "terminal finishReason"
                    )
                content = _get(raw_candidate, "content")
                if content is not None:
                    role = _get(content, "role")
                    if role is not None and role != "model":
                        raise OutputContractError(
                            "Google stream candidate role is not model"
                        )
                    _append_stream_parts(
                        cast(list[object], state["parts"]),
                        _get(content, "parts", []),
                        "stream candidate parts",
                    )
                finish_reason = _get(raw_candidate, "finishReason")
                if finish_reason is not None:
                    state["finishReason"] = finish_reason
                for name in ("citationMetadata", "urlContextMetadata"):
                    value = _get(raw_candidate, name)
                    if value is not None:
                        previous = state.get(name)
                        if previous is not None and previous != value:
                            raise OutputContractError(
                                f"Google stream changed {name} between events"
                            )
                        state[name] = value
                raw_grounding = _get(raw_candidate, "groundingMetadata")
                if raw_grounding is not None:
                    accumulated_grounding = state.get("groundingMetadata")
                    if accumulated_grounding is not None and not isinstance(
                        accumulated_grounding, dict
                    ):
                        raise AssertionError(
                            "Google grounding accumulator changed type"
                        )
                    state["groundingMetadata"] = _merge_stream_grounding_metadata(
                        cast(dict[str, object] | None, accumulated_grounding),
                        raw_grounding,
                        "groundingMetadata",
                    )
                raw_logprobs = _get(raw_candidate, "logprobsResult")
                if raw_logprobs is not None:
                    if not isinstance(raw_logprobs, Mapping):
                        raise OutputContractError(
                            "Google stream logprobsResult must be an object"
                        )
                    accumulated_raw = state.get("logprobsResult")
                    if accumulated_raw is None:
                        accumulated: dict[str, object] = {
                            "chosenCandidates": [],
                            "topCandidates": [],
                        }
                        state["logprobsResult"] = accumulated
                    elif isinstance(accumulated_raw, dict):
                        accumulated = cast(dict[str, object], accumulated_raw)
                    else:
                        raise AssertionError("Google logprobs accumulator changed type")
                    for field_name in ("chosenCandidates", "topCandidates"):
                        value = _get(raw_logprobs, field_name)
                        if value is not None:
                            target = accumulated.get(field_name)
                            if not isinstance(target, list):
                                raise AssertionError(
                                    "Google logprobs accumulator changed shape"
                                )
                            cast(list[object], target).extend(
                                _sequence(
                                    value,
                                    f"stream logprobsResult.{field_name}",
                                )
                            )
        candidates: list[JSONValue] = []
        for index in sorted(by_index):
            state = by_index[index]
            candidate: dict[str, JSONValue] = {
                "index": index,
                "content": {
                    "role": "model",
                    "parts": cast(list[JSONValue], state.pop("parts")),
                },
            }
            for key, value in state.items():
                candidate[key] = copy_json_value(value, f"stream.{key}")
            candidates.append(candidate)
        merged: dict[str, JSONValue] = {"candidates": candidates}
        terminal_usage = _get(chunks[-1], "usageMetadata")
        if terminal_usage is not None:
            merged["usageMetadata"] = copy_json_value(
                terminal_usage, "terminal stream usage"
            )
        if model_version is not None:
            merged["modelVersion"] = copy_json_value(model_version, "stream model")
        if prompt_feedback is not None:
            merged["promptFeedback"] = copy_json_value(
                prompt_feedback, "stream promptFeedback"
            )
        return self._lift(merged, context)

    async def execute(self, prepared: PreparedRequest) -> Response:
        context = prepared.context
        if not isinstance(context, _GoogleContext):
            raise DialectError("google_genai received invalid prepared context")
        operation_verifier = _OperationVerifier(
            self._model_name,
            self._wire_contract.api_version,
            context.request.scheduling.stream,
        )
        operation_mismatch = operation_verifier.verify_prepared(prepared)
        if operation_mismatch is not None:
            raise DialectError(
                f"google_genai received the wrong prepared operation: "
                f"{operation_mismatch}"
            )
        body = prepared.thaw_body()
        raw_contents = body.get("contents")
        if not isinstance(raw_contents, list):
            raise DialectError("prepared Google request has invalid contents")
        copied_contents = copy_json_value(raw_contents, "prepared Google contents")
        if not isinstance(copied_contents, list) or not all(
            isinstance(item, dict) for item in copied_contents
        ):
            raise DialectError("prepared Google request has invalid content items")
        headers = {"Content-Type": "application/json"}
        credential = self._connection.credential
        if credential is not None:
            headers["x-goog-api-key"] = credential
        request = self._connection.client.build_request(
            "POST",
            self._endpoint_root.join(prepared.operation),
            headers=headers,
            json=body,
        )
        operation_verifier.verify_http(request, self._endpoint_root)
        actual_body = _strict_json_loads(
            request.content.decode("utf-8"), "final Google request body"
        )
        if not isinstance(actual_body, dict):
            raise DialectError("final Google request body must be an object")
        validate_wire_body(actual_body, self._wire_contract)
        mismatch = _FunctionsVerifier(
            context.request.tools.functions, self._wire_contract
        ).verify(actual_body)
        if mismatch is not None:
            raise DialectError(f"final Google request: {mismatch}")
        _validate_final_request_constraints(actual_body, self._model_name)
        actual_contents = actual_body.get("contents")
        if not isinstance(actual_contents, list) or not all(
            isinstance(item, dict) for item in actual_contents
        ):
            raise DialectError("final Google request body has invalid contents")
        actual_content_tuple = tuple(
            cast(list[Mapping[str, JSONValue]], actual_contents)
        )
        _, _, _, pending_calls = _call_state_from_contents(
            actual_content_tuple,
            context.synthetic_call_ids,
        )
        _require_answered_function_calls(
            pending_calls, "final Google conversation history"
        )
        historical_function_call_ids, historical_server_call_ids = (
            _historical_output_call_ids(
                actual_content_tuple,
                context.synthetic_call_ids,
            )
        )
        raw_system = actual_body.get("systemInstruction")
        if raw_system is None:
            actual_system: Mapping[str, JSONValue] | None = None
        elif isinstance(raw_system, dict):
            actual_system = cast(Mapping[str, JSONValue], raw_system)
        else:
            raise DialectError(
                "final Google request body has invalid systemInstruction"
            )
        request_params: dict[str, JSONValue] = {
            # An HTTP envelope, not a flat body (see `request_definition`).
            REQUEST_ENVELOPE_KEY: REQUEST_ENVELOPE_HTTP,
            "method": request.method,
            "path": request.url.path,
            "body": {
                key: value
                for key, value in actual_body.items()
                if key not in {"contents", "systemInstruction"}
            },
        }
        actual_alt = request.url.params.get("alt")
        if actual_alt is not None:
            request_params["query"] = {"alt": actual_alt}
        execution_context = _GoogleExecutionContext(
            request=context.request,
            request_params=request_params,
            system_instruction=actual_system,
            input_contents=actual_content_tuple,
            synthetic_call_ids=context.synthetic_call_ids,
            historical_function_call_ids=historical_function_call_ids,
            historical_server_call_ids=historical_server_call_ids,
        )
        if context.request.scheduling.stream:
            response = await self._connection.client.send(request, stream=True)
            try:
                if not response.is_success:
                    await response.aread()
                _raise_for_google_status(response)
                return await self._lift_stream(response, execution_context)
            finally:
                await response.aclose()
        response = await self._connection.client.send(request)
        _raise_for_google_status(response)
        payload = _strict_json_loads(response.text, "Google response body")
        return self._lift(payload, execution_context)

    async def arun(self, req: Request) -> Response:
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


@dataclass(frozen=True)
class _RootOptionVerifier:
    key: str
    value: JSONValue

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "value",
            copy_json_value(self.value, f"Google dialect option {self.key}"),
        )

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        return (
            None
            if self.key in body and body[self.key] == self.value
            else f"Google dialect option {self.key!r} changed"
        )


def _reasoning_without_effect(
    req: Request,
    usage: UsageStats | None,
    saw_thoughts: bool,
    all_channels_optional: bool,
) -> tuple[str, ...]:
    """The reasoning controls Google accepted but the reply shows no sign of.

    Google may accept ``thinkingBudget`` and ``includeThoughts`` and ignore
    them: ``reasoning.budget_tokens`` when a budget got no thought tokens,
    ``reasoning.summary`` when requested thoughts got no thought part. A reply
    that stopped early says nothing either way.
    """
    if all_channels_optional:
        return ()
    found: list[str] = []
    thoughts = usage.reasoning_tokens if usage is not None else None
    budget = req.reasoning.budget_tokens
    if budget is not None and not thoughts:
        found.append("reasoning.budget_tokens")
    if req.reasoning.summary == "auto" and not saw_thoughts:
        found.append("reasoning.summary")
    return tuple(found)


@dataclass(frozen=True)
class _ThinkingEffortVerifier:
    effort: str

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        actual = _get(
            _get(_get(body, "generationConfig"), "thinkingConfig"), "thinkingLevel"
        )
        return None if actual == self.effort.upper() else "thinking level changed"


@dataclass(frozen=True)
class _ReasoningSummaryVerifier:
    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        actual = _get(
            _get(_get(body, "generationConfig"), "thinkingConfig"),
            "includeThoughts",
        )
        return None if actual is True else "visible-thought request changed"


@dataclass(frozen=True)
class _StructuredFormatVerifier:
    format: str

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        generation = _get(body, "generationConfig")
        if _get(generation, "responseMimeType") != "application/json":
            return "structured output MIME type changed"
        has_schema = _get(generation, "responseJsonSchema") is not None
        if has_schema != (self.format == "json_schema"):
            return "structured output schema presence changed"
        return None


@dataclass(frozen=True)
class _FunctionsVerifier:
    functions: tuple[Mapping[str, JSONValue], ...]
    contract: GoogleWireContract

    def __post_init__(self) -> None:
        copied: list[Mapping[str, JSONValue]] = []
        for index, function in enumerate(self.functions):
            item = copy_json_value(function, f"Google function {index}")
            if not isinstance(item, dict):
                raise TypeError("Google function declarations must be objects")
            copied.append(item)
        object.__setattr__(self, "functions", tuple(copied))

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        declarations: list[JSONValue] = []
        for function in self.functions:
            source = dict(function)
            nested = source.get("function")
            if source.get("type") == "function" and isinstance(nested, dict):
                source = dict(nested)
            else:
                source.pop("type", None)
            declaration: dict[str, JSONValue] = {"name": source.get("name")}
            if "description" in source:
                declaration["description"] = source["description"]
            if "parameters" in source:
                declaration["parametersJsonSchema"] = source["parameters"]
            declarations.append(declaration)
        # Inspect actual declarations; never run the lowering encoder on the
        # source schema. In v1, decode only Schema nodes before comparison.
        tools = body.get("tools", [])
        if not isinstance(tools, Sequence) or isinstance(tools, str | bytes):
            return "function tools changed"
        actual: list[JSONValue] = []
        saw_declarations = False
        for tool in tools:
            if not isinstance(tool, Mapping) or "functionDeclarations" not in tool:
                continue
            if (
                saw_declarations
                or not self.functions
                or set(tool) != {"functionDeclarations"}
            ):
                return "function tools changed"
            saw_declarations = True
            raw_declarations = _get(tool, "functionDeclarations")
            if not isinstance(raw_declarations, Sequence) or isinstance(
                raw_declarations, str | bytes
            ):
                return "function tools changed"
            for raw in raw_declarations:
                if not isinstance(raw, Mapping):
                    return "function tools changed"
                try:
                    declaration = dict(
                        cast(Mapping[str, JSONValue], copy_json_value(raw, "function"))
                    )
                except (TypeError, ValueError):
                    return "function tools changed"
                selected = self.contract.function_schema_field
                other = (
                    "parametersJsonSchema" if selected == "parameters" else "parameters"
                )
                if other in declaration:
                    return "function tools changed"
                if selected in declaration and selected == "parameters":
                    try:
                        declaration["parametersJsonSchema"] = inspect_function_schema(
                            declaration.pop("parameters")
                        )
                    except DialectError:
                        return "function tools changed"
                actual.append(declaration)
        # JSON comparison distinguishes booleans from numeric constraints.
        return (
            None
            if json.dumps(actual, sort_keys=True)
            == json.dumps(declarations, sort_keys=True)
            else "function tools changed"
        )


@dataclass(frozen=True)
class _HostedToolsVerifier:
    hosted: tuple[object, ...]
    has_functions: bool = False

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        tools = _get(body, "tools")
        expected: list[dict[str, JSONValue]] = []
        for item in self.hosted:
            kind = getattr(item, "kind", None)
            config = getattr(item, "config", None)
            if not isinstance(config, Mapping):
                return "hosted tool config is invalid"
            key = "googleSearch" if kind == "google_search" else "urlContext"
            expected.append({key: dict(config)})
        if not isinstance(tools, Sequence) or isinstance(tools, str | bytes):
            return "hosted tools changed"
        if self.has_functions:
            if (
                not tools
                or not isinstance(tools[0], Mapping)
                or set(tools[0]) != {"functionDeclarations"}
            ):
                return "hosted tools changed"
            actual = tools[1:]
        else:
            actual = tools
        try:
            matches = json.dumps(
                copy_json_value(actual, "verified Google hosted tools"), sort_keys=True
            ) == json.dumps(
                copy_json_value(expected, "requested Google hosted tools"),
                sort_keys=True,
            )
        except (TypeError, ValueError):
            matches = False
        if not matches:
            return "hosted tools changed"
        if (
            _get(_get(body, "toolConfig"), "includeServerSideToolInvocations")
            is not True
        ):
            return "hosted-tool invocation evidence was disabled"
        return None


@dataclass(frozen=True)
class _ToolChoiceVerifier:
    choice: JSONValue
    include_server_side_invocations: bool = False

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        if isinstance(self.choice, str):
            mode = {
                "auto": (
                    "VALIDATED" if self.include_server_side_invocations else "AUTO"
                ),
                "none": "NONE",
                "required": "ANY",
            }.get(self.choice)
            expected: JSONValue = {"functionCallingConfig": {"mode": mode}}
        elif isinstance(self.choice, dict):
            nested = self.choice.get("function")
            name = (
                nested.get("name")
                if isinstance(nested, dict)
                else self.choice.get("name")
            )
            expected = {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": [name],
                }
            }
        else:
            return "source tool choice is invalid"
        actual = _get(body, "toolConfig")
        if not isinstance(actual, Mapping):
            return "tool choice changed"
        actual_choice = {
            key: value
            for key, value in actual.items()
            if key != "includeServerSideToolInvocations"
        }
        return None if actual_choice == expected else "tool choice changed"
