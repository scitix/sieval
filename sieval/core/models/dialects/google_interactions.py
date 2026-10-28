"""Google Gemini native ``interactions`` dialect for the model IR.

The adapter speaks ``POST {root}/{version}/interactions`` through the
provider-neutral ``async_http_json`` connection.  Request leaves are lowered
only where the shared IR has a lossless Interactions representation; every
other active leaf is rejected before network I/O, and ambiguous replies fail
loudly instead of being reinterpreted.

The terminal ``status`` is recorded verbatim as the finish reason.

AI-Generated Code - Claude Sonnet 5.5 (Anthropic)
"""

import json
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

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
)
from sieval.core.models.ir import (
    REQUEST_ENVELOPE_HTTP,
    REQUEST_ENVELOPE_KEY,
    ChatInput,
    ChatMessage,
    Citation,
    FunctionToolCall,
    ImagePart,
    ReasoningOutput,
    Request,
    Response,
    ServerToolUse,
    StructuredOutput,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    UsageStats,
)
from sieval.core.types import JSONValue

from ._google_genai_wire import validate_function_parameters
from ._google_interactions_wire import resolve_interactions_endpoint
from ._google_json import (
    reject_duplicate_keys,
    reject_json_constant,
    strict_json_loads,
)
from ._http_status import raise_for_provider_status

_THINKING_LEVELS = frozenset({"minimal", "low", "medium", "high"})
_OPAQUE_VERSION = 1
_SUMMARY_MODES = frozenset({"none", "auto"})
#: ``flex`` and ``deferred`` may answer with a ``queued`` interaction, which a
#: synchronous call cannot wait out.
_SERVICE_TIERS = frozenset({"standard", "priority"})
_TERMINAL_STATUSES = frozenset({"completed", "incomplete", "requires_action"})
_GENERATION_FIELDS = (
    ("temperature", "temperature"),
    ("top_p", "top_p"),
    ("max_tokens", "max_output_tokens"),
    ("stop", "stop_sequences"),
    ("seed", "seed"),
)
_IR_OWNED_KEYS = BINDING_RESOURCE_KEYS | frozenset(
    {
        "model",
        "input",
        "system_instruction",
        "generation_config",
        "tools",
        "response_format",
        "stream",
        "previous_interaction_id",
    }
)
_UNSUPPORTED_OPTION_KEYS = MappingProxyType(
    {
        "agent": "google_interactions is bound to a model, not an agent",
        "agent_config": "google_interactions is bound to a model, not an agent",
        "background": (
            "background interactions cannot complete in this synchronous "
            "execution contract"
        ),
    }
)
_REJECTED_PATHS = MappingProxyType(
    {
        "scoring.input_scoring": "Interactions cannot score input tokens",
        "scoring.sampled_logprobs": "Interactions returns no logprobs",
        "scoring.top_logprobs": "Interactions returns no logprobs",
        "sampling.n": "Interactions has no candidate count",
        "sampling.top_k": "Interactions generation_config has no top_k field",
        "sampling.frequency_penalty": (
            "Interactions generation_config has no frequency_penalty"
        ),
        "sampling.presence_penalty": (
            "Interactions generation_config has no presence_penalty"
        ),
        "reasoning.budget_tokens": (
            "Interactions has no reasoning token-budget field; use reasoning.effort"
        ),
        "tools.parallel": "Interactions has no parallel tool switch",
        "structured_output.name": "Interactions response schemas have no name field",
        "structured_output.strict": (
            "Interactions response schemas have no strict switch"
        ),
        "input.modality.image.detail": "Interactions images have no per-image detail",
    }
)
_FUNCTION_NAME_PATTERN = re.compile(r"[A-Za-z0-9_:.\-]{1,128}")
_HOSTED_TOOL_KINDS = frozenset({"google_search", "url_context"})
_IMAGE_MEDIA_TYPES = frozenset(
    {
        "image/bmp",
        "image/gif",
        "image/heic",
        "image/heif",
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/webp",
    }
)


def _validate_reasoning_config(options: object) -> None:
    assert isinstance(options, ReasoningOptions)
    if options.budget_tokens is not None:
        raise ValueError("google_interactions has no reasoning token-budget field")
    if options.summary not in {None, *_SUMMARY_MODES}:
        raise ValueError(
            "google_interactions supports only reasoning summary 'none' or 'auto'"
        )
    if options.effort is not None and options.effort.lower() not in _THINKING_LEVELS:
        raise ValueError(f"unsupported Google thinking level {options.effort!r}")


def _validate_hosted_tools_config(options: object) -> None:
    assert isinstance(options, HostedToolsOptions)
    unsupported = set(options.kinds) - _HOSTED_TOOL_KINDS
    if unsupported:
        raise ValueError(
            "google_interactions supports only google_search and url_context "
            f"hosted tools; unsupported kinds: {sorted(unsupported)}"
        )


def _validate_multimodal_config(options: object) -> None:
    assert isinstance(options, MultimodalInputOptions)
    unsupported = set(options.modalities) - {"image"}
    if unsupported:
        raise ValueError(
            f"google_interactions does not support modalities {unsupported!r}"
        )


def _tuple_validator(item_type: type, channel: str) -> Callable[[object], None]:
    def validate(value: object) -> None:
        if not isinstance(value, tuple) or not all(
            isinstance(item, item_type) for item in value
        ):
            raise OutputContractError(f"{channel} channel has invalid shape")

    return validate


def _decisions() -> Mapping[CapabilityKey, DialectCapabilityDecision]:
    decisions: dict[CapabilityKey, DialectCapabilityDecision] = {
        "input_scoring": Unsupported("Interactions cannot score input tokens"),
        "sampled_logprobs": Unsupported("Interactions returns no logprobs"),
        "top_logprobs": Unsupported("Interactions returns no logprobs"),
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
            "Interactions evaluation is stateless: "
            "store=false prevents prompt retention"
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
        "prefill": Unsupported("Interactions has no assistant-prefill field"),
        "fim": Unsupported("Interactions has no completion suffix field"),
    }
    if set(decisions) != set(CAPABILITY_KEYS):
        raise AssertionError("Google Interactions capability row is incomplete")
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
        "logprobs": OutputRule(Guarantee.NEVER),
        "top_logprobs": OutputRule(Guarantee.NEVER),
        "choice_logprobs": OutputRule(Guarantee.NEVER),
        "input_scoring": OutputRule(Guarantee.NEVER),
        "citations": OutputRule(
            Guarantee.BEST_EFFORT, _tuple_validator(Citation, "citations")
        ),
        "grounding": OutputRule(Guarantee.NEVER),
        "session_id": OutputRule(Guarantee.NEVER),
        "usage": OutputRule(Guarantee.BEST_EFFORT),
    },
    empty_outcome_finish_reasons=frozenset({"incomplete"}),
)


@dataclass(frozen=True)
class _InteractionsContext:
    request: Request


@dataclass(frozen=True)
class _ExecutionContext:
    """Everything response lifting needs, taken from the final wire request.

    ``declared_functions`` and ``hosted_kinds`` are what the reply is allowed to
    call.
    """

    request: Request
    request_params: Mapping[str, JSONValue]
    declared_functions: frozenset[str] = frozenset()
    hosted_kinds: frozenset[str] = frozenset()
    tool_choice: JSONValue = None
    input_steps: tuple[JSONValue, ...] = ()
    system_instruction: str | None = None


@dataclass(frozen=True)
class _LegacyPlan:
    dialect_id: str = "google_interactions"
    available_capabilities: frozenset[str] = frozenset(
        key
        for key, decision in CAPABILITY_DECISIONS.items()
        if isinstance(decision, Supported)
    )
    capability_minimums: Mapping[str, Mapping[str, JSONValue]] = field(
        default_factory=dict
    )
    required_output_channels: frozenset[str] = frozenset()


def _nested(raw: object, *keys: str) -> object:
    current = raw
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = cast(Mapping[str, object], current).get(key)
    return current


def _mapping(raw: object, path: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise OutputContractError(f"{path} must be an object")
    return cast(Mapping[str, object], raw)


def _array(raw: object, path: str) -> list[object]:
    if not isinstance(raw, list):
        raise OutputContractError(f"{path} must be an array")
    return cast(list[object], raw)


def _optional_string(raw: object, path: str) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        raise OutputContractError(f"{path} must be a non-empty string")
    return raw


def _count(
    usage: Mapping[str, object], name: str, *, default: int | None
) -> int | None:
    if name not in usage:
        return default
    value = usage[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OutputContractError(f"usage.{name} must be a non-negative integer")
    return value


def _usage(raw: object) -> UsageStats | None:
    """Lift Interactions' counts onto the shared accounting.

    ``total_output_tokens`` excludes thoughts, so they fold into
    ``output_tokens`` to keep ``reasoning_tokens`` a breakdown of it, as in
    ``google_genai``.
    """

    if raw is None:
        return None
    usage = _mapping(raw, "usage")
    input_tokens = cast(int, _count(usage, "total_input_tokens", default=0))
    output_tokens = cast(int, _count(usage, "total_output_tokens", default=0))
    tool_use = _count(usage, "total_tool_use_tokens", default=None)
    thoughts = _count(usage, "total_thought_tokens", default=None)
    reported = _count(usage, "total_tokens", default=None)
    input_total = input_tokens + (tool_use or 0)
    output_total = output_tokens + (thoughts or 0)
    total = input_total + output_total
    return UsageStats(
        input_tokens=input_total,
        output_tokens=output_total,
        total_tokens=total,
        reasoning_tokens=thoughts,
        cached_tokens=_count(usage, "total_cached_tokens", default=None),
        reported_total_tokens=None if reported in (None, total) else reported,
    )


def _function_source(raw: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    """Normalize one declaration to ``{name, description?, parameters?}``."""

    data = dict(raw)
    if data.get("type") == "function":
        nested = data.get("function")
        if nested is not None:
            if set(data) != {"type", "function"} or not isinstance(nested, Mapping):
                raise DialectError("invalid OpenAI-shaped function declaration")
            data = dict(nested)
        else:
            data.pop("type")
    unknown = set(data) - {"name", "description", "parameters"}
    if unknown:
        raise DialectError(
            "unsupported Interactions function declaration field(s): "
            + ", ".join(sorted(unknown))
        )
    name = data.get("name")
    if not isinstance(name, str) or _FUNCTION_NAME_PATTERN.fullmatch(name) is None:
        raise DialectError(
            "Interactions function name must be 1-128 characters of letters, "
            "digits, underscores, colons, dots, or dashes"
        )
    description = data.get("description")
    if not isinstance(description, str) or not description.strip():
        raise DialectError(
            "Interactions function declaration requires a non-empty description"
        )
    result: dict[str, JSONValue] = {"type": "function", "name": name}
    result["description"] = description
    if "parameters" in data:
        parameters = data["parameters"]
        if not isinstance(parameters, Mapping):
            raise DialectError("Interactions function parameters must be an object")
        validate_function_parameters(parameters)
        result["parameters"] = dict(parameters)
    return result


def _tool_choice_wire(raw: JSONValue) -> JSONValue:
    """Lower the IR tool choice onto ``generation_config.tool_choice``.

    ``required`` is the IR spelling of "must call some function"; the wire says
    ``any``.
    """

    if isinstance(raw, str):
        modes = {"auto": "auto", "none": "none", "required": "any"}
        try:
            return modes[raw]
        except KeyError as exc:
            raise DialectError(f"unsupported Interactions tool choice {raw!r}") from exc
    if not isinstance(raw, Mapping):
        raise DialectError("Interactions tool choice must be a string or mapping")
    data = dict(raw)
    nested = data.get("function")
    if set(data) == {"type", "function"} and isinstance(nested, Mapping):
        name = nested.get("name")
    elif set(data) == {"type", "name"} and data.get("type") == "function":
        name = data.get("name")
    else:
        raise DialectError("invalid Interactions function-specific tool choice")
    if not isinstance(name, str) or not name:
        raise DialectError("Interactions function-specific tool choice needs a name")
    # The documented way to narrow the callable set to one function is
    # ``allowed_tools`` with mode ``any``; ``validated`` only checks schema
    # adherence and would let a plain-text reply through.
    return {"allowed_tools": {"mode": "any", "tools": [name]}}


def _declared_function_names(req: Request) -> frozenset[str]:
    """Functions a reply may call, derived from the source declarations."""

    if not req.tools.functions:
        return frozenset()
    names = frozenset(
        cast(str, _function_source(item)["name"]) for item in req.tools.functions
    )
    choice = req.tools.choice
    if choice == "none":
        return frozenset()
    if isinstance(choice, Mapping):
        data = dict(choice)
        nested = data.get("function")
        name = (
            nested.get("name")
            if set(data) == {"type", "function"} and isinstance(nested, Mapping)
            else data.get("name")
        )
        return frozenset({cast(str, name)})
    return names


def _hosted_tool_config_error(kind: str, config: object) -> str | None:
    if not isinstance(config, Mapping):
        return f"Interactions hosted tool {kind!r} config must be an object"
    data = cast(Mapping[str, object], config)
    if kind == "url_context":
        return "Interactions url_context takes no config" if data else None
    if kind != "google_search":
        return f"unsupported Interactions hosted tool {kind!r}"
    unknown = set(data) - {"search_types"}
    if unknown:
        return "Interactions google_search config has unsupported field(s): " + (
            ", ".join(sorted(str(key) for key in unknown))
        )
    search_types = data.get("search_types")
    if search_types is None:
        return None
    if not isinstance(search_types, Sequence) or isinstance(search_types, str | bytes):
        return "Interactions google_search.search_types must be an array"
    allowed = {"web_search", "image_search", "enterprise_web_search"}
    if not search_types or any(
        not isinstance(item, str) or item not in allowed for item in search_types
    ):
        return (
            "Interactions google_search.search_types must be a non-empty array of "
            "web_search, image_search, or enterprise_web_search"
        )
    if len(set(cast(Sequence[str], search_types))) != len(search_types):
        return "Interactions google_search.search_types must not repeat a type"
    return None


def _hosted_tool(spec: object) -> dict[str, JSONValue]:
    kind = getattr(spec, "kind", None)
    config = getattr(spec, "config", None)
    if not isinstance(kind, str):
        raise DialectError(f"unsupported Interactions hosted tool {kind!r}")
    if error := _hosted_tool_config_error(kind, config):
        raise DialectError(error)
    assert config is not None
    tool: dict[str, JSONValue] = {"type": kind}
    tool.update(dict(config))
    return tool


def _image_content(part: ImagePart) -> dict[str, JSONValue]:
    if part.detail is not None:
        raise DialectError("Interactions images have no per-image detail")
    if part.media_type is not None and part.media_type not in _IMAGE_MEDIA_TYPES:
        raise DialectError(
            f"unsupported Interactions image media type {part.media_type!r}"
        )
    if part.url is not None:
        if not part.url:
            raise DialectError("Interactions image URLs must not be empty")
        content: dict[str, JSONValue] = {"type": "image", "uri": part.url}
        if part.media_type is not None:
            content["mime_type"] = part.media_type
        return content
    if not part.data or not part.media_type:
        raise DialectError(
            "Interactions inline images require data and an explicit MIME type"
        )
    return {"type": "image", "data": part.data, "mime_type": part.media_type}


_TEXT_CONTENT_KEYS = frozenset({"type", "text", "annotations"})
_IMAGE_CONTENT_KEYS = frozenset({"type", "data", "mime_type", "resolution", "uri"})
_RESOLUTIONS = frozenset({"low", "medium", "high", "ultra_high"})


def _tool_result_annotation_error(item: Mapping[str, object], path: str) -> str | None:
    """Validate input annotation fields without converting a response citation."""

    if "type" not in item:
        return f"{path}.type is required"
    kind = item["type"]
    if kind != "url_citation":
        return f"{path} has citation type {kind!r}; only url_citation is supported"
    for field_name in ("url", "title"):
        if field_name in item and not isinstance(item[field_name], str):
            return f"{path}.{field_name} must be a string"
    for field_name in ("start_index", "end_index"):
        if field_name not in item:
            continue
        value = item[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return f"{path}.{field_name} must be a non-negative integer"
    return None


def _tool_result_content_error(item: Mapping[str, object], path: str) -> str | None:
    """Why one array entry cannot be a ``function_result`` content block."""

    kind = item.get("type")
    if kind == "text":
        if set(item) - _TEXT_CONTENT_KEYS or not isinstance(item.get("text"), str):
            return f"{path} is not a valid text content block"
        if "annotations" not in item:
            return None
        annotations = item["annotations"]
        if not isinstance(annotations, list):
            return f"{path}.annotations must be an array"
        for index, raw in enumerate(annotations):
            annotation_path = f"{path}.annotations[{index}]"
            if not isinstance(raw, Mapping):
                return f"{annotation_path} must be an object"
            annotation = cast(Mapping[str, object], raw)
            if error := _tool_result_annotation_error(annotation, annotation_path):
                return error
        return None
    if kind == "image":
        if set(item) - _IMAGE_CONTENT_KEYS:
            return f"{path} is not a valid image content block"
        for field_name in ("data", "uri"):
            if field_name in item and not isinstance(item[field_name], str):
                return f"{path}.{field_name} must be a string"
        source = [field_name for field_name in ("data", "uri") if item.get(field_name)]
        if len(source) != 1:
            return f"{path} is not a valid image content block"
        for field_name, allowed in (
            ("mime_type", _IMAGE_MEDIA_TYPES),
            ("resolution", _RESOLUTIONS),
        ):
            if field_name not in item:
                continue
            value = item[field_name]
            if not isinstance(value, str):
                return f"{path}.{field_name} must be a string"
            if value not in allowed:
                return f"{path}.{field_name} must be one of {sorted(allowed)!r}"
        return None
    return f"{path} has content type {kind!r}; only text and image blocks are accepted"


def _tool_result_error(result: JSONValue) -> str | None:
    """Why ``result`` cannot be a ``function_result.result``, if it cannot.

    The wire accepts a string, an object, or an array of text/image content
    blocks; anything else is refused rather than stringified.
    """

    if isinstance(result, str | dict):
        return None
    if not isinstance(result, list):
        return (
            "Interactions function results must be a string, an object, or an "
            f"array of text/image content, not {type(result).__name__}; "
            "serialize the value (for example with json.dumps) first"
        )
    for index, raw in enumerate(result):
        path = f"function result[{index}]"
        if not isinstance(raw, dict):
            return f"{path} must be a text or image content object"
        error = _tool_result_content_error(cast(Mapping[str, object], raw), path)
        if error is not None:
            return error
    return None


def _function_result_step(part: ToolResultPart) -> dict[str, JSONValue]:
    if error := _tool_result_error(part.result):
        raise DialectError(error)
    step: dict[str, JSONValue] = {
        "type": "function_result",
        "call_id": part.call_id,
        "result": copy_json_value(part.result, "function result"),
    }
    if part.is_error:
        step["is_error"] = True
    return step


_OPAQUE_STEP_KEYS = {
    "user_input": frozenset({"type", "content"}),
    "model_output": frozenset({"type", "content", "error"}),
    "thought": frozenset({"type", "signature", "summary"}),
    "function_call": frozenset({"type", "id", "name", "arguments", "signature"}),
    "function_result": frozenset(
        {"type", "call_id", "name", "result", "is_error", "signature"}
    ),
    "google_search_call": frozenset(
        {"type", "id", "arguments", "search_type", "signature"}
    ),
    "google_search_result": frozenset(
        {"type", "call_id", "result", "is_error", "signature"}
    ),
    "url_context_call": frozenset({"type", "id", "arguments", "signature"}),
    "url_context_result": frozenset(
        {"type", "call_id", "result", "is_error", "signature"}
    ),
}


def _opaque_step_error(step: Mapping[str, JSONValue], path: str) -> str | None:
    kind = step.get("type")
    if not isinstance(kind, str) or kind not in _OPAQUE_STEP_KEYS:
        return f"{path}.type is unsupported"
    if set(step) - _OPAQUE_STEP_KEYS[kind]:
        return f"{path} has unsupported fields"
    if "signature" in step and not isinstance(step["signature"], str):
        return f"{path}.signature must be a string"
    if kind in {"user_input", "model_output", "thought"}:
        if step.get("error") is not None:
            return f"{path} carries an error"
        field_name = "summary" if kind == "thought" else "content"
        if field_name not in step:
            return None
        content = step[field_name]
        if not isinstance(content, list):
            return f"{path}.{field_name} must be an array"
        for index, item in enumerate(content):
            item_path = f"{path}.{field_name}[{index}]"
            if not isinstance(item, dict):
                return f"{item_path} must be an object"
            if kind == "thought" and item.get("type") != "text":
                return f"{item_path} must be text"
            if error := _tool_result_content_error(item, item_path):
                return error
        return None
    identifier = "id" if kind.endswith("_call") else "call_id"
    if not isinstance(step.get(identifier), str) or not step[identifier]:
        return f"{path}.{identifier} must be a non-empty string"
    if (kind == "function_call" or "name" in step) and (
        not isinstance(step.get("name"), str) or not step["name"]
    ):
        return f"{path}.name must be a non-empty string"
    if kind.endswith("_call"):
        if not isinstance(step.get("arguments"), dict):
            return f"{path}.arguments must be an object"
        if "search_type" in step and step["search_type"] not in (
            "web_search",
            "image_search",
            "enterprise_web_search",
        ):
            return f"{path}.search_type is unsupported"
    else:
        if "result" not in step:
            return f"{path}.result is required"
        if "is_error" in step and not isinstance(step["is_error"], bool):
            return f"{path}.is_error must be a boolean"
        if kind == "function_result":
            if error := _tool_result_error(step["result"]):
                return f"{path}: {error}"
        elif not step.get("is_error", False) and not isinstance(step["result"], list):
            return f"{path}.result must be an array"
    return None


def _decode_opaque(value: str) -> tuple[str | None, tuple[JSONValue, ...]]:
    try:
        raw = strict_json_loads(value, "opaque continuation JSON")
    except OutputContractError as exc:
        raise DialectError(str(exc)) from exc
    if (
        not isinstance(raw, dict)
        or not {"version", "steps"}.issubset(raw)
        or set(raw) - {"version", "steps", "system_instruction"}
    ):
        raise DialectError("invalid Interactions opaque continuation envelope")
    if type(raw["version"]) is not int or raw["version"] != _OPAQUE_VERSION:
        raise DialectError("unsupported Interactions opaque continuation version")
    system = raw.get("system_instruction")
    if "system_instruction" in raw and not isinstance(system, str):
        raise DialectError("opaque system_instruction must be a string")
    steps = raw["steps"]
    if not isinstance(steps, list):
        raise DialectError("opaque steps must be an array")
    calls: dict[str, str] = {}
    call_names: dict[str, str] = {}
    answered: set[str] = set()
    for index, step in enumerate(steps):
        path = f"opaque.steps[{index}]"
        if not isinstance(step, dict):
            raise DialectError(f"{path} must be an object")
        if error := _opaque_step_error(step, path):
            raise DialectError(error)
        kind = cast(str, step["type"])
        if kind.endswith("_call"):
            call_id = cast(str, step["id"])
            if call_id in calls:
                raise DialectError(f"{path} repeats opaque call id {call_id!r}")
            calls[call_id] = kind
            if kind == "function_call":
                call_names[call_id] = cast(str, step["name"])
        elif kind.endswith("_result"):
            call_id = cast(str, step["call_id"])
            if calls.get(call_id) != kind.removesuffix("_result") + "_call":
                raise DialectError(f"{path} answers no matching earlier opaque call")
            if call_id in answered:
                raise DialectError(f"{path} answers an opaque call more than once")
            if (
                kind == "function_result"
                and "name" in step
                and step["name"] != call_names[call_id]
            ):
                raise DialectError(f"{path}.name differs from the earlier opaque call")
            answered.add(call_id)
    return cast(str | None, system), tuple(steps)


def _encode_opaque(system: str | None, steps: Sequence[JSONValue]) -> str:
    envelope: dict[str, JSONValue] = {
        "version": _OPAQUE_VERSION,
        "steps": copy_json_value(list(steps), "opaque history"),
    }
    if system is not None:
        envelope["system_instruction"] = system
    return json.dumps(
        envelope,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _resolve_opaque_system(current: str | None, prior: str | None) -> str | None:
    if prior is not None and current is not None and prior != current:
        raise DialectError(
            "opaque system_instruction conflicts with current system message"
        )
    return current if current is not None else prior


def _history_function_names(history: Sequence[JSONValue]) -> dict[str, str]:
    names: dict[str, str] = {}
    for raw in history:
        step = cast(dict[str, JSONValue], raw)
        if step["type"] == "function_call":
            names[cast(str, step["id"])] = cast(str, step["name"])
    return names


def _orphan_result_error(
    chat: ChatInput, history: Sequence[JSONValue] = ()
) -> str | None:
    """A tool result must answer a call in validated history or current input."""

    calls: set[str] = set()
    answered: set[str] = set()
    historical_ids: set[str] = set()
    for raw in history:
        step = cast(dict[str, JSONValue], raw)
        if cast(str, step["type"]).endswith("_call"):
            historical_ids.add(cast(str, step["id"]))
        if step["type"] == "function_call":
            calls.add(cast(str, step["id"]))
        elif step["type"] == "function_result":
            answered.add(cast(str, step["call_id"]))
    for message in chat.messages:
        for part in message.content:
            if isinstance(part, ToolCallPart):
                if history and part.call_id in historical_ids | calls:
                    return (
                        f"tool call {part.call_id!r} repeats a call in opaque "
                        "history or current input"
                    )
                calls.add(part.call_id)
            elif isinstance(part, ToolResultPart):
                if part.call_id not in calls:
                    return (
                        f"tool result {part.call_id!r} answers no earlier tool call "
                        "in this input; replay the call before its result"
                    )
                if part.call_id in answered:
                    return f"tool call {part.call_id!r} is answered more than once"
                answered.add(part.call_id)
    return None


def _chat_shape_error(chat: ChatInput, history: Sequence[JSONValue] = ()) -> str | None:
    """Return a lossless-mapping error for chat history, if any.

    Part kinds have their own request leaves; nothing here rejects a part.
    """

    system_messages = 0
    saw_content = False
    for index, message in enumerate(chat.messages):
        if message.role == "developer":
            return (
                "Interactions system_instruction cannot preserve the distinct "
                "developer message authority"
            )
        if message.role == "tool":
            saw_content = True
            for part in message.content:
                if isinstance(part, ToolResultPart) and (
                    error := _tool_result_error(part.result)
                ):
                    return error
            continue
        if message.role == "system":
            if saw_content:
                return "Interactions system messages must precede conversation turns"
            system_messages += 1
            if (
                system_messages > 1
                or len(message.content) != 1
                or not isinstance(message.content[0], TextPart)
            ):
                return (
                    "Interactions system_instruction is one text string, so it "
                    "needs exactly one system message with one text part"
                )
            continue
        saw_content = True
        if not message.content:
            return f"message {index} has no content to lower"
    if not saw_content:
        return "Interactions input needs at least one conversation turn"
    # A tool result ends a valid input: it is the documented way to hand a
    # function's output back to the model.  Only a trailing assistant turn is a
    # prefill the API has no field for.
    if chat.messages[-1].role not in ("user", "tool"):
        return "google_interactions does not support assistant-prefill input"
    return _orphan_result_error(chat, history)


def _part_to_content(part: object) -> dict[str, JSONValue]:
    if isinstance(part, TextPart):
        if not isinstance(part.text, str):
            raise DialectError("Interactions text parts must carry a string")
        return {"type": "text", "text": part.text}
    if isinstance(part, ImagePart):
        return _image_content(part)
    raise DialectError(f"google_interactions cannot lower {type(part).__name__}")


def _message_to_steps(message: ChatMessage) -> list[JSONValue]:
    """Lower one IR message to wire steps, preserving part order.

    ``function_call`` is a step of its own on the wire, never a content block of
    ``model_output``, so an assistant message becomes a run of ``model_output``
    steps (text/image) broken wherever a ``ToolCallPart`` sits.
    """

    if message.role == "tool":
        results = [part for part in message.content if isinstance(part, ToolResultPart)]
        if len(results) != 1 or len(message.content) != 1:
            raise DialectError(
                "a tool message must carry exactly one ToolResultPart; the IR has "
                "no way to group several results into one function_result step"
            )
        return [_function_result_step(results[0])]
    role = "model_output" if message.role == "assistant" else "user_input"
    steps: list[JSONValue] = []
    content: list[JSONValue] = []
    for part in message.content:
        if isinstance(part, ToolCallPart):
            if message.role != "assistant":
                raise DialectError("only an assistant message can carry a ToolCallPart")
            if (
                not part.call_id
                or not part.name
                or not isinstance(part.arguments, Mapping)
            ):
                raise DialectError(
                    "Interactions function calls need a non-empty id and name and "
                    "an object of arguments"
                )
            if content:
                steps.append({"type": role, "content": content})
                content = []
            steps.append(
                {
                    "type": "function_call",
                    "id": part.call_id,
                    "name": part.name,
                    "arguments": copy_json_value(
                        part.arguments, "function call arguments"
                    ),
                }
            )
            continue
        content.append(_part_to_content(part))
    if content:
        steps.append({"type": role, "content": content})
    if not steps:
        raise DialectError("Interactions steps cannot be empty")
    return steps


def _input_to_wire(
    chat: ChatInput, opaque: str | None = None
) -> tuple[str | None, list[JSONValue]]:
    prior_system, history = _decode_opaque(opaque) if opaque is not None else (None, ())
    if error := _chat_shape_error(chat, history):
        raise DialectError(error)
    system: str | None = None
    steps: list[JSONValue] = list(history)
    for message in chat.messages:
        if message.name is not None:
            raise DialectError("Interactions steps have no ChatMessage.name field")
        if message.role == "system":
            system = cast(TextPart, message.content[0]).text
            continue
        steps.extend(_message_to_steps(message))
    call_names = _history_function_names(history)
    for raw in steps[len(history) :]:
        step = cast(dict[str, JSONValue], raw)
        if step["type"] == "function_call":
            call_names[cast(str, step["id"])] = cast(str, step["name"])
        elif step["type"] == "function_result":
            step["name"] = call_names[cast(str, step["call_id"])]
    return _resolve_opaque_system(system, prior_system), steps


@dataclass(frozen=True)
class _ExpectedContent:
    """One text or image block of a ``user_input``/``model_output`` step."""

    kind: str
    text: str | None = None
    data: str | None = None
    uri: str | None = None
    mime_type: str | None = None


@dataclass(frozen=True)
class _ExpectedStep:
    """One input step, decoded by shape rather than compared as JSON.

    ``function_call`` and ``function_result`` are steps of their own;
    ``payload`` is the call's arguments or the result's value.
    """

    kind: str
    content: tuple[_ExpectedContent, ...] = ()
    call_id: str | None = None
    call_name: str | None = None
    payload: JSONValue = None
    is_error: bool = False


_STEP_KEYS = MappingProxyType(
    {
        "user_input": frozenset({"type", "content"}),
        "model_output": frozenset({"type", "content"}),
        "function_call": frozenset({"type", "id", "name", "arguments"}),
        "function_result": frozenset({"type", "call_id", "result", "is_error"}),
    }
)
_CONTENT_KEYS = MappingProxyType(
    {
        "text": frozenset({"type", "text"}),
        "image": frozenset({"type", "data", "uri", "mime_type"}),
    }
)


@dataclass(frozen=True)
class _InputVerifier:
    """Decode the wire input and compare it with the provider-neutral source.

    The expected steps are derived from the IR on their own, not from the
    lowering's output, so a lowering bug cannot vouch for itself.
    """

    system: str | None
    steps: tuple[_ExpectedStep, ...]
    history: tuple[JSONValue, ...] = ()

    @classmethod
    def from_request(
        cls, chat: ChatInput, opaque: str | None = None
    ) -> "_InputVerifier":
        prior_system, history = (
            _decode_opaque(opaque) if opaque is not None else (None, ())
        )
        call_names = _history_function_names(history)
        system: str | None = None
        steps: list[_ExpectedStep] = []
        for message in chat.messages:
            if message.role == "system":
                system = cast(TextPart, message.content[0]).text
                continue
            if message.role == "tool":
                result = cast(ToolResultPart, message.content[0])
                steps.append(
                    _ExpectedStep(
                        "function_result",
                        call_id=result.call_id,
                        call_name=call_names.get(result.call_id),
                        payload=result.result,
                        is_error=result.is_error,
                    )
                )
                continue
            kind = "model_output" if message.role == "assistant" else "user_input"
            content: list[_ExpectedContent] = []
            for part in message.content:
                if isinstance(part, ToolCallPart):
                    call_names[part.call_id] = part.name
                    if content:
                        steps.append(_ExpectedStep(kind, tuple(content)))
                        content = []
                    steps.append(
                        _ExpectedStep(
                            "function_call",
                            call_id=part.call_id,
                            call_name=part.name,
                            payload=part.arguments,
                        )
                    )
                elif isinstance(part, TextPart):
                    content.append(_ExpectedContent("text", text=part.text))
                elif isinstance(part, ImagePart):
                    expected = _image_content(part)
                    content.append(
                        _ExpectedContent(
                            "image",
                            data=expected.get("data"),  # ty: ignore[invalid-argument-type]
                            uri=expected.get("uri"),  # ty: ignore[invalid-argument-type]
                            mime_type=expected.get("mime_type"),  # ty: ignore[invalid-argument-type]
                        )
                    )
            if content:
                steps.append(_ExpectedStep(kind, tuple(content)))
        return cls(_resolve_opaque_system(system, prior_system), tuple(steps), history)

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        raw = body.get("input")
        if not isinstance(raw, list):
            return "input is not a step array"
        if len(raw) != len(self.history) + len(self.steps):
            return "input steps no longer match the provider-neutral input"
        if _encode_opaque(
            self.system, cast(list[JSONValue], raw[: len(self.history)])
        ) != _encode_opaque(self.system, self.history):
            return "input opaque history no longer matches the continuation"
        for index, (raw_step, expected) in enumerate(
            zip(raw[len(self.history) :], self.steps, strict=True),
            start=len(self.history),
        ):
            problem = _compare_step(raw_step, expected, f"input[{index}]")
            if problem is not None:
                return problem
        if body.get("system_instruction") != self.system:
            return "system_instruction no longer matches the system message"
        return None


def _compare_step(raw: object, expected: _ExpectedStep, path: str) -> str | None:
    if not isinstance(raw, Mapping):
        return f"{path} has an unexpected shape"
    step = cast(Mapping[str, object], raw)
    allowed = _STEP_KEYS[expected.kind]
    if expected.kind == "function_result" and expected.call_name is not None:
        allowed = allowed | {"name"}
    if step.get("type") != expected.kind or set(step) - allowed:
        return f"{path} has an unexpected shape"
    if expected.kind == "function_result":
        if step.get("call_id") != expected.call_id:
            return f"{path} answers a different call"
        if step.get("result") != expected.payload:
            return f"{path} result no longer matches the tool result"
        if expected.call_name is not None and step.get("name") != expected.call_name:
            return f"{path} function result name no longer matches the earlier call"
        flag = step.get("is_error")
        if expected.is_error and flag is not True:
            return f"{path} lost its is_error flag"
        if not expected.is_error and flag is not None and flag is not False:
            return f"{path} carries an unexpected is_error"
        return None
    if expected.kind == "function_call":
        if (
            step.get("id") != expected.call_id
            or step.get("name") != expected.call_name
            or step.get("arguments") != expected.payload
        ):
            return f"{path} function call no longer matches the provider-neutral input"
        return None
    content = step.get("content")
    if not isinstance(content, list) or len(content) != len(expected.content):
        return f"{path} content no longer matches the provider-neutral input"
    for item_index, (raw_item, want) in enumerate(
        zip(content, expected.content, strict=True)
    ):
        problem = _compare_content(raw_item, want, f"{path}.content[{item_index}]")
        if problem is not None:
            return problem
    return None


def _compare_content(raw: object, want: _ExpectedContent, path: str) -> str | None:
    if not isinstance(raw, Mapping):
        return f"{path} is not a content block"
    item = cast(Mapping[str, object], raw)
    if item.get("type") != want.kind:
        return f"{path} changed type"
    if set(item) - _CONTENT_KEYS[want.kind]:
        return f"{path} has unexpected field(s)"
    for field_name, value in (
        ("text", want.text),
        ("data", want.data),
        ("uri", want.uri),
        ("mime_type", want.mime_type),
    ):
        if value is not None and item.get(field_name) != value:
            return f"{path} {field_name} no longer matches the provider-neutral input"
    return None


@dataclass(frozen=True)
class _RouteVerifier:
    """Verify the route, bound model and stateless retention controls.

    Model, stream and retention defaults come from no request leaf, so their
    evidence must include the final prepared and serialized request.
    """

    model: str
    api_version: str

    def verify_body(self, body: Mapping[str, JSONValue]) -> str | None:
        if body.get("model") != self.model:
            return "model does not match the bound model"
        if body.get("stream") not in (None, True):
            return "stream must be true or absent"
        if body.get("store") is not False:
            return "store must be false to prevent evaluation prompt retention"
        if "previous_interaction_id" in body:
            return "previous_interaction_id is unavailable for stateless evaluation"
        if "background" in body:
            return "background execution is unavailable for synchronous evaluation"
        return None

    def verify_prepared(self, prepared: PreparedRequest) -> str | None:
        expected = f"{self.api_version}/interactions"
        if prepared.operation != expected:
            return f"operation {prepared.operation!r} does not match {expected!r}"
        return self.verify_body(prepared.body)

    def verify_http(self, request: httpx.Request, endpoint_root: httpx.URL) -> None:
        expected = endpoint_root.join(f"{self.api_version}/interactions")
        if request.method != "POST" or request.url != expected:
            raise DialectError(
                "final Interactions request URL or method does not match the "
                f"bound {self.api_version} operation"
            )


@dataclass(frozen=True)
class _ThinkingLevelVerifier:
    effort: str

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        actual = _nested(body, "generation_config", "thinking_level")
        return None if actual == self.effort.lower() else "thinking level changed"


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
        if isinstance(self.value, bool):
            return (
                None
                if self.key in body and body[self.key] is self.value
                else f"Google dialect option {self.key!r} changed"
            )
        return (
            None
            if self.key in body and body[self.key] == self.value
            else f"Google dialect option {self.key!r} changed"
        )


@dataclass(frozen=True)
class _FunctionsVerifier:
    """Rebuild the declarations from source and compare with the wire tools."""

    functions: tuple[Mapping[str, JSONValue], ...]

    def __post_init__(self) -> None:
        copied: list[Mapping[str, JSONValue]] = []
        for index, function in enumerate(self.functions):
            item = copy_json_value(function, f"Google function {index}")
            if not isinstance(item, dict):
                raise TypeError("Interactions function declarations must be objects")
            copied.append(item)
        object.__setattr__(self, "functions", tuple(copied))

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        expected: list[JSONValue] = []
        for function in self.functions:
            source = dict(function)
            nested = source.get("function")
            if source.get("type") == "function" and isinstance(nested, dict):
                source = dict(nested)
            else:
                source.pop("type", None)
            declaration: dict[str, JSONValue] = {
                "type": "function",
                "name": source.get("name"),
                "description": source.get("description"),
            }
            if "parameters" in source:
                declaration["parameters"] = source["parameters"]
            expected.append(declaration)
        raw_tools = body.get("tools", [])
        if not isinstance(raw_tools, Sequence) or isinstance(raw_tools, str | bytes):
            return "function tools changed"
        actual = [
            item
            for item in cast(Sequence[Mapping[str, object]], raw_tools)
            if item.get("type") == "function"
        ]
        # ``body`` arrives frozen; thaw both sides before serializing.
        try:
            actual_json = json.dumps(
                copy_json_value(actual, "verified Interactions function tools"),
                sort_keys=True,
            )
            expected_json = json.dumps(
                copy_json_value(expected, "requested Interactions function tools"),
                sort_keys=True,
            )
        except (TypeError, ValueError):
            return "function tools changed"
        return None if actual_json == expected_json else "function tools changed"


@dataclass(frozen=True)
class _HostedToolsVerifier:
    hosted: tuple[object, ...]

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        expected: list[JSONValue] = []
        for item in self.hosted:
            kind = getattr(item, "kind", None)
            config = getattr(item, "config", None)
            if not isinstance(kind, str) or not isinstance(config, Mapping):
                return "hosted tool config is invalid"
            tool: dict[str, JSONValue] = {"type": kind}
            tool.update(dict(config))
            expected.append(tool)
        raw_tools = body.get("tools")
        if not isinstance(raw_tools, Sequence) or isinstance(raw_tools, str | bytes):
            return "hosted tools changed"
        actual = [
            item
            for item in cast(Sequence[Mapping[str, object]], raw_tools)
            if item.get("type") != "function"
        ]
        try:
            matches = json.dumps(
                copy_json_value(actual, "verified Interactions hosted tools"),
                sort_keys=True,
            ) == json.dumps(
                copy_json_value(expected, "requested Interactions hosted tools"),
                sort_keys=True,
            )
        except (TypeError, ValueError):
            matches = False
        return None if matches else "hosted tools changed"


@dataclass(frozen=True)
class _ToolChoiceVerifier:
    """Rebuild the wire form from the IR value and compare."""

    choice: JSONValue

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        expected = _tool_choice_wire(self.choice)
        actual = _nested(body, "generation_config", "tool_choice")
        return None if actual == expected else "tool choice changed"


@dataclass(frozen=True)
class _StructuredFormatVerifier:
    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        response_format = _nested(body, "response_format")
        if not isinstance(response_format, Mapping):
            return "structured output response_format changed"
        mapping = cast(Mapping[str, object], response_format)
        if mapping.get("type") != "text":
            return "structured output type changed"
        if mapping.get("mime_type") != "application/json":
            return "structured output MIME type changed"
        return None


@dataclass(frozen=True)
class _StreamVerifier:
    model: str

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        if body.get("stream") is not True:
            return "stream flag changed"
        if body.get("model") != self.model:
            return "model does not match the bound model"
        return None


class GoogleInteractionsDialect:
    """Executable adapter for the Gemini native Interactions REST endpoint."""

    dialect_id = "google_interactions"
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
            Capability.StructuredOutput,
        }
    )

    def __init__(self, connection: object, requested_model_id: str):
        if not isinstance(connection, AsyncHTTPJSONConnection):
            raise TypeError("google_interactions requires AsyncHTTPJSONConnection")
        if not requested_model_id:
            raise ValueError("requested_model_id must not be empty")
        self._connection = connection
        self._requested_model_id = requested_model_id
        self._version, self._endpoint_root = resolve_interactions_endpoint(
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
        opaque_error: str | None = None
        history: tuple[JSONValue, ...] = ()
        if req.session.opaque_continuation is not None:
            try:
                _, history = _decode_opaque(req.session.opaque_continuation.value)
            except DialectError as exc:
                opaque_error = str(exc)
        chat_shape_error = (
            _chat_shape_error(req.input, history)
            if isinstance(req.input, ChatInput)
            else None
        )
        options = req.dialect_options
        invalid_option_paths = (
            {
                f"dialect_options.{key}"
                for key in options.values
                if not isinstance(key, str)
            }
            if options is not None
            else set()
        )
        for path in audit.active_paths:
            if path in _REJECTED_PATHS:
                audit.rejected(path, _REJECTED_PATHS[path])
            elif path == "input.chat":
                if chat_shape_error is not None:
                    audit.rejected(path, chat_shape_error)
            elif path == "session.opaque_continuation" and opaque_error is not None:
                audit.rejected(path, opaque_error)
            elif path == "input.chat.message.name":
                audit.rejected(
                    path, "Interactions steps have no ChatMessage.name field"
                )
            elif path.startswith("input.completion"):
                audit.rejected(path, "google_interactions requires ChatInput")
            elif (
                path == "sampling.max_tokens" and cast(int, req.sampling.max_tokens) < 1
            ):
                audit.rejected(path, "Google max_tokens must be positive")
            elif path == "reasoning.effort" and (
                not isinstance(req.reasoning.effort, str)
                or req.reasoning.effort.lower() not in _THINKING_LEVELS
            ):
                audit.rejected(path, "unsupported Google thinking level")
            elif path == "reasoning.summary" and (
                req.reasoning.summary not in _SUMMARY_MODES
            ):
                audit.rejected(path, "Google supports only summary='none' or 'auto'")
            elif path == "session.previous_response_id":
                if not req.session.previous_response_id:
                    audit.rejected(path, "previous_interaction_id must not be empty")
                else:
                    audit.rejected(
                        path,
                        "store=false is stateless and cannot return a reusable "
                        "session_id; resend the full conversation history",
                    )
            elif path == "tools.functions":
                try:
                    for function in req.tools.functions:
                        _function_source(function)
                except DialectError as exc:
                    audit.rejected(path, str(exc))
            elif path == "tools.choice":
                if not req.tools.functions:
                    audit.rejected(
                        path, "Interactions tool choice requires function tools"
                    )
                else:
                    try:
                        _tool_choice_wire(req.tools.choice)
                    except DialectError as exc:
                        audit.rejected(path, str(exc))
            elif path == "tools.hosted":
                for tool in req.tools.hosted:
                    error = _hosted_tool_config_error(tool.kind, tool.config)
                    if error is not None:
                        audit.rejected(path, error)
                        break
            elif path == "structured_output.format" and (
                req.structured_output.format != "json_schema"
            ):
                audit.rejected(
                    path,
                    "Interactions response_format only carries a JSON schema here",
                )
            elif path.startswith("dialect_options."):
                self._validate_option(path, req, audit, invalid_option_paths)

    @staticmethod
    def _validate_option(
        path: str, req: Request, audit: RequestAudit, invalid: set[str]
    ) -> None:
        key = path.removeprefix("dialect_options.")
        if path in invalid:
            audit.rejected(path, "Google dialect option keys must be strings")
            return
        if not key:
            audit.rejected(path, "dialect option keys must be non-empty")
            return
        if key in _IR_OWNED_KEYS:
            audit.rejected(path, f"{key!r} must use its provider-neutral field")
            return
        if key in _UNSUPPORTED_OPTION_KEYS:
            audit.rejected(path, _UNSUPPORTED_OPTION_KEYS[key])
            return
        value = (
            cast(object, req.dialect_options.values[key])
            if req.dialect_options
            else None
        )
        if key == "store" and value is not False:
            audit.rejected(
                path,
                "Interactions store must be a boolean false "
                "to prevent prompt retention",
            )
        elif key == "service_tier" and not (
            isinstance(value, str) and value in _SERVICE_TIERS
        ):
            audit.rejected(
                path,
                "service_tier must be 'standard' or 'priority'; other tiers can "
                "return a queued interaction",
            )

    def prepare(self, req: Request, audit: RequestAudit) -> PreparedRequest:
        audit.require_matches_request(req)
        audit.raise_rejections()
        if not isinstance(req.input, ChatInput):
            raise DialectError("google_interactions requires ChatInput")
        opaque = (
            req.session.opaque_continuation.value
            if req.session.opaque_continuation is not None
            else None
        )
        system, steps = _input_to_wire(req.input, opaque)
        body: dict[str, JSONValue] = {
            "model": self._requested_model_id,
            "input": steps,
            "store": False,
        }
        if system is not None:
            body["system_instruction"] = system
        input_verifier = _InputVerifier.from_request(req.input, opaque)
        route = _RouteVerifier(self._requested_model_id, self._version)
        # Only leaves this concrete request activates may receive a decision;
        # an inactive path is a hard audit error.
        active = frozenset(audit.active_paths)

        def consume(path: str, *evidence: WireEvidence) -> None:
            if path in active and path not in audit.decisions:
                audit.consumed(path, *evidence)

        consume(
            "input.chat",
            input_verifier,
            PreparedRequestEvidence(route),
        )
        for leaf in (
            "input.modality.text",
            "input.modality.image",
            "input.modality.image.media_type",
            "input.modality.tool_call",
            "input.modality.tool_result",
            "input.modality.tool_result.is_error",
        ):
            consume(leaf, input_verifier)
        if opaque is not None:
            consume("session.opaque_continuation", input_verifier)

        generation: dict[str, JSONValue] = {}
        for source, destination in _GENERATION_FIELDS:
            value = getattr(req.sampling, source)
            if value is None:
                continue
            generation[destination] = list(value) if isinstance(value, tuple) else value
            audit.consumed(
                f"sampling.{source}",
                WireObservation(("generation_config", destination)),
            )
        if req.reasoning.effort is not None:
            generation["thinking_level"] = req.reasoning.effort.lower()
            audit.consumed(
                "reasoning.effort", _ThinkingLevelVerifier(req.reasoning.effort)
            )
        if req.reasoning.summary is not None:
            generation["thinking_summaries"] = req.reasoning.summary
            audit.consumed(
                "reasoning.summary",
                WireObservation(("generation_config", "thinking_summaries")),
            )

        tools: list[JSONValue] = []
        if req.tools.functions:
            tools.extend(_function_source(item) for item in req.tools.functions)
            audit.consumed("tools.functions", _FunctionsVerifier(req.tools.functions))
        if req.tools.hosted:
            tools.extend(_hosted_tool(tool) for tool in req.tools.hosted)
            audit.consumed(
                "tools.hosted",
                _HostedToolsVerifier(req.tools.hosted),
            )
        if tools:
            body["tools"] = tools
        if req.tools.choice is not None:
            generation["tool_choice"] = _tool_choice_wire(req.tools.choice)
            audit.consumed("tools.choice", _ToolChoiceVerifier(req.tools.choice))

        if req.structured_output.format is not None:
            schema = req.structured_output.schema
            if schema is None:
                raise DialectError("Interactions response_format requires a schema")
            body["response_format"] = {
                "type": "text",
                "mime_type": "application/json",
                "schema": dict(schema),
            }
            audit.consumed("structured_output.format", _StructuredFormatVerifier())
            audit.consumed(
                "structured_output.schema",
                WireObservation(("response_format", "schema")),
            )

        if generation:
            body["generation_config"] = generation

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
                body[key] = value
                audit.consumed(path, _RootOptionVerifier(key, value))

        if req.scheduling.stream:
            body["stream"] = True
            audit.consumed(
                "scheduling.stream", _StreamVerifier(self._requested_model_id)
            )

        return PreparedRequest(
            f"{self._version}/interactions",
            body,
            _InteractionsContext(deepcopy(req)),
        )

    def _lift(self, raw: object, context: _ExecutionContext) -> Response:
        req = context.request
        reply = _mapping(raw, "response")
        kind = reply.get("object")
        if kind is not None and kind != "interaction":
            raise OutputContractError(f"response.object {kind!r} is not 'interaction'")
        status = reply.get("status")
        if not isinstance(status, str) or not status:
            raise OutputContractError("response.status must be a non-empty string")
        if status == "requires_action" and not req.tools.functions:
            raise OutputContractError(_status_error(status, reply))
        if status not in _TERMINAL_STATUSES:
            raise OutputContractError(_status_error(status, reply))

        lifted = _LiftState(
            request=req,
            declared_functions=context.declared_functions,
            hosted_kinds=context.hosted_kinds,
            tool_choice=context.tool_choice,
        )
        raw_steps = reply.get("steps")
        for index, raw_step in enumerate(
            [] if raw_steps is None else _array(raw_steps, "response.steps")
        ):
            lifted.absorb(_mapping(raw_step, f"response.steps[{index}]"), index)
        return lifted.finish(status, reply, context)

    async def execute(self, prepared: PreparedRequest) -> Response:
        context = prepared.context
        if not isinstance(context, _InteractionsContext):
            raise DialectError("google_interactions received invalid prepared context")
        route = _RouteVerifier(self._requested_model_id, self._version)
        mismatch = route.verify_prepared(prepared)
        if mismatch is not None:
            raise DialectError(
                f"google_interactions received the wrong prepared operation: {mismatch}"
            )
        headers = {"Content-Type": "application/json"}
        credential = self._connection.credential
        if credential is not None:
            headers["x-goog-api-key"] = credential
        request = self._connection.client.build_request(
            "POST",
            self._endpoint_root.join(prepared.operation),
            headers=headers,
            json=prepared.thaw_body(),
        )
        route.verify_http(request, self._endpoint_root)
        actual_body = strict_json_loads(
            request.content.decode("utf-8"), "final Interactions request body"
        )
        if not isinstance(actual_body, dict):
            raise DialectError("final Interactions request body must be an object")
        mismatch = route.verify_body(actual_body)
        if mismatch is not None:
            raise DialectError(f"final Interactions request: {mismatch}")
        stream = actual_body.get("stream", False)
        if not isinstance(stream, bool):
            raise DialectError("final Interactions request stream must be a boolean")
        opaque = context.request.session.opaque_continuation
        mismatch = _InputVerifier.from_request(
            cast(ChatInput, context.request.input),
            opaque.value if opaque is not None else None,
        ).verify(actual_body)
        if mismatch is not None:
            raise DialectError(f"final Interactions request: {mismatch}")
        execution = _ExecutionContext(
            request=context.request,
            request_params={
                REQUEST_ENVELOPE_KEY: REQUEST_ENVELOPE_HTTP,
                "method": request.method,
                "path": request.url.path,
                "body": {
                    key: value
                    for key, value in actual_body.items()
                    if key not in {"model", "input", "system_instruction"}
                },
            },
            declared_functions=_declared_function_names(context.request),
            hosted_kinds=frozenset(tool.kind for tool in context.request.tools.hosted),
            tool_choice=context.request.tools.choice,
            input_steps=tuple(cast(list[JSONValue], actual_body["input"])),
            system_instruction=cast(str | None, actual_body.get("system_instruction")),
        )
        if not stream:
            response = await self._connection.client.send(request)
            raise_for_provider_status(response, "Google")
            payload = strict_json_loads(response.text, "Interactions response body")
            return self._lift(payload, execution)
        response = await self._connection.client.send(request, stream=True)
        try:
            if not response.is_success:
                await response.aread()
            raise_for_provider_status(response, "Google")
            return await self._lift_stream(response, execution)
        finally:
            await response.aclose()

    async def _lift_stream(
        self, response: httpx.Response, context: _ExecutionContext
    ) -> Response:
        return _merge_stream(await _read_stream(response), context)

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


def _status_error(status: str, reply: Mapping[str, object]) -> str:
    if status == "requires_action":
        return (
            "Interactions reply requires an action, but no function tools were declared"
        )
    if status == "failed":
        errors = reply.get("errors")
        detail = ""
        if isinstance(errors, list) and errors and isinstance(errors[0], Mapping):
            message = cast(Mapping[str, object], errors[0]).get("message")
            detail = f": {message}" if isinstance(message, str) else ""
        return f"Interactions reply failed{detail}"
    if status == "budget_exceeded":
        return (
            "Interactions reply stopped at its budget (status 'budget_exceeded'); "
            "no partial result is reported"
        )
    if status in {"in_progress", "queued"}:
        return (
            f"Interactions reply is not terminal (status {status!r}); background "
            "execution is outside the synchronous contract"
        )
    if status == "cancelled":
        return "Interactions reply was cancelled"
    return f"Interactions reply has unknown status {status!r}"


def _thought_summary(step: Mapping[str, object], path: str) -> list[str]:
    """Extract Content text; any separators must come from the provider."""

    extra = set(step) - {"type", "signature", "summary"}
    if extra:
        raise OutputContractError(f"{path} has unsupported field(s) {sorted(extra)}")
    signature = step.get("signature")
    if signature is not None and not isinstance(signature, str):
        raise OutputContractError(f"{path}.signature must be a string")
    raw_summary = step.get("summary")
    if raw_summary is None:
        return []
    texts: list[str] = []
    for index, raw_item in enumerate(_array(raw_summary, f"{path}.summary")):
        item = _mapping(raw_item, f"{path}.summary[{index}]")
        text = item.get("text")
        if item.get("type") != "text" or set(item) - {"type", "text"}:
            raise OutputContractError(
                f"{path}.summary[{index}] is not a plain text summary"
            )
        if not isinstance(text, str):
            raise OutputContractError(f"{path}.summary[{index}].text must be a string")
        texts.append(text)
    return texts


def _annotation(item: Mapping[str, object], path: str) -> Citation | None:
    kind = item.get("type")
    if kind is None:
        return None
    if kind != "url_citation":
        raise OutputContractError(
            f"{path} has citation type {kind!r}, which the shared IR cannot carry"
        )
    url = item.get("url")
    if not isinstance(url, str) or not url:
        raise OutputContractError(f"{path}.url must be a non-empty string")
    title = item.get("title")
    if title is not None and not isinstance(title, str):
        raise OutputContractError(f"{path}.title must be a string")
    return Citation(url=url, title=title)


def _optional_bool(value: object, path: str) -> bool:
    """``is_error`` may be absent or a boolean; ``1``/``"yes"`` are not booleans."""

    if value is None:
        return False
    if not isinstance(value, bool):
        raise OutputContractError(f"{path}.is_error must be a boolean")
    return value


def _check_span(item: Mapping[str, object], path: str) -> None:
    start = item.get("start_index")
    end = item.get("end_index")
    for name, value in (("start_index", start), ("end_index", end)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise OutputContractError(f"{path}.{name} must be a non-negative integer")


def _content_text(item: Mapping[str, object], path: str) -> str:
    if item.get("type") != "text":
        raise OutputContractError(f"{path} has unsupported type {item.get('type')!r}")
    extra = set(item) - {"type", "text", "annotations"}
    if extra:
        raise OutputContractError(f"{path} has unsupported field(s) {sorted(extra)}")
    text = item.get("text")
    if not isinstance(text, str):
        raise OutputContractError(f"{path}.text must be a string")
    return text


@dataclass(frozen=True)
class _HostedResult:
    """A hosted tool's result, kept with the evidence the wire gave for it.

    ``failed`` keeps a failed lookup distinct from a successful empty one.
    """

    kind: str
    result: JSONValue
    failed: bool


#: ``ServerToolUse.error_code`` is the only persisted slot for a hosted-tool
#: failure, and the wire only says whether the call failed.
_HOSTED_ERROR_CODE = "is_error"

_UNSET = object()


def _signature_evidence(steps: Sequence[JSONValue]) -> dict[tuple[str, str], str]:
    evidence: dict[tuple[str, str], str] = {}
    occurrences: dict[str, int] = {}
    for raw in steps:
        step = cast(dict[str, JSONValue], raw)
        kind = cast(str, step["type"])
        occurrence = occurrences.get(kind, 0)
        occurrences[kind] = occurrence + 1
        signature = step.get("signature")
        if isinstance(signature, str):
            identity = step.get("id", step.get("call_id", str(occurrence)))
            evidence[kind, cast(str, identity)] = signature
    return evidence


def _projection_mismatch(left: "_LiftState", right: "_LiftState") -> str | None:
    """Compare observable output without requiring matching text segmentation."""

    for field_name, left_value, right_value in (
        ("texts", "".join(left.texts), "".join(right.texts)),
        ("tool_calls", left.tool_calls, right.tool_calls),
        ("server_uses", left.server_uses, right.server_uses),
        ("citations", left.citations, right.citations),
        ("summaries", "".join(left.summaries), "".join(right.summaries)),
        ("saw_thought", left.saw_thought, right.saw_thought),
    ):
        if left_value != right_value:
            return field_name
    left_signatures = _signature_evidence(left.native_steps)
    right_signatures = _signature_evidence(right.native_steps)
    if any(
        left_signatures[key] != right_signatures[key]
        for key in left_signatures.keys() & right_signatures.keys()
    ):
        return "signatures"
    left_results = {
        key: (value.kind, value.result, value.failed)
        for key, value in left.server_results.items()
    }
    right_results = {
        key: (value.kind, value.result, value.failed)
        for key, value in right.server_results.items()
    }
    return None if left_results == right_results else "server_results"


@dataclass
class _LiftState:
    """Accumulate steps from one non-streamed reply or one streamed event run.

    Both transports share it, so the two lifting paths cannot drift.
    """

    request: Request
    declared_functions: frozenset[str]
    hosted_kinds: frozenset[str]
    tool_choice: JSONValue
    texts: list[str] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)
    saw_thought: bool = False
    absorbed: int = 0
    tool_calls: list[FunctionToolCall] = field(default_factory=list)
    server_uses: list[ServerToolUse] = field(default_factory=list)
    hosted_call_kinds: dict[str, str] = field(default_factory=dict)
    citations: list[Citation] = field(default_factory=list)
    server_results: dict[str, _HostedResult] = field(default_factory=dict)
    native_steps: list[JSONValue] = field(default_factory=list)

    def absorb(self, step: Mapping[str, object], index: int) -> None:
        path = f"response.steps[{index}]"
        kind = step.get("type")
        if kind == "function_call" and isinstance(step.get("arguments"), str):
            raw = cast(str, step["arguments"])
            try:
                decoded = json.loads(
                    raw,
                    object_pairs_hook=reject_duplicate_keys,
                    parse_constant=reject_json_constant,
                )
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise OutputContractError(
                    f"{path}.arguments is not strict JSON"
                ) from exc
            if not isinstance(decoded, dict):
                raise OutputContractError(f"{path}.arguments must decode to an object")
            step = {**step, "arguments": decoded}
        if "signature" in step and not isinstance(step["signature"], str):
            raise OutputContractError(f"{path}.signature must be a string")
        self.absorbed += 1
        if kind == "thought":
            self.saw_thought = True
            self.summaries.extend(_thought_summary(step, path))
        elif kind == "model_output":
            self._model_output(step, path)
        elif kind == "function_call":
            self._function_call(step, path)
        elif kind == "google_search_call":
            self._hosted_call(step, path, "google_search")
        elif kind == "google_search_result":
            self._hosted_result(step, path, "google_search")
        elif kind == "url_context_call":
            self._hosted_call(step, path, "url_context")
        elif kind == "url_context_result":
            self._hosted_result(step, path, "url_context")
        else:
            raise OutputContractError(f"{path} has unsupported type {kind!r}")
        self.native_steps.append(copy_json_value(step, path))

    def _model_output(self, step: Mapping[str, object], path: str) -> None:
        extra = set(step) - {"type", "content", "error"}
        if extra:
            raise OutputContractError(
                f"{path} has unsupported field(s) {sorted(extra)}"
            )
        if step.get("error") is not None:
            raise OutputContractError(f"{path} carries an error result")
        content = step.get("content")
        if content is None:
            return
        for index, raw_item in enumerate(_array(content, f"{path}.content")):
            item_path = f"{path}.content[{index}]"
            item = _mapping(raw_item, item_path)
            self.texts.append(_content_text(item, item_path))
            annotations = item.get("annotations")
            if annotations is None:
                continue
            for annotation_index, raw_annotation in enumerate(
                _array(annotations, f"{item_path}.annotations")
            ):
                annotation_path = f"{item_path}.annotations[{annotation_index}]"
                annotation = _mapping(raw_annotation, annotation_path)
                _check_span(annotation, annotation_path)
                citation = _annotation(annotation, annotation_path)
                if citation is not None:
                    self.citations.append(citation)

    def _function_call(self, step: Mapping[str, object], path: str) -> None:
        extra = set(step) - {"type", "id", "name", "arguments", "signature"}
        if extra:
            raise OutputContractError(
                f"{path} has unsupported field(s) {sorted(extra)}"
            )
        call_id = step.get("id")
        name = step.get("name")
        arguments = step.get("arguments")
        if not isinstance(call_id, str) or not call_id:
            raise OutputContractError(f"{path}.id must be a non-empty string")
        if not isinstance(name, str) or not name:
            raise OutputContractError(f"{path}.name must be a non-empty string")
        if not isinstance(arguments, Mapping):
            raise OutputContractError(f"{path}.arguments must be an object")
        if (
            not self.request.tools.functions
            or name not in self.declared_functions
            or self.tool_choice == "none"
        ):
            raise OutputContractError(
                f"Interactions returned function {name!r} outside the request's "
                "allowed function set"
            )
        if any(call.call_id == call_id for call in self.tool_calls):
            raise OutputContractError(
                f"Interactions returned function call id {call_id!r} twice"
            )
        self.tool_calls.append(
            FunctionToolCall(
                call_id=call_id,
                name=name,
                arguments=cast(
                    dict[str, JSONValue],
                    copy_json_value(arguments, f"{path}.arguments"),
                ),
            )
        )

    def _hosted_call(self, step: Mapping[str, object], path: str, kind: str) -> None:
        if kind not in self.hosted_kinds:
            raise OutputContractError(
                f"Interactions returned an unrequested {kind} call"
            )
        extra = set(step) - {"type", "id", "arguments", "signature", "search_type"}
        if extra:
            raise OutputContractError(
                f"{path} has unsupported field(s) {sorted(extra)}"
            )
        call_id = step.get("id")
        arguments = step.get("arguments")
        if not isinstance(call_id, str) or not call_id:
            raise OutputContractError(f"{path}.id must be a non-empty string")
        if not isinstance(arguments, Mapping):
            raise OutputContractError(f"{path}.arguments must be an object")
        if call_id in self.hosted_call_kinds:
            raise OutputContractError(
                f"Interactions returned hosted tool call id {call_id!r} twice"
            )
        self.hosted_call_kinds[call_id] = kind
        self.server_uses.append(
            ServerToolUse(
                tool_type=kind,
                tool_use_id=call_id,
                input=cast(Mapping[str, JSONValue], dict(arguments)),
            )
        )

    def _hosted_result(self, step: Mapping[str, object], path: str, kind: str) -> None:
        extra = set(step) - {"type", "call_id", "result", "is_error", "signature"}
        if extra:
            raise OutputContractError(
                f"{path} has unsupported field(s) {sorted(extra)}"
            )
        call_id = step.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise OutputContractError(f"{path}.call_id must be a non-empty string")
        call_kind = self.hosted_call_kinds.get(call_id)
        if call_kind is None:
            raise OutputContractError(
                f"{path} answers {call_id!r}, which no earlier step called"
            )
        if call_kind != kind:
            raise OutputContractError(
                f"{path} is a {kind} result for a {call_kind} call {call_id!r}"
            )
        if call_id in self.server_results:
            raise OutputContractError(
                f"Interactions returned a second {kind} result for {call_id!r}"
            )
        failed = _optional_bool(step.get("is_error"), path)
        result = copy_json_value(step.get("result"), f"{path}.result")
        if not failed:
            for index, raw_item in enumerate(_array(result, f"{path}.result")):
                item_path = f"{path}.result[{index}]"
                item = _mapping(raw_item, item_path)
                if kind == "url_context":
                    url = item.get("url")
                    if not isinstance(url, str) or not url:
                        raise OutputContractError(
                            f"{item_path}.url must be a non-empty string"
                        )
                    self.citations.append(Citation(url=url))
        self.server_results[call_id] = _HostedResult(kind, result, failed)

    def finish(
        self,
        status: str,
        reply: Mapping[str, object],
        context: "_ExecutionContext",
        *,
        usage_raw: object = _UNSET,
    ) -> Response:
        summary = "".join(self.summaries)
        opaque: str | None = None
        if (
            self.saw_thought
            or any(
                "signature" in cast(dict[str, JSONValue], step)
                for step in self.native_steps
            )
            or self.request.session.opaque_continuation is not None
        ):
            opaque = _encode_opaque(
                context.system_instruction, [*context.input_steps, *self.native_steps]
            )
            try:
                _decode_opaque(opaque)
            except DialectError as exc:
                raise OutputContractError(
                    f"response generated invalid opaque history: {exc}"
                ) from exc
        reasoning = (
            ReasoningOutput(text=summary or None, opaque_roundtrip=opaque)
            if summary or opaque is not None
            else None
        )
        server_uses = [
            ServerToolUse(
                tool_type=use.tool_type,
                tool_use_id=use.tool_use_id,
                input=use.input,
                result=self.server_results[use.tool_use_id].result
                if use.tool_use_id in self.server_results
                else None,
                error_code=(
                    _HOSTED_ERROR_CODE
                    if use.tool_use_id in self.server_results
                    and self.server_results[use.tool_use_id].failed
                    else None
                ),
            )
            for use in self.server_uses
        ]
        unanswered = set(self.server_results) - {
            use.tool_use_id for use in self.server_uses
        }
        if unanswered:
            raise OutputContractError(
                "Interactions returned a tool result without its call"
            )
        text = "".join(self.texts)
        _optional_string(reply.get("id"), "response.id")
        usage_value = reply.get("usage") if usage_raw is _UNSET else usage_raw
        return Response(
            texts=(text,),
            reasoning=(
                (reasoning,)
                if reasoning is not None or self.request.reasoning.summary == "auto"
                else None
            ),
            finish_reasons=(status,),
            tool_calls=tuple(self.tool_calls) or None,
            server_tool_uses=tuple(server_uses) or None,
            structured_output=_structured_output(self.request, status, text),
            citations=tuple(self.citations) or None,
            session_id=None,
            usage=_usage(usage_value),
            request_params=context.request_params,
            response_model=_optional_string(reply.get("model"), "response.model"),
        )


def _structured_output(req: Request, status: str, text: str) -> StructuredOutput | None:
    if req.structured_output.format is None or status == "incomplete":
        return None
    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_json_constant,
        )
        return StructuredOutput(cast(JSONValue, value))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise OutputContractError(
            "Interactions structured output is not valid JSON"
        ) from exc


def _stream_event(payload: str) -> dict[str, object]:
    raw = strict_json_loads(payload, "Interactions stream event")
    return dict(_mapping(raw, "Interactions stream event"))


#: Delta types that carry a whole value for their step, with the fields each
#: may set.  Everything else a delta can do (text, arguments, signature,
#: summary, annotations) is an incremental append handled in ``_apply_delta``.
_WHOLE_VALUE_DELTAS = MappingProxyType(
    {
        "google_search_call": frozenset({"arguments", "signature"}),
        "google_search_result": frozenset({"result", "is_error", "signature"}),
        "url_context_call": frozenset({"arguments", "signature"}),
        "url_context_result": frozenset({"result", "is_error", "signature"}),
    }
)


@dataclass
class _StreamState:
    """Reassemble `step.start` / `step.delta` / `step.stop` into whole steps."""

    partial: dict[int, dict[str, object]] = field(default_factory=dict)
    stopped: set[int] = field(default_factory=set)
    stopped_steps: dict[int, dict[str, object]] = field(default_factory=dict)
    cumulative_usage: object = None

    def absorb_in_index_order(self, state: "_LiftState") -> None:
        """Apply the stopped steps by index, not by the order they stopped."""

        for index in sorted(self.stopped_steps):
            state.absorb(self.stopped_steps[index], index)

    def consume(self, event: Mapping[str, object]) -> None:
        event_type = event.get("event_type")
        if event_type == "step.start":
            index = _stream_index(event)
            step = _mapping(event.get("step"), "step.start.step")
            if index in self.partial or index in self.stopped:
                raise OutputContractError(f"stream restarted step {index}")
            self.partial[index] = dict(step)
            return
        if event_type == "step.delta":
            index = _stream_index(event)
            if index not in self.partial:
                raise OutputContractError(f"stream delta for unknown step {index}")
            self._apply_delta(
                self.partial[index], _mapping(event.get("delta"), "step.delta.delta")
            )
            if "metadata" in event:
                metadata = _mapping(event["metadata"], "step.delta.metadata")
                if "total_usage" in metadata:
                    usage = _mapping(
                        metadata["total_usage"], "step.delta.metadata.total_usage"
                    )
                    try:
                        _usage(usage)
                    except OutputContractError as exc:
                        raise OutputContractError(
                            f"step.delta.metadata.total_usage: {exc}"
                        ) from exc
                    self.cumulative_usage = usage
            return
        if event_type == "step.stop":
            index = _stream_index(event)
            usage = event.get("usage")
            if usage is not None:
                self.cumulative_usage = usage
            step = self.partial.pop(index, None)
            if step is None:
                raise OutputContractError(f"stream stopped unknown step {index}")
            self.stopped.add(index)
            self.stopped_steps[index] = step
            return
        raise OutputContractError(f"stream emitted unexpected event {event_type!r}")

    def _apply_delta(
        self, step: dict[str, object], delta: Mapping[str, object]
    ) -> None:
        kind = delta.get("type")
        if kind == "text":
            text = delta.get("text")
            if not isinstance(text, str):
                raise OutputContractError("stream text delta must carry a string")
            content = step.get("content")
            if not isinstance(content, list):
                content = []
                step["content"] = content
            blocks = cast(list[object], content)
            if not blocks:
                blocks.append({"type": "text", "text": ""})
            last = cast(dict[str, object], blocks[-1])
            last["text"] = f"{last.get('text', '')}{text}"
            return
        if kind == "arguments_delta":
            arguments = delta.get("arguments")
            if not isinstance(arguments, str):
                raise OutputContractError("stream arguments delta must be a string")
            current = step.get("arguments")
            # ``step.start`` seeds ``arguments`` as ``{}``; the deltas carry the
            # JSON text, accumulated here and decoded once the step stops.
            buffer = (
                ""
                if current is None or (isinstance(current, Mapping) and not current)
                else current
            )
            if not isinstance(buffer, str):
                raise OutputContractError("stream accumulated function arguments twice")
            step["arguments"] = f"{buffer}{arguments}"
            return
        if kind == "thought_signature":
            signature = delta.get("signature")
            if not isinstance(signature, str):
                raise OutputContractError("stream signature delta must be a string")
            step["signature"] = signature
            return
        if kind == "thought_summary":
            content = _mapping(delta.get("content"), "thought_summary.content")
            summary = step.get("summary")
            if summary is None:
                summary = []
                step["summary"] = summary
            if not isinstance(summary, list):
                raise OutputContractError("thought summary is not an array")
            cast(list[object], summary).append(dict(content))
            return
        if kind in _WHOLE_VALUE_DELTAS:
            allowed = _WHOLE_VALUE_DELTAS[kind]
            extra = set(delta) - allowed - {"type"}
            if extra:
                raise OutputContractError(
                    f"stream {kind} delta has unsupported field(s) {sorted(extra)}"
                )
            step.update({key: value for key, value in delta.items() if key != "type"})
            return
        if kind == "text_annotation_delta":
            annotations = delta.get("annotations")
            if not isinstance(annotations, list):
                raise OutputContractError(
                    "stream text_annotation_delta must carry an annotations array"
                )
            content = step.get("content")
            if not isinstance(content, list) or not content:
                raise OutputContractError(
                    "stream text_annotation_delta arrived before any text"
                )
            block = cast(dict[str, object], cast(list[object], content)[-1])
            existing = block.get("annotations")
            if existing is None:
                existing = []
                block["annotations"] = existing
            if not isinstance(existing, list):
                raise OutputContractError("stream annotations are not an array")
            cast(list[object], existing).extend(annotations)
            return
        raise OutputContractError(f"stream emitted unsupported delta {kind!r}")


def _stream_index(event: Mapping[str, object]) -> int:
    index = event.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise OutputContractError("stream event index must be a non-negative integer")
    return index


async def _read_stream(response: httpx.Response) -> list[dict[str, object]]:
    """Parse SSE frames up to and including ``[DONE]``.

    ``id`` and ``event_id`` lines are tolerated, never required.
    """

    events: list[dict[str, object]] = []
    data_lines: list[str] = []

    def flush() -> None:
        if not data_lines:
            return
        payload = "\n".join(data_lines)
        data_lines.clear()
        if payload.strip() == "[DONE]":
            events.append({"event_type": "done"})
            return
        event = _stream_event(payload)
        event_type = event.get("event_type")
        if not isinstance(event_type, str) or not event_type:
            raise OutputContractError("stream event has no event_type")
        events.append(event)

    async for line in response.aiter_lines():
        if not line:
            flush()
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if not separator:
            raise OutputContractError("Interactions stream emitted a malformed line")
        if field == "data":
            data_lines.append(value.removeprefix(" "))
        elif field not in {"event", "id", "retry", "event_id"}:
            raise OutputContractError(
                f"Interactions stream emitted unsupported field {field!r}"
            )
    flush()
    if not events:
        raise OutputContractError("Interactions stream emitted no events")
    return events


def _completed_interaction(event: Mapping[str, object]) -> dict[str, object]:
    return dict(_mapping(event.get("interaction"), "interaction.completed.interaction"))


def _reconcile_terminal_steps(
    terminal: Mapping[str, object], lifted: "_LiftState"
) -> None:
    """Use the terminal snapshot when it is the only evidence, and check it
    against the deltas otherwise.

    ``steps`` is optional on the terminal event; a snapshot that disagrees
    with the deltas fails rather than picking a side.
    """

    raw = terminal.get("steps")
    if raw is None:
        return
    snapshot = _LiftState(
        request=lifted.request,
        declared_functions=lifted.declared_functions,
        hosted_kinds=lifted.hosted_kinds,
        tool_choice=lifted.tool_choice,
    )
    for index, raw_step in enumerate(_array(raw, "interaction.completed.steps")):
        snapshot.absorb(
            _mapping(raw_step, f"interaction.completed.steps[{index}]"), index
        )
    if lifted.absorbed == 0:
        # No delta evidence: the snapshot is the whole reply.
        lifted.texts = snapshot.texts
        lifted.summaries = snapshot.summaries
        lifted.saw_thought = snapshot.saw_thought
        lifted.absorbed = snapshot.absorbed
        lifted.tool_calls = snapshot.tool_calls
        lifted.server_uses = snapshot.server_uses
        lifted.hosted_call_kinds = snapshot.hosted_call_kinds
        lifted.citations = snapshot.citations
        lifted.server_results = snapshot.server_results
        lifted.native_steps = snapshot.native_steps
        return
    mismatch = _projection_mismatch(lifted, snapshot)
    if mismatch is not None:
        raise OutputContractError(
            "interaction.completed steps disagree with the streamed steps on "
            f"{mismatch!r}"
        )


def _merge_stream(
    events: list[dict[str, object]], context: "_ExecutionContext"
) -> Response:
    lifted = _LiftState(
        request=context.request,
        declared_functions=context.declared_functions,
        hosted_kinds=context.hosted_kinds,
        tool_choice=context.tool_choice,
    )
    steps = _StreamState()
    terminal: dict[str, object] | None = None
    done = False
    for event in events:
        event_type = event.get("event_type")
        if done:
            raise OutputContractError(
                "Interactions stream sent an event after its terminal marker"
            )
        if event_type == "interaction.created":
            created = _mapping(
                event.get("interaction"), "interaction.created.interaction"
            )
            if created.get("status") not in (None, "in_progress"):
                raise OutputContractError(
                    "interaction.created reported a status other than in_progress"
                )
        elif event_type == "interaction.status_update":
            if event.get("status") not in (None, "in_progress"):
                raise OutputContractError(
                    "interaction.status_update reported a terminal status"
                )
        elif event_type == "error":
            error = _mapping(event.get("error"), "error")
            message = error.get("message")
            raise OutputContractError(
                "Interactions stream reported an error"
                + (f": {message}" if isinstance(message, str) else "")
            )
        elif event_type == "interaction.completed":
            if terminal is not None:
                raise OutputContractError(
                    "Interactions stream sent interaction.completed twice"
                )
            terminal = _completed_interaction(event)
        elif event_type == "done":
            if terminal is None:
                raise OutputContractError(
                    "Interactions stream ended without interaction.completed"
                )
            done = True
        elif event_type in {"step.start", "step.delta", "step.stop"}:
            if terminal is not None:
                raise OutputContractError(
                    "Interactions stream sent a step after interaction.completed"
                )
            steps.consume(event)
        else:
            raise OutputContractError(
                f"stream emitted unsupported event_type {event_type!r}"
            )
    if terminal is None:
        raise OutputContractError(
            "Interactions stream ended without interaction.completed"
        )
    if steps.partial:
        raise OutputContractError(
            f"Interactions stream ended with an unfinished step {sorted(steps.partial)}"
        )
    steps.absorb_in_index_order(lifted)
    _reconcile_terminal_steps(terminal, lifted)
    status = terminal.get("status")
    if not isinstance(status, str) or not status:
        raise OutputContractError("interaction.completed has no status")
    if status == "requires_action" and not context.request.tools.functions:
        raise OutputContractError(_status_error(status, terminal))
    if status not in _TERMINAL_STATUSES:
        raise OutputContractError(_status_error(status, terminal))
    # ``step.stop`` carries cumulative usage; the terminal event may or may not.
    usage = terminal.get("usage", steps.cumulative_usage)
    return lifted.finish(status, terminal, context, usage_raw=usage)
