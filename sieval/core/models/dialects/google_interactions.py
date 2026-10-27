"""Google Gemini Interactions API (``POST /{version}/interactions``) dialect.

Interactions is the endpoint Google recommends for new Gemini work, beside
``generateContent`` (``google_genai``). It is a separate dialect because the
two expose different eval controls: Interactions has no candidate count, no
``top_k`` and no logprobs, so those requests are refused here and stay on
``google_genai``.

Every request sends ``store: false``: Google otherwise retains the prompt and
reply server-side by default, and eval prompts must not be retained. That also
rules out ``previous_interaction_id``, so multi-turn input is stateless — the
whole history is resent as steps, and signed thought steps round-trip through
``opaque_continuation``.

The termination signal is the interaction ``status``, recorded verbatim as the
finish reason. ``incomplete`` covers a hit token budget and other early stops
alike; the reply does not say which.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
from collections.abc import Mapping, Sequence
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
    StructuredOutput,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    UsageStats,
)
from sieval.core.types import JSONValue

from ._google_genai_wire import resolve_wire_endpoint
from ._http_status import raise_for_provider_status

_THINKING_LEVELS = frozenset({"minimal", "low", "medium", "high"})
_THINKING_SUMMARIES = frozenset({"none", "auto"})
_IMAGE_MEDIA_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/heic",
        "image/heif",
        "image/gif",
        "image/bmp",
        "image/tiff",
    }
)
_CHAT_ROLES = frozenset({"system", "user", "assistant", "tool"})
_OPAQUE_VERSION = 1
#: Statuses a create call can end in. ``in_progress``/``queued`` only occur
#: with ``background``, which ``store: false`` rules out.
_TERMINAL_STATUSES = frozenset({"completed", "incomplete", "requires_action"})
#: A status that may leave requested channels empty.
_EMPTY_OUTCOME_STATUSES = frozenset({"incomplete"})
#: Body keys this dialect owns; a dialect option may not overwrite them.
_IR_OWNED_KEYS = BINDING_RESOURCE_KEYS | frozenset(
    {
        "model",
        "agent",
        "agent_config",
        "input",
        "system_instruction",
        "generation_config",
        "tools",
        "response_format",
        "stream",
        "store",
        "background",
        "previous_interaction_id",
    }
)


def _validate_reasoning_config(options: object) -> None:
    assert isinstance(options, ReasoningOptions)
    if options.budget_tokens is not None:
        raise ValueError("google_interactions has no thinking budget; use effort")
    if options.summary not in {None, *_THINKING_SUMMARIES}:
        raise ValueError(
            "google_interactions supports only reasoning summary 'none' or 'auto'"
        )
    if options.effort is not None and options.effort not in _THINKING_LEVELS:
        raise ValueError(f"unsupported Interactions thinking level {options.effort!r}")


def _validate_multimodal_config(options: object) -> None:
    assert isinstance(options, MultimodalInputOptions)
    unsupported = set(options.modalities) - {"image"}
    if unsupported:
        raise ValueError(
            f"google_interactions does not support modalities {unsupported!r}"
        )


def _decisions() -> Mapping[CapabilityKey, DialectCapabilityDecision]:
    decisions: dict[CapabilityKey, DialectCapabilityDecision] = {
        "input_scoring": Unsupported("Interactions cannot score input tokens"),
        "sampled_logprobs": Unsupported("Interactions returns no logprobs"),
        "top_logprobs": Unsupported("Interactions returns no logprobs"),
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
        "hosted_tools": Unsupported(
            "google_interactions does not lift hosted tool steps yet"
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
        "stateful_session": Unsupported("store=false disables previous_interaction_id"),
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
        "prefill": Unsupported("Interactions has no assistant-prefill input"),
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
        "server_tool_uses": OutputRule(Guarantee.NEVER),
        "structured_output": OutputRule(
            Guarantee.PRESENT_OR_ERROR, validate_structured_output
        ),
        "logprobs": OutputRule(Guarantee.NEVER),
        "top_logprobs": OutputRule(Guarantee.NEVER),
        "choice_logprobs": OutputRule(Guarantee.NEVER),
        "input_scoring": OutputRule(Guarantee.NEVER),
        "citations": OutputRule(Guarantee.BEST_EFFORT),
        "grounding": OutputRule(Guarantee.NEVER),
        "session_id": OutputRule(Guarantee.NEVER),
        "usage": OutputRule(Guarantee.BEST_EFFORT),
    },
    empty_outcome_finish_reasons=_EMPTY_OUTCOME_STATUSES,
)


@dataclass(frozen=True)
class _InteractionsContext:
    request: Request


@dataclass(frozen=True)
class _ExecutionContext:
    request: Request
    request_params: Mapping[str, JSONValue]
    system_instruction: str | None
    input_steps: tuple[JSONValue, ...]


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


def _get(raw: object, name: str, default: object = None) -> object:
    if isinstance(raw, Mapping):
        return cast(Mapping[str, object], raw).get(name, default)
    return default


def _sequence(raw: object, path: str) -> Sequence[object]:
    if not isinstance(raw, list):
        raise OutputContractError(f"{path} must be a list")
    return cast(list[object], raw)


def _required_string(raw: object, path: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise OutputContractError(f"{path} must be a non-empty string")
    return raw


def _optional_string(raw: object, path: str) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise OutputContractError(f"{path} must be a string")
    return raw


def _nonnegative_int(raw: object, path: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise OutputContractError(f"{path} must be a non-negative integer")
    return raw


def _reject_duplicate_keys(pairs: list[tuple[str, JSONValue]]) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


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


def _function_declaration(raw: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    """Lower one OpenAI-shaped or bare function declaration to a function tool."""

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
            "unsupported Interactions function field(s): " + ", ".join(sorted(unknown))
        )
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise DialectError("Interactions function declaration requires a name")
    result: dict[str, JSONValue] = {"type": "function", "name": name}
    if "description" in data:
        if not isinstance(data["description"], str):
            raise DialectError("Interactions function description must be a string")
        result["description"] = data["description"]
    if "parameters" in data:
        if not isinstance(data["parameters"], Mapping):
            raise DialectError("Interactions function parameters must be an object")
        result["parameters"] = copy_json_value(
            data["parameters"], "function parameters"
        )
    return result


def _tool_choice(raw: JSONValue, declared: frozenset[str]) -> JSONValue:
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
        raise DialectError("Interactions function-specific tool choice requires a name")
    if name not in declared:
        raise DialectError("Interactions tool choice names an undeclared function")
    return {"allowed_tools": {"mode": "any", "tools": [name]}}


def _declared_names(req: Request) -> frozenset[str]:
    names: set[str] = set()
    for function in req.tools.functions:
        name = _function_declaration(function).get("name")
        if isinstance(name, str):
            names.add(name)
    return frozenset(names)


def _image_content(part: ImagePart) -> dict[str, JSONValue]:
    if part.detail is not None:
        raise DialectError("Interactions images have no per-image detail field")
    if part.media_type is not None and part.media_type not in _IMAGE_MEDIA_TYPES:
        raise DialectError(
            f"unsupported Interactions image media type {part.media_type!r}"
        )
    content: dict[str, JSONValue] = {"type": "image"}
    if part.url is not None:
        if not part.url:
            raise DialectError("Interactions image URLs must not be empty")
        content["uri"] = part.url
    else:
        if not part.data or not part.media_type:
            raise DialectError(
                "Interactions inline images require data and an explicit MIME type"
            )
        content["data"] = part.data
    if part.media_type is not None:
        content["mime_type"] = part.media_type
    return content


def _chat_shape_error(request_input: ChatInput) -> str | None:
    for index, message in enumerate(request_input.messages):
        if message.role == "developer":
            return (
                "Interactions system_instruction cannot preserve the distinct "
                "developer message authority"
            )
        if message.role not in _CHAT_ROLES:
            return f"Interactions message {index} has invalid role {message.role!r}"
    if request_input.messages[-1].role == "assistant":
        return "google_interactions does not support assistant-prefill input"
    return None


def _message_steps(
    message: ChatMessage, call_names: dict[str, str]
) -> list[dict[str, JSONValue]]:
    """Lower one chat message to steps, keeping its part order.

    Consecutive text/image parts share one ``user_input`` or ``model_output``
    step; each tool call or result is a step of its own.
    """

    content_type = "model_output" if message.role == "assistant" else "user_input"
    steps: list[dict[str, JSONValue]] = []
    content: list[JSONValue] = []

    def flush() -> None:
        if content:
            steps.append({"type": content_type, "content": list(content)})
            content.clear()

    for part in message.content:
        if isinstance(part, TextPart):
            content.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            if message.role == "assistant":
                raise DialectError("Interactions model output cannot carry images")
            content.append(_image_content(part))
        elif isinstance(part, ToolCallPart):
            if message.role != "assistant":
                raise DialectError("only assistant messages may carry tool calls")
            if not isinstance(part.arguments, Mapping):
                raise DialectError(
                    "Interactions function call arguments must be an object"
                )
            flush()
            call_names[part.call_id] = part.name
            steps.append(
                {
                    "type": "function_call",
                    "id": part.call_id,
                    "name": part.name,
                    "arguments": copy_json_value(part.arguments, "tool call"),
                }
            )
        elif isinstance(part, ToolResultPart):
            if message.role not in {"user", "tool"}:
                raise DialectError("only user or tool messages may carry tool results")
            flush()
            result: dict[str, JSONValue] = {
                "type": "function_result",
                "call_id": part.call_id,
                "result": copy_json_value(part.result, "tool result"),
            }
            name = call_names.get(part.call_id)
            if name is not None:
                result["name"] = name
            if part.is_error:
                result["is_error"] = True
            steps.append(result)
        else:
            raise DialectError(f"unsupported chat part {type(part).__name__}")
    flush()
    return steps


def _decode_opaque(value: str) -> tuple[str | None, list[JSONValue]]:
    try:
        raw = json.loads(
            value,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise DialectError("invalid Interactions opaque continuation JSON") from exc
    if (
        not isinstance(raw, dict)
        or not {"version", "steps"}.issubset(raw)
        or set(raw) - {"version", "steps", "system_instruction"}
    ):
        raise DialectError("invalid Interactions opaque continuation envelope")
    if raw["version"] != _OPAQUE_VERSION:
        raise DialectError("unsupported Interactions opaque continuation version")
    steps = raw["steps"]
    if not isinstance(steps, list) or not all(
        isinstance(step, dict) and isinstance(step.get("type"), str) for step in steps
    ):
        raise DialectError("Interactions opaque continuation steps are invalid")
    system = raw.get("system_instruction")
    if system is not None and not isinstance(system, str):
        raise DialectError("Interactions opaque system_instruction must be a string")
    return system, cast(list[JSONValue], steps)


def _input_to_wire(
    request_input: ChatInput, opaque: str | None
) -> tuple[str | None, list[JSONValue]]:
    """Lower chat input, after any opaque history, to a system string and steps."""

    if error := _chat_shape_error(request_input):
        raise DialectError(error)
    opaque_system, steps = _decode_opaque(opaque) if opaque is not None else (None, [])
    steps = list(steps)
    call_names: dict[str, str] = {}
    for step in steps:
        call_id = _get(step, "id")
        name = _get(step, "name")
        if (
            _get(step, "type") == "function_call"
            and isinstance(call_id, str)
            and isinstance(name, str)
        ):
            call_names[call_id] = name
    system_texts: list[str] = []
    saw_conversation = False
    for message in request_input.messages:
        if message.name is not None:
            raise DialectError("Interactions steps have no ChatMessage.name field")
        if message.role == "system":
            if saw_conversation:
                raise DialectError(
                    "Interactions system messages must precede conversation turns"
                )
            for part in message.content:
                if not isinstance(part, TextPart):
                    raise DialectError("Interactions system_instruction is text-only")
                system_texts.append(part.text)
            continue
        saw_conversation = True
        steps.extend(_message_steps(message, call_names))
    if len(system_texts) > 1:
        raise DialectError("Interactions system_instruction is a single string")
    current_system = system_texts[0] if system_texts else None
    if (
        opaque_system is not None
        and current_system is not None
        and opaque_system != current_system
    ):
        raise DialectError(
            "Interactions opaque continuation system_instruction conflicts with "
            "the current request"
        )
    if not steps or _get(steps[-1], "type") not in {"user_input", "function_result"}:
        raise DialectError(
            "Interactions input must end with a user input or function result step"
        )
    return (current_system if current_system is not None else opaque_system), steps


def _usage(raw: object) -> UsageStats | None:
    """Lift Interactions usage onto the shared accounting.

    Like ``generateContent``, ``total_output_tokens`` excludes thoughts and
    ``total_input_tokens`` excludes the tool-use prompt; folding both in keeps
    ``reasoning_tokens`` a breakdown of ``output_tokens``.
    """

    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise OutputContractError("Interactions usage must be an object")
    prompt = _nonnegative_int(
        _get(raw, "total_input_tokens"), "usage.total_input_tokens"
    )
    output = _nonnegative_int(
        _get(raw, "total_output_tokens"), "usage.total_output_tokens"
    )
    thoughts = _nonnegative_int(
        _get(raw, "total_thought_tokens"), "usage.total_thought_tokens"
    )
    tool_use = _nonnegative_int(
        _get(raw, "total_tool_use_tokens"), "usage.total_tool_use_tokens"
    )
    cached = _nonnegative_int(
        _get(raw, "total_cached_tokens"), "usage.total_cached_tokens"
    )
    reported = _nonnegative_int(_get(raw, "total_tokens"), "usage.total_tokens")
    input_tokens = (prompt or 0) + (tool_use or 0)
    output_tokens = (output or 0) + (thoughts or 0)
    total = input_tokens + output_tokens
    return UsageStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total,
        reasoning_tokens=thoughts,
        cached_tokens=cached,
        reported_total_tokens=reported
        if reported is not None and reported != total
        else None,
    )


def _citations(annotations: object, path: str) -> list[Citation]:
    if annotations is None:
        return []
    citations: list[Citation] = []
    for index, annotation in enumerate(_sequence(annotations, path)):
        if _get(annotation, "type") != "url_citation":
            continue
        citations.append(
            Citation(
                url=_required_string(_get(annotation, "url"), f"{path}[{index}].url"),
                title=_optional_string(
                    _get(annotation, "title"), f"{path}[{index}].title"
                ),
            )
        )
    return citations


def _structured_output(req: Request, text: str) -> StructuredOutput | None:
    if req.structured_output.format is None:
        return None
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise OutputContractError(
            "Interactions structured output is not valid JSON"
        ) from exc
    return StructuredOutput(cast(JSONValue, value))


def _failure_message(raw: object) -> str:
    messages: list[str] = []
    errors = _get(raw, "errors")
    if isinstance(errors, list):
        for error in errors:
            message = _get(error, "message")
            if isinstance(message, str) and message:
                messages.append(message)
    return "Interactions request failed" + (
        ": " + "; ".join(messages) if messages else ""
    )


def _append_text(items: list[object], text: str) -> None:
    """Append streamed text to the last text item, or start one."""

    last = items[-1] if items else None
    if isinstance(last, dict) and _get(last, "type") == "text":
        item = cast(dict[str, object], last)
        previous = item.get("text")
        item["text"] = (previous if isinstance(previous, str) else "") + text
    else:
        items.append({"type": "text", "text": text})


class GoogleInteractionsDialect:
    """Executable adapter for the Gemini Interactions REST endpoint."""

    dialect_id = "google_interactions"
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
        }
    )

    def __init__(self, connection: object, requested_model_id: str):
        if not isinstance(connection, AsyncHTTPJSONConnection):
            raise TypeError("google_interactions requires AsyncHTTPJSONConnection")
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
        wire_contract, self._endpoint_root = resolve_wire_endpoint(
            connection.client.base_url
        )
        self._operation = f"{wire_contract.api_version}/interactions"

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
        for path in audit.active_paths:
            if path == "input.chat":
                if chat_shape_error is not None:
                    audit.rejected(path, chat_shape_error)
            elif path == "input.chat.message.name":
                audit.rejected(
                    path, "Interactions steps have no ChatMessage.name field"
                )
            elif path.startswith("input.completion"):
                audit.rejected(path, "google_interactions requires ChatInput")
            elif path == "input.modality.image.detail":
                audit.rejected(path, "Interactions images have no per-image detail")
            elif path == "sampling.n":
                audit.rejected(
                    path, "Interactions has no candidate count; use google_genai"
                )
            elif path == "sampling.top_k":
                audit.rejected(path, "Interactions has no top_k; use google_genai")
            elif path in {"sampling.frequency_penalty", "sampling.presence_penalty"}:
                audit.rejected(path, "Interactions has no frequency/presence penalty")
            elif path in {
                "scoring.input_scoring",
                "scoring.sampled_logprobs",
                "scoring.top_logprobs",
            }:
                audit.rejected(
                    path, "Interactions returns no logprobs; use google_genai"
                )
            elif path == "reasoning.budget_tokens":
                audit.rejected(path, "Interactions has no thinking budget; use effort")
            elif (
                path == "reasoning.effort"
                and req.reasoning.effort not in _THINKING_LEVELS
            ):
                audit.rejected(path, "unsupported Interactions thinking level")
            elif (
                path == "reasoning.summary"
                and req.reasoning.summary not in _THINKING_SUMMARIES
            ):
                audit.rejected(
                    path, "Interactions supports only summary='none' or 'auto'"
                )
            elif path == "tools.parallel":
                audit.rejected(path, "Interactions has no parallel tool switch")
            elif path == "tools.hosted":
                audit.rejected(
                    path, "google_interactions does not support hosted tools"
                )
            elif path == "tools.functions":
                try:
                    for function in req.tools.functions:
                        _function_declaration(function)
                except DialectError as exc:
                    audit.rejected(path, str(exc))
            elif path == "tools.choice":
                if not req.tools.functions:
                    audit.rejected(
                        path, "Interactions tool choice requires function tools"
                    )
                else:
                    try:
                        _tool_choice(req.tools.choice, _declared_names(req))
                    except DialectError as exc:
                        audit.rejected(path, str(exc))
            elif path == "session.previous_response_id":
                audit.rejected(path, "store=false disables previous_interaction_id")
            elif path == "structured_output.schema" and (
                req.structured_output.format != "json_schema"
            ):
                audit.rejected(path, "a response schema requires format='json_schema'")
            elif path == "structured_output.name":
                audit.rejected(path, "Interactions response schemas have no name")
            elif path == "structured_output.strict":
                audit.rejected(path, "Interactions has no JSON-schema strict switch")
            elif path.startswith("dialect_options."):
                key = path.removeprefix("dialect_options.")
                if not key:
                    audit.rejected(path, "dialect option keys must be non-empty")
                elif key in _IR_OWNED_KEYS:
                    audit.rejected(path, f"{key!r} is owned by google_interactions")

    def prepare(self, req: Request, audit: RequestAudit) -> PreparedRequest:
        audit.require_matches_request(req)
        audit.raise_rejections()
        if not isinstance(req.input, ChatInput):
            raise DialectError("google_interactions requires ChatInput")
        active = frozenset(audit.active_paths)

        def consume(path: str, *evidence: WireEvidence) -> None:
            if path in active and path not in audit.decisions:
                audit.consumed(path, *evidence)

        opaque = (
            req.session.opaque_continuation.value
            if req.session.opaque_continuation is not None
            else None
        )
        system, steps = _input_to_wire(req.input, opaque)
        body: dict[str, JSONValue] = {
            "model": self._model_name,
            "input": steps,
            "store": False,
        }
        if system is not None:
            body["system_instruction"] = system
        input_verifier = _InputVerifier(system, tuple(steps))
        envelope = PreparedRequestEvidence(
            _EnvelopeVerifier(self._operation, self._model_name, req.scheduling.stream)
        )
        consume("input.chat", input_verifier, envelope)
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
        fields = {
            "temperature": "temperature",
            "top_p": "top_p",
            "max_tokens": "max_output_tokens",
            "stop": "stop_sequences",
            "seed": "seed",
        }
        for source, destination in fields.items():
            value = getattr(req.sampling, source)
            if value is not None:
                generation[destination] = (
                    list(value) if isinstance(value, tuple) else value
                )
                consume(
                    f"sampling.{source}",
                    WireObservation(("generation_config", destination)),
                )
        if req.reasoning.effort is not None:
            generation["thinking_level"] = req.reasoning.effort
            consume(
                "reasoning.effort",
                WireObservation(("generation_config", "thinking_level")),
            )
        if req.reasoning.summary is not None:
            generation["thinking_summaries"] = req.reasoning.summary
            consume(
                "reasoning.summary",
                WireObservation(("generation_config", "thinking_summaries")),
            )

        if req.tools.functions:
            body["tools"] = [
                _function_declaration(item) for item in req.tools.functions
            ]
            consume("tools.functions", _FunctionsVerifier(req.tools.functions))
        if req.tools.choice is not None:
            generation["tool_choice"] = _tool_choice(
                req.tools.choice, _declared_names(req)
            )
            consume("tools.choice", _ToolChoiceVerifier(req.tools.choice))
        if generation:
            body["generation_config"] = generation

        if req.structured_output.format is not None:
            response_format: dict[str, JSONValue] = {
                "type": "text",
                "mime_type": "application/json",
            }
            if req.structured_output.schema is not None:
                response_format["schema"] = copy_json_value(
                    dict(req.structured_output.schema), "response schema"
                )
                consume(
                    "structured_output.schema",
                    WireObservation(("response_format", "schema")),
                )
            body["response_format"] = response_format
            consume(
                "structured_output.format",
                _StructuredFormatVerifier(req.structured_output.format),
            )

        if req.scheduling.stream:
            body["stream"] = True
            consume("scheduling.stream", WireObservation(("stream",)))

        options = req.dialect_options
        if options is not None:
            if not options.values:
                audit.noop(
                    "dialect_options", "empty matching options have no wire semantics"
                )
            for key, value in options.values.items():
                path = f"dialect_options.{key}"
                if path in audit.decisions:
                    continue
                body[key] = value
                audit.consumed(path, _RootOptionVerifier(key, value))

        return PreparedRequest(
            self._operation, body, _InteractionsContext(deepcopy(req))
        )

    def _lift(self, raw: object, context: _ExecutionContext) -> Response:
        req = context.request
        if not isinstance(raw, Mapping):
            raise OutputContractError("Interactions response must be an object")
        status = _required_string(_get(raw, "status"), "interaction.status")
        if status == "failed":
            raise OutputContractError(_failure_message(raw))
        if status not in _TERMINAL_STATUSES:
            raise OutputContractError(
                f"Interactions returned non-terminal status {status!r}"
            )
        raw_steps = _get(raw, "steps")
        steps = () if raw_steps is None else _sequence(raw_steps, "interaction.steps")
        declared = _declared_names(req)
        text_parts: list[str] = []
        thought_texts: list[str] = []
        calls: list[FunctionToolCall] = []
        citations: list[Citation] = []
        output_steps: list[JSONValue] = []
        saw_signature = False
        for index, step in enumerate(steps):
            path = f"interaction.steps[{index}]"
            step_type = _required_string(_get(step, "type"), f"{path}.type")
            if step_type == "model_output":
                error = _get(step, "error")
                if error is not None:
                    message = _get(error, "message")
                    raise OutputContractError(
                        "Interactions model output step failed"
                        + (f": {message}" if isinstance(message, str) else "")
                    )
                content = _get(step, "content")
                items = () if content is None else _sequence(content, f"{path}.content")
                for item_index, item in enumerate(items):
                    item_path = f"{path}.content[{item_index}]"
                    item_type = _get(item, "type")
                    if item_type != "text":
                        raise OutputContractError(
                            f"{item_path} has unsupported type {item_type!r}"
                        )
                    text = _get(item, "text")
                    if not isinstance(text, str):
                        raise OutputContractError(f"{item_path}.text must be a string")
                    text_parts.append(text)
                    citations.extend(
                        _citations(
                            _get(item, "annotations"), f"{item_path}.annotations"
                        )
                    )
            elif step_type == "thought":
                signature = _get(step, "signature")
                if signature is not None:
                    _required_string(signature, f"{path}.signature")
                    saw_signature = True
                summary = _get(step, "summary")
                for item in (
                    () if summary is None else _sequence(summary, f"{path}.summary")
                ):
                    if _get(item, "type") == "text":
                        text = _get(item, "text")
                        if not isinstance(text, str):
                            raise OutputContractError(
                                f"{path}.summary text must be a string"
                            )
                        thought_texts.append(text)
            elif step_type == "function_call":
                call_id = _required_string(_get(step, "id"), f"{path}.id")
                name = _required_string(_get(step, "name"), f"{path}.name")
                arguments = _get(step, "arguments")
                if not isinstance(arguments, Mapping):
                    raise OutputContractError(f"{path}.arguments must be an object")
                if name not in declared:
                    raise OutputContractError(
                        f"Interactions called undeclared function {name!r}"
                    )
                calls.append(
                    FunctionToolCall(
                        call_id=call_id,
                        name=name,
                        arguments=copy_json_value(arguments, f"{path}.arguments"),
                    )
                )
            elif step_type == "user_input":
                # A stored interaction echoes its input; create returns only
                # generated steps, so an echo here is not model output.
                raise OutputContractError("Interactions reply echoed a user_input step")
            else:
                raise OutputContractError(
                    f"Interactions returned unsupported step type {step_type!r}"
                )
            output_steps.append(copy_json_value(step, path))
        if status == "requires_action" and not calls:
            raise OutputContractError(
                "Interactions status requires_action without a function call"
            )
        text = "".join(text_parts)
        empty_outcome = status in _EMPTY_OUTCOME_STATUSES
        if (
            req.reasoning.summary == "auto"
            and not empty_outcome
            and saw_signature
            and not "".join(thought_texts).strip()
        ):
            raise OutputContractError(
                "Interactions reply returned an empty requested thought summary"
            )

        prior_opaque = (
            req.session.opaque_continuation.value
            if req.session.opaque_continuation is not None
            else None
        )
        history = [*context.input_steps, *output_steps]
        opaque: str | None = None
        if saw_signature or prior_opaque is not None:
            envelope: dict[str, JSONValue] = {
                "version": _OPAQUE_VERSION,
                "steps": history,
            }
            if context.system_instruction is not None:
                envelope["system_instruction"] = context.system_instruction
            opaque = json.dumps(
                envelope,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        thought_text = "\n".join(item for item in thought_texts if item) or None
        reasoning_item = (
            ReasoningOutput(text=thought_text, opaque_roundtrip=opaque)
            if thought_text is not None or opaque is not None
            else None
        )
        structured = (
            None
            if empty_outcome or (status == "requires_action" and not text)
            else _structured_output(req, text)
        )
        return Response(
            texts=(text,),
            reasoning=(
                (reasoning_item,)
                if reasoning_item is not None or req.reasoning.summary == "auto"
                else None
            ),
            finish_reasons=(status,),
            tool_calls=tuple(calls) or None,
            structured_output=structured,
            citations=tuple(citations) or None,
            usage=_usage(_get(raw, "usage")),
            request_params=context.request_params,
            response_model=_optional_string(_get(raw, "model"), "interaction.model"),
        )

    async def _lift_stream(
        self, response: httpx.Response, context: _ExecutionContext
    ) -> Response:
        events: list[Mapping[str, object]] = []
        data_lines: list[str] = []

        def flush_event() -> None:
            if not data_lines:
                return
            payload = "\n".join(data_lines)
            data_lines.clear()
            event = _strict_json_loads(payload, "Interactions stream event")
            if not isinstance(event, dict):
                raise OutputContractError("Interactions stream event must be an object")
            events.append(cast(Mapping[str, object], event))

        async for line in response.aiter_lines():
            if not line:
                flush_event()
                continue
            if line.startswith(":"):
                continue
            name, separator, value = line.partition(":")
            if not separator:
                raise OutputContractError(
                    "Interactions stream emitted a malformed SSE line"
                )
            if name == "data":
                data_lines.append(value.removeprefix(" "))
            elif name not in {"event", "id", "retry"}:
                raise OutputContractError(
                    f"Interactions stream emitted unsupported SSE field {name!r}"
                )
        flush_event()
        return self._lift(_merge_stream(events), context)

    async def execute(self, prepared: PreparedRequest) -> Response:
        context = prepared.context
        if not isinstance(context, _InteractionsContext):
            raise DialectError("google_interactions received invalid prepared context")
        stream = context.request.scheduling.stream
        envelope_verifier = _EnvelopeVerifier(self._operation, self._model_name, stream)
        mismatch = envelope_verifier.verify_prepared(prepared)
        if mismatch is not None:
            raise DialectError(f"google_interactions prepared request: {mismatch}")
        headers = {"Content-Type": "application/json"}
        if stream:
            headers["Accept"] = "text/event-stream"
        credential = self._connection.credential
        if credential is not None:
            headers["x-goog-api-key"] = credential
        request = self._connection.client.build_request(
            "POST",
            self._endpoint_root.join(prepared.operation),
            headers=headers,
            json=prepared.thaw_body(),
        )
        expected_url = self._endpoint_root.join(self._operation)
        if request.method != "POST" or request.url != expected_url:
            raise DialectError(
                "final Interactions request URL or method does not match the "
                "bound operation"
            )
        actual_body = _strict_json_loads(
            request.content.decode("utf-8"), "final Interactions request body"
        )
        if not isinstance(actual_body, dict):
            raise DialectError("final Interactions request body must be an object")
        mismatch = envelope_verifier.verify_body(actual_body)
        if mismatch is not None:
            raise DialectError(f"final Interactions request: {mismatch}")
        raw_input = actual_body.get("input")
        if not isinstance(raw_input, list):
            raise DialectError("final Interactions request has invalid input steps")
        raw_system = actual_body.get("system_instruction")
        if raw_system is not None and not isinstance(raw_system, str):
            raise DialectError("final Interactions system_instruction must be a string")
        request_params: dict[str, JSONValue] = {
            # An HTTP envelope, not a flat body (see `request_definition`).
            REQUEST_ENVELOPE_KEY: REQUEST_ENVELOPE_HTTP,
            "method": request.method,
            "path": request.url.path,
            "body": {
                key: value
                for key, value in actual_body.items()
                if key not in {"input", "system_instruction"}
            },
        }
        execution_context = _ExecutionContext(
            request=context.request,
            request_params=request_params,
            system_instruction=raw_system,
            input_steps=tuple(cast(list[JSONValue], raw_input)),
        )
        if stream:
            response = await self._connection.client.send(request, stream=True)
            try:
                if not response.is_success:
                    await response.aread()
                raise_for_provider_status(response, "Google")
                return await self._lift_stream(response, execution_context)
            finally:
                await response.aclose()
        response = await self._connection.client.send(request)
        raise_for_provider_status(response, "Google")
        payload = _strict_json_loads(response.text, "Interactions response body")
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


def _merge_stream(events: Sequence[Mapping[str, object]]) -> dict[str, JSONValue]:
    """Fold SSE events into the Interaction object a non-streaming call returns.

    ``step.start`` opens a step, ``step.delta`` appends to it, and the terminal
    ``interaction.completed`` carries the status. Usage comes from the latest
    event that reports it, since every report is cumulative.
    """

    steps: dict[int, dict[str, object]] = {}
    arguments: dict[int, list[str]] = {}
    status: object = None
    usage: object = None
    model: object = None
    completed = False
    for event in events:
        event_type = event.get("event_type")
        if completed:
            raise OutputContractError(
                "Interactions stream continued after interaction.completed"
            )
        if event_type in {"interaction.created", "interaction.completed"}:
            interaction = event.get("interaction")
            if not isinstance(interaction, Mapping):
                raise OutputContractError(f"{event_type} carries no interaction")
            status = _get(interaction, "status", status)
            usage = _get(interaction, "usage", usage)
            model = _get(interaction, "model", model)
            if event_type == "interaction.completed":
                completed = True
                if _get(interaction, "errors") is not None:
                    raise OutputContractError(_failure_message(interaction))
        elif event_type == "interaction.status_update":
            status = event.get("status", status)
            usage = event.get("usage", usage)
        elif event_type == "step.start":
            index = event.get("index")
            step = event.get("step")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise OutputContractError("Interactions step.start index is invalid")
            if not isinstance(step, Mapping) or index in steps:
                raise OutputContractError(f"Interactions step {index} started twice")
            steps[index] = dict(cast(Mapping[str, object], step))
        elif event_type == "step.delta":
            _apply_delta(steps, arguments, event)
            metadata = event.get("metadata")
            usage = _get(metadata, "total_usage", usage)
        elif event_type == "step.stop":
            if event.get("index") not in steps:
                raise OutputContractError("Interactions step.stop for an unknown step")
            usage = event.get("usage", usage)
        elif event_type == "error":
            error = event.get("error")
            message = _get(error, "message")
            code = _get(error, "code")
            raise OutputContractError(
                "Interactions stream error"
                + (f" ({code})" if isinstance(code, str) and code else "")
                + (f": {message}" if isinstance(message, str) and message else "")
            )
        else:
            raise OutputContractError(
                f"Interactions stream emitted unsupported event {event_type!r}"
            )
    if not completed:
        raise OutputContractError("Interactions stream ended before completion")
    if sorted(steps) != list(range(len(steps))):
        raise OutputContractError("Interactions stream step indexes are not contiguous")
    for index, chunks in arguments.items():
        decoded = _strict_json_loads("".join(chunks), f"stream step {index} arguments")
        if not isinstance(decoded, dict):
            raise OutputContractError("streamed function arguments must be an object")
        steps[index]["arguments"] = decoded
    merged: dict[str, object] = {
        "status": status,
        "steps": [steps[index] for index in sorted(steps)],
    }
    if usage is not None:
        merged["usage"] = usage
    if model is not None:
        merged["model"] = model
    return cast(dict[str, JSONValue], copy_json_value(merged, "merged stream"))


def _apply_delta(
    steps: dict[int, dict[str, object]],
    arguments: dict[int, list[str]],
    event: Mapping[str, object],
) -> None:
    index = event.get("index")
    step = steps.get(index) if isinstance(index, int) else None
    if step is None:
        raise OutputContractError("Interactions step.delta for an unknown step")
    delta = event.get("delta")
    delta_type = _get(delta, "type")
    step_type = step.get("type")
    if delta_type == "text" and step_type == "model_output":
        text = _get(delta, "text")
        if not isinstance(text, str):
            raise OutputContractError("Interactions text delta must carry text")
        content = step.setdefault("content", [])
        if not isinstance(content, list):
            raise OutputContractError("Interactions model_output content is invalid")
        _append_text(cast(list[object], content), text)
    elif delta_type == "thought_signature" and step_type == "thought":
        signature = _get(delta, "signature")
        if not isinstance(signature, str):
            raise OutputContractError("Interactions thought signature must be a string")
        step["signature"] = cast(str, step.get("signature") or "") + signature
    elif delta_type == "thought_summary" and step_type == "thought":
        item = _get(delta, "content")
        summary = step.setdefault("summary", [])
        if not isinstance(summary, list):
            raise OutputContractError("Interactions thought summary is invalid")
        text = _get(item, "text")
        if _get(item, "type") == "text" and isinstance(text, str):
            _append_text(cast(list[object], summary), text)
        elif item is not None:
            cast(list[object], summary).append(item)
    elif delta_type == "arguments_delta" and step_type == "function_call":
        chunk = _get(delta, "arguments")
        if not isinstance(chunk, str):
            raise OutputContractError("Interactions arguments delta must be a string")
        arguments.setdefault(cast(int, index), []).append(chunk)
    else:
        raise OutputContractError(
            f"Interactions stream delta {delta_type!r} does not fit a "
            f"{step_type!r} step"
        )


@dataclass(frozen=True)
class _EnvelopeVerifier:
    """Check the route, bound model, and retention flag of a request."""

    operation: str
    model_name: str
    stream: bool

    def verify_prepared(self, prepared: PreparedRequest) -> str | None:
        if prepared.operation != self.operation:
            return f"operation {prepared.operation!r} does not match {self.operation!r}"
        return self.verify_body(prepared.body)

    def verify_body(self, body: Mapping[str, JSONValue]) -> str | None:
        if body.get("model") != self.model_name:
            return "model does not match the bound model"
        if body.get("store") is not False:
            return "store must be false so Google does not retain the prompt"
        for key in ("background", "previous_interaction_id", "agent"):
            if key in body:
                return f"{key!r} must not be sent"
        if (body.get("stream") is True) != self.stream:
            return "stream flag does not match the request"
        return None


@dataclass(frozen=True)
class _InputVerifier:
    expected_system: str | None
    expected_steps: tuple[JSONValue, ...]

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        if body.get("input") != list(self.expected_steps):
            return "input steps no longer match the provider-neutral input"
        if body.get("system_instruction") != self.expected_system:
            return "system_instruction no longer matches the system message"
        return None


@dataclass(frozen=True)
class _RootOptionVerifier:
    key: str
    value: JSONValue

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "value", copy_json_value(self.value, f"dialect option {self.key}")
        )

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        return (
            None
            if self.key in body and body[self.key] == self.value
            else f"dialect option {self.key!r} changed"
        )


@dataclass(frozen=True)
class _FunctionsVerifier:
    functions: tuple[Mapping[str, JSONValue], ...]

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        expected = [_function_declaration(item) for item in self.functions]
        return None if body.get("tools") == expected else "function tools changed"


@dataclass(frozen=True)
class _ToolChoiceVerifier:
    choice: JSONValue

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        if isinstance(self.choice, str):
            expected: JSONValue = {
                "auto": "auto",
                "none": "none",
                "required": "any",
            }.get(self.choice)
        else:
            nested = _get(self.choice, "function")
            name = (
                _get(nested, "name")
                if nested is not None
                else _get(self.choice, "name")
            )
            expected = {"allowed_tools": {"mode": "any", "tools": [cast(str, name)]}}
        actual = _get(body.get("generation_config"), "tool_choice")
        return None if actual == expected else "tool choice changed"


@dataclass(frozen=True)
class _StructuredFormatVerifier:
    format: str

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        response_format = body.get("response_format")
        if (
            _get(response_format, "type") != "text"
            or _get(response_format, "mime_type") != "application/json"
        ):
            return "structured output format changed"
        if self.format == "json_object" and _get(response_format, "schema") is not None:
            return "json_object output gained a schema"
        return None
