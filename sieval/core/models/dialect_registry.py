"""Stable dialect descriptors and executable PR-1 binders.

Descriptors are serializable symbol metadata.  Binders are ordinary functions
and exist only for dialect packages that are executable in the current delivery
step.  Keeping those registries separate makes a reserved dialect visible
without pretending that it can be bound.  Legacy wrapper identities also
register here so ``Model`` can recognize their exact classes without importing
those wrappers back and recreating a runtime dependency cycle.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, cast

from sieval.core.types import JSONValue

from .capabilities import (
    CAPABILITY_KEYS,
    CapabilityKey,
    DialectCapabilityDecision,
    DialectCapabilityStatus,
    Supported,
    Unsupported,
    normalize_dialect_capability_outcomes,
)
from .connection_factory import CONNECTION_FACTORY_REGISTRY
from .deployment import ConnectionPool, Deployment, RouteIntent, resolve_route
from .dialect import Dialect, active_request_leaves
from .dialects.openai_chat import (
    CAPABILITY_DECISIONS as OPENAI_CHAT_CAPABILITY_DECISIONS,
)
from .dialects.openai_chat import OpenAIChatDialect
from .dialects.openai_completions import (
    CAPABILITY_DECISIONS as OPENAI_COMPLETIONS_CAPABILITY_DECISIONS,
)
from .dialects.openai_completions import OpenAICompletionsDialect
from .ir import (
    ChatInput,
    ChatMessage,
    CompletionInput,
    ImagePart,
    Request,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    response_field_contract,
)

if TYPE_CHECKING:
    from .reconcile import RuntimeBindingPlan


_COMPAT_MODEL_INPUT_KINDS: dict[type[Any], str] = {}


def _register_compat_model_type(model_type: type[Any], input_kind: str) -> None:
    """Register one exact legacy wrapper class and its provider-neutral input kind."""

    _COMPAT_MODEL_INPUT_KINDS[model_type] = input_kind


def _compat_model_input_kind(model_type: type[Any]) -> str | None:
    """Return the input kind registered for an exact legacy wrapper class."""

    return _COMPAT_MODEL_INPUT_KINDS.get(model_type)


class DialectRegistryError(ValueError):
    """Base class for descriptor or binder lookup failures."""


class UnknownDialect(DialectRegistryError):
    """The requested identifier is outside the stable seven-dialect space."""


class DialectNotImplemented(DialectRegistryError):
    """A stable reserved dialect has no executable binder in this delivery."""


class DialectImplementationStatus(StrEnum):
    """Whether an executable binder exists in the current delivery."""

    ACTIVE = "active"
    RESERVED = "reserved"


class RequestSeedSupport(StrEnum):
    """Whether one dialect can transmit a per-request sampling seed."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    RESERVED = "reserved"


@dataclass(frozen=True)
class DialectSpec:
    """Serializable descriptor for one stable provider wire dialect."""

    dialect_id: str
    connection_family: str
    implementation_status: DialectImplementationStatus
    capability_outcomes: Mapping[CapabilityKey, DialectCapabilityStatus]
    request_seed_support: RequestSeedSupport
    input_kinds: tuple[str, ...]
    input_modalities: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.dialect_id:
            raise ValueError("DialectSpec.dialect_id must not be empty")
        if not self.connection_family:
            raise ValueError("DialectSpec.connection_family must not be empty")
        if not isinstance(self.implementation_status, DialectImplementationStatus):
            raise TypeError(
                "implementation_status must be a DialectImplementationStatus"
            )
        if not isinstance(self.request_seed_support, RequestSeedSupport):
            raise TypeError("request_seed_support must be a RequestSeedSupport")
        outcomes = normalize_dialect_capability_outcomes(
            cast(
                Mapping[str, DialectCapabilityStatus | str],
                self.capability_outcomes,
            )
        )
        if (
            self.implementation_status is DialectImplementationStatus.ACTIVE
            and DialectCapabilityStatus.RESERVED in outcomes.values()
        ):
            raise ValueError("an active dialect cannot reserve a capability outcome")
        if (
            self.implementation_status is DialectImplementationStatus.ACTIVE
            and self.request_seed_support is RequestSeedSupport.RESERVED
        ):
            raise ValueError("an active dialect cannot reserve request-seed support")
        object.__setattr__(
            self,
            "capability_outcomes",
            MappingProxyType(outcomes),
        )
        object.__setattr__(
            self,
            "input_kinds",
            _normalized_symbols(self.input_kinds, "input_kinds"),
        )
        object.__setattr__(
            self,
            "input_modalities",
            _normalized_symbols(self.input_modalities, "input_modalities"),
        )

    def to_json_value(self) -> dict[str, JSONValue]:
        """Return deterministic descriptor data with no executable objects."""

        return {
            "dialect_id": self.dialect_id,
            "connection_family": self.connection_family,
            "implementation_status": self.implementation_status.value,
            "request_seed_support": self.request_seed_support.value,
            "capability_outcomes": {
                key: self.capability_outcomes[key].value for key in CAPABILITY_KEYS
            },
            "input_kinds": list(self.input_kinds),
            "input_modalities": list(self.input_modalities),
        }


class DialectBinder(Protocol):
    """An ordinary executable function with its package's decision row."""

    capability_decisions: Mapping[str, DialectCapabilityDecision]
    compatibility_factory: Callable[..., Any] | None

    def __call__(self, connection: Any, requested_model_id: str) -> Dialect: ...


def _normalized_symbols(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if any(not isinstance(value, str) or not value for value in values):
        raise TypeError(f"{name} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(sorted(values))


def _outcomes_from_decisions(
    decisions: Mapping[str, DialectCapabilityDecision],
) -> dict[CapabilityKey, DialectCapabilityStatus]:
    if set(decisions) != set(CAPABILITY_KEYS):
        missing = sorted(set(CAPABILITY_KEYS) - set(decisions))
        extra = sorted(set(decisions) - set(CAPABILITY_KEYS))
        raise ValueError(
            f"dialect decision row incomplete: missing={missing}, extra={extra}"
        )
    request_schema = _request_schema_leaves()
    response_channels = {
        name
        for name, (role, _) in response_field_contract().items()
        if role == "channel"
    }
    request_owners: dict[str, CapabilityKey] = {}
    outcomes: dict[CapabilityKey, DialectCapabilityStatus] = {}
    for key in CAPABILITY_KEYS:
        decision = decisions[key]
        if isinstance(decision, Supported):
            binding = decision.binding
            if binding.key != key:
                raise ValueError(
                    f"dialect decision key {key!r} does not match binding key "
                    f"{binding.key!r}"
                )
            for leaf in binding.request_leaves:
                if leaf not in request_schema:
                    raise ValueError(
                        f"capability {key!r} declares unknown Request leaf {leaf!r}"
                    )
                previous = request_owners.setdefault(leaf, key)
                if previous != key:
                    raise ValueError(
                        f"Request leaf {leaf!r} is owned by both {previous!r} "
                        f"and {key!r}"
                    )
            for channel in binding.response_channels:
                if channel not in response_channels:
                    raise ValueError(
                        f"capability {key!r} declares unknown Response channel "
                        f"{channel!r}"
                    )
            outcomes[key] = DialectCapabilityStatus.SUPPORTED
        elif isinstance(decision, Unsupported):
            outcomes[key] = DialectCapabilityStatus.UNSUPPORTED
        else:
            raise TypeError(f"invalid decision for capability {key!r}")
    return outcomes


def _request_schema_leaves() -> frozenset[str]:
    """Derive the finite provider-neutral Request leaf schema mechanically."""

    completion = Request(CompletionInput("", suffix="suffix"))
    chat = Request(
        ChatInput(
            (
                ChatMessage(
                    "user",
                    (
                        TextPart("text"),
                        ImagePart(
                            url="https://schema.invalid/image",
                            media_type="image/png",
                            detail="high",
                        ),
                        ToolCallPart("call", "tool", {}),
                        ToolResultPart("call", None, is_error=True),
                    ),
                    name="schema-message",
                ),
            )
        )
    )
    leaves = set(active_request_leaves(completion))
    leaves.update(active_request_leaves(chat))
    for item in fields(Request):
        if item.name in {"input", "dialect_options"}:
            continue
        group = getattr(completion, item.name)
        leaves.update(f"{item.name}.{member.name}" for member in fields(group))
    return frozenset(leaves)


def _reserved_outcomes() -> dict[CapabilityKey, DialectCapabilityStatus]:
    return dict.fromkeys(CAPABILITY_KEYS, DialectCapabilityStatus.RESERVED)


def _spec(
    dialect_id: str,
    connection_family: str,
    *,
    decisions: Mapping[str, DialectCapabilityDecision] | None = None,
    request_seed_support: RequestSeedSupport,
    input_kinds: tuple[str, ...],
    input_modalities: tuple[str, ...],
) -> DialectSpec:
    status = (
        DialectImplementationStatus.ACTIVE
        if decisions is not None
        else DialectImplementationStatus.RESERVED
    )
    outcomes = (
        _outcomes_from_decisions(decisions)
        if decisions is not None
        else _reserved_outcomes()
    )
    return DialectSpec(
        dialect_id=dialect_id,
        connection_family=connection_family,
        implementation_status=status,
        capability_outcomes=outcomes,
        request_seed_support=request_seed_support,
        input_kinds=input_kinds,
        input_modalities=input_modalities,
    )


def _bind_openai_chat(connection: Any, requested_model_id: str) -> Dialect:
    return OpenAIChatDialect(connection, requested_model_id)


def _bind_openai_completions(connection: Any, requested_model_id: str) -> Dialect:
    return OpenAICompletionsDialect(connection, requested_model_id)


def _binder(
    function: Callable[[Any, str], Dialect],
    decisions: Mapping[str, DialectCapabilityDecision],
    *,
    compatibility_factory: Callable[..., Any] | None = None,
) -> DialectBinder:
    """Attach serializable package decisions while retaining a real function."""

    dynamic_function = cast(Any, function)
    dynamic_function.capability_decisions = decisions
    dynamic_function.compatibility_factory = compatibility_factory
    return cast(DialectBinder, function)


DIALECT_BINDERS: Mapping[str, DialectBinder] = MappingProxyType(
    {
        "openai_chat": _binder(
            _bind_openai_chat,
            OPENAI_CHAT_CAPABILITY_DECISIONS,
        ),
        "openai_completions": _binder(
            _bind_openai_completions,
            cast(
                Mapping[str, DialectCapabilityDecision],
                OPENAI_COMPLETIONS_CAPABILITY_DECISIONS,
            ),
        ),
    }
)


def _registered_decisions(
    dialect_id: str,
) -> Mapping[str, DialectCapabilityDecision] | None:
    binder = DIALECT_BINDERS.get(dialect_id)
    return None if binder is None else binder.capability_decisions


DIALECT_SPECS: Mapping[str, DialectSpec] = MappingProxyType(
    {
        "openai_chat": _spec(
            "openai_chat",
            "openai_sdk",
            decisions=_registered_decisions("openai_chat"),
            request_seed_support=RequestSeedSupport.SUPPORTED,
            input_kinds=("chat",),
            input_modalities=("text", "image", "tool_call", "tool_result"),
        ),
        "openai_completions": _spec(
            "openai_completions",
            "openai_sdk",
            decisions=_registered_decisions("openai_completions"),
            request_seed_support=RequestSeedSupport.SUPPORTED,
            input_kinds=("completion",),
            input_modalities=("text",),
        ),
        "openai_responses": _spec(
            "openai_responses",
            "openai_sdk",
            request_seed_support=RequestSeedSupport.RESERVED,
            input_kinds=("chat",),
            input_modalities=("text", "image", "tool_call", "tool_result"),
        ),
        "anthropic_messages": _spec(
            "anthropic_messages",
            "async_http_json",
            request_seed_support=RequestSeedSupport.RESERVED,
            input_kinds=("chat",),
            input_modalities=("text", "image", "tool_call", "tool_result"),
        ),
        "google_genai": _spec(
            "google_genai",
            "async_http_json",
            request_seed_support=RequestSeedSupport.RESERVED,
            input_kinds=("chat",),
            input_modalities=("text", "image", "tool_call", "tool_result"),
        ),
        "sglang_native": _spec(
            "sglang_native",
            "async_http_json",
            request_seed_support=RequestSeedSupport.RESERVED,
            input_kinds=("completion",),
            input_modalities=("text",),
        ),
        "vllm_native": _spec(
            "vllm_native",
            "async_http_json",
            request_seed_support=RequestSeedSupport.RESERVED,
            input_kinds=("completion",),
            input_modalities=("text",),
        ),
    }
)


def get_dialect_spec(dialect_id: str) -> DialectSpec:
    """Resolve one descriptor without falling back to a similar protocol."""

    try:
        return DIALECT_SPECS[dialect_id]
    except KeyError as exc:
        known = ", ".join(DIALECT_SPECS)
        raise UnknownDialect(
            f"unknown dialect {dialect_id!r}; expected one of: {known}"
        ) from exc


def dialect_is_bindable(dialect_id: str) -> bool:
    """Whether the stable dialect currently has an executable binder."""

    get_dialect_spec(dialect_id)
    return dialect_id in DIALECT_BINDERS


def compatibility_factory_for(
    dialect_id: str,
) -> Callable[..., Any] | None:
    """Return an optional one-cycle constructor adapter owned by the binder.

    Reserved dialects return ``None``.  A later binder can activate a legacy
    facade without editing that facade by registering its compatibility
    factory together with the executable binder entry.
    """

    get_dialect_spec(dialect_id)
    binder = DIALECT_BINDERS.get(dialect_id)
    return None if binder is None else binder.compatibility_factory


def capability_decisions_for(
    dialect_id: str,
) -> Mapping[CapabilityKey, DialectCapabilityDecision]:
    """Return the executable capability row or a named reserved error."""

    spec = get_dialect_spec(dialect_id)
    try:
        binder = DIALECT_BINDERS[dialect_id]
        return cast(
            Mapping[CapabilityKey, DialectCapabilityDecision],
            binder.capability_decisions,
        )
    except KeyError as exc:
        raise DialectNotImplemented(_not_implemented_message(spec.dialect_id)) from exc


def bind_dialect(
    dialect_id: str,
    requested_model_id: str,
    deployment: Deployment,
    pool: ConnectionPool[Any],
    runtime_plan: "RuntimeBindingPlan",
) -> Dialect:
    """Validate a complete runtime binding and construct its bound dialect."""

    spec = get_dialect_spec(dialect_id)
    try:
        binder = DIALECT_BINDERS[dialect_id]
    except KeyError as exc:
        raise DialectNotImplemented(_not_implemented_message(spec.dialect_id)) from exc

    if not requested_model_id:
        raise ValueError("requested_model_id must not be empty")
    if runtime_plan.dialect_id != dialect_id:
        raise DialectRegistryError(
            "runtime plan dialect mismatch: "
            f"{runtime_plan.dialect_id!r} != {dialect_id!r}"
        )
    if runtime_plan.requested_model_id != requested_model_id:
        raise DialectRegistryError(
            "runtime plan requested-model mismatch: "
            f"{runtime_plan.requested_model_id!r} != {requested_model_id!r}"
        )
    if runtime_plan.deployment_fingerprint != deployment.fingerprint:
        raise DialectRegistryError("runtime plan targets another deployment snapshot")
    if runtime_plan.resolved_route.connection_family != spec.connection_family:
        raise DialectRegistryError("runtime plan route has the wrong connection family")

    expected_route = resolve_route(
        deployment,
        dialect_id,
        spec.connection_family,
        RouteIntent(runtime_plan.resolved_route.service_role),
    )
    if expected_route != runtime_plan.resolved_route:
        raise DialectRegistryError("runtime plan route does not match the deployment")

    pool.verify_route(runtime_plan.resolved_route)
    pool.verify_identity(runtime_plan.connection_identity)
    dialect = binder(pool.connection, requested_model_id)
    if dialect.dialect_id != dialect_id:
        raise DialectRegistryError("binder returned a dialect with the wrong id")
    if dialect.connection_family != spec.connection_family:
        raise DialectRegistryError(
            "binder returned a dialect with the wrong connection family"
        )
    return dialect


def _not_implemented_message(dialect_id: str) -> str:
    if dialect_id == "sglang_native":
        return (
            "dialect 'sglang_native' has no PR-1 binder; the one-cycle SGLang "
            "compatibility path remains an explicit legacy bypass"
        )
    if dialect_id == "vllm_native":
        return "dialect 'vllm_native' is reserved and explicitly deferred after #25"
    return f"dialect {dialect_id!r} is reserved for a later #25 adapter PR"


def dialect_registry_to_json() -> dict[str, JSONValue]:
    """Serialize the complete stable descriptor registry deterministically."""

    return {
        dialect_id: cast(JSONValue, DIALECT_SPECS[dialect_id].to_json_value())
        for dialect_id in DIALECT_SPECS
    }


if any(
    DIALECT_SPECS[dialect_id].implementation_status
    is not DialectImplementationStatus.ACTIVE
    for dialect_id in DIALECT_BINDERS
):
    raise RuntimeError("every executable binder must have an active descriptor")

_missing_connection_families = {
    spec.connection_family for spec in DIALECT_SPECS.values()
} - CONNECTION_FACTORY_REGISTRY.families
if _missing_connection_families:
    raise RuntimeError(
        "dialect descriptors reference unregistered connection families: "
        + ", ".join(sorted(_missing_connection_families))
    )
