"""Typed capability declarations for the provider-neutral Model IR.

Capabilities are stable string keys with typed configuration.  They are joined
with caller-supplied dialect and model outcomes; this module deliberately owns
neither a dialect registry nor serving-condition algebra.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import MISSING, dataclass, field, fields
from enum import Enum, StrEnum, auto
from types import MappingProxyType
from typing import Literal, cast

from sieval.core.types import JSONValue

from ._shared import copy_json_value, validate_nonempty_string

type CapabilityKey = Literal[
    "input_scoring",
    "sampled_logprobs",
    "top_logprobs",
    "reasoning",
    "function_tools",
    "hosted_tools",
    "structured_output",
    "stateful_session",
    "opaque_continuation",
    "multimodal_input",
    "prefill",
    "fim",
]

CAPABILITY_KEYS: tuple[CapabilityKey, ...] = (
    "input_scoring",
    "sampled_logprobs",
    "top_logprobs",
    "reasoning",
    "function_tools",
    "hosted_tools",
    "structured_output",
    "stateful_session",
    "opaque_continuation",
    "multimodal_input",
    "prefill",
    "fim",
)


_LEGACY_ARGUMENT_CAPABILITIES: Mapping[str, tuple[CapabilityKey, ...]] = (
    MappingProxyType(
        {
            "echo": ("input_scoring",),
            "score_input": ("input_scoring",),
            "return_logprobs": ("sampled_logprobs",),
            "top_logprobs": ("top_logprobs",),
            "reasoning": ("reasoning",),
            "reasoning_effort": ("reasoning",),
            "tools": ("function_tools",),
            "tool_choice": ("function_tools",),
            "parallel_tool_calls": ("function_tools",),
            "server_tools": ("hosted_tools",),
            "response_format": ("structured_output",),
            "previous_response_id": ("stateful_session",),
            "session_id": ("stateful_session",),
            "opaque_continuation": ("opaque_continuation",),
            "prefill": ("prefill",),
            "prefix": ("prefill",),
            "suffix": ("fim",),
        }
    )
)

_LEGACY_WIRE_CONTAINERS = frozenset({"extra_body", "extra_wire_params"})


class CapabilityConfigError(ValueError):
    """A capability declaration or outcome mapping is invalid."""


def legacy_capability_ambiguities(
    declarations: Mapping[str, object],
    legacy_arguments: Mapping[str, object],
) -> dict[CapabilityKey, tuple[str, ...]]:
    """Find semantics expressed by both canonical and legacy config surfaces.

    The migration rule is presence-based: even two equal values or an explicit
    ``false`` have two owners and would make later precedence significant.
    Ordinary sampling arguments are intentionally absent from the mapping.
    ``logprobs`` is value-sensitive because the legacy completion spelling can
    carry both the sampled-logprob switch and an alternative-token breadth.
    """

    canonical = set(declarations) & set(CAPABILITY_KEYS)
    paths: dict[CapabilityKey, set[str]] = {}

    def inspect(name: object, value: object, path: str) -> None:
        if not isinstance(name, str):
            return
        capabilities = _LEGACY_ARGUMENT_CAPABILITIES.get(name, ())
        if name == "logprobs":
            capabilities = ("sampled_logprobs",)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                capabilities += ("top_logprobs",)
        elif (
            name == "top_logprobs"
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
        ):
            capabilities += ("sampled_logprobs",)
        for capability in capabilities:
            if capability in canonical:
                paths.setdefault(capability, set()).add(path)

    for name, value in legacy_arguments.items():
        inspect(name, value, str(name))
        if name not in _LEGACY_WIRE_CONTAINERS or not isinstance(value, Mapping):
            continue
        for nested_name, nested_value in value.items():
            inspect(nested_name, nested_value, f"{name}.{nested_name}")

    return {
        capability: tuple(sorted(argument_paths))
        for capability, argument_paths in sorted(paths.items())
    }


def validate_no_legacy_capability_ambiguity(
    declarations: Mapping[str, object],
    legacy_arguments: Mapping[str, object],
    *,
    canonical_source: str,
    legacy_source: str,
) -> None:
    """Reject a canonical/legacy double declaration before runtime binding."""

    ambiguities = legacy_capability_ambiguities(declarations, legacy_arguments)
    if not ambiguities:
        return
    detail = "; ".join(
        f"{capability} via {', '.join(paths)}"
        for capability, paths in ambiguities.items()
    )
    raise CapabilityConfigError(
        f"{canonical_source} and {legacy_source} both express canonical "
        f"capability semantics ({detail}); remove the legacy argument(s) or "
        "the canonical declaration"
    )


def _validate_string_tuple(
    value: object,
    name: str,
    *,
    allowed: frozenset[str] | None = None,
) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple of strings")
    seen: set[str] = set()
    for item in value:
        validate_nonempty_string(item, f"{name} item")
        assert isinstance(item, str)
        if item in seen:
            raise ValueError(f"{name} contains duplicate value {item!r}")
        if allowed is not None and item not in allowed:
            expected = ", ".join(sorted(allowed))
            raise ValueError(f"{name} value {item!r} is not one of: {expected}")
        seen.add(item)


@dataclass(frozen=True)
class InputScoringOptions:
    """Configuration for complete input-token scoring."""


@dataclass(frozen=True)
class SampledLogprobsOptions:
    """Configuration for sampled-output token log probabilities."""


@dataclass(frozen=True)
class TopLogprobsOptions:
    """Minimum alternative-token breadth requested by a caller."""

    minimum: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.minimum, bool) or not isinstance(self.minimum, int):
            raise TypeError("minimum must be an integer")
        if self.minimum < 1:
            raise ValueError("minimum must be >= 1")


@dataclass(frozen=True)
class ReasoningOptions:
    """Provider-neutral reasoning controls and visible-summary preference."""

    effort: str | None = None
    budget_tokens: int | None = None
    summary: str | None = None

    def __post_init__(self) -> None:
        if self.effort is not None:
            validate_nonempty_string(self.effort, "effort")
        if self.budget_tokens is not None:
            if isinstance(self.budget_tokens, bool) or not isinstance(
                self.budget_tokens, int
            ):
                raise TypeError("budget_tokens must be an integer")
            if self.budget_tokens < 1:
                raise ValueError("budget_tokens must be >= 1")
        if self.effort is not None and self.budget_tokens is not None:
            raise ValueError("effort and budget_tokens are mutually exclusive")
        if self.summary is not None:
            validate_nonempty_string(self.summary, "summary")
            if self.summary not in {"none", "auto", "concise", "detailed"}:
                raise ValueError(
                    "summary must be one of: auto, concise, detailed, none"
                )


@dataclass(frozen=True)
class FunctionToolsOptions:
    """Function-tool controls shared by dialects."""

    parallel: bool | None = None

    def __post_init__(self) -> None:
        if self.parallel is not None and not isinstance(self.parallel, bool):
            raise TypeError("parallel must be a boolean")


@dataclass(frozen=True)
class HostedToolsOptions:
    """Kinds of provider-hosted tools requested by the user."""

    kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_string_tuple(self.kinds, "kinds")


_STRUCTURED_OUTPUT_FORMATS = frozenset({"json_object", "json_schema"})


@dataclass(frozen=True)
class StructuredOutputOptions:
    """Structured formats that must be accepted end to end."""

    formats: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_string_tuple(
            self.formats,
            "formats",
            allowed=_STRUCTURED_OUTPUT_FORMATS,
        )


@dataclass(frozen=True)
class StatefulSessionOptions:
    """Configuration for provider/session response-id continuation."""


@dataclass(frozen=True)
class OpaqueContinuationOptions:
    """Configuration for round-tripping provider-originated opaque state."""


_MULTIMODAL_INPUT_MODALITIES = frozenset({"image"})


@dataclass(frozen=True)
class MultimodalInputOptions:
    """Non-text input modalities with executable shared-IR support."""

    modalities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_string_tuple(
            self.modalities,
            "modalities",
            allowed=_MULTIMODAL_INPUT_MODALITIES,
        )


@dataclass(frozen=True)
class PrefillOptions:
    """Configuration for a dialect-native assistant prefill."""


@dataclass(frozen=True)
class FimOptions:
    """Configuration for prefix/suffix fill-in-the-middle input."""


type CapabilityOptions = (
    InputScoringOptions
    | SampledLogprobsOptions
    | TopLogprobsOptions
    | ReasoningOptions
    | FunctionToolsOptions
    | HostedToolsOptions
    | StructuredOutputOptions
    | StatefulSessionOptions
    | OpaqueContinuationOptions
    | MultimodalInputOptions
    | PrefillOptions
    | FimOptions
)


@dataclass(frozen=True)
class CapabilitySpec:
    """Stable declaration key and the dataclass used to normalize its options."""

    key: CapabilityKey
    options_type: type[CapabilityOptions]


CAPABILITY_SPECS: Mapping[CapabilityKey, CapabilitySpec] = MappingProxyType(
    {
        "input_scoring": CapabilitySpec("input_scoring", InputScoringOptions),
        "sampled_logprobs": CapabilitySpec("sampled_logprobs", SampledLogprobsOptions),
        "top_logprobs": CapabilitySpec("top_logprobs", TopLogprobsOptions),
        "reasoning": CapabilitySpec("reasoning", ReasoningOptions),
        "function_tools": CapabilitySpec("function_tools", FunctionToolsOptions),
        "hosted_tools": CapabilitySpec("hosted_tools", HostedToolsOptions),
        "structured_output": CapabilitySpec(
            "structured_output", StructuredOutputOptions
        ),
        "stateful_session": CapabilitySpec("stateful_session", StatefulSessionOptions),
        "opaque_continuation": CapabilitySpec(
            "opaque_continuation", OpaqueContinuationOptions
        ),
        "multimodal_input": CapabilitySpec("multimodal_input", MultimodalInputOptions),
        "prefill": CapabilitySpec("prefill", PrefillOptions),
        "fim": CapabilitySpec("fim", FimOptions),
    }
)

_SEQUENCE_OPTIONS: frozenset[tuple[CapabilityKey, str]] = frozenset(
    {
        ("hosted_tools", "kinds"),
        ("structured_output", "formats"),
        ("multimodal_input", "modalities"),
    }
)


class DialectCapabilityStatus(StrEnum):
    """Descriptor-level outcome for one dialect/capability cell."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    RESERVED = "reserved"


type NormalizedCapabilityValue = CapabilityOptions | Literal[False]


def normalize_dialect_capability_outcomes(
    outcomes: Mapping[str, DialectCapabilityStatus | str],
) -> dict[CapabilityKey, DialectCapabilityStatus]:
    """Validate and normalize a complete selected-dialect outcome row."""
    unknown = sorted(set(outcomes) - set(CAPABILITY_KEYS))
    if unknown:
        raise CapabilityConfigError(
            f"unknown capability outcome key(s): {', '.join(unknown)}"
        )
    missing = sorted(set(CAPABILITY_KEYS) - set(outcomes))
    if missing:
        raise CapabilityConfigError(
            f"missing capability outcome(s): {', '.join(missing)}"
        )

    normalized: dict[CapabilityKey, DialectCapabilityStatus] = {}
    for key in CAPABILITY_KEYS:
        value = outcomes[key]
        try:
            normalized[key] = DialectCapabilityStatus(value)
        except (TypeError, ValueError) as exc:
            raise CapabilityConfigError(
                f"invalid outcome for capability {key!r}: {value!r}"
            ) from exc
    return normalized


def _parse_options(
    spec: CapabilitySpec,
    raw: Mapping[object, object],
) -> CapabilityOptions:
    field_names = {item.name for item in fields(spec.options_type)}
    nonstring = [key for key in raw if not isinstance(key, str)]
    if nonstring:
        raise CapabilityConfigError(
            f"capability {spec.key!r} option names must be strings"
        )
    unknown = sorted(cast(str, key) for key in raw if key not in field_names)
    if unknown:
        raise CapabilityConfigError(
            f"unknown option(s) for capability {spec.key!r}: {', '.join(unknown)}"
        )

    kwargs: dict[str, object] = {}
    for untyped_name, value in raw.items():
        name = cast(str, untyped_name)
        if (spec.key, name) in _SEQUENCE_OPTIONS:
            if not isinstance(value, (list, tuple)) or isinstance(value, str):
                raise CapabilityConfigError(
                    f"{spec.key}.{name} must be a sequence of strings"
                )
            value = tuple(value)
        kwargs[name] = value

    try:
        constructor = cast(Callable[..., CapabilityOptions], spec.options_type)
        return constructor(**kwargs)
    except (TypeError, ValueError) as exc:
        raise CapabilityConfigError(f"invalid {spec.key} options: {exc}") from exc


def _normalize_capability_value(
    spec: CapabilitySpec,
    raw: object,
) -> NormalizedCapabilityValue:
    if raw is None:
        raise CapabilityConfigError(
            f"capability {spec.key!r} cannot be null; omit it or use false"
        )
    if raw is False:
        return False
    if raw is True:
        return spec.options_type()
    if not isinstance(raw, Mapping):
        raise CapabilityConfigError(
            f"capability {spec.key!r} must be false, true, or a mapping"
        )
    return _parse_options(spec, cast(Mapping[object, object], raw))


def normalize_capability_declarations(
    raw: object,
    *,
    dialect_id: str,
    outcomes: Mapping[str, DialectCapabilityStatus | str],
) -> dict[CapabilityKey, NormalizedCapabilityValue]:
    """Normalize one model's capability subtree for its selected dialect.

    A missing capability is represented by an absent mapping key.  Explicit
    ``false`` is retained; ``true`` and an empty mapping both construct the
    typed option dataclass with its defaults.
    """
    validate_nonempty_string(dialect_id, "dialect_id")
    if not isinstance(raw, Mapping):
        raise CapabilityConfigError("capabilities must be a mapping")
    raw_mapping = cast(Mapping[object, object], raw)
    normalized_outcomes = normalize_dialect_capability_outcomes(outcomes)

    nonstring = [key for key in raw_mapping if not isinstance(key, str)]
    if nonstring:
        raise CapabilityConfigError("capability names must be strings")
    unknown = sorted(
        cast(str, key) for key in raw_mapping if key not in CAPABILITY_SPECS
    )
    if unknown:
        raise CapabilityConfigError(f"unknown capability key(s): {', '.join(unknown)}")

    normalized: dict[CapabilityKey, NormalizedCapabilityValue] = {}
    for key in CAPABILITY_KEYS:
        if key not in raw_mapping:
            continue
        status = normalized_outcomes[key]
        if status is DialectCapabilityStatus.RESERVED:
            raise CapabilityConfigError(
                f"capability {key!r} is reserved for dialect {dialect_id!r}"
            )
        value = _normalize_capability_value(CAPABILITY_SPECS[key], raw_mapping[key])
        if status is DialectCapabilityStatus.UNSUPPORTED and value is not False:
            raise CapabilityConfigError(
                f"dialect {dialect_id!r} does not support capability {key!r}"
            )
        normalized[key] = value
    return normalized


def _same_json_value(left: JSONValue, right: JSONValue) -> bool:
    """Compare JSON values without collapsing distinct JSON scalar types."""
    return json.dumps(
        copy_json_value(left, "left"),
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ) == json.dumps(
        copy_json_value(right, "right"),
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _options_to_json(options: CapabilityOptions) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {}
    for item in fields(options):
        value = getattr(options, item.name)
        if item.default is not MISSING and value == item.default:
            continue
        if value is None:
            continue
        result[item.name] = copy_json_value(value, item.name)
    return result


def capability_declarations_to_json(
    declarations: Mapping[CapabilityKey, NormalizedCapabilityValue],
) -> dict[str, JSONValue]:
    """Return the canonical JSON-value form of normalized declarations."""
    result: dict[str, JSONValue] = {}
    for key in sorted(declarations):
        if key not in CAPABILITY_SPECS:
            raise CapabilityConfigError(f"unknown capability key: {key!r}")
        value = declarations[key]
        if value is False:
            result[key] = False
            continue
        spec = CAPABILITY_SPECS[key]
        if not isinstance(value, spec.options_type):
            raise CapabilityConfigError(
                f"capability {key!r} has options for the wrong capability"
            )
        result[key] = _options_to_json(value)
    return result


def canonical_capability_json(
    declarations: Mapping[CapabilityKey, NormalizedCapabilityValue],
) -> str:
    """Serialize normalized declarations deterministically for evidence hashes."""
    return json.dumps(
        capability_declarations_to_json(declarations),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class RequestDefaults:
    """Serializable model-level defaults keyed by provider-neutral leaf path."""

    values: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        copied = copy_json_value(self.values, "request_defaults")
        assert isinstance(copied, dict)
        object.__setattr__(self, "values", MappingProxyType(copied))

    def to_json_value(self) -> dict[str, JSONValue]:
        """Return a detached JSON-compatible mapping."""
        copied = copy_json_value(self.values, "request_defaults")
        assert isinstance(copied, dict)
        return cast(dict[str, JSONValue], copied)


@dataclass(frozen=True)
class CapabilityIntent:
    """One normalized semantic requirement without serving-policy decisions."""

    key: CapabilityKey
    required: bool
    minimums: Mapping[str, JSONValue] = field(default_factory=dict)
    request_defaults: RequestDefaults = field(default_factory=RequestDefaults)
    sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.key not in CAPABILITY_SPECS:
            raise ValueError(f"unknown capability key: {self.key!r}")
        if not isinstance(self.required, bool):
            raise TypeError("required must be a boolean")
        copied = copy_json_value(self.minimums, "minimums")
        assert isinstance(copied, dict)
        object.__setattr__(self, "minimums", MappingProxyType(copied))
        _validate_string_tuple(self.sources, "sources")
        object.__setattr__(self, "sources", tuple(sorted(self.sources)))

    def to_json_value(self) -> dict[str, JSONValue]:
        """Return deterministic record data suitable for plan serialization."""
        minimums = copy_json_value(self.minimums, "minimums")
        assert isinstance(minimums, dict)
        return {
            "key": self.key,
            "required": self.required,
            "minimums": minimums,
            "request_defaults": self.request_defaults.to_json_value(),
            "sources": list(self.sources),
        }


def legacy_capability_intents(
    legacy_arguments: Mapping[str, object],
    *,
    source: str,
) -> dict[CapabilityKey, CapabilityIntent]:
    """Project active legacy request defaults onto the capability plane.

    This is a migration adapter, not a second declaration vocabulary.  It
    records only the semantic capability and any correctness minimum needed by
    reconciliation; the original argument remains the request builder's value
    source.  Ordinary sampling controls deliberately produce no intent.
    """

    validate_nonempty_string(source, "legacy intent source")
    projected: list[CapabilityIntent] = []

    def add(
        capability: CapabilityKey,
        path: str,
        *,
        minimums: Mapping[str, JSONValue] | None = None,
    ) -> None:
        projected.append(
            CapabilityIntent(
                capability,
                True,
                minimums={} if minimums is None else minimums,
                sources=(f"{source}.{path}",),
            )
        )

    def inspect(name: object, value: object, path: str) -> None:
        if not isinstance(name, str):
            return

        if name in {"echo", "score_input", "return_logprobs"}:
            if value is not False and value is not None:
                add(
                    "input_scoring"
                    if name in {"echo", "score_input"}
                    else "sampled_logprobs",
                    path,
                )
            return

        if name == "logprobs":
            if value is False or value is None:
                return
            add("sampled_logprobs", path)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                add("top_logprobs", path, minimums={"minimum": value})
            elif not isinstance(value, bool | int):
                # Invalid values still express an attempted breadth request;
                # the request builder retains responsibility for its type
                # diagnostic, while setup must not miss the capability.
                add("top_logprobs", path)
            return

        if name == "top_logprobs":
            if value in (None, False, 0):
                return
            add("sampled_logprobs", path)
            minimums = (
                {"minimum": value}
                if isinstance(value, int) and not isinstance(value, bool) and value > 0
                else None
            )
            add("top_logprobs", path, minimums=minimums)
            return

        capability_values = _LEGACY_ARGUMENT_CAPABILITIES.get(name, ())
        if not capability_values or value is None:
            return
        if name in {"tools", "server_tools"} and not value:
            return
        for capability in capability_values:
            add(capability, path)

    for name, value in legacy_arguments.items():
        inspect(name, value, str(name))
        if name not in _LEGACY_WIRE_CONTAINERS or not isinstance(value, Mapping):
            continue
        for nested_name, nested_value in value.items():
            inspect(nested_name, nested_value, f"{name}.{nested_name}")

    return aggregate_capability_intents(projected)


def aggregate_capability_intents(
    intents: list[CapabilityIntent] | tuple[CapabilityIntent, ...],
) -> dict[CapabilityKey, CapabilityIntent]:
    """OR/max-aggregate already-normalized intents while retaining sources."""

    grouped: dict[CapabilityKey, list[CapabilityIntent]] = {}
    for intent in intents:
        if not isinstance(intent, CapabilityIntent):
            raise TypeError("intents must contain CapabilityIntent values")
        grouped.setdefault(intent.key, []).append(intent)

    result: dict[CapabilityKey, CapabilityIntent] = {}
    for capability in CAPABILITY_KEYS:
        values = grouped.get(capability)
        if not values:
            continue
        minimums: dict[str, JSONValue] = {}
        defaults: dict[str, JSONValue] = {}
        sources: set[str] = set()
        for intent in values:
            sources.update(intent.sources)
            for name, value in intent.minimums.items():
                if name not in minimums:
                    minimums[name] = value
                    continue
                previous = minimums[name]
                if (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and isinstance(previous, int)
                    and not isinstance(previous, bool)
                ):
                    minimums[name] = max(previous, value)
                elif _same_json_value(previous, value):
                    minimums[name] = value
                else:
                    raise CapabilityConfigError(
                        f"legacy capability {capability!r} has incompatible "
                        f"minimum {name!r} from {', '.join(sorted(sources))}"
                    )
            for path, value in intent.request_defaults.values.items():
                if path in defaults and not _same_json_value(defaults[path], value):
                    raise CapabilityConfigError(
                        f"legacy capability {capability!r} has competing request "
                        f"default {path!r}"
                    )
                defaults[path] = value
        result[capability] = CapabilityIntent(
            capability,
            any(intent.required for intent in values),
            minimums=minimums,
            request_defaults=RequestDefaults(defaults),
            sources=tuple(sorted(sources)),
        )
    return result


class ModelCapabilityStatus(StrEnum):
    """A model-profile claim, independent of dialect and serving state."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ModelCapabilityEntry:
    """One sourced model-profile outcome."""

    status: ModelCapabilityStatus
    source: str
    reason: str | None = None
    verifier: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ModelCapabilityStatus):
            raise TypeError("status must be a ModelCapabilityStatus")
        validate_nonempty_string(self.source, "source")
        if self.reason is not None:
            validate_nonempty_string(self.reason, "reason")
        if self.verifier is not None:
            validate_nonempty_string(self.verifier, "verifier")
        if self.status is ModelCapabilityStatus.UNSUPPORTED and self.reason is None:
            raise ValueError("unsupported model capability requires a reason")
        if self.status is ModelCapabilityStatus.UNKNOWN and self.verifier is None:
            raise ValueError("unknown model capability requires a verifier")


@dataclass(frozen=True)
class ModelCapabilityProfile:
    """Sourced capability outcomes for one checkpoint or hosted model."""

    entries: Mapping[CapabilityKey, ModelCapabilityEntry]
    authoritative: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.authoritative, bool):
            raise TypeError("authoritative must be a boolean")
        copied: dict[CapabilityKey, ModelCapabilityEntry] = {}
        for key, entry in self.entries.items():
            if key not in CAPABILITY_SPECS:
                raise ValueError(f"unknown capability key: {key!r}")
            if not isinstance(entry, ModelCapabilityEntry):
                raise TypeError(f"profile entry {key!r} must be ModelCapabilityEntry")
            copied[key] = entry
        object.__setattr__(self, "entries", MappingProxyType(copied))

    def to_json_value(self) -> dict[str, JSONValue]:
        """Return deterministically ordered model-profile data."""
        entries: dict[str, JSONValue] = {}
        for key in sorted(self.entries):
            entry = self.entries[key]
            value: dict[str, JSONValue] = {
                "status": entry.status.value,
                "source": entry.source,
            }
            if entry.reason is not None:
                value["reason"] = entry.reason
            if entry.verifier is not None:
                value["verifier"] = entry.verifier
            entries[key] = value
        return {"authoritative": self.authoritative, "entries": entries}


type CapabilityConfigValidator = Callable[[CapabilityOptions], None]


def _accept_capability_config(options: CapabilityOptions) -> None:
    del options


@dataclass(frozen=True)
class DialectCapabilityBinding:
    """Executable request/output projection for one dialect capability."""

    key: CapabilityKey
    request_leaves: tuple[str, ...] = ()
    response_channels: tuple[str, ...] = ()
    _config_validator: CapabilityConfigValidator = field(
        default=_accept_capability_config,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.key not in CAPABILITY_SPECS:
            raise ValueError(f"unknown capability key: {self.key!r}")
        _validate_string_tuple(self.request_leaves, "request_leaves")
        _validate_string_tuple(self.response_channels, "response_channels")
        if any("*" in path for path in self.request_leaves):
            raise ValueError("request_leaves cannot contain wildcards")
        if not callable(self._config_validator):
            raise TypeError("config validator must be callable")

    def validate_config(self, options: CapabilityOptions) -> None:
        """Validate normalized options against this dialect-specific binding."""
        expected = CAPABILITY_SPECS[self.key].options_type
        if not isinstance(options, expected):
            raise CapabilityConfigError(
                f"binding {self.key!r} received options for another capability"
            )
        self._config_validator(options)


@dataclass(frozen=True)
class Supported:
    """An executable dialect capability binding."""

    binding: DialectCapabilityBinding


@dataclass(frozen=True)
class Unsupported:
    """An explicit dialect capability absence with an actionable reason."""

    reason: str

    def __post_init__(self) -> None:
        validate_nonempty_string(self.reason, "reason")


type DialectCapabilityDecision = Supported | Unsupported


class Capability(Enum):
    """Deprecated non-composable namespace kept during the PR-1 migration.

    Canonical code uses :data:`CAPABILITY_SPECS` string keys.  This ordinary
    ``Enum`` intentionally makes legacy ``Capability.A | Capability.B`` fail.
    """

    Completion = auto()
    Chat = auto()
    FunctionCalling = auto()
    ServerTools = auto()
    Reasoning = auto()
    ReasoningEffort = auto()
    TopKLogprobs = auto()
    InputScoring = auto()
    SampledLogprobs = auto()
    SampledLogprobsWithTokenIds = auto()
    StructuredOutput = auto()
    Prefill = auto()
    FIM = auto()
