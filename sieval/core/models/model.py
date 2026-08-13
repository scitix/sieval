"""Bound model orchestration and one-cycle compatibility bridges.

``Model`` is the provider-neutral composition root.  It borrows the connection
owned by a :class:`ConnectionPool`, runs the residual request audit before
admission, delegates wire work to a bound :class:`Dialect`, validates the
response contract, and attaches immutable provenance.  It never creates or
closes a provider client.

The legacy ``agenerate``/``alogprobs`` builders remain for one compatibility
cycle.  Their defaults are deliberately separate from the reconciled runtime
plan: ``with_args`` can change only those builder defaults, schema-external
metadata, and the model-local concurrency cap.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

import copy
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from types import TracebackType
from typing import Any, NotRequired, Self, TypedDict, cast
from uuid import uuid4

import anyio
from openai import AsyncOpenAI

from sieval.core.types import JSONValue
from sieval.core.utils.serialization import sieval_record

from ._fingerprint import fingerprint_mapping
from .capabilities import (
    Capability,
    RequestDefaults,
    Supported,
)
from .deployment import (
    BINDING_RESOURCE_KEYS,
    ConnectionIdentity,
    ConnectionPool,
    Deployment,
    Engine,
    ServingFacts,
    resolve_route,
)
from .dialect import (
    Dialect,
    DialectError,
    RequestAudit,
    active_request_capabilities,
    active_request_leaves,
    validate_request_invariants,
    validate_runtime_binding_plan,
)
from .dialect_registry import (
    _compat_model_input_kind,
    bind_dialect,
    capability_decisions_for,
    get_dialect_spec,
)
from .exceptions import CapabilityError
from .ir import (
    CapabilityEvidence,
    ChatInput,
    CompletionInput,
    DialectOptions,
    ModelIdentity,
    ModelInput,
    ModelProvenance,
    OpaqueContinuation,
    ReasoningParams,
    Request,
    Response,
    SamplingParams,
    SchedulingParams,
    ScoringParams,
    SessionParams,
    StructuredOutputParams,
    TokenLogprob,
    ToolParams,
)
from .reconcile import RuntimeBindingPlan


class ModelUsage(TypedDict):
    """Token usage statistics from a single model API call."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


class ModelMeta(TypedDict):
    """Persisted identity and defaults for one model call."""

    model: str
    api_base: str | None
    default_params: dict[str, JSONValue]
    extra: NotRequired[dict[str, JSONValue]]
    provenance: NotRequired[ModelProvenance]


class ModelCallMeta(TypedDict):
    """Per-API-call metadata: model info, usage, params, finish reasons."""

    model: ModelMeta
    usage: NotRequired[ModelUsage]
    request_params: NotRequired[dict[str, JSONValue]]
    finish_reasons: NotRequired[list[str]]
    response_model: NotRequired[str]
    system_fingerprint: NotRequired[str | None]


class ModelQuotaSnapshot(TypedDict):
    """Snapshot of one concurrency limiter."""

    available: int
    total: int


class ModelQuotaInfo(TypedDict):
    """Combined shared and model-local limiter quota information."""

    available: int | float
    total: int | float
    parent: ModelQuotaSnapshot | None
    child: ModelQuotaSnapshot | None


@sieval_record
@dataclass
class ModelOutput:
    """Legacy return type preserved while tasks migrate to ``Response``."""

    model: ModelMeta
    texts: list[str]
    finish_reasons: list[str] | None = None
    reasoning_texts: list[str] | None = None
    logprobs_tokens: list[str] | None = None
    logprobs: list[float | None] | None = None
    top_logprobs: list[dict[str, float]] | None = None
    usage: ModelUsage | None = None
    request_params: dict[str, JSONValue] | None = None
    response_model: str | None = None
    system_fingerprint: str | None = None


@dataclass(frozen=True)
class _LegacyOpenAIBinding:
    deployment: Deployment
    pool: ConnectionPool[Any]
    runtime_plan: RuntimeBindingPlan
    local_limiter: anyio.CapacityLimiter | None
    parent_limiter: anyio.CapacityLimiter | None


_LEGACY_CAPABILITY_MAP: Mapping[str, tuple[Capability, ...]] = {
    "fim": (Capability.FIM,),
    "function_tools": (Capability.FunctionCalling,),
    "hosted_tools": (Capability.ServerTools,),
    "input_scoring": (Capability.InputScoring,),
    "prefill": (Capability.Prefill,),
    "reasoning": (Capability.Reasoning, Capability.ReasoningEffort),
    "sampled_logprobs": (Capability.SampledLogprobs,),
    "structured_output": (Capability.StructuredOutput,),
    "top_logprobs": (Capability.TopKLogprobs,),
}

_RESPONSE_CHANNEL_BY_CAPABILITY: Mapping[str, str] = {
    "function_tools": "tool_calls",
    "hosted_tools": "server_tool_uses",
    "input_scoring": "input_scoring",
    "opaque_continuation": "reasoning",
    "reasoning": "reasoning",
    "sampled_logprobs": "logprobs",
    "stateful_session": "session_id",
    "structured_output": "structured_output",
    "top_logprobs": "top_logprobs",
}

_REQUEST_CHECK_VERIFIERS = frozenset({"validate_response_channel"})
_REMOVED_SUBCLASS_HOOKS = frozenset({"_agenerate_impl", "_alogprobs_impl"})


def _named_json_value(value: object, name: str) -> JSONValue:
    """Validate and detach a JSON value, naming the offending leaf on failure.

    Sequences are ``list``/``tuple`` only: the result is persisted, and a
    ``set`` would serialize in hash order while a generator would serialize
    as ``[]`` once consumed.
    """
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must not contain a non-finite float")
        return value
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, Mapping):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} keys must be strings")
            result[key] = _named_json_value(item, f"{name}.{key}")
        return result
    if isinstance(value, list | tuple):
        return [_named_json_value(item, name) for item in value]
    raise TypeError(f"{name} must be JSON-compatible, got {type(value).__name__}")


def _checked_builder_defaults(values: Mapping[str, object]) -> dict[str, object]:
    """Reject builder defaults that ``meta()`` could not persist.

    ``meta()`` runs once per response, so an unpersistable default would
    otherwise raise only after a call had been billed. Values are stored
    unconverted -- the request builders need them as given.
    """
    for key, value in values.items():
        _named_json_value(value, f"default_params.{key}")
    return dict(values)


def _optional_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _pop_compatible_alias(
    values: dict[str, object],
    canonical: str,
    legacy: str,
    *,
    default: object,
) -> object:
    """Pop one semantic value without silently resolving two conflicting owners."""

    missing = object()
    canonical_value = values.pop(canonical, missing)
    legacy_value = values.pop(legacy, missing)
    if (
        canonical_value is not missing
        and legacy_value is not missing
        and canonical_value is not None
        and legacy_value is not None
        and canonical_value != legacy_value
    ):
        raise ValueError(
            f"{canonical} conflicts with its legacy alias {legacy}: "
            f"{canonical_value!r} != {legacy_value!r}"
        )
    if canonical_value is not missing and canonical_value is not None:
        return canonical_value
    if legacy_value is not missing:
        return legacy_value
    if canonical_value is not missing:
        return canonical_value
    return default


def _legacy_runtime_plan(
    *,
    dialect_id: str,
    requested_model_id: str,
    deployment: Deployment,
    identity: ConnectionIdentity,
) -> RuntimeBindingPlan:
    """Build honest external-Python binding evidence for legacy wrappers."""

    spec = get_dialect_spec(dialect_id)
    route = resolve_route(deployment, dialect_id, spec.connection_family)
    decisions = capability_decisions_for(dialect_id)
    available = frozenset(
        key for key, decision in decisions.items() if isinstance(decision, Supported)
    )
    effective: dict[str, JSONValue] = {key: {} for key in sorted(available)}
    identity_fingerprint = fingerprint_mapping(
        {
            "endpoint": identity.endpoint,
            "connection_family": identity.connection_family,
            "credential_scope": identity.credential_scope,
            "retry_policy": identity.retry_policy,
            "quota_scope": identity.quota_scope,
        }
    )
    identity_suffix = identity_fingerprint.removeprefix("sha256:")[:16]
    binding_id = f"legacy:{dialect_id}:{requested_model_id}:{identity_suffix}"
    root_deployment_key = f"legacy:{deployment.fingerprint}:{identity_suffix}"
    binding_plan_fingerprint = fingerprint_mapping(
        {
            "available_capabilities": sorted(available),
            "binding_id": binding_id,
            "capability_minimums": {},
            "declared_capabilities": effective,
            "dialect_id": dialect_id,
            "effective_capabilities": effective,
            "request_checks": [],
            "request_defaults": {},
            "requested_model_id": requested_model_id,
            "required_output_channels": [],
            "root_deployment_key": root_deployment_key,
        }
    )
    return RuntimeBindingPlan(
        binding_id=binding_id,
        root_deployment_key=root_deployment_key,
        requested_model_id=requested_model_id,
        dialect_id=dialect_id,
        declared_capabilities=effective,
        effective_capabilities=effective,
        available_capabilities=available,
        capability_minimums={},
        request_defaults=RequestDefaults(),
        required_output_channels=frozenset(),
        request_checks=(),
        deployment_fingerprint=deployment.fingerprint,
        resolved_route=route,
        connection_identity=identity,
        binding_plan_fingerprint=binding_plan_fingerprint,
        deployment_plan_fingerprint="external:none",
    )


def build_legacy_openai_binding(
    *,
    dialect_id: str,
    model: str,
    api_base: str | None,
    api_key: str | None,
    max_retries: int,
    concurrency_limit: int | None,
    parent_limiter: anyio.CapacityLimiter | None,
) -> _LegacyOpenAIBinding:
    """Create the wrapper-owned client/pool boundary used for one cycle."""

    client = AsyncOpenAI(
        base_url=api_base,
        api_key=api_key,
        max_retries=max_retries,
    )
    endpoint = str(client.base_url).rstrip("/")
    local_limiter = (
        anyio.CapacityLimiter(concurrency_limit)
        if concurrency_limit is not None
        else None
    )
    shared_limiter = parent_limiter
    if shared_limiter is None:
        shared_limiter = local_limiter

    private_scope = uuid4().hex
    identity = ConnectionIdentity(
        endpoint=endpoint,
        connection_family="openai_sdk",
        credential_scope=(
            f"legacy-private:{private_scope}:explicit-credential"
            if api_key is not None
            else f"legacy-private:{private_scope}:environment-credential"
        ),
        retry_policy=f"openai-sdk:max-retries={max_retries}",
        quota_scope=f"legacy-private:{private_scope}",
    )
    deployment = Deployment(
        deployment_id=None,
        plan=None,
        engine=Engine("unknown"),
        engine_source="unknown",
        api_base=endpoint,
        endpoints={},
        topology=None,
        metrics_url=None,
        facts=ServingFacts(),
    )
    pool = ConnectionPool(client, identity, shared_limiter)
    runtime_plan = _legacy_runtime_plan(
        dialect_id=dialect_id,
        requested_model_id=model,
        deployment=deployment,
        identity=identity,
    )
    return _LegacyOpenAIBinding(
        deployment=deployment,
        pool=pool,
        runtime_plan=runtime_plan,
        local_limiter=local_limiter,
        parent_limiter=parent_limiter,
    )


def _coerce_structured_output(value: object) -> StructuredOutputParams:
    if value is None:
        return StructuredOutputParams()
    if isinstance(value, StructuredOutputParams):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("response_format must be a mapping")
    value_mapping = cast(Mapping[str, object], value)
    format_ = value_mapping.get("type")
    if format_ == "json_object":
        return StructuredOutputParams(format="json_object")
    if format_ != "json_schema":
        raise ValueError(f"unsupported response_format type: {format_!r}")
    raw_schema = value_mapping.get("json_schema")
    if not isinstance(raw_schema, Mapping):
        raise TypeError("json_schema response_format requires a mapping")
    raw_schema_mapping = cast(Mapping[str, object], raw_schema)
    schema = raw_schema_mapping.get("schema")
    if not isinstance(schema, Mapping):
        raise TypeError("json_schema response_format requires `schema`")
    name = raw_schema_mapping.get("name")
    strict = raw_schema_mapping.get("strict")
    if name is not None and not isinstance(name, str):
        raise TypeError("json_schema name must be a string")
    if strict is not None and not isinstance(strict, bool):
        raise TypeError("json_schema strict must be a bool")
    return StructuredOutputParams(
        format="json_schema",
        schema=cast(Mapping[str, JSONValue], _named_json_value(schema, "schema")),
        name=name,
        strict=strict,
    )


def _apply_request_defaults(req: Request, defaults: RequestDefaults) -> Request:
    """Project reconciled defaults onto provider-neutral leaves still unset."""

    result = req
    for path, raw_value in sorted(defaults.values.items()):
        parts = path.split(".")
        if len(parts) != 2:
            raise DialectError(f"invalid request-default leaf path {path!r}")
        group_name, field_name = parts
        if group_name in {"input", "dialect_options"} or not hasattr(
            result, group_name
        ):
            raise DialectError(f"request default cannot target {path!r}")
        group = getattr(result, group_name)
        dataclass_fields = {item.name for item in fields(group)}
        if field_name not in dataclass_fields:
            raise DialectError(f"unknown request-default leaf path {path!r}")
        default_group = type(group)()
        if getattr(group, field_name) != getattr(default_group, field_name):
            continue
        value: object = raw_value
        if group_name == "sampling" and field_name == "stop":
            if not isinstance(raw_value, list) or not all(
                isinstance(item, str) for item in raw_value
            ):
                raise DialectError("sampling.stop request default must be strings")
            value = tuple(raw_value)
        new_group = replace(group, **{field_name: value})
        result = replace(result, **{group_name: new_group})
    return result


class Model:
    """A reconciled deployment/pool/dialect binding.

    Construct canonical models with :meth:`bind`.  Compatibility wrappers use
    the same internal initializer after creating their private pool.
    """

    def __init_subclass__(cls) -> None:
        """Reject legacy execution hooks that the Request/Response path ignores."""

        super().__init_subclass__()
        removed = sorted(_REMOVED_SUBCLASS_HOOKS & cls.__dict__.keys())
        if removed:
            raise TypeError(
                f"{cls.__name__} defines removed Model hook(s): "
                f"{', '.join(removed)}. Compatibility wrappers must override "
                "_build_default_transport(); canonical code must bind a Dialect."
            )

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError(
            "Model cannot infer a client or dialect; use Model.bind("
            "deployment, pool, runtime_plan)"
        )

    @classmethod
    def bind(
        cls,
        deployment: Deployment,
        pool: ConnectionPool[Any],
        runtime_plan: RuntimeBindingPlan,
        *,
        local_limiter: anyio.CapacityLimiter | None = None,
        extra: Mapping[str, JSONValue] | None = None,
    ) -> "Model":
        """Bind an already-owned compatible pool to a reconciled runtime plan."""

        if cls is not Model:
            raise TypeError("canonical binding is Model.bind(...), not a wrapper bind")
        dialect = bind_dialect(
            runtime_plan.dialect_id,
            runtime_plan.requested_model_id,
            deployment,
            pool,
            runtime_plan,
        )
        model = object.__new__(Model)
        model._initialize(
            deployment=deployment,
            pool=pool,
            runtime_plan=runtime_plan,
            dialect=dialect,
            local_limiter=(
                local_limiter
                if local_limiter is not None
                else cast(anyio.CapacityLimiter | None, pool.shared_limiter)
            ),
            parent_limiter=(
                cast(anyio.CapacityLimiter | None, pool.shared_limiter)
                if local_limiter is not None
                else None
            ),
            builder_defaults={},
            extra=extra,
            api_base=deployment.api_base,
            lifecycle_owner=None,
        )
        return model

    def _initialize(
        self,
        *,
        deployment: Deployment,
        pool: ConnectionPool[Any],
        runtime_plan: RuntimeBindingPlan,
        dialect: Dialect,
        local_limiter: anyio.CapacityLimiter | None,
        parent_limiter: anyio.CapacityLimiter | None,
        builder_defaults: Mapping[str, object],
        extra: Mapping[str, JSONValue] | None,
        api_base: str | None,
        lifecycle_owner: "Model | None",
    ) -> None:
        if dialect.dialect_id != runtime_plan.dialect_id:
            raise ValueError("bound dialect does not match the runtime plan")
        if dialect.connection_family != runtime_plan.resolved_route.connection_family:
            raise ValueError("bound dialect does not match the connection family")
        self._deployment = deployment
        self._pool = pool
        self._runtime_plan = runtime_plan
        self._dialect = dialect
        self._transport = dialect  # one-cycle private compatibility alias
        self._model = runtime_plan.requested_model_id
        self._api_base = api_base
        self._kwargs = _checked_builder_defaults(builder_defaults)
        self._extra = dict(extra) if extra is not None else None
        self._limiter = local_limiter
        self._parent_limiter = parent_limiter
        self._lifecycle_owner = lifecycle_owner
        self._client = pool.connection  # borrowed compatibility view; never owned here

    @property
    def dialect_id(self) -> str:
        return self._runtime_plan.dialect_id

    @property
    def runtime_plan(self) -> RuntimeBindingPlan | None:
        return self._runtime_plan

    @property
    def deployment(self) -> Deployment:
        return self._deployment

    @property
    def pool(self) -> ConnectionPool[Any]:
        return self._pool

    def with_args(
        self,
        concurrency_limit: int | None = None,
        extra: dict[str, JSONValue] | None = None,
        **kwargs: object,
    ) -> Self:
        """Derive legacy builder defaults and an optional model-local cap."""

        reserved = sorted(BINDING_RESOURCE_KEYS & set(kwargs))
        if reserved:
            raise ValueError(
                "with_args cannot change binding resources: " + ", ".join(reserved)
            )
        has_local_subquota = (
            self._limiter is not None and self._limiter is not self._pool.shared_limiter
        )
        if concurrency_limit is not None and has_local_subquota:
            raise ValueError(
                "Cannot create multi-level model derivation. Multi-level "
                "resource pools are not supported. Fork from the base model."
            )
        if isinstance(concurrency_limit, bool) or (
            concurrency_limit is not None and concurrency_limit < 1
        ):
            raise ValueError("concurrency_limit must be a positive integer")

        new_model = copy.copy(self)
        new_model._kwargs = _checked_builder_defaults({**self._kwargs, **kwargs})
        if extra is not None:
            new_model._extra = dict(extra)
        if concurrency_limit is not None:
            new_model._limiter = anyio.CapacityLimiter(concurrency_limit)
            new_model._parent_limiter = cast(
                anyio.CapacityLimiter | None,
                self._pool.shared_limiter,
            )
        return new_model

    def with_dialect(
        self,
        dialect_id: str,
        runtime_plan: RuntimeBindingPlan,
    ) -> "Model":
        """Rebind through an externally reconciled target plan and the same pool."""

        if runtime_plan.dialect_id != dialect_id:
            raise ValueError("target dialect does not match runtime_plan.dialect_id")
        if runtime_plan.root_deployment_key != self._runtime_plan.root_deployment_key:
            raise ValueError("target runtime plan belongs to another deployment root")
        if runtime_plan.deployment_fingerprint != self._deployment.fingerprint:
            raise ValueError("target runtime plan belongs to another deployment")
        if runtime_plan.requested_model_id != self._runtime_plan.requested_model_id:
            raise ValueError("dialect rebinding cannot change requested_model_id")
        dialect = bind_dialect(
            dialect_id,
            runtime_plan.requested_model_id,
            self._deployment,
            self._pool,
            runtime_plan,
        )
        result = object.__new__(Model)
        result._initialize(
            deployment=self._deployment,
            pool=self._pool,
            runtime_plan=runtime_plan,
            dialect=dialect,
            local_limiter=self._limiter,
            parent_limiter=self._parent_limiter,
            builder_defaults=self._kwargs,
            extra=self._extra,
            api_base=self._api_base,
            lifecycle_owner=self._lifecycle_owner,
        )
        return result

    def as_compat_type(self, model_type: type["Model"]) -> "Model":
        """Expose a truthful one-cycle wrapper over this exact binding.

        This is a shape adapter for legacy task constructors, not dialect
        conversion.  The current runtime plan, dialect, pool, and limiters are
        retained by identity, and the returned wrapper never acquires pool
        lifecycle ownership.
        """

        expected_input_kind = _compat_model_input_kind(model_type)
        if expected_input_kind is None:
            raise TypeError("model_type must be exactly ChatModel or GenModel")

        dialect_spec = get_dialect_spec(self.dialect_id)
        if expected_input_kind not in dialect_spec.input_kinds:
            raise ValueError(
                f"{model_type.__name__} requires a dialect accepting "
                f"{expected_input_kind!r} input, but {self.dialect_id!r} accepts "
                f"{list(dialect_spec.input_kinds)!r}"
            )

        result = object.__new__(model_type)
        result._initialize(
            deployment=self._deployment,
            pool=self._pool,
            runtime_plan=self._runtime_plan,
            dialect=self._dialect,
            local_limiter=self._limiter,
            parent_limiter=self._parent_limiter,
            builder_defaults=self._kwargs,
            extra=self._extra,
            api_base=self._api_base,
            lifecycle_owner=None,
        )
        return result

    @property
    def capabilities(self) -> frozenset[Capability]:
        """Deprecated enum projection of the keyed runtime capability plan."""

        result: set[Capability] = set()
        try:
            spec = get_dialect_spec(self.dialect_id)
        except ValueError:
            spec = None
        if spec is not None:
            if "chat" in spec.input_kinds:
                result.add(Capability.Chat)
            if "completion" in spec.input_kinds:
                result.add(Capability.Completion)
        for key in self._runtime_plan.available_capabilities:
            result.update(_LEGACY_CAPABILITY_MAP.get(key, ()))
        return frozenset(result)

    def assert_capability(self, *caps: Capability) -> None:
        """Deprecated compatibility check over the keyed runtime plan."""

        missing = frozenset(caps) - self.capabilities
        if missing:
            names = ", ".join(sorted(item.name for item in missing))
            raise CapabilityError(f"{self.dialect_id} does not support: {names}")

    @staticmethod
    def _quota_value(limiter: object | None, attribute: str) -> int | None:
        value = getattr(limiter, attribute, None)
        return value if isinstance(value, int) else None

    def get_available_quota(self) -> int | float:
        values = [
            value
            for value in (
                self._quota_value(self._parent_limiter, "available_tokens"),
                self._quota_value(self._limiter, "available_tokens"),
            )
            if value is not None
        ]
        return min(values) if values else float("inf")

    def get_total_quota(self) -> int | float:
        value = self._quota_value(self._limiter, "total_tokens")
        return value if value is not None else float("inf")

    def get_quota_info(self) -> ModelQuotaInfo:
        info: ModelQuotaInfo = {
            "available": self.get_available_quota(),
            "total": self.get_total_quota(),
            "parent": None,
            "child": None,
        }
        parent_available = self._quota_value(self._parent_limiter, "available_tokens")
        parent_total = self._quota_value(self._parent_limiter, "total_tokens")
        if parent_available is not None and parent_total is not None:
            info["parent"] = {
                "available": parent_available,
                "total": parent_total,
            }
        child_available = self._quota_value(self._limiter, "available_tokens")
        child_total = self._quota_value(self._limiter, "total_tokens")
        if child_available is not None and child_total is not None:
            info["child"] = {"available": child_available, "total": child_total}
        return info

    @property
    def extra(self) -> dict[str, JSONValue]:
        return dict(self._extra) if self._extra else {}

    def meta(self) -> ModelMeta:
        result: ModelMeta = {
            "model": self._model,
            "api_base": self._api_base,
            "default_params": {
                key: _named_json_value(value, f"default_params.{key}")
                for key, value in self._kwargs.items()
            },
        }
        if self._extra:
            result["extra"] = dict(self._extra)
        return result

    async def arun(self, req: Request) -> Response:
        """Execute the fixed pre-I/O audit, dialect call, and provenance flow."""

        if not isinstance(req, Request):
            raise TypeError(f"arun requires Request, got {type(req).__name__}")
        projected = _apply_request_defaults(req, self._runtime_plan.request_defaults)
        validate_request_invariants(projected)
        validate_runtime_binding_plan(self._runtime_plan, projected)
        active_capabilities = active_request_capabilities(projected)
        audit = RequestAudit(active_request_leaves(projected))
        self._dialect.validate_request(projected, audit, self._runtime_plan)
        audit.raise_rejections()
        unknown_verifiers = sorted(
            {
                check.verifier
                for check in self._runtime_plan.request_checks
                if check.capability in active_capabilities
                if check.verifier not in _REQUEST_CHECK_VERIFIERS
            }
        )
        if unknown_verifiers:
            raise DialectError(
                "unknown request-check verifier: " + ", ".join(unknown_verifiers)
            )
        prepared = self._dialect.prepare(projected, audit)
        audit.finish(prepared)

        async with self._pool.acquire(self._limiter):
            response = await self._dialect.execute(prepared)
        self._dialect.output_contract.validate(
            self._runtime_plan,
            projected,
            response,
        )
        self._run_request_checks(projected, response, active_capabilities)
        return replace(response, provenance=self._provenance(response))

    def _run_request_checks(
        self,
        req: Request,
        response: Response,
        active_capabilities: frozenset[str],
    ) -> None:
        """Run the small PR-1 named response-postcondition dispatch."""

        for check in self._runtime_plan.request_checks:
            if check.capability not in active_capabilities:
                continue
            if check.verifier != "validate_response_channel":
                # Names were rejected before admission; retain a defensive
                # branch in case the call sequence is edited later.
                raise DialectError(f"unknown request-check verifier: {check.verifier}")
            channel = _RESPONSE_CHANNEL_BY_CAPABILITY.get(check.capability)
            if channel is None:
                raise DialectError(
                    "validate_response_channel has no channel for capability "
                    f"{check.capability!r}"
                )
            value = getattr(response, channel)
            if value is None:
                raise DialectError(
                    f"request check {check.verifier!r} failed: {channel!r} absent"
                )
            # Presence is the only generic response postcondition.  In
            # particular, candidate availability is not observable separately
            # from a returned top-logprob position: a short position can be
            # valid constrained decoding rather than provider truncation, so
            # consumers validate the concrete alternatives they require.

    def _provenance(self, response: Response) -> ModelProvenance:
        plan = self._runtime_plan
        deployment = self._deployment
        return ModelProvenance(
            dialect_id=plan.dialect_id,
            engine_id=deployment.engine.engine_id,
            engine_source=deployment.engine_source,
            deployment_id=deployment.deployment_id,
            model_identity=ModelIdentity(
                requested_model_id=plan.requested_model_id,
                provider_reported_model_id=response.response_model,
            ),
            engine_version=deployment.facts.engine_version,
            deployment_fingerprint=deployment.fingerprint,
            capabilities=CapabilityEvidence(
                declared=plan.declared_capabilities,
                effective=plan.effective_capabilities,
                plan_fingerprint=plan.fingerprint,
                verification_fingerprint=plan.verification_fingerprint,
            ),
        )

    async def agenerate(self, prompt: object, **kwargs: object) -> ModelOutput:
        req = self._build_generate_request(prompt, **kwargs)
        return self._response_to_model_output(await self.arun(req))

    async def alogprobs(
        self,
        prompt: object,
        *,
        max_tokens: int = 1,
        logprobs: int = 5,
        echo: bool = True,
        temperature: float = 0.0,
        **kwargs: object,
    ) -> ModelOutput:
        if echo:
            self.assert_capability(Capability.InputScoring)
        req = self._build_logprobs_request(
            prompt,
            max_tokens=max_tokens,
            logprobs=logprobs,
            score_input=echo,
            temperature=temperature,
            **kwargs,
        )
        response = await self.arun(req)
        if (
            response.logprobs is None
            and response.input_scoring is None
            and response.top_logprobs is None
        ):
            raise RuntimeError("logprobs requested but the server returned none.")
        return self._response_to_model_output(response)

    @staticmethod
    def _validate_n(kwargs: Mapping[str, object]) -> int:
        n = kwargs.get("n", 1)
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(f"n must be an int, got {type(n).__name__}: {n!r}")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        return n

    def _coerce_input(self, prompt: object) -> ModelInput:
        if isinstance(prompt, CompletionInput | ChatInput):
            return prompt
        raise TypeError(
            "plain Model does not guess input modality; pass CompletionInput "
            "or ChatInput"
        )

    def _kwargs_to_request(
        self,
        input_: ModelInput,
        final_kwargs: Mapping[str, object],
    ) -> Request:
        kw = dict(final_kwargs)
        reserved = sorted(BINDING_RESOURCE_KEYS & set(kw))
        if reserved:
            raise ValueError(
                "request arguments cannot change binding resources: "
                + ", ".join(reserved)
            )
        n = self._validate_n(kw)
        kw.pop("n", None)

        stream = kw.pop("stream", True)
        if not isinstance(stream, bool):
            raise TypeError("stream must be a bool")

        max_tokens = _pop_compatible_alias(
            kw,
            "max_tokens",
            "max_completion_tokens",
            default=None,
        )
        temperature_value = kw.pop("temperature", None)
        top_p_value = kw.pop("top_p", None)
        top_k_value = kw.pop("top_k", None)
        seed_value = kw.pop("seed", None)
        frequency_penalty_value = kw.pop("frequency_penalty", None)
        presence_penalty_value = kw.pop("presence_penalty", None)
        stop = kw.pop("stop", None)
        stop_value: tuple[str, ...] | None = None
        if stop is not None:
            if isinstance(stop, str):
                stop_value = (stop,)
            elif isinstance(stop, list | tuple):
                values = tuple(stop)
                if not all(isinstance(item, str) for item in values):
                    raise TypeError("stop must contain only strings")
                stop_value = cast(tuple[str, ...], values)
            else:
                # ``list``/``tuple`` only, matching ``_named_json_value``: this
                # value is echoed into the persisted ``request_params``, and a
                # ``set`` would land there in hash order.
                raise TypeError("stop must be a string, list, or tuple of strings")
        sampling = SamplingParams(
            temperature=_optional_float(temperature_value, "temperature"),
            top_p=_optional_float(top_p_value, "top_p"),
            top_k=_optional_int(top_k_value, "top_k"),
            max_tokens=_optional_int(max_tokens, "max_tokens"),
            stop=stop_value,
            seed=_optional_int(seed_value, "seed"),
            frequency_penalty=_optional_float(
                frequency_penalty_value, "frequency_penalty"
            ),
            presence_penalty=_optional_float(
                presence_penalty_value, "presence_penalty"
            ),
            n=n,
        )

        return_logprobs = kw.pop("return_logprobs", False)
        if not isinstance(return_logprobs, bool):
            raise TypeError("return_logprobs must be a bool")
        top_logprobs = kw.pop("top_logprobs", None)
        legacy_logprobs = kw.pop("logprobs", None)
        sampled = return_logprobs
        breadth = 0
        if isinstance(legacy_logprobs, bool):
            sampled = sampled or legacy_logprobs
        elif legacy_logprobs is not None:
            if isinstance(legacy_logprobs, bool) or not isinstance(
                legacy_logprobs, int
            ):
                raise TypeError("logprobs must be a bool or integer")
            sampled = True
            breadth = legacy_logprobs
        if top_logprobs is not None:
            if isinstance(top_logprobs, bool) or not isinstance(top_logprobs, int):
                raise TypeError("top_logprobs must be an integer")
            breadth = top_logprobs
            sampled = sampled or top_logprobs > 0
        score_input = _pop_compatible_alias(
            kw,
            "score_input",
            "echo",
            default=False,
        )
        if not isinstance(score_input, bool):
            raise TypeError("echo/score_input must be a bool")
        scoring = ScoringParams(
            input_scoring=score_input,
            sampled_logprobs=sampled,
            top_logprobs=breadth,
        )

        reasoning = kw.pop("reasoning", None)
        effort = kw.pop("reasoning_effort", None)
        if effort is not None and not isinstance(effort, str):
            raise TypeError("reasoning_effort must be a string")
        if reasoning is None:
            reasoning_params = ReasoningParams(effort=effort)
        elif isinstance(reasoning, ReasoningParams):
            if effort is not None:
                raise ValueError("reasoning and reasoning_effort cannot both be set")
            reasoning_params = reasoning
        else:
            raise TypeError("reasoning must be ReasoningParams")

        raw_tools = kw.pop("tools", ())
        if raw_tools is None:
            raw_tools = ()
        if not isinstance(raw_tools, Iterable) or isinstance(
            raw_tools, Mapping | str | bytes
        ):
            raise TypeError("tools must be an iterable of mappings")
        functions: list[Mapping[str, JSONValue]] = []
        for index, tool in enumerate(raw_tools):
            if not isinstance(tool, Mapping):
                raise TypeError("tools must contain mappings")
            functions.append(
                cast(
                    Mapping[str, JSONValue],
                    _named_json_value(tool, f"tools[{index}]"),
                )
            )
        choice = _named_json_value(kw.pop("tool_choice", None), "tool_choice")
        parallel = kw.pop("parallel_tool_calls", None)
        if parallel is not None and not isinstance(parallel, bool):
            raise TypeError("parallel_tool_calls must be a bool")
        tools = ToolParams(
            functions=tuple(functions),
            choice=choice,
            parallel=parallel,
        )

        structured_output = _coerce_structured_output(kw.pop("response_format", None))
        previous_response_id = _pop_compatible_alias(
            kw,
            "previous_response_id",
            "session_id",
            default=None,
        )
        if previous_response_id is not None and not isinstance(
            previous_response_id, str
        ):
            raise TypeError("previous_response_id must be a string")
        continuation = kw.pop("opaque_continuation", None)
        if continuation is not None and not isinstance(
            continuation, OpaqueContinuation
        ):
            raise TypeError("opaque_continuation must be OpaqueContinuation")

        suffix = kw.pop("suffix", None)
        if suffix is not None:
            if not isinstance(suffix, str):
                raise TypeError("suffix must be a string")
            if not isinstance(input_, CompletionInput):
                raise TypeError("suffix requires CompletionInput")
            if input_.suffix is not None and input_.suffix != suffix:
                raise ValueError("suffix conflicts with CompletionInput.suffix")
            input_ = replace(input_, suffix=suffix)

        options: dict[str, JSONValue] = {}
        explicit_options = kw.pop("dialect_options", None)
        if explicit_options is not None:
            if not isinstance(explicit_options, DialectOptions):
                raise TypeError("dialect_options must be DialectOptions")
            if explicit_options.dialect_id != self.dialect_id:
                raise ValueError("dialect_options target another dialect")
            reserved = sorted(BINDING_RESOURCE_KEYS & set(explicit_options.values))
            if reserved:
                raise ValueError(
                    "dialect_options cannot contain binding resources: "
                    + ", ".join(reserved)
                )
            options.update(explicit_options.values)
        for container_name in ("extra_body", "extra_wire_params"):
            raw = kw.pop(container_name, None)
            if raw is None:
                continue
            if not isinstance(raw, Mapping):
                raise TypeError(f"{container_name} must be a mapping")
            for key, value in raw.items():
                if not isinstance(key, str):
                    raise TypeError(f"{container_name} keys must be strings")
                if key in BINDING_RESOURCE_KEYS:
                    raise ValueError(
                        f"{container_name} cannot contain binding resource {key!r}"
                    )
                if key in options:
                    raise ValueError(f"duplicate dialect option {key!r}")
                options[key] = _named_json_value(value, f"{container_name}.{key}")
        for key, value in kw.items():
            if key in options:
                raise ValueError(f"duplicate dialect option {key!r}")
            options[key] = _named_json_value(value, key)

        return Request(
            input=input_,
            sampling=sampling,
            scoring=scoring,
            reasoning=reasoning_params,
            tools=tools,
            structured_output=structured_output,
            session=SessionParams(
                previous_response_id=previous_response_id,
                opaque_continuation=continuation,
            ),
            scheduling=SchedulingParams(stream=stream),
            dialect_options=(
                DialectOptions(self.dialect_id, options) if options else None
            ),
        )

    def _build_generate_request(self, prompt: object, **kwargs: object) -> Request:
        final_kwargs = {**self._kwargs, **kwargs}
        return self._kwargs_to_request(self._coerce_input(prompt), final_kwargs)

    def _build_logprobs_request(
        self,
        prompt: object,
        *,
        max_tokens: int,
        logprobs: int,
        score_input: bool,
        temperature: float,
        **kwargs: object,
    ) -> Request:
        if isinstance(logprobs, bool) or not isinstance(logprobs, int):
            raise TypeError("logprobs must be an integer")
        final_kwargs = {**self._kwargs, **kwargs}
        # ``logprobs`` is a legacy builder alias.  The explicit alogprobs
        # arguments below are authoritative and must not leave that alias
        # active alongside input scoring.
        final_kwargs.pop("logprobs", None)
        final_kwargs.update(
            {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "return_logprobs": not score_input,
                "top_logprobs": logprobs if not score_input else 0,
                "score_input": score_input,
            }
        )
        if self._validate_n(final_kwargs) > 1:
            raise ValueError("alogprobs only supports n=1")
        return self._kwargs_to_request(self._coerce_input(prompt), final_kwargs)

    def _response_to_model_output(self, response: Response) -> ModelOutput:
        segments: list[TokenLogprob] = []
        if response.input_scoring is not None:
            segments.extend(response.input_scoring.token_logprobs)
        if response.logprobs is not None:
            segments.extend(response.logprobs)
        logprobs_present = (
            response.input_scoring is not None or response.logprobs is not None
        )
        logprobs_tokens = (
            [item.token for item in segments] if logprobs_present else None
        )
        logprobs = [item.logprob for item in segments] if logprobs_present else None

        top_logprobs: list[dict[str, float]] | None = None
        if response.top_logprobs is not None:
            top_logprobs = []
            for position in response.top_logprobs:
                merged: dict[str, float] = {}
                for item in position:
                    previous = merged.get(item.token)
                    if previous is None or item.logprob > previous:
                        merged[item.token] = item.logprob
                top_logprobs.append(merged)
            top_logprobs = top_logprobs or None

        usage: ModelUsage | None = None
        if response.usage is not None:
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        reasoning_texts: list[str] | None = None
        if response.reasoning is not None:
            reasoning_texts = [
                item.text or "" if item is not None else ""
                for item in response.reasoning
            ]
            if not any(reasoning_texts):
                reasoning_texts = None

        model_meta = self.meta()
        if response.provenance is not None:
            model_meta["provenance"] = response.provenance
        return ModelOutput(
            model=model_meta,
            texts=list(response.texts),
            finish_reasons=(
                list(response.finish_reasons)
                if response.finish_reasons is not None
                else None
            ),
            reasoning_texts=reasoning_texts,
            logprobs_tokens=logprobs_tokens,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            usage=usage,
            request_params=(
                dict(response.request_params)
                if response.request_params is not None
                else None
            ),
            response_model=response.response_model,
            system_fingerprint=response.system_fingerprint,
        )

    async def aclose(self) -> None:
        """Close the wrapper-owned pool; canonical models leave ownership outside."""

        owner = self._lifecycle_owner
        if owner is None:
            raise RuntimeError("canonical Model does not own its ConnectionPool")
        await owner._pool.aclose()

    async def __aenter__(self) -> Self:
        owner = self._lifecycle_owner
        if owner is None:
            raise RuntimeError("use the canonical model's ConnectionPool context")
        await owner._pool.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        await self.aclose()
