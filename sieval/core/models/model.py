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
import json
from collections.abc import Mapping
from dataclasses import fields, replace
from types import TracebackType
from typing import Any, Protocol, Self, TypedDict, cast

import anyio

from sieval.core.types import JSONValue

from ._legacy_binding import _legacy_provenance_projector_for_plan
from ._legacy_bridge import (
    ModelMeta,
    ModelOutput,
    kwargs_to_request,
    response_to_model_output,
    validate_n,
)
from ._shared import named_json_value
from .capabilities import (
    Capability,
    RequestDefaults,
)
from .deployment import (
    BINDING_RESOURCE_KEYS,
    ConnectionIdentity,
    ConnectionPool,
    Deployment,
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
    get_dialect_spec,
)
from .exceptions import CapabilityError
from .ir import (
    CapabilityEvidence,
    ChatInput,
    CompletionInput,
    ModelIdentity,
    ModelInput,
    ModelProvenance,
    Request,
    Response,
)
from .reconcile import CheckStage, DeferredCheck, RuntimeBindingPlan


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


class _ProvenanceProjector(Protocol):
    """Legacy seam for deriving stable evidence from a runtime plan."""

    def __call__(
        self, runtime_plan: RuntimeBindingPlan
    ) -> RuntimeBindingPlan | None: ...


_PROVENANCE_PROJECTABLE_PLAN_FIELDS = frozenset(
    {
        "binding_id",
        "root_deployment_key",
        "binding_plan_fingerprint",
        "deployment_plan_fingerprint",
    }
)
_PROVENANCE_SEMANTIC_PLAN_FIELDS = frozenset(
    {
        "requested_model_id",
        "dialect_id",
        "declared_capabilities",
        "effective_capabilities",
        "available_capabilities",
        "capability_minimums",
        "request_defaults",
        "required_output_channels",
        "deployment_fingerprint",
        "resolved_route",
    }
)
_PROVENANCE_SPECIAL_PLAN_FIELDS = frozenset({"request_checks", "connection_identity"})
_PROVENANCE_COMPUTED_PLAN_FIELDS = frozenset(
    {"fingerprint", "verification_fingerprint"}
)
_PROVENANCE_PROJECTABLE_CONNECTION_FIELDS = frozenset(
    {"credential_scope", "quota_scope"}
)
_PROVENANCE_SEMANTIC_CONNECTION_FIELDS = frozenset(
    {"endpoint", "connection_family", "retry_policy"}
)
_PROVENANCE_PROJECTABLE_CHECK_FIELDS = frozenset({"reason"})
_PROVENANCE_SEMANTIC_CHECK_FIELDS = frozenset({"capability", "stage", "verifier"})


def _validate_provenance_schema() -> None:
    """Fail when a plan field has no explicit provenance policy."""

    plan_fields = {item.name for item in fields(RuntimeBindingPlan)}
    classified_plan_fields = (
        _PROVENANCE_PROJECTABLE_PLAN_FIELDS
        | _PROVENANCE_SEMANTIC_PLAN_FIELDS
        | _PROVENANCE_SPECIAL_PLAN_FIELDS
        | _PROVENANCE_COMPUTED_PLAN_FIELDS
    )
    if plan_fields != classified_plan_fields:
        missing = sorted(plan_fields - classified_plan_fields)
        stale = sorted(classified_plan_fields - plan_fields)
        raise RuntimeError(
            "RuntimeBindingPlan provenance policy is incomplete; "
            f"missing={missing!r}, stale={stale!r}"
        )

    identity_fields = {item.name for item in fields(ConnectionIdentity)}
    classified_identity_fields = (
        _PROVENANCE_PROJECTABLE_CONNECTION_FIELDS
        | _PROVENANCE_SEMANTIC_CONNECTION_FIELDS
    )
    if identity_fields != classified_identity_fields:
        missing = sorted(identity_fields - classified_identity_fields)
        stale = sorted(classified_identity_fields - identity_fields)
        raise RuntimeError(
            "ConnectionIdentity provenance policy is incomplete; "
            f"missing={missing!r}, stale={stale!r}"
        )

    check_fields = {item.name for item in fields(DeferredCheck)}
    classified_check_fields = (
        _PROVENANCE_PROJECTABLE_CHECK_FIELDS | _PROVENANCE_SEMANTIC_CHECK_FIELDS
    )
    if check_fields != classified_check_fields:
        missing = sorted(check_fields - classified_check_fields)
        stale = sorted(classified_check_fields - check_fields)
        raise RuntimeError(
            "DeferredCheck provenance policy is incomplete; "
            f"missing={missing!r}, stale={stale!r}"
        )


def _request_check_semantics(
    checks: tuple[DeferredCheck, ...],
) -> tuple[tuple[str, CheckStage, str], ...]:
    """Return the executable part of checks; reasons are diagnostic evidence."""

    return tuple((check.capability, check.stage, check.verifier) for check in checks)


def _same_json_value(left: JSONValue, right: JSONValue) -> bool:
    """Compare JSON values with the same type-sensitive rules as fingerprints."""

    def encode(value: JSONValue) -> str:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    return encode(left) == encode(right)


def _validate_provenance_plan(
    runtime_plan: RuntimeBindingPlan,
    provenance_plan: RuntimeBindingPlan,
) -> None:
    """Reject projections that change request or capability semantics."""

    _validate_provenance_schema()
    runtime_value = runtime_plan.to_json_value()
    provenance_value = provenance_plan.to_json_value()
    changed = [
        name
        for name in sorted(_PROVENANCE_SEMANTIC_PLAN_FIELDS)
        if not _same_json_value(runtime_value[name], provenance_value[name])
    ]
    if _request_check_semantics(provenance_plan.request_checks) != (
        _request_check_semantics(runtime_plan.request_checks)
    ):
        changed.append("request_checks")
    changed.extend(
        f"connection_identity.{name}"
        for name in sorted(_PROVENANCE_SEMANTIC_CONNECTION_FIELDS)
        if getattr(provenance_plan.connection_identity, name)
        != getattr(runtime_plan.connection_identity, name)
    )
    if changed:
        raise ValueError(
            "provenance plan changes runtime semantics: " + ", ".join(changed)
        )


def _checked_builder_defaults(values: Mapping[str, object]) -> dict[str, object]:
    """Reject builder defaults that ``meta()`` could not persist.

    ``meta()`` runs once per response, so an unpersistable default would
    otherwise raise only after a call had been billed. Values are stored
    unconverted -- the request builders need them as given.
    """
    for key, value in values.items():
        named_json_value(value, f"default_params.{key}")
    return dict(values)


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
        provenance_projector: _ProvenanceProjector | None = None,
        provenance_plan: RuntimeBindingPlan | None = None,
    ) -> None:
        if dialect.dialect_id != runtime_plan.dialect_id:
            raise ValueError("bound dialect does not match the runtime plan")
        if dialect.connection_family != runtime_plan.resolved_route.connection_family:
            raise ValueError("bound dialect does not match the connection family")
        if provenance_projector is None:
            provenance_projector = _legacy_provenance_projector_for_plan(runtime_plan)
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
        # Canonical bindings persist their runtime plan verbatim. Legacy
        # wrappers retain volatile pool ownership at runtime while projecting
        # every derived/reconciled plan into stable persisted evidence.
        self._provenance_projector = provenance_projector
        if provenance_plan is None:
            provenance_plan = (
                runtime_plan
                if provenance_projector is None
                else provenance_projector(runtime_plan)
            )
        if provenance_plan is not None:
            _validate_provenance_plan(runtime_plan, provenance_plan)
        self._provenance_plan = provenance_plan

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

    def without_args(self, *names: str) -> Self:
        """Derive a model with selected builder defaults removed.

        This is intentionally distinct from ``with_args(name=None)``: ``None``
        is a user-visible explicit default that belongs in model metadata,
        whereas removing an inherited orchestration default must leave no
        request leaf and no persisted ``default_params`` entry.
        """

        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("without_args requires non-empty argument names")
        reserved = sorted(BINDING_RESOURCE_KEYS & set(names))
        if reserved:
            raise ValueError(
                "without_args cannot change binding resources: " + ", ".join(reserved)
            )
        new_model = copy.copy(self)
        new_model._kwargs = {
            key: value for key, value in self._kwargs.items() if key not in names
        }
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
        provenance_plan = (
            self._provenance_plan if runtime_plan == self._runtime_plan else None
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
            provenance_projector=self._provenance_projector,
            provenance_plan=provenance_plan,
        )
        return result

    def _with_provenance_plan(
        self,
        provenance_plan: RuntimeBindingPlan,
    ) -> Self:
        """Return a copy carrying trusted composition-layer provenance."""

        if self._provenance_projector is None and provenance_plan != self._runtime_plan:
            raise ValueError(
                "canonical models must persist their runtime plan verbatim"
            )
        _validate_provenance_plan(self._runtime_plan, provenance_plan)
        if provenance_plan == self._provenance_plan:
            return self
        result = copy.copy(self)
        result._provenance_plan = provenance_plan
        return result

    def _require_provenance_plan(self) -> RuntimeBindingPlan:
        """Return complete persisted evidence or fail before request I/O."""

        provenance_plan = self._provenance_plan
        if provenance_plan is None:
            raise RuntimeError(
                "model provenance is incomplete: the reconciled binding and "
                "deployment evidence must be attached by the composition layer"
            )
        return provenance_plan

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
            provenance_projector=self._provenance_projector,
            provenance_plan=self._provenance_plan,
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
                key: named_json_value(value, f"default_params.{key}")
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
        self._require_provenance_plan()
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
        provenance_plan = self._require_provenance_plan()
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
                plan_fingerprint=provenance_plan.fingerprint,
                verification_fingerprint=provenance_plan.verification_fingerprint,
            ),
        )

    async def agenerate(self, prompt: object, **kwargs: object) -> ModelOutput:
        req = self._build_generate_request(prompt, **kwargs)
        return response_to_model_output(self.meta(), await self.arun(req))

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
        return response_to_model_output(self.meta(), response)

    def _coerce_input(self, prompt: object) -> ModelInput:
        if isinstance(prompt, CompletionInput | ChatInput):
            return prompt
        raise TypeError(
            "plain Model does not guess input modality; pass CompletionInput "
            "or ChatInput"
        )

    def _build_generate_request(self, prompt: object, **kwargs: object) -> Request:
        final_kwargs = {**self._kwargs, **kwargs}
        return kwargs_to_request(
            self.dialect_id, self._coerce_input(prompt), final_kwargs
        )

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
        if validate_n(final_kwargs) > 1:
            raise ValueError("alogprobs only supports n=1")
        return kwargs_to_request(
            self.dialect_id, self._coerce_input(prompt), final_kwargs
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
