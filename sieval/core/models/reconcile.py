"""Pure capability reconciliation for model dialect bindings.

The join order is deliberately explicit: task and declaration intent, dialect
decision, model-profile outcome, then injected serving outcome.  This module
contains no network I/O, launch code, engine predicates, or condition algebra.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Never, Protocol, Self, cast

from sieval.core.types import JSONValue

from ._fingerprint import fingerprint_mapping
from ._shared import copy_json_value
from .capabilities import (
    CAPABILITY_KEYS,
    CAPABILITY_SPECS,
    CapabilityConfigError,
    CapabilityIntent,
    CapabilityKey,
    DialectCapabilityStatus,
    FunctionToolsOptions,
    HostedToolsOptions,
    ModelCapabilityProfile,
    ModelCapabilityStatus,
    MultimodalInputOptions,
    NormalizedCapabilityValue,
    ReasoningOptions,
    RequestDefaults,
    StructuredOutputOptions,
    Supported,
    TopLogprobsOptions,
    Unsupported,
    capability_declarations_to_json,
    normalize_capability_declarations,
)
from .deployment import (
    ConnectionIdentity,
    Deployment,
    DeploymentPlanProjection,
    ResolvedRoute,
    RouteIntent,
    resolve_route,
)
from .dialect_registry import (
    DialectNotImplemented,
    DialectSpec,
    UnknownDialect,
    capability_decisions_for,
    get_dialect_spec,
)
from .requirements import AggregatedTaskRequirements, InputModality


class ReconcileSeverity(StrEnum):
    """Machine-readable importance of one deterministic diagnostic."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class CheckStage(StrEnum):
    """Where a named deferred verifier can safely run."""

    SETUP = "setup"
    REQUEST = "request"


@dataclass(frozen=True)
class ReconcileDiagnostic:
    """One source-preserving capability reconciliation diagnostic."""

    severity: ReconcileSeverity
    code: str
    message: str
    binding_id: str | None = None
    root_deployment_key: str | None = None
    capability: CapabilityKey | None = None
    sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.severity, ReconcileSeverity):
            raise TypeError("severity must be a ReconcileSeverity")
        _nonempty(self.code, "diagnostic code")
        _nonempty(self.message, "diagnostic message")
        if self.binding_id is not None:
            _nonempty(self.binding_id, "diagnostic binding_id")
        if self.root_deployment_key is not None:
            _nonempty(self.root_deployment_key, "diagnostic root_deployment_key")
        if self.capability is not None and self.capability not in CAPABILITY_SPECS:
            raise ValueError(f"unknown diagnostic capability {self.capability!r}")
        object.__setattr__(self, "sources", _source_tuple(self.sources))

    def to_json_value(self) -> dict[str, JSONValue]:
        value: dict[str, JSONValue] = {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "sources": list(self.sources),
        }
        if self.binding_id is not None:
            value["binding_id"] = self.binding_id
        if self.root_deployment_key is not None:
            value["root_deployment_key"] = self.root_deployment_key
        if self.capability is not None:
            value["capability"] = self.capability
        return value


@dataclass(frozen=True)
class DeferredCheck:
    """A named, serializable setup or request-time verification obligation."""

    capability: CapabilityKey
    stage: CheckStage
    verifier: str
    reason: str

    def __post_init__(self) -> None:
        if self.capability not in CAPABILITY_SPECS:
            raise ValueError(f"unknown deferred capability {self.capability!r}")
        if not isinstance(self.stage, CheckStage):
            raise TypeError("stage must be a CheckStage")
        _nonempty(self.verifier, "deferred verifier")
        _nonempty(self.reason, "deferred reason")

    def to_json_value(self) -> dict[str, JSONValue]:
        return {
            "capability": self.capability,
            "stage": self.stage.value,
            "verifier": self.verifier,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ServingRequirement:
    """Provider-neutral input to an injected #47 serving reconciler."""

    capability: CapabilityKey
    minimums: Mapping[str, JSONValue] = field(default_factory=dict)
    sources: tuple[str, ...] = ()
    verifier: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.capability not in CAPABILITY_SPECS:
            raise ValueError(f"unknown serving capability {self.capability!r}")
        object.__setattr__(self, "minimums", _freeze_json(self.minimums, "minimums"))
        object.__setattr__(self, "sources", _source_tuple(self.sources))
        if self.verifier is not None:
            _nonempty(self.verifier, "serving verifier")
        if self.reason is not None:
            _nonempty(self.reason, "serving reason")

    def to_json_value(self) -> dict[str, JSONValue]:
        value: dict[str, JSONValue] = {
            "capability": self.capability,
            "minimums": _thaw_json(self.minimums),
            "sources": list(self.sources),
        }
        if self.verifier is not None:
            value["verifier"] = self.verifier
        if self.reason is not None:
            value["reason"] = self.reason
        return value


@dataclass(frozen=True)
class Configured:
    """A serving requirement is satisfied or safely configured."""

    launch_patch: Mapping[str, JSONValue] = field(default_factory=dict)
    request_checks: tuple[DeferredCheck, ...] = ()
    evidence: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "launch_patch", _freeze_json(self.launch_patch, "launch_patch")
        )
        object.__setattr__(
            self, "evidence", _freeze_json(self.evidence, "serving evidence")
        )
        if any(check.stage is not CheckStage.REQUEST for check in self.request_checks):
            raise ValueError("Configured.request_checks must use request stage")


@dataclass(frozen=True)
class CannotVerify:
    """A named verifier exists, but available setup data cannot discharge it."""

    stage: CheckStage
    verifier: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.stage, CheckStage):
            raise TypeError("stage must be a CheckStage")
        _nonempty(self.verifier, "cannot-verify verifier")
        _nonempty(self.reason, "cannot-verify reason")


@dataclass(frozen=True)
class ServingUnsupported:
    """The realized/configurable service cannot satisfy a capability."""

    reason: str
    remedy: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.reason, "unsupported reason")
        if self.remedy is not None:
            _nonempty(self.remedy, "unsupported remedy")


type ServingOutcome = Configured | CannotVerify | ServingUnsupported


class ServingReconciler(Protocol):
    """Injected #47 seam; called once per root-deployment requirement group."""

    def reconcile(
        self,
        requirements: tuple[ServingRequirement, ...],
        deployment: "DeploymentReconcileInput",
    ) -> Mapping[CapabilityKey, ServingOutcome]: ...


@dataclass(frozen=True)
class ConnectionScope:
    """Secret-free pool-sharing scopes known before route resolution."""

    credential_scope: str
    retry_policy: str
    quota_scope: str

    def __post_init__(self) -> None:
        _nonempty(self.credential_scope, "credential_scope")
        _nonempty(self.retry_policy, "retry_policy")
        _nonempty(self.quota_scope, "quota_scope")


@dataclass(frozen=True)
class BindingReconcileInput:
    """Typed inputs for one logical model binding."""

    binding_id: str
    root_deployment_key: str
    requested_model_id: str
    dialect_id: str
    requirements: AggregatedTaskRequirements
    model_profile: ModelCapabilityProfile
    connection_scope: ConnectionScope
    declarations: Mapping[str, JSONValue] = field(default_factory=dict)
    request_intents: Mapping[CapabilityKey, CapabilityIntent] = field(
        default_factory=dict
    )
    route_intent: RouteIntent = field(default_factory=RouteIntent)
    declaration_source: str = "model.capabilities"

    def __post_init__(self) -> None:
        for name in (
            "binding_id",
            "root_deployment_key",
            "requested_model_id",
            "dialect_id",
            "declaration_source",
        ):
            _nonempty(getattr(self, name), name)
        if not isinstance(self.requirements, AggregatedTaskRequirements):
            raise TypeError("requirements must be AggregatedTaskRequirements")
        if not isinstance(self.model_profile, ModelCapabilityProfile):
            raise TypeError("model_profile must be ModelCapabilityProfile")
        if not isinstance(self.connection_scope, ConnectionScope):
            raise TypeError("connection_scope must be ConnectionScope")
        if not isinstance(self.route_intent, RouteIntent):
            raise TypeError("route_intent must be RouteIntent")
        object.__setattr__(
            self,
            "declarations",
            _freeze_json(self.declarations, "capability declarations"),
        )
        request_intents: dict[CapabilityKey, CapabilityIntent] = {}
        for key, intent in self.request_intents.items():
            if key not in CAPABILITY_SPECS:
                raise ValueError(f"unknown request intent capability {key!r}")
            if not isinstance(intent, CapabilityIntent):
                raise TypeError("request_intents values must be CapabilityIntent")
            if intent.key != key:
                raise ValueError("request intent key does not match its mapping key")
            request_intents[key] = intent
        object.__setattr__(
            self,
            "request_intents",
            MappingProxyType(request_intents),
        )


@dataclass(frozen=True)
class DeploymentReconcileInput:
    """Desired and optionally realized state for one root deployment."""

    root_deployment_key: str
    engine_id: str
    deployment: Deployment | None = None
    plan: DeploymentPlanProjection | None = None
    recipe_parameters: Mapping[str, JSONValue] = field(default_factory=dict)
    explicit_parameters: Mapping[str, JSONValue] = field(default_factory=dict)
    prelaunch_plan: "DeploymentCapabilityPlan | None" = None

    def __post_init__(self) -> None:
        _nonempty(self.root_deployment_key, "root_deployment_key")
        _nonempty(self.engine_id, "engine_id")
        if self.plan is not None and self.plan.engine_id != self.engine_id:
            raise ValueError("desired plan engine does not match engine_id")
        if self.deployment is not None:
            if self.engine_id != self.deployment.engine.engine_id:
                raise ValueError("deployment engine does not match engine_id")
            if self.plan is not None and self.plan != self.deployment.plan:
                raise ValueError("desired plan does not match realized deployment plan")
        if self.prelaunch_plan is not None:
            if self.deployment is None:
                raise ValueError(
                    "prelaunch_plan is only valid for a realized deployment"
                )
            if self.prelaunch_plan.root_deployment_key != self.root_deployment_key:
                raise ValueError("prelaunch_plan belongs to another root deployment")
            if (
                self.prelaunch_plan.engine_id != "unknown"
                and self.prelaunch_plan.engine_id != self.engine_id
            ):
                raise ValueError("prelaunch_plan belongs to another engine")
        object.__setattr__(
            self,
            "recipe_parameters",
            _freeze_json(self.recipe_parameters, "recipe_parameters"),
        )
        object.__setattr__(
            self,
            "explicit_parameters",
            _freeze_json(self.explicit_parameters, "explicit_parameters"),
        )

    @property
    def effective_plan(self) -> DeploymentPlanProjection | None:
        if self.deployment is not None:
            return self.deployment.plan
        return self.plan

    def to_json_value(self) -> dict[str, JSONValue]:
        plan = self.effective_plan
        value: dict[str, JSONValue] = {
            "root_deployment_key": self.root_deployment_key,
            "engine_id": self.engine_id,
            "recipe_parameters": _thaw_json(self.recipe_parameters),
            "explicit_parameters": _thaw_json(self.explicit_parameters),
            "plan": (
                None
                if plan is None
                else {
                    "fingerprint": plan.fingerprint,
                    "engine_id": plan.engine_id,
                    "service_roles": list(plan.service_roles),
                }
            ),
        }
        if self.deployment is not None:
            value["deployment_fingerprint"] = self.deployment.fingerprint
        if self.prelaunch_plan is not None:
            value["prelaunch_plan_fingerprint"] = self.prelaunch_plan.fingerprint
        return value


@dataclass(frozen=True)
class ReconcileBatch:
    """One deterministic batch of bindings and their root deployments."""

    bindings: tuple[BindingReconcileInput, ...]
    deployments: Mapping[str, DeploymentReconcileInput]

    def __post_init__(self) -> None:
        binding_ids = [binding.binding_id for binding in self.bindings]
        if len(set(binding_ids)) != len(binding_ids):
            raise ValueError("binding ids must be unique within a reconcile batch")
        copied = dict(self.deployments)
        for key, deployment in copied.items():
            if key != deployment.root_deployment_key:
                raise ValueError("deployment mapping key does not match its root key")
        object.__setattr__(self, "deployments", MappingProxyType(copied))


@dataclass(frozen=True)
class BindingCapabilityPlan:
    """Dialect/model join for one binding, before root serving resolution."""

    binding_id: str
    root_deployment_key: str
    requested_model_id: str
    dialect_id: str
    declared_capabilities: Mapping[str, JSONValue]
    intents: Mapping[CapabilityKey, CapabilityIntent]
    required_capabilities: frozenset[CapabilityKey]
    available_capabilities: frozenset[CapabilityKey]
    pending_capabilities: frozenset[CapabilityKey]
    unavailable_capabilities: Mapping[CapabilityKey, str]
    capability_minimums: Mapping[CapabilityKey, Mapping[str, JSONValue]]
    request_defaults: RequestDefaults
    output_channels: frozenset[str]
    required_output_channels: frozenset[str]
    serving_requirements: tuple[ServingRequirement, ...]
    route_intent: RouteIntent
    connection_scope: ConnectionScope
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_defaults",
            _freeze_request_defaults(self.request_defaults, "request_defaults"),
        )
        object.__setattr__(
            self,
            "declared_capabilities",
            _freeze_json(self.declared_capabilities, "declared_capabilities"),
        )
        object.__setattr__(
            self,
            "intents",
            MappingProxyType(
                {
                    key: _freeze_capability_intent(intent, f"intents.{key}")
                    for key, intent in self.intents.items()
                }
            ),
        )
        for name in (
            "required_capabilities",
            "available_capabilities",
            "pending_capabilities",
            "output_channels",
            "required_output_channels",
        ):
            object.__setattr__(self, name, frozenset(getattr(self, name)))
        object.__setattr__(
            self, "serving_requirements", tuple(self.serving_requirements)
        )
        object.__setattr__(
            self,
            "unavailable_capabilities",
            MappingProxyType(dict(self.unavailable_capabilities)),
        )
        object.__setattr__(
            self,
            "capability_minimums",
            MappingProxyType(
                {
                    key: _freeze_json(value, f"capability_minimums.{key}")
                    for key, value in self.capability_minimums.items()
                }
            ),
        )
        object.__setattr__(
            self, "fingerprint", fingerprint_mapping(self._plan_payload())
        )

    def _plan_payload(self) -> dict[str, JSONValue]:
        return {
            "binding_id": self.binding_id,
            "root_deployment_key": self.root_deployment_key,
            "requested_model_id": self.requested_model_id,
            "dialect_id": self.dialect_id,
            "declared_capabilities": _thaw_json(self.declared_capabilities),
            "intents": {
                key: self.intents[key].to_json_value() for key in sorted(self.intents)
            },
            "required_capabilities": cast(
                JSONValue, sorted(self.required_capabilities)
            ),
            "available_capabilities": cast(
                JSONValue, sorted(self.available_capabilities)
            ),
            "pending_capabilities": cast(JSONValue, sorted(self.pending_capabilities)),
            "unavailable_capabilities": {
                key: self.unavailable_capabilities[key]
                for key in sorted(self.unavailable_capabilities)
            },
            "capability_minimums": {
                key: _thaw_json(self.capability_minimums[key])
                for key in sorted(self.capability_minimums)
            },
            "request_defaults": self.request_defaults.to_json_value(),
            "output_channels": cast(JSONValue, sorted(self.output_channels)),
            "required_output_channels": cast(
                JSONValue, sorted(self.required_output_channels)
            ),
            "serving_requirements": [
                item.to_json_value() for item in self.serving_requirements
            ],
            "route_intent": {"service_role": self.route_intent.service_role},
            "connection_scope": {
                "credential_scope": self.connection_scope.credential_scope,
                "retry_policy": self.connection_scope.retry_policy,
                "quota_scope": self.connection_scope.quota_scope,
            },
        }

    def to_json_value(self) -> dict[str, JSONValue]:
        value = self._plan_payload()
        value["fingerprint"] = self.fingerprint
        return value


@dataclass(frozen=True)
class DeploymentCapabilityPlan:
    """One root's aggregated launch patch and named verification checks."""

    root_deployment_key: str
    engine_id: str
    desired_plan_fingerprint: str | None
    recipe_parameters: Mapping[str, JSONValue]
    explicit_parameters: Mapping[str, JSONValue]
    serving_requirements: tuple[ServingRequirement, ...]
    launch_patch: Mapping[str, JSONValue]
    setup_checks: tuple[DeferredCheck, ...]
    request_checks: tuple[DeferredCheck, ...]
    outcome_kinds: Mapping[CapabilityKey, str]
    outcome_evidence: Mapping[CapabilityKey, Mapping[str, JSONValue]]
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if self.desired_plan_fingerprint is not None:
            _nonempty(self.desired_plan_fingerprint, "desired_plan_fingerprint")
        object.__setattr__(
            self,
            "recipe_parameters",
            _freeze_json(self.recipe_parameters, "recipe_parameters"),
        )
        object.__setattr__(
            self,
            "explicit_parameters",
            _freeze_json(self.explicit_parameters, "explicit_parameters"),
        )
        object.__setattr__(
            self,
            "launch_patch",
            _freeze_json(self.launch_patch, "launch_patch"),
        )
        for name in ("serving_requirements", "setup_checks", "request_checks"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(
            self,
            "outcome_kinds",
            MappingProxyType(dict(self.outcome_kinds)),
        )
        object.__setattr__(
            self,
            "outcome_evidence",
            MappingProxyType(
                {
                    key: _freeze_json(value, f"outcome_evidence.{key}")
                    for key, value in self.outcome_evidence.items()
                }
            ),
        )
        object.__setattr__(
            self, "fingerprint", fingerprint_mapping(self._plan_payload())
        )

    def _plan_payload(self) -> dict[str, JSONValue]:
        return {
            "root_deployment_key": self.root_deployment_key,
            "engine_id": self.engine_id,
            "desired_plan_fingerprint": self.desired_plan_fingerprint,
            "recipe_parameters": _thaw_json(self.recipe_parameters),
            "explicit_parameters": _thaw_json(self.explicit_parameters),
            "serving_requirements": [
                item.to_json_value() for item in self.serving_requirements
            ],
            "launch_patch": _thaw_json(self.launch_patch),
            "setup_checks": [check.to_json_value() for check in self.setup_checks],
            "request_checks": [check.to_json_value() for check in self.request_checks],
            "outcome_kinds": {
                key: self.outcome_kinds[key] for key in sorted(self.outcome_kinds)
            },
            "outcome_evidence": {
                key: _thaw_json(self.outcome_evidence[key])
                for key in sorted(self.outcome_evidence)
            },
        }

    def to_json_value(self) -> dict[str, JSONValue]:
        value = self._plan_payload()
        value["fingerprint"] = self.fingerprint
        return value


@dataclass(frozen=True)
class RuntimeBindingPlan:
    """Immutable, bindable plan for one realized deployment and route."""

    binding_id: str
    root_deployment_key: str
    requested_model_id: str
    dialect_id: str
    declared_capabilities: Mapping[str, JSONValue]
    effective_capabilities: Mapping[str, JSONValue]
    available_capabilities: frozenset[str]
    capability_minimums: Mapping[str, Mapping[str, JSONValue]]
    request_defaults: RequestDefaults
    required_output_channels: frozenset[str]
    request_checks: tuple[DeferredCheck, ...]
    deployment_fingerprint: str
    resolved_route: ResolvedRoute
    connection_identity: ConnectionIdentity
    binding_plan_fingerprint: str
    deployment_plan_fingerprint: str
    fingerprint: str = field(init=False)
    verification_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_defaults",
            _freeze_request_defaults(self.request_defaults, "request_defaults"),
        )
        object.__setattr__(
            self,
            "declared_capabilities",
            _freeze_json(self.declared_capabilities, "declared_capabilities"),
        )
        object.__setattr__(
            self,
            "effective_capabilities",
            _freeze_json(self.effective_capabilities, "effective_capabilities"),
        )
        object.__setattr__(
            self, "available_capabilities", frozenset(self.available_capabilities)
        )
        object.__setattr__(
            self,
            "required_output_channels",
            frozenset(self.required_output_channels),
        )
        object.__setattr__(self, "request_checks", tuple(self.request_checks))
        object.__setattr__(
            self,
            "capability_minimums",
            MappingProxyType(
                {
                    key: _freeze_json(value, f"capability_minimums.{key}")
                    for key, value in self.capability_minimums.items()
                }
            ),
        )
        verification_fingerprint = fingerprint_mapping(self._verification_payload())
        object.__setattr__(self, "verification_fingerprint", verification_fingerprint)
        fingerprint_payload = self._plan_payload()
        fingerprint_payload["verification_fingerprint"] = verification_fingerprint
        object.__setattr__(
            self, "fingerprint", fingerprint_mapping(fingerprint_payload)
        )

    def _verification_payload(self) -> dict[str, JSONValue]:
        """Return the realized capability evidence covered by verification."""

        return {
            "binding_plan_fingerprint": self.binding_plan_fingerprint,
            "deployment_plan_fingerprint": self.deployment_plan_fingerprint,
            "deployment_fingerprint": self.deployment_fingerprint,
            "dialect_id": self.dialect_id,
            "requested_model_id": self.requested_model_id,
            "effective_capabilities": _thaw_json(self.effective_capabilities),
            "available_capabilities": cast(
                JSONValue, sorted(self.available_capabilities)
            ),
            "capability_minimums": {
                key: _thaw_json(self.capability_minimums[key])
                for key in sorted(self.capability_minimums)
            },
            "request_defaults": self.request_defaults.to_json_value(),
            "required_output_channels": cast(
                JSONValue, sorted(self.required_output_channels)
            ),
            "request_checks": [check.to_json_value() for check in self.request_checks],
        }

    def _plan_payload(self) -> dict[str, JSONValue]:
        """Return every caller-independent field in the bindable plan."""

        return {
            "binding_id": self.binding_id,
            "root_deployment_key": self.root_deployment_key,
            "requested_model_id": self.requested_model_id,
            "dialect_id": self.dialect_id,
            "declared_capabilities": _thaw_json(self.declared_capabilities),
            "effective_capabilities": _thaw_json(self.effective_capabilities),
            "available_capabilities": cast(
                JSONValue, sorted(self.available_capabilities)
            ),
            "capability_minimums": {
                key: _thaw_json(self.capability_minimums[key])
                for key in sorted(self.capability_minimums)
            },
            "request_defaults": self.request_defaults.to_json_value(),
            "required_output_channels": cast(
                JSONValue, sorted(self.required_output_channels)
            ),
            "request_checks": [check.to_json_value() for check in self.request_checks],
            "deployment_fingerprint": self.deployment_fingerprint,
            "resolved_route": {
                "service_role": self.resolved_route.service_role,
                "endpoint": self.resolved_route.endpoint,
                "connection_family": self.resolved_route.connection_family,
                "fingerprint": self.resolved_route.fingerprint,
            },
            "connection_identity": {
                "endpoint": self.connection_identity.endpoint,
                "connection_family": self.connection_identity.connection_family,
                "credential_scope": self.connection_identity.credential_scope,
                "retry_policy": self.connection_identity.retry_policy,
                "quota_scope": self.connection_identity.quota_scope,
            },
            "binding_plan_fingerprint": self.binding_plan_fingerprint,
            "deployment_plan_fingerprint": self.deployment_plan_fingerprint,
        }

    def to_json_value(self) -> dict[str, JSONValue]:
        value = self._plan_payload()
        value["fingerprint"] = self.fingerprint
        value["verification_fingerprint"] = self.verification_fingerprint
        return value


@dataclass(frozen=True)
class ReconcileResult:
    """All successful plans and all deterministic diagnostics for one batch."""

    binding_plans: Mapping[str, BindingCapabilityPlan]
    deployment_plans: Mapping[str, DeploymentCapabilityPlan]
    runtime_plans: Mapping[str, RuntimeBindingPlan]
    diagnostics: tuple[ReconcileDiagnostic, ...]

    def __post_init__(self) -> None:
        for name in ("binding_plans", "deployment_plans", "runtime_plans"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    @property
    def errors(self) -> tuple[ReconcileDiagnostic, ...]:
        return tuple(
            item
            for item in self.diagnostics
            if item.severity is ReconcileSeverity.ERROR
        )

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_json_value(self) -> dict[str, JSONValue]:
        return {
            "binding_plans": {
                key: self.binding_plans[key].to_json_value()
                for key in sorted(self.binding_plans)
            },
            "deployment_plans": {
                key: self.deployment_plans[key].to_json_value()
                for key in sorted(self.deployment_plans)
            },
            "runtime_plans": {
                key: self.runtime_plans[key].to_json_value()
                for key in sorted(self.runtime_plans)
            },
            "diagnostics": [item.to_json_value() for item in self.diagnostics],
        }


@dataclass(frozen=True)
class BindingReconcileResult:
    plan: BindingCapabilityPlan | None
    diagnostics: tuple[ReconcileDiagnostic, ...]


def reconcile_binding(binding: BindingReconcileInput) -> BindingReconcileResult:
    """Join task/declaration, dialect, and model columns for one binding."""

    diagnostics: list[ReconcileDiagnostic] = []
    try:
        spec = get_dialect_spec(binding.dialect_id)
        decisions = capability_decisions_for(binding.dialect_id)
    except (UnknownDialect, DialectNotImplemented) as exc:
        diagnostics.append(_binding_error(binding, "dialect_unavailable", str(exc)))
        return BindingReconcileResult(None, tuple(diagnostics))

    _validate_task_shape(binding, spec, diagnostics)
    try:
        normalized = normalize_capability_declarations(
            binding.declarations,
            dialect_id=binding.dialect_id,
            outcomes=cast(
                Mapping[str, DialectCapabilityStatus | str],
                spec.capability_outcomes,
            ),
        )
    except CapabilityConfigError as exc:
        diagnostics.append(
            _binding_error(binding, "invalid_capability_declaration", str(exc))
        )
        return BindingReconcileResult(None, tuple(diagnostics))

    task_intents = _task_intents(binding.requirements)
    required_intents, request_diagnostics = _merge_request_intents(
        binding,
        task_intents,
        binding.request_intents,
    )
    diagnostics.extend(request_diagnostics)
    intents, disabled, intent_diagnostics = _merge_declarations(
        binding, normalized, required_intents
    )
    diagnostics.extend(intent_diagnostics)

    available: set[CapabilityKey] = set()
    pending: set[CapabilityKey] = set()
    unavailable: dict[CapabilityKey, str] = {}
    output_channels: set[str] = set()
    serving_requirements: list[ServingRequirement] = []
    default_values: dict[str, JSONValue] = {}

    for key in CAPABILITY_KEYS:
        intent = intents.get(key)
        required = intent is not None and intent.required
        if key in disabled:
            unavailable[key] = "explicitly disabled"
            continue

        decision = decisions[key]
        if isinstance(decision, Unsupported):
            unavailable[key] = decision.reason
            if required:
                assert intent is not None
                diagnostics.append(
                    _binding_error(
                        binding,
                        "dialect_unsupported",
                        decision.reason,
                        capability=key,
                        sources=intent.sources,
                    )
                )
            continue

        assert isinstance(decision, Supported)
        output_channels.update(decision.binding.response_channels)
        declared = normalized.get(key)
        if declared is not None and declared is not False:
            try:
                decision.binding.validate_config(declared)
            except (CapabilityConfigError, TypeError, ValueError) as exc:
                diagnostics.append(
                    _binding_error(
                        binding,
                        "dialect_config_invalid",
                        str(exc),
                        capability=key,
                        sources=intents[key].sources if required else (),
                    )
                )

        entry = binding.model_profile.entries.get(key)
        if entry is None:
            reason = "model profile provides no support outcome"
            unavailable[key] = reason
            if required:
                assert intent is not None
                diagnostics.append(
                    _binding_error(
                        binding,
                        "model_outcome_missing",
                        reason,
                        capability=key,
                        sources=intent.sources,
                    )
                )
            continue
        if entry.status is ModelCapabilityStatus.UNSUPPORTED:
            reason = entry.reason or "model profile marks capability unsupported"
            unavailable[key] = reason
            if required:
                assert intent is not None
                diagnostics.append(
                    _binding_error(
                        binding,
                        "model_unsupported",
                        reason,
                        capability=key,
                        sources=intent.sources,
                    )
                )
            continue
        if not required:
            unavailable[key] = "capability was not required for this binding"
            continue

        assert intent is not None
        if entry.status is ModelCapabilityStatus.UNKNOWN:
            assert entry.verifier is not None
            reason = entry.reason or "model support is unknown"
            verifier = entry.verifier
        else:
            reason = (
                entry.reason
                or "model support is declared, but serving support still requires "
                "an explicit outcome"
            )
            verifier = entry.verifier

        unavailable[key] = reason
        pending.add(key)
        serving_requirements.append(
            ServingRequirement(
                capability=key,
                minimums=intent.minimums,
                sources=intent.sources,
                verifier=verifier,
                reason=reason,
            )
        )
        for path, value in intent.request_defaults.values.items():
            if path in default_values:
                diagnostics.append(
                    _binding_error(
                        binding,
                        "duplicate_request_default_owner",
                        f"request default {path!r} has more than one owner",
                        capability=key,
                        sources=intent.sources,
                    )
                )
            default_values[path] = value

    required_output_channels = _required_task_outputs(binding.requirements)
    missing_output = required_output_channels - output_channels
    if missing_output:
        diagnostics.append(
            _binding_error(
                binding,
                "required_output_unmapped",
                "dialect has no output mapping for: "
                + ", ".join(sorted(missing_output)),
            )
        )

    if any(item.severity is ReconcileSeverity.ERROR for item in diagnostics):
        return BindingReconcileResult(None, tuple(diagnostics))

    minimums = {
        key: intent.minimums for key, intent in intents.items() if intent.minimums
    }
    declared_json = capability_declarations_to_json(normalized)
    plan = BindingCapabilityPlan(
        binding_id=binding.binding_id,
        root_deployment_key=binding.root_deployment_key,
        requested_model_id=binding.requested_model_id,
        dialect_id=binding.dialect_id,
        declared_capabilities=MappingProxyType(declared_json),
        intents=MappingProxyType(dict(intents)),
        required_capabilities=frozenset(
            key for key, intent in intents.items() if intent.required
        ),
        available_capabilities=frozenset(available),
        pending_capabilities=frozenset(pending),
        unavailable_capabilities=MappingProxyType(unavailable),
        capability_minimums=MappingProxyType(minimums),
        request_defaults=RequestDefaults(default_values),
        output_channels=frozenset(output_channels),
        required_output_channels=frozenset(required_output_channels),
        serving_requirements=tuple(
            sorted(serving_requirements, key=lambda item: item.capability)
        ),
        route_intent=binding.route_intent,
        connection_scope=binding.connection_scope,
    )
    return BindingReconcileResult(plan, tuple(diagnostics))


def reconcile(
    batch: ReconcileBatch,
    serving_reconciler: ServingReconciler | None = None,
) -> ReconcileResult:
    """Purely reconcile every binding, root deployment, and realized route."""

    binding_plans: dict[str, BindingCapabilityPlan] = {}
    deployment_plans: dict[str, DeploymentCapabilityPlan] = {}
    runtime_plans: dict[str, RuntimeBindingPlan] = {}
    diagnostics: list[ReconcileDiagnostic] = []

    for binding in sorted(batch.bindings, key=lambda item: item.binding_id):
        result = reconcile_binding(binding)
        diagnostics.extend(result.diagnostics)
        if result.plan is not None:
            binding_plans[binding.binding_id] = result.plan

    plans_by_root: dict[str, list[BindingCapabilityPlan]] = {}
    for plan in binding_plans.values():
        plans_by_root.setdefault(plan.root_deployment_key, []).append(plan)

    for root_key in sorted(plans_by_root):
        deployment_input = batch.deployments.get(root_key)
        if deployment_input is None:
            diagnostics.append(
                ReconcileDiagnostic(
                    ReconcileSeverity.ERROR,
                    "deployment_input_missing",
                    f"no deployment input exists for root {root_key!r}",
                    root_deployment_key=root_key,
                )
            )
            continue
        root_plans = sorted(plans_by_root[root_key], key=lambda item: item.binding_id)
        deployment_plan, outcomes, deployment_diagnostics = reconcile_deployment(
            root_plans, deployment_input, serving_reconciler
        )
        diagnostics.extend(deployment_diagnostics)
        deployment_plans[root_key] = deployment_plan
        if any(
            diagnostic.severity is ReconcileSeverity.ERROR
            for diagnostic in deployment_diagnostics
        ):
            continue
        if deployment_input.deployment is None or deployment_plan.setup_checks:
            continue
        for binding_plan in root_plans:
            try:
                runtime = _finalize_runtime_plan(
                    binding_plan,
                    deployment_input.deployment,
                    deployment_plan,
                    outcomes,
                )
            except ValueError as exc:
                diagnostics.append(
                    ReconcileDiagnostic(
                        ReconcileSeverity.ERROR,
                        "route_resolution_failed",
                        str(exc),
                        binding_id=binding_plan.binding_id,
                        root_deployment_key=root_key,
                    )
                )
            else:
                runtime_plans[binding_plan.binding_id] = runtime

    if any(
        diagnostic.severity is ReconcileSeverity.ERROR for diagnostic in diagnostics
    ):
        runtime_plans.clear()

    return ReconcileResult(
        binding_plans=binding_plans,
        deployment_plans=deployment_plans,
        runtime_plans=runtime_plans,
        diagnostics=tuple(diagnostics),
    )


def reconcile_deployment(
    binding_plans: Iterable[BindingCapabilityPlan],
    deployment: DeploymentReconcileInput,
    serving_reconciler: ServingReconciler | None = None,
) -> tuple[
    DeploymentCapabilityPlan,
    Mapping[CapabilityKey, ServingOutcome],
    tuple[ReconcileDiagnostic, ...],
]:
    """Aggregate serving requirements once and reconcile one root deployment."""

    plans = tuple(sorted(binding_plans, key=lambda item: item.binding_id))
    if any(
        plan.root_deployment_key != deployment.root_deployment_key for plan in plans
    ):
        raise ValueError("binding plan belongs to another root deployment")
    diagnostics: list[ReconcileDiagnostic] = []
    desired_plan = deployment.effective_plan
    desired_roles = () if desired_plan is None else desired_plan.service_roles
    if desired_roles:
        for binding in plans:
            requested_role = binding.route_intent.service_role
            if requested_role is not None and requested_role not in desired_roles:
                diagnostics.append(
                    ReconcileDiagnostic(
                        ReconcileSeverity.ERROR,
                        "route_role_missing_from_plan",
                        f"requested service role {requested_role!r} is absent from "
                        f"desired deployment roles {list(desired_roles)!r}",
                        binding_id=binding.binding_id,
                        root_deployment_key=deployment.root_deployment_key,
                    )
                )
            elif requested_role is None and len(desired_roles) > 1:
                diagnostics.append(
                    ReconcileDiagnostic(
                        ReconcileSeverity.ERROR,
                        "route_role_ambiguous_prelaunch",
                        "desired deployment has multiple service roles; set "
                        "model service_role before launch",
                        binding_id=binding.binding_id,
                        root_deployment_key=deployment.root_deployment_key,
                    )
                )

    try:
        requirements = _aggregate_serving_requirements(plans)
    except ValueError as exc:
        requirements = ()
        diagnostics.append(
            _deployment_error(
                deployment,
                "serving_verifier_conflict",
                str(exc),
            )
        )
    if serving_reconciler is None:
        outcomes: dict[CapabilityKey, ServingOutcome] = {
            requirement.capability: CannotVerify(
                CheckStage.SETUP,
                requirement.verifier or "unconfigured_serving_reconciler",
                requirement.reason
                or "no serving reconciler can verify this requirement",
            )
            for requirement in requirements
        }
    else:
        outcomes = dict(serving_reconciler.reconcile(requirements, deployment))

    expected = {requirement.capability for requirement in requirements}
    extra = set(outcomes) - expected
    missing = expected - set(outcomes)
    if extra:
        diagnostics.append(
            _deployment_error(
                deployment,
                "serving_outcome_extra",
                "serving reconciler returned unrequested capabilities: "
                + ", ".join(sorted(extra)),
            )
        )
    for capability in sorted(missing):
        diagnostics.append(
            _deployment_error(
                deployment,
                "serving_outcome_missing",
                "serving reconciler omitted a required outcome",
                capability=capability,
            )
        )

    launch_patch: dict[str, JSONValue] = {}
    setup_checks: list[DeferredCheck] = []
    request_checks: list[DeferredCheck] = []
    outcome_kinds: dict[CapabilityKey, str] = {}
    outcome_evidence: dict[CapabilityKey, Mapping[str, JSONValue]] = {}
    requirement_by_key = {item.capability: item for item in requirements}
    for capability in sorted(expected & set(outcomes)):
        outcome = outcomes[capability]
        requirement = requirement_by_key[capability]
        if isinstance(outcome, Configured):
            outcome_kinds[capability] = "configured"
            outcome_evidence[capability] = outcome.evidence
            _merge_launch_patch(
                launch_patch,
                outcome.launch_patch,
                deployment,
                capability,
                diagnostics,
            )
            request_checks.extend(outcome.request_checks)
        elif isinstance(outcome, CannotVerify):
            outcome_kinds[capability] = "cannot_verify"
            check = DeferredCheck(
                capability,
                outcome.stage,
                outcome.verifier,
                outcome.reason,
            )
            if outcome.stage is CheckStage.REQUEST:
                request_checks.append(check)
                severity = ReconcileSeverity.WARNING
            else:
                setup_checks.append(check)
                severity = (
                    ReconcileSeverity.WARNING
                    if deployment.deployment is None and desired_plan is not None
                    else ReconcileSeverity.ERROR
                )
            diagnostics.append(
                ReconcileDiagnostic(
                    severity,
                    "cannot_verify",
                    outcome.reason,
                    root_deployment_key=deployment.root_deployment_key,
                    capability=capability,
                    sources=requirement.sources,
                )
            )
        elif isinstance(outcome, ServingUnsupported):
            outcome_kinds[capability] = "unsupported"
            message = outcome.reason
            if outcome.remedy is not None:
                message = f"{message}; remedy: {outcome.remedy}"
            diagnostics.append(
                ReconcileDiagnostic(
                    ReconcileSeverity.ERROR,
                    "serving_unsupported",
                    message,
                    root_deployment_key=deployment.root_deployment_key,
                    capability=capability,
                    sources=requirement.sources,
                )
            )
        else:
            raise TypeError(f"invalid serving outcome for {capability!r}")

    if launch_patch and desired_plan is None:
        diagnostics.append(
            _deployment_error(
                deployment,
                "launch_patch_unavailable",
                "serving reconciliation derived launch parameters for an external "
                "deployment with no SiEval deployment plan",
            )
        )
    elif (
        launch_patch
        and deployment.deployment is not None
        and deployment.prelaunch_plan is None
    ):
        diagnostics.append(
            _deployment_error(
                deployment,
                "unfrozen_realized_launch_patch",
                "serving reconciliation derived launch parameters for an already "
                "realized deployment without a frozen pre-launch plan",
            )
        )

    plan = DeploymentCapabilityPlan(
        root_deployment_key=deployment.root_deployment_key,
        engine_id=deployment.engine_id,
        desired_plan_fingerprint=(
            None if desired_plan is None else desired_plan.fingerprint
        ),
        recipe_parameters=deployment.recipe_parameters,
        explicit_parameters=deployment.explicit_parameters,
        serving_requirements=requirements,
        launch_patch=MappingProxyType(launch_patch),
        setup_checks=tuple(setup_checks),
        request_checks=tuple(request_checks),
        outcome_kinds=MappingProxyType(outcome_kinds),
        outcome_evidence=MappingProxyType(outcome_evidence),
    )
    if (
        deployment.prelaunch_plan is not None
        and deployment.prelaunch_plan.desired_plan_fingerprint
        != plan.desired_plan_fingerprint
    ):
        diagnostics.append(
            _deployment_error(
                deployment,
                "postlaunch_plan_drift",
                "realized deployment plan does not match the pre-launch desired "
                "plan; binding is refused",
            )
        )
    if deployment.prelaunch_plan is not None and (
        deployment.prelaunch_plan.recipe_parameters != plan.recipe_parameters
        or deployment.prelaunch_plan.explicit_parameters != plan.explicit_parameters
    ):
        diagnostics.append(
            _deployment_error(
                deployment,
                "postlaunch_parameter_drift",
                "post-launch reconciliation received different recipe or explicit "
                "engine parameters",
            )
        )
    if (
        deployment.prelaunch_plan is not None
        and deployment.prelaunch_plan.launch_patch != plan.launch_patch
    ):
        diagnostics.append(
            _deployment_error(
                deployment,
                "postlaunch_patch_drift",
                "post-launch verification derived a different launch patch; "
                "relaunch is required",
            )
        )
    return plan, MappingProxyType(outcomes), tuple(diagnostics)


def _task_intents(
    requirements: AggregatedTaskRequirements,
) -> dict[CapabilityKey, CapabilityIntent]:
    intents: dict[CapabilityKey, CapabilityIntent] = {}
    if requirements.input_scoring:
        intents["input_scoring"] = CapabilityIntent(
            "input_scoring",
            True,
            sources=tuple(requirements.input_scoring_sources),
        )
    if requirements.sampled_logprobs:
        intents["sampled_logprobs"] = CapabilityIntent(
            "sampled_logprobs",
            True,
            sources=tuple(requirements.sampled_logprobs_sources),
        )
    if requirements.min_top_logprobs is not None:
        sources = {
            source
            for minimum, names in requirements.min_top_logprobs_sources.items()
            if minimum <= requirements.min_top_logprobs
            for source in names
        }
        intents["top_logprobs"] = CapabilityIntent(
            "top_logprobs",
            True,
            minimums={"minimum": requirements.min_top_logprobs},
            sources=tuple(sorted(sources)),
        )
    if InputModality.IMAGE in requirements.input_modalities:
        sources = set(requirements.modality_sources.get(InputModality.IMAGE, ()))
        intents["multimodal_input"] = CapabilityIntent(
            "multimodal_input",
            True,
            minimums={"modalities": [InputModality.IMAGE.value]},
            sources=tuple(sorted(sources)),
        )
    return intents


def _merge_declarations(
    binding: BindingReconcileInput,
    declarations: Mapping[CapabilityKey, NormalizedCapabilityValue],
    required_intents: Mapping[CapabilityKey, CapabilityIntent],
) -> tuple[
    dict[CapabilityKey, CapabilityIntent],
    set[CapabilityKey],
    tuple[ReconcileDiagnostic, ...],
]:
    intents = dict(required_intents)
    disabled: set[CapabilityKey] = set()
    diagnostics: list[ReconcileDiagnostic] = []
    for key in CAPABILITY_KEYS:
        declaration = declarations.get(key)
        if declaration is None:
            continue
        if declaration is False:
            disabled.add(key)
            required_intent = required_intents.get(key)
            if required_intent is not None and required_intent.required:
                diagnostics.append(
                    _binding_error(
                        binding,
                        "task_capability_disabled",
                        "explicit false conflicts with a task/request requirement",
                        capability=key,
                        sources=required_intent.sources,
                    )
                )
            continue

        declared_intent = _declaration_intent(
            key, declaration, binding.declaration_source
        )
        required_intent = required_intents.get(key)
        if required_intent is None:
            intents[key] = declared_intent
            continue
        merged, conflict = _merge_two_intents(required_intent, declared_intent)
        if conflict is not None:
            diagnostics.append(
                _binding_error(
                    binding,
                    "capability_minimum_weakened",
                    conflict,
                    capability=key,
                    sources=required_intent.sources,
                )
            )
        else:
            intents[key] = merged
    return intents, disabled, tuple(diagnostics)


def _merge_request_intents(
    binding: BindingReconcileInput,
    task_intents: Mapping[CapabilityKey, CapabilityIntent],
    request_intents: Mapping[CapabilityKey, CapabilityIntent],
) -> tuple[
    dict[CapabilityKey, CapabilityIntent],
    tuple[ReconcileDiagnostic, ...],
]:
    """Join dynamic legacy request intent without making it a default owner."""

    merged_intents = dict(task_intents)
    diagnostics: list[ReconcileDiagnostic] = []
    for key in CAPABILITY_KEYS:
        request_intent = request_intents.get(key)
        if request_intent is None:
            continue
        task_intent = task_intents.get(key)
        if task_intent is None:
            merged_intents[key] = request_intent
            continue
        merged, conflict = _merge_two_intents(task_intent, request_intent)
        if conflict is not None:
            diagnostics.append(
                _binding_error(
                    binding,
                    "request_capability_minimum_weakened",
                    conflict,
                    capability=key,
                    sources=request_intent.sources,
                )
            )
        else:
            merged_intents[key] = merged
    return merged_intents, tuple(diagnostics)


def _declaration_intent(
    key: CapabilityKey,
    value: NormalizedCapabilityValue,
    source: str,
) -> CapabilityIntent:
    assert value is not False
    minimums: dict[str, JSONValue] = {}
    defaults: dict[str, JSONValue] = {}
    if isinstance(value, TopLogprobsOptions):
        minimums["minimum"] = value.minimum
    elif isinstance(value, ReasoningOptions):
        for name in ("effort", "budget_tokens", "summary"):
            option = getattr(value, name)
            if option is not None:
                defaults[f"reasoning.{name}"] = option
    elif isinstance(value, FunctionToolsOptions) and value.parallel is not None:
        defaults["tools.parallel"] = value.parallel
    elif isinstance(value, HostedToolsOptions) and value.kinds:
        minimums["kinds"] = list(value.kinds)
    elif isinstance(value, StructuredOutputOptions) and value.formats:
        minimums["formats"] = list(value.formats)
    elif isinstance(value, MultimodalInputOptions) and value.modalities:
        minimums["modalities"] = list(value.modalities)
    return CapabilityIntent(
        key=key,
        required=True,
        minimums=minimums,
        request_defaults=RequestDefaults(defaults),
        sources=(source,),
    )


def _merge_two_intents(
    task: CapabilityIntent,
    declared: CapabilityIntent,
) -> tuple[CapabilityIntent, str | None]:
    task_minimum = task.minimums.get("minimum")
    declared_minimum = declared.minimums.get("minimum")
    if (
        isinstance(task_minimum, int)
        and isinstance(declared_minimum, int)
        and declared_minimum < task_minimum
    ):
        return task, (
            f"declared minimum {declared_minimum} is weaker than task minimum "
            f"{task_minimum}"
        )
    for name, task_value in task.minimums.items():
        declared_value = declared.minimums.get(name)
        if not isinstance(task_value, list) or not isinstance(declared_value, list):
            continue
        missing = set(task_value) - set(declared_value)
        if missing:
            return task, (
                f"declared {name} omits task-required value(s): "
                + ", ".join(sorted(str(value) for value in missing))
            )
    minimums = dict(task.minimums)
    minimums.update(declared.minimums)
    return (
        CapabilityIntent(
            task.key,
            task.required or declared.required,
            minimums=minimums,
            request_defaults=declared.request_defaults,
            sources=tuple(sorted(set(task.sources) | set(declared.sources))),
        ),
        None,
    )


def _required_task_outputs(
    requirements: AggregatedTaskRequirements,
) -> set[str]:
    channels: set[str] = set()
    if requirements.input_scoring:
        channels.add("input_scoring")
    if requirements.sampled_logprobs:
        channels.add("logprobs")
    if requirements.min_top_logprobs is not None:
        channels.add("top_logprobs")
    return channels


def _validate_task_shape(
    binding: BindingReconcileInput,
    spec: DialectSpec,
    diagnostics: list[ReconcileDiagnostic],
) -> None:
    unsupported_kinds = {
        kind
        for kind in binding.requirements.input
        if kind.value not in spec.input_kinds
    }
    for kind in sorted(unsupported_kinds):
        diagnostics.append(
            _binding_error(
                binding,
                "input_kind_unsupported",
                f"dialect {spec.dialect_id!r} does not accept {kind.value!r} input",
                sources=tuple(binding.requirements.input_sources.get(kind, ())),
            )
        )
    unsupported_modalities = {
        modality
        for modality in binding.requirements.input_modalities
        if modality.value not in spec.input_modalities
    }
    for modality in sorted(unsupported_modalities):
        diagnostics.append(
            _binding_error(
                binding,
                "input_modality_unsupported",
                f"dialect {spec.dialect_id!r} does not accept {modality.value!r} input",
                sources=tuple(binding.requirements.modality_sources.get(modality, ())),
            )
        )


def _aggregate_serving_requirements(
    plans: Iterable[BindingCapabilityPlan],
) -> tuple[ServingRequirement, ...]:
    grouped: dict[CapabilityKey, list[ServingRequirement]] = {}
    for plan in plans:
        for requirement in plan.serving_requirements:
            grouped.setdefault(requirement.capability, []).append(requirement)
    aggregated: list[ServingRequirement] = []
    for capability in CAPABILITY_KEYS:
        requirements = grouped.get(capability)
        if not requirements:
            continue
        minimums: dict[str, JSONValue] = {}
        sources: set[str] = set()
        verifiers: set[str] = set()
        reasons: set[str] = set()
        for requirement in requirements:
            sources.update(requirement.sources)
            if requirement.verifier is not None:
                verifiers.add(requirement.verifier)
            if requirement.reason is not None:
                reasons.add(requirement.reason)
            for key, value in requirement.minimums.items():
                if key not in minimums:
                    minimums[key] = value
                    continue
                current = minimums[key]
                if (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and isinstance(current, int)
                    and not isinstance(current, bool)
                ):
                    minimums[key] = max(value, current)
                elif _canonical_json(current) == _canonical_json(value):
                    minimums[key] = value
                else:
                    values = {
                        _canonical_json(item): item
                        for item in (
                            *_json_sequence(current),
                            *_json_sequence(value),
                        )
                    }
                    minimums[key] = cast(
                        JSONValue,
                        sorted(
                            values.values(),
                            key=_canonical_json,
                        ),
                    )
        if len(verifiers) > 1:
            raise ValueError(
                f"capability {capability!r} has conflicting serving verifiers: "
                + ", ".join(sorted(verifiers))
            )
        aggregated.append(
            ServingRequirement(
                capability,
                minimums=minimums,
                sources=tuple(sorted(sources)),
                verifier=" + ".join(sorted(verifiers)) if verifiers else None,
                reason="; ".join(sorted(reasons)) if reasons else None,
            )
        )
    return tuple(aggregated)


def _merge_launch_patch(
    target: dict[str, JSONValue],
    patch: Mapping[str, JSONValue],
    deployment: DeploymentReconcileInput,
    capability: CapabilityKey,
    diagnostics: list[ReconcileDiagnostic],
) -> None:
    for key in sorted(patch):
        value = patch[key]
        explicit = deployment.explicit_parameters.get(key)
        if key in deployment.explicit_parameters:
            if _canonical_json(explicit) != _canonical_json(value):
                safe_value = _canonical_json(value)
                diagnostics.append(
                    _deployment_error(
                        deployment,
                        "explicit_engine_parameter_conflict",
                        f"explicit parameter {key!r} conflicts with the "
                        "serving reconciler's correctness-safe value; set "
                        f"{key!r} to {safe_value}",
                        capability=capability,
                    )
                )
            continue
        if key in target and _canonical_json(target[key]) != _canonical_json(value):
            diagnostics.append(
                _deployment_error(
                    deployment,
                    "contradictory_launch_patch",
                    f"capabilities derived contradictory values for {key!r}",
                    capability=capability,
                )
            )
            continue
        target[key] = value


def _finalize_runtime_plan(
    binding: BindingCapabilityPlan,
    deployment: Deployment,
    deployment_plan: DeploymentCapabilityPlan,
    outcomes: Mapping[CapabilityKey, ServingOutcome],
) -> RuntimeBindingPlan:
    spec = get_dialect_spec(binding.dialect_id)
    route = resolve_route(
        deployment,
        binding.dialect_id,
        spec.connection_family,
        binding.route_intent,
    )
    scope = binding.connection_scope
    identity = ConnectionIdentity(
        endpoint=route.endpoint,
        connection_family=route.connection_family,
        credential_scope=scope.credential_scope,
        retry_policy=scope.retry_policy,
        quota_scope=scope.quota_scope,
    )
    satisfied_pending = {
        capability
        for capability in binding.pending_capabilities
        if isinstance(outcomes.get(capability), (Configured, CannotVerify))
        and not (
            isinstance(outcomes.get(capability), CannotVerify)
            and cast(CannotVerify, outcomes[capability]).stage is CheckStage.SETUP
        )
    }
    available = binding.available_capabilities | satisfied_pending
    request_checks = tuple(
        check
        for check in deployment_plan.request_checks
        if check.capability in binding.required_capabilities
    )
    effective_capabilities: dict[str, JSONValue] = {
        "available": cast(JSONValue, sorted(available)),
        "required": cast(JSONValue, sorted(binding.required_capabilities)),
        "minimums": {
            key: _thaw_json(value)
            for key, value in sorted(binding.capability_minimums.items())
        },
        "request_defaults": binding.request_defaults.to_json_value(),
    }
    return RuntimeBindingPlan(
        binding_id=binding.binding_id,
        root_deployment_key=binding.root_deployment_key,
        requested_model_id=binding.requested_model_id,
        dialect_id=binding.dialect_id,
        declared_capabilities=binding.declared_capabilities,
        effective_capabilities=MappingProxyType(effective_capabilities),
        available_capabilities=frozenset(available),
        capability_minimums=MappingProxyType(
            {str(key): value for key, value in binding.capability_minimums.items()}
        ),
        request_defaults=binding.request_defaults,
        required_output_channels=binding.required_output_channels,
        request_checks=request_checks,
        deployment_fingerprint=deployment.fingerprint,
        resolved_route=route,
        connection_identity=identity,
        binding_plan_fingerprint=binding.fingerprint,
        deployment_plan_fingerprint=deployment_plan.fingerprint,
    )


def _binding_error(
    binding: BindingReconcileInput,
    code: str,
    message: str,
    *,
    capability: CapabilityKey | None = None,
    sources: tuple[str, ...] = (),
) -> ReconcileDiagnostic:
    return ReconcileDiagnostic(
        ReconcileSeverity.ERROR,
        code,
        message,
        binding_id=binding.binding_id,
        root_deployment_key=binding.root_deployment_key,
        capability=capability,
        sources=sources,
    )


def _deployment_error(
    deployment: DeploymentReconcileInput,
    code: str,
    message: str,
    *,
    capability: CapabilityKey | None = None,
) -> ReconcileDiagnostic:
    return ReconcileDiagnostic(
        ReconcileSeverity.ERROR,
        code,
        message,
        root_deployment_key=deployment.root_deployment_key,
        capability=capability,
    )


def _source_tuple(values: Iterable[str]) -> tuple[str, ...]:
    result = tuple(sorted(set(values)))
    if any(not isinstance(value, str) or not value for value in result):
        raise TypeError("sources must contain non-empty strings")
    return result


def _nonempty(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")


class _FrozenJSONList(list[JSONValue]):
    """List-shaped JSON value that rejects mutation while preserving equality."""

    def _immutable(self, *args: object, **kwargs: object) -> Never:
        del self, args, kwargs
        raise TypeError("frozen JSON sequences do not support mutation")

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        del memo
        return self


# ``list`` exposes several mutation spellings, including in-place operators.
# Install the same guard for all of them so nested JSON sequences remain
# list-compatible without leaving a normal mutation path open.
for _method_name in (
    "append",
    "clear",
    "extend",
    "insert",
    "pop",
    "remove",
    "reverse",
    "sort",
    "__delitem__",
    "__iadd__",
    "__imul__",
    "__setitem__",
):
    setattr(_FrozenJSONList, _method_name, _FrozenJSONList._immutable)
del _method_name


def _freeze_json_value(value: object, path: str) -> JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            copied[key] = _freeze_json_value(item, f"{path}.{key}")
        return MappingProxyType(copied)
    if isinstance(value, (list, tuple)):
        return _FrozenJSONList(
            _freeze_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{path} contains non-JSON value {type(value).__name__}")


def _freeze_json(value: Mapping[str, JSONValue], path: str) -> Mapping[str, JSONValue]:
    frozen = _freeze_json_value(value, path)
    assert isinstance(frozen, Mapping)
    return cast(Mapping[str, JSONValue], frozen)


def _freeze_request_defaults(value: RequestDefaults, path: str) -> RequestDefaults:
    """Detach and deeply freeze defaults before they become plan evidence."""

    if not isinstance(value, RequestDefaults):
        raise TypeError(f"{path} must be RequestDefaults")
    frozen = RequestDefaults()
    object.__setattr__(frozen, "values", _freeze_json(value.values, path))
    return frozen


def _freeze_capability_intent(value: CapabilityIntent, path: str) -> CapabilityIntent:
    """Detach the nested JSON owned by an otherwise frozen intent record."""

    if not isinstance(value, CapabilityIntent):
        raise TypeError(f"{path} must be CapabilityIntent")
    frozen = CapabilityIntent(
        key=value.key,
        required=value.required,
        request_defaults=_freeze_request_defaults(
            value.request_defaults, f"{path}.request_defaults"
        ),
        sources=value.sources,
    )
    object.__setattr__(
        frozen,
        "minimums",
        _freeze_json(value.minimums, f"{path}.minimums"),
    )
    return frozen


def _thaw_json(value: object) -> JSONValue:
    return copy_json_value(value, "serialized value")


def _json_sequence(value: JSONValue) -> tuple[str | int | float | bool | None, ...]:
    if isinstance(value, list):
        if any(isinstance(item, (list, Mapping)) for item in value):
            raise TypeError("minimum sequence values must be scalar")
        return tuple(cast(str | int | float | bool | None, item) for item in value)
    if isinstance(value, Mapping):
        raise TypeError("minimum values cannot merge incompatible mappings")
    return (value,)


def _canonical_json(value: JSONValue) -> str:
    """Return a deterministic, type-preserving identity for one JSON value."""

    return json.dumps(
        _thaw_json(value),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
