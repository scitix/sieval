"""Binding construction for the legacy ``ChatModel``/``GenModel`` constructors.

These wrappers accept a bare ``model=``/``api_base=`` pair and must still
present a truthful :class:`RuntimeBindingPlan`, so this module fabricates one
from an externally-owned connection rather than from a reconciled deployment.
The canonical ``Model.bind`` path also uses the narrow reconstruction helper
below so converting a wrapper plan cannot discard its provenance policy.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

import anyio
from openai import AsyncOpenAI

from sieval.core.types import JSONValue

from ._fingerprint import fingerprint_mapping
from .capabilities import RequestDefaults, Supported
from .connection_factory import DEFAULT_REQUEST_TIMEOUT
from .deployment import (
    ConnectionIdentity,
    ConnectionPool,
    Deployment,
    Engine,
    ServingFacts,
    resolve_route,
)
from .dialect_registry import capability_decisions_for, get_dialect_spec
from .reconcile import RuntimeBindingPlan


@dataclass(frozen=True)
class _LegacyOpenAIBinding:
    deployment: Deployment
    pool: ConnectionPool[Any]
    runtime_plan: RuntimeBindingPlan
    provenance_projector: "_LegacyProvenanceProjector"
    local_limiter: anyio.CapacityLimiter | None
    parent_limiter: anyio.CapacityLimiter | None


@dataclass(frozen=True)
class _LegacyProvenanceProjector:
    """Project a UUID-scoped runtime plan into stable persisted evidence."""

    connection_identity: ConnectionIdentity
    runtime_binding_id: str
    runtime_root_deployment_key: str
    runtime_binding_plan_fingerprint: str
    runtime_deployment_plan_fingerprint: str

    def _identity_suffix(self) -> str:
        identity_fingerprint = fingerprint_mapping(
            {
                "endpoint": self.connection_identity.endpoint,
                "connection_family": self.connection_identity.connection_family,
                "credential_scope": self.connection_identity.credential_scope,
                "retry_policy": self.connection_identity.retry_policy,
                "quota_scope": self.connection_identity.quota_scope,
            }
        )
        return identity_fingerprint.removeprefix("sha256:")[:16]

    def _stable_binding_id(
        self,
        runtime_plan: RuntimeBindingPlan,
        stable_base_id: str,
    ) -> str | None:
        """Classify and project only wrapper-owned binding identities.

        ``with_dialect`` accepts plans for sibling logical bindings.  Preserve
        a sibling suffix in an injective, namespaced encoding.  An unrelated
        binding ID is opaque: retaining it could persist a runtime UUID, while
        treating it as the base binding could collapse two logical bindings.
        """

        if runtime_plan.binding_id == self.runtime_binding_id:
            return stable_base_id
        sibling_prefix = f"{self.runtime_binding_id}:"
        if runtime_plan.binding_id.startswith(sibling_prefix):
            suffix = runtime_plan.binding_id.removeprefix(sibling_prefix)
            encoded_suffix = suffix.encode("utf-8").hex()
            return f"{stable_base_id}:sibling:{encoded_suffix}"
        return None

    def _project_identity(
        self, runtime_plan: RuntimeBindingPlan
    ) -> RuntimeBindingPlan | None:
        """Replace proven wrapper-owned identities without touching evidence."""

        if runtime_plan.root_deployment_key != self.runtime_root_deployment_key:
            return None
        identity_suffix = self._identity_suffix()
        stable_base_id = (
            f"legacy:{runtime_plan.dialect_id}:"
            f"{runtime_plan.requested_model_id}:{identity_suffix}"
        )
        binding_id = self._stable_binding_id(runtime_plan, stable_base_id)
        if binding_id is None:
            return None
        root_deployment_key = (
            f"legacy:{runtime_plan.deployment_fingerprint}:{identity_suffix}"
        )
        return replace(
            runtime_plan,
            binding_id=binding_id,
            root_deployment_key=root_deployment_key,
            connection_identity=self.connection_identity,
        )

    def __call__(self, runtime_plan: RuntimeBindingPlan) -> RuntimeBindingPlan | None:
        """Return stable evidence only when no opaque proof was replaced.

        Reconciliation owns binding/deployment plan fingerprints.  A bare
        ``with_dialect`` call cannot interpret a changed opaque fingerprint,
        so it must leave provenance incomplete for the composition layer to
        fill rather than silently replacing that evidence.
        """

        if (
            runtime_plan.binding_plan_fingerprint
            != self.runtime_binding_plan_fingerprint
            or runtime_plan.deployment_plan_fingerprint
            != self.runtime_deployment_plan_fingerprint
        ):
            return None
        projected_identity = self._project_identity(runtime_plan)
        if projected_identity is None:
            return None

        semantic_plan = projected_identity.to_json_value()
        for key in (
            "binding_plan_fingerprint",
            "deployment_plan_fingerprint",
            "fingerprint",
            "verification_fingerprint",
        ):
            semantic_plan.pop(key)
        binding_plan_fingerprint = fingerprint_mapping(
            {"legacy_provenance_plan": semantic_plan}
        )
        projected = replace(
            projected_identity,
            binding_plan_fingerprint=binding_plan_fingerprint,
            deployment_plan_fingerprint="external:none",
        )
        return projected


def _legacy_private_credential_kind(
    runtime_plan: RuntimeBindingPlan,
) -> str | None:
    """Return the stable credential category for a wrapper-private identity."""

    identity = runtime_plan.connection_identity
    quota_scope = identity.quota_scope
    prefix = "legacy-private:"
    credential_scope = identity.credential_scope
    uses_legacy_namespace = (
        quota_scope == "legacy-private"
        or quota_scope.startswith(prefix)
        or credential_scope == "legacy-private"
        or credential_scope.startswith(prefix)
    )
    if not uses_legacy_namespace:
        return None
    if not quota_scope.startswith(prefix):
        raise ValueError(
            "legacy-private connection identity has an invalid quota scope"
        )
    private_scope = quota_scope.removeprefix(prefix)
    if len(private_scope) != 32 or any(
        character not in "0123456789abcdef" for character in private_scope
    ):
        raise ValueError(
            "legacy-private connection identity has an invalid runtime UUID"
        )
    credential_prefix = f"{quota_scope}:"
    if not credential_scope.startswith(credential_prefix):
        raise ValueError(
            "legacy-private connection identity has mismatched credential and "
            "quota scopes"
        )
    credential_kind = credential_scope.removeprefix(credential_prefix)
    if credential_kind not in {
        "explicit-credential",
        "environment-credential",
    }:
        raise ValueError(
            "legacy-private connection identity has an invalid credential category"
        )
    return credential_kind


def _legacy_capability_state(
    dialect_id: str,
) -> tuple[frozenset[str], dict[str, JSONValue]]:
    decisions = capability_decisions_for(dialect_id)
    available = frozenset(
        key for key, decision in decisions.items() if isinstance(decision, Supported)
    )
    effective: dict[str, JSONValue] = {key: {} for key in sorted(available)}
    return available, effective


def _legacy_initial_binding_plan_fingerprint(
    *,
    dialect_id: str,
    requested_model_id: str,
    binding_id: str,
    root_deployment_key: str,
    available: frozenset[str],
    effective: dict[str, JSONValue],
) -> str:
    """Return the opaque proof emitted by the unreconciled wrapper plan."""

    return fingerprint_mapping(
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


def _legacy_provenance_projector_for_plan(
    runtime_plan: RuntimeBindingPlan,
) -> _LegacyProvenanceProjector | None:
    """Recover the stable policy carried implicitly by a legacy runtime plan.

    ``Model.bind`` is public and may be handed a plan obtained from a legacy
    wrapper.  Reconstructing the projector here prevents that shape conversion
    from silently turning private pool ownership into persisted provenance.
    Malformed legacy-looking identities fail closed instead of being treated as
    canonical plans.
    """

    credential_kind = _legacy_private_credential_kind(runtime_plan)
    if credential_kind is None:
        return None

    runtime_identity = runtime_plan.connection_identity
    identity_fingerprint = fingerprint_mapping(
        {
            "endpoint": runtime_identity.endpoint,
            "connection_family": runtime_identity.connection_family,
            "credential_scope": runtime_identity.credential_scope,
            "retry_policy": runtime_identity.retry_policy,
            "quota_scope": runtime_identity.quota_scope,
        }
    )
    runtime_suffix = identity_fingerprint.removeprefix("sha256:")[:16]
    runtime_binding_id = (
        f"legacy:{runtime_plan.dialect_id}:"
        f"{runtime_plan.requested_model_id}:{runtime_suffix}"
    )
    runtime_root_deployment_key = (
        f"legacy:{runtime_plan.deployment_fingerprint}:{runtime_suffix}"
    )
    if (
        runtime_plan.binding_id != runtime_binding_id
        and not runtime_plan.binding_id.startswith(f"{runtime_binding_id}:")
    ):
        raise ValueError("UUID-scoped legacy plan has an inconsistent binding identity")
    if runtime_plan.root_deployment_key != runtime_root_deployment_key:
        raise ValueError("UUID-scoped legacy plan has an inconsistent deployment root")

    available, effective = _legacy_capability_state(runtime_plan.dialect_id)
    initial_binding_plan_fingerprint = _legacy_initial_binding_plan_fingerprint(
        dialect_id=runtime_plan.dialect_id,
        requested_model_id=runtime_plan.requested_model_id,
        binding_id=runtime_binding_id,
        root_deployment_key=runtime_root_deployment_key,
        available=available,
        effective=effective,
    )
    if (
        runtime_plan.binding_plan_fingerprint != initial_binding_plan_fingerprint
        or runtime_plan.deployment_plan_fingerprint != "external:none"
    ):
        raise ValueError(
            "UUID-scoped legacy plan contains opaque reconciled provenance; "
            "attach its stable provenance through the composition layer"
        )

    stable_identity = ConnectionIdentity(
        endpoint=runtime_identity.endpoint,
        connection_family=runtime_identity.connection_family,
        credential_scope=f"legacy-private:{credential_kind}",
        retry_policy=runtime_identity.retry_policy,
        quota_scope="legacy-private",
    )
    return _LegacyProvenanceProjector(
        stable_identity,
        runtime_binding_id,
        runtime_root_deployment_key,
        runtime_plan.binding_plan_fingerprint,
        runtime_plan.deployment_plan_fingerprint,
    )


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
    available, effective = _legacy_capability_state(dialect_id)
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
    binding_plan_fingerprint = _legacy_initial_binding_plan_fingerprint(
        dialect_id=dialect_id,
        requested_model_id=requested_model_id,
        binding_id=binding_id,
        root_deployment_key=root_deployment_key,
        available=available,
        effective=effective,
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
        timeout=DEFAULT_REQUEST_TIMEOUT,
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
    # The UUID above is required runtime identity: independently constructed
    # legacy wrappers must never share quota or credential ownership by
    # accident.  Persisted provenance records the same semantic binding with
    # only a credential category and a stable private-pool scope, so object
    # allocation does not make equivalent run artifacts differ.
    provenance_projector = _legacy_provenance_projector_for_plan(runtime_plan)
    assert provenance_projector is not None
    return _LegacyOpenAIBinding(
        deployment=deployment,
        pool=pool,
        runtime_plan=runtime_plan,
        provenance_projector=provenance_projector,
        local_limiter=local_limiter,
        parent_limiter=parent_limiter,
    )
