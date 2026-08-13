"""Binding construction for the legacy ``ChatModel``/``GenModel`` constructors.

These wrappers accept a bare ``model=``/``api_base=`` pair and must still
present a truthful :class:`RuntimeBindingPlan`, so this module fabricates one
from an externally-owned connection rather than from a reconciled deployment.
Kept apart from ``model`` by lifetime: nothing here is reachable from the
canonical ``Model.bind`` path, so it goes when the wrappers do.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import anyio
from openai import AsyncOpenAI

from sieval.core.types import JSONValue

from ._fingerprint import fingerprint_mapping
from .capabilities import RequestDefaults, Supported
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
    local_limiter: anyio.CapacityLimiter | None
    parent_limiter: anyio.CapacityLimiter | None


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
