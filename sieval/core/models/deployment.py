"""Realized deployments, route resolution, and connection ownership.

This module is deliberately below dialect binding and reconciliation.  It
describes where a model is served and owns the one client used to reach that
route; it does not infer capabilities or interpret provider wire formats.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from types import MappingProxyType, TracebackType
from typing import Any, Self, cast
from urllib.parse import urlsplit

import anyio

from sieval.core.utils.concurrency import CompositeLimiter

from ._engine_source import EngineSource
from ._fingerprint import fingerprint_mapping

# These names select or mutate binding resources; they are never provider wire
# extensions.  Composition layers and the legacy ``Model`` bridge share this
# policy so a misspelled placement cannot silently turn a credential, endpoint,
# or connection choice into ``extra_body``.
BINDING_RESOURCE_KEYS = frozenset(
    {
        "api_base",
        "api_key",
        "api_key_env",
        "authorization",
        "base_url",
        "capabilities",
        "connection",
        "connection_family",
        "connection_pool",
        "credential",
        "credentials",
        "deployment",
        "dialect",
        "dialect_id",
        "endpoint",
        "engine",
        "max_retries",
        "model",
        "pool",
        "retry_policy",
        "runtime_plan",
        "service_role",
        "transport",
    }
)


@dataclass(frozen=True)
class ServingFacts:
    """Provider-neutral facts observed from a realized deployment."""

    engine_version: str | None = None
    tokenizer_available: bool | None = None
    prefix_cache_enabled: bool | None = None
    max_top_logprobs: int | None = None


@dataclass(frozen=True)
class DeploymentPlanProjection:
    """Core-layer projection of the desired infer-layer deployment plan."""

    fingerprint: str
    engine_id: str
    service_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.fingerprint:
            raise ValueError("DeploymentPlanProjection.fingerprint must not be empty")
        if not self.engine_id:
            raise ValueError("DeploymentPlanProjection.engine_id must not be empty")
        if not isinstance(self.service_roles, tuple) or any(
            not isinstance(role, str) or not role for role in self.service_roles
        ):
            raise TypeError("DeploymentPlanProjection.service_roles must be strings")
        if tuple(sorted(set(self.service_roles))) != self.service_roles:
            raise ValueError(
                "DeploymentPlanProjection.service_roles must be sorted and unique"
            )


@dataclass(frozen=True)
class DeploymentTopology:
    """Realized serving topology without infer-layer implementation types."""

    is_disaggregated: bool
    roles: tuple[str, ...]
    total_gpus: int


@dataclass(frozen=True)
class RouteIntent:
    """Optional service role requested by a dialect binding."""

    service_role: str | None = None


@dataclass(frozen=True)
class ResolvedRoute:
    """One unambiguous endpoint selected for a dialect binding."""

    service_role: str
    endpoint: str
    connection_family: str
    fingerprint: str


@dataclass(frozen=True)
class ConnectionIdentity:
    """Secret-free identity of one connection and its sharing boundaries.

    Scope values are stable identifiers supplied by composition code, never
    credentials, headers, or other raw secrets.
    """

    endpoint: str
    connection_family: str
    credential_scope: str
    retry_policy: str
    quota_scope: str

    def __post_init__(self) -> None:
        _validate_endpoint(self.endpoint)
        for name in (
            "connection_family",
            "credential_scope",
            "retry_policy",
            "quota_scope",
        ):
            if not getattr(self, name):
                raise ValueError(f"ConnectionIdentity.{name} must not be empty")


DeploymentProbe = Callable[["Deployment"], Awaitable[ServingFacts]]


@dataclass(frozen=True)
class Engine:
    """Static engine identity and the default deployment-verification seam."""

    engine_id: str

    def __post_init__(self) -> None:
        if not self.engine_id:
            raise ValueError("Engine.engine_id must not be empty")

    async def verify_deployment(
        self,
        deployment: "Deployment",
        probe: DeploymentProbe | None = None,
    ) -> ServingFacts:
        """Return current facts, or delegate observation to ``probe``.

        Engine-specific verification may override this method.  Concrete
        constraints and remedies remain outside the deployment/connection
        ownership layer.
        """

        if probe is None:
            return deployment.facts
        return await probe(deployment)


@dataclass(frozen=True)
class Deployment:
    """Immutable snapshot of a realized model-serving deployment."""

    deployment_id: str | None
    plan: DeploymentPlanProjection | None
    engine: Engine
    engine_source: EngineSource
    api_base: str
    endpoints: Mapping[str, str]
    topology: DeploymentTopology | None
    metrics_url: str | None
    facts: ServingFacts

    def __post_init__(self) -> None:
        if self.engine_source not in ("deployment", "config", "unknown"):
            raise ValueError(
                "Deployment.engine_source must be 'deployment', 'config', or 'unknown'"
            )
        engine_is_unknown = self.engine.engine_id == "unknown"
        source_is_unknown = self.engine_source == "unknown"
        if engine_is_unknown != source_is_unknown:
            raise ValueError(
                "Deployment.engine_id and engine_source must agree on whether "
                "the engine identity is unknown"
            )
        if self.deployment_id is not None and (
            not isinstance(self.deployment_id, str) or not self.deployment_id
        ):
            raise ValueError("Deployment.deployment_id must be a non-empty string")
        if (self.deployment_id is None) != (self.plan is None):
            raise ValueError(
                "Deployment.deployment_id and plan must either both be present "
                "for a managed deployment or both be None for an external deployment"
            )
        copied_endpoints = dict(self.endpoints)
        for role, endpoint in copied_endpoints.items():
            if not role:
                raise ValueError("Deployment endpoint roles must not be empty")
            _validate_endpoint(endpoint)
        if self.api_base:
            _validate_endpoint(self.api_base)
        if self.metrics_url:
            _validate_endpoint(self.metrics_url)
        if self.plan is not None and self.plan.engine_id != self.engine.engine_id:
            raise ValueError(
                "Deployment plan engine_id does not match the realized engine: "
                f"{self.plan.engine_id!r} != {self.engine.engine_id!r}"
            )
        object.__setattr__(
            self,
            "endpoints",
            MappingProxyType(copied_endpoints),
        )

    @property
    def fingerprint(self) -> str:
        """Return deterministic, secret-free evidence for this snapshot."""

        return deployment_fingerprint(self)


def deployment_fingerprint(deployment: Deployment) -> str:
    """Fingerprint all realized deployment state relevant to model binding."""

    plan = deployment.plan
    topology = deployment.topology
    facts = deployment.facts
    payload = {
        "api_base": deployment.api_base,
        "deployment_id": deployment.deployment_id,
        "endpoints": sorted(deployment.endpoints.items()),
        "engine": {
            "engine_id": deployment.engine.engine_id,
            "source": deployment.engine_source,
        },
        "facts": {
            "engine_version": facts.engine_version,
            "max_top_logprobs": facts.max_top_logprobs,
            "prefix_cache_enabled": facts.prefix_cache_enabled,
            "tokenizer_available": facts.tokenizer_available,
        },
        "metrics_url": deployment.metrics_url,
        "plan": (
            None
            if plan is None
            else {"engine_id": plan.engine_id, "fingerprint": plan.fingerprint}
        ),
        "topology": (
            None
            if topology is None
            else {
                "is_disaggregated": topology.is_disaggregated,
                "roles": sorted(topology.roles),
                "total_gpus": topology.total_gpus,
            }
        ),
    }
    return fingerprint_mapping(payload)


def resolve_route(
    deployment: Deployment,
    dialect_id: str,
    connection_family: str,
    intent: RouteIntent | None = None,
) -> ResolvedRoute:
    """Resolve exactly one endpoint or reject an absent/ambiguous route."""

    if not dialect_id:
        raise ValueError("dialect_id must not be empty")
    if not connection_family:
        raise ValueError("connection_family must not be empty")

    requested_role = intent.service_role if intent is not None else None
    endpoints = deployment.endpoints
    if requested_role is not None:
        if endpoints:
            try:
                endpoint = endpoints[requested_role]
            except KeyError as exc:
                roles = ", ".join(sorted(endpoints))
                raise ValueError(
                    f"Deployment has no endpoint for service role "
                    f"{requested_role!r}; available roles: {roles}"
                ) from exc
        elif deployment.api_base:
            endpoint = deployment.api_base
        else:
            raise ValueError(
                f"Deployment has no endpoint for service role {requested_role!r}"
            )
        service_role = requested_role
    elif len(endpoints) == 1:
        service_role, endpoint = next(iter(endpoints.items()))
    elif len(endpoints) > 1:
        roles = ", ".join(sorted(endpoints))
        raise ValueError(
            "Deployment route is ambiguous; specify RouteIntent.service_role "
            f"from: {roles}"
        )
    elif deployment.api_base:
        service_role = "default"
        endpoint = deployment.api_base
    else:
        raise ValueError("Deployment has no routable endpoint")

    fingerprint = fingerprint_mapping(
        {
            "connection_family": connection_family,
            "dialect_id": dialect_id,
            "endpoint": endpoint,
            "service_role": service_role,
        }
    )
    return ResolvedRoute(
        service_role=service_role,
        endpoint=endpoint,
        connection_family=connection_family,
        fingerprint=fingerprint,
    )


type Limiter = AbstractAsyncContextManager[Any]


class ConnectionPool[ConnectionT]:
    """Own one connection, its shared quota, and its drain-before-close state."""

    def __init__(
        self,
        connection: ConnectionT,
        identity: ConnectionIdentity,
        shared_limiter: Limiter | None = None,
    ) -> None:
        self._connection = connection
        self._identity = identity
        self._shared_limiter = shared_limiter
        self._state_lock = anyio.Lock()
        self._drained = anyio.Event()
        self._close_complete = anyio.Event()
        self._active = 0
        self._closing = False
        self._closed = False
        self._close_error: BaseException | None = None
        self._connection_closer = _connection_closer(connection)

    @property
    def identity(self) -> ConnectionIdentity:
        """Return the complete secret-free sharing identity."""

        return self._identity

    @property
    def connection(self) -> ConnectionT:
        """Return the owned client for dialect construction.

        Call execution must still enter :meth:`acquire`; exposing this
        read-only reference does not transfer ownership to the dialect.
        """

        return self._connection

    @property
    def shared_limiter(self) -> Limiter | None:
        """Return the pool-owned root quota for read-only diagnostics."""

        return self._shared_limiter

    @property
    def is_closing(self) -> bool:
        """Whether close has started and new admission is disabled."""

        return self._closing

    @property
    def is_closed(self) -> bool:
        """Whether the owned connection's one close attempt has completed."""

        return self._closed

    def verify_family(self, connection_family: str) -> None:
        """Reject binding a dialect from another connection family."""

        if connection_family != self._identity.connection_family:
            raise ValueError(
                "Connection family mismatch: pool owns "
                f"{self._identity.connection_family!r}, requested "
                f"{connection_family!r}"
            )

    def verify_route(self, route: ResolvedRoute) -> None:
        """Reject binding a route other than the one owned by this pool."""

        self.verify_family(route.connection_family)
        if route.endpoint != self._identity.endpoint:
            raise ValueError(
                "Connection route mismatch: pool owns endpoint "
                f"{self._identity.endpoint!r}, requested {route.endpoint!r}"
            )

    def verify_identity(self, identity: ConnectionIdentity) -> None:
        """Reject sharing across credential, retry, or quota boundaries."""

        if identity != self._identity:
            raise ValueError(
                "Connection identity mismatch; construct a distinct ConnectionPool"
            )

    @asynccontextmanager
    async def acquire(
        self,
        local_limiter: Limiter | None = None,
    ) -> AsyncIterator[ConnectionT]:
        """Admit one request, enter shared/local limits, and yield the client."""

        async with self._state_lock:
            if self._closing:
                raise RuntimeError("ConnectionPool is closing; admission rejected")
            self._active += 1

        try:
            if local_limiter is self._shared_limiter:
                local_limiter = None
            async with CompositeLimiter(self._shared_limiter, local_limiter):
                yield self._connection
        finally:
            with anyio.CancelScope(shield=True):
                async with self._state_lock:
                    self._active -= 1
                    if self._closing and self._active == 0:
                        self._drained.set()

    async def aclose(self) -> None:
        """Stop admission, drain admitted requests, and close exactly once."""

        with anyio.CancelScope(shield=True):
            async with self._state_lock:
                close_owner = not self._closing
                if close_owner:
                    self._closing = True
                    if self._active == 0:
                        self._drained.set()

            if close_owner:
                await self._drained.wait()
                try:
                    result = self._connection_closer()
                    if inspect.isawaitable(result):
                        await result
                except BaseException as exc:
                    self._close_error = exc
                finally:
                    async with self._state_lock:
                        self._closed = True
                        self._close_complete.set()
            else:
                await self._close_complete.wait()

        if self._close_error is not None:
            raise self._close_error

    async def __aenter__(self) -> Self:
        if self._closing:
            raise RuntimeError("ConnectionPool is closing; context entry rejected")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


def _validate_endpoint(endpoint: str) -> None:
    if not endpoint:
        raise ValueError("Endpoint must not be empty")
    parsed = urlsplit(endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Endpoint must be an absolute URL with scheme and host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Endpoint identity must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("Endpoint identity must not contain query data or fragments")


def _connection_closer(connection: object) -> Callable[[], object]:
    for method_name in ("aclose", "close"):
        closer = getattr(connection, method_name, None)
        if callable(closer):
            return cast(Callable[[], object], closer)
    raise TypeError("ConnectionPool connection must provide close() or aclose()")
