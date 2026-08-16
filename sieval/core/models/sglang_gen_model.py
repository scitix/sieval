"""Explicit one-cycle bypass for SGLang's native ``/generate`` protocol.

``sglang_native`` has no executable PR-1 binder, so this facade intentionally
does not fabricate a ``RuntimeBindingPlan`` or enter the canonical dialect
registry.  It preserves the existing native wire implementation until PR 5
activates that dialect, while exposing a truthful ``sglang_legacy`` identity to
the temporary task compatibility checks.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

from types import MappingProxyType, TracebackType
from typing import Any, Self, cast
from uuid import uuid4

import anyio
from openai import AsyncOpenAI

from sieval.core.types import JSONValue

from .capabilities import Capability
from .connection_factory import DEFAULT_REQUEST_TIMEOUT
from .deployment import (
    ConnectionIdentity,
    ConnectionPool,
    Deployment,
    Engine,
    ServingFacts,
)
from .dialect import (
    RequestAuditError,
    active_request_leaves,
    validate_request_invariants,
)
from .dialect_registry import compatibility_factory_for
from .ir import CompletionInput, ModelInput, Request, Response
from .model import Model
from .transports.sglang import (
    SGLANG_LEGACY_DIALECT_OPTION_KEYS,
    SglangTransport,
)

_SGLANG_LEGACY_LOWERED_LEAVES = frozenset(
    {
        "input.completion",
        "sampling.temperature",
        "sampling.top_p",
        "sampling.top_k",
        "sampling.max_tokens",
        "sampling.stop",
        "sampling.frequency_penalty",
        "sampling.presence_penalty",
        "sampling.n",
        "scoring.input_scoring",
        "scoring.sampled_logprobs",
        "scoring.top_logprobs",
    }
)
_SGLANG_LEGACY_NOOP_LEAVES = MappingProxyType(
    {
        # Native /generate is always one POST. The compatibility builder
        # defaults stream=True, so this is an explicit scheduling-only no-op.
        "scheduling.stream": "native /generate is a single non-streaming POST",
        # SGLang has no per-request seed on /generate. Deterministic managed
        # deployments pin the process seed through DeploymentPlan.seed and
        # --random-seed; external deployments remain explicitly best-effort.
        "sampling.seed": (
            "native /generate has no per-request seed; managed deterministic "
            "serving pins DeploymentPlan.seed at engine startup"
        ),
    }
)
_SGLANG_LEGACY_CANONICAL_OPTION_OWNERS = {
    "max_tokens": "sampling.max_tokens",
    "temperature": "sampling.temperature",
    "top_p": "sampling.top_p",
    "top_k": "sampling.top_k",
    "stop": "sampling.stop",
    "frequency_penalty": "sampling.frequency_penalty",
    "presence_penalty": "sampling.presence_penalty",
}


def _validate_legacy_request_leaves(req: Request) -> None:
    """Reject every active leaf the temporary native transport cannot lower."""

    options = req.dialect_options
    if options is not None and options.dialect_id != "sglang_legacy":
        raise RequestAuditError(
            f"dialect options target {options.dialect_id!r}, but the legacy "
            "model is bound to 'sglang_legacy'"
        )

    active = active_request_leaves(req)
    rejected: list[str] = []
    for path, value in active.items():
        if path in _SGLANG_LEGACY_LOWERED_LEAVES:
            continue
        if path in _SGLANG_LEGACY_NOOP_LEAVES or path == "dialect_options":
            continue
        if not path.startswith("dialect_options."):
            rejected.append(path)
            continue

        key = path.removeprefix("dialect_options.")
        if key not in SGLANG_LEGACY_DIALECT_OPTION_KEYS:
            rejected.append(path)
            continue
        owner = _SGLANG_LEGACY_CANONICAL_OPTION_OWNERS.get(key)
        if owner is not None:
            rejected.append(f"{path} (use canonical request leaf {owner})")
            continue
        if value is None:
            rejected.append(f"{path} (null values are not lowered)")
            continue

    if options is not None and {"prefill", "prefix"} <= set(options.values):
        rejected.append("dialect_options.prefix (duplicates dialect_options.prefill)")

    if rejected:
        raise RequestAuditError(
            "sglang_legacy cannot lower request leaves: " + ", ".join(sorted(rejected))
        )


class SglangGenModel(Model):
    """Legacy native SGLang facade, deliberately outside canonical binding."""

    def __new__(cls, *args: object, **kwargs: object) -> Any:
        if cls is SglangGenModel:
            factory = compatibility_factory_for("sglang_native")
            if factory is not None:
                return factory(*args, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        concurrency_limit: int | None = None,
        parent_limiter: anyio.CapacityLimiter | None = None,
        extra: dict[str, JSONValue] | None = None,
        transport: SglangTransport | None = None,
        **kwargs: object,
    ) -> None:
        self._model = model
        self._api_base = api_base
        self._client = AsyncOpenAI(
            base_url=api_base,
            api_key=api_key,
            max_retries=max_retries,
            timeout=DEFAULT_REQUEST_TIMEOUT,
        )
        self._kwargs = dict(kwargs)
        self._extra = dict(extra) if extra is not None else None
        self._parent_limiter = parent_limiter
        self._limiter = (
            anyio.CapacityLimiter(concurrency_limit)
            if concurrency_limit is not None
            else None
        )
        endpoint = str(self._client.base_url).rstrip("/")
        shared_limiter = parent_limiter if parent_limiter is not None else self._limiter
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
        self._deployment = Deployment(
            deployment_id=None,
            plan=None,
            engine=Engine("sglang"),
            engine_source="config",
            api_base=endpoint,
            endpoints={},
            topology=None,
            metrics_url=None,
            facts=ServingFacts(),
        )
        self._pool = ConnectionPool(self._client, identity, shared_limiter)
        legacy_transport = (
            transport if transport is not None else self._build_default_transport()
        )
        self._legacy_transport = legacy_transport
        self._transport = cast(Any, legacy_transport)
        self._lifecycle_owner = self

    def _build_default_transport(self) -> SglangTransport:
        return SglangTransport(self._client, self._model, self._api_base)

    @property
    def dialect_id(self) -> str:
        return "sglang_legacy"

    @property
    def runtime_plan(self) -> None:
        return None

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._legacy_transport.capabilities

    def _coerce_input(self, prompt: object) -> ModelInput:
        if isinstance(prompt, CompletionInput):
            return prompt
        if isinstance(prompt, str):
            return CompletionInput(prompt)
        raise TypeError("SglangGenModel prompt must be text or CompletionInput")

    async def arun(self, req: Request) -> Response:
        if not isinstance(req, Request):
            raise TypeError(f"arun requires Request, got {type(req).__name__}")
        validate_request_invariants(req)
        _validate_legacy_request_leaves(req)
        async with self._pool.acquire(self._limiter):
            return await self._legacy_transport.arun(req)

    def with_dialect(self, dialect_id: str, runtime_plan: Any) -> Model:
        del dialect_id, runtime_plan
        raise RuntimeError(
            "sglang_legacy cannot rebind before the sglang_native PR-5 binder"
        )

    def _legacy_lifecycle_owner(self) -> "SglangGenModel":
        owner = self._lifecycle_owner
        if not isinstance(owner, SglangGenModel):
            raise RuntimeError("sglang_legacy binding has no lifecycle owner")
        return owner

    async def aclose(self) -> None:
        owner = self._legacy_lifecycle_owner()
        await owner._pool.aclose()

    async def __aenter__(self) -> Self:
        owner = self._legacy_lifecycle_owner()
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
