"""One-cycle ``sglang_native`` compatibility wrapper over a private
``async_http_json`` pool and a real ``RuntimeBindingPlan``.

It reads no environment variable: ``api_base`` is required, and a bearer
token is sent only for an explicit ``api_key`` (``OPENAI_API_KEY`` is not
read, unlike the ``AsyncOpenAI`` client it replaced).

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from collections.abc import Mapping
from typing import cast

import anyio
import httpx

from sieval.core.types import JSONValue

from ._legacy_binding import build_legacy_http_binding
from .deployment import Engine
from .dialect import Dialect
from .dialects.sglang_native import SglangNativeDialect
from .ir import CompletionInput, ModelInput
from .model import Model


class SglangGenModel(Model):
    """Deprecated constructor wrapper selecting ``sglang_native``."""

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        concurrency_limit: int | None = None,
        parent_limiter: anyio.CapacityLimiter | None = None,
        extra: dict[str, JSONValue] | None = None,
        transport: Dialect | None = None,
        client_timeout: httpx.Timeout | Mapping[str, float] | None = None,
        **kwargs: object,
    ) -> None:
        binding = build_legacy_http_binding(
            dialect_id="sglang_native",
            model=model,
            api_base=api_base,
            api_key=api_key,
            max_retries=max_retries,
            concurrency_limit=concurrency_limit,
            parent_limiter=parent_limiter,
            engine=Engine("sglang"),
            client_timeout=client_timeout,
        )
        self._client = binding.pool.connection
        self._model = model
        self._api_base = api_base
        dialect = (
            transport if transport is not None else self._build_default_transport()
        )
        self._initialize(
            deployment=binding.deployment,
            pool=binding.pool,
            runtime_plan=binding.runtime_plan,
            dialect=dialect,
            local_limiter=(
                binding.local_limiter
                if binding.local_limiter is not None
                else cast(anyio.CapacityLimiter | None, binding.pool.shared_limiter)
            ),
            parent_limiter=binding.parent_limiter,
            builder_defaults=kwargs,
            extra=extra,
            api_base=api_base,
            lifecycle_owner=self,
            provenance_projector=binding.provenance_projector,
        )

    def _build_default_transport(self) -> Dialect:
        """Deprecated hook retained for downstream test/subclass adapters."""

        return SglangNativeDialect(self._client, self._model)

    def _coerce_input(self, prompt: object) -> ModelInput:
        if isinstance(prompt, CompletionInput):
            return prompt
        if isinstance(prompt, str):
            return CompletionInput(prompt)
        raise TypeError("SglangGenModel prompt must be text or CompletionInput")
