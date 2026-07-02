"""Abstract model base class and shared types for model backends."""

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NotRequired, Self, TypedDict

import anyio
from openai import AsyncOpenAI

from sieval.core.types import JSONValue
from sieval.core.utils.concurrency import CompositeLimiter
from sieval.core.utils.serialization import sieval_record


class ModelUsage(TypedDict):
    """Token usage statistics from a single model API call."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


class ModelMeta(TypedDict):
    """Model identification metadata: name, endpoint, and default parameters."""

    model: str
    api_base: str | None
    default_params: dict[str, JSONValue]
    extra: NotRequired[dict[str, JSONValue]]


class ModelCallMeta(TypedDict):
    """Per-API-call metadata: model info, usage, params, finish reasons."""

    model: ModelMeta
    usage: NotRequired[ModelUsage]
    request_params: NotRequired[dict[str, JSONValue]]
    finish_reasons: NotRequired[list[str]]
    response_model: NotRequired[str]
    system_fingerprint: NotRequired[str | None]


class ModelQuotaSnapshot(TypedDict):
    """Snapshot of a single limiter's available and total concurrency tokens."""

    available: int
    total: int


class ModelQuotaInfo(TypedDict):
    """Combined parent (shared) and child (local) limiter quota info."""

    available: int | float
    total: int | float
    parent: ModelQuotaSnapshot | None
    child: ModelQuotaSnapshot | None


@sieval_record
@dataclass
class ModelOutput:
    """Standard return type from model calls."""

    model: ModelMeta  # should be auto-attached by Model implementations
    texts: list[str]
    finish_reasons: list[str] | None = None
    reasoning_texts: list[str] | None = None
    # Token texts follow the OpenAI / literal-whitespace convention (leading
    # spaces preserved, e.g. " A"). Backends that emit other markers (e.g.
    # sglang byte-level "ĠA") normalize to this contract before populating it,
    # so consumers can match on literal whitespace rather than per-tokenizer
    # markers.
    logprobs_tokens: list[str] | None = None
    logprobs: list[float | None] | None = None
    top_logprobs: list[dict[str, float]] | None = None
    usage: ModelUsage | None = None
    request_params: dict[str, JSONValue] | None = None
    # From API response — what the server actually used
    response_model: str | None = None
    system_fingerprint: str | None = None


class Model[TModelInput](ABC):
    """Abstract base for all model backends.

    Uses an OpenAI-compatible AsyncClient. Provides two-level concurrency
    control: ``_parent_limiter`` (shared API quota from a base model) and
    ``_limiter`` (this model's reserved quota). Both limiters are acquired
    before every ``agenerate`` / ``alogprobs`` call.
    """

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        concurrency_limit: int | None = None,
        parent_limiter: anyio.CapacityLimiter | None = None,
        extra: dict[str, JSONValue] | None = None,
        **kwargs,
    ):
        self._model = model
        self._api_base = api_base

        self._client = AsyncOpenAI(
            base_url=api_base,
            api_key=api_key,
            max_retries=max_retries,
        )
        # Default kwargs for the model
        self._kwargs = kwargs
        # Catch-all for user-defined, schema-external config (e.g., sequence_wrappers).
        # Stored separately from _kwargs — not passed to the API backend.
        self._extra = extra

        # Two-level concurrency control:
        # - parent_limiter: shared API quota (from base model)
        # - _limiter: this model's reserved quota
        self._parent_limiter = parent_limiter
        self._limiter = (
            anyio.CapacityLimiter(concurrency_limit)
            if concurrency_limit is not None
            else None
        )

    def with_args(
        self,
        concurrency_limit: int | None = None,
        extra: dict[str, JSONValue] | None = None,
        **kwargs,
    ) -> Self:
        """Create a derived model sharing the same resource pool.

        A non-``None`` *concurrency_limit* reserves a sub-quota from this
        model's limiter; ``None`` shares the existing limiter.
        Multi-level derivation is forbidden.

        Example::

            child = base.with_args(concurrency_limit=64)
        """
        # Prevent multi-level derivation
        if concurrency_limit is not None and self._parent_limiter is not None:
            raise ValueError(
                "Cannot create multi-level model derivation. "
                "Multi-level resource pools are not supported. "
                "Please fork from the base model instead."
            )

        new_model = copy.copy(self)
        new_model._kwargs = {**self._kwargs, **kwargs}

        if extra is not None:
            new_model._extra = extra

        # If specifying a new concurrency_limit, create a new limiter
        if concurrency_limit is not None:
            new_model._limiter = anyio.CapacityLimiter(concurrency_limit)
            # Parent's limiter becomes our parent
            new_model._parent_limiter = self._limiter
        # Otherwise, share the same limiter and parent_limiter

        return new_model

    def as_type(self, model_type: type[Self]) -> Self:
        """Re-type this model (e.g. GenModel ↔ ChatModel), sharing client and limiters.

        Example::

            chat_model = gen_model.as_type(ChatModel)
        """
        if not isinstance(model_type, type) or not issubclass(model_type, Model):
            raise TypeError(f"model_type must be a Model subclass, got {model_type}")

        # Copy and change class type
        # This is safe because GenModel and ChatModel only differ in methods,
        # not in instance attributes
        new_model = copy.copy(self)
        new_model.__class__ = model_type

        return new_model

    def get_available_quota(self) -> int | float:
        """Return the minimum available tokens across both limiters."""
        capacities = []

        if self._parent_limiter is not None:
            capacities.append(self._parent_limiter.available_tokens)

        if self._limiter is not None:
            capacities.append(self._limiter.available_tokens)

        if not capacities:
            return float("inf")

        return min(capacities)

    def get_total_quota(self) -> int | float:
        """Return the child limiter's total tokens (ignores parent)."""
        if self._limiter is None:
            return float("inf")
        return self._limiter.total_tokens

    def get_quota_info(self) -> ModelQuotaInfo:
        """Return a structured breakdown of both parent and child limiter quotas."""
        info: ModelQuotaInfo = {
            "available": self.get_available_quota(),
            "total": self.get_total_quota(),
            "parent": None,
            "child": None,
        }

        if self._parent_limiter is not None:
            info["parent"] = {
                "available": self._parent_limiter.available_tokens,
                "total": self._parent_limiter.total_tokens,
            }

        if self._limiter is not None:
            info["child"] = {
                "available": self._limiter.available_tokens,
                "total": self._limiter.total_tokens,
            }

        return info

    @property
    def extra(self) -> dict[str, JSONValue]:
        """User-defined, schema-external model config, not sent to the API backend.

        Returns a shallow copy; mutating the returned dict does not affect
        the model instance.
        """
        return dict(self._extra) if self._extra else {}

    def meta(self) -> ModelMeta:
        """Return a ``ModelMeta`` dict identifying this model."""
        result: ModelMeta = {
            "model": self._model,
            "api_base": self._api_base,
            "default_params": dict(self._kwargs),
        }
        if self._extra:
            result["extra"] = dict(self._extra)
        return result

    async def agenerate(self, prompt: TModelInput, **kwargs) -> ModelOutput:
        """Generate text; acquires both limiters first."""
        async with CompositeLimiter(self._parent_limiter, self._limiter):
            return await self._agenerate_impl(prompt, **kwargs)

    async def alogprobs(
        self,
        prompt: str,
        *,
        max_tokens: int = 1,
        logprobs: int = 5,
        echo: bool = True,
        temperature: float = 0.0,
        **kwargs,
    ) -> ModelOutput:
        """Extract logprobs; acquires both limiters first."""
        async with CompositeLimiter(self._parent_limiter, self._limiter):
            return await self._alogprobs_impl(
                prompt,
                max_tokens=max_tokens,
                logprobs=logprobs,
                echo=echo,
                temperature=temperature,
                **kwargs,
            )

    @abstractmethod
    async def _agenerate_impl(self, prompt: TModelInput, **kwargs) -> ModelOutput: ...

    @abstractmethod
    async def _alogprobs_impl(
        self,
        prompt: str,
        *,
        max_tokens: int,
        logprobs: int,
        echo: bool,
        temperature: float,
        **kwargs,
    ) -> ModelOutput: ...
