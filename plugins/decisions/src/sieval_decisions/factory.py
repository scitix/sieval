"""Explicit provider selection shared by every decision benchmark.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

from collections.abc import Mapping

import httpx

from sieval.core.types import JSONValue

from .client import DecisionsClient
from .providers import OpenRouterDecisionsClient


def create_decisions_client(
    provider: str,
    model_name: str,
    *,
    base_url: str | None = None,
    api_key_env: str | None = None,
    revision: str | None = None,
    options: Mapping[str, JSONValue] | None = None,
    timeout: float = 60,
    max_retries: int = 2,
    transport: httpx.AsyncBaseTransport | None = None,
) -> DecisionsClient:
    implementations = {
        "openrouter": OpenRouterDecisionsClient,
    }
    implementation = implementations.get(provider)
    if implementation is None:
        raise ValueError(
            f"unknown decisions provider {provider!r}; choose {sorted(implementations)}"
        )
    return implementation(
        model_name,
        base_url=base_url,
        api_key_env=api_key_env,
        revision=revision,
        options=options,
        timeout=timeout,
        max_retries=max_retries,
        transport=transport,
    )
