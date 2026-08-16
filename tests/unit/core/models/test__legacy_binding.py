"""Tests for the legacy ``ChatModel``/``GenModel`` binding construction.

Scoped to what this path owes *independently* of ``connection_factory``: it
builds its own ``AsyncOpenAI`` instead of going through a factory, so a contract
asserted only against the factory does not reach it.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sieval.core.models._legacy_binding import build_legacy_openai_binding
from sieval.core.models.connection_factory import DEFAULT_REQUEST_TIMEOUT


class TestLegacyBindingClient:
    def test_client_declares_the_shared_request_timeout(self) -> None:
        """The wrapper path owes the same declared bound as the factory.

        ``ChatModel`` and ``GenModel`` both build their client here, so this is
        the construction that serves runs today -- and the one place a drift back
        to whichever ``timeout`` the OpenAI SDK happens to default to would go
        unnoticed, since the fallback is currently the same value.
        """
        client = SimpleNamespace(
            base_url="https://legacy.example/v1/",
            close=AsyncMock(),
        )
        with patch(
            "sieval.core.models._legacy_binding.AsyncOpenAI",
            return_value=client,
        ) as client_factory:
            build_legacy_openai_binding(
                dialect_id="openai_chat",
                model="m",
                api_base="https://legacy.example/v1",
                api_key="sk-runtime-only",
                max_retries=4,
                concurrency_limit=None,
                parent_limiter=None,
            )

        client_factory.assert_called_once_with(
            base_url="https://legacy.example/v1",
            api_key="sk-runtime-only",
            max_retries=4,
            timeout=DEFAULT_REQUEST_TIMEOUT,
        )
