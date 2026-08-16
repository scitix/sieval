"""Tests for family-keyed connection construction and credential boundaries.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

from typing import Any, cast
from unittest.mock import MagicMock, patch

import httpx
import pytest

from sieval.core.models.connection_factory import (
    CONNECTION_FACTORY_REGISTRY,
    DEFAULT_REQUEST_TIMEOUT,
    AsyncHTTPJSONConnection,
    ConnectionFactoryRegistry,
    ConnectionFactorySpec,
    ConnectionRequest,
    UnknownConnectionFamily,
)
from sieval.core.models.dialect_registry import DIALECT_SPECS


class TestConnectionFactoryRegistry:
    def test_every_dialect_family_has_exactly_one_factory(self) -> None:
        descriptor_families = {
            spec.connection_family for spec in DIALECT_SPECS.values()
        }

        assert descriptor_families == {
            "openai_sdk",
            "async_http_json",
        }
        assert CONNECTION_FACTORY_REGISTRY.families == descriptor_families

    def test_unknown_family_fails_before_any_builder_runs(self) -> None:
        builder = MagicMock()
        registry = ConnectionFactoryRegistry(
            (
                ConnectionFactorySpec(
                    "known_family",
                    "known:max-retries=",
                    builder,
                ),
            )
        )
        request = ConnectionRequest(
            endpoint="https://models.example/v1",
            credential="secret",
            max_retries=2,
        )

        with pytest.raises(UnknownConnectionFamily, match="unknown connection family"):
            registry.create("missing_family", request)

        builder.assert_not_called()

    def test_retry_policy_is_family_owned_and_strict(self) -> None:
        assert (
            CONNECTION_FACTORY_REGISTRY.retry_policy("openai_sdk", 4)
            == "openai-sdk:max-retries=4"
        )
        assert (
            CONNECTION_FACTORY_REGISTRY.retry_policy("async_http_json", 4)
            == "httpx-transport:max-connect-retries=4"
        )
        with pytest.raises(ValueError, match="does not belong"):
            CONNECTION_FACTORY_REGISTRY.validate_retry_policy(
                "async_http_json",
                "openai-sdk:max-retries=4",
            )

    @pytest.mark.parametrize(
        ("changes", "error", "message"),
        [
            ({"endpoint": ""}, ValueError, "endpoint"),
            ({"endpoint": cast(Any, 7)}, ValueError, "endpoint"),
            ({"credential": cast(Any, 7)}, TypeError, "credential"),
            ({"max_retries": cast(Any, True)}, TypeError, "max_retries"),
            ({"max_retries": cast(Any, "2")}, TypeError, "max_retries"),
            ({"max_retries": -1}, ValueError, "non-negative"),
        ],
    )
    def test_connection_request_rejects_invalid_runtime_inputs(
        self,
        changes: dict[str, object],
        error: type[Exception],
        message: str,
    ) -> None:
        values: dict[str, object] = {
            "endpoint": "https://models.example/v1",
            "credential": None,
            "max_retries": 1,
            **changes,
        }

        with pytest.raises(error, match=message):
            ConnectionRequest(**cast(Any, values))

    @pytest.mark.parametrize(
        ("family", "prefix", "builder", "error", "message"),
        [
            ("", "retry=", MagicMock(), ValueError, "connection_family"),
            (cast(Any, 7), "retry=", MagicMock(), ValueError, "connection_family"),
            ("family", "", MagicMock(), ValueError, "retry_policy_prefix"),
            ("family", "retry", MagicMock(), ValueError, "end with '='"),
            ("family", "retry=", cast(Any, object()), TypeError, "callable"),
        ],
    )
    def test_factory_spec_rejects_invalid_registry_metadata(
        self,
        family: object,
        prefix: object,
        builder: object,
        error: type[Exception],
        message: str,
    ) -> None:
        with pytest.raises(error, match=message):
            ConnectionFactorySpec(
                cast(Any, family), cast(Any, prefix), cast(Any, builder)
            )

    @pytest.mark.parametrize("value", [cast(Any, True), cast(Any, "2"), -1])
    def test_factory_retry_policy_rejects_invalid_retry_count(
        self, value: object
    ) -> None:
        spec = ConnectionFactorySpec("family", "retry=", MagicMock())

        with pytest.raises((TypeError, ValueError), match="integer|non-negative"):
            spec.retry_policy(cast(Any, value))

    @pytest.mark.parametrize(
        "policy",
        ["retry=", "retry=two", "retry=２", "retry=-1"],
    )
    def test_factory_rejects_malformed_retry_policy_suffix(self, policy: str) -> None:
        spec = ConnectionFactorySpec("family", "retry=", MagicMock())

        with pytest.raises(ValueError, match="invalid retry policy"):
            spec.parse_retry_policy(policy)

    def test_registry_rejects_non_specs_and_duplicate_families(self) -> None:
        with pytest.raises(TypeError, match="factory specs"):
            ConnectionFactoryRegistry([cast(Any, object())])

        first = ConnectionFactorySpec("same", "one=", MagicMock())
        second = ConnectionFactorySpec("same", "two=", MagicMock())
        with pytest.raises(ValueError, match="duplicate connection family"):
            ConnectionFactoryRegistry((first, second))

    def test_registry_extension_is_immutable_and_executable(self) -> None:
        first_builder = MagicMock(return_value="first")
        second_builder = MagicMock(return_value="second")
        original = ConnectionFactoryRegistry(
            (ConnectionFactorySpec("first", "first=", first_builder),)
        )
        extended = original.with_factory(
            ConnectionFactorySpec("second", "second=", second_builder)
        )
        request = ConnectionRequest("https://models.example/v1", None, 3)

        assert original.families == {"first"}
        assert extended.families == {"first", "second"}
        assert tuple(extended.specs) == ("first", "second")
        assert extended.get("first").parse_retry_policy("first=3") == 3
        assert extended.validate_retry_policy("second", "second=4") == 4
        assert extended.create("second", request) == "second"
        second_builder.assert_called_once_with(request)


class TestBuiltInConnectionFactories:
    @pytest.mark.anyio
    async def test_async_http_json_type_credential_boundary_and_close(self) -> None:
        secret = "provider-secret-value"
        request = ConnectionRequest(
            endpoint="https://models.example/v1",
            credential=secret,
            max_retries=2,
        )

        connection = CONNECTION_FACTORY_REGISTRY.create(
            "async_http_json",
            request,
        )

        assert isinstance(connection, AsyncHTTPJSONConnection)
        assert isinstance(connection.client, httpx.AsyncClient)
        assert connection.credential == secret
        assert connection.client.base_url == httpx.URL("https://models.example/v1/")
        lowered_headers = {name.lower() for name in connection.client.headers}
        assert "authorization" not in lowered_headers
        assert "x-api-key" not in lowered_headers
        assert "x-goog-api-key" not in lowered_headers
        assert secret not in repr(request)
        assert secret not in repr(connection)

        await connection.aclose()
        assert connection.client.is_closed

    def test_openai_factory_preserves_previous_constructor_contract(self) -> None:
        connection = object()
        request = ConnectionRequest(
            endpoint="https://openai-compatible.example/v1",
            credential="sk-runtime-only",
            max_retries=7,
        )

        with patch(
            "sieval.core.models.connection_factory.AsyncOpenAI",
            return_value=connection,
        ) as client_type:
            created = CONNECTION_FACTORY_REGISTRY.create("openai_sdk", request)

        assert created is connection
        client_type.assert_called_once_with(
            base_url="https://openai-compatible.example/v1",
            api_key="sk-runtime-only",
            max_retries=7,
            timeout=DEFAULT_REQUEST_TIMEOUT,
        )

    def test_the_request_timeout_is_declared_here_not_inherited(self) -> None:
        """Asserted literally on purpose.

        Reading the numbers off ``httpx`` or the OpenAI SDK would re-derive them
        from the very defaults this constant exists to stop depending on, so the
        assertion would survive the upstream change it is meant to catch.
        """
        assert DEFAULT_REQUEST_TIMEOUT.read == 600.0
        assert DEFAULT_REQUEST_TIMEOUT.write == 600.0
        assert DEFAULT_REQUEST_TIMEOUT.pool == 600.0
        assert DEFAULT_REQUEST_TIMEOUT.connect == 5.0

    @pytest.mark.anyio
    async def test_both_families_bound_a_request_by_the_same_declared_timeout(
        self,
    ) -> None:
        """One declared bound, not one per library.

        The two library defaults differ by 120x, so before this was declared a
        dialect's connection family decided whether a long generation could
        finish.
        """
        request = ConnectionRequest(
            endpoint="https://models.example/v1",
            credential="runtime-only",
            max_retries=2,
        )

        http_json = CONNECTION_FACTORY_REGISTRY.create("async_http_json", request)
        assert isinstance(http_json, AsyncHTTPJSONConnection)
        assert http_json.client.timeout == DEFAULT_REQUEST_TIMEOUT
        await http_json.aclose()

        openai_sdk = CONNECTION_FACTORY_REGISTRY.create("openai_sdk", request)
        assert cast(Any, openai_sdk).timeout == DEFAULT_REQUEST_TIMEOUT
        await cast(Any, openai_sdk).close()
