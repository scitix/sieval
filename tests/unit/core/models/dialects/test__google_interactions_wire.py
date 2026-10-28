"""Contract tests for Gemini Interactions route resolution.

AI-Generated Code - Claude Sonnet 5.5 (Anthropic)
"""

import httpx
import pytest

from sieval.core.models.dialects._google_interactions_wire import (
    resolve_interactions_endpoint,
)


class TestResolveInteractionsEndpoint:
    @pytest.mark.parametrize(
        ("path", "version", "root"),
        [
            ("", "v1beta", "/"),
            ("/", "v1beta", "/"),
            ("/v1", "v1", "/"),
            ("/v1beta/", "v1beta", "/"),
            ("/gateway/google", "v1beta", "/gateway/google/"),
            ("/gateway/google/v1/", "v1", "/gateway/google/"),
            ("/gateway/v2/google", "v1beta", "/gateway/v2/google/"),
            ("/gateway/v2/google/v1beta", "v1beta", "/gateway/v2/google/"),
        ],
    )
    def test_root_and_explicit_version_preserve_gateway_prefix(
        self, path: str, version: str, root: str
    ) -> None:
        url = httpx.URL(f"https://example.test{path}")
        resolved_version, endpoint_root = resolve_interactions_endpoint(url)
        assert resolved_version == version
        assert endpoint_root == httpx.URL(f"https://example.test{root}")
        assert str(url) == str(httpx.URL(f"https://example.test{path}"))

    @pytest.mark.parametrize(
        ("path", "operation"),
        [
            ("", "https://example.test/v1beta/interactions"),
            ("/v1", "https://example.test/v1/interactions"),
            ("/gw/v1beta", "https://example.test/gw/v1beta/interactions"),
        ],
    )
    def test_joined_operation_has_no_models_segment(
        self, path: str, operation: str
    ) -> None:
        version, endpoint_root = resolve_interactions_endpoint(
            httpx.URL(f"https://example.test{path}")
        )
        assert endpoint_root.join(f"{version}/interactions") == httpx.URL(operation)

    @pytest.mark.parametrize(
        "path",
        [
            "/v2",
            "/v1alpha/",
            "/gateway/v9beta",
            "/V1",
            "/v1beta/interactions",
            "/v1/interactions/abc",
            "/interactions",
            "/v1?key=secret",
            "/v1#fragment",
        ],
    )
    def test_ambiguous_endpoint_is_rejected(self, path: str) -> None:
        with pytest.raises(ValueError, match="Google api_base"):
            resolve_interactions_endpoint(httpx.URL(f"https://example.test{path}"))

    def test_interactions_like_gateway_segment_is_not_rejected(self) -> None:
        # Only the exact path segment names the resource; a gateway may route
        # through a longer name.
        version, endpoint_root = resolve_interactions_endpoint(
            httpx.URL("https://example.test/interactions-proxy/v1")
        )
        assert version == "v1"
        assert endpoint_root == httpx.URL("https://example.test/interactions-proxy/")
