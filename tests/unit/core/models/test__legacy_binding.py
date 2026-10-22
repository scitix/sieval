"""Tests for the legacy ``ChatModel``/``GenModel`` binding construction.

Scoped to what this path owes *independently* of ``connection_factory``, which it
bypasses.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import ANY, AsyncMock, patch

import httpx
import pytest

from sieval.core.models._legacy_binding import (
    _legacy_provenance_projector_for_plan,
    _LegacyOpenAIBinding,
    build_legacy_openai_binding,
)
from sieval.core.models.connection_factory import DEFAULT_REQUEST_TIMEOUT
from sieval.core.models.reconcile import RuntimeBindingPlan
from sieval.core.models.request_log import RecordingOpenAIClient


def _a_binding(*, api_key: str | None = "sk-runtime-only") -> _LegacyOpenAIBinding:
    """Build a wrapper binding without opening a client."""

    client = SimpleNamespace(base_url="https://legacy.example/v1/", close=AsyncMock())
    with patch(
        "sieval.core.models._legacy_binding.AsyncOpenAI",
        return_value=client,
    ):
        return build_legacy_openai_binding(
            dialect_id="openai_chat",
            model="m",
            api_base="https://legacy.example/v1",
            api_key=api_key,
            max_retries=4,
            concurrency_limit=None,
            parent_limiter=None,
        )


def _projected_plan() -> RuntimeBindingPlan:
    """Return the stable provenance plan a wrapper persists."""

    binding = _a_binding()
    projected = binding.provenance_projector(binding.runtime_plan)
    assert projected is not None
    return projected


class TestLegacyBindingClient:
    def test_constructor_builds_provenance_without_recovering_its_own_plan(
        self,
    ) -> None:
        client = SimpleNamespace(
            base_url="https://legacy.example/v1/",
            close=AsyncMock(),
        )
        with (
            patch(
                "sieval.core.models._legacy_binding.AsyncOpenAI",
                return_value=client,
            ),
            patch(
                "sieval.core.models._legacy_binding."
                "_legacy_provenance_projector_for_plan",
                side_effect=AssertionError("constructor must not use recovery"),
            ),
        ):
            binding = build_legacy_openai_binding(
                dialect_id="openai_chat",
                model="m",
                api_base="https://legacy.example/v1",
                api_key="sk-runtime-only",
                max_retries=4,
                concurrency_limit=None,
                parent_limiter=None,
            )

        assert binding.provenance_projector(binding.runtime_plan) is not None

    def test_client_declares_the_shared_request_timeout(self) -> None:
        """The wrapper path owes the same declared bound as the factory.

        ``ChatModel`` and ``GenModel`` both build their client here, so this is
        the construction serving runs today. Its fallback is the SDK's default,
        which is the same value -- so a drift would be silent without this.
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
            http_client=ANY,
        )
        recording = client_factory.call_args.kwargs["http_client"]
        assert isinstance(recording, RecordingOpenAIClient)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (
                {"read": 1800},
                httpx.Timeout(connect=5.0, read=1800.0, write=600.0, pool=600.0),
            ),
            (httpx.Timeout(7.0), httpx.Timeout(7.0)),
        ],
    )
    def test_client_resolves_a_mapping_or_timeout(
        self, value: object, expected: httpx.Timeout
    ) -> None:
        """``ChatModel(**grader)`` hands over the YAML mapping as is."""
        client = SimpleNamespace(
            base_url="https://legacy.example/v1/", close=AsyncMock()
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
                client_timeout=cast(Any, value),
            )

        assert client_factory.call_args.kwargs["timeout"] == expected

    def test_client_refuses_an_unknown_timeout_phase(self) -> None:
        with pytest.raises(ValueError, match=r"Model 'm' 'client_timeout' has unknown"):
            build_legacy_openai_binding(
                dialect_id="openai_chat",
                model="m",
                api_base="https://legacy.example/v1",
                api_key="sk-runtime-only",
                max_retries=4,
                concurrency_limit=None,
                parent_limiter=None,
                client_timeout=cast(Any, {"conect": 30}),
            )

    def test_persisted_fingerprints_exclude_the_private_runtime_scope(self) -> None:
        client = SimpleNamespace(
            base_url="https://legacy.example/v1/",
            close=AsyncMock(),
        )
        with (
            patch(
                "sieval.core.models._legacy_binding.AsyncOpenAI",
                return_value=client,
            ),
            patch(
                "sieval.core.models._legacy_binding.uuid4",
                side_effect=[
                    SimpleNamespace(hex="0" * 32),
                    SimpleNamespace(hex="1" * 32),
                ],
            ),
        ):
            first = build_legacy_openai_binding(
                dialect_id="openai_chat",
                model="m",
                api_base="https://legacy.example/v1",
                api_key="sk-runtime-only",
                max_retries=4,
                concurrency_limit=None,
                parent_limiter=None,
            )
            second = build_legacy_openai_binding(
                dialect_id="openai_chat",
                model="m",
                api_base="https://legacy.example/v1",
                api_key="sk-runtime-only",
                max_retries=4,
                concurrency_limit=None,
                parent_limiter=None,
            )

        assert first.pool.identity != second.pool.identity
        assert first.runtime_plan.fingerprint != second.runtime_plan.fingerprint
        first_provenance = first.provenance_projector(first.runtime_plan)
        second_provenance = second.provenance_projector(second.runtime_plan)
        assert first_provenance is not None
        assert second_provenance is not None
        assert first_provenance.fingerprint == second_provenance.fingerprint
        assert (
            first_provenance.verification_fingerprint
            == second_provenance.verification_fingerprint
        )

    def test_projection_preserves_distinct_sibling_binding_identity(self) -> None:
        client = SimpleNamespace(
            base_url="https://legacy.example/v1/",
            close=AsyncMock(),
        )
        with patch(
            "sieval.core.models._legacy_binding.AsyncOpenAI",
            return_value=client,
        ):
            binding = build_legacy_openai_binding(
                dialect_id="openai_chat",
                model="m",
                api_base="https://legacy.example/v1",
                api_key="sk-runtime-only",
                max_retries=4,
                concurrency_limit=None,
                parent_limiter=None,
            )

        base = binding.provenance_projector(binding.runtime_plan)
        assert base is not None
        sibling_runtime = replace(
            binding.runtime_plan,
            binding_id=f"{binding.runtime_plan.binding_id}:sibling",
        )
        sibling = binding.provenance_projector(sibling_runtime)
        assert sibling is not None

        assert sibling.binding_id.startswith(f"{base.binding_id}:sibling:")
        assert sibling.root_deployment_key == base.root_deployment_key
        assert sibling.fingerprint != base.fingerprint
        assert sibling.verification_fingerprint != base.verification_fingerprint

    def test_projection_rejects_foreign_binding_that_matches_stable_base(self) -> None:
        client = SimpleNamespace(
            base_url="https://legacy.example/v1/",
            close=AsyncMock(),
        )
        with patch(
            "sieval.core.models._legacy_binding.AsyncOpenAI",
            return_value=client,
        ):
            binding = build_legacy_openai_binding(
                dialect_id="openai_chat",
                model="m",
                api_base="https://legacy.example/v1",
                api_key="sk-runtime-only",
                max_retries=4,
                concurrency_limit=None,
                parent_limiter=None,
            )

        base = binding.provenance_projector(binding.runtime_plan)
        assert base is not None
        foreign = replace(binding.runtime_plan, binding_id=base.binding_id)

        assert binding.provenance_projector(foreign) is None

    def test_projection_rejects_a_plan_from_another_runtime_root(self) -> None:
        client = SimpleNamespace(
            base_url="https://legacy.example/v1/",
            close=AsyncMock(),
        )
        with patch(
            "sieval.core.models._legacy_binding.AsyncOpenAI",
            return_value=client,
        ):
            binding = build_legacy_openai_binding(
                dialect_id="openai_chat",
                model="m",
                api_base="https://legacy.example/v1",
                api_key="sk-runtime-only",
                max_retries=4,
                concurrency_limit=None,
                parent_limiter=None,
            )

        foreign_root = replace(
            binding.runtime_plan,
            root_deployment_key="legacy:foreign-runtime-root",
        )

        assert binding.provenance_projector(foreign_root) is None

    def test_projection_rejects_changed_opaque_plan_evidence(self) -> None:
        client = SimpleNamespace(
            base_url="https://legacy.example/v1/",
            close=AsyncMock(),
        )
        with patch(
            "sieval.core.models._legacy_binding.AsyncOpenAI",
            return_value=client,
        ):
            binding = build_legacy_openai_binding(
                dialect_id="openai_chat",
                model="m",
                api_base="https://legacy.example/v1",
                api_key="sk-runtime-only",
                max_retries=4,
                concurrency_limit=None,
                parent_limiter=None,
            )

        assert (
            binding.provenance_projector(
                replace(
                    binding.runtime_plan,
                    binding_plan_fingerprint="opaque:binding-proof",
                )
            )
            is None
        )
        assert (
            binding.provenance_projector(
                replace(
                    binding.runtime_plan,
                    deployment_plan_fingerprint="opaque:deployment-proof",
                )
            )
            is None
        )

    def test_persisted_fingerprints_record_credential_category_not_secret(self) -> None:
        def build(api_key: str):
            client = SimpleNamespace(
                base_url="https://legacy.example/v1/",
                close=AsyncMock(),
            )
            with patch(
                "sieval.core.models._legacy_binding.AsyncOpenAI",
                return_value=client,
            ):
                return build_legacy_openai_binding(
                    dialect_id="openai_chat",
                    model="m",
                    api_base="https://legacy.example/v1",
                    api_key=api_key,
                    max_retries=4,
                    concurrency_limit=None,
                    parent_limiter=None,
                )

        first = build("first-secret")
        second = build("second-secret")

        assert first.runtime_plan.fingerprint != second.runtime_plan.fingerprint
        first_provenance = first.provenance_projector(first.runtime_plan)
        second_provenance = second.provenance_projector(second.runtime_plan)
        assert first_provenance is not None
        assert second_provenance is not None
        assert first_provenance.fingerprint == second_provenance.fingerprint
        assert "first-secret" not in repr(first)
        assert "second-secret" not in repr(second)

    def test_persisted_fingerprint_changes_with_semantic_binding_inputs(self) -> None:
        def build(
            *,
            dialect_id: str = "openai_chat",
            model: str = "m",
            api_base: str = "https://legacy.example/v1",
            api_key: str | None = "sk-runtime-only",
            max_retries: int = 4,
        ):
            client = SimpleNamespace(
                base_url=f"{api_base.rstrip('/')}/",
                close=AsyncMock(),
            )
            with patch(
                "sieval.core.models._legacy_binding.AsyncOpenAI",
                return_value=client,
            ):
                return build_legacy_openai_binding(
                    dialect_id=dialect_id,
                    model=model,
                    api_base=api_base,
                    api_key=api_key,
                    max_retries=max_retries,
                    concurrency_limit=None,
                    parent_limiter=None,
                )

        baseline = build()
        variants = (
            build(model="another-model"),
            build(max_retries=5),
            build(api_base="https://other.example/v1"),
            build(dialect_id="openai_completions"),
            build(api_key=None),
        )

        baseline_provenance = baseline.provenance_projector(baseline.runtime_plan)
        variant_provenance = [
            variant.provenance_projector(variant.runtime_plan) for variant in variants
        ]
        assert baseline_provenance is not None
        assert all(projected is not None for projected in variant_provenance)
        assert all(
            projected.fingerprint != baseline_provenance.fingerprint
            for projected in variant_provenance
            if projected is not None
        )
        assert all(
            projected.verification_fingerprint
            != baseline_provenance.verification_fingerprint
            for projected in variant_provenance
            if projected is not None
        )

    def test_projected_plan_is_not_reported_as_a_malformed_runtime_identity(
        self,
    ) -> None:
        client = SimpleNamespace(
            base_url="https://legacy.example/v1/",
            close=AsyncMock(),
        )
        with patch(
            "sieval.core.models._legacy_binding.AsyncOpenAI",
            return_value=client,
        ):
            binding = build_legacy_openai_binding(
                dialect_id="openai_chat",
                model="m",
                api_base="https://legacy.example/v1",
                api_key="sk-runtime-only",
                max_retries=4,
                concurrency_limit=None,
                parent_limiter=None,
            )

        projected = binding.provenance_projector(binding.runtime_plan)
        assert projected is not None
        assert _legacy_provenance_projector_for_plan(projected) is None

    def test_projected_plan_with_a_tampered_credential_scope_fails_closed(
        self,
    ) -> None:
        """A stable quota scope does not license an arbitrary credential scope.

        Falling through as canonical would persist forged evidence verbatim.
        """

        projected = _projected_plan()
        forged = replace(
            projected,
            connection_identity=replace(
                projected.connection_identity,
                credential_scope="legacy-private:forged-credential",
            ),
        )

        with pytest.raises(ValueError, match="invalid credential scope"):
            _legacy_provenance_projector_for_plan(forged)

    def test_legacy_credential_scope_without_a_legacy_quota_scope_fails_closed(
        self,
    ) -> None:
        """A half-legacy identity is malformed, not canonical."""

        plan = _a_binding().runtime_plan
        mismatched = replace(
            plan,
            connection_identity=replace(
                plan.connection_identity,
                quota_scope="shared-pool",
            ),
        )

        with pytest.raises(ValueError, match="invalid quota scope"):
            _legacy_provenance_projector_for_plan(mismatched)

    def test_recovery_rejects_a_plan_whose_binding_identity_was_rewritten(
        self,
    ) -> None:
        """Recovery re-derives the binding id and will not adopt a foreign one."""

        plan = _a_binding().runtime_plan
        rewritten = replace(plan, binding_id="legacy:someone-elses:binding:0000")

        with pytest.raises(ValueError, match="inconsistent binding identity"):
            _legacy_provenance_projector_for_plan(rewritten)

    def test_environment_credential_plans_recover_their_stable_identity(self) -> None:
        """The environment-credential category survives a ``Model.bind`` round-trip."""

        binding = _a_binding(api_key=None)
        runtime_scope = binding.runtime_plan.connection_identity.credential_scope
        assert runtime_scope.endswith(":environment-credential")

        projector = _legacy_provenance_projector_for_plan(binding.runtime_plan)
        assert projector is not None
        assert (
            projector.connection_identity.credential_scope
            == "legacy-private:environment-credential"
        )
        # Recovery must land on exactly what the constructor already built.
        assert projector.connection_identity == (
            binding.provenance_projector.connection_identity
        )
