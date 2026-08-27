"""Tests for the legacy ``ChatModel``/``GenModel`` binding construction.

Scoped to what this path owes *independently* of ``connection_factory``, which it
bypasses.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sieval.core.models._legacy_binding import build_legacy_openai_binding
from sieval.core.models.connection_factory import DEFAULT_REQUEST_TIMEOUT


class TestLegacyBindingClient:
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
                    SimpleNamespace(hex="first-runtime-scope"),
                    SimpleNamespace(hex="second-runtime-scope"),
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
