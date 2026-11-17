"""Decision capability setup through a test-only supported descriptor.

AI-Generated Code - GPT-6 (OpenAI)
"""

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest

from sieval.cli.resolution import ConfigModelTypeResolution
from sieval.cli.session import EvalSession
from sieval.core.models import DecisionsOptions
from sieval.core.models import dialect_registry as registry
from sieval.core.models.capabilities import (
    DialectCapabilityBinding,
    DialectCapabilityStatus,
    ModelCapabilityStatus,
    Supported,
)
from sieval.core.models.reconcile import (
    BindingReconcileInput,
    ConnectionScope,
    reconcile_binding,
)
from sieval.core.models.requirements import (
    AggregatedTaskRequirements,
    InlineModelBinding,
    InputKind,
)


@pytest.fixture
def decision_binding(monkeypatch, tmp_path):
    original = registry.get_dialect_spec("openai_chat")
    spec = replace(
        original,
        dialect_id="test_decisions",
        input_kinds=("context",),
        capability_outcomes={
            **original.capability_outcomes,
            "decisions": DialectCapabilityStatus.SUPPORTED,
        },
    )
    decisions = {
        **registry.capability_decisions_for("openai_chat"),
        "decisions": Supported(
            DialectCapabilityBinding(
                "decisions",
                request_leaves=("input.context", "decisions.questions"),
                response_channels=("decisions",),
            )
        ),
    }
    monkeypatch.setattr(
        registry, "DIALECT_SPECS", {**registry.DIALECT_SPECS, spec.dialect_id: spec}
    )
    monkeypatch.setattr(
        registry,
        "DIALECT_BINDERS",
        {
            **registry.DIALECT_BINDERS,
            spec.dialect_id: cast(
                registry.DialectBinder, SimpleNamespace(capability_decisions=decisions)
            ),
        },
    )
    path = tmp_path / "config.yaml"
    path.write_text("models: {}\ntasks: {}\n")
    session = EvalSession(config_path=str(path))
    binding = InlineModelBinding(
        binding_id="decision",
        root_deployment_key="root",
        requested_model_id="model",
        config={},
        dialect_id=spec.dialect_id,
    )
    return session, binding


def _plan(session, binding, declarations):
    result = reconcile_binding(
        BindingReconcileInput(
            binding_id=binding.binding_id,
            root_deployment_key=binding.root_deployment_key,
            requested_model_id=binding.requested_model_id,
            dialect_id=binding.dialect_id,
            declarations=declarations,
            model_profile=session._model_profile_for(binding),
            requirements=AggregatedTaskRequirements(
                input=frozenset({InputKind.CONTEXT}),
                decisions=DecisionsOptions(),
                decisions_sources=frozenset({"task"}),
            ),
            connection_scope=ConnectionScope("test", "standard", "root"),
        )
    )
    assert result.plan is not None, result.diagnostics
    return result.plan


def test_default_profile_admits_decisions_with_response_time_verification(
    decision_binding,
):
    session, binding = decision_binding
    profile = session._model_profile_for(binding)
    entry = profile.entries["decisions"]
    assert entry.status is ModelCapabilityStatus.UNKNOWN
    assert entry.verifier == "validate_response_channel"
    plan = _plan(session, binding, {})
    assert plan.required_output_channels == {"decisions"}
    assert "decisions" in plan.pending_capabilities


def test_decision_declaration_order_does_not_change_plan_identity(decision_binding):
    session, binding = decision_binding
    first = {
        "decisions": {
            "question_types": ["predicate", "choice"],
            "evidence_kinds": ["text", "json"],
            "guidance_kinds": ["text", "json", "null"],
        }
    }
    reversed_forms = {
        "decisions": {
            key: list(reversed(value)) for key, value in first["decisions"].items()
        }
    }
    plans = [
        _plan(session, binding, declaration) for declaration in (first, reversed_forms)
    ]
    assert plans[0].declared_capabilities == plans[1].declared_capabilities
    assert plans[0].capability_minimums == plans[1].capability_minimums
    assert plans[0].fingerprint == plans[1].fingerprint


@pytest.mark.anyio
async def test_both_serving_entry_points_refuse_context_with_the_same_message(
    tmp_path, monkeypatch
):
    import yaml

    from sieval.cli.infer.recipe import resolve_infer_config
    from sieval.cli.run import _prepare_launch_batch

    config = {
        "models": {"decision": {"infer": {"backend": "sglang", "checkpoint": "unused"}}}
    }
    path = tmp_path / "context.yaml"
    path.write_text(yaml.safe_dump(config))
    resolution = ConfigModelTypeResolution(
        model_types_by_root={"root": None}, model_types_by_config={"decision": None}
    )
    monkeypatch.setattr(
        "sieval.cli.run.resolve_config_model_types", lambda _: resolution
    )
    with pytest.raises(ValueError) as recipe_error:
        await resolve_infer_config(path, model_type_resolution=resolution)
    with pytest.raises(ValueError) as run_error:
        await _prepare_launch_batch(path, config, effective_deterministic=False)
    assert str(recipe_error.value) == str(run_error.value)
    assert str(run_error.value) == (
        "Model 'decision' uses context input; generation serving recipes "
        "cannot serve it. "
        "Configure an external endpoint with an explicit dialect."
    )
