"""Decision requests through the shared audit and output contract.

AI-Generated Code - GPT-6 (OpenAI)
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from unittest.mock import AsyncMock

import pytest

from sieval.core.models import (
    ChoiceAnswer,
    ChoiceOption,
    ChoiceProbability,
    ChoiceQuestion,
    ContextInput,
    DecisionParams,
    DecisionResult,
    DecisionsOptions,
    JsonPart,
    Model,
    PredicateAnswer,
    PredicateQuestion,
    Request,
    RequestDefaults,
    Response,
    SamplingParams,
    SchedulingParams,
    TextPart,
)
from sieval.core.models.dialect import (
    DialectError,
    Guarantee,
    OutputContract,
    OutputContractError,
    OutputRule,
    PreparedRequest,
    RequestAudit,
    RequestAuditError,
    active_request_capabilities,
    active_request_leaves,
    required_response_channels,
    validate_request_invariants,
)
from sieval.core.models.dialect_registry import DIALECT_BINDERS
from sieval.core.models.ir import response_field_contract
from sieval.core.models.reconcile import RuntimeBindingPlan
from sieval.core.types import JSONValue
from tests.conftest import HandlerTransport
from tests.unit.core.models.test_dialect_registry import _runtime_binding


@dataclass(frozen=True)
class Plan:
    dialect_id: str = "test_decisions"
    available_capabilities: frozenset[str] = frozenset({"decisions"})
    capability_minimums: Mapping[str, Mapping[str, JSONValue]] = field(
        default_factory=dict
    )
    required_output_channels: frozenset[str] = frozenset({"decisions"})


def request():
    return Request(
        ContextInput((TextPart("A damaged item"),)),
        decisions=DecisionParams((PredicateQuestion("defect", "Is it damaged?"),)),
    )


def contract():
    rules = {
        name: OutputRule(Guarantee.NEVER)
        for name, (role, _) in response_field_contract().items()
        if role == "channel"
    }
    rules["decisions"] = OutputRule(Guarantee.PRESENT_OR_ERROR)
    return OutputContract(rules)


def test_decision_leaves_activate_one_capability_and_channel():
    req = request()
    assert active_request_capabilities(req) == {"decisions"}
    assert required_response_channels(req) == {"decisions"}
    assert set(active_request_leaves(req)) == {
        "input.context",
        "input.context.text",
        "decisions.questions",
    }


def test_all_question_and_evidence_data_are_subject_to_audit():
    req = request()
    audit = RequestAudit(active_request_leaves(req))
    audit.noop("input.context.text", "The context record carries its text")
    with pytest.raises(RequestAuditError, match="unaccounted"):
        audit.finish(PreparedRequest("test", {"state": "A damaged item"}))


def test_nontext_decision_response_has_no_placeholder_generation():
    response = Response(
        texts=(), decisions=DecisionResult((PredicateAnswer("defect", 0.9),))
    )
    contract().validate(Plan(), request(), response)
    with pytest.raises(OutputContractError, match="placeholder"):
        contract().validate(
            Plan(), request(), Response(texts=("",), decisions=response.decisions)
        )


def test_decision_finish_reason_is_one_call_level_value_without_texts():
    response = Response(
        texts=(),
        decisions=DecisionResult((PredicateAnswer("defect", 0.9),)),
        finish_reasons=("length",),
        finish_reason_sources=("test.call",),
    )
    contract().validate(Plan(), request(), response)
    assert response.finish_reasons == ("length",)
    with pytest.raises(ValueError, match="finish_reasons"):
        Response(
            texts=(), decisions=response.decisions, finish_reasons=("stop", "length")
        )


def test_decision_refusal_finish_reason_cannot_bypass_invalid_answers():
    response = Response(
        texts=(), decisions=DecisionResult(()), finish_reasons=("refusal",)
    )
    with pytest.raises(OutputContractError, match="identity"):
        contract().validate(Plan(), request(), response)


def test_finish_reason_cannot_bypass_missing_decision_answers():
    with pytest.raises(OutputContractError, match="absent"):
        contract().validate(Plan(), request(), Response(texts=(), finish_reasons=()))


@pytest.mark.parametrize(
    "sampling,stream", [(SamplingParams(n=2), False), (SamplingParams(), True)]
)
def test_decisions_do_not_reuse_generation_n_or_streaming(sampling, stream):
    req = Request(
        request().input,
        sampling=sampling,
        scheduling=SchedulingParams(stream=stream),
        decisions=request().decisions,
    )
    with pytest.raises(ValueError):
        validate_request_invariants(req)


def test_context_is_explicit_and_requires_questions():
    with pytest.raises(ValueError, match="questions"):
        validate_request_invariants(Request(request().input))
    with pytest.raises(ValueError, match="object or array"):
        ContextInput((JsonPart.from_value(None),))


@pytest.mark.anyio
@pytest.mark.parametrize("guarantee", ["require_probabilities", "require_confidence"])
async def test_task_view_projects_and_enforces_guarantees_without_changing_shared_model(
    guarantee, monkeypatch
):
    deployment, pool, route = _runtime_binding("openai_chat")
    plan = RuntimeBindingPlan(
        binding_id="task-view",
        root_deployment_key="root",
        requested_model_id="model",
        dialect_id="openai_chat",
        declared_capabilities={},
        effective_capabilities={},
        available_capabilities=frozenset({"decisions"}),
        capability_minimums={},
        request_defaults=RequestDefaults(),
        required_output_channels=frozenset({"decisions"}),
        request_checks=(),
        deployment_fingerprint=deployment.fingerprint,
        resolved_route=route.resolved_route,
        connection_identity=pool.identity,
        binding_plan_fingerprint="binding",
        deployment_plan_fingerprint="deployment",
    )
    model = Model.bind(deployment, pool, plan)

    async def execute(_req):
        return Response(texts=(), decisions=DecisionResult((ChoiceAnswer("q", "a"),)))

    dialect = HandlerTransport(execute, "openai_chat")
    monkeypatch.setattr(model, "_dialect", dialect)
    options = DecisionsOptions(
        require_probabilities=guarantee == "require_probabilities",
        require_confidence=guarantee == "require_confidence",
    )
    view = model.with_decision_requirements(options)
    req = Request(
        ContextInput((TextPart("evidence"),)),
        decisions=DecisionParams(
            (ChoiceQuestion("q", "Choose", (ChoiceOption("a"), ChoiceOption("b"))),)
        ),
    )
    try:
        with pytest.raises(
            OutputContractError, match=guarantee.removeprefix("require_")
        ):
            await view.arun(req)
        await model.arun(req)
        assert getattr(dialect.requests[0].decisions, guarantee) is True
        assert getattr(dialect.requests[1].decisions, guarantee) is False
        assert getattr(req.decisions, guarantee) is False
        assert view.runtime_plan is model.runtime_plan
    finally:
        await pool.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("guarantee", ["require_probabilities", "require_confidence"])
@pytest.mark.parametrize("declaration", ["absent", "empty", "false", "true"])
async def test_binding_declarations_enforce_output_guarantees_but_task_floors_do_not(
    guarantee, declaration, monkeypatch
):
    deployment, pool, route = _runtime_binding("openai_chat")
    declared: dict[str, JSONValue] = {}
    if declaration != "absent":
        declared["decisions"] = (
            {} if declaration == "empty" else {guarantee: declaration == "true"}
        )
    plan = RuntimeBindingPlan(
        binding_id="declaration",
        root_deployment_key="root",
        requested_model_id="model",
        dialect_id="openai_chat",
        declared_capabilities=declared,
        effective_capabilities={},
        available_capabilities=frozenset({"decisions"}),
        # An aggregated task floor must not strengthen other consumers' calls.
        capability_minimums={"decisions": {guarantee: True}},
        request_defaults=RequestDefaults(),
        required_output_channels=frozenset({"decisions"}),
        request_checks=(),
        deployment_fingerprint=deployment.fingerprint,
        resolved_route=route.resolved_route,
        connection_identity=pool.identity,
        binding_plan_fingerprint="binding",
        deployment_plan_fingerprint="deployment",
    )
    model = Model.bind(deployment, pool, plan)
    answer = ChoiceAnswer("q", "a")

    async def execute(_req):
        return Response(texts=(), decisions=DecisionResult((answer,)))

    dialect = HandlerTransport(execute, "openai_chat")
    monkeypatch.setattr(model, "_dialect", dialect)
    req = Request(
        ContextInput((TextPart("evidence"),)),
        decisions=DecisionParams(
            (ChoiceQuestion("q", "Choose", (ChoiceOption("a"), ChoiceOption("b"))),)
        ),
    )
    other_guarantee = (
        "require_confidence"
        if guarantee == "require_probabilities"
        else "require_probabilities"
    )
    local = model.with_decision_requirements(
        DecisionsOptions(
            require_probabilities=other_guarantee == "require_probabilities",
            require_confidence=other_guarantee == "require_confidence",
        )
    )
    try:
        if declaration == "true":
            with pytest.raises(
                OutputContractError, match=guarantee.removeprefix("require_")
            ):
                await model.arun(req)
        else:
            await model.arun(req)
        assert getattr(dialect.requests[-1].decisions, guarantee) is (
            declaration == "true"
        )
        assert getattr(dialect.requests[-1].decisions, other_guarantee) is False

        with pytest.raises(OutputContractError):
            await local.arun(req)
        assert getattr(dialect.requests[-1].decisions, guarantee) is (
            declaration == "true"
        )
        assert getattr(dialect.requests[-1].decisions, other_guarantee) is True
        assert req.decisions is not None
        assert req.decisions.require_probabilities is False
        assert req.decisions.require_confidence is False

        # Rebinding reads the new configuration while retaining task-local flags.
        rebound_plan = replace(plan, declared_capabilities={})
        rebound = local.with_dialect("openai_chat", rebound_plan)
        monkeypatch.setattr(rebound, "_dialect", dialect)
        with pytest.raises(
            OutputContractError, match=other_guarantee.removeprefix("require_")
        ):
            await rebound.arun(req)
        assert getattr(dialect.requests[-1].decisions, guarantee) is False
        assert getattr(dialect.requests[-1].decisions, other_guarantee) is True

        strengthened = replace(
            req,
            decisions=replace(
                req.decisions, require_probabilities=True, require_confidence=True
            ),
        )
        with pytest.raises(OutputContractError):
            await model.arun(strengthened)
        lowered = dialect.requests[-1].decisions
        assert isinstance(lowered, DecisionParams)
        assert lowered.require_probabilities is True
        assert lowered.require_confidence is True
        answer = ChoiceAnswer(
            "q",
            "a",
            probabilities=(ChoiceProbability("a", 0.8), ChoiceProbability("b", 0.2)),
            confidence=0.9,
        )
        response = await local.arun(strengthened)
        assert response.decisions == DecisionResult((answer,))
    finally:
        await pool.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("dialect_id", tuple(DIALECT_BINDERS))
async def test_existing_dialects_reject_decisions_before_execution(
    dialect_id, monkeypatch
):
    deployment, pool, route = _runtime_binding(dialect_id)
    plan = RuntimeBindingPlan(
        binding_id="test",
        root_deployment_key="root",
        requested_model_id="requested-model",
        dialect_id=dialect_id,
        declared_capabilities={},
        effective_capabilities={},
        available_capabilities=frozenset(),
        capability_minimums={},
        request_defaults=RequestDefaults(),
        required_output_channels=frozenset(),
        request_checks=(),
        deployment_fingerprint=deployment.fingerprint,
        resolved_route=route.resolved_route,
        connection_identity=pool.identity,
        binding_plan_fingerprint="binding",
        deployment_plan_fingerprint="deployment",
    )
    model = Model.bind(deployment, pool, plan)
    execute = AsyncMock(side_effect=AssertionError("unsupported decisions executed"))
    monkeypatch.setattr(model._dialect, "execute", execute)
    try:
        with pytest.raises(DialectError, match="decisions"):
            await model.arun(request())
        execute.assert_not_awaited()
    finally:
        await pool.aclose()
