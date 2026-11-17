"""Typed decisions and model-call metadata through the existing disk pipeline.

The dialect seam is a test double; no native decision API is implemented here.

AI-Generated Code - GPT-6 (OpenAI)
"""

import json
from dataclasses import replace

import pytest

from sieval.core.models import (
    ContextInput,
    DecisionParams,
    DecisionResult,
    DecisionsOptions,
    JsonPart,
    Model,
    ModelProvenance,
    PredicateAnswer,
    PredicateQuestion,
    Request,
    RequestDefaults,
    Response,
    UsageStats,
)
from sieval.core.models.reconcile import RuntimeBindingPlan
from sieval.core.models.requirements import InputKind, TaskRequirements
from sieval.core.runners import TaskRunner
from sieval.core.tasks import TaskStageOutput
from sieval.core.tasks.consts import TaskStage
from sieval.core.tasks.loader import TaskLoader
from sieval.core.tasks.task import Task
from sieval.core.utils.meta import build_model_call_meta_from_response, build_stage_meta
from tests.conftest import HandlerTransport, MockDataset, make_config
from tests.unit.core.models.test_dialect_registry import _runtime_binding

EVIDENCE = {"customer": "Ada", "damage": True, "notes": None, "history": [None]}


class DecisionRecordTask(Task):
    tags = frozenset({"decisions"})
    requires = TaskRequirements(
        input=InputKind.CONTEXT,
        decisions=DecisionsOptions(
            question_types=("predicate",), evidence_kinds=("json",)
        ),
    )

    def __init__(self, dataset, model, name=None, *, provide_meta=True):
        self._provide_meta = provide_meta
        super().__init__(dataset=dataset, model=model, name=name)

    async def preprocess(self, raw, ctx):
        return {
            "evidence": JsonPart.from_value(EVIDENCE),
            "questions": DecisionParams(
                (PredicateQuestion("damage", "Is there damage?"),)
            ),
        }

    async def infer(self, pre, ctx):
        response = await self.model.arun(
            Request(
                ContextInput((pre["evidence"],)),
                decisions=pre["questions"],
            )
        )
        if not self._provide_meta:
            return TaskStageOutput(value=response)
        return TaskStageOutput(
            value=response,
            meta=build_stage_meta(
                model_calls=[
                    build_model_call_meta_from_response(self.model.meta(), response)
                ]
            ),
        )

    async def postprocess(self, inf, ctx):
        return inf.value.decisions.answers[0].probability

    async def feedback(self, post, ctx):
        return True, {"correct": post > 0.5}

    async def report(self, finals, fails):
        return {
            "correct": sum(item.feedback_result["correct"] for item in finals),
            "failed": len(fails),
        }


class ConfidenceDecisionRecordTask(DecisionRecordTask):
    requires = replace(
        DecisionRecordTask.requires,
        decisions=DecisionsOptions(
            question_types=("predicate",),
            evidence_kinds=("json",),
            require_confidence=True,
        ),
    )


@pytest.mark.anyio
@pytest.mark.parametrize("record_meta", [True, False])
@pytest.mark.parametrize("record_each_stage", [True, False])
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
@pytest.mark.parametrize("provide_meta", [True, False])
@pytest.mark.parametrize(
    "task_type", [DecisionRecordTask, ConfidenceDecisionRecordTask]
)
async def test_decision_records_and_usage_survive_fresh_disk_load(
    tmp_path,
    monkeypatch,
    record_meta,
    record_each_stage,
    finish_reason,
    provide_meta,
    task_type,
):
    deployment, pool, route = _runtime_binding("openai_chat")
    plan = RuntimeBindingPlan(
        binding_id="test-decision",
        root_deployment_key="root",
        requested_model_id="test-model",
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
        binding_plan_fingerprint="test-binding",
        deployment_plan_fingerprint="test-deployment",
    )
    model = Model.bind(deployment, pool, plan)

    async def execute(req):
        assert isinstance(req.input, ContextInput)
        assert req.input.parts[0].value == EVIDENCE
        requires_confidence = task_type is ConfidenceDecisionRecordTask
        assert req.decisions.require_confidence is requires_confidence
        return Response(
            texts=(),
            decisions=DecisionResult(
                (
                    PredicateAnswer(
                        "damage", 0.85, confidence=0.8 if requires_confidence else None
                    ),
                )
            ),
            finish_reasons=(finish_reason,),
            usage=UsageStats(
                input_tokens=10,
                output_tokens=2,
                total_tokens=12,
                cached_tokens=0,
            ),
            request_params={"operation": "test-decision"},
            response_model="served-model",
        )

    # Supply the future dialect's descriptor and executor at their test seams.
    monkeypatch.setattr(
        "sieval.core.tasks.task._dialect_input_shape",
        lambda _: (frozenset({"context"}), frozenset({"text"})),
    )
    dialect = HandlerTransport(execute, "openai_chat")
    monkeypatch.setattr(model, "_dialect", dialect)
    task = task_type(
        dataset=MockDataset([{"id": "one"}]),
        model=model,
        name="decision-record",
        provide_meta=provide_meta,
    )
    config = make_config(
        tmp_path,
        record_meta=record_meta,
        record_each_stage=record_each_stage,
        profile_usage=True,
        detect_anomalies=True,
    )
    runner = TaskRunner(task, config)
    try:
        report = await runner.arun()
        assert report["correct"] == 1
        assert report["failed"] == 0
        n_truncated = int(finish_reason == "length")
        assert report["n_truncated"] == n_truncated
        assert report["n_scored_rollouts"] == 1
        saved_report = json.loads((runner.root_dir / "report.json").read_text())
        assert saved_report["failed"] == 0
        assert saved_report["n_truncated"] == n_truncated
        assert saved_report["n_scored_rollouts"] == 1
        anomalies = json.loads((runner.root_dir / "anomalies.json").read_text())
        assert (
            sum(
                "truncated_output" in rules
                for iterations in anomalies["samples"].values()
                for rules in iterations.values()
            )
            == n_truncated
        )
        loader = TaskLoader(task=task, root_dir=runner.root_dir)
        contexts = await loader.load_initial_state()
        await loader.hydrate(
            contexts,
            set(),
            include_stages={TaskStage.FINAL},
            record_each_stage=record_each_stage,
        )
        assert len(contexts) == 1
        ctx = next(iter(contexts.values()))
        assert ctx.stage is TaskStage.FINAL
        assert isinstance(ctx.preprocess_result, dict)
        assert ctx.preprocess_result["evidence"].value == EVIDENCE
        assert isinstance(ctx.preprocess_result["questions"], DecisionParams)
        assert isinstance(ctx.infer_result, TaskStageOutput)
        response = ctx.infer_result.value
        assert isinstance(response, Response)
        assert response.texts == ()
        assert response.decisions == DecisionResult(
            (
                PredicateAnswer(
                    "damage",
                    0.85,
                    confidence=0.8
                    if task_type is ConfidenceDecisionRecordTask
                    else None,
                ),
            )
        )
        assert response.finish_reasons == (finish_reason,)
        assert isinstance(response.provenance, ModelProvenance)
        assert response.request_params == {"operation": "test-decision"}
        assert response.usage is not None
        assert response.usage.cached_tokens == 0
        profile = json.loads((runner.root_dir / "profile.json").read_text())
        assert profile["token_usage"][TaskStage.INFERRED.value]["input"]["total"] == 10
        assert profile["token_usage"][TaskStage.INFERRED.value]["output"]["total"] == 2
        if record_meta:
            calls = ctx.stage_meta["inferred"][0]["model_calls"]
            assert len(calls) == 1
            assert calls[0]["model"]["provenance"] == response.provenance
            assert calls[0]["usage"]["total_tokens"] == 12
            assert calls[0]["usage"]["cached_tokens"] == 0
            assert calls[0]["finish_reasons"] == [finish_reason]
            assert "reasoning_tokens" not in calls[0]["usage"]
        else:
            assert ctx.stage_meta == {}
        (runner.root_dir / "report.json").unlink()
        resumed = TaskRunner(task, replace(config, resume=True))
        resumed_report = await resumed.arun()
        assert resumed_report["correct"] == 1
        assert resumed_report["failed"] == 0
        if record_meta:
            assert resumed_report["n_truncated"] == n_truncated
            assert resumed_report["n_scored_rollouts"] == 1
        else:
            assert "n_truncated" not in resumed_report
            assert "n_scored_rollouts" not in resumed_report
        if record_meta:
            resumed_profile = json.loads((runner.root_dir / "profile.json").read_text())
            assert resumed_profile["token_usage"]["inferred"]["input"]["total"] == 10
            assert resumed_profile["token_usage"]["inferred"]["output"]["total"] == 2
        assert len(dialect.requests) == 1
    finally:
        await pool.aclose()
