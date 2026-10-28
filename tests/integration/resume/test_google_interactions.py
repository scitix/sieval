"""Native Interactions request records survive disk-backed resume.

AI-Generated Code - GPT-6 (OpenAI)
"""

import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import yaml

from sieval.cli.session import EvalSession
from sieval.core.models import Model, ModelOutput
from sieval.core.models.ir import ChatInput, ChatMessage, TextPart
from sieval.core.models.requirements import InputKind, TaskRequirements
from sieval.core.runners import TaskRunner
from sieval.core.tasks.consts import TaskStage
from sieval.core.tasks.context import TaskContext
from sieval.core.tasks.loader import TaskLoader
from sieval.core.tasks.run_meta import (
    aggregate_digest,
    call_key,
    request_definition,
    request_digest,
)
from tests.conftest import MockDataset, make_config

from .conftest import ResumeTask

FIXTURES = (
    Path(__file__).parents[2] / "unit/core/models/dialects/fixtures/google_interactions"
)


class InteractionsResumeTask(ResumeTask):
    requires = TaskRequirements(input=InputKind.CHAT)

    async def infer(self, pre: str, ctx: TaskContext) -> ModelOutput:
        return await self.model.agenerate(
            ChatInput((ChatMessage("user", (TextPart(pre),)),))
        )


@asynccontextmanager
async def _bound_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    stream: bool,
) -> AsyncIterator[Model]:
    monkeypatch.setenv("SIEVAL_INTERACTIONS_TEST_KEY", "offline-key")
    monkeypatch.setattr(
        "sieval.core.models.connection_factory.httpx.AsyncHTTPTransport",
        lambda **_: httpx.MockTransport(handler),
    )
    config_path = tmp_path / "interactions.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "candidate": {
                        "name": "gemini-test-alias",
                        "dialect": "google_interactions",
                        "api_base": "https://gateway.example/google/v1beta",
                        "api_key_env": "SIEVAL_INTERACTIONS_TEST_KEY",
                        "client_timeout": {"connect": 45, "read": 1800},
                        "args": {"temperature": 0.3, "stream": stream},
                    }
                },
                "tasks": {
                    "interactions": {
                        "class": "fake.InteractionsResumeTask",
                        "dataset": {"class": "fake.Dataset"},
                        "model": "candidate",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    session = EvalSession(config_path)
    try:
        with patch(
            "sieval.cli.session.resolve_task_class",
            return_value=InteractionsResumeTask,
        ):
            session._setup_prelaunch_reconciliation()
            session._setup_postlaunch_reconciliation()
            session._setup_models()
        yield session.models["candidate"]
    finally:
        await session._close_owned_model_resources()


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("record_each_stage", [False, True])
async def test_request_identity_survives_disk_reload_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    record_each_stage: bool,
) -> None:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if stream:
            return httpx.Response(
                200,
                content=(FIXTURES / "stream_text.sse").read_bytes(),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json=json.loads((FIXTURES / "completed_text.json").read_text()),
        )

    dataset = MockDataset([{"question": "ping", "answer": "391" if stream else "pong"}])
    async with _bound_model(tmp_path, monkeypatch, handler, stream=stream) as model:
        task = InteractionsResumeTask(dataset, model, name="interactions_resume")
        runner = TaskRunner(
            task, make_config(tmp_path, record_each_stage=record_each_stage)
        )
        report = await runner.arun()
        assert report["accuracy"] == 1.0
        assert len(sent) == 1
        assert sent[0].url.path == "/google/v1beta/interactions"
        assert sent[0].headers["x-goog-api-key"] == "offline-key"
        assert sent[0].extensions["timeout"] == {
            "connect": 45.0,
            "read": 1800.0,
            "write": 600.0,
            "pool": 600.0,
        }

        loader = TaskLoader(task=task, root_dir=runner.root_dir)
        contexts = await loader.load_initial_state()
        await loader.hydrate(
            contexts,
            set(),
            include_stages={TaskStage.FINAL},
            record_each_stage=record_each_stage,
        )
        assert len(contexts) == 1
        context = next(iter(contexts.values()))
        output = context.infer_result
        assert isinstance(output, ModelOutput)
        assert output.request_params is not None
        assert output.request_params["envelope"] == "http"
        definition = request_definition(output.request_params)
        wire_body = json.loads(sent[0].content)
        assert wire_body["store"] is False
        assert "previous_interaction_id" not in wire_body
        assert output.request_params["body"]["store"] is False
        assert definition == {
            "body": {
                key: value
                for key, value in wire_body.items()
                if key not in {"model", "stream", "input"}
            },
            "stream": stream,
        }
        assert "client_timeout" not in model.meta()["default_params"]
        calls = [
            call
            for entry in context.stage_meta[TaskStage.INFERRED.value]
            for call in entry["model_calls"]
        ]
        sent_digest = request_digest(calls, call_key(model.meta()))
        assert sent_digest is not None
        original_meta = json.loads((runner.root_dir / "meta.json").read_text())
        assert original_meta["identity"]["inference"]["requests"] == aggregate_digest(
            [(context.sample_id, context.repeat_index, sent_digest)]
        )

        resumed = TaskRunner(
            InteractionsResumeTask(dataset, model, name="interactions_resume"),
            make_config(tmp_path, resume=True, record_each_stage=record_each_stage),
        )
        assert await resumed.arun() == report
        assert len(sent) == 1
        resumed_meta = json.loads((resumed.root_dir / "meta.json").read_text())
        assert resumed_meta["identity"] == original_meta["identity"]


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
async def test_provider_error_is_persisted_in_failed_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    reason = "Thinking level is not supported for this model"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": reason}})

    async with _bound_model(tmp_path, monkeypatch, handler, stream=stream) as model:
        task = InteractionsResumeTask(
            MockDataset([{"question": "ping", "answer": "pong"}]),
            model,
            name="interactions_failure",
        )
        runner = TaskRunner(task, make_config(tmp_path, max_retries=0))
        report = await runner.arun()
        assert report["failed"] == 1
        loader = TaskLoader(task=task, root_dir=runner.root_dir)
        contexts = await loader.load_initial_state()
        await loader.hydrate(contexts, set(), include_stages={TaskStage.FAILED})
        assert len(contexts) == 1
        context = next(iter(contexts.values()))
        assert context.error_msg is not None
        assert f"Google: {reason}" in context.error_msg
