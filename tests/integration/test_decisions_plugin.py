"""The standalone Decisions plugin uses SiEval's public subject boundary.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "plugins" / "decisions" / "src")
)

from sieval_decisions.client import DecisionResponse  # noqa: E402
from sieval_decisions.dataset import DecisionJSONLDataset  # noqa: E402
from sieval_decisions.factory import create_decisions_client  # noqa: E402
from sieval_decisions.task import DecisionJSONLTask  # noqa: E402

from sieval.cli.session import EvalSession  # noqa: E402
from sieval.core.runners import StageResumeError, TaskRunner  # noqa: E402
from sieval.core.tasks.consts import TaskStage  # noqa: E402
from sieval.core.tasks.loader import TaskLoader  # noqa: E402
from tests.conftest import make_config  # noqa: E402


def write_data(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps(
            {
                "input": [
                    {
                        "type": "json",
                        "data": {"message": "Charged twice", "filter": None},
                    }
                ],
                "questions": [
                    {
                        "id": "duplicate",
                        "type": "predicate",
                        "instructions": "Was the customer charged twice?",
                    }
                ],
                "reference": {"duplicate": True},
            }
        )
        + "\n"
    )
    return path


def response_body():
    answer = {"type": "noul", "noul": 0.8, "confidence": 0.8}
    return {
        "model": "pinned",
        "answers": {"duplicate": answer},
        "usage": {"input_tokens": 7, "output_tokens": 0, "total_tokens": 7},
    }


def attach_client(task, provider, transport):
    task._client = create_decisions_client(
        provider,
        "pinned",
        base_url="https://decisions.invalid/v1",
        max_retries=0,
        transport=transport,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("provider", ["openrouter"])
@pytest.mark.parametrize("record_each_stage", [False, True])
async def test_same_task_runs_all_providers_and_resumes_on_recorded_decisions(
    tmp_path, provider, record_each_stage, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    sent = []

    def respond(request):
        sent.append(json.loads(request.read()))
        return httpx.Response(200, json=response_body())

    transport = httpx.MockTransport(respond)
    dataset = DecisionJSONLDataset(str(write_data(tmp_path)))
    first = DecisionJSONLTask(
        dataset,
        None,
        provider=provider,
        model_name="pinned",
        base_url="https://decisions.invalid/v1",
    )
    attach_client(first, provider, transport)
    report = await TaskRunner(
        first, make_config(tmp_path, record_each_stage=record_each_stage)
    ).arun()
    assert report["accuracy"] == 1
    assert report["predicate_brier_score"] == pytest.approx(0.04)
    assert "reference" not in sent[0]
    root = tmp_path / "results"
    meta = json.loads((root / "meta.json").read_text())
    requests = meta["identity"]["inference"]["requests"]
    assert meta["model"] == "pinned"
    assert requests["digest"] is not None
    profile = json.loads((root / "profile.json").read_text())
    assert profile["token_usage"]["inferred"]["input"]["total"] == 7

    loader = TaskLoader(task=first, root_dir=root)
    contexts = await loader.load_initial_state()
    await loader.hydrate(
        contexts,
        set(),
        include_stages=set(TaskStage),
        record_each_stage=record_each_stage,
    )
    assert isinstance(contexts[0].infer_result, DecisionResponse)
    assert contexts[0].infer_result.result.answers[0].probability == 0.8
    assert contexts[0].infer_result.raw.value == response_body()

    # A grading argument changes the verdict without sending a second request.
    second = DecisionJSONLTask(
        dataset,
        None,
        provider=provider,
        model_name="pinned",
        base_url="https://decisions.invalid/v1",
        predicate_threshold=0.9,
    )
    attach_client(second, provider, transport)
    updated = await TaskRunner(
        second, make_config(tmp_path, resume=True, record_each_stage=record_each_stage)
    ).arun()
    assert updated["accuracy"] == 0
    assert len(sent) == 1
    assert (
        json.loads((root / "meta.json").read_text())["identity"]["inference"][
            "requests"
        ]
        == requests
    )

    changed = DecisionJSONLTask(
        dataset,
        None,
        provider=provider,
        model_name="another-model",
        base_url="https://decisions.invalid/v1",
    )
    before = (root / "meta.json").read_bytes()
    with pytest.raises(StageResumeError, match="inference"):
        await TaskRunner(
            changed,
            make_config(tmp_path, resume=True, record_each_stage=record_each_stage),
        ).plan_run()
    assert (root / "meta.json").read_bytes() == before


@pytest.mark.anyio
async def test_plugin_runs_through_cli_configuration_without_models(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")

    def respond(_request):
        return httpx.Response(200, json=response_body())

    from sieval_decisions.client import HTTPDecisionsClient

    original = HTTPDecisionsClient.__init__

    def initialize(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(respond)
        original(self, *args, **kwargs)

    monkeypatch.setattr(HTTPDecisionsClient, "__init__", initialize)
    config = {
        "result_dir": str(tmp_path / "results"),
        "datasets": {
            "questions": {
                "class": "sieval_decisions.dataset.DecisionJSONLDataset",
                "args": {"name_or_path": str(write_data(tmp_path))},
            }
        },
        "tasks": {
            "decisions": {
                "class": "sieval_decisions.task.DecisionJSONLTask",
                "dataset": "questions",
                "args": {"provider": "openrouter", "model_name": "pinned"},
                "runner_config": {
                    "show_progress": False,
                    "detect_anomalies": False,
                    "dump_progress": False,
                },
            }
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(config))
    result = await EvalSession(config_path=str(path)).arun()
    assert result["decisions"]["accuracy"] == 1
    assert (tmp_path / "results" / "decisions" / "requests.jsonl").exists()
