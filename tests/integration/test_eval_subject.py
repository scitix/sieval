"""Contract tests for external EvalSubject tasks and model-less CLI runs.

AI-Generated Code - GPT-6.1 (OpenAI)
"""

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from sieval.cli import app
from sieval.cli.session import EvalSession
from sieval.cli.validation import run_dry_run, validate_eval_config
from sieval.core.models import (
    EvalSubject,
    Model,
    RecordingAsyncClient,
    ServingFingerprint,
    SubjectMeta,
    subject_call,
)
from sieval.core.runners import StageResumeError
from sieval.core.tasks.run_meta import (
    canonical_json,
    digest,
    inference_identity,
    serving_fingerprint,
)
from sieval.core.tasks.task import Task
from tests.conftest import MockChatModel


class _ReferenceSubject:
    def __init__(self, revision: str):
        self.revision = revision

    def subject_meta(self) -> SubjectMeta:
        return {"model": "reference-service", "revision": self.revision}

    def call_key(self) -> str:
        return f"reference-service:{self.revision}"

    def request_args(self) -> dict[str, Any]:
        return {"protocol": "reference-http-v1"}

    def serving(self) -> ServingFingerprint | None:
        return None


class ReferenceApiTask(Task):
    """A reference plugin that evaluates a service without a Model client."""

    requires_model = False

    def __init__(
        self,
        dataset,
        model: Model | None,
        name: str | None = None,
        revision: str = "v1",
        suffix: str = "",
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._subject = _ReferenceSubject(revision)
        self._revision = revision
        self._suffix = suffix

    @property
    def subject(self) -> EvalSubject:
        return self._subject

    async def preprocess(self, raw, ctx):
        return raw["question"]

    async def infer(self, pre, ctx):
        payload = {"question": pre}

        def respond(request: httpx.Request) -> httpx.Response:
            request.read()
            return httpx.Response(200, json={"answer": "ok"})

        with subject_call(self.subject) as call:
            async with RecordingAsyncClient(
                transport=httpx.MockTransport(respond),
                base_url="https://reference.invalid",
            ) as client:
                response = await client.post("/evaluate", json=payload)
            response.raise_for_status()
            call.record(
                usage={
                    "input_tokens": 2,
                    "output_tokens": 1,
                    "total_tokens": 3,
                },
                request_params={"body": payload},
                response_model=self._revision,
            )
        return response.json()["answer"]

    async def postprocess(self, inf, ctx):
        return f"{inf}{self._suffix}"

    async def feedback(self, post, ctx):
        return True, {"prediction": post}

    async def report(self, finals, fails):
        return {
            "total": len(finals) + len(fails),
            "predictions": [final.postprocess_result for final in finals],
        }


class _NullableRequestTask(ReferenceApiTask):
    async def infer(self, pre, ctx):
        payload = {"question": pre, "filter": None}

        with subject_call(self.subject) as call:
            async with RecordingAsyncClient(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json={"answer": "ok"})
                ),
                base_url="https://reference.invalid",
            ) as client:
                response = await client.post("/evaluate", json=payload)
            response.raise_for_status()
            call.record(request_params=payload)
        return response.json()["answer"]


class _SnapshotRequestTask(ReferenceApiTask):
    def __init__(
        self, dataset, model=None, name=None, revision="v1", suffix="", immutable=False
    ):
        super().__init__(dataset, model, name, revision, suffix)
        self._immutable = immutable
        self.received = []

    async def infer(self, pre, ctx):
        item = {"index": 0, "filter": None}
        body = {"question": pre, "items": [item]}

        def respond(request):
            self.received.append(json.loads(request.read())["items"][0]["index"])
            return httpx.Response(200, json={"answer": "ok"})

        async with RecordingAsyncClient(
            transport=httpx.MockTransport(respond), base_url="https://reference.invalid"
        ) as client:
            for index in (0, 1):
                item["index"] = index
                with subject_call(self.subject) as call:
                    response = await client.post("/evaluate", json=body)
                    response.raise_for_status()
                    params = (
                        MappingProxyType(
                            {
                                "body": MappingProxyType(
                                    {
                                        "question": pre,
                                        "items": [MappingProxyType(item)],
                                    }
                                )
                            }
                        )
                        if self._immutable
                        else {"body": body}
                    )
                    call.record(request_params=params)
        return "ok"


class _MixedReferenceApiTask(ReferenceApiTask):
    requires_model = True


class _MissingSubjectApiTask(ReferenceApiTask):
    subject = Task.subject


@pytest.mark.anyio
@pytest.mark.parametrize("immutable", [False, True])
@pytest.mark.parametrize("record_each_stage", [False, True])
async def test_record_captures_sent_parameters_and_normalizes_json_mappings(
    tmp_path, immutable, record_each_stage
):
    from sieval.core.runners import TaskRunner
    from tests.conftest import MockDataset, make_config

    dataset = MockDataset(samples=[{"question": "q", "answer": "ok"}])
    task = _SnapshotRequestTask(dataset, immutable=immutable)
    await TaskRunner(
        task, make_config(tmp_path, record_each_stage=record_each_stage)
    ).arun()
    assert task.received == [0, 1]
    root = tmp_path / "results"
    stage = "inferred" if record_each_stage else "final"
    stored = json.loads(next((root / "0" / stage).glob("*.jsonl")).read_text())
    meta = (
        stored["meta_last"]
        if record_each_stage
        else stored["stage_meta"]["inferred"][-1]
    )
    assert [
        call["request_params"]["body"]["items"][0]["index"]
        for call in meta["model_calls"]
    ] == [0, 1]
    recorded = json.loads((root / "meta.json").read_text())
    requests = recorded["identity"]["inference"]["requests"]
    resumed_task = _SnapshotRequestTask(dataset, suffix="!", immutable=immutable)
    await TaskRunner(
        resumed_task,
        make_config(tmp_path, resume=True, record_each_stage=record_each_stage),
    ).arun()
    resumed = json.loads((root / "meta.json").read_text())
    assert resumed["identity"]["inference"]["requests"] == requests
    assert resumed_task.received == []
    assert len((root / "requests.jsonl").read_text().splitlines()) == 2


@pytest.mark.anyio
@pytest.mark.parametrize("record_each_stage", [False, True])
async def test_nullable_request_digest_survives_postprocess_resume(
    tmp_path, record_each_stage
):
    from sieval.core.runners import TaskRunner
    from sieval.core.tasks.run_meta import aggregate_digest, request_digest
    from tests.conftest import MockDataset, make_config

    dataset = MockDataset(samples=[{"question": "q", "answer": "ok"}])
    sample_digest = request_digest(
        [
            {
                "model": {"model": "reference-service"},
                "call_key": "reference-service:v1",
                "request_params": {"question": "q", "filter": None},
            }
        ],
        "reference-service:v1",
    )
    expected = aggregate_digest([(0, None, sample_digest)])["digest"]
    for suffix in ("", "!", "!!"):
        runner = TaskRunner(
            _NullableRequestTask(dataset, None, suffix=suffix),
            make_config(
                tmp_path, resume=bool(suffix), record_each_stage=record_each_stage
            ),
        )
        result = await runner.arun()
        assert result["predictions"] == [f"ok{suffix}"]
        meta = json.loads((tmp_path / "results" / "meta.json").read_text())
        assert meta["identity"]["inference"]["requests"]["digest"] == expected
    assert len((tmp_path / "results" / "requests.jsonl").read_text().splitlines()) == 1


def _write_config(
    tmp_path: Path,
    *,
    revision: str = "v1",
    suffix: str = "",
    result_dir: Path | None = None,
) -> Path:
    result_dir = result_dir or tmp_path / "results"
    config = {
        "result_dir": str(result_dir),
        "datasets": {
            "samples": {
                "class": "tests.conftest.MockDataset",
                "args": {
                    "samples": [{"question": "hello", "answer": "ok"}],
                },
            }
        },
        "tasks": {
            "external": {
                "class": "tests.integration.test_eval_subject.ReferenceApiTask",
                "dataset": "samples",
                "args": {"revision": revision, "suffix": suffix},
                "runner_config": {
                    "show_progress": False,
                    "detect_anomalies": False,
                    "profile_io": False,
                    "profile_stages": False,
                    "profile_usage": True,
                    "dump_progress": False,
                },
            }
        },
    }
    path = tmp_path / "eval.yaml"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_model_less_subject_runs_through_cli_and_records_calls(tmp_path: Path) -> None:
    path = _write_config(tmp_path)

    result = CliRunner().invoke(app, ["eval", str(path), "-o", "json"])

    assert result.exit_code == 0, result.stdout
    task_dir = tmp_path / "results" / "external"
    report = json.loads((task_dir / "report.json").read_text(encoding="utf-8"))
    assert report["predictions"] == ["ok"]
    meta = json.loads((task_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["model"] == "reference-service"
    requests = meta["identity"]["inference"]["requests"]
    assert requests["digest"].startswith("sha256:")
    assert (task_dir / "requests.jsonl").is_file()
    [request] = [
        json.loads(line)
        for line in (task_dir / "requests.jsonl").read_text().splitlines()
    ]
    assert request["role"] == "subject"
    assert request["status"] == 200
    assert "serving" not in meta["runs"][-1]
    record = json.loads(next((task_dir / "0" / "inferred").glob("*.jsonl")).read_text())
    [call] = record["meta_last"]["model_calls"]
    assert call["model"]["revision"] == "v1"
    assert call["usage"]["total_tokens"] == 3
    profile = json.loads((task_dir / "profile.json").read_text(encoding="utf-8"))
    assert profile["token_usage"]["inferred"]["input"]["total"] == 2
    assert profile["token_usage"]["inferred"]["output"]["total"] == 1


@pytest.mark.anyio
async def test_subject_change_refuses_resume_at_inference(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    await EvalSession(config_path=str(path)).arun()
    path = _write_config(tmp_path, revision="v2")

    with pytest.raises(ExceptionGroup) as exc_info:
        await EvalSession(config_path=str(path), resume=True).arun()
    pending: list[BaseException] = list(exc_info.value.exceptions)
    while pending:
        error = pending.pop()
        if isinstance(error, BaseExceptionGroup):
            pending.extend(error.exceptions)
        elif isinstance(error, StageResumeError):
            assert "stage `inference` differs" in str(error)
            return
    pytest.fail("resume did not refuse the changed subject at inference")


@pytest.mark.anyio
async def test_postprocess_change_resumes_from_postprocess(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    await EvalSession(config_path=str(path)).arun()
    path = _write_config(tmp_path, suffix="!")

    result = await EvalSession(config_path=str(path), resume=True).arun()

    assert result["external"]["predictions"] == ["ok!"]
    report = json.loads(
        (tmp_path / "results" / "external" / "report.json").read_text(encoding="utf-8")
    )
    assert report["predictions"] == ["ok!"]
    requests = (tmp_path / "results" / "external" / "requests.jsonl").read_text()
    assert len(requests.splitlines()) == 1


def test_validation_and_dry_run_explain_model_less_contract(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    config = json.loads(path.read_text(encoding="utf-8"))

    validated = validate_eval_config(config)
    assert validated.ok
    assert any("EvalSubject" in warning for warning in validated.warnings)

    dry_run = run_dry_run(path)
    checks = {check["name"]: check for check in dry_run["checks"]}
    assert dry_run["n_errors"] == 0
    assert any(
        "requires_model=False" in warning for warning in checks["schema"]["warnings"]
    )
    assert (
        "EvalSubject protocol is task-owned" in checks["capability_reconcile"]["detail"]
    )


@pytest.mark.anyio
async def test_cli_mixed_subject_does_not_inherit_helper_weights_or_context(
    tmp_path: Path,
) -> None:
    path = _write_config(tmp_path)
    config = json.loads(path.read_text())
    config["models"] = {
        "helper": {
            "name": "helper-model",
            "api_base": "https://helper.invalid/v1",
            "api_key": "fake",
            "weights": "helper-w",
            "context_limit": 8192,
        }
    }
    config["tasks"]["external"]["model"] = "helper"
    config["tasks"]["external"]["class"] = (
        "tests.integration.test_eval_subject._MixedReferenceApiTask"
    )
    for resume in (False, True):
        path.write_text(json.dumps(config))
        result = await EvalSession(config_path=str(path), resume=resume).arun()
        assert result["external"]["predictions"] == ["ok"]
        meta = json.loads((tmp_path / "results" / "external" / "meta.json").read_text())
        assert meta["model"] == "reference-service"
        assert "weights" not in meta
        config["models"]["helper"]["context_limit"] = 16384
        config["models"]["helper"]["weights"] = "other-helper-w"
    requests = tmp_path / "results" / "external" / "requests.jsonl"
    assert len(requests.read_text().splitlines()) == 1


def test_fresh_cli_dry_run_rejects_a_missing_subject(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    config = json.loads(path.read_text())
    config["tasks"]["external"]["class"] = (
        "tests.integration.test_eval_subject._MissingSubjectApiTask"
    )
    path.write_text(json.dumps(config))
    result = CliRunner().invoke(app, ["run", str(path), "--dry-run", "-o", "json"])
    assert result.exit_code != 0, result.stdout
    assert json.loads(result.stdout)["data"]["n_errors"] > 0


@pytest.mark.anyio
async def test_model_implements_eval_subject_without_changing_model_identity() -> None:
    # ``canonical_json`` historically accepted extended values in ``extra``.
    # Keep the EvalSubject bridge byte-identical for those models too.
    model = MockChatModel(extra={"tags": {"x"}})
    metadata = model.meta()
    legacy_model_args = {"params": dict(metadata["default_params"])}
    if "extra" in metadata:
        legacy_model_args["extra"] = dict(metadata["extra"])
    plan = model.runtime_plan
    if plan is not None:
        if plan.request_defaults.values:
            legacy_model_args["request_defaults"] = (
                plan.request_defaults.to_json_value()
            )
        legacy_model_args["dialect"] = plan.dialect_id

    legacy_call_key = canonical_json(
        {
            key: value
            for key, value in metadata.items()
            if key not in {"api_base", "api_base_env", "provenance"}
        }
    )
    runner_keys = {
        "deterministic": False,
        "max_iterations": None,
        "max_retries": None,
    }
    legacy_inference_digest = digest(
        {
            "model_args": legacy_model_args,
            "task_args": {},
            "runner": runner_keys,
            "context_limit": None,
        }
    )
    legacy_request_args = dict(legacy_model_args)
    legacy_request_args.pop("extra", None)
    params = legacy_request_args.get("params")
    if isinstance(params, dict):
        legacy_request_args["params"] = {
            key: value
            for key, value in params.items()
            if key not in {"model", "stream", "stream_options", "timeout"}
        }
    legacy_request_key = digest({"model_args": legacy_request_args, "task_args": {}})

    assert isinstance(model, EvalSubject)
    assert model.subject_meta() == metadata
    assert model.call_key() == legacy_call_key
    assert model.request_args() == legacy_model_args
    assert model.serving() == serving_fingerprint(model, None)

    inference = inference_identity(
        model=model,
        task_args={},
        runner_keys=runner_keys,
        context_limit=None,
    )
    assert inference["digest"] == legacy_inference_digest
    assert inference["request_key"] == legacy_request_key
