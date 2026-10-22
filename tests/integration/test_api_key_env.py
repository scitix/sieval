"""
Pattern: a model's credential read from an environment variable.

A YAML run whose candidate and judge name their keys with ``api_key_env``
sends each key on the wire, never records it, and resumes after the keys
rotate without rerunning inference or regrading.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from loguru import logger
from openai import AsyncOpenAI

from sieval.cli.leaderboard.session import arun_session
from sieval.core.models import Model, connection_factory
from sieval.core.models.request_log import RecordingOpenAIClient
from sieval.core.tasks import InputKind, Task, TaskRequirements

SAMPLES = [
    {"question": "Q1", "answer": "A1"},
    {"question": "Q2", "answer": "A2"},
]
CANDIDATE_HOST = "candidate.example"
JUDGE_HOST = "judge.example"


class JudgedTask(Task):
    """Candidate answers, then a ``grader`` judge rules on each answer."""

    model_type = "chat"

    def __init__(self, dataset, model, name=None, grader=None, models_by_role=None):
        super().__init__(dataset=dataset, model=model, name=name)
        self._grader = self._resolve_role_model(
            "grader", grader, models_by_role, build=self._no_inline_build
        )

    @staticmethod
    def _no_inline_build(grader: object) -> Model:
        raise ValueError("JudgedTask takes its grader from models_by_role")

    @classmethod
    def model_requirements_for(cls, context):
        candidate = super().model_requirements_for(context)
        grader = cls._bind_role_requirement(
            context, "grader", TaskRequirements(input=InputKind.CHAT)
        )
        return candidate + grader

    def judge_prompts(self):
        return {"grader": "Is {answer} right?"}

    async def preprocess(self, raw, ctx):
        return raw["question"]

    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

    async def postprocess(self, inf, ctx):
        return inf.texts[0].strip()

    async def feedback(self, post, ctx):
        verdict = await self._grader.agenerate(f"Is {post} right?")
        return True, {"correct": "yes" in verdict.texts[0]}

    async def report(self, finals, fails):
        correct = sum(1 for f in finals if f.feedback_result["correct"])
        return {"accuracy": correct / len(SAMPLES), "total": len(finals)}


def _sse(events: list[Mapping[str, Any]], *, done: bool) -> bytes:
    lines = [f"data: {json.dumps(event)}\n\n" for event in events]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _openai_chat(answer: str) -> httpx.Response:
    base = {"id": "c1", "object": "chat.completion.chunk", "created": 0}
    body = _sse(
        [
            {
                **base,
                "model": "org/candidate",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": answer},
                        "finish_reason": None,
                    }
                ],
            },
            {
                **base,
                "model": "org/candidate",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                },
            },
        ],
        done=True,
    )
    return httpx.Response(
        200, content=body, headers={"content-type": "text/event-stream"}
    )


def _google_verdict() -> httpx.Response:
    body = _sse(
        [
            {
                "candidates": [
                    {
                        "index": 0,
                        "content": {"role": "model", "parts": [{"text": "yes"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 4,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 5,
                },
                "modelVersion": "gemini-judge",
            }
        ],
        done=False,
    )
    return httpx.Response(
        200, content=body, headers={"content-type": "text/event-stream"}
    )


class _Wire:
    """Answers both hosts and records each request's credential header."""

    def __init__(self) -> None:
        self.candidate_keys: list[str] = []
        self.judge_keys: list[str] = []
        #: Answers a request before the hosts do, when it returns a response.
        self.before: Callable[[httpx.Request], httpx.Response | None] | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.before is not None and (early := self.before(request)) is not None:
            return early
        if request.url.host == CANDIDATE_HOST:
            self.candidate_keys.append(request.headers["authorization"])
            question = json.loads(request.content)["messages"][-1]["content"]
            return _openai_chat({"Q1": "A1", "Q2": "A2"}[question])
        assert request.url.host == JUDGE_HOST
        self.judge_keys.append(request.headers["x-goog-api-key"])
        return _google_verdict()


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Wire]:
    """Route both connection families to one in-process handler."""

    handler = _Wire()
    real_openai = AsyncOpenAI

    def openai_client(**kwargs: Any) -> AsyncOpenAI:
        # The factory's own recording client, on an in-process transport.
        kwargs["http_client"] = RecordingOpenAIClient(
            timeout=kwargs["timeout"], transport=httpx.MockTransport(handler)
        )
        return real_openai(**kwargs)

    monkeypatch.setattr(connection_factory, "AsyncOpenAI", openai_client)
    monkeypatch.setattr(
        connection_factory.httpx,
        "AsyncHTTPTransport",
        lambda **_: httpx.MockTransport(handler),
    )
    yield handler


@pytest.fixture
def captured_logs() -> Iterator[list[str]]:
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(str(message)), level="TRACE")
    yield lines
    logger.remove(sink)


def _write_config(
    tmp_path: Path,
    *,
    candidate_credential: str = "api_key_env: CANDIDATE_KEY",
    judge_credential: str = "api_key_env: JUDGE_KEY",
) -> Path:
    config = tmp_path / "eval.yaml"
    config.write_text(
        f"""\
result_dir: "{tmp_path / "results"}"
models:
  candidate:
    name: org/candidate
    api_base: https://{CANDIDATE_HOST}/v1
    {candidate_credential}
datasets:
  ds:
    class: tests.conftest.MockDataset
    args:
      samples: {json.dumps(SAMPLES)}
tasks:
  judged:
    class: tests.integration.test_api_key_env.JudgedTask
    dataset: ds
    model: candidate
    args:
      grader:
        model: gemini-judge
        dialect: google_genai
        api_base: https://{JUDGE_HOST}/
        {judge_credential}
        temperature: 0
    runner_config:
      show_progress: false
      detect_anomalies: false
      profile_io: false
      profile_stages: false
      profile_usage: false
      dump_progress: false
""",
        encoding="utf-8",
    )
    return config


def _files_containing(root: Path, needle: str) -> list[str]:
    encoded = needle.encode()
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and encoded in path.read_bytes()
    )


class TestApiKeyEnvRun:
    @pytest.mark.anyio
    async def test_keys_reach_the_wire_never_the_records_and_rotate_on_resume(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        wire: _Wire,
        captured_logs: list[str],
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        config = _write_config(tmp_path)
        results = tmp_path / "results"

        monkeypatch.setenv("CANDIDATE_KEY", "SENTINEL-CANDIDATE-A")
        monkeypatch.setenv("JUDGE_KEY", "SENTINEL-JUDGE-A")
        first = await arun_session(config)

        assert first["judged"]["accuracy"] == 1.0
        assert first["judged"]["total"] == 2
        assert set(wire.candidate_keys) == {"Bearer SENTINEL-CANDIDATE-A"}
        assert set(wire.judge_keys) == {"SENTINEL-JUDGE-A"}
        meta_path = results / "judged" / "meta.json"
        first_identity = json.loads(meta_path.read_text())["identity"]

        wire.candidate_keys.clear()
        wire.judge_keys.clear()
        monkeypatch.setenv("CANDIDATE_KEY", "SENTINEL-CANDIDATE-B")
        monkeypatch.setenv("JUDGE_KEY", "SENTINEL-JUDGE-B")
        second = await arun_session(config, resume=True)

        assert second["judged"]["accuracy"] == first["judged"]["accuracy"]
        # Nothing reran: no inference and no regrade under the rotated keys.
        assert wire.candidate_keys == []
        assert wire.judge_keys == []
        meta = json.loads(meta_path.read_text())
        assert meta["identity"] == first_identity
        assert [run["start"] for run in meta["runs"]] == ["fresh", "resume"]
        assert meta["runs"][-1]["reran"] == []
        assert "config" not in meta["runs"][-1]

        # The variable names are recorded; the values are not, anywhere.
        effective = (results / "effective_config.yaml").read_text()
        assert "api_key_env: CANDIDATE_KEY" in effective
        assert "api_key_env: JUDGE_KEY" in effective
        out, err = capfd.readouterr()
        streams = "\n".join([*captured_logs, out, err])
        for value in (
            "SENTINEL-CANDIDATE-A",
            "SENTINEL-CANDIDATE-B",
            "SENTINEL-JUDGE-A",
            "SENTINEL-JUDGE-B",
        ):
            assert _files_containing(results, value) == []
            assert value not in streams
        assert "SENTINEL" not in json.dumps(meta)

    @pytest.mark.anyio
    async def test_switching_from_a_literal_key_resumes_without_rerun(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        wire: _Wire,
    ) -> None:
        results = tmp_path / "results"
        config = _write_config(
            tmp_path,
            candidate_credential="api_key: SENTINEL-CANDIDATE-A",
            judge_credential="api_key: SENTINEL-JUDGE-A",
        )
        await arun_session(config)
        meta_path = results / "judged" / "meta.json"
        first_identity = json.loads(meta_path.read_text())["identity"]

        wire.candidate_keys.clear()
        wire.judge_keys.clear()
        monkeypatch.setenv("CANDIDATE_KEY", "SENTINEL-CANDIDATE-B")
        monkeypatch.setenv("JUDGE_KEY", "SENTINEL-JUDGE-B")
        config = _write_config(tmp_path)
        await arun_session(config, resume=True)

        assert wire.candidate_keys == []
        assert wire.judge_keys == []
        meta = json.loads(meta_path.read_text())
        assert meta["identity"] == first_identity
        assert meta["runs"][-1]["reran"] == []
        # The resume records that the credential moved, never the old key.
        changes = meta["runs"][-1]["config"]
        assert "models.candidate.api_key: <redacted> → None" in changes
        assert "models.candidate.api_key_env: None → 'CANDIDATE_KEY'" in changes
        assert "tasks.judged.args.grader.api_key: <redacted> → None" in changes
        assert "SENTINEL" not in json.dumps(meta)

    @pytest.mark.anyio
    async def test_literal_keys_are_redacted_from_audit_files(
        self,
        tmp_path: Path,
        wire: _Wire,
        captured_logs: list[str],
    ) -> None:
        results = tmp_path / "results"
        config = _write_config(
            tmp_path,
            candidate_credential="api_key: SENTINEL-CANDIDATE-LITERAL",
            judge_credential="api_key: SENTINEL-JUDGE-LITERAL",
        )
        await arun_session(config)
        assert set(wire.judge_keys) == {"SENTINEL-JUDGE-LITERAL"}

        effective = results / "effective_config.yaml"
        assert "SENTINEL" not in effective.read_text()
        persisted = yaml.safe_load(effective.read_text())
        assert persisted["models"]["candidate"]["api_key"] == "<redacted>"
        grader = persisted["tasks"]["judged"]["args"]["grader"]
        assert grader["api_key"] == "<redacted>"
        assert _files_containing(results, "SENTINEL") == []
        assert not any("SENTINEL" in line for line in captured_logs)

        # A resume compares the redacted record with the live config: no
        # rerun, no regrade, and the recorded file stays as it was.
        meta_path = results / "judged" / "meta.json"
        first_identity = json.loads(meta_path.read_text())["identity"]
        recorded = effective.read_text()
        wire.candidate_keys.clear()
        wire.judge_keys.clear()
        await arun_session(config, resume=True)

        meta = json.loads(meta_path.read_text())
        assert meta["identity"] == first_identity
        assert meta["runs"][-1]["reran"] == []
        assert "config" not in meta["runs"][-1]
        assert wire.candidate_keys == wire.judge_keys == []
        assert effective.read_text() == recorded

        # Rerunning the persisted file is refused on the redacted key, before
        # any request: "<redacted>" is never sent as a credential.
        with pytest.raises(ValueError, match="persisted with its key removed"):
            await arun_session(effective, result_dir=str(tmp_path / "rerun"))
        assert wire.candidate_keys == wire.judge_keys == []

    @pytest.mark.anyio
    async def test_unset_variable_fails_before_any_request(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        wire: _Wire,
    ) -> None:
        config = _write_config(tmp_path)
        monkeypatch.setenv("CANDIDATE_KEY", "SENTINEL-CANDIDATE-A")
        monkeypatch.delenv("JUDGE_KEY", raising=False)

        with pytest.raises(ValueError, match="JUDGE_KEY, which is unset"):
            await arun_session(config)

        assert wire.candidate_keys == []
        assert wire.judge_keys == []
