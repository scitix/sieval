"""Pattern: ``sieval run`` launches an engine only for samples left to infer.

A served model goes through the real session, runner and connection factory.
Only the engine is a stub: ``launch_model`` returns the handles a launch of the
translated commands would, and an in-process transport answers at their
endpoint. GPU detection is faked.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import orjson
import pytest
import yaml
from openai import AsyncOpenAI

from sieval.cli.run import _run_all
from sieval.core.models import connection_factory
from sieval.core.models.request_log import RecordingOpenAIClient
from sieval.core.tasks import Task
from sieval.infer.backends.translator import BackendCommand
from sieval.infer.config import InferHandle
from sieval.infer.deployer import command_endpoint
from sieval.infer.introspect import GPUInfo

H100 = GPUInfo(model="NVIDIA H100 80GB HBM3", count=8, memory_mib=81559)
QUESTIONS = ["Q1", "Q2", "Q3"]


class EchoTask(Task):
    """Asks each question once; right when the reply names the question."""

    model_type = "chat"

    def __init__(self, dataset, model, name=None, strict=False):
        super().__init__(dataset=dataset, model=model, name=name)
        self._strict = strict

    async def preprocess(self, raw, ctx):
        return raw["question"]

    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

    async def postprocess(self, inf, ctx):
        return inf.texts[0].strip()

    async def feedback(self, post, ctx):
        expected = f"A{ctx.raw_sample['question'][1:]}"
        return True, {"correct": not self._strict and post == expected}

    async def report(self, finals, fails):
        correct = sum(1 for f in finals if f.feedback_result["correct"])
        return {"correct": correct, "total": len(finals) + len(fails)}


def _sse_answer(answer: str) -> httpx.Response:
    base = {"id": "c1", "object": "chat.completion.chunk", "created": 0}
    events = [
        {
            **base,
            "model": "Qwen3-4B",
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
            "model": "Qwen3-4B",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        },
    ]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return httpx.Response(
        200, content=body.encode(), headers={"content-type": "text/event-stream"}
    )


class Engine:
    """The stub engine: launches by returning handles, answers over HTTP."""

    def __init__(self) -> None:
        self.launches: list[str] = []
        self.requests: list[str] = []

    async def launch(
        self, model_name: str, commands: list[BackendCommand], **_: Any
    ) -> tuple[list[InferHandle], None]:
        self.launches.append(model_name)
        handles = [
            InferHandle(
                backend=cmd.backend,
                handle_id="4242",
                endpoint=command_endpoint(cmd),
                metadata={"role": cmd.role},
            )
            for cmd in commands
        ]
        return handles, None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        question = json.loads(request.content)["messages"][-1]["content"]
        self.requests.append(question)
        return _sse_answer(f"A{question[1:]}")


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Engine]:
    stub = Engine()
    real_openai = AsyncOpenAI

    def openai_client(**kwargs: Any) -> AsyncOpenAI:
        kwargs["http_client"] = RecordingOpenAIClient(
            timeout=kwargs["timeout"], transport=httpx.MockTransport(stub)
        )
        return real_openai(**kwargs)

    monkeypatch.setattr(connection_factory, "AsyncOpenAI", openai_client)
    monkeypatch.setattr("sieval.cli.infer.lifecycle.HANDLE_DIR", tmp_path / "handles")
    with (
        patch("sieval.cli.run.launch_model", new=stub.launch),
        patch("sieval.cli.run.cleanup_model", new=AsyncMock()),
    ):
        yield stub


def _gpu(gpu: GPUInfo | None):
    """This node's GPUs, as ``nvidia-smi`` would report them."""
    return patch(
        "sieval.infer.introspect.detect_local_gpu", new=AsyncMock(return_value=gpu)
    )


def _write(tmp_path: Path, questions: list[str], strict: bool = False) -> Path:
    checkpoint = tmp_path / "Qwen3-4B"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "vocab_size": 151936,
                "hidden_size": 2560,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "intermediate_size": 9216,
                "torch_dtype": "bfloat16",
            }
        )
    )
    config = tmp_path / "eval.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "result_dir": str(tmp_path / "results"),
                "models": {"subject": {"path": str(checkpoint)}},
                "datasets": {
                    "ds": {
                        "class": "tests.conftest.MockDataset",
                        "args": {"samples": [{"question": q} for q in questions]},
                    }
                },
                "tasks": {
                    "echo": {
                        "class": "tests.integration.test_lazy_launch.EchoTask",
                        "dataset": "ds",
                        "model": "subject",
                        "args": {"strict": strict},
                        "runner_config": {
                            "show_progress": False,
                            "dump_progress": False,
                        },
                    }
                },
            }
        )
    )
    return config


def _meta(tmp_path: Path) -> dict:
    return orjson.loads((tmp_path / "results" / "echo" / "meta.json").read_bytes())


async def _run(config: Path, **kwargs: Any) -> dict:
    # `resolve_infer_config`'s path, through auto_resolve_plan's detection.
    with patch(
        "sieval.infer.topology.resolver.detect_local_gpu",
        new=AsyncMock(return_value=H100),
    ):
        return await _run_all(config, **kwargs)


class TestLazyLaunch:
    @pytest.mark.anyio
    async def test_a_resume_with_nothing_to_infer_starts_no_engine(
        self, tmp_path: Path, engine: Engine
    ) -> None:
        config = _write(tmp_path, QUESTIONS[:2])
        with _gpu(H100):
            first = await _run(config)
        assert engine.launches == ["subject"]
        assert sorted(engine.requests) == ["Q1", "Q2"]
        identity = _meta(tmp_path)["identity"]

        # Every sample is reused, so this resume runs on a node with no GPU.
        engine.launches.clear()
        engine.requests.clear()
        with _gpu(None):
            second = await _run(config, resume=True)

        assert engine.launches == []
        assert engine.requests == []
        assert second == first
        meta = _meta(tmp_path)
        assert meta["identity"]["inference"] == identity["inference"]
        assert [run["start"] for run in meta["runs"]] == ["fresh", "resume"]
        assert meta["runs"][-1]["reran"] == []
        assert not (tmp_path / "results" / "logs" / "subject").exists()

    @pytest.mark.anyio
    async def test_a_regrade_starts_no_engine(
        self, tmp_path: Path, engine: Engine
    ) -> None:
        with _gpu(H100):
            await _run(_write(tmp_path, QUESTIONS[:2]))
        identity = _meta(tmp_path)["identity"]
        engine.launches.clear()
        engine.requests.clear()

        # Another grader: both samples regrade on their recorded inference.
        with _gpu(None):
            report = await _run(
                _write(tmp_path, QUESTIONS[:2], strict=True), resume=True
            )

        assert engine.launches == []
        assert engine.requests == []
        assert report["echo"]["correct"] == 0
        meta = _meta(tmp_path)
        assert meta["identity"]["inference"] == identity["inference"]
        assert meta["identity"]["feedback"] != identity["feedback"]
        # A task argument is read by every grading stage.
        assert meta["runs"][-1]["reran"] == ["postprocess", "feedback", "report"]

    @pytest.mark.anyio
    async def test_a_resume_with_samples_to_infer_starts_the_engine(
        self, tmp_path: Path, engine: Engine
    ) -> None:
        with _gpu(H100):
            await _run(_write(tmp_path, QUESTIONS[:2]))
            first = _meta(tmp_path)["identity"]["inference"]
            engine.launches.clear()
            engine.requests.clear()
            report = await _run(_write(tmp_path, QUESTIONS), resume=True)

        assert engine.launches == ["subject"]
        assert engine.requests == ["Q3"]
        assert (report["echo"]["correct"], report["echo"]["total"]) == (3, 3)
        inference = _meta(tmp_path)["identity"]["inference"]
        assert inference["digest"] == first["digest"]

        # What a fresh run of all three sends, digest for digest.
        fresh = tmp_path / "fresh"
        fresh.mkdir()
        with _gpu(H100):
            await _run(_write(fresh, QUESTIONS))
        assert _meta(fresh)["identity"]["inference"] == inference

    @pytest.mark.anyio
    async def test_samples_to_infer_on_a_node_without_gpus_are_refused_first(
        self, tmp_path: Path, engine: Engine
    ) -> None:
        with _gpu(H100):
            await _run(_write(tmp_path, QUESTIONS[:2]))
        before = (tmp_path / "results" / "echo" / "meta.json").read_bytes()
        engine.launches.clear()

        with _gpu(None), pytest.raises(RuntimeError, match="no GPU on this node"):
            await _run(_write(tmp_path, QUESTIONS), resume=True)

        assert engine.launches == []
        assert (tmp_path / "results" / "echo" / "meta.json").read_bytes() == before

    @pytest.mark.anyio
    async def test_a_fresh_run_is_not_planned(
        self, tmp_path: Path, engine: Engine
    ) -> None:
        planned = AsyncMock()
        with _gpu(H100), patch("sieval.cli.run._plan_tasks", new=planned):
            await _run(_write(tmp_path, QUESTIONS[:1]))
        planned.assert_not_awaited()
        assert engine.launches == ["subject"]

    @pytest.mark.anyio
    async def test_only_the_model_with_samples_to_infer_starts(
        self, tmp_path: Path, engine: Engine
    ) -> None:
        config = _write(tmp_path, QUESTIONS[:2])
        cfg = yaml.safe_load(config.read_text())
        cfg["models"]["other"] = {"path": cfg["models"]["subject"]["path"]}
        cfg["tasks"]["echo_other"] = {**cfg["tasks"]["echo"], "model": "other"}
        config.write_text(yaml.safe_dump(cfg))
        with _gpu(H100):
            await _run(config)
        assert sorted(engine.launches) == ["other", "subject"]
        engine.launches.clear()
        engine.requests.clear()

        # A third question for `other`'s task only.
        datasets = cfg["datasets"]
        datasets["ds3"] = {
            **datasets["ds"],
            "args": {"samples": [{"question": q} for q in QUESTIONS]},
        }
        cfg["tasks"]["echo_other"]["dataset"] = "ds3"
        config.write_text(yaml.safe_dump(cfg))
        with _gpu(H100):
            reports = await _run(config, resume=True)

        assert engine.launches == ["other"]
        assert engine.requests == ["Q3"]
        assert reports["echo"]["total"] == 2
        assert reports["echo_other"]["total"] == 3
