"""Pattern: ``sieval run`` fails fast when an engine it launched dies mid-run.

The engine is a real process launched by the real deployer: a stdlib HTTP
server answering OpenAI chat completions, which exits on its own after a set
number of answers, as an engine that ran out of memory would. The session,
runner, connection factory and request path are the real ones. GPU detection
is faked.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
import socket
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import orjson
import pytest
import yaml

from sieval.cli import run as run_module
from sieval.cli.run import ENGINE_LOST_EXIT_CODE, EngineLostError, _run_all
from sieval.core.tasks import Task
from sieval.core.tasks.consts import ENGINE_DIED_REASON, ENGINE_DIED_UNSENT_REASON
from sieval.infer import deployer as deployer_module
from sieval.infer.backends.process import pid_alive
from sieval.infer.backends.translator import BackendCommand
from sieval.infer.introspect import GPUInfo

H100 = GPUInfo(model="NVIDIA H100 80GB HBM3", count=8, memory_mib=81559)
QUESTIONS = [f"Q{i}" for i in range(1, 9)]

# Answers each chat completion with "A<n>" for question "Q<n>". After
# argv[2] answers it writes a last log line, then under argv[3] "exit" exits
# 137, as the kernel's OOM killer leaves an engine; under "hang" it stays alive
# and answers nothing more, health checks included.
_ENGINE = textwrap.dedent(
    """
    import json, os, sys, threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    port, budget, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    answered = 0
    wedged = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if wedged.is_set():
                threading.Event().wait()
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            global answered
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            question = body["messages"][-1]["content"]
            if answered >= budget:
                print("torch.OutOfMemoryError: CUDA out of memory", flush=True)
                if mode == "exit":
                    os._exit(137)
                wedged.set()
                threading.Event().wait()
            answered += 1
            reply = {
                "id": "c1",
                "object": "chat.completion",
                "created": 0,
                "model": "stub",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "A" + question[1:]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3
                },
            }
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    print("engine ready", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    """
)


class EchoTask(Task):
    """Asks each question once; right when the reply names the question."""

    model_type = "chat"

    async def preprocess(self, raw, ctx):
        return raw["question"]

    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre, stream=False)

    async def postprocess(self, inf, ctx):
        return inf.texts[0].strip()

    async def feedback(self, post, ctx):
        expected = f"A{ctx.raw_sample['question'][1:]}"
        return True, {"correct": post == expected}

    async def report(self, finals, fails):
        correct = sum(1 for f in finals if f.feedback_result["correct"])
        return {"correct": correct, "total": len(finals) + len(fails)}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class StubEngine:
    """Replaces the translated launch command with the stub engine's."""

    def __init__(self, tmp_path: Path) -> None:
        self.script = tmp_path / "engine.py"
        self.script.write_text(_ENGINE)
        self.budget = 1000
        self.mode = "exit"
        self.launches = 0

    def translate(self, plan: Any) -> list[BackendCommand]:
        self.launches += 1
        port = _free_port()
        argv = [
            sys.executable,
            str(self.script),
            str(port),
            str(self.budget),
            self.mode,
        ]
        return [
            BackendCommand(
                cli_args=argv,
                backend="sglang",
                health_url=f"http://127.0.0.1:{port}/health",
            )
        ]


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[StubEngine]:
    stub = StubEngine(tmp_path)
    translator = type("T", (), {"translate": staticmethod(stub.translate)})
    monkeypatch.setattr("sieval.cli.infer.lifecycle.HANDLE_DIR", tmp_path / "handles")
    monkeypatch.setattr(deployer_module, "ENGINE_POLL_INTERVAL", 0.05)
    gpu = AsyncMock(return_value=H100)
    with (
        patch.object(run_module, "get_translator", return_value=translator),
        patch("sieval.infer.introspect.detect_local_gpu", new=gpu),
        patch("sieval.infer.topology.resolver.detect_local_gpu", new=gpu),
        # The engine is a few lines of Python: no framework to probe.
        patch(
            "sieval.cli.infer.lifecycle.collect_basic_env",
            new=AsyncMock(return_value=None),
        ),
    ):
        yield stub


def _write(tmp_path: Path, *, concurrency: int = 1, client_retries: int = 5) -> Path:
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
                # Retries against a dead engine are what made the old run slow.
                "models": {
                    "subject": {"path": str(checkpoint), "max_retries": client_retries}
                },
                "datasets": {
                    "ds": {
                        "class": "tests.conftest.MockDataset",
                        "args": {"samples": [{"question": q} for q in QUESTIONS]},
                    }
                },
                "tasks": {
                    "echo": {
                        "class": "tests.integration.test_engine_death.EchoTask",
                        "dataset": "ds",
                        "model": "subject",
                        "runner_config": {
                            "show_progress": False,
                            "dump_progress": False,
                            "concurrency_limit": concurrency,
                            "max_retries": 1,
                        },
                    }
                },
            }
        )
    )
    return config


def _terminal(tmp_path: Path) -> dict[int, dict]:
    """The last terminal record of each sample, by sample id."""
    out: dict[int, dict] = {}
    task_dir = tmp_path / "results" / "echo" / "0"
    for stage in ("failed", "final"):
        for shard in sorted((task_dir / stage).glob("*.jsonl")):
            for line in shard.read_bytes().splitlines():
                record = orjson.loads(line)
                out[record["sample_id"]] = record
    return out


class TestEngineDeath:
    @pytest.mark.anyio
    async def test_remaining_samples_fail_fast_and_a_resume_retries_them(
        self, tmp_path: Path, engine: StubEngine
    ) -> None:
        engine.budget = 3
        config = _write(tmp_path)

        started = time.monotonic()
        with pytest.raises(EngineLostError) as caught:
            await _run_all(config)
        elapsed = time.monotonic() - started

        # Five retries with backoff per sample took over a minute before; the
        # run now ends within a poll of the engine's exit.
        assert elapsed < 10
        error = caught.value
        assert set(error.lost) == {"subject"}
        assert error.lost["subject"].exit_code == 137
        assert "CUDA out of memory" in error.lost["subject"].log
        assert "exited with code 137" in str(error)
        assert error.failed == {"echo": 5}
        assert error.to_json()["outcome"] == "engine_died"
        # The task finished: its report counts the failures.
        report = cast(dict, error.reports["echo"])
        assert (report["correct"], report["total"]) == (3, 8)
        assert report["sieval"]["failed"] == 5
        terminal = _terminal(tmp_path)
        assert len(terminal) == 8
        failed = {sid: r for sid, r in terminal.items() if r["stage"] == "failed"}
        assert len(failed) == 5
        # One request was in flight; the other four were never sent.
        reasons = sorted(r["error_reason"] for r in failed.values())
        assert reasons == [ENGINE_DIED_REASON] + [ENGINE_DIED_UNSENT_REASON] * 4
        assert all("engine subject lost" in r["error_msg"] for r in failed.values())
        # The engine's log stays in the result directory.
        log = tmp_path / "results" / "logs" / "subject" / "full.log"
        assert "CUDA out of memory" in log.read_text()
        assert not list((tmp_path / "handles").glob("run-*/*.json"))

        # The resume launches a fresh engine and retries exactly those five.
        engine.budget = 1000
        report = cast(dict, (await _run_all(config, resume=True))["echo"])

        assert engine.launches == 2
        assert (report["correct"], report["total"]) == (8, 8)
        terminal = _terminal(tmp_path)
        assert all(r["stage"] == "final" for r in terminal.values())
        # Only the sample in flight, which may have brought it down, spent one.
        assert sorted(r.get("retry_count", 0) for r in terminal.values()) == [0] * 7 + [
            1
        ]

    @pytest.mark.anyio
    async def test_an_engine_that_stops_answering_is_lost_and_stopped(
        self, tmp_path: Path, engine: StubEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Alive but wedged: the request in flight would wait out the client
        # timeout. The failed health checks end it instead.
        monkeypatch.setattr(deployer_module, "ENGINE_HEALTH_INTERVAL", 0.1)
        monkeypatch.setattr(deployer_module, "ENGINE_UNHEALTHY_AFTER", 0.5)
        engine.budget, engine.mode = 3, "hang"

        started = time.monotonic()
        with pytest.raises(EngineLostError) as caught:
            await _run_all(_write(tmp_path, concurrency=2))

        assert time.monotonic() - started < 30
        exit = caught.value.lost["subject"]
        assert exit.exit_code is None
        assert exit.reason == "health_check_timeout"
        assert "failed its health check" in str(caught.value)
        assert caught.value.failed == {"echo": 5}
        failed = [r for r in _terminal(tmp_path).values() if r["stage"] == "failed"]
        assert {r["error_reason"] for r in failed} == {
            ENGINE_DIED_REASON,
            ENGINE_DIED_UNSENT_REASON,
        }
        # Stopped like any engine once the run ends: nothing is left running.
        assert not pid_alive(int(exit.pid))

    @pytest.mark.anyio
    async def test_requests_in_flight_that_fail_before_the_watch_sees_it(
        self, tmp_path: Path, engine: StubEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Every remaining request is in flight when the engine exits, and fails
        # on the reset at once, before the next poll at the real interval.
        monkeypatch.setattr(deployer_module, "ENGINE_POLL_INTERVAL", 1.0)
        engine.budget = 3

        with pytest.raises(EngineLostError) as caught:
            await _run_all(_write(tmp_path, concurrency=8, client_retries=0))

        assert caught.value.failed == {"echo": 5}
        failed = [r for r in _terminal(tmp_path).values() if r["stage"] == "failed"]
        assert [r["error_reason"] for r in failed] == [ENGINE_DIED_REASON] * 5

    @pytest.mark.anyio
    async def test_an_engine_that_outlives_its_samples_is_no_loss(
        self, tmp_path: Path, engine: StubEngine
    ) -> None:
        # Every sample answered; the engine is stopped on purpose afterwards,
        # which the watch, already ended, never sees.
        report = cast(dict, (await _run_all(_write(tmp_path)))["echo"])

        assert (report["correct"], report["total"]) == (8, 8)


def test_the_exit_code_is_not_a_refusal_code() -> None:
    from sieval.cli.refusal import EXIT_CODES

    assert ENGINE_LOST_EXIT_CODE not in {0, 1, 2, *EXIT_CODES.values(), 130, 143}
