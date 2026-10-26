"""Pattern: a served model whose tasks score input launches with headroom.

``sieval run`` serves a base checkpoint to a task that scores its prompts
through SGLang's native ``/generate``. Only the engine is a stub: launches
return handles and an in-process transport answers ``/generate``. The
headroom changes the launch only, so a directory recorded before it resumes.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import AsyncMock, patch

import anyio
import httpx
import orjson
import pytest
import yaml
from openai import AsyncOpenAI
from typer.testing import CliRunner

import sieval.cli.serving
from sieval.cli.main import app
from sieval.cli.run import _run_all
from sieval.core.models import sglang_gen_model
from sieval.core.models.request_log import RecordingOpenAIClient
from sieval.core.models.requirements import TaskRequirements
from sieval.core.tasks import Task
from sieval.core.tasks.report_metrics import report_metrics
from sieval.infer.backends.translator import BackendCommand
from sieval.infer.config import InferHandle
from sieval.infer.deployer import command_endpoint
from sieval.infer.introspect import GPUInfo

H100 = GPUInfo(model="NVIDIA H100 80GB HBM3", count=8, memory_mib=81559)
H100_TEXT = "NVIDIA H100 80GB HBM3:8:81559"
QUESTIONS = ["Q1", "Q2", "Q3"]


class ScoreTask(Task):
    """Scores each question's own tokens, as a perplexity task does."""

    model_type = "gen"
    requires: ClassVar[TaskRequirements] = TaskRequirements(input_scoring=True)
    report_metrics = report_metrics("scored", "total")

    async def preprocess(self, raw, ctx):
        return raw["question"]

    async def infer(self, pre, ctx):
        return await self.model.alogprobs(pre, echo=True, logprobs=0)

    async def postprocess(self, inf, ctx):
        return sum(lp for lp in inf.logprobs or () if lp is not None)

    async def feedback(self, post, ctx):
        return True, {"scored": post < 0}

    async def report(self, finals, fails):
        scored = sum(1 for f in finals if f.feedback_result["scored"])
        return {"scored": scored, "total": len(finals) + len(fails)}


class Engine:
    """The stub engine: launches by returning handles, scores over HTTP."""

    def __init__(self) -> None:
        self.argv: list[list[str]] = []
        self.prompts: list[str] = []

    async def launch(
        self, model_name: str, commands: list[BackendCommand], **_: Any
    ) -> tuple[list[InferHandle], None]:
        self.argv.extend(list(cmd.cli_args) for cmd in commands)
        return [
            InferHandle(
                backend=cmd.backend,
                handle_id="4242",
                endpoint=command_endpoint(cmd),
                metadata={"role": cmd.role},
            )
            for cmd in commands
        ], None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/generate"
        prompt = json.loads(request.content)["text"]
        self.prompts.append(prompt)
        meta = {
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "cached_tokens": 0,
            "finish_reason": {"type": "length"},
            "input_token_logprobs": [[None, 1, prompt], [-0.5, 2, "!"]],
            "output_token_logprobs": [[-0.1, 3, "?"]],
        }
        return httpx.Response(200, json={"text": "?", "meta_info": meta})


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Engine]:
    stub = Engine()
    real_openai = AsyncOpenAI

    def openai_client(**kwargs: Any) -> AsyncOpenAI:
        kwargs["http_client"] = RecordingOpenAIClient(
            timeout=kwargs["timeout"], transport=httpx.MockTransport(stub)
        )
        return real_openai(**kwargs)

    # `sglang_legacy` serves through SglangGenModel, which builds its own.
    monkeypatch.setattr(sglang_gen_model, "AsyncOpenAI", openai_client)
    monkeypatch.setattr("sieval.cli.infer.lifecycle.HANDLE_DIR", tmp_path / "handles")
    monkeypatch.setenv("SIEVAL_TARGET_GPU", H100_TEXT)
    gpu = AsyncMock(return_value=H100)
    with (
        patch("sieval.cli.run.launch_model", new=stub.launch),
        patch("sieval.cli.run.cleanup_model", new=AsyncMock()),
        patch("sieval.infer.introspect.detect_local_gpu", new=gpu),
        patch("sieval.infer.topology.resolver.detect_local_gpu", new=gpu),
    ):
        yield stub


def _write(tmp_path: Path, questions: list[str]) -> Path:
    checkpoint = tmp_path / "Qwen3-4B-Base"
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
                "models": {"subject": {"path": str(checkpoint), "type": "gen"}},
                "datasets": {
                    "ds": {
                        "class": "tests.conftest.MockDataset",
                        "args": {"samples": [{"question": q} for q in questions]},
                    }
                },
                "tasks": {
                    "score": {
                        "class": "tests.integration.test_input_scoring_headroom."
                        "ScoreTask",
                        "dataset": "ds",
                        "model": "subject",
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
    return orjson.loads((tmp_path / "results" / "score" / "meta.json").read_bytes())


def _flag(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


def _engine_params(serving: dict) -> dict:
    return serving["recipe"]["assignments"][0]["engine_params"]


def _no_headroom(plan, tasks, **_: Any):
    """The serving plan of a release before the headroom."""
    return plan, sieval.cli.serving.ScoringHeadroom(tasks, "unchanged")


def _run(config: Path, **kwargs: Any) -> dict:
    return anyio.run(lambda: _run_all(config, **kwargs))


class TestInputScoringHeadroom:
    def test_the_dry_run_reports_the_launch_the_run_records(
        self, tmp_path: Path, engine: Engine
    ) -> None:
        config = _write(tmp_path, QUESTIONS)
        result = CliRunner().invoke(
            app, ["run", str(config), "--dry-run", "-o", "json"]
        )
        assert result.exit_code == 0, result.stdout
        data = json.loads(result.stdout)["data"]
        spec = data["serving"]["default"]["engines"]["subject"]
        headroom = spec["headroom"]
        assert headroom["tasks"] == ["score"]
        # The qwen3-4b H100 recipe launches at 0.90 and no prefill cap.
        assert headroom["params"]["mem_fraction_static"]["recipe"] == 0.90
        assert headroom["params"]["chunked_prefill_size"]["recipe"] is None
        cap = headroom["prefill_tokens"]
        argv = spec["launch"][0]["argv"]
        assert _flag(argv, "--chunked-prefill-size") == str(cap)
        assert float(_flag(argv, "--mem-fraction-static") or 0) < 0.90
        assert "input scoring by score" in headroom["reason"]

        _run(config)
        assert engine.argv == [argv]
        plans = yaml.safe_load((tmp_path / "results" / "infer_plans.yaml").read_text())
        assert plans["models"]["subject"] == data["models"]["subject"]["plan"]
        # What the headroom started from, for a resume to reuse.
        recipe = plans["recipes"]["subject"]["assignments"][0]["engine_params"]
        assert recipe["mem_fraction_static"] == 0.90

    def test_a_directory_recorded_before_the_headroom_resumes(
        self, tmp_path: Path, engine: Engine
    ) -> None:
        with patch.object(
            sieval.cli.serving, "reserve_scoring_headroom", new=_no_headroom
        ):
            _run(_write(tmp_path, QUESTIONS[:2]))
        before = _meta(tmp_path)
        assert engine.argv and "--chunked-prefill-size" not in engine.argv[0]
        engine.argv.clear()
        engine.prompts.clear()

        report = _run(_write(tmp_path, QUESTIONS), resume=True)

        assert engine.prompts == ["Q3"]
        assert "--chunked-prefill-size" in engine.argv[0]
        assert (report["score"]["scored"], report["score"]["total"]) == (3, 3)
        meta = _meta(tmp_path)
        assert (
            meta["identity"]["inference"]["digest"]
            == (before["identity"]["inference"]["digest"])
        )
        assert meta["runs"][-1]["resume"]["kept"] == 2
        first, second = (_engine_params(run["serving"]) for run in meta["runs"])
        assert first["mem_fraction_static"] == 0.90
        assert "chunked_prefill_size" not in first
        assert second["mem_fraction_static"] < 0.90
        assert second["chunked_prefill_size"] > 0

    def test_a_resume_reapplies_the_headroom_to_the_recorded_recipe(
        self, tmp_path: Path, engine: Engine
    ) -> None:
        """Under ``SIEVAL_TARGET_GPU`` a resume reuses the recorded plan; it
        starts from ``recipes``, so the headroom is not taken twice."""
        _run(_write(tmp_path, QUESTIONS[:2]))
        first = engine.argv[0]
        engine.argv.clear()

        _run(_write(tmp_path, QUESTIONS), resume=True)

        assert engine.argv == [first]
