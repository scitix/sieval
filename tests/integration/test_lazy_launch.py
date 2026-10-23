"""Pattern: ``sieval run`` launches an engine only for samples left to infer.

A served model goes through the real session, runner and connection factory.
Only the engine is a stub: ``launch_model`` returns the handles a launch of the
translated commands would, and an in-process transport answers at their
endpoint. GPU detection is faked.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import importlib
import json
import sys
import textwrap
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


def _sync_run(config: Path, **kwargs: Any) -> dict:
    """``_run`` from a test that is not async, as the CLI is not."""
    import anyio

    return anyio.run(lambda: _run(config, **kwargs))


def _dry_run(config: Path, *args: str) -> tuple[int, dict]:
    from typer.testing import CliRunner

    from sieval.cli.main import app

    result = CliRunner().invoke(
        app, ["run", str(config), "--dry-run", *args, "-o", "json"]
    )
    return result.exit_code, json.loads(result.stdout)


def _resume_plan(**counts: int) -> dict[str, int]:
    """A task's ``resume_plan``: every outcome, zero unless given."""
    keys = ("kept", "rereported", "regraded", "repostprocessed", "restarted")
    return {**dict.fromkeys((*keys, "dropped"), 0), **counts}


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


class TestDryRunReportsWhatARunWouldDo:
    """``sieval run --dry-run -o json`` on a CPU node, with declared GPUs."""

    H100_TEXT = "NVIDIA H100 80GB HBM3:8:81559"

    def test_a_resume_reports_its_plan_and_reads_dirs_read_only(
        self, tmp_path: Path, engine: Engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with _gpu(H100):
            _sync_run(_write(tmp_path, QUESTIONS[:2]))
        recorded = _tree(tmp_path / "results")
        engine.launches.clear()
        config = _write(tmp_path, QUESTIONS)
        monkeypatch.setenv("SIEVAL_TARGET_GPU", self.H100_TEXT)
        cpu = AsyncMock(return_value=None)

        with (
            _gpu(None),
            patch("sieval.infer.topology.resolver.detect_local_gpu", new=cpu),
        ):
            code, out = _dry_run(config, "--resume")

        cpu.assert_not_awaited()  # nvidia-smi is never asked
        assert code == 0, out
        data = out["data"]
        assert {c["name"] for c in data["checks"]} >= {"schema", "capability_reconcile"}
        assert data["needs_inference"] is True
        assert data["gpu"] == self.H100_TEXT
        model = data["models"]["subject"]
        assert model["needs_inference"] is True
        assert model["plan"]["backend"] == "sglang"
        checkpoint = str(tmp_path / "Qwen3-4B")
        assert model["launch"] == [
            {
                "role": "full",
                "argv": [
                    "sglang",
                    "serve",
                    "--model-path",
                    checkpoint,
                    *model["launch"][0]["argv"][4:],
                ],
            }
        ]
        assert "--port" in model["launch"][0]["argv"]
        assert data["tasks"]["echo"] == {
            "model": "subject",
            "start": "resume",
            "recorded": str(tmp_path / "results" / "echo"),
            "samples": 3,
            "resume_plan": _resume_plan(kept=2),
            "stages": data["tasks"]["echo"]["stages"],
            "new": 1,
            "to_infer": 1,
            "to_grade": 1,
            "needs_inference": True,
        }
        stages = data["tasks"]["echo"]["stages"]
        assert stages["identity_schema"] == 3
        assert stages["rerun_from"] is None
        # A third sample changes the set's data identity; each recorded sample
        # still matches its own digest, so both are kept.
        assert stages["stages"]["data"] == "differs"
        # Preprocess is code: the same task compares the same.
        assert stages["stages"]["preprocess"] == "same"
        assert stages["legacy_preprocess"] is False
        assert stages["stages"]["inference"] == "same"
        assert engine.launches == []
        assert _tree(tmp_path / "results") == recorded

        # The plan the dry-run reports is the one the run then records.
        with _gpu(H100):
            _sync_run(config, resume=True)
        plans = yaml.safe_load((tmp_path / "results" / "infer_plans.yaml").read_text())
        assert plans["models"]["subject"] == model["plan"]

    def test_nothing_to_infer(
        self, tmp_path: Path, engine: Engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with _gpu(H100):
            _sync_run(_write(tmp_path, QUESTIONS[:2]))
        monkeypatch.setenv("SIEVAL_TARGET_GPU", self.H100_TEXT)
        # A task argument: both samples rerun from postprocess.
        config = _write(tmp_path, QUESTIONS[:2], strict=True)

        with _gpu(None):
            code, out = _dry_run(config, "--resume")

        assert code == 0, out
        data = out["data"]
        assert data["needs_inference"] is False
        assert data["models"]["subject"]["needs_inference"] is False
        echo = data["tasks"]["echo"]
        assert echo["resume_plan"] == _resume_plan(repostprocessed=2)
        assert echo["stages"]["rerun_from"] == "postprocess"
        assert echo["needs_inference"] is False
        assert (echo["to_infer"], echo["to_grade"]) == (0, 2)

    def test_resume_from_another_dir_copies_nothing(
        self, tmp_path: Path, engine: Engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with _gpu(H100):
            _sync_run(_write(tmp_path, QUESTIONS))
        recorded = tmp_path / "results"
        before = _tree(recorded)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        config = _write(elsewhere, QUESTIONS)
        monkeypatch.setenv("SIEVAL_TARGET_GPU", self.H100_TEXT)

        with _gpu(None):
            code, out = _dry_run(config, "--resume", str(recorded))

        assert code == 0, out
        echo = out["data"]["tasks"]["echo"]
        assert (echo["start"], echo["recorded"]) == ("resume", str(recorded / "echo"))
        assert out["data"]["needs_inference"] is False
        assert _tree(recorded) == before
        assert not (elsewhere / "results").exists()

    def test_a_refused_resume_fails_the_dry_run(
        self, tmp_path: Path, engine: Engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with _gpu(H100):
            _sync_run(_write(tmp_path, QUESTIONS[:2]))
        cfg = yaml.safe_load(_write(tmp_path, QUESTIONS[:2]).read_text())
        cfg["tasks"]["echo"]["infer_args"] = {"temperature": 0.9}  # another inference
        config = tmp_path / "eval.yaml"
        config.write_text(yaml.safe_dump(cfg))
        monkeypatch.setenv("SIEVAL_TARGET_GPU", self.H100_TEXT)

        with _gpu(None):
            code, out = _dry_run(config, "--resume")

        assert code == 1
        data = out["data"]
        assert "inference" in data["tasks"]["echo"]["error"]
        assert [c["name"] for c in data["checks"] if not c["ok"]] == ["resume_plan"]

    def test_a_fresh_run_infers_everything(
        self, tmp_path: Path, engine: Engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SIEVAL_TARGET_GPU", self.H100_TEXT)
        with _gpu(None):
            code, out = _dry_run(_write(tmp_path, QUESTIONS))
        assert code == 0, out
        data = out["data"]
        assert data["needs_inference"] is True
        assert data["models"]["subject"]["needs_inference"] is True
        # No recorded samples, so no resume plan.
        assert data["tasks"]["echo"] == {
            "model": "subject",
            "start": "fresh",
            "recorded": None,
            "samples": 3,
            "resume_plan": None,
            "stages": None,
            "new": 3,
            "to_infer": 3,
            "to_grade": 3,
            "needs_inference": True,
        }
        assert not (tmp_path / "results").exists()

    def test_a_finished_run_is_reused(
        self, tmp_path: Path, engine: Engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with _gpu(H100):
            _sync_run(_write(tmp_path, QUESTIONS))
        monkeypatch.setenv("SIEVAL_TARGET_GPU", self.H100_TEXT)

        with _gpu(None):
            code, out = _dry_run(_write(tmp_path, QUESTIONS), "--resume")

        assert code == 0, out
        echo = out["data"]["tasks"]["echo"]
        assert echo["start"] == "reuse"
        assert echo["resume_plan"] == _resume_plan(kept=3)
        assert out["data"]["needs_inference"] is False


def _patch_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``EchoTask.report`` from a package on disk, as a plugin patch does."""
    patches = tmp_path / "patches"
    (patches / "echo_report_patch").mkdir(parents=True, exist_ok=True)
    (patches / "echo_report_patch" / "__init__.py").write_text(
        textwrap.dedent(
            """
            async def report(self, finals, fails):
                correct = sum(1 for f in finals if f.feedback_result["correct"])
                return {"correct": correct, "total": len(finals) + len(fails),
                        "patched": True}
            """
        )
    )
    monkeypatch.syspath_prepend(str(patches))
    monkeypatch.delitem(sys.modules, "echo_report_patch", raising=False)
    patch_module = importlib.import_module("echo_report_patch")
    monkeypatch.setattr(EchoTask, "report", patch_module.report)


class TestDryRunStagePlan:
    """The dry-run's per-task stage plan, and whether the run then infers."""

    H100_TEXT = TestDryRunReportsWhatARunWouldDo.H100_TEXT

    def test_a_report_only_change_rereports_and_infers_nothing(
        self, tmp_path: Path, engine: Engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with _gpu(H100):
            _sync_run(_write(tmp_path, QUESTIONS[:2]))
        engine.launches.clear()
        engine.requests.clear()
        _patch_report(tmp_path, monkeypatch)
        monkeypatch.setenv("SIEVAL_TARGET_GPU", self.H100_TEXT)
        config = _write(tmp_path, QUESTIONS[:2])

        with _gpu(None):
            code, out = _dry_run(config, "--resume")

        assert code == 0, out
        data = out["data"]
        echo = data["tasks"]["echo"]
        assert echo["start"] == "reuse"
        assert echo["resume_plan"] == _resume_plan(kept=2, rereported=2)
        assert (echo["to_infer"], echo["needs_inference"]) == (0, False)
        assert echo["stages"]["rerun_from"] == "report"
        assert echo["stages"]["stages"]["report"] == "differs"
        assert echo["stages"]["stages"]["feedback"] == "same"
        assert data["needs_inference"] is False

        # The run re-reports on a node with no GPU: no engine, no request.
        with _gpu(None):
            report = _sync_run(config, resume=True)
        assert engine.launches == [] and engine.requests == []
        assert report["echo"]["patched"] is True
        assert _meta(tmp_path)["runs"][-1]["resume"] == _resume_plan(
            kept=2, rereported=2
        )

    @pytest.mark.parametrize(
        ("change", "rerun_from", "outcomes"),
        [
            ("none", None, {"kept": 2}),
            ("report", "report", {"kept": 2, "rereported": 2}),
            ("grader", "postprocess", {"repostprocessed": 2}),
            ("new_sample", None, {"kept": 2}),
            ("fewer_samples", None, {"kept": 1, "dropped": 1}),
        ],
    )
    def test_needs_inference_agrees_with_what_the_run_launches(
        self,
        tmp_path: Path,
        engine: Engine,
        monkeypatch: pytest.MonkeyPatch,
        change: str,
        rerun_from: str | None,
        outcomes: dict[str, int],
    ) -> None:
        with _gpu(H100):
            _sync_run(_write(tmp_path, QUESTIONS[:2]))
        engine.launches.clear()
        engine.requests.clear()
        questions = {"new_sample": QUESTIONS, "fewer_samples": QUESTIONS[:1]}.get(
            change, QUESTIONS[:2]
        )
        if change == "report":
            _patch_report(tmp_path, monkeypatch)
        config = _write(tmp_path, questions, strict=change == "grader")

        monkeypatch.setenv("SIEVAL_TARGET_GPU", self.H100_TEXT)
        with _gpu(None):
            code, out = _dry_run(config, "--resume")
        monkeypatch.delenv("SIEVAL_TARGET_GPU")
        assert code == 0, out
        data = out["data"]
        echo = data["tasks"]["echo"]
        assert echo["resume_plan"] == _resume_plan(**outcomes)
        assert echo["stages"]["rerun_from"] == rerun_from

        with _gpu(H100):
            _sync_run(config, resume=True)

        assert data["needs_inference"] is (engine.launches == ["subject"])
        assert data["needs_inference"] is bool(engine.launches)
        assert echo["needs_inference"] is bool(engine.requests)
        assert len(engine.requests) == echo["to_infer"]
        assert _meta(tmp_path)["runs"][-1]["resume"] == echo["resume_plan"]
