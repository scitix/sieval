"""Pattern: ``sieval run`` with model lifecycles and a launched judge.

The subject serves ``until: inference`` and the judge ``from: inference``: one
run starts the subject, stops it once every sample has answered, then starts
the judge on the same GPUs. The real session, runner and connection factory
run; only the engines are stubs (``launch_model`` returns handles, an
in-process transport answers by served model name) and GPU detection is faked.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import httpx
import orjson
import pytest
import yaml
from openai import AsyncOpenAI

from sieval.cli.lifecycle import LifecycleError, LifecycleFitError
from sieval.cli.run import _run_all, _run_dry_run
from sieval.core.models import Model, connection_factory
from sieval.core.models.request_log import RecordingOpenAIClient
from sieval.core.tasks import (
    InputKind,
    RequirementContext,
    Task,
    TaskModelRequirement,
    TaskRequirements,
)
from sieval.core.tasks.report_metrics import report_metrics
from sieval.infer.backends.translator import BackendCommand
from sieval.infer.config import InferHandle
from sieval.infer.deployer import command_endpoint
from sieval.infer.introspect import GPUInfo

H100 = GPUInfo(model="NVIDIA H100 80GB HBM3", count=8, memory_mib=81559)
QUESTIONS = ["Q1", "Q2", "Q3"]
_THIS = "tests.integration.test_staged_run"


class JudgedTask(Task):
    """Asks each question; a judge decides whether the reply is right."""

    model_type = "chat"
    report_metrics = report_metrics("correct", "total")

    @classmethod
    def model_requirements_for(
        cls, context: RequirementContext
    ) -> tuple[TaskModelRequirement, ...]:
        grader = cls._bind_role_requirement(
            context, "grader", TaskRequirements(input=InputKind.CHAT)
        )
        return super().model_requirements_for(context) + grader

    def __init__(
        self,
        dataset,
        model,
        name=None,
        grader: object = None,
        models_by_role: dict[str, Model] | None = None,
    ):
        super().__init__(dataset=dataset, model=model, name=name)

        def refuse(_: object) -> Model:
            raise ValueError("JudgedTask needs a grader")

        self._grader = self._resolve_role_model(
            "grader", grader, models_by_role, build=refuse
        )

    async def preprocess(self, raw, ctx):
        return raw["question"]

    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre)

    async def postprocess(self, inf, ctx):
        return inf.texts[0].strip()

    async def feedback(self, post, ctx):
        verdict = await self._grader.agenerate(f"JUDGE {post}")
        return True, {"correct": verdict.texts[0].strip() == "yes"}

    async def report(self, finals, fails):
        correct = sum(1 for f in finals if f.feedback_result["correct"])
        return {"correct": correct, "total": len(finals) + len(fails)}


def _sse(model: str, answer: str) -> httpx.Response:
    base = {"id": "c1", "object": "chat.completion.chunk", "created": 0}
    events = [
        {
            **base,
            "model": model,
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
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        },
    ]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return httpx.Response(
        200, content=body.encode(), headers={"content-type": "text/event-stream"}
    )


class Engines:
    """Stub engines: a launch returns handles; requests answer by model name.

    ``events`` is the order engines start and stop, and requests reach them.
    A request to an engine that is not up fails, as a connection would.
    """

    def __init__(self) -> None:
        self.events: list[str] = []
        self.commands: dict[str, list[BackendCommand]] = {}
        self.up: set[str] = set()

    async def launch(
        self, model_name: str, commands: list[BackendCommand], **_: Any
    ) -> tuple[list[InferHandle], None]:
        self.events.append(f"start {model_name}")
        self.commands[model_name] = commands
        self.up.add(model_name)
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

    async def cleanup(self, model_name: str, *_: Any, **__: Any) -> None:
        if model_name in self.up:
            self.events.append(f"stop {model_name}")
            self.up.discard(model_name)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        model = body["model"]
        engine = "judge" if model == "judge-model" else "subject"
        if engine not in self.up:
            raise httpx.ConnectError(f"{engine} is not running")
        content = body["messages"][-1]["content"]
        self.events.append(f"ask {engine}")
        if engine == "judge":
            return _sse(model, "yes" if content.startswith("JUDGE A") else "no")
        return _sse(model, f"A{content[1:]}")

    def asked(self, engine: str) -> int:
        return self.events.count(f"ask {engine}")


@pytest.fixture
def engines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Engines]:
    stub = Engines()
    real_openai = AsyncOpenAI

    def openai_client(**kwargs: Any) -> AsyncOpenAI:
        kwargs["http_client"] = RecordingOpenAIClient(
            timeout=kwargs["timeout"], transport=httpx.MockTransport(stub)
        )
        return real_openai(**kwargs)

    monkeypatch.setattr(connection_factory, "AsyncOpenAI", openai_client)
    monkeypatch.setattr("sieval.cli.infer.lifecycle.HANDLE_DIR", tmp_path / "handles")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with (
        patch("sieval.cli.run.launch_model", new=stub.launch),
        patch("sieval.cli.run.cleanup_model", new=stub.cleanup),
        patch(
            "sieval.infer.introspect.detect_local_gpu",
            new=AsyncMock(return_value=H100),
        ),
        patch(
            "sieval.infer.topology.resolver.detect_local_gpu",
            new=AsyncMock(return_value=H100),
        ),
        # The GPUs the plan fits engines on, as `nvidia-smi` reports them.
        patch("sieval.cli.run.detect_local_gpu", new=AsyncMock(return_value=H100)),
    ):
        yield stub


def _checkpoint(root: Path, name: str) -> str:
    path = root / name / "Qwen3-4B"
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
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
    return str(path)


def _write(
    tmp_path: Path,
    *,
    subject: dict | None = None,
    judge: dict | None = None,
    runner: dict | None = None,
    questions: list[str] = QUESTIONS,
    extra_tasks: dict | None = None,
) -> Path:
    config = {
        "result_dir": str(tmp_path / "results"),
        "runner_config": {"show_progress": False, "dump_progress": False}
        | (runner or {}),
        "models": {
            "subject": {
                "name": "subject-model",
                "path": _checkpoint(tmp_path, "subject"),
                "weights": "sha256:subject",
            }
            | (subject or {}),
            "judge": {
                "name": "judge-model",
                "path": _checkpoint(tmp_path, "judge"),
                "weights": "sha256:judge",
            }
            | (judge or {}),
        },
        "datasets": {
            "ds": {
                "class": "tests.conftest.MockDataset",
                "args": {"samples": [{"question": q} for q in questions]},
            }
        },
        "tasks": {
            "judged": {
                "class": f"{_THIS}.JudgedTask",
                "dataset": "ds",
                "model": "subject",
                "args": {"grader": "judge"},
            },
            **(extra_tasks or {}),
        },
    }
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


SUBJECT_UNTIL: dict[str, Any] = {"lifecycle": {"until": "inference"}}
JUDGE_FROM: dict[str, Any] = {"lifecycle": {"from": "inference"}}


def _staged(tmp_path: Path, **kwargs: Any) -> Path:
    """The config with subject `until: inference` and judge `from: inference`."""
    return _write(tmp_path, subject=SUBJECT_UNTIL, judge=JUDGE_FROM, **kwargs)


async def _run(config: Path, **kwargs: Any) -> dict[str, Any]:
    return cast(dict[str, Any], await _run_all(config, **kwargs))


async def _dry(config: Path, **kwargs: Any) -> dict[str, Any]:
    return cast(dict[str, Any], await _run_dry_run(config, **kwargs))


def _halves(config: Path, keep_lifecycles: bool = False) -> Path:
    """*config* with each model pinned to 4 GPUs; without its lifecycles both
    are alive for the whole run, side by side."""
    cfg = yaml.safe_load(config.read_text())
    for name in ("subject", "judge"):
        model = cfg["models"][name]
        if not keep_lifecycles:
            model.pop("lifecycle", None)
        model["infer"] = {
            "backend": "sglang",
            "checkpoint": model.pop("path"),
            "overrides": {"dp_size": 4},
        }
    config.write_text(yaml.safe_dump(cfg))
    return config


def _meta(tmp_path: Path, task: str = "judged") -> dict:
    return orjson.loads((tmp_path / "results" / task / "meta.json").read_bytes())


class TestStagedRun:
    @pytest.mark.anyio
    async def test_the_judge_takes_the_subjects_gpus_after_inference(
        self, tmp_path: Path, engines: Engines
    ) -> None:
        reports = await _run(_staged(tmp_path))

        assert reports["judged"]["correct"] == 3
        events = engines.events
        # The subject answers every question, stops, and only then the judge
        # starts: never both up at once.
        assert events[0] == "start subject"
        stop = events.index("stop subject")
        assert events.index("start judge") == stop + 1
        assert events[1:stop] == ["ask subject"] * 3
        assert events[stop + 2 :] == ["ask judge"] * 3 + ["stop judge"]
        # Sequential lifecycles share every GPU and the default port.
        for name in ("subject", "judge"):
            (command,) = engines.commands[name]
            assert "CUDA_VISIBLE_DEVICES" not in command.env
            assert command.cli_args[command.cli_args.index("--port") + 1] == "30000"

        meta = _meta(tmp_path)
        serving = meta["runs"][-1]["serving"]
        assert serving["lifecycles"] == {
            "judge": {"from": "inference"},
            "subject": {"until": "inference"},
        }
        (judge,) = meta["identity"]["feedback"]["judges"]
        assert (judge["role"], judge["weights"]) == ("grader", "sha256:judge")

    @pytest.mark.anyio
    async def test_a_lifecycle_changes_no_identity(
        self, tmp_path: Path, engines: Engines
    ) -> None:
        staged, plain = tmp_path / "staged", tmp_path / "plain"
        staged.mkdir()
        plain.mkdir()
        await _run(_halves(_staged(staged), keep_lifecycles=True))
        await _run(_halves(_write(plain)))
        assert _meta(staged)["identity"] == _meta(plain)["identity"]
        assert "lifecycles" not in _meta(plain)["runs"][-1]["serving"]

        # Removing the lifecycles on resume reruns nothing.
        engines.events.clear()
        await _run(_halves(_write(staged)), resume=True)
        assert engines.events == []
        assert _meta(staged)["runs"][-1]["reran"] == []

    @pytest.mark.anyio
    async def test_a_judge_stated_by_weights_regrades_when_they_change(
        self, tmp_path: Path, engines: Engines
    ) -> None:
        await _run(_staged(tmp_path))
        engines.events.clear()
        judge = {**JUDGE_FROM, "weights": "sha256:other-judge"}
        await _run(_write(tmp_path, subject=SUBJECT_UNTIL, judge=judge), resume=True)
        assert engines.events == ["start judge"] + ["ask judge"] * 3 + ["stop judge"]
        assert _meta(tmp_path)["runs"][-1]["reran"] == ["feedback", "report"]


class TestRuleTable:
    @pytest.mark.anyio
    async def test_stop_after_inference_does_not_start_the_judge(
        self, tmp_path: Path, engines: Engines
    ) -> None:
        reports = await _run(_staged(tmp_path, runner={"stop_after": "inference"}))
        assert reports == {"judged": {"stopped_after": "inference"}}
        assert engines.events == ["start subject"] + ["ask subject"] * 3 + [
            "stop subject"
        ]
        assert _meta(tmp_path)["runs"][-1]["stopped_after"] == "inference"

        # The resume continues from postprocess, and starts only the judge.
        engines.events.clear()
        report = await _run(_staged(tmp_path), resume=True)
        assert report["judged"]["correct"] == 3
        assert engines.events == ["start judge"] + ["ask judge"] * 3 + ["stop judge"]
        runs = _meta(tmp_path)["runs"]
        assert runs[-1]["reran"] == ["postprocess", "feedback", "report"]

        fresh = tmp_path / "fresh"
        fresh.mkdir()
        await _run(_staged(fresh))
        assert _meta(tmp_path)["identity"] == _meta(fresh)["identity"]

    @pytest.mark.anyio
    async def test_a_report_only_resume_starts_nothing(
        self, tmp_path: Path, engines: Engines
    ) -> None:
        await _run(_staged(tmp_path))
        engines.events.clear()
        await _run(_staged(tmp_path), resume=True)
        assert engines.events == []

    @pytest.mark.anyio
    async def test_stop_after_before_until_stops_the_model_with_the_run(
        self, tmp_path: Path, engines: Engines
    ) -> None:
        reports = await _run(_staged(tmp_path, runner={"stop_after": "preprocess"}))
        assert reports == {"judged": {"stopped_after": "preprocess"}}
        assert engines.asked("subject") == 0
        assert "start judge" not in engines.events
        assert engines.up == set()

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("lifecycle", "message"),
        [
            ({"until": "infer"}, "until must be one of"),
            ({"from": "feedback", "until": "inference"}, "is not before until"),
            ({"from": "inference", "until": "inference"}, "is not before until"),
            ({"after": "inference"}, "unknown key"),
        ],
    )
    async def test_a_malformed_lifecycle_is_refused_before_launch(
        self, tmp_path: Path, engines: Engines, lifecycle: dict, message: str
    ) -> None:
        with pytest.raises(LifecycleError, match=message):
            await _run(_write(tmp_path, judge={"lifecycle": lifecycle}))
        assert engines.events == []


class TestPlacement:
    @pytest.mark.anyio
    async def test_models_alive_together_get_disjoint_devices_and_ports(
        self, tmp_path: Path, engines: Engines
    ) -> None:
        config = _halves(_write(tmp_path))

        result = await _dry(config)
        (phase,) = result["phases"]
        devices = {name: spec["devices"] for name, spec in phase["models"].items()}
        assert devices == {"judge": [0, 1, 2, 3], "subject": [4, 5, 6, 7]}
        await _run(config)
        ports = {}
        for name, (command,) in engines.commands.items():
            assert command.env["CUDA_VISIBLE_DEVICES"] == ",".join(
                map(str, devices[name])
            )
            ports[name] = command.cli_args[command.cli_args.index("--port") + 1]
            assert f":{ports[name]}/" in (command.health_url or "")
        assert ports["judge"] != ports["subject"]

    @pytest.mark.anyio
    async def test_devices_select_from_an_inherited_visible_device_list(
        self, tmp_path: Path, engines: Engines, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "8,9,10,11,12,13,14,15")
        await _run(_halves(_write(tmp_path)))
        devices = {
            name: command.env["CUDA_VISIBLE_DEVICES"]
            for name, (command,) in engines.commands.items()
        }
        assert devices == {"judge": "8,9,10,11", "subject": "12,13,14,15"}

    @pytest.mark.anyio
    async def test_fewer_visible_devices_than_the_node_has_are_the_limit(
        self, tmp_path: Path, engines: Engines, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
        with pytest.raises(LifecycleFitError, match="the run has 4"):
            await _run(_halves(_write(tmp_path)))

    @pytest.mark.anyio
    async def test_models_alive_together_that_do_not_fit_are_refused_first(
        self, tmp_path: Path, engines: Engines
    ) -> None:
        # Each recipe takes all 8 GPUs; alive together they need 16.
        with pytest.raises(LifecycleFitError, match="need 16 GPU"):
            await _run(_write(tmp_path))
        assert engines.events == []
        assert not (tmp_path / "results" / "judged").exists()


class TestDryRun:
    @pytest.mark.anyio
    async def test_lists_barriers_phases_and_launches(
        self, tmp_path: Path, engines: Engines
    ) -> None:
        result = await _dry(_staged(tmp_path, runner={"stop_after": "feedback"}))
        assert result["n_errors"] == 0
        assert result["barriers"] == [
            {
                "stage": "inference",
                "implied_by": ["judge.lifecycle.from", "subject.lifecycle.until"],
            },
            {"stage": "feedback", "implied_by": ["stop_after"]},
        ]
        assert [list(p["models"]) for p in result["phases"]] == [
            ["subject"],
            ["judge"],
        ]
        assert result["launches"] == ["judge", "subject"]
        assert result["needs_inference"] is True
        assert result["models"]["judge"]["lifecycle"] == {"from": "inference"}

    @pytest.mark.anyio
    async def test_a_resume_with_complete_inference_launches_only_the_judge(
        self, tmp_path: Path, engines: Engines
    ) -> None:
        await _run(_staged(tmp_path, runner={"stop_after": "inference"}))
        result = await _dry(_staged(tmp_path), resume=True)
        assert result["launches"] == ["judge"]
        assert result["needs_inference"] is False

        await _run(_staged(tmp_path), resume=True)
        result = await _dry(_staged(tmp_path), resume=True)
        assert result["launches"] == []

    @pytest.mark.anyio
    async def test_reports_a_lifecycle_refusal_as_a_failed_check(
        self, tmp_path: Path, engines: Engines
    ) -> None:
        result = await _dry(_write(tmp_path))
        (check,) = [c for c in result["checks"] if c["name"] == "lifecycle"]
        assert not check["ok"]
        assert "need 16 GPU" in check["detail"]
        assert result["n_errors"] >= 1

    @pytest.mark.anyio
    async def test_refuses_a_barrier_an_iterating_task_cannot_pass(
        self, tmp_path: Path, engines: Engines
    ) -> None:
        result = await _dry(_staged(tmp_path, runner={"max_iterations": 2}))
        (check,) = [c for c in result["checks"] if c["name"] == "lifecycle"]
        assert "can iterate" in check["detail"]
