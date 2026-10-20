"""Tests for `sieval infer start`'s capability-layer selection.

Checkpoint mode is the one entry point with no task context to derive the model
type from, so the operator declares it with ``--model-type``. These tests pin
the translation into the recipe's own vocabulary and the refusal to accept the
flag in YAML mode, where the config already answers the question.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from sieval.cli.infer import infer_app
from sieval.infer.backends.translator import BackendCommand
from sieval.infer.config import InferHandle
from sieval.infer.topology.models import (
    DeploymentPlan,
    DeviceGroup,
    ParallelTopology,
    RoleAssignment,
    WellKnownRole,
)
from sieval.infer.topology.resolver import ResolveResult

runner = CliRunner()


def _failure_text(result) -> str:
    """Everywhere a failed command's message can surface, joined.

    A raised exception and rendered stdout are not interchangeable, and which
    one carries the message is a property of the CLI's error plumbing, not of
    the behavior under test. Today an escaping `ValueError` reaches
    `result.exception`; once command failures are funnelled through `render()`
    (scitix/sieval#85) the same message arrives on stdout with
    `result.exception` set to `SystemExit`. Asserting on both keeps these tests
    about the message, so they do not depend on which change lands first.
    """
    return f"{result.output}\n{result.exception!r}"


def _fake_plan() -> DeploymentPlan:
    return DeploymentPlan(
        checkpoint="/ckpt",
        backend="sglang",
        assignments=(
            RoleAssignment(
                role=WellKnownRole.FULL,
                devices=DeviceGroup(count=1, gpu_model="H100"),
                topology=ParallelTopology(tp=1, dp=1, pp=1),
                engine_params={"context_length": 32768},
            ),
        ),
    )


def _write_checkpoint(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"})
    )
    return path


class TestInferStartModelType:
    """`--model-type` selects the recipe capability layer in checkpoint mode."""

    def _capability_for(self, tmp_path: Path, args: list[str]) -> str:
        """Run `infer start --dry-run` and return the resolved capability."""
        resolve = AsyncMock(return_value=ResolveResult(plan=_fake_plan(), steps=()))
        translator = MagicMock()
        cmd = MagicMock()
        cmd.cli_args = ["sglang", "--model-path", "/ckpt"]
        cmd.health_url = "http://localhost:8000/health"
        cmd.env = {}
        translator.translate.return_value = [cmd]

        with (
            patch("sieval.cli.infer.commands.auto_resolve_plan", new=resolve),
            patch(
                "sieval.cli.infer.commands.get_translator",
                return_value=translator,
            ),
            patch("sieval.cli.infer.commands.validate_plan", return_value=[]),
        ):
            result = runner.invoke(infer_app, args)

        assert result.exit_code == 0, result.output
        resolve.assert_awaited_once()
        call = resolve.await_args
        assert call is not None
        return call.kwargs["capability"]

    def test_default_is_instruct(self, tmp_path: Path):
        ckpt = _write_checkpoint(tmp_path / "Qwen3-4B")
        assert (
            self._capability_for(tmp_path, ["start", str(ckpt), "--dry-run"])
            == "instruct"
        )

    def test_gen_selects_the_base_layer(self, tmp_path: Path):
        """The point of the flag: serve a base checkpoint without instruct params."""
        ckpt = _write_checkpoint(tmp_path / "Qwen3-4B-Base")
        capability = self._capability_for(
            tmp_path,
            ["start", str(ckpt), "--dry-run", "--model-type", "gen"],
        )
        assert capability == "base"

    def test_chat_selects_the_instruct_layer(self, tmp_path: Path):
        ckpt = _write_checkpoint(tmp_path / "Qwen3-4B")
        capability = self._capability_for(
            tmp_path,
            ["start", str(ckpt), "--dry-run", "--model-type", "chat"],
        )
        assert capability == "instruct"

    @pytest.mark.parametrize("bad", ["base", "instruct", "cht"])
    def test_recipe_vocabulary_and_typos_are_rejected(self, tmp_path: Path, bad: str):
        """The flag takes config words; `--model-type base` must not silently pass.

        `base` is the recipe layer's own name for what `gen` selects, so it is
        the likely slip — and accepting it as an unknown default would pick the
        *instruct* layer, the opposite of what was asked for.
        """
        ckpt = _write_checkpoint(tmp_path / "Qwen3-4B")
        result = runner.invoke(
            infer_app,
            ["start", str(ckpt), "--dry-run", "--model-type", bad],
        )
        assert result.exit_code != 0
        assert "expected 'chat' or 'gen'" in _failure_text(result)

    def test_rejected_in_yaml_mode(self, tmp_path: Path):
        """YAML already carries the type; accepting the flag would let them differ."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            yaml.safe_dump(
                {
                    "models": {
                        "m": {
                            "type": "chat",
                            "infer": {"backend": "sglang", "recipe": "qwen3-4b"},
                        }
                    }
                }
            )
        )

        result = runner.invoke(
            infer_app,
            ["start", str(cfg), "--dry-run", "--model-type", "gen"],
        )

        assert result.exit_code != 0
        # Typer renders the message inside a wrapped Rich panel, so match a
        # fragment short enough to survive line breaking.
        assert "applies to checkpoint mode" in _failure_text(result)


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


class TestPerOwnerHandles:
    """`infer list/stop/start` over the per-owner handle layout."""

    @pytest.fixture
    def handle_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        handle_dir = tmp_path / "handles"
        handle_dir.mkdir()
        monkeypatch.setattr("sieval.cli.infer.lifecycle.HANDLE_DIR", handle_dir)
        return handle_dir

    def test_list_shows_run_handles_and_sweeps_dead_runs(self, handle_dir: Path):
        (handle_dir / "m.json").write_text(
            json.dumps({"phase": "pending", "owner_pid": os.getpid()})
        )
        live = handle_dir / f"run-{os.getpid()}"
        live.mkdir()
        (live / "m.json").write_text(
            json.dumps({"phase": "pending", "pid": os.getpid()})
        )
        dead = handle_dir / f"run-{_dead_pid()}"
        dead.mkdir()
        (dead / "m.json").write_text(
            json.dumps({"phase": "pending", "owner_pid": _dead_pid()})
        )

        result = runner.invoke(infer_app, ["list", "-o", "json"])

        assert result.exit_code == 0, result.output
        rows = json.loads(result.stdout)["data"]
        assert [r["model"] for r in rows] == ["m", f"run-{os.getpid()}/m"]
        assert all(r["status"] == "Pending" for r in rows)
        assert not dead.exists()

    def test_list_probes_a_running_old_format_handle(self, handle_dir: Path):
        from sieval.infer.config import InferCondition, InferPhase

        (handle_dir / "m.json").write_text(
            json.dumps(
                {
                    "phase": "running",
                    "backend": "vllm",
                    "endpoint": "http://localhost:8000/v1",
                    "handle_id": "4242",
                    "pid": 1,
                }
            )
        )
        status = AsyncMock(
            return_value=(InferPhase.RUNNING, {"ready": InferCondition(True)})
        )
        with patch("sieval.cli.infer.lifecycle.default_deployer.status", status):
            result = runner.invoke(infer_app, ["list", "-o", "json"])

        assert result.exit_code == 0, result.output
        [row] = json.loads(result.stdout)["data"]
        assert row["status"] == "Ready"
        assert row["engine_pid"] == "4242"
        assert status.await_args is not None
        assert status.await_args.args[0].handle_id == "4242"
        saved = json.loads((handle_dir / "m.json").read_text())
        assert saved["engine_pid"] == "4242"
        assert "handle_id" not in saved

    def test_list_removes_a_stopping_handle_whose_engine_exited(
        self, handle_dir: Path, tmp_path: Path
    ):
        log = tmp_path / "full-x.log"
        log.write_text("x")
        (handle_dir / "m.json").write_text(
            json.dumps(
                {
                    "phase": "stopping",
                    "handle_id": str(_dead_pid()),
                    "keep_logs": 0,
                    "metadata": {"log_file": str(log)},
                }
            )
        )
        result = runner.invoke(infer_app, ["list", "-o", "json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["data"] == []
        assert not (handle_dir / "m.json").exists()
        assert not log.exists()

    def test_stop_with_keep_zero_deletes_the_log(
        self, handle_dir: Path, tmp_path: Path
    ):
        from sieval.infer.config import InferCondition, InferPhase

        log = tmp_path / "full-x.log"
        log.write_text("x")
        (handle_dir / "m.json").write_text(
            json.dumps(
                {
                    "phase": "running",
                    "backend": "sglang",
                    "engine_pid": 4242,
                    "keep_logs": 0,
                    "metadata": {"log_file": str(log)},
                }
            )
        )
        status = AsyncMock(
            return_value=(InferPhase.STOPPED, {"ready": InferCondition(False)})
        )
        with patch("sieval.cli.infer.lifecycle.default_deployer") as deployer:
            deployer.delete = AsyncMock()
            deployer.status = status
            with patch("sieval.cli.infer.commands.default_deployer", deployer):
                result = runner.invoke(infer_app, ["stop", "m", "-o", "json"])

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["data"]["stopped"] is True
        assert deployer.delete.await_args is not None
        assert deployer.delete.await_args.args[0].handle_id == "4242"
        assert not (handle_dir / "m.json").exists()
        assert not log.exists()

    def test_stop_reaches_a_run_handle(self, handle_dir: Path):
        from sieval.infer.config import InferPhase

        run_dir = handle_dir / "run-1"
        run_dir.mkdir()
        (run_dir / "m.json").write_text(
            json.dumps({"phase": "running", "engine_pid": 4242})
        )
        with patch("sieval.cli.infer.lifecycle.default_deployer") as deployer:
            deployer.delete = AsyncMock()
            deployer.status = AsyncMock(return_value=(InferPhase.STOPPED, {}))
            with patch("sieval.cli.infer.commands.default_deployer", deployer):
                result = runner.invoke(infer_app, ["stop", "run-1/m", "-o", "json"])
        assert result.exit_code == 0, result.output
        assert not (run_dir / "m.json").exists()

    @pytest.mark.parametrize(
        ("args", "env", "expected"),
        [
            ([], None, 3),
            ([], "5", 5),
            (["--keep-logs", "0"], "5", 0),
            (["--keep-logs", "-1"], None, -1),
        ],
    )
    def test_start_passes_keep_logs(
        self,
        handle_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        args: list[str],
        env: str | None,
        expected: int,
    ):
        if env is None:
            monkeypatch.delenv("SIEVAL_KEEP_LOGS", raising=False)
        else:
            monkeypatch.setenv("SIEVAL_KEEP_LOGS", env)
        ckpt = _write_checkpoint(tmp_path / "Qwen3-4B")
        resolve = AsyncMock(return_value=ResolveResult(plan=_fake_plan(), steps=()))
        translator = MagicMock()
        translator.translate.return_value = [
            BackendCommand(cli_args=["sglang"], backend="sglang", role="full")
        ]
        handle = InferHandle(backend="sglang", endpoint=None, handle_id="1")
        launch = AsyncMock(return_value=([handle], None))
        with (
            patch("sieval.cli.infer.commands.auto_resolve_plan", new=resolve),
            patch("sieval.cli.infer.commands.get_translator", return_value=translator),
            patch("sieval.cli.infer.commands.validate_plan", return_value=[]),
            patch("sieval.cli.infer.commands.launch_model", new=launch),
        ):
            result = runner.invoke(infer_app, ["start", str(ckpt), "-o", "json", *args])

        assert result.exit_code == 0, result.output
        assert launch.await_args is not None
        kwargs = launch.await_args.kwargs
        assert kwargs["keep_logs"] == expected
        assert kwargs["handle_path"] == handle_dir / "qwen3-4b.json"
        assert json.loads(result.stdout)["data"]["engine_pid"] == "1"

    def test_start_refuses_a_live_handle(self, handle_dir: Path, tmp_path: Path):
        (handle_dir / "qwen3-4b.json").write_text(
            json.dumps({"phase": "running", "handle_id": str(os.getpid())})
        )
        ckpt = _write_checkpoint(tmp_path / "Qwen3-4B")
        resolve = AsyncMock(return_value=ResolveResult(plan=_fake_plan(), steps=()))
        translator = MagicMock()
        translator.translate.return_value = [
            BackendCommand(cli_args=["sglang"], backend="sglang", role="full")
        ]
        launch = AsyncMock()
        with (
            patch("sieval.cli.infer.commands.auto_resolve_plan", new=resolve),
            patch("sieval.cli.infer.commands.get_translator", return_value=translator),
            patch("sieval.cli.infer.commands.validate_plan", return_value=[]),
            patch("sieval.cli.infer.commands.launch_model", new=launch),
        ):
            result = runner.invoke(infer_app, ["start", str(ckpt), "-o", "json"])
        assert result.exit_code == 1
        assert "already running" in json.loads(result.stdout)["error"]
        launch.assert_not_awaited()


class TestInferStopRace:
    """`infer stop` probes outside the lock; the handle may change meanwhile."""

    @pytest.fixture
    def handle_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        d = tmp_path / "handles"
        d.mkdir()
        monkeypatch.setattr("sieval.cli.infer.lifecycle.HANDLE_DIR", d)
        return d

    def test_stop_keeps_a_handle_another_start_reclaimed(self, handle_dir: Path):
        from sieval.infer.config import InferPhase

        path = handle_dir / "m.json"
        path.write_text(json.dumps({"phase": "running", "engine_pid": 4242}))

        async def _reclaimed(_handle):  # noqa: ANN001
            # A new `infer start` took the name while the old engine exited.
            path.write_text(json.dumps({"phase": "running", "engine_pid": 5151}))
            return InferPhase.STOPPED, {}

        with patch("sieval.cli.infer.lifecycle.default_deployer") as deployer:
            deployer.delete = AsyncMock()
            deployer.status = AsyncMock(side_effect=_reclaimed)
            with patch("sieval.cli.infer.commands.default_deployer", deployer):
                result = runner.invoke(infer_app, ["stop", "m", "-o", "json"])

        assert result.exit_code == 0, result.output
        assert json.loads(path.read_text()) == {"phase": "running", "engine_pid": 5151}
