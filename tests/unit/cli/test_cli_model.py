"""Tests for sieval.cli._cli_model — ``--checkpoint`` / ``--api-base``.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from sieval.cli import app
from sieval.cli._cli_model import CliModel, apply_cli_model
from sieval.cli.session import EvalSession
from sieval.cli.validation import run_dry_run

runner = CliRunner()

_TASK_ONLY = (
    "tasks:\n"
    "  eval:\n"
    "    class: tests.unit.core.runners.test_runner.MockTask\n"
    "    dataset:\n"
    "      class: tests.conftest.MockDataset\n"
)


def _task_only(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "benchmark.yaml"
    path.write_text(extra + _TASK_ONLY, encoding="utf-8")
    return path


class TestFromOptions:
    def test_no_flag_is_no_model(self):
        assert CliModel.from_options(checkpoint=None, api_base=None, model="m") is None

    def test_checkpoint_needs_no_model_name(self):
        cli_model = CliModel.from_options(
            checkpoint="/models/sft", api_base=None, model=None
        )
        assert cli_model == CliModel(checkpoint="/models/sft")

    def test_relative_checkpoint_is_made_absolute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.chdir(tmp_path)
        cli_model = CliModel.from_options(
            checkpoint="ckpt/sft", api_base=None, model=None
        )
        assert cli_model == CliModel(checkpoint=str(tmp_path / "ckpt" / "sft"))

    def test_both_flags_are_refused(self):
        with pytest.raises(ValueError, match="--checkpoint and --api-base"):
            CliModel.from_options(checkpoint="/m", api_base="http://h/v1", model="m")

    def test_api_base_without_model_is_refused(self):
        with pytest.raises(ValueError, match="--api-base needs --model"):
            CliModel.from_options(checkpoint=None, api_base="http://h/v1", model=None)

    @pytest.mark.parametrize("flag", ["checkpoint", "api_base"])
    def test_empty_value_is_refused(self, flag: str):
        with pytest.raises(ValueError, match="must not be empty"):
            CliModel(**{flag: ""})

    def test_exactly_one_source(self):
        with pytest.raises(ValueError, match="exactly one"):
            CliModel()


class TestApplyCliModel:
    def test_checkpoint_entry_is_a_bare_path(self):
        cfg: dict = {"tasks": {}}
        apply_cli_model(cfg, CliModel(checkpoint="/models/my-sft/"), None)
        assert cfg["models"] == {"my-sft": {"path": "/models/my-sft/"}}

    def test_checkpoint_entry_is_keyed_by_model_name(self):
        cfg: dict = {}
        apply_cli_model(cfg, CliModel(checkpoint="/models/x"), "org/served")
        assert cfg["models"] == {"served": {"path": "/models/x", "name": "org/served"}}

    def test_api_base_entry_is_keyed_by_model_name(self):
        cfg: dict = {"models": {}}
        apply_cli_model(cfg, CliModel(api_base="http://h/v1"), "Qwen/Qwen3-8B")
        assert cfg["models"] == {
            "Qwen3-8B": {"api_base": "http://h/v1", "name": "Qwen/Qwen3-8B"}
        }

    def test_unnameable_entry_is_refused(self):
        with pytest.raises(ValueError, match="pass --model"):
            CliModel(checkpoint="/").entry(None)

    def test_no_cli_model_leaves_config_alone(self):
        cfg: dict = {"models": {"m": {"name": "x"}}}
        assert apply_cli_model(cfg, None, None) == {"models": {"m": {"name": "x"}}}

    def test_config_with_models_is_refused(self):
        cfg: dict = {"models": {"b": {}, "a": {}}}
        with pytest.raises(ValueError) as exc:
            apply_cli_model(cfg, CliModel(api_base="http://h/v1"), "m")
        assert "--api-base is for a config with no 'models:'" in str(exc.value)
        assert "defines a, b" in str(exc.value)
        assert cfg == {"models": {"b": {}, "a": {}}}


class TestSessionRecordsTheEntry:
    @pytest.mark.anyio
    async def test_persisted_config_reruns_with_no_flags(self, tmp_path: Path):
        cfg_path = _task_only(tmp_path)
        result_dir = tmp_path / "out"
        session = EvalSession(
            cfg_path,
            model_override="served",
            result_dir_override=str(result_dir),
            cli_model=CliModel(api_base="http://h:8000/v1"),
        )
        first_plan = session.prepare_prelaunch().to_json_value()
        await session._persist_effective_config()

        persisted_path = result_dir / "effective_config.yaml"
        persisted = yaml.safe_load(persisted_path.read_text())
        assert persisted["models"] == {
            "served": {"api_base": "http://h:8000/v1", "name": "served"}
        }

        replay = EvalSession(persisted_path, result_dir_override=str(result_dir))
        assert replay.prepare_prelaunch().to_json_value() == first_plan

    def test_task_only_config_without_flags_still_fails(self, tmp_path: Path):
        with pytest.raises(ValueError, match="no models defined in config"):
            EvalSession(_task_only(tmp_path)).prepare_prelaunch()

    def test_config_with_models_refuses_the_flag(self, tmp_path: Path):
        cfg_path = _task_only(tmp_path, "models:\n  m:\n    name: x\n")
        with pytest.raises(ValueError, match="this config defines m"):
            EvalSession(cfg_path, cli_model=CliModel(checkpoint="/models/x"))

    def test_task_naming_another_model_still_fails(self, tmp_path: Path):
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(_TASK_ONLY + "    model: judge\n", encoding="utf-8")
        session = EvalSession(
            cfg_path,
            model_override="m",
            cli_model=CliModel(api_base="http://h/v1"),
        )
        with pytest.raises(ValueError, match="unknown model 'judge'"):
            session.prepare_prelaunch()


class TestDryRun:
    def test_shows_the_entry_and_passes(self, tmp_path: Path):
        result = run_dry_run(
            _task_only(tmp_path),
            model_override="served",
            cli_model=CliModel(api_base="http://h/v1"),
        )
        checks = {check["name"]: check for check in result["checks"]}
        assert checks["cli_model"] == {
            "name": "cli_model",
            "ok": True,
            "plan": {
                "models": {"served": {"api_base": "http://h/v1", "name": "served"}}
            },
        }
        assert checks["capability_reconcile"]["ok"] is True
        assert result["n_errors"] == 0

    def test_refusal_is_a_failed_check(self, tmp_path: Path):
        cfg_path = _task_only(tmp_path, "models:\n  m:\n    name: x\n")
        result = run_dry_run(
            cfg_path, model_override="m", cli_model=CliModel(api_base="http://h/v1")
        )
        assert result["checks"][-1]["name"] == "cli_model"
        assert result["checks"][-1]["ok"] is False
        assert "this config defines m" in result["checks"][-1]["detail"]
        assert result["n_errors"] == 1

    def test_non_mapping_config_fails_the_check(self, tmp_path: Path):
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("- a\n", encoding="utf-8")
        result = run_dry_run(
            cfg_path, model_override="m", cli_model=CliModel(api_base="http://h/v1")
        )
        assert result["checks"][-1] == {
            "name": "cli_model",
            "ok": False,
            "detail": "evaluation config must be a mapping",
        }
        assert result["n_errors"] == 1

    def test_without_flags_the_config_still_fails(self, tmp_path: Path):
        result = run_dry_run(_task_only(tmp_path))
        assert result["n_errors"] == 1
        assert "no models defined in config" in json.dumps(result)


class TestEvalCommand:
    def test_api_base_runs_a_task_only_config(self, tmp_path: Path):
        cfg_path = _task_only(tmp_path)
        with patch(
            "sieval.cli.session.arun_session",
            new=AsyncMock(return_value={}),
        ) as session:
            result = runner.invoke(
                app,
                ["eval", str(cfg_path), "--api-base", "http://h/v1", "-m", "served"],
            )
        assert result.exit_code == 0, result.output
        session.assert_awaited_once()
        assert session.await_args is not None
        assert session.await_args.kwargs["cli_model"] == CliModel(
            api_base="http://h/v1"
        )

    def test_eval_has_no_checkpoint_flag(self, tmp_path: Path):
        result = runner.invoke(
            app, ["eval", str(_task_only(tmp_path)), "--checkpoint", "/m"]
        )
        assert result.exit_code == 2
        assert "No such option" in result.output

    def test_api_base_without_model_fails_before_running(self, tmp_path: Path):
        with patch("sieval.cli.session.arun_session", new=AsyncMock()) as session:
            result = runner.invoke(
                app,
                ["eval", str(_task_only(tmp_path)), "--api-base", "http://h/v1"]
                + ["-o", "json"],
            )
        assert result.exit_code == 1
        assert "--api-base needs --model" in json.loads(result.stdout)["error"]
        session.assert_not_awaited()

    def test_dry_run_shows_the_entry(self, tmp_path: Path):
        result = runner.invoke(
            app,
            ["eval", str(_task_only(tmp_path)), "--dry-run", "-o", "json"]
            + ["--api-base", "http://h/v1", "-m", "served"],
        )
        assert result.exit_code == 0, result.output
        checks = json.loads(result.stdout)["data"]["checks"]
        entry = next(check for check in checks if check["name"] == "cli_model")
        assert entry["plan"] == {
            "models": {"served": {"api_base": "http://h/v1", "name": "served"}}
        }


class TestRunCommand:
    def test_both_flags_fail_before_anything_launches(self, tmp_path: Path):
        with patch("sieval.cli.run._run_all", new=AsyncMock()) as run_all:
            result = runner.invoke(
                app,
                ["run", str(_task_only(tmp_path)), "--checkpoint", "/m"]
                + ["--api-base", "http://h/v1", "-m", "x", "-o", "json"],
            )
        assert result.exit_code == 1
        assert "--checkpoint and --api-base" in json.loads(result.stdout)["error"]
        run_all.assert_not_awaited()

    def test_dry_run_shows_a_refusal(self, tmp_path: Path):
        cfg_path = _task_only(tmp_path, "models:\n  m:\n    name: x\n")
        with patch("sieval.cli.output.log_user") as log:
            result = runner.invoke(
                app, ["run", str(cfg_path), "--checkpoint", "/models/x", "--dry-run"]
            )
        assert result.exit_code == 1
        logged = " ".join(str(arg) for call in log.call_args_list for arg in call.args)
        assert "this config defines m" in logged

    def test_checkpoint_reaches_run_all(self, tmp_path: Path):
        with patch(
            "sieval.cli.run._run_all", new=AsyncMock(return_value={})
        ) as run_all:
            result = runner.invoke(
                app, ["run", str(_task_only(tmp_path)), "--checkpoint", "/models/x"]
            )
        assert result.exit_code == 0, result.output
        assert run_all.await_args is not None
        assert run_all.await_args.kwargs["cli_model"] == CliModel(
            checkpoint="/models/x"
        )

    @pytest.mark.anyio
    async def test_checkpoint_is_launched_with_the_default_plan(self, tmp_path: Path):
        from sieval.cli.run import _run_all
        from sieval.infer.topology.models import ResolveResult
        from tests.unit.cli.test_run import _fake_plan

        resolve = AsyncMock(return_value=ResolveResult(plan=_fake_plan(), steps=()))
        translator = MagicMock()
        translator.translate.return_value = []
        with (
            patch("sieval.cli.run.auto_resolve_plan", new=resolve),
            patch("sieval.cli.run.get_translator", return_value=translator),
            patch("sieval.cli.run.LocalDeployer"),
            patch("sieval.infer.topology.validator.validate_plan", return_value=[]),
            patch("sieval.cli.validation.prepare_prelaunch_reconciliation") as pre,
            patch(
                "sieval.cli.run.launch_model", new=AsyncMock(return_value=([], None))
            ),
            patch("sieval.cli.run.cleanup_model", new=AsyncMock()),
            patch(
                "sieval.cli.session.arun_session",
                new=AsyncMock(return_value={}),
            ) as session,
        ):
            pre.return_value.deployment_plans = {}
            await _run_all(
                _task_only(tmp_path, f"result_dir: {tmp_path / 'out'}\n"),
                cli_model=CliModel(checkpoint="/models/my-sft"),
            )

        assert resolve.await_args is not None
        assert resolve.await_args.kwargs["checkpoint"] == "/models/my-sft"
        assert session.await_args is not None
        assert session.await_args.kwargs["self_managed_endpoints"] == {"my-sft"}
        assert session.await_args.kwargs["cli_model"] == CliModel(
            checkpoint="/models/my-sft"
        )

    @pytest.mark.anyio
    async def test_api_base_turns_auto_serve_off(self, tmp_path: Path):
        from sieval.cli.run import _run_all

        resolve = AsyncMock()
        with (
            patch("sieval.cli.run.auto_resolve_plan", new=resolve),
            patch("sieval.cli.validation.prepare_prelaunch_reconciliation") as pre,
            patch("sieval.cli.run.launch_model", new=AsyncMock()) as launch,
            patch(
                "sieval.cli.session.arun_session",
                new=AsyncMock(return_value={}),
            ),
        ):
            pre.return_value.deployment_plans = {}
            await _run_all(
                _task_only(tmp_path),
                model="served",
                cli_model=CliModel(api_base="http://h/v1"),
            )
        resolve.assert_not_awaited()
        launch.assert_not_awaited()
        assert pre.call_args.kwargs["cli_model"] == CliModel(api_base="http://h/v1")
