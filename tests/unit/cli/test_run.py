"""Tests for sieval.cli.run — run command (all-in-one).

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from sieval.cli import app
from sieval.cli.run import _RecordedPlans
from sieval.cli.serving import SessionServing
from sieval.core.models import (
    Deployment,
    DeploymentTopology,
    Engine,
    ServingFacts,
)
from sieval.infer.backends.translator import BackendCommand
from sieval.infer.config import InferHandle
from sieval.infer.topology.models import (
    DETERMINISTIC_DEFAULT_SEED,
    DeploymentPlan,
    DeviceGroup,
    ParallelTopology,
    ResolveResult,
    RoleAssignment,
    WellKnownRole,
    deployment_plan_projection,
)
from sieval.tasks import MATH500ZeroShotGenTask

runner = CliRunner()


def _fake_plan(deterministic: bool = False) -> DeploymentPlan:
    return DeploymentPlan(
        checkpoint="/tmp/ckpt",
        backend="vllm",
        assignments=(
            RoleAssignment(
                role=WellKnownRole.FULL,
                devices=DeviceGroup(count=1, gpu_model="H100"),
                topology=ParallelTopology(tp=1, dp=1, pp=1),
                engine_params={},
            ),
        ),
        deterministic=deterministic,
    )


def _fake_handle(
    *,
    endpoint: str = "http://localhost:8000/v1",
    handle_id: str = "12345",
    role: str = "full",
) -> InferHandle:
    return InferHandle(
        backend="vllm",
        endpoint=endpoint,
        handle_id=handle_id,
        metadata={"role": role},
    )


def _make_translate_capture() -> tuple[
    list[DeploymentPlan], Callable[[DeploymentPlan], list[BackendCommand]]
]:
    """Return ``(captured_plans, side_effect_fn)`` for a mocked translator.

    The returned list fills as the translator is called; the callback
    always returns a single canonical ``BackendCommand``.
    """
    captured: list[DeploymentPlan] = []

    def capture(plan: DeploymentPlan) -> list[BackendCommand]:
        captured.append(plan)
        return [
            BackendCommand(
                cli_args=["vllm", "serve"],
                backend="vllm",
                host="localhost",
                role="full",
                health_url="http://localhost:8000/health",
            )
        ]

    return captured, capture


@pytest.fixture(autouse=True)
def _a_gpu_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every launch here is faked, so fake the GPUs it would need too.

    A test that means a node without GPUs patches ``detect_local_gpu`` itself.
    """
    from sieval.infer.introspect import GPUInfo

    monkeypatch.setattr(
        "sieval.infer.introspect.detect_local_gpu",
        AsyncMock(return_value=GPUInfo(model="NVIDIA H100", count=8, memory_mib=81559)),
    )


class _UndeclaredTask(MATH500ZeroShotGenTask):
    """A plugin task whose own report declares no metrics."""

    async def report(self, finals, fails):
        return {"mine": 1.0}


class _MalformedTask(MATH500ZeroShotGenTask):
    """A plugin task whose declaration is not a ``ReportMetrics``."""

    report_metrics = frozenset({"score"})


class TestRunCommand:
    def test_run_help(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "all-in-one" in result.output.lower()

    def test_run_file_not_found(self):
        result = runner.invoke(app, ["run", "nonexistent.yaml"])
        assert result.exit_code != 0

    def test_result_dir_exists_hint_uses_cli_flags(self, tmp_path: Path):
        from sieval.core.runners import ResultDirExistsError

        config = tmp_path / "test.yaml"
        config.write_text("models: {}\ntasks: {}")
        existing = tmp_path / "prior_run"

        mock_run_all = AsyncMock(side_effect=ResultDirExistsError(existing))

        with patch("sieval.cli.run._run_all", mock_run_all):
            result = runner.invoke(app, ["run", str(config), "-o", "json"])

        assert result.exit_code != 0
        parsed = json.loads(result.stdout)
        assert parsed["ok"] is False
        error = parsed["error"]
        assert "--resume" in error
        assert "--result-dir" in error
        assert str(existing) in error
        assert "resume=True" not in error

    def test_run_help_shows_ready_timeout(self):
        """`sieval run` exposes the readiness budget it has always enforced."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--ready-timeout" in re.sub(r"\s+", " ", plain)

    def test_ready_timeout_option_reaches_run_all(self, tmp_path: Path):
        """The typer option must arrive at ``_run_all``, not stop at the parser.

        Its neighbours start *at* ``_run_all`` and stop at ``--help``; the
        hand-off in ``register_run_command`` sits between them. Dropping that
        one line restores the original defect and leaves both of them green.
        """
        config = tmp_path / "test.yaml"
        config.write_text("models: {}\ntasks: {}")

        mock_run_all = AsyncMock(return_value={})

        with patch("sieval.cli.run._run_all", mock_run_all):
            result = runner.invoke(
                app, ["run", str(config), "--ready-timeout", "1234", "-o", "json"]
            )

        assert result.exit_code == 0, result.output
        # A patch target that resolves but no longer intercepts would leave the
        # kwargs assertion below unreached, and the test vacuously green.
        assert mock_run_all.await_count == 1
        await_args = mock_run_all.await_args
        assert await_args is not None
        assert await_args.kwargs["ready_timeout"] == 1234.0

    def test_ready_timeout_rejects_a_non_positive_budget(self, tmp_path: Path):
        """A zero/negative budget must be refused before anything is launched.

        Without ``min=`` the poll loop takes it and times out on its first
        pass, so a bad argument reads as a broken engine.
        """
        config = tmp_path / "test.yaml"
        config.write_text("models: {}\ntasks: {}")

        mock_run_all = AsyncMock(return_value={})
        for bad in ("0", "-30"):
            with patch("sieval.cli.run._run_all", mock_run_all):
                result = runner.invoke(
                    app, ["run", str(config), "--ready-timeout", bad]
                )
            assert result.exit_code == 2, f"{bad} was accepted: {result.output}"
        assert mock_run_all.await_count == 0

    def test_infer_start_accepts_both_readiness_option_names(self):
        """Both spellings resolve to the same parameter, so neither is wrong."""
        import click
        import typer.main

        root = typer.main.get_command(app)
        assert isinstance(root, click.Group)
        infer = root.commands["infer"]
        assert isinstance(infer, click.Group)

        param = next(p for p in infer.commands["start"].params if p.name == "timeout")
        assert set(param.opts) == {"--timeout", "--ready-timeout"}
        # Same non-positive guard as `run --ready-timeout`.
        assert isinstance(param.type, click.FloatRange)
        assert param.type.min == 1.0

    def test_ready_timeout_default_is_shared_with_infer_start(self):
        """One named default, so the two commands cannot drift apart again.

        `sieval run` enforced a hardcoded 300s that no flag could reach while
        `sieval infer start` had its own literal default. Three copies of the
        same number is how that happened; this pins them to one.
        """
        import inspect

        from sieval.cli.infer.commands import infer_start
        from sieval.cli.infer.lifecycle import launch_model
        from sieval.cli.run import _run_all
        from sieval.infer.deployer import DEFAULT_READY_TIMEOUT, LocalDeployer

        assert (
            inspect.signature(LocalDeployer.deploy).parameters["timeout"].default
            == DEFAULT_READY_TIMEOUT
        )
        assert (
            inspect.signature(launch_model).parameters["timeout"].default
            == DEFAULT_READY_TIMEOUT
        )
        assert (
            inspect.signature(_run_all).parameters["ready_timeout"].default
            == DEFAULT_READY_TIMEOUT
        )
        # typer wraps the default in the parameter's own default slot
        assert (
            inspect.signature(infer_start).parameters["timeout"].default
            == DEFAULT_READY_TIMEOUT
        )

    def test_run_help_shows_deterministic_flag_only(self):
        """--deterministic appears; --no-deterministic does not (monotone).

        With ``bool | None`` (tri-state) and a single-name
        ``typer.Option("--deterministic")``, typer emits only the positive
        form. The ``bool`` + single-name combination would instead emit
        ``--deterministic/--no-deterministic``.
        """
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--deterministic" in result.output
        assert "--no-deterministic" not in result.output


class TestPrelaunchSequencing:
    @pytest.mark.anyio
    async def test_dry_run_reuses_managed_plan_batch_and_cli_overrides(
        self, tmp_path: Path
    ) -> None:
        from sieval.cli.run import _run_dry_run

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "models": {"model_a": {"path": "/tmp/a"}},
                    "tasks": {},
                }
            )
        )
        plan_dicts = {"model_a": {"backend": "vllm", "assignments": []}}
        prepared = {"model_a": (_fake_plan(), [])}
        dry_result = {"checks": [], "n_errors": 0, "n_warnings": 0}

        with (
            patch(
                "sieval.cli.run._prepare_launch_batch",
                new=AsyncMock(return_value=(plan_dicts, prepared)),
            ) as prepare,
            patch(
                "sieval.cli.validation.run_dry_run",
                return_value=dry_result,
            ) as validate,
            # The tasks' resume plans, which the CLI tests cover.
            patch(
                "sieval.cli.run._plan_tasks",
                new=AsyncMock(return_value=({}, {})),
            ),
        ):
            result = await _run_dry_run(
                config_path,
                resume=True,
                model="override/model",
                result_dir="result-override",
                deterministic=True,
            )

        assert result["checks"] == [{"name": "resume_plan", "ok": True, "detail": ANY}]
        assert result["n_errors"] == 0
        assert result["models"] == {
            "model_a": {
                "plan": plan_dicts["model_a"],
                "launch": [],
                "needs_inference": False,
            }
        }
        assert (result["tasks"], result["needs_inference"]) == ({}, False)
        prepare.assert_awaited_once()
        assert prepare.call_args.kwargs["effective_deterministic"] is True
        assert validate.call_args.kwargs["infer_plans"] == plan_dicts
        assert validate.call_args.kwargs["model_override"] == "override/model"
        assert validate.call_args.kwargs["resume"] is True
        assert validate.call_args.kwargs["result_dir_override"] == "result-override"
        assert validate.call_args.kwargs["deterministic_override"] is True
        assert validate.call_args.kwargs["self_managed_endpoints"] == {"model_a"}

    @staticmethod
    async def _dry_run(
        tmp_path: Path,
        *,
        n_errors: int = 0,
        resume: bool = True,
        plan_tasks: AsyncMock | None = None,
        tasks: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """``_run_dry_run`` with ``model_a`` served and ``api`` not."""
        from sieval.cli.run import _run_dry_run

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(yaml.safe_dump({"models": {}, "tasks": tasks or {}}))
        plan_dicts = {"model_a": {"backend": "vllm", "assignments": []}}
        checks = [{"name": "schema", "ok": not n_errors}]
        with (
            patch.dict(os.environ, {"SIEVAL_TARGET_GPU": "NVIDIA H100:8:81559"}),
            patch(
                "sieval.cli.run._prepare_launch_batch",
                new=AsyncMock(
                    return_value=(plan_dicts, {"model_a": (_fake_plan(), [])})
                ),
            ),
            patch(
                "sieval.cli.validation.run_dry_run",
                return_value={"checks": checks, "n_errors": n_errors, "n_warnings": 0},
            ),
            patch(
                "sieval.cli.run._plan_tasks",
                new=plan_tasks or AsyncMock(return_value=({}, {})),
            ) as planned,
        ):
            result = await _run_dry_run(config_path, resume=resume)
        result["_planned"] = planned.await_count
        return result

    @staticmethod
    def _plan(to_infer: int) -> dict[str, object]:
        return {
            "start": "resume",
            "recorded": "r/t",
            "samples": 2,
            "outcomes": {"kept": 2 - to_infer},
            "stages": {"rerun_from": None},
            "new": to_infer,
            "to_infer": to_infer,
            "to_grade": to_infer,
        }

    @pytest.mark.anyio
    async def test_needs_inference_is_whether_an_engine_starts(
        self, tmp_path: Path
    ) -> None:
        # `api` infers through an api_base model: no engine starts for it.
        plans = AsyncMock(
            return_value=(
                {"t_api": "api", "t_a": "model_a"},
                {"t_api": self._plan(2), "t_a": self._plan(0)},
            )
        )
        result = await self._dry_run(tmp_path, plan_tasks=plans)
        models = cast(dict, result["models"])
        assert models["model_a"]["needs_inference"] is False
        assert result["needs_inference"] is False
        tasks = cast(dict, result["tasks"])
        assert tasks["t_api"]["to_infer"] == 2
        # A task infers through `api` without starting an engine.
        assert tasks["t_api"]["needs_inference"] is True
        assert tasks["t_a"]["needs_inference"] is False
        assert tasks["t_a"]["stages"] == {"rerun_from": None}

    @pytest.mark.anyio
    async def test_each_task_reports_its_metric_names(self, tmp_path: Path) -> None:
        tasks = {
            "math": {"class": "MATH500ZeroShotGenTask"},
            "mmlu": {"class": "sieval.tasks.mmlu_kshot_clp.MMLUFewShotCLPTask"},
            "mine": {"class": "tests.unit.cli.test_run._UndeclaredTask"},
            # Advisory: a class this cannot resolve, or none, is unknown.
            "gone": {"class": "no_such_plugin.Task"},
            "bare": {"dataset": "d"},
        }
        plans = AsyncMock(return_value=({}, {name: self._plan(0) for name in tasks}))
        result = await self._dry_run(tmp_path, plan_tasks=plans, tasks=tasks)
        found = {
            name: entry["metrics"]
            for name, entry in cast(dict, result["tasks"]).items()
        }
        assert {"score", "pass@1", "n_truncated"} <= set(found["math"]["names"])
        assert found["math"]["patterns"] == []
        # A clp task has no rollouts for the runner to count.
        assert "n_truncated" not in found["mmlu"]["names"]
        assert found["mmlu"]["patterns"] == ["score_{category}"]
        assert found["mine"] is None
        assert (found["gone"], found["bare"]) == (None, None)
        assert result["n_errors"] == 0
        assert "report_metrics" not in {c["name"] for c in cast(list, result["checks"])}

    @pytest.mark.anyio
    async def test_a_malformed_declaration_fails_the_dry_run(
        self, tmp_path: Path
    ) -> None:
        # Reading it as unknown would let a check pass that the author broke.
        tasks = {
            "bad": {"class": "tests.unit.cli.test_run._MalformedTask"},
            "math": {"class": "MATH500ZeroShotGenTask"},
        }
        plans = AsyncMock(return_value=({}, {name: self._plan(0) for name in tasks}))
        result = await self._dry_run(tmp_path, plan_tasks=plans, tasks=tasks)
        entries = cast(dict, result["tasks"])
        assert entries["bad"]["metrics"] is None
        assert entries["math"]["metrics"] is not None
        (check,) = [
            c for c in cast(list, result["checks"]) if c["name"] == "report_metrics"
        ]
        assert check["ok"] is False
        assert check["detail"].startswith("bad: ")
        assert "ReportMetrics or None" in check["detail"]
        assert result["n_errors"] == 1

    @pytest.mark.anyio
    async def test_a_fresh_run_starts_every_served_engine(self, tmp_path: Path) -> None:
        # `_models_to_infer` does not plan a run where nothing resumes.
        result = await self._dry_run(tmp_path, resume=False)
        assert cast(dict, result["models"])["model_a"]["needs_inference"] is True
        assert result["needs_inference"] is True

    @pytest.mark.anyio
    async def test_failed_checks_leave_the_plan_unknown(self, tmp_path: Path) -> None:
        result = await self._dry_run(tmp_path, n_errors=1)
        assert result["_planned"] == 0
        assert (result["tasks"], result["needs_inference"]) == (None, None)
        assert cast(dict, result["models"])["model_a"]["needs_inference"] is None
        assert result["gpu"] == "NVIDIA H100:8:81559"

    @pytest.mark.anyio
    async def test_a_planning_error_fails_resume_plan(self, tmp_path: Path) -> None:
        boom = AsyncMock(side_effect=ValueError("result_dir is not a directory"))
        result = await self._dry_run(tmp_path, plan_tasks=boom)
        assert result["n_errors"] == 1
        assert cast(list, result["checks"])[-1] == {
            "name": "resume_plan",
            "ok": False,
            "detail": "result_dir is not a directory",
        }
        assert (result["tasks"], result["needs_inference"]) == (None, None)

    @pytest.mark.anyio
    async def test_a_missing_config_still_has_every_key(self, tmp_path: Path) -> None:
        from sieval.cli.run import _run_dry_run

        result = await _run_dry_run(tmp_path / "absent.yaml")
        assert result["n_errors"] == 1
        for key in ("gpu", "models", "tasks", "needs_inference"):
            assert result[key] is None

    @pytest.mark.anyio
    async def test_binding_error_prevents_every_managed_launch(self, tmp_path: Path):
        from sieval.cli.run import _run_all

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "models": {
                        "model_a": {
                            "path": "/tmp/a",
                            "infer": {"backend": "vllm", "recipe": "test"},
                        },
                        "model_b": {
                            "path": "/tmp/b",
                            "infer": {"backend": "vllm", "recipe": "test"},
                        },
                    },
                    "tasks": {},
                }
            )
        )

        async def resolve(
            _config_path: Path,
            model_name: str,
            *,
            model_type_resolution=None,
        ):
            assert model_type_resolution is not None
            return model_name, _fake_plan(), {}

        translator = MagicMock()
        translator.translate.return_value = [
            BackendCommand(
                cli_args=["vllm", "serve"],
                backend="vllm",
                role="full",
                health_url="http://localhost:8000/health",
            )
        ]
        prepare_prelaunch = MagicMock(
            side_effect=ValueError("binding capability mismatch")
        )
        launch = AsyncMock()
        cleanup = AsyncMock()

        with (
            patch(
                "sieval.cli.run.resolve_infer_config",
                new=AsyncMock(side_effect=resolve),
            ) as resolve_mock,
            patch("sieval.cli.run.get_translator", return_value=translator),
            patch("sieval.cli.run.launch_model", new=launch),
            patch("sieval.cli.run.cleanup_model", new=cleanup),
            patch(
                "sieval.cli.validation.prepare_prelaunch_reconciliation",
                new=prepare_prelaunch,
            ),
            pytest.raises(ValueError, match="binding capability mismatch"),
        ):
            await _run_all(config_path)

        assert resolve_mock.await_count == 2
        assert translator.translate.call_count == 2
        launch.assert_not_awaited()
        cleanup.assert_not_awaited()
        prepare_prelaunch.assert_called_once()
        # Vendored services and grader packages are checked before any engine starts.
        assert prepare_prelaunch.call_args.kwargs["check_services"] is True
        assert prepare_prelaunch.call_args.kwargs["check_python_requires"] is True
        assert set(prepare_prelaunch.call_args.kwargs["infer_plans"]) == {
            "model_a",
            "model_b",
        }

    @pytest.mark.anyio
    async def test_unhandled_launch_patch_fails_before_launch(self, tmp_path: Path):
        from sieval.cli.run import _run_all

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "models": {
                        "model_a": {
                            "path": "/tmp/a",
                            "infer": {"backend": "vllm", "recipe": "test"},
                        }
                    },
                    "tasks": {},
                }
            )
        )

        translator = MagicMock()
        translator.translate.return_value = []
        deployment_plan = MagicMock()
        deployment_plan.launch_patch = {"disable_prefix_cache": True}
        prelaunch_result = MagicMock()
        prelaunch_result.deployment_plans = {"model:model_a": deployment_plan}
        launch = AsyncMock()

        with (
            patch(
                "sieval.cli.run.resolve_infer_config",
                new=AsyncMock(return_value=("model_a", _fake_plan(), {})),
            ),
            patch("sieval.cli.run.get_translator", return_value=translator),
            patch("sieval.cli.run.launch_model", new=launch),
            patch("sieval.cli.run.cleanup_model", new=AsyncMock()),
            patch(
                "sieval.cli.validation.prepare_prelaunch_reconciliation",
                return_value=prelaunch_result,
            ),
            pytest.raises(RuntimeError, match="#47 launch-patch translator"),
        ):
            await _run_all(config_path)

        launch.assert_not_awaited()


class TestReservedServiceLogDir:
    @pytest.mark.anyio
    async def test_a_model_named_services_is_refused_before_launch(
        self, tmp_path: Path
    ):
        """Vendored services log under `logs/services/`; a served model of that
        name would share it."""
        from sieval.cli.run import _run_all

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "models": {
                        "services": {
                            "path": "/tmp/a",
                            "infer": {"backend": "vllm", "recipe": "test"},
                        }
                    },
                    "tasks": {},
                }
            )
        )
        translator = MagicMock()
        translator.translate.return_value = []
        prelaunch_result = MagicMock()
        prelaunch_result.deployment_plans = {}
        launch = AsyncMock()
        with (
            patch(
                "sieval.cli.run.resolve_infer_config",
                new=AsyncMock(return_value=("services", _fake_plan(), {})),
            ),
            patch("sieval.cli.run.get_translator", return_value=translator),
            patch("sieval.cli.run.launch_model", new=launch),
            patch(
                "sieval.cli.validation.prepare_prelaunch_reconciliation",
                return_value=prelaunch_result,
            ),
            pytest.raises(ValueError, match="reserved"),
        ):
            await _run_all(config_path)
        launch.assert_not_awaited()


class TestDeterministicPlanPropagation:
    """Verify --deterministic CLI flag reaches DeploymentPlan.deterministic."""

    @pytest.mark.anyio
    async def test_cli_deterministic_sets_plan_deterministic(self, tmp_path: Path):
        """`sieval run --deterministic` sets `DeploymentPlan.deterministic=True`."""
        from sieval.cli.run import _run_all

        config = {
            "models": {
                "model_a": {
                    "path": "/tmp/ckpt",
                    "infer": {"backend": "vllm", "recipe": "test"},
                }
            },
            "result_dir": str(tmp_path / "out"),
            "tasks": {},
        }
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(yaml.safe_dump(config))

        translated_plans, capture_translate = _make_translate_capture()
        mock_translator = MagicMock()
        mock_translator.translate.side_effect = capture_translate
        mock_handle = _fake_handle()

        with (
            patch(
                "sieval.cli.run.resolve_infer_config",
                new=AsyncMock(return_value=("model_a", _fake_plan(), {})),
            ),
            patch(
                "sieval.cli.run.get_translator",
                return_value=mock_translator,
            ),
            patch(
                "sieval.cli.run.launch_model",
                new=AsyncMock(return_value=([mock_handle], None)),
            ),
            patch("sieval.cli.run.cleanup_model", new=AsyncMock()),
            patch(
                "sieval.cli.run.validate_plan",
                create=True,
                return_value=[],
            ),
            patch(
                "sieval.infer.topology.validator.validate_plan",
                return_value=[],
            ),
            patch(
                "sieval.cli.session.arun_session",
                new=AsyncMock(return_value={}),
            ),
        ):
            await _run_all(
                config_path=config_path,
                verbose=False,
                resume=False,
                deterministic=True,
            )

        # Translator received a plan with deterministic=True
        assert len(translated_plans) == 1
        assert translated_plans[0].deterministic is True

    @pytest.mark.anyio
    async def test_yaml_deterministic_also_sets_plan_deterministic(
        self, tmp_path: Path
    ):
        """YAML `deterministic: true` (no CLI) propagates to DeploymentPlan.

        The YAML→plan mapping is performed inside ``resolve_infer_config``
        (see ``tests/unit/cli/infer/test_resolve.py``); here we confirm
        ``_run_all`` passes that plan through to the translator unchanged
        (no spurious force-on/off in the middle).
        """
        from sieval.cli.run import _run_all

        config = {
            "deterministic": True,  # YAML-level
            "models": {
                "model_a": {
                    "path": "/tmp/ckpt",
                    "infer": {"backend": "vllm", "recipe": "test"},
                }
            },
            "result_dir": str(tmp_path / "out"),
            "tasks": {},
        }
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(yaml.safe_dump(config))

        translated_plans, capture_translate = _make_translate_capture()
        mock_translator = MagicMock()
        mock_translator.translate.side_effect = capture_translate
        mock_handle = _fake_handle()

        # Simulate real resolve_infer_config behavior for this YAML: it
        # reads `deterministic: true` and stamps it onto the plan.
        yaml_stamped_plan = _fake_plan(deterministic=True)

        with (
            patch(
                "sieval.cli.run.resolve_infer_config",
                new=AsyncMock(return_value=("model_a", yaml_stamped_plan, {})),
            ),
            patch(
                "sieval.cli.run.get_translator",
                return_value=mock_translator,
            ),
            patch(
                "sieval.cli.run.launch_model",
                new=AsyncMock(return_value=([mock_handle], None)),
            ),
            patch("sieval.cli.run.cleanup_model", new=AsyncMock()),
            patch(
                "sieval.infer.topology.validator.validate_plan",
                return_value=[],
            ),
            patch(
                "sieval.cli.session.arun_session",
                new=AsyncMock(return_value={}),
            ),
        ):
            await _run_all(
                config_path=config_path,
                verbose=False,
                resume=False,
                # No CLI deterministic — relying on YAML value
            )

        assert len(translated_plans) == 1
        assert translated_plans[0].deterministic is True

    @pytest.mark.anyio
    async def test_deterministic_pins_plan_seed_to_default(self, tmp_path: Path):
        """Under deterministic mode, ``plan.seed`` is pinned to
        ``DETERMINISTIC_DEFAULT_SEED`` (0). Per-request reproducibility is
        governed by YAML ``args.seed``; engine seed is just a fallback."""
        from sieval.cli.run import _run_all

        config = {
            "models": {
                "model_a": {
                    "path": "/tmp/ckpt",
                    "infer": {"backend": "vllm", "recipe": "test"},
                }
            },
            "result_dir": str(tmp_path / "out"),
            "tasks": {},
        }
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(yaml.safe_dump(config))

        translated_plans, capture_translate = _make_translate_capture()
        mock_translator = MagicMock()
        mock_translator.translate.side_effect = capture_translate
        mock_handle = _fake_handle()

        with (
            patch(
                "sieval.cli.run.resolve_infer_config",
                new=AsyncMock(return_value=("model_a", _fake_plan(), {})),
            ),
            patch("sieval.cli.run.get_translator", return_value=mock_translator),
            patch(
                "sieval.cli.run.launch_model",
                new=AsyncMock(return_value=([mock_handle], None)),
            ),
            patch("sieval.cli.run.cleanup_model", new=AsyncMock()),
            patch(
                "sieval.infer.topology.validator.validate_plan",
                return_value=[],
            ),
            patch(
                "sieval.cli.session.arun_session",
                new=AsyncMock(return_value={}),
            ),
        ):
            await _run_all(
                config_path=config_path,
                verbose=False,
                resume=False,
                deterministic=True,
            )

        assert translated_plans[0].deterministic is True
        assert translated_plans[0].seed == DETERMINISTIC_DEFAULT_SEED

    @pytest.mark.anyio
    async def test_path_only_model_inherits_yaml_deterministic(self, tmp_path: Path):
        """Path-only model (no `infer:` section) must still honor YAML
        `deterministic: true` — the branch skips `resolve_infer_config`
        and goes through `auto_resolve_plan`, which doesn't see the YAML.
        `_run_all` applies `resolve_deterministic(cli, config)` uniformly
        so the effective flag reaches both branches.
        """
        from sieval.cli.run import _run_all

        config = {
            "deterministic": True,  # YAML-level
            "models": {
                "model_a": {"path": "/tmp/ckpt"},  # path-only, no `infer:`
            },
            "result_dir": str(tmp_path / "out"),
            "tasks": {},
        }
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(yaml.safe_dump(config))

        translated_plans, capture_translate = _make_translate_capture()
        mock_translator = MagicMock()
        mock_translator.translate.side_effect = capture_translate
        mock_handle = _fake_handle()

        # auto_resolve_plan doesn't read YAML; it returns a plain non-
        # deterministic plan. The fix in `_run_all` must stamp the YAML
        # deterministic onto this plan before it reaches the translator.
        resolve_result = ResolveResult(plan=_fake_plan(), steps=())

        with (
            patch(
                "sieval.cli.run.auto_resolve_plan",
                new=AsyncMock(return_value=resolve_result),
            ),
            patch("sieval.cli.run.get_translator", return_value=mock_translator),
            patch(
                "sieval.cli.run.launch_model",
                new=AsyncMock(return_value=([mock_handle], None)),
            ),
            patch("sieval.cli.run.cleanup_model", new=AsyncMock()),
            patch(
                "sieval.infer.topology.validator.validate_plan",
                return_value=[],
            ),
            patch(
                "sieval.cli.session.arun_session",
                new=AsyncMock(return_value={}),
            ),
        ):
            await _run_all(
                config_path=config_path,
                verbose=False,
                resume=False,
                # No CLI --deterministic — YAML must drive it end-to-end.
            )

        assert len(translated_plans) == 1
        assert translated_plans[0].deterministic is True


class TestPathOnlyCapabilityLayer:
    """Path-only mode must translate the config model type into the recipe's
    capability vocabulary before handing it to ``auto_resolve_plan``.

    The two vocabularies sit one call apart (``chat``/``gen`` in the config,
    ``instruct``/``base`` in the recipe). Every other test here patches
    ``auto_resolve_plan`` wholesale, so without these assertions dropping the
    ``capability_model_type`` translation — passing a raw ``"chat"`` straight
    through — leaves the whole suite green while an instruct model silently
    loses its parser params.
    """

    async def _capability_for(self, tmp_path: Path, config: dict) -> str:
        from sieval.cli.run import _run_all

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(yaml.safe_dump(config))

        resolve = AsyncMock(return_value=ResolveResult(plan=_fake_plan(), steps=()))
        mock_translator = MagicMock()
        mock_translator.translate.side_effect = _make_translate_capture()[1]
        mock_handle = _fake_handle()

        with (
            patch("sieval.cli.run.auto_resolve_plan", new=resolve),
            patch("sieval.cli.run.get_translator", return_value=mock_translator),
            patch(
                "sieval.cli.run.launch_model",
                new=AsyncMock(return_value=([mock_handle], None)),
            ),
            patch("sieval.cli.run.cleanup_model", new=AsyncMock()),
            patch("sieval.infer.topology.validator.validate_plan", return_value=[]),
            patch(
                "sieval.cli.session.arun_session",
                new=AsyncMock(return_value={}),
            ),
        ):
            await _run_all(config_path=config_path, verbose=False, resume=False)

        resolve.assert_awaited_once()
        call = resolve.await_args
        assert call is not None
        return call.kwargs["capability"]

    @pytest.mark.anyio
    async def test_chat_config_resolves_instruct_layer(self, tmp_path: Path):
        capability = await self._capability_for(
            tmp_path,
            {
                "models": {"model_a": {"path": "/tmp/ckpt", "type": "chat"}},
                "result_dir": str(tmp_path / "out"),
                "tasks": {},
            },
        )
        assert capability == "instruct"

    @pytest.mark.anyio
    async def test_gen_config_resolves_base_layer(self, tmp_path: Path):
        capability = await self._capability_for(
            tmp_path,
            {
                "models": {"model_a": {"path": "/tmp/ckpt", "type": "gen"}},
                "result_dir": str(tmp_path / "out"),
                "tasks": {},
            },
        )
        assert capability == "base"

    @pytest.mark.anyio
    async def test_undeclared_type_is_derived_from_the_tasks(self, tmp_path: Path):
        """The load-bearing case: no `type:` in the config, a gen task on it."""
        capability = await self._capability_for(
            tmp_path,
            {
                "models": {"model_a": {"path": "/tmp/ckpt"}},
                "result_dir": str(tmp_path / "out"),
                "tasks": {
                    "arc_ppl": {
                        "model": "model_a",
                        "class": "ARCEasyFewShotPplTask",
                        "dataset": {"class": "fake.Dataset"},
                    },
                },
            },
        )
        assert capability == "base"


class TestDeterministicPassedToSession:
    """`_run_all` forwards the raw CLI ``deterministic`` value to
    ``arun_session``; EvalSession computes the monotone OR with YAML
    internally (single source of truth per layer).
    """

    @pytest.mark.anyio
    async def test_raw_cli_value_is_forwarded(self, tmp_path: Path):
        from sieval.cli.run import _run_all

        config = {
            "deterministic": True,  # YAML-only; CLI left unset
            "models": {
                "model_a": {
                    "path": "/tmp/ckpt",
                    "infer": {"backend": "vllm", "recipe": "test"},
                }
            },
            "result_dir": str(tmp_path / "out"),
            "tasks": {},
        }
        config_path = tmp_path / "user-facing.yaml"
        config_path.write_text(yaml.safe_dump(config))

        mock_handle = _fake_handle()
        mock_translator = MagicMock()
        mock_translator.translate.return_value = [
            BackendCommand(
                cli_args=["vllm", "serve"],
                backend="vllm",
                host="localhost",
                role="full",
                health_url="http://localhost:8000/health",
            )
        ]
        arun_session_mock = AsyncMock(return_value={})

        with (
            patch(
                "sieval.cli.run.resolve_infer_config",
                new=AsyncMock(return_value=("model_a", _fake_plan(), {})),
            ),
            patch("sieval.cli.run.get_translator", return_value=mock_translator),
            patch(
                "sieval.cli.run.launch_model",
                new=AsyncMock(return_value=([mock_handle], None)),
            ),
            patch("sieval.cli.run.cleanup_model", new=AsyncMock()),
            patch(
                "sieval.infer.topology.validator.validate_plan",
                return_value=[],
            ),
            patch(
                "sieval.cli.session.arun_session",
                new=arun_session_mock,
            ),
        ):
            await _run_all(
                config_path=config_path,
                verbose=False,
                resume=False,
                # CLI deterministic left as None — YAML should drive it.
            )

        arun_session_mock.assert_called_once()
        kwargs = arun_session_mock.call_args.kwargs
        # Raw CLI value (None) is forwarded; EvalSession resolves the
        # YAML leg internally via `resolve_deterministic`. End-to-end
        # YAML→session semantics are covered by TestDeterministicMode
        # in test_session.py.
        assert kwargs["deterministic"] is None

    @pytest.mark.anyio
    async def test_ready_timeout_reaches_the_deploy_call(self, tmp_path: Path):
        """--ready-timeout must arrive at launch_model, not stop at the CLI.

        The readiness budget was previously unreachable from `sieval run`: the
        default was applied three call-frames down and nothing threaded a
        caller's value to it.
        """
        from sieval.cli.run import _run_all

        config = {
            "models": {
                "model_a": {
                    "path": "/tmp/ckpt",
                    "infer": {"backend": "vllm", "recipe": "test"},
                }
            },
            "result_dir": str(tmp_path / "out"),
            "tasks": {},
        }
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(yaml.safe_dump(config))

        mock_translator = MagicMock()
        mock_translator.translate.return_value = [
            BackendCommand(
                cli_args=["vllm", "serve"],
                backend="vllm",
                host="localhost",
                role="full",
                health_url="http://localhost:8000/health",
            )
        ]
        launch = AsyncMock(return_value=([_fake_handle()], None))

        with (
            patch(
                "sieval.cli.run.resolve_infer_config",
                new=AsyncMock(return_value=("model_a", _fake_plan(), {})),
            ),
            patch("sieval.cli.run.get_translator", return_value=mock_translator),
            patch("sieval.cli.run.launch_model", new=launch),
            patch("sieval.cli.run.cleanup_model", new=AsyncMock()),
            patch("sieval.infer.topology.validator.validate_plan", return_value=[]),
            patch(
                "sieval.cli.session.arun_session",
                new=AsyncMock(return_value={}),
            ),
        ):
            await _run_all(config_path=config_path, ready_timeout=1234.0)

        launch.assert_awaited_once()
        await_args = launch.await_args
        assert await_args is not None
        assert await_args.kwargs["timeout"] == 1234.0


class TestRunHandlesAndLogs:
    """`sieval run` owns run-<pid>/ handles and keeps engine logs with results."""

    def _config(self, tmp_path: Path, result_dir: Path | None) -> Path:
        config: dict[str, object] = {
            "models": {
                "model_a": {
                    "path": "/tmp/ckpt",
                    "infer": {"backend": "vllm", "recipe": "test"},
                }
            },
            "tasks": {},
        }
        if result_dir is not None:
            config["result_dir"] = str(result_dir)
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(yaml.safe_dump(config))
        return config_path

    async def _run(
        self, tmp_path: Path, config_path: Path, **kwargs: object
    ) -> tuple[AsyncMock, AsyncMock]:
        launch = AsyncMock(return_value=([_fake_handle()], None))
        cleanup = AsyncMock()
        with patch("sieval.cli.run.launch_model", new=launch):
            await self._run_patched(tmp_path, config_path, cleanup=cleanup, **kwargs)
        return launch, cleanup

    async def _run_patched(
        self,
        tmp_path: Path,
        config_path: Path,
        cleanup: AsyncMock | None = None,
        **kwargs: object,
    ) -> None:
        """Run `_run_all` with `launch_model` left to the caller to patch."""
        from sieval.cli.run import _run_all

        translator = MagicMock()
        translator.translate.return_value = [
            BackendCommand(cli_args=["vllm", "serve"], backend="vllm", role="full")
        ]
        with (
            patch("sieval.cli.infer.lifecycle.HANDLE_DIR", tmp_path / "handles"),
            patch(
                "sieval.cli.run.resolve_infer_config",
                new=AsyncMock(return_value=("model_a", _fake_plan(), {})),
            ),
            patch("sieval.cli.run.get_translator", return_value=translator),
            patch("sieval.cli.run.cleanup_model", new=cleanup or AsyncMock()),
            patch(
                "sieval.cli.session.arun_session",
                new=AsyncMock(return_value={}),
            ),
        ):
            await _run_all(config_path, **kwargs)  # type: ignore[arg-type]

    @pytest.mark.anyio
    async def test_handle_under_run_pid_and_logs_under_result_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("SIEVAL_KEEP_LOGS", raising=False)
        out = tmp_path / "out"
        run_dir = tmp_path / "handles" / f"run-{os.getpid()}"
        run_dir.mkdir(parents=True)
        launch, cleanup = await self._run(tmp_path, self._config(tmp_path, out))

        assert launch.await_args is not None
        kwargs = launch.await_args.kwargs
        assert kwargs["handle_path"] == run_dir / "model_a.json"
        assert kwargs["log_dir"] == out / "logs" / "model_a"
        assert cleanup.await_args is not None
        assert cleanup.await_args.kwargs["handle_path"] == run_dir / "model_a.json"
        assert not run_dir.exists()

    @pytest.mark.anyio
    async def test_without_result_dir_logs_go_to_the_shared_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("SIEVAL_KEEP_LOGS", "7")
        config_path = self._config(tmp_path, None)
        launch, _ = await self._run(tmp_path, config_path)
        assert launch.await_args is not None
        assert launch.await_args.kwargs["log_dir"] is None
        assert launch.await_args.kwargs["keep_logs"] == 7

        launch, _ = await self._run(tmp_path, config_path, keep_logs=1)
        assert launch.await_args is not None
        assert launch.await_args.kwargs["keep_logs"] == 1

    @pytest.mark.anyio
    async def test_result_dir_run_ignores_keep_logs_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Retention never applies under result_dir, so a bad env var is moot."""
        monkeypatch.setenv("SIEVAL_KEEP_LOGS", "not-a-number")
        config_path = self._config(tmp_path, tmp_path / "out")
        launch, _ = await self._run(tmp_path, config_path, keep_logs=5)
        assert launch.await_args is not None
        assert launch.await_args.kwargs["keep_logs"] is None

    @pytest.mark.anyio
    async def test_held_handle_names_the_handle_to_stop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from sieval.cli.infer import HandleHeldError

        run_dir = f"run-{os.getpid()}"
        held = HandleHeldError(
            tmp_path / "handles" / run_dir / "model_a.json",
            {"phase": "running"},
            4242,
        )
        with (
            patch("sieval.cli.run.launch_model", new=AsyncMock(side_effect=held)),
            pytest.raises(HandleHeldError) as excinfo,
        ):
            await self._run_patched(tmp_path, self._config(tmp_path, None))
        message = str(excinfo.value)
        assert "PID 4242" in message
        assert f"sieval infer stop {run_dir}/model_a" in message

    @pytest.mark.anyio
    async def test_result_dir_flag_wins_over_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        config_path = self._config(tmp_path, tmp_path / "from-config")
        launch, _ = await self._run(
            tmp_path, config_path, result_dir=str(tmp_path / "from-flag")
        )
        assert launch.await_args is not None
        assert launch.await_args.kwargs["log_dir"] == (
            tmp_path / "from-flag" / "logs" / "model_a"
        )

    def test_keep_logs_flag_reaches_run_all(self, tmp_path: Path):
        config_path = self._config(tmp_path, None)
        with patch(
            "sieval.cli.run._run_all", new=AsyncMock(return_value={})
        ) as run_all:
            result = runner.invoke(app, ["run", str(config_path), "--keep-logs", "0"])
        assert result.exit_code == 0, result.output
        assert run_all.await_args is not None
        assert run_all.await_args.kwargs["keep_logs"] == 0


class TestDeploymentPropagation:
    """Auto-serve hands off complete typed state without endpoint projection."""

    @pytest.mark.anyio
    async def test_multi_endpoint_deployment_reaches_arun_session_unchanged(
        self, tmp_path: Path
    ):
        from sieval.cli.run import _run_all

        plan = DeploymentPlan(
            checkpoint="/tmp/ckpt",
            backend="sglang",
            assignments=(
                RoleAssignment(
                    role=WellKnownRole.PREFILL,
                    devices=DeviceGroup(count=4, gpu_model="H100"),
                    topology=ParallelTopology(tp=4),
                ),
                RoleAssignment(
                    role=WellKnownRole.DECODE,
                    devices=DeviceGroup(count=4, gpu_model="H100"),
                    topology=ParallelTopology(tp=2, dp=2),
                ),
            ),
        )
        config = {
            "models": {
                "model_a": {
                    "path": "/tmp/ckpt",
                    "infer": {"backend": "sglang", "recipe": "test"},
                    "service_role": WellKnownRole.DECODE,
                }
            },
            "result_dir": str(tmp_path / "out"),
            "tasks": {},
        }
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(yaml.safe_dump(config))

        handles = [
            _fake_handle(
                endpoint="http://localhost:30000/v1",
                handle_id="prefill-handle",
                role=WellKnownRole.PREFILL,
            ),
            _fake_handle(
                endpoint="http://localhost:30001/v1",
                handle_id="decode-handle",
                role=WellKnownRole.DECODE,
            ),
        ]
        facts = ServingFacts(
            engine_version="0.4.6",
            tokenizer_available=True,
            prefix_cache_enabled=False,
            max_top_logprobs=20,
        )
        realized = Deployment(
            deployment_id="served-model-a",
            plan=deployment_plan_projection(plan),
            engine=Engine("sglang"),
            engine_source="deployment",
            api_base="http://localhost:30001/v1",
            endpoints={
                WellKnownRole.PREFILL: "http://localhost:30000/v1",
                WellKnownRole.DECODE: "http://localhost:30001/v1",
            },
            topology=DeploymentTopology(
                is_disaggregated=True,
                roles=(WellKnownRole.PREFILL, WellKnownRole.DECODE),
                total_gpus=8,
            ),
            metrics_url="http://localhost:30001/metrics",
            facts=facts,
        )
        mock_translator = MagicMock()
        mock_translator.translate.return_value = [
            BackendCommand(
                cli_args=["sglang", "serve"],
                backend="sglang",
                host="localhost",
                role="decode",
                health_url="http://localhost:30001/health",
            )
        ]
        arun_session_mock = AsyncMock(return_value={})

        with (
            patch(
                "sieval.cli.run.resolve_infer_config",
                new=AsyncMock(return_value=("model_a", plan, {})),
            ),
            patch("sieval.cli.run.get_translator", return_value=mock_translator),
            patch(
                "sieval.cli.run.launch_model",
                new=AsyncMock(return_value=(handles, None)),
            ),
            patch(
                "sieval.cli.run.LocalDeployer.build_deployment",
                return_value=realized,
            ) as build_deployment,
            patch("sieval.cli.run.cleanup_model", new=AsyncMock()),
            patch(
                "sieval.infer.topology.validator.validate_plan",
                return_value=[],
            ),
            patch(
                "sieval.cli.session.arun_session",
                new=arun_session_mock,
            ),
        ):
            await _run_all(config_path=config_path, verbose=False, resume=False)

        arun_session_mock.assert_called_once()
        kwargs = arun_session_mock.call_args.kwargs
        args = arun_session_mock.call_args.args
        # First positional is config_path — the ORIGINAL (no tempfile)
        assert args[0] == str(config_path) or args[0] == config_path
        assert "endpoint_map" not in kwargs
        assert "model_a" in kwargs["infer_plans"]
        assert kwargs["infer_plans"]["model_a"]["backend"] == "sglang"
        deployment = kwargs["realized_deployments"]["model_a"]
        assert deployment is realized
        assert deployment.endpoints == {
            "prefill": "http://localhost:30000/v1",
            "decode": "http://localhost:30001/v1",
        }
        assert deployment.topology is realized.topology
        assert deployment.engine is realized.engine
        assert deployment.facts is facts
        build_deployment.assert_called_once_with(plan, handles)

    @pytest.mark.anyio
    async def test_session_failure_still_cleans_realized_deployment(
        self, tmp_path: Path
    ):
        from sieval.cli.run import _run_all

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "models": {
                        "model_a": {
                            "path": "/tmp/ckpt",
                            "infer": {"backend": "vllm", "recipe": "test"},
                        }
                    },
                    "tasks": {},
                }
            )
        )
        handle = _fake_handle()
        translator = MagicMock()
        translator.translate.return_value = [
            BackendCommand(
                cli_args=["vllm", "serve"],
                backend="vllm",
                role="full",
                health_url="http://localhost:8000/health",
            )
        ]
        cleanup = AsyncMock()

        with (
            patch(
                "sieval.cli.run.resolve_infer_config",
                new=AsyncMock(return_value=("model_a", _fake_plan(), {})),
            ),
            patch("sieval.cli.run.get_translator", return_value=translator),
            patch(
                "sieval.cli.run.launch_model",
                new=AsyncMock(return_value=([handle], None)),
            ),
            patch("sieval.cli.run.cleanup_model", new=cleanup),
            patch(
                "sieval.cli.session.arun_session",
                new=AsyncMock(side_effect=RuntimeError("evaluation failed")),
            ),
            pytest.raises(RuntimeError, match="evaluation failed"),
        ):
            await _run_all(config_path)

        cleanup.assert_awaited_once_with(
            "model_a", [handle], deployer=ANY, handle_path=ANY
        )

    def test_run_module_does_not_import_tempfile(self):
        """Regression guard: the tempfile-YAML-patch path is gone — `tempfile`
        must not be re-imported into sieval.cli.run. A future edit that brings
        back ``import tempfile`` (and presumably the tempfile dance) will trip
        this assertion before any behavioral test has a chance to."""
        import sieval.cli.run as run_module

        assert not hasattr(run_module, "tempfile"), (
            "sieval.cli.run should no longer import tempfile — the former "
            "tempfile YAML-patch path was replaced by endpoint_map kwarg "
            "propagation through EvalSession."
        )


class TestExplicitEndpointRefusal:
    """``args.api_base`` next to a checkpoint is refused before launch (#189)."""

    @staticmethod
    def _write(tmp_path: Path, model: dict) -> Path:
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "models": {"m": {"name": "org/model", "type": "gen", **model}},
                    "result_dir": str(tmp_path / "out"),
                    "tasks": {},
                }
            )
        )
        return config_path

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("model", "source", "serve_key"),
        [
            (
                {"infer": {"backend": "sglang", "checkpoint": "/tmp/ckpt"}},
                "infer.checkpoint",
                "infer",
            ),
            ({"path": "/tmp/ckpt"}, "path", "path"),
            (
                {"infer": {"backend": "sglang"}, "path": "/tmp/ckpt"},
                "path",
                "path",
            ),
        ],
        ids=["checkpoint", "path", "infer_without_checkpoint"],
    )
    @pytest.mark.parametrize("dry_run", [False, True])
    async def test_args_api_base_with_checkpoint_refuses_before_resolution(
        self,
        tmp_path: Path,
        model: dict,
        source: str,
        serve_key: str,
        dry_run: bool,
    ):
        from sieval.cli.run import _run_all, _run_dry_run

        config_path = self._write(
            tmp_path, {**model, "args": {"api_base": "https://proxy.example/v1"}}
        )
        resolve_infer = AsyncMock()
        resolve_path = AsyncMock()
        launch = AsyncMock()

        with (
            patch("sieval.cli.run.resolve_infer_config", new=resolve_infer),
            patch("sieval.cli.run.auto_resolve_plan", new=resolve_path),
            patch("sieval.cli.run.launch_model", new=launch),
            pytest.raises(ValueError) as excinfo,
        ):
            if dry_run:
                await _run_dry_run(config_path)
            else:
                await _run_all(config_path=config_path, verbose=False, resume=False)

        message = str(excinfo.value)
        assert f"sets both 'args.api_base' and '{source}'" in message
        assert f"remove '{serve_key}' to evaluate" in message
        assert "'https://proxy.example/v1'" in message
        resolve_infer.assert_not_awaited()
        resolve_path.assert_not_awaited()
        launch.assert_not_awaited()

    @pytest.mark.anyio
    async def test_empty_args_api_base_is_not_an_endpoint(self, tmp_path: Path):
        from sieval.cli.run import _prepare_launch_batch

        config_path = self._write(
            tmp_path, {"path": "/tmp/ckpt", "args": {"api_base": ""}}
        )
        plan = MagicMock(deterministic=False, backend="sglang")
        result = MagicMock(plan=plan)
        with (
            patch(
                "sieval.cli.run.auto_resolve_plan", new=AsyncMock(return_value=result)
            ) as resolve,
            patch("sieval.cli.run.unwrap_proxies", return_value={}),
            patch("sieval.infer.topology.validator.validate_plan", return_value=[]),
            patch("sieval.cli.run.get_translator"),
            patch("sieval.cli.run.inject_user_env"),
        ):
            await _prepare_launch_batch(
                config_path,
                yaml.safe_load(config_path.read_text()),
                effective_deterministic=False,
            )

        resolve.assert_awaited_once()

    @pytest.mark.anyio
    async def test_top_level_api_base_still_skips_the_launch(self, tmp_path: Path):
        from sieval.cli.run import _prepare_launch_batch

        model = {
            "api_base": "https://proxy.example/v1",
            "args": {"api_base": "https://proxy.example/v1"},
            "infer": {"backend": "sglang", "checkpoint": "/tmp/ckpt"},
        }
        config_path = self._write(tmp_path, model)

        plans, launches = await _prepare_launch_batch(
            config_path,
            yaml.safe_load(config_path.read_text()),
            effective_deterministic=False,
        )

        assert plans == {}
        assert launches == {}


class TestEffectiveConfigRerunSafety:
    """End-to-end: `sieval run` produces an effective_config.yaml without
    baked-in api_base/api_key/auto-filled name. Rerun via `sieval run` would
    see the same raw YAML (no api_base) and re-launch services correctly."""

    @pytest.mark.anyio
    async def test_persisted_config_has_no_injected_api_base(self, tmp_path: Path):
        from sieval.cli.run import _run_all
        from sieval.cli.session import EvalSession

        config = {
            "models": {
                "model_a": {
                    "path": "/tmp/ckpt",
                    "infer": {"backend": "vllm", "recipe": "test"},
                }
            },
            "result_dir": str(tmp_path / "out"),
            "tasks": {},
        }
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(yaml.safe_dump(config))

        mock_handle = _fake_handle()
        mock_translator = MagicMock()
        mock_translator.translate.return_value = [
            BackendCommand(
                cli_args=["vllm", "serve"],
                backend="vllm",
                host="localhost",
                role="full",
                health_url="http://localhost:8000/health",
            )
        ]

        # Stub out _prepare_execution so we don't actually load models/datasets/tasks;
        # but keep _persist_effective_config + _persist_infer_plans real.
        async def stub_prepare(self):
            self.runner = MagicMock()
            self.runner.arun = AsyncMock(return_value={})

        with (
            patch(
                "sieval.cli.run.resolve_infer_config",
                new=AsyncMock(return_value=("model_a", _fake_plan(), {})),
            ),
            patch("sieval.cli.run.get_translator", return_value=mock_translator),
            patch(
                "sieval.cli.run.launch_model",
                new=AsyncMock(return_value=([mock_handle], None)),
            ),
            patch("sieval.cli.run.cleanup_model", new=AsyncMock()),
            patch(
                "sieval.infer.topology.validator.validate_plan",
                return_value=[],
            ),
            patch.object(EvalSession, "_prepare_execution", stub_prepare),
        ):
            await _run_all(config_path=config_path, verbose=False, resume=False)

        persisted = tmp_path / "out" / "effective_config.yaml"
        assert persisted.exists(), "effective_config.yaml was not written"

        loaded = yaml.safe_load(persisted.read_text())
        m = loaded["models"]["model_a"]
        # api_base / api_key / auto-filled name must NOT be in the persisted body
        assert "api_base" not in m, (
            f"effective_config.yaml contains injected api_base: {m.get('api_base')}"
        )
        assert "api_key" not in m, (
            f"effective_config.yaml contains injected api_key: {m.get('api_key')}"
        )
        # The original path IS preserved so sieval run can re-launch
        assert m["path"] == "/tmp/ckpt"

    @pytest.mark.anyio
    async def test_infer_plans_yaml_written_for_auto_serve(self, tmp_path: Path):
        from sieval.cli.run import _run_all
        from sieval.cli.session import EvalSession

        config = {
            "models": {
                "model_a": {
                    "path": "/tmp/ckpt",
                    "infer": {"backend": "vllm", "recipe": "test"},
                }
            },
            "result_dir": str(tmp_path / "out"),
            "tasks": {},
        }
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(yaml.safe_dump(config))

        mock_handle = _fake_handle()
        mock_translator = MagicMock()
        mock_translator.translate.return_value = [
            BackendCommand(
                cli_args=["vllm", "serve"],
                backend="vllm",
                host="localhost",
                role="full",
                health_url="http://localhost:8000/health",
            )
        ]

        async def stub_prepare(self):
            self.runner = MagicMock()
            self.runner.arun = AsyncMock(return_value={})

        with (
            patch(
                "sieval.cli.run.resolve_infer_config",
                new=AsyncMock(return_value=("model_a", _fake_plan(), {})),
            ),
            patch("sieval.cli.run.get_translator", return_value=mock_translator),
            patch(
                "sieval.cli.run.launch_model",
                new=AsyncMock(return_value=([mock_handle], None)),
            ),
            patch("sieval.cli.run.cleanup_model", new=AsyncMock()),
            patch(
                "sieval.infer.topology.validator.validate_plan",
                return_value=[],
            ),
            patch.object(EvalSession, "_prepare_execution", stub_prepare),
        ):
            await _run_all(config_path=config_path, verbose=False, resume=False)

        infer_plans = tmp_path / "out" / "infer_plans.yaml"
        assert infer_plans.exists()
        loaded = yaml.safe_load(infer_plans.read_text())
        assert "model_a" in loaded["models"]
        assert loaded["models"]["model_a"]["backend"] == "vllm"


class TestDeclaredTargetGpu:
    """``SIEVAL_TARGET_GPU`` resolves on any node the plan a node with those
    GPUs resolves, and a resume reuses the plan it recorded for them."""

    H100 = "NVIDIA H100 80GB HBM3:8:81559"

    def _config(self, tmp_path: Path) -> Path:
        checkpoint = tmp_path / "Qwen3-32B"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["Qwen3ForCausalLM"],
                    "model_type": "qwen3",
                    "vocab_size": 151936,
                    "hidden_size": 5120,
                    "num_hidden_layers": 64,
                    "num_attention_heads": 64,
                    "num_key_value_heads": 8,
                    "intermediate_size": 25600,
                    "torch_dtype": "bfloat16",
                }
            )
        )
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "models": {"m": {"path": str(checkpoint), "type": "chat"}},
                    "result_dir": str(tmp_path / "out"),
                    "tasks": {},
                }
            )
        )
        return config_path

    async def _run(
        self, config_path: Path, *, gpu: object, resume: bool = False
    ) -> tuple[AsyncMock, AsyncMock]:
        """Run with this node's GPUs faked as *gpu*; return the launch and
        session mocks."""
        from sieval.cli.run import _run_all

        launch = AsyncMock(return_value=([_fake_handle()], None))
        session = AsyncMock(return_value={})
        nvidia_smi = AsyncMock(return_value=gpu)
        with (
            patch("sieval.infer.topology.resolver.detect_local_gpu", new=nvidia_smi),
            patch("sieval.infer.introspect.detect_local_gpu", new=nvidia_smi),
            patch("sieval.cli.run.detect_local_gpu", new=nvidia_smi),
            patch("sieval.cli.run.launch_model", new=launch),
            patch("sieval.cli.run.cleanup_model", new=AsyncMock()),
            patch("sieval.cli.session.arun_session", new=session),
        ):
            await _run_all(config_path, resume=resume)
        return launch, session

    @staticmethod
    def _gpu(text: str) -> object:
        from sieval.infer.introspect import parse_target_gpu

        return parse_target_gpu(text)

    @pytest.mark.anyio
    async def test_a_declared_plan_is_the_detected_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        config_path = self._config(tmp_path)
        _, detected = await self._run(config_path, gpu=self._gpu(self.H100))

        monkeypatch.setenv("SIEVAL_TARGET_GPU", self.H100)
        _, declared = await self._run(config_path, gpu=self._gpu(self.H100))

        assert declared.await_args is not None and detected.await_args is not None
        plans = declared.await_args.kwargs["infer_plans"]
        assert plans == detected.await_args.kwargs["infer_plans"]
        # Both name the GPUs the plans are for, as the variable spells them.
        assert declared.await_args.kwargs["infer_gpu"] == self.H100
        assert detected.await_args.kwargs["infer_gpu"] == self.H100

    @pytest.mark.anyio
    async def test_declared_gpus_other_than_this_nodes_refuse_the_launch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("SIEVAL_TARGET_GPU", self.H100)
        with pytest.raises(RuntimeError, match="no GPU"):
            await self._run(self._config(tmp_path), gpu=None)

    def _record(self, tmp_path: Path, config_path: Path, plan: dict) -> None:
        """What a session that served ``m`` with *plan* on the H100s left."""
        out = tmp_path / "out"
        out.mkdir(exist_ok=True)
        (out / "effective_config.yaml").write_text(config_path.read_text())
        (out / "infer_plans.yaml").write_text(
            yaml.safe_dump({"gpu": self.H100, "models": {"m": plan}}, sort_keys=False)
        )

    @pytest.mark.anyio
    async def test_a_resume_reuses_the_plan_recorded_for_the_declared_gpus(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from sieval.cli.session import unwrap_proxies

        config_path = self._config(tmp_path)
        # Not what this SiEval resolves, so only a reuse produces it.
        recorded = unwrap_proxies(_fake_plan())
        recorded["checkpoint"] = yaml.safe_load(config_path.read_text())["models"]["m"][
            "path"
        ]
        self._record(tmp_path, config_path, recorded)
        monkeypatch.setenv("SIEVAL_TARGET_GPU", self.H100)
        resolve = AsyncMock()
        with patch("sieval.cli.run.auto_resolve_plan", new=resolve):
            _, session = await self._run(
                config_path, gpu=self._gpu(self.H100), resume=True
            )

        resolve.assert_not_awaited()
        assert session.await_args is not None
        assert session.await_args.kwargs["infer_plans"] == {"m": recorded}

    @pytest.mark.anyio
    async def test_a_plan_recorded_with_a_secret_is_resolved_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """``infer_plans.yaml`` is written redacted, so reusing a plan with an
        engine ``api_key`` would launch the engine with the marker as its key."""
        from sieval.cli.session import redact_secrets, unwrap_proxies

        config_path = self._config(tmp_path)
        recorded = unwrap_proxies(_fake_plan())
        recorded["checkpoint"] = yaml.safe_load(config_path.read_text())["models"]["m"][
            "path"
        ]
        recorded["assignments"][0]["engine_params"]["api_key"] = "sk-engine"
        self._record(tmp_path, config_path, redact_secrets(recorded))
        monkeypatch.setenv("SIEVAL_TARGET_GPU", self.H100)
        resolve = AsyncMock(return_value=ResolveResult(plan=_fake_plan(), steps=()))
        # The config has no task; say one still infers with `m`, so it launches.
        with (
            patch("sieval.cli.run.auto_resolve_plan", new=resolve),
            patch("sieval.cli.run._models_to_infer", new=AsyncMock(return_value={"m"})),
        ):
            launch, _ = await self._run(
                config_path, gpu=self._gpu(self.H100), resume=True
            )

        resolve.assert_awaited_once()
        assert launch.await_count == 1
        assert "<redacted>" not in str(launch.await_args_list)

    def _recorded_plans(
        self, tmp_path: Path, plan: dict | None = None
    ) -> _RecordedPlans:
        from sieval.cli.session import unwrap_proxies

        config_path = self._config(tmp_path)
        raw = plan if plan is not None else unwrap_proxies(_fake_plan())
        self._record(tmp_path, config_path, raw)
        return _RecordedPlans(tmp_path / "out", self.H100)

    def _model_config(self, tmp_path: Path) -> dict:
        return yaml.safe_load((tmp_path / "cfg.yaml").read_text())["models"]["m"]

    def test_a_recorded_plan_applies_only_to_the_same_serving(self, tmp_path: Path):
        recorded = self._recorded_plans(tmp_path)
        model = self._model_config(tmp_path)
        plan_for = recorded.plan_for

        assert plan_for("m", model, "instruct", False) == _fake_plan()
        assert plan_for("other", model, "instruct", False) is None
        assert plan_for("m", {**model, "path": "/elsewhere"}, "instruct", False) is None
        assert plan_for("m", model, "base", False) is None
        assert plan_for("m", model, "instruct", True) is None

    def test_a_serving_group_run_alone_reuses_its_recorded_recipe(self, tmp_path: Path):
        """A ``--serving-group`` job writes only ``infer_plans.<key>.yaml``;
        its ``recipes`` are the plans its scaled ones started from."""
        from sieval.cli.session import unwrap_proxies

        self._recorded_plans(tmp_path)
        out = tmp_path / "out"
        recipe = yaml.safe_load((out / "infer_plans.yaml").read_text())
        (out / "infer_plans.yaml").unlink()
        scaled = unwrap_proxies(_fake_plan())
        scaled["assignments"][0]["engine_params"]["context_length"] = 100000
        (out / "infer_plans.m@100000.yaml").write_text(
            yaml.safe_dump(
                {"gpu": self.H100, "models": {"m": scaled}, "recipes": recipe["models"]}
            )
        )

        recorded = _RecordedPlans(out, self.H100)
        model = self._model_config(tmp_path)
        assert recorded.plan_for("m", model, "instruct", False) == _fake_plan()

    def test_a_record_this_version_cannot_read_is_not_reused(self, tmp_path: Path):
        from sieval.cli.session import unwrap_proxies

        raw = unwrap_proxies(_fake_plan())
        raw["added_later"] = 1
        recorded = self._recorded_plans(tmp_path, raw)
        model = self._model_config(tmp_path)
        assert recorded.plan_for("m", model, "instruct", False) is None

    @pytest.mark.parametrize(
        "damage", ["no_plans_file", "no_models", "no_config", "bad_config"]
    )
    def test_an_incomplete_record_reuses_nothing(self, tmp_path: Path, damage: str):
        self._recorded_plans(tmp_path)
        out = tmp_path / "out"
        if damage == "no_plans_file":
            (out / "infer_plans.yaml").unlink()
        if damage == "no_models":
            (out / "infer_plans.yaml").write_text(yaml.safe_dump({"gpu": self.H100}))
        if damage == "no_config":
            (out / "effective_config.yaml").unlink()
        if damage == "bad_config":
            # A model the session's type resolution refuses.
            (out / "effective_config.yaml").write_text(
                yaml.safe_dump({"models": {"m": "not a mapping"}, "tasks": {}})
            )
        recorded = _RecordedPlans(out, self.H100)
        model = self._model_config(tmp_path)
        assert recorded.plan_for("m", model, "instruct", False) is None

    @pytest.mark.parametrize(
        ("resume", "expected"),
        [
            (True, "out"),
            (False, None),
            (None, None),
            ("session", "session"),
            ("session/task", "session"),
        ],
    )
    def test_the_recorded_session_dir(
        self, tmp_path: Path, resume: bool | str | None, expected: str | None
    ):
        from sieval.cli.run import _recorded_session_dir

        (tmp_path / "session" / "task").mkdir(parents=True)
        (tmp_path / "session" / "task" / "meta.json").write_text("{}")
        target = tmp_path / resume if isinstance(resume, str) else resume
        found = _recorded_session_dir(target, str(tmp_path / "out"))
        assert found == (tmp_path / expected if expected else None)
        # A plain resume without a result_dir has nothing to read.
        assert _recorded_session_dir(True, None) is None

    @pytest.mark.anyio
    @pytest.mark.parametrize("change", ["other_gpus", "fresh_run", "changed_model"])
    async def test_otherwise_the_plan_is_resolved_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
    ):
        from sieval.cli.session import unwrap_proxies

        config_path = self._config(tmp_path)
        self._record(tmp_path, config_path, unwrap_proxies(_fake_plan()))
        declared = self.H100
        if change == "other_gpus":
            declared = "NVIDIA A100-SXM4-80GB:8:81920"
        if change == "changed_model":
            cfg = yaml.safe_load(config_path.read_text())
            cfg["models"]["m"]["infer"] = {"backend": "vllm"}
            cfg["models"]["m"]["infer"]["overrides"] = {"tp": 8}
            config_path.write_text(yaml.safe_dump(cfg))
        monkeypatch.setenv("SIEVAL_TARGET_GPU", declared)
        resolve = AsyncMock(return_value=ResolveResult(plan=_fake_plan(), steps=()))
        resolve_yaml = AsyncMock(return_value=("m", _fake_plan(), {}))
        with (
            patch("sieval.cli.run.auto_resolve_plan", new=resolve),
            patch("sieval.cli.run.resolve_infer_config", new=resolve_yaml),
        ):
            await self._run(
                config_path,
                gpu=self._gpu(declared),
                resume=change != "fresh_run",
            )

        assert resolve.await_count + resolve_yaml.await_count == 1


def _sglang_plan(window: int) -> DeploymentPlan:
    return DeploymentPlan(
        checkpoint="/tmp/ckpt",
        backend="sglang",
        assignments=(
            RoleAssignment(
                role=WellKnownRole.FULL,
                devices=DeviceGroup(count=1, gpu_model="H100"),
                topology=ParallelTopology(tp=1, dp=1, pp=1),
                engine_params={"context_length": window},
            ),
        ),
    )


class TestServingGroups:
    """One engine per serving plan: a need beyond the native window gets its own."""

    @staticmethod
    def _config(tmp_path: Path) -> Path:
        config = {
            "result_dir": str(tmp_path / "out"),
            "models": {
                "m": {
                    "path": "/tmp/ckpt",
                    "context": {
                        "native": 32768,
                        "max": 131072,
                        "scaling": {"json_model_override_args": "{}"},
                    },
                }
            },
            "datasets": {"d": {"class": "GSM8KDataset"}},
            "tasks": {
                "short": {"class": "GSM8KZeroShotGenTask", "dataset": "d"},
                "long": {
                    "class": "GSM8KZeroShotGenTask",
                    "dataset": "d",
                    "infer_args": {"max_tokens": 95904},
                },
            },
        }
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(config))
        return path

    @staticmethod
    def _translator() -> MagicMock:
        translator = MagicMock()
        translator.translate.side_effect = lambda plan: [
            BackendCommand(
                cli_args=[
                    "sglang",
                    "--context-length",
                    str(plan.assignments[0].engine_params["context_length"]),
                ],
                backend="sglang",
                role="full",
            )
        ]
        return translator

    @pytest.mark.anyio
    async def test_groups_launch_one_after_another_with_their_own_sessions(
        self, tmp_path: Path
    ) -> None:
        from sieval.cli.run import _run_all

        events: list[str] = []

        async def launch(name: str, commands: list[BackendCommand], **_: object):
            events.append(f"launch {name} {commands[0].cli_args[-1]}")
            return [_fake_handle()], None

        async def cleanup(name: str, *_: object, **__: object) -> None:
            events.append(f"stop {name}")

        async def session(*_: object, **kwargs: object) -> dict[str, object]:
            tasks = cast(SessionServing, kwargs["serving"]).tasks or frozenset()
            events.append(f"session {sorted(tasks)}")
            return dict.fromkeys(tasks, "ok")

        prepare = MagicMock()
        prepare.return_value.deployment_plans = {}
        sessions = AsyncMock(side_effect=session)
        with (
            patch("sieval.cli.infer.lifecycle.HANDLE_DIR", tmp_path / "handles"),
            patch(
                "sieval.cli.run.auto_resolve_plan",
                new=AsyncMock(
                    return_value=ResolveResult(plan=_sglang_plan(32768), steps=())
                ),
            ),
            patch("sieval.cli.run.get_translator", return_value=self._translator()),
            patch("sieval.cli.run.launch_model", new=AsyncMock(side_effect=launch)),
            patch("sieval.cli.run.cleanup_model", new=AsyncMock(side_effect=cleanup)),
            patch("sieval.cli.run.check_launch_gpu", new=AsyncMock()),
            patch("sieval.cli.run._plan_gpu_text", new=AsyncMock(return_value=None)),
            patch(
                "sieval.cli.validation.prepare_prelaunch_reconciliation", new=prepare
            ),
            patch("sieval.cli.session.arun_session", new=sessions),
        ):
            reports = await _run_all(self._config(tmp_path))

        assert events == [
            "launch m 32768",
            "session ['short']",
            "stop m",
            "launch m@100000 100000",
            "session ['long']",
            "stop m@100000",
        ]
        assert reports == {"short": "ok", "long": "ok"}
        # The services check runs once; every group's plans reconcile first.
        assert [c.kwargs["check_services"] for c in prepare.call_args_list] == [
            True,
            False,
        ]
        long_call = sessions.await_args_list[1].kwargs
        assert long_call["serving"].group == "m@100000"
        assert long_call["serving"].infer_plans_file == "infer_plans.m@100000.yaml"
        params = long_call["infer_plans"]["m"]["assignments"][0]["engine_params"]
        assert params["json_model_override_args"] == "{}"
        # The recipe plan it scaled, so a resume of this group alone reuses it.
        recipe = long_call["serving"].recipes["m"]["assignments"][0]["engine_params"]
        assert recipe["context_length"] == 32768

    @pytest.mark.anyio
    async def test_a_root_with_every_task_scaled_starts_no_default_engine(
        self, tmp_path: Path
    ) -> None:
        from sieval.cli.run import _run_all

        config_path = self._config(tmp_path)
        cfg = yaml.safe_load(config_path.read_text())
        del cfg["tasks"]["short"]
        config_path.write_text(yaml.safe_dump(cfg))
        launch = AsyncMock(return_value=([_fake_handle()], None))
        prepare = MagicMock()
        prepare.return_value.deployment_plans = {}
        with (
            patch("sieval.cli.infer.lifecycle.HANDLE_DIR", tmp_path / "handles"),
            patch(
                "sieval.cli.run.auto_resolve_plan",
                new=AsyncMock(
                    return_value=ResolveResult(plan=_sglang_plan(32768), steps=())
                ),
            ),
            patch("sieval.cli.run.get_translator", return_value=self._translator()),
            patch("sieval.cli.run.launch_model", new=launch),
            patch("sieval.cli.run.cleanup_model", new=AsyncMock()),
            patch("sieval.cli.run.check_launch_gpu", new=AsyncMock()),
            patch("sieval.cli.run._plan_gpu_text", new=AsyncMock(return_value=None)),
            patch(
                "sieval.cli.validation.prepare_prelaunch_reconciliation", new=prepare
            ),
            patch("sieval.cli.session.arun_session", new=AsyncMock(return_value={})),
        ):
            await _run_all(config_path)

        assert [c.args[0] for c in launch.await_args_list] == ["m@100000"]

    @pytest.mark.anyio
    async def test_one_group_runs_alone_and_an_unknown_key_is_refused(
        self, tmp_path: Path
    ) -> None:
        from sieval.cli.run import _run_all

        launch = AsyncMock(return_value=([_fake_handle()], None))
        sessions = AsyncMock(return_value={})
        prepare = MagicMock()
        prepare.return_value.deployment_plans = {}
        with (
            patch("sieval.cli.infer.lifecycle.HANDLE_DIR", tmp_path / "handles"),
            patch(
                "sieval.cli.run.auto_resolve_plan",
                new=AsyncMock(
                    return_value=ResolveResult(plan=_sglang_plan(32768), steps=())
                ),
            ),
            patch("sieval.cli.run.get_translator", return_value=self._translator()),
            patch("sieval.cli.run.launch_model", new=launch),
            patch("sieval.cli.run.cleanup_model", new=AsyncMock()),
            patch("sieval.cli.run.check_launch_gpu", new=AsyncMock()),
            patch("sieval.cli.run._plan_gpu_text", new=AsyncMock(return_value=None)),
            patch(
                "sieval.cli.validation.prepare_prelaunch_reconciliation", new=prepare
            ),
            patch("sieval.cli.session.arun_session", new=sessions),
        ):
            await _run_all(self._config(tmp_path), serving_group="m@100000")
            with pytest.raises(ValueError, match="Unknown serving group 'nope'"):
                await _run_all(self._config(tmp_path), serving_group="nope")

        assert launch.await_count == 1
        assert launch.await_args is not None
        assert launch.await_args.args[0] == "m@100000"
        (call,) = sessions.await_args_list
        assert call.kwargs["serving"].tasks == {"long"}

    @pytest.mark.anyio
    async def test_dry_run_lists_each_group_with_its_engine(
        self, tmp_path: Path
    ) -> None:
        from sieval.cli.run import _run_dry_run

        plan_tasks = AsyncMock(return_value=({}, {}))
        with (
            patch(
                "sieval.cli.run.auto_resolve_plan",
                new=AsyncMock(
                    return_value=ResolveResult(plan=_sglang_plan(32768), steps=())
                ),
            ),
            patch("sieval.cli.run.get_translator", return_value=self._translator()),
            patch("sieval.cli.run._plan_gpu_text", new=AsyncMock(return_value=None)),
            patch(
                "sieval.cli.validation.run_dry_run",
                return_value={"checks": [], "n_errors": 0, "n_warnings": 0},
            ),
            patch("sieval.cli.run._plan_tasks", new=plan_tasks),
        ):
            result = await _run_dry_run(self._config(tmp_path))

        serving = cast(dict, result["serving"])
        assert list(serving) == ["default", "m@100000"]
        assert serving["default"]["tasks"] == ["short"]
        assert serving["m@100000"]["tasks"] == ["long"]
        assert serving["m@100000"]["windows"] == {"long": 100000}
        engine = serving["m@100000"]["engines"]["m@100000"]
        assert engine["model"] == "m"
        assert engine["window"] == 100000
        assert engine["launch"][0]["argv"][-1] == "100000"
        # A fresh run starts every group's engine.
        assert serving["m@100000"]["needs_inference"] is True
        check = cast(list, result["checks"])[0]
        assert check == {
            "name": "serving",
            "ok": True,
            "detail": "default: 1 task(s), m@100000: 1 task(s)",
        }
        assert plan_tasks.await_count == 2
        assert {
            frozenset(c.kwargs["serving"].tasks) for c in plan_tasks.await_args_list
        } == {frozenset({"short"}), frozenset({"long"})}

    @pytest.mark.anyio
    async def test_dry_run_fails_the_serving_check_on_a_shortfall(
        self, tmp_path: Path
    ) -> None:
        from sieval.cli.run import _run_dry_run

        config_path = self._config(tmp_path)
        cfg = yaml.safe_load(config_path.read_text())
        del cfg["models"]["m"]["context"]["max"]
        config_path.write_text(yaml.safe_dump(cfg))
        with (
            patch(
                "sieval.cli.run.auto_resolve_plan",
                new=AsyncMock(
                    return_value=ResolveResult(plan=_sglang_plan(32768), steps=())
                ),
            ),
            patch("sieval.cli.run.get_translator", return_value=self._translator()),
            patch("sieval.cli.run._plan_gpu_text", new=AsyncMock(return_value=None)),
            patch(
                "sieval.cli.validation.run_dry_run",
                return_value={"checks": [], "n_errors": 0, "n_warnings": 0},
            ),
        ):
            result = await _run_dry_run(config_path)

        (check,) = cast(list, result["checks"])
        assert check["name"] == "serving"
        assert check["ok"] is False
        assert "task 'long' needs 100000 tokens" in check["detail"]
        assert result["n_errors"] == 1
        assert (result["tasks"], result["serving"]) == (None, None)
