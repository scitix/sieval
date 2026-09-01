"""Tests for sieval.cli.run — run command (all-in-one).

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import json
import re
from collections.abc import Callable
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from sieval.cli import app
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
        assert "auto_resume=True" not in error

    def test_run_help_shows_ready_timeout(self):
        """`sieval run` exposes the readiness budget it has always enforced."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--ready-timeout" in re.sub(r"\s+", " ", plain)

    def test_ready_timeout_option_reaches_run_all(self, tmp_path: Path):
        """The typer option must arrive at ``_run_all``, not stop at the parser.

        ``test_ready_timeout_reaches_the_deploy_call`` starts *at* ``_run_all``
        and ``test_run_help_shows_ready_timeout`` stops at ``--help``; between
        them sits the hand-off in ``register_run_command``. Dropping that one
        line restores the original defect — an option the user can type that
        never reaches the deployer — while leaving both neighbours green.
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
        """A zero/negative budget is a bad argument, not an engine failure.

        Without ``min=``, the poll loop accepts it and raises
        ``DeployTimeoutError`` on its first pass, which reads as a broken
        engine. Typer must refuse it before anything is launched.
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
        """One spelling works across both commands.

        ``run`` names the budget ``--ready-timeout``; ``infer start`` shipped
        it as ``--timeout``. Both names resolve to the same parameter here, so
        neither spelling is wrong depending on which command you reached for.
        """
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
        ):
            result = await _run_dry_run(
                config_path,
                resume=True,
                model="override/model",
                result_dir="result-override",
                deterministic=True,
            )

        assert result == dry_result
        prepare.assert_awaited_once()
        assert prepare.call_args.kwargs["effective_deterministic"] is True
        assert validate.call_args.kwargs["infer_plans"] == plan_dicts
        assert validate.call_args.kwargs["model_override"] == "override/model"
        assert validate.call_args.kwargs["resume"] is True
        assert validate.call_args.kwargs["result_dir_override"] == "result-override"
        assert validate.call_args.kwargs["deterministic_override"] is True
        assert validate.call_args.kwargs["self_managed_endpoints"] == {"model_a"}

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
                "sieval.cli.leaderboard.session.arun_session",
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
                "sieval.cli.leaderboard.session.arun_session",
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
                "sieval.cli.leaderboard.session.arun_session",
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
                "sieval.cli.leaderboard.session.arun_session",
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
                "sieval.cli.leaderboard.session.arun_session",
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
                "sieval.cli.leaderboard.session.arun_session",
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
                "sieval.cli.leaderboard.session.arun_session",
                new=AsyncMock(return_value={}),
            ),
        ):
            await _run_all(config_path=config_path, ready_timeout=1234.0)

        launch.assert_awaited_once()
        await_args = launch.await_args
        assert await_args is not None
        assert await_args.kwargs["timeout"] == 1234.0


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
                "sieval.cli.leaderboard.session.arun_session",
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
                "sieval.cli.leaderboard.session.arun_session",
                new=AsyncMock(side_effect=RuntimeError("evaluation failed")),
            ),
            pytest.raises(RuntimeError, match="evaluation failed"),
        ):
            await _run_all(config_path)

        cleanup.assert_awaited_once_with("model_a", [handle], deployer=ANY)

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


class TestEffectiveConfigRerunSafety:
    """End-to-end: `sieval run` produces an effective_config.yaml without
    baked-in api_base/api_key/auto-filled name. Rerun via `sieval run` would
    see the same raw YAML (no api_base) and re-launch services correctly."""

    @pytest.mark.anyio
    async def test_persisted_config_has_no_injected_api_base(self, tmp_path: Path):
        from sieval.cli.leaderboard.session import EvalSession
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
        from sieval.cli.leaderboard.session import EvalSession
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
