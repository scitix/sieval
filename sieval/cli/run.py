"""
sieval run — all-in-one: serve -> eval -> cleanup.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import dataclasses
import shlex
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, cast

import anyio
import typer
import yaml
from loguru import logger

from sieval.cli.infer import cleanup_model, launch_model, resolve_infer_config
from sieval.cli.leaderboard.session import resolve_deterministic, unwrap_proxies
from sieval.cli.output import (
    CommandResult,
    OutputFormat,
    cli_command,
    cli_error_message,
    render,
)
from sieval.cli.resolution import resolve_config_model_types
from sieval.core.models import Deployment
from sieval.core.utils.logging import configure_logging, log_user
from sieval.infer.backends import get_translator
from sieval.infer.backends.translator import BackendCommand, inject_user_env
from sieval.infer.config import InferHandle
from sieval.infer.deployer import DEFAULT_READY_TIMEOUT, LocalDeployer
from sieval.infer.recipes import capability_model_type
from sieval.infer.topology import DeploymentPlan
from sieval.infer.topology.resolver import auto_resolve_plan


def _needs_serve(model_config: dict) -> bool:
    """Determine whether a model config requires auto-serve.

    A model needs serve when it has a local checkpoint (``path`` or
    ``infer.checkpoint``) and no pre-existing ``api_base``.
    """
    if model_config.get("api_base"):
        return False
    if model_config.get("path"):
        return True
    infer_dict = model_config.get("infer")
    return infer_dict is not None and bool(infer_dict.get("checkpoint"))


async def _prepare_launch_batch(
    config_path: Path,
    config: dict[str, Any],
    *,
    effective_deterministic: bool,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, tuple[DeploymentPlan, list[BackendCommand]]],
]:
    """Resolve and translate every managed deployment without launching it."""

    models = config.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("models must be a mapping")
    model_type_resolution = resolve_config_model_types(config)
    plan_dicts: dict[str, dict[str, Any]] = {}
    prepared_launches: dict[str, tuple[DeploymentPlan, list[BackendCommand]]] = {}

    for model_name, model_config in models.items():
        if not isinstance(model_name, str) or not isinstance(model_config, dict):
            raise ValueError("models entries must map names to model configurations")
        if not _needs_serve(model_config):
            continue

        infer_dict = model_config.get("infer")
        if infer_dict is not None:
            _, plan, user_env = await resolve_infer_config(
                config_path,
                model_name,
                model_type_resolution=model_type_resolution,
            )
        else:
            checkpoint = model_config["path"]
            result = await auto_resolve_plan(
                checkpoint=checkpoint,
                capability=capability_model_type(
                    model_type_resolution.model_types_by_config[model_name]
                ),
            )
            plan = result.plan
            user_env = {}

        if effective_deterministic and not plan.deterministic:
            plan = dataclasses.replace(plan, deterministic=True)

        plan_dicts[model_name] = unwrap_proxies(plan)

        from sieval.infer.topology.validator import validate_plan

        errors = validate_plan(plan)
        if errors:
            raise RuntimeError(
                f"Invalid deployment plan for {model_name}: " + "; ".join(errors)
            )

        translator = get_translator(plan.backend)
        commands = translator.translate(plan)
        inject_user_env(commands, user_env)
        prepared_launches[model_name] = (plan, commands)

    return plan_dicts, prepared_launches


async def _run_dry_run(
    config_path: Path,
    *,
    resume: bool = False,
    model: str | None = None,
    result_dir: str | None = None,
    deterministic: bool | None = None,
) -> dict[str, object]:
    """Resolve the same managed batch and return the pure prelaunch plan."""

    from sieval.cli.validation import run_dry_run

    if not config_path.exists():
        return dict(
            cast(
                Mapping[str, object],
                run_dry_run(
                    config_path,
                    model_override=model,
                    resume=resume,
                    result_dir_override=result_dir,
                    deterministic_override=deterministic,
                    invocation=shlex.join(sys.argv),
                ),
            )
        )

    with open(config_path) as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError("evaluation config must be a mapping")
    effective_deterministic = resolve_deterministic(deterministic, loaded)
    plan_dicts, prepared_launches = await _prepare_launch_batch(
        config_path,
        loaded,
        effective_deterministic=effective_deterministic,
    )

    return dict(
        cast(
            Mapping[str, object],
            run_dry_run(
                config_path,
                model_override=model,
                resume=resume,
                result_dir_override=result_dir,
                deterministic_override=deterministic,
                infer_plans=plan_dicts or None,
                invocation=shlex.join(sys.argv),
                self_managed_endpoints=frozenset(prepared_launches),
            ),
        )
    )


async def _run_all(
    config_path: Path,
    verbose: bool = False,
    resume: bool = False,
    model: str | None = None,
    result_dir: str | None = None,
    deterministic: bool | None = None,
    ready_timeout: float = DEFAULT_READY_TIMEOUT,
) -> dict[str, object]:
    """Orchestrate: start services -> run eval -> stop services.

    Two triggers cause a model to be auto-served:
      1. Explicit ``infer`` section in the model config.
      2. Top-level ``path`` field without ``api_base`` (path-only shortcut).

    Returns:
        Report dict from arun_session (task_name → report string).
    """
    configure_logging(verbose)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Path-only models skip resolve_infer_config (which reads YAML
    # `deterministic: true`), so compute the effective flag here and apply
    # it uniformly to every plan below. Without this, YAML deterministic
    # would be lost for `path:`-only models even though the session layer
    # still honors it.
    effective_deterministic = resolve_deterministic(deterministic, config)

    launched: dict[str, list[InferHandle]] = {}
    plan_dicts: dict[str, dict[str, Any]] = {}
    prepared_launches: dict[str, tuple[DeploymentPlan, list[BackendCommand]]] = {}
    realized_deployments: dict[str, Deployment] = {}
    deployer = LocalDeployer()
    invocation = shlex.join(sys.argv)

    try:
        # Resolve the complete desired batch before starting any process.  The
        # dry-run entry point calls this exact helper and omits only launch and
        # post-launch verification.
        plan_dicts, prepared_launches = await _prepare_launch_batch(
            config_path,
            config,
            effective_deterministic=effective_deterministic,
        )

        # Task requirements, model declarations, and every managed desired
        # plan must reconcile as one batch before the first subprocess starts.
        # The final EvalSession intentionally repeats this pure check so its
        # execution setup never trusts orchestration-only state.
        from sieval.cli.leaderboard.session import arun_session
        from sieval.cli.validation import prepare_prelaunch_reconciliation

        prelaunch_result = prepare_prelaunch_reconciliation(
            config_path,
            model_override=model,
            resume=resume,
            result_dir_override=result_dir,
            deterministic_override=deterministic,
            infer_plans=plan_dicts or None,
            invocation=invocation,
            self_managed_endpoints=frozenset(prepared_launches),
        )
        launch_patches = {
            root_key: dict(deployment_plan.launch_patch)
            for root_key, deployment_plan in prelaunch_result.deployment_plans.items()
            if deployment_plan.launch_patch
        }
        if launch_patches:
            details = "; ".join(
                f"{root_key}: {patch_values!r}"
                for root_key, patch_values in sorted(launch_patches.items())
            )
            raise RuntimeError(
                "Pre-launch reconciliation produced engine launch parameters, "
                "but `sieval run` has no #47 launch-patch translator yet; "
                f"refusing to ignore them: {details}"
            )

        for model_name, (plan, commands) in prepared_launches.items():
            log_user("Starting inference for model: {}", model_name)

            last_status = None
            last_log_time = 0.0

            # Capture model_name in closure via default arg
            def _progress(
                elapsed: float,
                status_value: str,
                _name: str = model_name,
            ) -> None:
                nonlocal last_status, last_log_time
                if sys.stderr.isatty():
                    logger.opt(raw=True).log(
                        "USER",
                        f"\r\x1b[K[{_name}] Waiting... "
                        f"(elapsed {elapsed:.0f}s, "
                        f"status: {status_value})",
                    )
                else:
                    now = time.perf_counter()
                    status_changed = status_value != last_status
                    heartbeat_due = (now - last_log_time) >= 15.0
                    if status_changed or heartbeat_due:
                        last_status = status_value
                        last_log_time = now
                        log_user(
                            "[{}] Status: {} (elapsed {:.0f}s)",
                            _name,
                            status_value,
                            elapsed,
                        )

            # Claim → deploy → save handle (shared with infer start)
            try:
                new_handles, env = await launch_model(
                    model_name,
                    commands,
                    backend=plan.backend,
                    deployer=deployer,
                    on_progress=_progress,
                    timeout=ready_timeout,
                )
            except FileExistsError:
                msg = (
                    f"Model {model_name!r} is already claimed by another"
                    " process — check `sieval infer list`"
                )
                raise FileExistsError(msg) from None

            if sys.stderr.isatty():
                logger.opt(raw=True).log("USER", "\n")
            launched[model_name] = new_handles

            deployment = deployer.build_deployment(plan, new_handles)
            realized_deployments[model_name] = deployment

            if env is not None:
                log_user(
                    "  env: {} / CUDA {} / {} x {}",
                    env.framework or "unknown",
                    env.cuda_version or "?",
                    env.gpu_model or "?",
                    env.gpu_count,
                )

        # `self_managed_endpoints` scopes the best-effort warning away
        # from endpoints we launched ourselves.
        reports = await arun_session(
            config_path,
            model=model,
            resume=resume,
            result_dir=result_dir,
            deterministic=deterministic,
            self_managed_endpoints=frozenset(realized_deployments),
            infer_plans=plan_dicts or None,
            invocation=invocation,
            realized_deployments=realized_deployments or None,
        )
        return reports

    finally:
        # Stop all launched services and remove handle files
        for name, model_handles in launched.items():
            await cleanup_model(name, model_handles, deployer=deployer)


def register_run_command(app: typer.Typer) -> None:
    """Register the run command directly on the main app."""

    @app.command()
    @cli_command
    def run(
        config: Annotated[
            Path,
            typer.Argument(help="Path to evaluation YAML config"),
        ],
        model: Annotated[
            str | None,
            typer.Option(
                "--model",
                "-m",
                help="Override model name for all base models",
            ),
        ] = None,
        resume: Annotated[
            bool,
            typer.Option(
                "--resume",
                "-r",
                help="Enable auto-resume for all tasks",
            ),
        ] = False,
        result_dir: Annotated[
            str | None,
            typer.Option(
                "--result-dir",
                help="Override result directory",
            ),
        ] = None,
        deterministic: Annotated[
            bool | None,
            typer.Option(
                "--deterministic",
                help=(
                    "Force deterministic inference mode on. "
                    "Monotone: cannot disable a YAML-level `deterministic: true`."
                ),
            ),
        ] = None,
        ready_timeout: Annotated[
            float,
            typer.Option(
                "--ready-timeout",
                # Zero or negative makes the poll loop time out on its first
                # pass, so a bad argument reads as a broken engine.
                min=1.0,
                help=(
                    "Seconds to wait for each auto-served model to become "
                    "ready. Applies per model, not to the run as a whole."
                ),
            ),
        ] = DEFAULT_READY_TIMEOUT,
        verbose: Annotated[
            bool,
            typer.Option("--verbose", "-v", help="Verbose output"),
        ] = False,
        dry_run: Annotated[
            bool,
            typer.Option("--dry-run", help="Validate config without running"),
        ] = False,
        output: Annotated[
            OutputFormat,
            typer.Option("-o", "--output", help="Output format"),
        ] = OutputFormat.TEXT,
    ) -> None:
        """All-in-one: launch inference services, run evaluation, and cleanup."""
        if dry_run:
            configure_logging(verbose)

            async def _go_dry() -> dict[str, object]:
                return await _run_dry_run(
                    config,
                    resume=resume,
                    model=model,
                    result_dir=result_dir,
                    deterministic=deterministic,
                )

            try:
                dry_result = anyio.run(_go_dry)
            except Exception as exc:
                result = CommandResult(
                    command="run.dry_run",
                    ok=False,
                    error=cli_error_message(exc),
                )
                render(result, output)
                raise typer.Exit(1) from exc
            n_errors = dry_result.get("n_errors")
            if not isinstance(n_errors, int):
                raise RuntimeError("dry-run result omitted integer n_errors")
            result = CommandResult(
                command="run.dry_run",
                ok=n_errors == 0,
                data=dict(dry_result),
                error="Dry-run failed" if n_errors > 0 else None,
            )
            render(result, output)
            if not result.ok:
                raise typer.Exit(1)
            return

        if not config.exists():
            cmd_result = CommandResult(
                command="run", ok=False, error=f"Config file not found: {config}"
            )
            render(cmd_result, output)
            raise typer.Exit(1)

        async def _go() -> dict[str, object]:
            return await _run_all(
                config,
                verbose=verbose,
                resume=resume,
                model=model,
                result_dir=result_dir,
                deterministic=deterministic,
                ready_timeout=ready_timeout,
            )

        try:
            reports = anyio.run(_go)
        except KeyboardInterrupt:
            # Ctrl-C is not a command failure: exit 130 without a result
            # payload. Everything else is left to `@cli_command`.
            sys.exit(130)

        tasks_data = {
            task_name: {"report": report} for task_name, report in reports.items()
        }
        cmd_result = CommandResult(
            command="run",
            ok=True,
            data={"tasks": tasks_data},
        )
        render(cmd_result, output)
