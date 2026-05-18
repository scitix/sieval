"""
sieval run — all-in-one: serve -> eval -> cleanup.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import dataclasses
import shlex
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import anyio
import typer
import yaml
from loguru import logger

from sieval.cli.infer import cleanup_model, launch_model, resolve_infer_config
from sieval.cli.leaderboard.session import resolve_deterministic, unwrap_proxies
from sieval.cli.output import CommandResult, OutputFormat, cli_error_message, render
from sieval.core.utils.logging import configure_logging, log_user
from sieval.infer.backends import get_translator
from sieval.infer.backends.translator import inject_user_env
from sieval.infer.config import InferHandle
from sieval.infer.deployer import LocalDeployer
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


async def _run_all(
    config_path: Path,
    verbose: bool = False,
    resume: bool = False,
    model: str | None = None,
    result_dir: str | None = None,
    deterministic: bool | None = None,
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

    models = config.get("models", {})
    launched: dict[str, list[InferHandle]] = {}
    endpoint_map: dict[str, str] = {}
    plan_dicts: dict[str, dict[str, Any]] = {}
    deployer = LocalDeployer()

    try:
        for model_name, model_config in models.items():
            if not _needs_serve(model_config):
                # Malformed infer sections are flagged by schema validation.
                continue

            log_user("Starting inference for model: {}", model_name)

            # Resolve: explicit infer section or path-only auto-resolve
            infer_dict = model_config.get("infer")
            if infer_dict is not None:
                _, plan, user_env = await resolve_infer_config(
                    config_path,
                    model_name,
                )
            else:
                # Path-only mode: auto-resolve from checkpoint
                checkpoint = model_config["path"]
                result = await auto_resolve_plan(
                    checkpoint=checkpoint,
                )
                plan = result.plan
                user_env = {}  # path-only mode has no YAML env section

            # Stamp effective (YAML ∪ CLI) deterministic onto the plan.
            # resolve_infer_config handles the YAML leg for `infer:` models;
            # path-only models reach here without it, so we re-apply.
            if effective_deterministic and not plan.deterministic:
                plan = dataclasses.replace(plan, deterministic=True)

            plan_dicts[model_name] = unwrap_proxies(plan)

            # Validate plan before translation
            from sieval.infer.topology.validator import validate_plan

            errors = validate_plan(plan)
            if errors:
                raise RuntimeError(
                    f"Invalid deployment plan for {model_name}: " + "; ".join(errors)
                )

            # Translate plan → commands
            translator = get_translator(plan.backend)
            commands = translator.translate(plan)

            # Inject user-specified environment variables from YAML config
            inject_user_env(commands, user_env)

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

            # Use first handle's endpoint
            if new_handles and new_handles[0].endpoint:
                endpoint_map[model_name] = new_handles[0].endpoint

            if env is not None:
                log_user(
                    "  env: {} / CUDA {} / {} x {}",
                    env.framework or "unknown",
                    env.cuda_version or "?",
                    env.gpu_model or "?",
                    env.gpu_count,
                )

        from sieval.cli.leaderboard.session import arun_session

        # `self_managed_endpoints` scopes the best-effort warning away
        # from endpoints we launched ourselves.
        reports = await arun_session(
            config_path,
            model=model,
            resume=resume,
            result_dir=result_dir,
            deterministic=deterministic,
            self_managed_endpoints=frozenset(endpoint_map.keys()),
            endpoint_map=endpoint_map or None,
            infer_plans=plan_dicts or None,
            invocation=shlex.join(sys.argv),
        )
        return reports

    finally:
        # Stop all launched services and remove handle files
        for name, model_handles in launched.items():
            await cleanup_model(name, model_handles, deployer=deployer)


def register_run_command(app: typer.Typer) -> None:
    """Register the run command directly on the main app."""

    @app.command()
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
            from sieval.cli.validation import run_dry_run

            configure_logging(verbose)
            dry_result = run_dry_run(config)
            result = CommandResult(
                command="run.dry_run",
                ok=dry_result["n_errors"] == 0,
                data=dict(dry_result),
                error="Dry-run failed" if dry_result["n_errors"] > 0 else None,
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
            )

        try:
            reports = anyio.run(_go)
        except KeyboardInterrupt:
            sys.exit(130)
        except Exception as e:
            cmd_result = CommandResult(
                command="run", ok=False, error=cli_error_message(e)
            )
            render(cmd_result, output)
            raise typer.Exit(1) from e

        tasks_data = {
            task_name: {"report": report} for task_name, report in reports.items()
        }
        cmd_result = CommandResult(
            command="run",
            ok=True,
            data={"tasks": tasks_data},
        )
        render(cmd_result, output)
