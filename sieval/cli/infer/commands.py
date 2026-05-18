"""
Typer app and subcommands for inference service management.

Provides the ``infer_app`` Typer instance with five subcommands:
start, list, show, stop, logs.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import contextlib
import dataclasses
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import anyio
import typer
from loguru import logger

from sieval.cli.output import CommandResult, OutputFormat, render
from sieval.core.utils.logging import configure_logging, log_user
from sieval.infer.backends import get_translator
from sieval.infer.backends.process import pid_alive
from sieval.infer.backends.translator import inject_user_env
from sieval.infer.config import (
    InferEnv,
    InferHandle,
    InferPhase,
    ParamValue,
)
from sieval.infer.deployer import DeployError, DeployTimeoutError
from sieval.infer.params import merge_params
from sieval.infer.topology.models import (
    ResolveResult,
    RoleAssignment,
)
from sieval.infer.topology.resolver import auto_resolve_plan
from sieval.infer.topology.validator import validate_plan

from .lifecycle import (
    HANDLE_DIR,
    HANDLE_EXTRA_KEYS,
    claim_handle,
    default_deployer,
    display_status,
    launch_model,
    load_env,
    load_handle,
    parse_phase,
    probe_and_sync,
)
from .recipe import ResolvedInferConfig, resolve_infer_config

infer_app = typer.Typer(
    name="infer",
    help="Manage inference services (start, list, show, stop, logs).",
)


@infer_app.callback()
def _infer_callback(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose (DEBUG) logging"),
    ] = False,
) -> None:
    """Shared pre-command hook: configure logging for all infer subcommands."""
    configure_logging(verbose)


def _derive_model_name(checkpoint: str) -> str:
    """Derive a short model name from a checkpoint path.

    Examples:
        /models/Qwen3-4B       -> qwen3-4b
        /models/Qwen3-4B-AWQ   -> qwen3-4b-awq
        Qwen/Qwen3-4B          -> qwen3-4b
    """
    name = Path(checkpoint).name
    return name.lower()


def _parse_engine_args(args: list[str]) -> dict[str, ParamValue]:
    """Parse extra engine arguments (after ``--``) into a param dict.

    Handles ``--flag value``, ``--flag=value``, and bare ``--flag`` (bool).
    Values are auto-coerced: int -> int, float -> float, else str.
    """
    params: dict[str, ParamValue] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if not arg.startswith("--"):
            logger.warning(
                "Ignoring unexpected positional argument {!r} "
                "(expected --flag or --flag=value)",
                arg,
            )
            i += 1
            continue

        if "=" in arg:
            key, _, raw_value = arg.partition("=")
            key = key.lstrip("-").replace("-", "_")
            params[key] = _coerce_value(raw_value)
        elif i + 1 < len(args) and not args[i + 1].startswith("--"):
            key = arg.lstrip("-").replace("-", "_")
            params[key] = _coerce_value(args[i + 1])
            i += 1
        else:
            # Bare flag -> bool True
            key = arg.lstrip("-").replace("-", "_")
            params[key] = True
        i += 1
    return params


def _coerce_value(raw: str) -> ParamValue:
    """Try bool, int, float, then str."""
    lower = raw.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


@infer_app.command(
    "start",
    context_settings={
        "allow_extra_args": True,
        "allow_interspersed_args": True,
    },
)
def infer_start(
    ctx: typer.Context,
    target: Annotated[
        str,
        typer.Argument(help="Checkpoint path (auto-resolve) or YAML config file"),
    ],
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Which model to serve (for YAML mode)",
        ),
    ] = None,
    backend: Annotated[
        str,
        typer.Option("--backend", "-b", help="Infer backend"),
    ] = "sglang",
    detach: Annotated[
        bool,
        typer.Option(
            "--detach",
            help="Return immediately after submission",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Only print the launch command"),
    ] = False,
    output: Annotated[
        OutputFormat,
        typer.Option("-o", "--output", help="Output format"),
    ] = OutputFormat.TEXT,
    timeout: Annotated[
        float,
        typer.Option("--timeout", help="Seconds to wait for ready"),
    ] = 300.0,
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Override handle name"),
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
) -> None:
    """Launch an inference service.\n
    1. Auto-resolve (recommended):\n
       sieval infer start /path/to/model\n
       Reads config.json, detects GPU, matches recipe, launches.\n
    2. YAML config:\n
       sieval infer start config.yaml\n
       Uses explicit infer config from the YAML file.\n
    Extra engine arguments can be passed after ``--``:\n
       sieval infer start /path/to/model -- --served-model-name my-model
    """
    # Parse extra engine arguments (everything after --)
    engine_overrides = _parse_engine_args(ctx.args) if ctx.args else {}

    target_path = Path(target)

    # Decide mode: YAML file vs checkpoint directory
    if target_path.suffix in (".yaml", ".yml") and target_path.is_file():
        # YAML mode
        async def _resolve_yaml() -> ResolvedInferConfig:
            return await resolve_infer_config(target_path, model)

        model_name, plan, user_env = anyio.run(_resolve_yaml)

        # Engine overrides from CLI: inject into first assignment
        if engine_overrides:
            a = plan.assignments[0]
            merged = merge_params(a.engine_params, engine_overrides)
            new_a = RoleAssignment(
                role=a.role,
                devices=a.devices,
                topology=a.topology,
                replicas=a.replicas,
                engine_params=merged,
            )
            # `replace` preserves every plan field set upstream; a
            # field-by-field rebuild silently drops new fields on `DeploymentPlan`.
            plan = dataclasses.replace(
                plan,
                assignments=(new_a,) + plan.assignments[1:],
            )
    else:
        # Auto-resolve mode: target is a checkpoint path
        async def _resolve() -> ResolveResult:
            return await auto_resolve_plan(
                target,
                backend=backend,
                overrides=engine_overrides or None,
            )

        resolve_result = anyio.run(_resolve)
        plan = resolve_result.plan
        model_name = name or _derive_model_name(target)
        user_env = {}  # auto-resolve inherits shell env via Popen

    # CLI force-on leg; YAML leg is handled by resolve_infer_config.
    if deterministic and not plan.deterministic:
        plan = dataclasses.replace(plan, deterministic=True)

    # Validate plan before translation
    errors = validate_plan(plan)
    if errors:
        result = CommandResult(command="infer.start", ok=False, error="; ".join(errors))
        render(result, output)
        raise typer.Exit(1)

    # Translate plan → commands
    translator = get_translator(plan.backend)
    commands = translator.translate(plan)

    # Inject user-specified environment variables from YAML config
    inject_user_env(commands, user_env)

    if dry_run:
        cmd = commands[0]
        info: dict[str, object] = {
            "model": model_name,
            "command": cmd.cli_args,
            "health_check": cmd.health_url,
        }
        if cmd.env:
            info["env"] = cmd.env
        dr_result = CommandResult(command="infer.dry_run", ok=True, data=info)
        render(dr_result, output)
        return

    # Guard: atomically claim the handle file so concurrent starts on
    # the same model_name are mutually exclusive.
    handle_path = HANDLE_DIR / f"{model_name}.json"

    def _reclaim(reason: str) -> None:
        """Remove existing handle and reclaim; raise on lost race.

        Note: there is a small TOCTOU window between ``unlink`` and
        ``claim_handle`` where another process could claim the name.
        The ``FileExistsError`` branch handles this gracefully.
        """
        handle_path.unlink(missing_ok=True)
        try:
            claim_handle(model_name, backend=plan.backend)
        except FileExistsError:
            result = CommandResult(
                command="infer.start",
                ok=False,
                error=(
                    f"Another process claimed {model_name!r} while we "
                    f"were cleaning up ({reason}). Retry or remove "
                    f"{handle_path} manually."
                ),
            )
            render(result, output)
            raise typer.Exit(1) from None

    def _try_claim() -> None:
        """Attempt to claim the handle; clean stale entries and retry once."""
        try:
            claim_handle(model_name, backend=plan.backend)
        except FileExistsError:
            # Handle file already exists — inspect it
            try:
                data = json.loads(handle_path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("Removing corrupted handle file for {}", model_name)
                _reclaim("corrupted handle")
                return

            phase = parse_phase(data)
            if phase == InferPhase.PENDING and "pid" in data:
                raw_pid = data["pid"]
            elif "handle_id" in data:
                raw_pid = data["handle_id"]
            else:
                raw_pid = None
            try:
                old_pid: int | None = int(raw_pid) if raw_pid is not None else None
            except (ValueError, TypeError):
                # handle_id is not a numeric PID (e.g. container ID) —
                # cannot check liveness, treat as stale
                old_pid = None

            if old_pid is not None and pid_alive(old_pid):
                if phase == InferPhase.PENDING:
                    msg = (
                        f"Model {model_name!r} is already starting "
                        f"(PID {old_pid}). Wait for it to finish or "
                        f"remove {handle_path}."
                    )
                else:
                    msg = (
                        f"Model {model_name!r} is already running "
                        f"(PID {old_pid}). "
                        f"Run 'sieval infer stop {model_name}' first."
                    )
                result = CommandResult(command="infer.start", ok=False, error=msg)
                render(result, output)
                raise typer.Exit(1) from None

            # Stale handle — process is dead, clean up and reclaim
            if old_pid is not None:
                logger.info(
                    "Cleaning stale handle for {} (PID {} no longer alive)",
                    model_name,
                    old_pid,
                )
            else:
                logger.info(
                    "Cleaning stale handle for {} (no PID recorded)",
                    model_name,
                )
            _reclaim("stale handle")

    _try_claim()

    last_status = None
    last_log_time = 0.0

    def _progress(elapsed: float, status_value: str) -> None:
        nonlocal last_status, last_log_time
        if sys.stderr.isatty():
            logger.opt(raw=True).log(
                "USER",
                f"\r\x1b[KWaiting for service... "
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
                    model_name,
                    status_value,
                    elapsed,
                )

    async def _launch() -> tuple[list[InferHandle], InferEnv | None]:
        return await launch_model(
            model_name,
            commands,
            backend=plan.backend,
            detach=detach,
            timeout=timeout,
            on_progress=None if detach else _progress,
            already_claimed=True,
        )

    try:
        handles, env = anyio.run(_launch)
    except KeyboardInterrupt:
        handle_path.unlink(missing_ok=True)
        sys.exit(130)
    except (DeployError, DeployTimeoutError) as exc:
        handle_path.unlink(missing_ok=True)
        result = CommandResult(command="infer.start", ok=False, error=str(exc))
        render(result, output)
        raise typer.Exit(1) from exc
    except BaseException:
        handle_path.unlink(missing_ok=True)
        raise

    if not detach and sys.stderr.isatty():
        logger.opt(raw=True).log("USER", "\n")

    handle = handles[0]
    start_data: dict[str, object] = {
        "model": model_name,
        "backend": handle.backend,
        "endpoint": handle.endpoint,
        "handle_id": handle.handle_id,
        "handle_path": str(handle_path),
        "metadata": dict(handle.metadata) if handle.metadata else {},
    }
    start_result = CommandResult(command="infer.start", ok=True, data=start_data)
    render(start_result, output)


@infer_app.command("list")
def infer_list(
    output: Annotated[
        OutputFormat,
        typer.Option("-o", "--output", help="Output format"),
    ] = OutputFormat.TEXT,
) -> None:
    """List all tracked inference services and their status."""
    if not HANDLE_DIR.exists():
        result = CommandResult(command="infer.list", ok=True, data=[])
        render(result, output)
        return

    handle_files = sorted(HANDLE_DIR.glob("*.json"))
    if not handle_files:
        result = CommandResult(command="infer.list", ok=True, data=[])
        render(result, output)
        return

    async def _collect_statuses() -> list[dict]:
        rows: list[dict] = []
        for handle_file in handle_files:
            mname = handle_file.stem
            try:
                data = json.loads(handle_file.read_text())

                phase_from_file = parse_phase(data)

                # Pending handle — process is still starting up
                if phase_from_file == InferPhase.PENDING:
                    starter_pid = data.get("pid")
                    try:
                        alive = starter_pid is not None and pid_alive(int(starter_pid))
                    except (ValueError, TypeError):
                        alive = False
                    if not alive:
                        handle_file.unlink(missing_ok=True)
                        continue
                    rows.append(
                        {
                            "model": mname,
                            "phase": "pending",
                            "status": "Pending",
                            "endpoint": None,
                            "backend": data.get("backend", ""),
                            "handle_id": data.get("handle_id", ""),
                            "conditions": data.get("conditions", {}),
                        }
                    )
                    continue

                # Stopping handle — process is being shut down.
                # Uses "handle_id" (deployed process PID), not "pid"
                # (starter PID used in the PENDING block above).
                if phase_from_file == InferPhase.STOPPING:
                    raw_pid = data.get("handle_id")
                    try:
                        alive = raw_pid is not None and pid_alive(int(raw_pid))
                    except (ValueError, TypeError):
                        alive = False
                    if not alive:
                        handle_file.unlink(missing_ok=True)
                        continue
                    endpoint = data.get("endpoint") or None
                    rows.append(
                        {
                            "model": mname,
                            "phase": "stopping",
                            "status": "Stopping",
                            "endpoint": endpoint,
                            "backend": data.get("backend", ""),
                            "handle_id": data.get("handle_id", ""),
                            "conditions": data.get("conditions", {}),
                        }
                    )
                    continue

                # Build InferHandle for deployer probe
                handle_data = {
                    k: v for k, v in data.items() if k not in HANDLE_EXTRA_KEYS
                }
                handle = InferHandle(**handle_data)
                phase, conditions = await probe_and_sync(handle, handle_file)

                status_display = display_status(phase, conditions)
                cond_dict = {
                    k: {"status": v.status, "reason": v.reason or ""}
                    for k, v in conditions.items()
                }
                rows.append(
                    {
                        "model": mname,
                        "phase": phase.value,
                        "status": status_display,
                        "endpoint": handle.endpoint,
                        "backend": handle.backend,
                        "handle_id": handle.handle_id,
                        "conditions": cond_dict,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "model": mname,
                        "phase": "error",
                        "status": "error",
                        "endpoint": None,
                        "backend": "",
                        "handle_id": "",
                        "conditions": {},
                        "error": str(exc),
                    }
                )
        return rows

    data = anyio.run(_collect_statuses)
    result = CommandResult(command="infer.list", ok=True, data=data)
    render(result, output)


@infer_app.command("show")
def infer_show(
    name: Annotated[
        str,
        typer.Argument(help="Model name"),
    ],
    output: Annotated[
        OutputFormat,
        typer.Option("-o", "--output", help="Output format"),
    ] = OutputFormat.TEXT,
) -> None:
    """Show detailed information about an inference service."""
    try:
        handle = load_handle(name)
    except typer.BadParameter as e:
        result = CommandResult(command="infer.show", ok=False, error=str(e))
        render(result, output)
        raise typer.Exit(1) from e

    handle_path = HANDLE_DIR / f"{name}.json"
    phase, conditions = anyio.run(probe_and_sync, handle, handle_path)

    cond_dict = {
        k: {"status": v.status, "reason": v.reason or ""} for k, v in conditions.items()
    }

    env = load_env(name)
    env_dict = None
    if env is not None:
        env_dict = asdict(env)

    data = {
        "model": name,
        "status": display_status(phase, conditions),
        "phase": phase.value,
        "backend": handle.backend,
        "endpoint": handle.endpoint,
        "handle_id": handle.handle_id,
        "conditions": cond_dict,
        "metadata": dict(handle.metadata) if handle.metadata else {},
        "env": env_dict,
    }
    result = CommandResult(command="infer.show", ok=True, data=data)
    render(result, output)


@infer_app.command("stop")
def infer_stop(
    name: Annotated[
        str,
        typer.Argument(help="Model name (as defined in YAML config)"),
    ],
    output: Annotated[
        OutputFormat,
        typer.Option("-o", "--output", help="Output format"),
    ] = OutputFormat.TEXT,
) -> None:
    """Stop an inference service by model name."""
    try:
        handle = load_handle(name)
    except typer.BadParameter as e:
        result = CommandResult(command="infer.stop", ok=False, error=str(e))
        render(result, output)
        raise typer.Exit(1) from e

    # Mark phase as stopping so concurrent `infer list` shows "Stopping"
    # instead of "NotReady" during the SIGTERM→exit window.
    handle_path = HANDLE_DIR / f"{name}.json"
    if handle_path.exists():
        tmp = handle_path.with_suffix(".tmp")
        try:
            data = json.loads(handle_path.read_text())
            data["phase"] = InferPhase.STOPPING.value
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(str(tmp), str(handle_path))
        except (json.JSONDecodeError, OSError):
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)

    anyio.run(default_deployer.delete, handle)

    # Verify the process actually stopped
    phase, conditions = anyio.run(probe_and_sync, handle, handle_path)
    if phase in (InferPhase.STOPPED, InferPhase.FAILED):
        if handle_path.exists():
            handle_path.unlink()
        result = CommandResult(
            command="infer.stop",
            ok=True,
            data={"model": name, "stopped": True, "phase": phase.value},
        )
    else:
        # ok=True: the stop *command* succeeded (signal sent).  The
        # process hasn't exited yet — callers check data["stopped"] to
        # distinguish "fully terminated" from "still shutting down".
        result = CommandResult(
            command="infer.stop",
            ok=True,
            data={
                "model": name,
                "stopped": False,
                "phase": phase.value,
                "handle_id": handle.handle_id,
            },
        )
    render(result, output)


@infer_app.command("logs")
def infer_logs(
    name: Annotated[
        str,
        typer.Argument(help="Model name"),
    ],
    tail: Annotated[
        int,
        typer.Option(
            "--tail",
            "-n",
            help="Number of trailing lines to show",
        ),
    ] = 50,
    follow: Annotated[
        bool,
        typer.Option("--follow", "-f", help="Follow log output"),
    ] = False,
) -> None:
    """Show engine logs for an inference service."""
    handle = load_handle(name)

    async def _print_logs() -> None:
        async for line in default_deployer.logs(handle, tail=tail, follow=follow):
            log_user("{}", line)

    with contextlib.suppress(KeyboardInterrupt):
        anyio.run(_print_logs)
