"""
sieval run — all-in-one: serve -> eval -> cleanup.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import dataclasses
import shlex
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated, Any, cast

import anyio
import typer
import yaml
from loguru import logger

from sieval.cli._cli_model import (
    ApiBaseOption,
    CheckpointOption,
    CliModel,
    apply_cli_model,
)
from sieval.cli._resume_option import (
    ResumeCommand,
    ResumeFromOption,
    ResumeOption,
    cli_resume,
)
from sieval.cli.infer import (
    HandleHeldError,
    cleanup_model,
    launch_model,
    remove_run_handle_dir,
    resolve_infer_config,
    resolve_keep_logs,
    run_handle_dir,
)
from sieval.cli.output import (
    CommandResult,
    OutputFormat,
    cli_command,
    cli_error_message,
    render,
)
from sieval.cli.resolution import resolve_config_model_types
from sieval.cli.services import SERVICE_LOG_DIR
from sieval.cli.session import (
    config_resumes,
    holds_secret,
    read_persisted_config,
    redact_secrets,
    resolve_deterministic,
    unwrap_proxies,
)
from sieval.cli.signals import terminate_as_interrupt
from sieval.core.models import Deployment
from sieval.core.runners import RunPlan
from sieval.core.utils.logging import configure_logging, log_user
from sieval.infer.backends import get_translator
from sieval.infer.backends.translator import BackendCommand, inject_user_env
from sieval.infer.config import InferHandle
from sieval.infer.deployer import (
    DEFAULT_READY_TIMEOUT,
    LocalDeployer,
    unlaunched_deployment,
)
from sieval.infer.introspect import (
    check_launch_gpu,
    declared_target_gpu,
    detect_local_gpu,
    format_target_gpu,
)
from sieval.infer.recipes import capability_model_type
from sieval.infer.topology import DeploymentPlan, plan_from_mapping
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


def _refuse_explicit_endpoint(model_name: str, model_config: dict) -> None:
    """Refuse ``args.api_base`` on a model that ``sieval run`` would serve.

    Unlike a top-level ``api_base``, it does not turn auto-serve off.
    """
    args = model_config.get("args")
    if not isinstance(args, dict) or not args.get("api_base"):
        return
    infer_dict = model_config.get("infer")
    if isinstance(infer_dict, dict) and infer_dict.get("checkpoint"):
        source, serve_key = "infer.checkpoint", "infer"
    else:
        source, serve_key = "path", "path"
    raise ValueError(
        f"Model '{model_name}' sets both 'args.api_base' and '{source}'. "
        "Remove 'args.api_base' to evaluate the launched engine, or remove "
        f"'{serve_key}' to evaluate {args['api_base']!r}."
    )


def _recorded_session_dir(
    resume: bool | str | Path | None, result_dir: str | None
) -> Path | None:
    """The session ``result_dir`` a resume continues, whose plans it may reuse.

    ``--resume DIR`` names it, or names a task directory inside it.
    """
    if resume is True:
        return Path(result_dir) if result_dir else None
    if isinstance(resume, (str, Path)):
        recorded = Path(resume)
        return recorded.parent if (recorded / "meta.json").is_file() else recorded
    return None


class _RecordedPlans:
    """What a resumed session recorded about its served models, read-only.

    Its plans are reused only when ``SIEVAL_TARGET_GPU`` declares the GPUs they
    were resolved for, and only for a model whose serving config is unchanged:
    a recipe the installed SiEval resolves differently would otherwise change
    the inference identity and refuse the resume.
    """

    def __init__(self, session_dir: Path | None, gpu: str | None) -> None:
        self._plans: dict[str, Any] = {}
        self._models: dict[str, Any] = {}
        self._capabilities: dict[str, str] = {}
        if session_dir is None or gpu is None:
            return
        recorded = read_persisted_config(session_dir / "infer_plans.yaml")
        if recorded is None or recorded.get("gpu") != gpu:
            return
        plans = recorded.get("models")
        config = read_persisted_config(session_dir / "effective_config.yaml")
        if not isinstance(plans, dict) or config is None:
            return
        models = config.get("models")
        try:
            types = resolve_config_model_types(config).model_types_by_config
        except ValueError:
            return
        if isinstance(models, dict):
            self._plans, self._models = plans, models
            self._capabilities = {
                name: capability_model_type(kind) for name, kind in types.items()
            }

    def plan_for(
        self,
        model_name: str,
        model_config: dict[str, Any],
        capability: str,
        deterministic: bool,
    ) -> DeploymentPlan | None:
        """The recorded plan of *model_name*, when this run would serve it alike."""
        raw = self._plans.get(model_name)
        recorded_model = self._models.get(model_name)
        if not isinstance(raw, dict) or not isinstance(recorded_model, dict):
            return None
        # `infer_plans.yaml` is written redacted: a plan that held a secret
        # (an engine `api_key`) reads back holding the marker, and would
        # launch the engine with it.
        if holds_secret(raw):
            return None
        # Compared redacted, as `effective_config.yaml` records them.
        current = redact_secrets(model_config)
        if any(recorded_model.get(k) != current.get(k) for k in ("path", "infer")):
            return None
        if self._capabilities.get(model_name) != capability:
            return None
        try:
            plan = plan_from_mapping(raw)
        except ValueError:
            return None
        if plan.deterministic != deterministic:
            return None
        return plan


def _declared_gpu_text() -> str | None:
    """``SIEVAL_TARGET_GPU``, normalized; ``None`` when nothing is declared."""
    declared = declared_target_gpu()
    return format_target_gpu(declared) if declared is not None else None


async def _plan_gpu_text() -> str | None:
    """The GPUs this run's plans were resolved for, as ``SIEVAL_TARGET_GPU``
    spells them: the declared ones, else this node's."""
    gpu = declared_target_gpu() or await detect_local_gpu()
    return format_target_gpu(gpu) if gpu is not None else None


def _recorded_plans(
    resume: bool | str | Path | None, result_dir: str | None
) -> _RecordedPlans:
    return _RecordedPlans(
        _recorded_session_dir(resume, result_dir), _declared_gpu_text()
    )


async def _prepare_launch_batch(
    config_path: Path,
    config: dict[str, Any],
    *,
    effective_deterministic: bool,
    recorded: _RecordedPlans | None = None,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, tuple[DeploymentPlan, list[BackendCommand]]],
]:
    """Resolve and translate every managed deployment without launching it.

    A plan *recorded* by the session a resume continues is reused when it
    applies (:class:`_RecordedPlans`).
    """

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
        _refuse_explicit_endpoint(model_name, model_config)

        infer_dict = model_config.get("infer")
        capability = capability_model_type(
            model_type_resolution.model_types_by_config[model_name]
        )
        reused = (
            recorded.plan_for(
                model_name, model_config, capability, effective_deterministic
            )
            if recorded is not None
            else None
        )
        if reused is not None:
            logger.info("Reusing the recorded serving plan of {}", model_name)
            plan = reused
            user_env = {
                str(k): str(v) for k, v in ((infer_dict or {}).get("env") or {}).items()
            }
        elif infer_dict is not None:
            _, plan, user_env = await resolve_infer_config(
                config_path,
                model_name,
                model_type_resolution=model_type_resolution,
            )
        else:
            checkpoint = model_config["path"]
            result = await auto_resolve_plan(
                checkpoint=checkpoint, capability=capability
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


def _resumes(resume: bool | str | Path | None, config: Mapping[str, Any]) -> bool:
    """Whether some task may resume: ``--resume``, or a ``runner_config`` key."""
    return bool(resume) or config_resumes(config)


def _unlaunched(
    prepared_launches: Mapping[str, tuple[DeploymentPlan, list[BackendCommand]]],
    names: set[str],
) -> dict[str, Deployment]:
    return {
        name: unlaunched_deployment(*prepared_launches[name]) for name in sorted(names)
    }


async def _plan_tasks(
    config_path: Path,
    prepared_launches: Mapping[str, tuple[DeploymentPlan, list[BackendCommand]]],
    plan_dicts: dict[str, dict[str, Any]],
    *,
    model: str | None,
    resume: bool | str | Path | None,
    result_dir: str | None,
    deterministic: bool | None,
    invocation: str,
    cli_model: CliModel | None,
) -> tuple[dict[str, str], dict[str, RunPlan | Exception]]:
    """Each task's served model and resume plan, with no engine launched.

    The one call that turns resume planning into a launch decision.
    """
    from sieval.cli.session import aplan_session

    return await aplan_session(
        config_path,
        model=model,
        resume=resume,
        result_dir=result_dir,
        deterministic=deterministic,
        infer_plans=plan_dicts or None,
        invocation=invocation,
        self_managed_endpoints=frozenset(prepared_launches),
        unlaunched_deployments=_unlaunched(prepared_launches, set(prepared_launches)),
        cli_model=cli_model,
    )


def _models_to_launch(
    served: Iterable[str],
    task_models: Mapping[str, str],
    plans: Mapping[str, RunPlan | Exception],
) -> set[str]:
    """The *served* models some task still has samples to infer with.

    A task whose plan raises infers nothing: its run raises the same before
    its first sample.
    """
    return {
        task_models[task]
        for task, plan in plans.items()
        if _infers(plan) and task_models.get(task) in served
    }


def _infers(plan: RunPlan | Exception) -> bool:
    """Whether a task's run sends any sample to its model under test."""
    return not isinstance(plan, Exception) and plan["to_infer"] > 0


async def _models_to_infer(
    config_path: Path,
    config: Mapping[str, Any],
    prepared_launches: Mapping[str, tuple[DeploymentPlan, list[BackendCommand]]],
    plan_dicts: dict[str, dict[str, Any]],
    **session: Any,
) -> set[str]:
    """The served models to launch: all of them for a run where nothing
    resumes, which is not planned; else :func:`_models_to_launch`."""
    if not prepared_launches:
        return set()
    if not _resumes(session["resume"], config):
        return set(prepared_launches)
    task_models, plans = await _plan_tasks(
        config_path, prepared_launches, plan_dicts, **session
    )
    return _models_to_launch(prepared_launches, task_models, plans)


def _launch_json(command: BackendCommand) -> dict[str, object]:
    """One process a launch starts: its role, argv and the env it adds."""
    out: dict[str, object] = {"role": command.role, "argv": list(command.cli_args)}
    if command.env:
        out["env"] = dict(command.env)
    if command.working_dir:
        out["cwd"] = command.working_dir
    return out


def _plan_summary(plans: Mapping[str, RunPlan | Exception]) -> str:
    """One line for the ``resume_plan`` check: what the tasks would do."""
    ok = [p for p in plans.values() if not isinstance(p, Exception)]
    to_infer = sum(p["to_infer"] for p in ok)
    to_grade = sum(p["to_grade"] for p in ok)
    return f"{len(ok)} task(s): {to_infer} sample(s) to infer, {to_grade} to grade"


def _task_json(model: str | None, plan: RunPlan | Exception) -> dict[str, object]:
    """A task's entry in the dry-run report.

    ``resume_plan`` is the plan's ``outcomes``, the counts ``runs[].resume``
    records, and ``stages`` its stage comparison; both ``None`` for a fresh
    start, which has no recorded samples. ``needs_inference`` is
    :func:`_infers`, the test :func:`_models_to_launch` launches by.
    """
    if isinstance(plan, Exception):
        return {"model": model, "error": cli_error_message(plan)}
    return {
        "model": model,
        "start": plan["start"],
        "recorded": plan["recorded"],
        "samples": plan["samples"],
        "resume_plan": plan["outcomes"],
        "stages": plan["stages"],
        "new": plan["new"],
        "to_infer": plan["to_infer"],
        "to_grade": plan["to_grade"],
        "needs_inference": _infers(plan),
    }


async def _run_dry_run(
    config_path: Path,
    *,
    resume: bool | str | Path | None = None,
    model: str | None = None,
    result_dir: str | None = None,
    deterministic: bool | None = None,
    cli_model: CliModel | None = None,
) -> dict[str, object]:
    """Report what the run would do, launching no engine.

    The checks, then each served model's plan and launch argv, and each task's
    plan; see ``docs/guide/configuration.md``. Reads ``--resume`` directories in
    place and writes no result directory. Planning loads the datasets and
    starts each vendored service a task declares, as the run does.
    """

    from sieval.cli.validation import run_dry_run

    def checks(
        infer_plans: dict[str, dict[str, Any]] | None = None,
        self_managed_endpoints: frozenset[str] = frozenset(),
    ) -> dict[str, object]:
        return dict(
            cast(
                Mapping[str, object],
                run_dry_run(
                    config_path,
                    model_override=model,
                    resume=resume,
                    result_dir_override=result_dir,
                    deterministic_override=deterministic,
                    infer_plans=infer_plans,
                    invocation=shlex.join(sys.argv),
                    self_managed_endpoints=self_managed_endpoints,
                    cli_model=cli_model,
                ),
            )
        )

    def unplanned(result: dict[str, object]) -> dict[str, object]:
        # Every report key is present; ``None`` where the dry-run stopped first.
        for key in ("gpu", "models", "tasks", "needs_inference"):
            result.setdefault(key, None)
        return result

    if not config_path.exists():
        return unplanned(checks())

    with open(config_path) as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError("evaluation config must be a mapping")
    try:
        apply_cli_model(loaded, cli_model, model)
    except ValueError:
        return unplanned(checks())  # reported as the failed cli_model check
    effective_deterministic = resolve_deterministic(deterministic, loaded)
    plan_dicts, prepared_launches = await _prepare_launch_batch(
        config_path,
        loaded,
        effective_deterministic=effective_deterministic,
        recorded=_recorded_plans(resume, result_dir or loaded.get("result_dir")),
    )
    result = checks(plan_dicts or None, frozenset(prepared_launches))
    # ``needs_inference`` stays ``None`` until the tasks are planned.
    models: dict[str, dict[str, object]] = {
        name: {
            "plan": plan_dicts[name],
            "launch": [_launch_json(command) for command in commands],
            "needs_inference": None,
        }
        for name, (_, commands) in prepared_launches.items()
    }
    result["gpu"] = await _plan_gpu_text() if prepared_launches else None
    result["models"] = models
    if result["n_errors"]:
        return unplanned(result)

    # What the run would do: each task's plan, from the call the run makes.
    # Planning sets the session up as the run does: it loads every dataset
    # (downloading one not yet cached) and starts each vendored service a task
    # declares, writing nothing under the result directory.
    checks_list = cast(list[dict[str, object]], result["checks"])
    try:
        task_models, plans = await _plan_tasks(
            config_path,
            prepared_launches,
            plan_dicts,
            model=model,
            resume=resume,
            result_dir=result_dir,
            deterministic=deterministic,
            invocation=shlex.join(sys.argv),
            cli_model=cli_model,
        )
    except Exception as exc:
        # Raised before any task would start: the run fails the same way.
        detail = cli_error_message(exc)
        checks_list.append({"name": "resume_plan", "ok": False, "detail": detail})
        result["n_errors"] = cast(int, result["n_errors"]) + 1
        return unplanned(result)
    refused = {t: p for t, p in plans.items() if isinstance(p, Exception)}
    result["tasks"] = {
        task: _task_json(task_models.get(task), plan) for task, plan in plans.items()
    }
    if refused:
        checks_list.append(
            {
                "name": "resume_plan",
                "ok": False,
                "detail": "; ".join(
                    f"{task}: {cli_error_message(exc)}"
                    for task, exc in sorted(refused.items())
                ),
            }
        )
        result["n_errors"] = cast(int, result["n_errors"]) + 1
    else:
        checks_list.append(
            {"name": "resume_plan", "ok": True, "detail": _plan_summary(plans)}
        )
    # The launch decision `_models_to_infer` makes for the run.
    launches = (
        _models_to_launch(prepared_launches, task_models, plans)
        if _resumes(resume, loaded)
        else set(prepared_launches)
    )
    for name, entry in models.items():
        entry["needs_inference"] = name in launches
    # Whether an engine starts. A task inferring through an `api_base` model
    # needs no engine, so it does not count.
    result["needs_inference"] = bool(launches)
    return result


async def _run_all(
    config_path: Path,
    verbose: bool = False,
    resume: bool | str | Path | None = None,
    model: str | None = None,
    result_dir: str | None = None,
    deterministic: bool | None = None,
    ready_timeout: float = DEFAULT_READY_TIMEOUT,
    keep_logs: int | None = None,
    cli_model: CliModel | None = None,
) -> dict[str, object]:
    """Orchestrate: start services -> run eval -> stop services.

    Two triggers cause a model to be auto-served:
      1. Explicit ``infer`` section in the model config.
      2. Top-level ``path`` field without ``api_base`` (path-only shortcut).

    *cli_model* is added to the config before either of those is decided.

    Handles go under ``run-<pid>/``, removed when the run ends. Engine logs go
    to ``<result_dir>/logs/<model>/<role>.log`` when a result_dir is set,
    otherwise to ``~/.sieval/logs/<model>/`` pruned to *keep_logs* per role.

    Returns:
        Report dict from arun_session (task_name → report string).
    """
    configure_logging(verbose)

    with open(config_path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("evaluation config must be a mapping")
    apply_cli_model(config, cli_model, model)

    # Path-only models skip resolve_infer_config (which reads YAML
    # `deterministic: true`), so compute the effective flag here and apply
    # it uniformly to every plan below. Without this, YAML deterministic
    # would be lost for `path:`-only models even though the session layer
    # still honors it.
    effective_deterministic = resolve_deterministic(deterministic, config)

    launched: dict[str, list[InferHandle]] = {}
    handle_dir = run_handle_dir()
    effective_result_dir = result_dir or config.get("result_dir")
    # Retention applies only to ~/.sieval/logs, not to result_dir logs.
    keep = None if effective_result_dir else resolve_keep_logs(keep_logs)
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
            recorded=_recorded_plans(resume, effective_result_dir),
        )

        # Task requirements, model declarations, and every managed desired
        # plan must reconcile as one batch before the first subprocess starts.
        # The final EvalSession intentionally repeats this pure check so its
        # execution setup never trusts orchestration-only state.
        from sieval.cli.session import arun_session
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
            check_services=True,
            cli_model=cli_model,
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

        if SERVICE_LOG_DIR in prepared_launches:
            raise ValueError(
                f"Model name {SERVICE_LOG_DIR!r} is reserved for "
                f"<result_dir>/logs/{SERVICE_LOG_DIR}/; rename the model."
            )
        # Plan the resume first: an engine starts only for a model some task
        # still has samples to infer with.
        to_launch = await _models_to_infer(
            config_path,
            config,
            prepared_launches,
            plan_dicts,
            model=model,
            resume=resume,
            result_dir=result_dir,
            deterministic=deterministic,
            invocation=invocation,
            cli_model=cli_model,
        )
        if to_launch:
            await check_launch_gpu(required=True)
        for model_name in sorted(set(prepared_launches) - to_launch):
            log_user("No inference needed for model {}; not starting it", model_name)
        for model_name, (plan, commands) in prepared_launches.items():
            if model_name not in to_launch:
                continue
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

            # Claim → deploy → save handle (shared with infer start). The
            # handle lives under this process's own run-<pid>/ directory, so
            # an `infer start` of the same model name does not collide.
            try:
                new_handles, env = await launch_model(
                    model_name,
                    commands,
                    backend=plan.backend,
                    deployer=deployer,
                    handle_path=handle_dir / f"{model_name}.json",
                    log_dir=(
                        Path(effective_result_dir) / "logs" / model_name
                        if effective_result_dir
                        else None
                    ),
                    keep_logs=keep,
                    on_progress=_progress,
                    timeout=ready_timeout,
                    # A leftover run-<pid>/ from an earlier process with this PID.
                    replace_stale=True,
                )
            except HandleHeldError as held:
                msg = (
                    f"Model {model_name!r}: engine PID {held.pid} from an earlier "
                    f"process still holds {held.path}. Run `sieval infer stop "
                    f"{handle_dir.name}/{model_name}`, then retry."
                )
                raise HandleHeldError(held.path, held.data, held.pid, msg) from None

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
            self_managed_endpoints=frozenset(prepared_launches),
            infer_plans=plan_dicts or None,
            invocation=invocation,
            realized_deployments=realized_deployments or None,
            unlaunched_deployments=(
                _unlaunched(prepared_launches, set(prepared_launches) - to_launch)
                or None
            ),
            cli_model=cli_model,
            infer_gpu=await _plan_gpu_text() if plan_dicts else None,
        )
        return reports

    finally:
        # Stop all launched services, then remove this run's handle directory
        for name, model_handles in launched.items():
            await cleanup_model(
                name,
                model_handles,
                deployer=deployer,
                handle_path=handle_dir / f"{name}.json",
            )
        remove_run_handle_dir(handle_dir)


def register_run_command(app: typer.Typer) -> None:
    """Register the run command directly on the main app."""

    @app.command(cls=ResumeCommand)
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
        checkpoint: CheckpointOption = None,
        api_base: ApiBaseOption = None,
        resume: ResumeOption = None,
        result_dir: Annotated[
            str | None,
            typer.Option(
                "--result-dir",
                help="Override result directory",
            ),
        ] = None,
        resume_from: ResumeFromOption = None,
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
        keep_logs: Annotated[
            int | None,
            typer.Option(
                "--keep-logs",
                help=(
                    "Engine logs to keep per role in ~/.sieval/logs/<model>, "
                    "counting this launch; ignored with a result_dir. 0 deletes "
                    "at the end; negative keeps all. "
                    "Default: $SIEVAL_KEEP_LOGS, else 3."
                ),
            ),
        ] = None,
        verbose: Annotated[
            bool,
            typer.Option("--verbose", "-v", help="Verbose output"),
        ] = False,
        dry_run: Annotated[
            bool,
            typer.Option(
                "--dry-run",
                help=(
                    "Validate the config, then report each model's plan and each "
                    "task's resume plan, launching no engine. Loads the datasets "
                    "and starts any vendored service, as the run does."
                ),
            ),
        ] = False,
        output: Annotated[
            OutputFormat,
            typer.Option("-o", "--output", help="Output format"),
        ] = OutputFormat.TEXT,
    ) -> None:
        """All-in-one: launch inference services, run evaluation, and cleanup."""
        configure_logging(verbose)
        resumed = cli_resume(resume, resume_from)
        cli_model = CliModel.from_options(
            checkpoint=checkpoint, api_base=api_base, model=model
        )
        if dry_run:

            async def _go_dry() -> dict[str, object]:
                return await _run_dry_run(
                    config,
                    resume=resumed,
                    model=model,
                    result_dir=result_dir,
                    deterministic=deterministic,
                    cli_model=cli_model,
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
                resume=resumed,
                model=model,
                result_dir=result_dir,
                deterministic=deterministic,
                ready_timeout=ready_timeout,
                keep_logs=keep_logs,
                cli_model=cli_model,
            )

        try:
            with terminate_as_interrupt() as terminated:
                reports = anyio.run(_go)
        except KeyboardInterrupt:
            # Ctrl-C/SIGTERM is not a command failure: exit 130/143 without a
            # result payload. Everything else is left to `@cli_command`.
            sys.exit(terminated.exit_code)

        tasks_data = {
            task_name: {"report": report} for task_name, report in reports.items()
        }
        cmd_result = CommandResult(
            command="run",
            ok=True,
            data={"tasks": tasks_data},
        )
        render(cmd_result, output)
