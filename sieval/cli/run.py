"""
sieval run — all-in-one: serve -> eval -> cleanup.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import dataclasses
import functools
import shlex
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from pathlib import Path
from typing import Annotated, Any, cast

import anyio
import typer
import yaml
from anyio.abc import TaskGroup
from loguru import logger

from sieval.cli._cli_model import (
    ApiBaseOption,
    CheckpointOption,
    CliModel,
    apply_cli_model,
)
from sieval.cli._resume_option import (
    ResumeCommand,
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
from sieval.cli.lifecycle import (
    Lifecycle,
    LifecycleError,
    Placement,
    check_iteration_barriers,
    check_lifecycles_launched,
    effective_barriers,
    model_lifecycles,
    phases_json,
    place,
    placed_commands,
    plan_phases,
    role_models,
    root_model,
    runs_before_stop,
    session_stop_after,
    visible_devices,
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
from sieval.cli.serving import (
    DEFAULT_GROUP,
    ServingGroup,
    SessionServing,
    plan_serving_groups,
    served_window,
)
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
from sieval.core.runners import EngineLoss, RunPlan
from sieval.core.runners.barriers import BARRIER_STAGES
from sieval.core.utils.logging import configure_logging, log_user
from sieval.infer.backends import get_translator
from sieval.infer.backends.translator import BackendCommand, inject_user_env
from sieval.infer.config import InferHandle
from sieval.infer.deployer import (
    DEFAULT_READY_TIMEOUT,
    ENGINE_POLL_INTERVAL,
    EngineExit,
    LocalDeployer,
    unlaunched_deployment,
)
from sieval.infer.introspect import (
    GPUInfo,
    check_launch_gpu,
    declared_target_gpu,
    detect_local_gpu,
    format_target_gpu,
)
from sieval.infer.params import merge_params
from sieval.infer.recipes import capability_model_type
from sieval.infer.topology import DeploymentPlan, plan_from_mapping
from sieval.infer.topology.memory import MemoryFit, engine_fit
from sieval.infer.topology.models import TP_KEYS
from sieval.infer.topology.resolver import auto_resolve_plan

#: A launched engine was lost while samples needed it. Not a refusal (3/4/5):
#: samples ran, and a resume retries the ones it failed.
ENGINE_LOST_EXIT_CODE = 6

# How long a stage that failed on a launched engine waits for the watch to see
# that engine lost: a connection drops before the next poll sees the exit.
_ENGINE_LOSS_SETTLE = 3 * ENGINE_POLL_INTERVAL


class EngineLostError(RuntimeError):
    """An engine ``sieval run`` launched was lost while samples needed it.

    *lost*: how each engine was lost. *failed*: samples failed per task.
    *skipped*: serving groups not started. *reports*: the reports written.
    """

    def __init__(
        self,
        lost: Mapping[str, EngineExit],
        failed: Mapping[str, int],
        skipped: list[str],
        reports: Mapping[str, object],
    ) -> None:
        self.lost = dict(lost)
        self.failed = dict(failed)
        self.skipped = list(skipped)
        self.reports = dict(reports)
        total = sum(self.failed.values())
        lines = [
            f"Engine {name} was lost mid-run: {engine_exit.summary()}."
            for name, engine_exit in sorted(self.lost.items())
        ]
        lines.append(
            f"{total} sample(s) failed with the engine ("
            + ", ".join(f"{task}: {n}" for task, n in sorted(self.failed.items()))
            + "); `sieval run --resume` retries them."
        )
        if self.skipped:
            lines.append("Serving groups not started: " + ", ".join(self.skipped))
        super().__init__("\n".join(lines) + "".join(e.log for e in self.lost.values()))

    def to_json(self) -> dict[str, object]:
        """The ``data`` of the failed ``run`` result."""
        return {
            "outcome": "engine_died",
            "engines": {
                name: {
                    "role": e.role,
                    "pid": e.pid,
                    "exit_code": e.exit_code,
                    "unhealthy_for": e.unhealthy_for,
                    "reason": e.reason or None,
                    "log": e.log.strip(),
                }
                for name, e in sorted(self.lost.items())
            },
            "failed": dict(sorted(self.failed.items())),
            "skipped_groups": self.skipped,
            "tasks": {name: {"report": r} for name, r in self.reports.items()},
        }


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
    the inference identity and refuse the resume. Each serving group's
    ``infer_plans[.<key>].yaml`` is read, a group's ``recipes`` standing for
    the plans its changed ones (scaled, or given input-scoring headroom)
    started from.
    """

    def __init__(self, session_dir: Path | None, gpu: str | None) -> None:
        self._plans: dict[str, Any] = {}
        self._models: dict[str, Any] = {}
        self._capabilities: dict[str, str] = {}
        if session_dir is None or gpu is None:
            return
        plans: dict[str, Any] = {}
        for path in [
            session_dir / "infer_plans.yaml",
            *sorted(session_dir.glob("infer_plans.*.yaml")),
        ]:
            recorded = read_persisted_config(path)
            if recorded is None or recorded.get("gpu") != gpu:
                continue
            for key in ("recipes", "models"):
                found = recorded.get(key)
                if isinstance(found, dict):
                    plans = {**found, **plans}
        config = read_persisted_config(session_dir / "effective_config.yaml")
        if not plans or config is None:
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


async def _plan_gpu() -> GPUInfo | None:
    """The GPUs this run's plans are resolved for: the declared ones, else
    this node's."""
    return declared_target_gpu() or await detect_local_gpu()


async def _plan_gpu_text() -> str | None:
    """:func:`_plan_gpu`, as ``SIEVAL_TARGET_GPU`` spells it."""
    gpu = await _plan_gpu()
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
        prepared_launches[model_name] = (plan, _translate(model_name, plan, user_env))

    return plan_dicts, prepared_launches


def _translate(
    model_name: str, plan: DeploymentPlan, user_env: dict[str, str]
) -> list[BackendCommand]:
    """Validate *plan* and translate it into its launch commands."""
    from sieval.infer.topology.validator import validate_plan

    errors = validate_plan(plan)
    if errors:
        raise RuntimeError(
            f"Invalid deployment plan for {model_name}: " + "; ".join(errors)
        )
    commands = get_translator(plan.backend).translate(plan)
    inject_user_env(commands, user_env)
    return commands


def _user_env(model_config: Mapping[str, Any]) -> dict[str, str]:
    """The ``infer.env`` a model's launch commands carry."""
    infer_dict = model_config.get("infer") or {}
    return {str(k): str(v) for k, v in (infer_dict.get("env") or {}).items()}


@dataclasses.dataclass(frozen=True)
class _GroupLaunch:
    """One serving group, ready to launch: what its session binds and runs.

    *prepared* and *plan_dicts* hold every served root, this group's plans over
    the default group's, so a task outside the group still binds; *own* are the
    roots this group launches. A root's engine is named *names*[root]: its
    model name in the default group, the group key otherwise. *recipes* are
    the recipe plans of the roots whose plan this group changed, recorded for
    a resume.
    """

    group: ServingGroup
    prepared: dict[str, tuple[DeploymentPlan, list[BackendCommand]]]
    plan_dicts: dict[str, dict[str, Any]]
    own: frozenset[str]
    names: dict[str, str]
    recipes: dict[str, dict[str, Any]]

    @property
    def key(self) -> str:
        return self.group.key

    def serving(self) -> SessionServing:
        """What this group's session runs and records. The default group
        holds every task no other group took, so a config with one group runs
        all its tasks and writes ``infer_plans.yaml`` as before."""
        return SessionServing(
            tasks=frozenset(self.group.tasks),
            group=self.key,
            infer_plans_file=(
                "infer_plans.yaml"
                if self.key == DEFAULT_GROUP
                else f"infer_plans.{self.key}.yaml"
            ),
            recipes=self.recipes,
        )


def _serving_groups(
    config: Mapping[str, Any],
    plan_dicts: Mapping[str, dict[str, Any]],
    prepared_launches: Mapping[str, tuple[DeploymentPlan, list[BackendCommand]]],
    gpu: GPUInfo | None = None,
) -> list[_GroupLaunch]:
    """:func:`plan_serving_groups` for *gpu*, each group with its launches
    translated; an unchanged plan keeps what :func:`_prepare_launch_batch`
    resolved, a changed one records its recipe in ``recipes``."""
    models = config.get("models") or {}
    groups = plan_serving_groups(
        config,
        {name: plan for name, (plan, _) in prepared_launches.items()},
        gpu.memory_mib if gpu is not None else None,
    )

    def entry(
        name: str, plan: DeploymentPlan
    ) -> tuple[tuple[DeploymentPlan, list[BackendCommand]], dict[str, Any]]:
        base, commands = prepared_launches[name]
        if plan == base:
            return (plan, commands), plan_dicts[name]
        user_env = _user_env(models.get(name) or {})
        return (plan, _translate(name, plan, user_env)), unwrap_proxies(plan)

    default = {name: entry(name, plan) for name, plan in groups[0].plans.items()}
    out: list[_GroupLaunch] = []
    for group in groups:
        entries = dict(default)
        names = {name: name for name in default}
        recipes: dict[str, dict[str, Any]] = {}
        if group.key == DEFAULT_GROUP:
            recipes = {
                name: plan_dicts[name]
                for name, plan in group.plans.items()
                if plan != prepared_launches[name][0]
            }
        else:
            for name, plan in group.plans.items():
                entries[name] = entry(name, plan)
                names[name] = group.key
                recipes[name] = plan_dicts[name]
        out.append(
            _GroupLaunch(
                group=group,
                prepared={name: launch for name, (launch, _) in entries.items()},
                plan_dicts={name: raw for name, (_, raw) in entries.items()},
                own=group.launches,
                names=names,
                recipes=recipes,
            )
        )
    return out


@dataclasses.dataclass(frozen=True)
class _StagedPlan:
    """The run's barriers, phases and engine placements, decided before launch.

    *barriers* maps each effective barrier stage to what implied it; *stop*
    is ``runner_config.stop_after``. *lifecycles* and *placements* are by root
    model; *roles* maps each task to the roots it names as ``grader`` or
    ``extractor``.
    """

    barriers: dict[str, list[str]]
    stop: str | None
    lifecycles: dict[str, Lifecycle]
    phases: list[Any]
    placements: dict[str, Placement]
    roles: dict[str, set[str]]

    @property
    def stages(self) -> tuple[str, ...]:
        return tuple(self.barriers)

    def starts(self, root: str) -> str | None:
        life = self.lifecycles.get(root)
        return None if life is None else life.start

    def stops(self, root: str) -> str | None:
        life = self.lifecycles.get(root)
        return None if life is None else life.stop

    def runs(self, root: str) -> bool:
        """Whether *root* has a phase in this run: one starting at or after
        ``stop_after`` never does."""
        return runs_before_stop(self.lifecycles.get(root), self.stop)

    def lifecycles_json(self, roots: Iterable[str]) -> dict[str, dict[str, str]]:
        return {
            root: self.lifecycles[root].to_json()
            for root in sorted(roots)
            if root in self.lifecycles
        }


def _staged_plan(
    config: Mapping[str, Any],
    prepared_launches: Mapping[str, tuple[DeploymentPlan, list[BackendCommand]]],
    gpu: GPUInfo | None,
) -> _StagedPlan:
    """Barriers, phases and placements for *config*; refuses, before launch, a
    malformed lifecycle, one on a model the run does not launch, a barrier an
    iterating task cannot pass, and models alive together that do not fit."""
    lifecycles = model_lifecycles(config)
    check_lifecycles_launched(lifecycles, prepared_launches)
    barriers = effective_barriers(config, lifecycles)
    check_iteration_barriers(config, barriers)
    stop = session_stop_after(config)
    phases = plan_phases(tuple(barriers), lifecycles, prepared_launches, stop)
    # `nvidia-smi` lists every GPU on the node; an inherited
    # CUDA_VISIBLE_DEVICES is all this run may use.
    visible = visible_devices()
    gpu_count = gpu.count if gpu is not None else None
    if gpu_count is not None and visible is not None:
        gpu_count = min(gpu_count, len(visible))
    placements = place(
        phases,
        {name: plan.total_gpus for name, (plan, _) in prepared_launches.items()},
        gpu_count,
        {name: len(commands) for name, (_, commands) in prepared_launches.items()},
    )
    roles = {
        task: {root_model(config, name) for name in names}
        for task, names in role_models(config).items()
    }
    return _StagedPlan(barriers, stop, lifecycles, phases, placements, roles)


def _placed_groups(
    groups: list[_GroupLaunch], staged: _StagedPlan
) -> list[_GroupLaunch]:
    """*groups* with each engine on its placement's devices and port, and each
    group launching the roots its tasks name as a role too."""
    visible = visible_devices()
    out = []
    for group in groups:
        prepared = {
            name: (
                plan,
                placed_commands(commands, staged.placements.get(name), visible),
            )
            for name, (plan, commands) in group.prepared.items()
        }
        roles = set().union(
            *(staged.roles.get(task, set()) for task in group.group.tasks)
        )
        out.append(
            dataclasses.replace(
                group,
                prepared=prepared,
                own=frozenset(group.own | (roles & set(prepared))),
            )
        )
    return out


def _select_group(groups: list[_GroupLaunch], key: str | None) -> list[_GroupLaunch]:
    """The groups a run executes: all of them, or the one *key* names."""
    if key is None:
        return groups
    for group in groups:
        if group.key == key:
            return [group]
    known = ", ".join(repr(group.key) for group in groups)
    raise ValueError(f"Unknown serving group {key!r}; this config has {known}")


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
    serving: SessionServing | None = None,
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
        serving=serving,
    )


def _models_to_launch(
    served: Iterable[str],
    task_models: Mapping[str, str],
    plans: Mapping[str, RunPlan | Exception],
    roles: Mapping[str, set[str]] | None = None,
) -> set[str]:
    """The *served* models some task still has work for: its model under test
    while it has samples to infer, each model it names in *roles* (``grader``,
    ``extractor``) while it has samples to grade.

    A task whose plan raises infers nothing: its run raises the same before
    its first sample.
    """
    served = set(served)
    out = {
        task_models[task]
        for task, plan in plans.items()
        if _infers(plan) and task_models.get(task) in served
    }
    for task, plan in plans.items():
        if isinstance(plan, Exception) or not (plan["to_grade"] or plan["to_infer"]):
            continue
        out |= (roles or {}).get(task, set()) & served
    return out


def _infers(plan: RunPlan | Exception) -> bool:
    """Whether a task's run sends any sample to its model under test."""
    return not isinstance(plan, Exception) and plan["to_infer"] > 0


async def _models_to_infer(
    config_path: Path,
    config: Mapping[str, Any],
    prepared_launches: Mapping[str, tuple[DeploymentPlan, list[BackendCommand]]],
    plan_dicts: dict[str, dict[str, Any]],
    roles: Mapping[str, set[str]] | None = None,
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
    return _models_to_launch(prepared_launches, task_models, plans, roles)


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


def _task_json(
    model: str | None,
    plan: RunPlan | Exception,
    metrics: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    """A task's entry in the dry-run report.

    ``resume_plan`` is the plan's ``outcomes``, the counts ``runs[].resume``
    records, and ``stages`` its stage comparison; both ``None`` for a fresh
    start, which has no recorded samples. ``needs_inference`` is
    :func:`_infers`, the test :func:`_models_to_launch` launches by.
    ``metrics`` is :func:`~sieval.cli.validation.task_report_metrics_json`,
    kept beside a refusal.
    """
    if isinstance(plan, Exception):
        return {"model": model, "error": cli_error_message(plan), "metrics": metrics}
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
        "metrics": metrics,
    }


async def _run_dry_run(
    config_path: Path,
    *,
    resume: bool | str | Path | None = None,
    model: str | None = None,
    result_dir: str | None = None,
    deterministic: bool | None = None,
    cli_model: CliModel | None = None,
    serving_group: str | None = None,
) -> dict[str, object]:
    """Report what the run would do, launching no engine.

    The checks, then each served model's plan and launch argv, each serving
    group (*serving_group* only, when given) and each task's plan; see
    ``docs/guide/configuration.md``. Reads ``--resume`` directories in
    place and writes no result directory. Planning loads the datasets and
    starts each vendored service a task declares, as the run does.
    """

    from sieval.cli.validation import run_dry_run, task_report_metrics_json

    def checks(
        infer_plans: dict[str, dict[str, Any]] | None = None,
        self_managed_endpoints: frozenset[str] = frozenset(),
        serving: SessionServing | None = None,
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
                    serving=serving,
                ),
            )
        )

    def unplanned(result: dict[str, object]) -> dict[str, object]:
        # Every report key is present; ``None`` where the dry-run stopped first.
        for key in (
            "gpu",
            "models",
            "serving",
            "tasks",
            "needs_inference",
            "barriers",
            "phases",
            "launches",
        ):
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
    batch_plans, prepared_launches = await _prepare_launch_batch(
        config_path,
        loaded,
        effective_deterministic=effective_deterministic,
        recorded=_recorded_plans(resume, result_dir or loaded.get("result_dir")),
    )
    groups: list[_GroupLaunch] = []
    serving_error: Exception | None = None
    staged: _StagedPlan | None = None
    gpu = await _plan_gpu() if prepared_launches else None
    try:
        staged = _staged_plan(loaded, prepared_launches, gpu)
        groups = _placed_groups(
            _select_group(
                _serving_groups(loaded, batch_plans, prepared_launches, gpu),
                serving_group,
            ),
            staged,
        )
    except LifecycleError as exc:
        serving_error = exc
        staged = None
    except ValueError as exc:
        serving_error = exc
    # The plans the first group launches: the recipe plans, unless a shortfall
    # left no group to read them from.
    first = groups[0] if groups else None
    plan_dicts = first.plan_dicts if first is not None else batch_plans
    prepared = first.prepared if first is not None else dict(prepared_launches)
    result = checks(
        plan_dicts or None,
        frozenset(prepared),
        first.serving() if first is not None else None,
    )
    # ``needs_inference`` stays ``None`` until the tasks are planned.
    models: dict[str, dict[str, object]] = {
        name: {
            "plan": plan_dicts[name],
            "launch": [_launch_json(command) for command in commands],
            "needs_inference": None,
        }
        for name, (_, commands) in prepared.items()
    }
    result["gpu"] = format_target_gpu(gpu) if gpu is not None else None
    result["models"] = models
    checks_list = cast(list[dict[str, object]], result["checks"])
    if staged is not None:
        # Every barrier the session waits at, and what implied each.
        result["barriers"] = [
            {"stage": stage, "implied_by": implied}
            for stage, implied in staged.barriers.items()
        ]
        result["phases"] = phases_json(
            staged.phases, staged.lifecycles, staged.placements
        )
        for name, entry in models.items():
            life = staged.lifecycles.get(name)
            entry["lifecycle"] = life.to_json() if life is not None else None
    if serving_error is not None:
        checks_list.append(
            {
                "name": (
                    "lifecycle"
                    if isinstance(serving_error, LifecycleError)
                    else "serving"
                ),
                "ok": False,
                "detail": cli_error_message(serving_error),
            }
        )
        result["n_errors"] = cast(int, result["n_errors"]) + 1
    sizing: list[str] = []
    serving = {group.key: _serving_json(group, loaded, gpu, sizing) for group in groups}
    result["serving"] = serving or None
    if serving_error is None and (
        len(groups) > 1 or serving_group is not None or sizing
    ):
        serving_check: dict[str, object] = {
            "name": "serving",
            "ok": True,
            "detail": ", ".join(
                f"{group.key}: {len(group.group.tasks)} task(s)" for group in groups
            ),
        }
        if sizing:
            serving_check["warnings"] = sizing
            result["n_warnings"] = cast(int, result["n_warnings"]) + len(sizing)
        checks_list.append(serving_check)
    if result["n_errors"]:
        return unplanned(result)

    # What the run would do: each task's plan, from the call the run makes.
    # Planning sets the session up as the run does: it loads every dataset
    # (downloading one not yet cached) and starts each vendored service a task
    # declares, writing nothing under the result directory.
    task_models: dict[str, str] = {}
    plans: dict[str, RunPlan | Exception] = {}
    launches_by_group: dict[str, set[str]] = {}
    for group in groups:
        try:
            group_models, group_plans = await _plan_tasks(
                config_path,
                group.prepared,
                group.plan_dicts,
                model=model,
                resume=resume,
                result_dir=result_dir,
                deterministic=deterministic,
                invocation=shlex.join(sys.argv),
                cli_model=cli_model,
                serving=group.serving(),
            )
        except Exception as exc:
            # Raised before any task would start: the run fails the same way.
            detail = cli_error_message(exc)
            checks_list.append({"name": "resume_plan", "ok": False, "detail": detail})
            result["n_errors"] = cast(int, result["n_errors"]) + 1
            return unplanned(result)
        task_models |= group_models
        plans |= group_plans
        # The launch decision `_models_to_infer` makes for the run.
        roles = staged.roles if staged is not None else {}
        launches_by_group[group.key] = {
            root
            for root in group.own
            & (
                _models_to_launch(group.prepared, group_models, group_plans, roles)
                if _resumes(resume, loaded)
                else set(group.prepared)
            )
            if staged is None or staged.runs(root)
        }
    refused = {t: p for t, p in plans.items() if isinstance(p, Exception)}
    metrics: dict[str, dict[str, list[str]] | None] = {}
    malformed: dict[str, TypeError] = {}
    for task in plans:
        try:
            metrics[task] = task_report_metrics_json(loaded, task)
        except TypeError as exc:
            metrics[task], malformed[task] = None, exc
    result["tasks"] = {
        task: _task_json(task_models.get(task), plan, metrics[task])
        for task, plan in plans.items()
    }
    if malformed:
        checks_list.append(
            {
                "name": "report_metrics",
                "ok": False,
                "detail": "; ".join(f"{t}: {e}" for t, e in sorted(malformed.items())),
            }
        )
        result["n_errors"] = cast(int, result["n_errors"]) + 1
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
    first_launches = launches_by_group.get(first.key, set()) if first else set()
    for name, entry in models.items():
        entry["needs_inference"] = name in first_launches
    for key, entry in serving.items():
        entry["needs_inference"] = bool(launches_by_group.get(key))
    # Whether a model under test starts: a judge or extractor alone does not
    # count. A task inferring through an `api_base` model needs no engine.
    candidates = set(task_models.values())
    graders = (
        set().union(*staged.roles.values()) - candidates
        if staged is not None
        else set()
    )
    result["needs_inference"] = any(
        launched - graders for launched in launches_by_group.values()
    )
    # Every engine the run starts, judges included: what a caller deciding
    # whether the run needs GPUs at all reads.
    result["launches"] = sorted(set().union(*launches_by_group.values()))
    return result


def _states_tp(model_config: Mapping[str, Any]) -> bool:
    """Whether *model_config*'s ``infer.overrides`` sets tp, which sizing keeps."""
    overrides = (model_config.get("infer") or {}).get("overrides") or {}
    return any(key in TP_KEYS for key in merge_params(overrides))


def _engine_fits(
    plan: DeploymentPlan, gpu: GPUInfo | None
) -> list[tuple[str, MemoryFit]] | None:
    """Each role of *plan* on one of *gpu*'s GPUs, with the params it launches
    with; ``None`` when the GPU memory or the checkpoint is unknown."""
    if gpu is None:
        return None
    fits: list[tuple[str, MemoryFit]] = []
    for assignment in plan.assignments:
        fit = engine_fit(
            checkpoint=plan.checkpoint,
            backend=plan.backend,
            tp=assignment.topology.tp,
            params=dict(assignment.engine_params),
            gpu_memory_mib=gpu.memory_mib,
        )
        if fit is None:
            return None
        fits.append((assignment.role, fit))
    return fits


def _serving_json(
    group: _GroupLaunch,
    config: Mapping[str, Any],
    gpu: GPUInfo | None,
    warnings: list[str],
) -> dict[str, object]:
    """A serving group's entry in the dry-run report: the tasks it runs, each
    one's served window, and the engines it launches (with any headroom).

    Each engine's ``memory`` is how each role fits its GPUs, ``None`` when
    unknown; a fit with little KV room appends the warning
    :func:`~sieval.infer.topology.memory.size_tp` logs to *warnings*.
    """
    models = config.get("models") or {}
    engines: dict[str, dict[str, object]] = {}
    for name, (plan, commands) in group.prepared.items():
        if name not in group.own:
            continue
        engine_name = group.names[name]
        engine: dict[str, object] = {
            "model": name,
            "window": served_window(plan, None),
            "launch": [_launch_json(command) for command in commands],
        }
        headroom = group.group.headroom.get(name)
        if headroom is not None:
            engine["headroom"] = headroom.to_json()
        fits = _engine_fits(plan, gpu)
        source = "stated" if _states_tp(models.get(name) or {}) else "sized"
        engine["memory"] = (
            None
            if fits is None
            else [{"role": role, **fit.to_json(source)} for role, fit in fits]
        )
        for role, fit in fits or ():
            warning = fit.low_room_warning()
            if warning is not None:
                where = engine_name if len(fits or ()) == 1 else f"{engine_name}.{role}"
                warnings.append(f"engine {where}: {warning}")
        engines[engine_name] = engine
    return {
        "tasks": list(group.group.tasks),
        "windows": {
            task: window.window for task, window in group.group.windows.items()
        },
        "engines": engines,
        "needs_inference": None,
    }


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
    serving_group: str | None = None,
) -> dict[str, object]:
    """Orchestrate: start services -> run eval -> stop services.

    Two triggers cause a model to be auto-served:
      1. Explicit ``infer`` section in the model config.
      2. Top-level ``path`` field without ``api_base`` (path-only shortcut).

    *cli_model* is added to the config before either of those is decided.

    Serving groups run one after another, each group's engines stopped before
    the next start; *serving_group* runs one. A scaled group's engine is named
    by its key.

    Handles go under ``run-<pid>/``, removed when the run ends. Engine logs go
    to ``<result_dir>/logs/<engine>/<role>.log`` when a result_dir is set,
    otherwise to ``~/.sieval/logs/<engine>/`` pruned to *keep_logs* per role.

    A launched engine lost while samples needed it (:func:`_session_watching`)
    lets its group finish and write reports, starts no later group, and raises
    :class:`EngineLostError`.

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

        # Shortfalls, malformed lifecycles and models alive together that do
        # not fit are refused before the first subprocess starts.
        gpu = await _plan_gpu() if prepared_launches else None
        staged = _staged_plan(config, prepared_launches, gpu)
        groups = _placed_groups(
            _select_group(
                _serving_groups(config, plan_dicts, prepared_launches, gpu),
                serving_group,
            ),
            staged,
        )
        if SERVICE_LOG_DIR in {n for g in groups for n in g.names.values()}:
            raise ValueError(
                f"Engine name {SERVICE_LOG_DIR!r} is reserved for "
                f"<result_dir>/logs/{SERVICE_LOG_DIR}/; rename the model."
            )

        # Task requirements, model declarations, and every managed desired
        # plan must reconcile as one batch before the first subprocess starts.
        # The final EvalSession intentionally repeats this pure check so its
        # execution setup never trusts orchestration-only state.
        from sieval.cli.validation import prepare_prelaunch_reconciliation

        for index, group in enumerate(groups):
            prelaunch_result = prepare_prelaunch_reconciliation(
                config_path,
                model_override=model,
                resume=resume,
                result_dir_override=result_dir,
                deterministic_override=deterministic,
                infer_plans=group.plan_dicts or None,
                invocation=invocation,
                self_managed_endpoints=frozenset(group.prepared),
                # Each covers every task, whichever group runs it: once.
                check_services=index == 0,
                check_python_requires=index == 0,
                cli_model=cli_model,
                serving=group.serving(),
            )
            launch_patches = {
                root_key: dict(deployment_plan.launch_patch)
                for root_key, deployment_plan in (
                    prelaunch_result.deployment_plans.items()
                )
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

        engines = _Engines(
            deployer=deployer,
            launched=launched,
            handle_dir=handle_dir,
            result_dir=effective_result_dir,
            keep_logs=keep,
            ready_timeout=ready_timeout,
        )
        # Models that start at a barrier launch once, after every serving group
        # has reached it: the groups first run up to the earliest such barrier,
        # then each continues from it with those models already up.
        spanning = sorted(
            {
                staged.starts(root)
                for root in prepared_launches
                if staged.starts(root) is not None and staged.runs(root)
            },
            key=BARRIER_STAGES.index,
        )
        segments: list[tuple[str | None, bool]] = [(None, False)]
        if spanning and len(groups) > 1:
            if not effective_result_dir:
                raise ValueError(
                    f"A model starting at the barrier on {spanning[0]!r} serves "
                    "every serving group, so the groups run up to it and then "
                    "resume from it: set result_dir"
                )
            segments = [(spanning[0], False), (None, True)]
        reports: dict[str, object] = {}
        for segment_stop, continuing in segments:
            last_segment = segment_stop is None
            for index, group in enumerate(groups):
                if len(groups) > 1:
                    log_user(
                        "Serving group {}: {}",
                        group.key,
                        ", ".join(group.group.tasks),
                    )
                lost, failed = await _run_group(
                    config_path,
                    config,
                    group,
                    staged,
                    engines,
                    resume=True if continuing else resume,
                    stop_override=segment_stop,
                    # A model that starts at a barrier outlives one group's
                    # run only while another group still needs it.
                    keep_spanning=last_segment
                    and index < len(groups) - 1
                    or not last_segment,
                    spanning_segment=not continuing and segment_stop is not None,
                    model=model,
                    result_dir=result_dir,
                    deterministic=deterministic,
                    invocation=invocation,
                    cli_model=cli_model,
                    reports=reports,
                )
                if lost:
                    raise EngineLostError(
                        lost,
                        failed,
                        [g.key for g in groups[index + 1 :]],
                        reports,
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


class _Watches:
    """The engines a session watches, by root model. A lifecycle starts and
    stops engines mid-session: :meth:`stop` ends a watch before a deliberate
    stop, so the stop is not a loss."""

    def __init__(self, engine_loss: EngineLoss, deployer: LocalDeployer) -> None:
        self._loss = engine_loss
        self._deployer = deployer
        self._tg: TaskGroup | None = None
        self._scopes: dict[str, tuple[anyio.CancelScope, anyio.Event]] = {}
        self.lost: dict[str, EngineExit] = {}

    def start(self, root: str, handles: list[InferHandle]) -> None:
        if self._tg is None or root in self._scopes:
            return
        scope, done = anyio.CancelScope(), anyio.Event()
        self._scopes[root] = (scope, done)
        self._tg.start_soon(self._watch, root, handles, scope, done)

    async def stop(self, root: str) -> None:
        entry = self._scopes.pop(root, None)
        if entry is not None:
            entry[0].cancel()
            await entry[1].wait()

    async def _watch(
        self,
        root: str,
        handles: list[InferHandle],
        scope: anyio.CancelScope,
        done: anyio.Event,
    ) -> None:
        try:
            with scope:
                engine_exit = await self._deployer.watch(handles)
                summary = engine_exit.summary()
                self.lost[root] = engine_exit
                logger.error("Engine {} lost: {}", root, summary)
                self._loss.mark_lost(root, f"engine {root} lost: {summary}")
        except Exception as exc:
            # A broken watch must not take the run down with it.
            logger.warning("Stopped watching engine {}: {!r}", root, exc)
        finally:
            done.set()


async def _session_watching(
    session: Callable[[], Awaitable[dict[str, Any]]],
    engines: Mapping[str, list[InferHandle]],
    engine_loss: EngineLoss,
    *,
    deployer: LocalDeployer,
    reports: dict[str, object],
    watches: _Watches | None = None,
) -> dict[str, EngineExit]:
    """Run *session* into *reports* while watching *engines* (by root model),
    and any *watches* starts while it runs.

    A lost engine is marked in *engine_loss*. Returns the lost engines some
    sample failed for; a loss after the last sample costs nothing. A session
    error after such a loss is logged and reported as the loss.
    """
    watches = watches or _Watches(engine_loss, deployer)

    # Held, not raised inside the task group, which would wrap it in a group.
    error: BaseException | None = None
    async with anyio.create_task_group() as tg:
        watches._tg = tg
        for root, handles in engines.items():
            watches.start(root, handles)
        try:
            reports |= await session()
        except (Exception, KeyboardInterrupt) as exc:
            error = exc
        finally:
            watches._tg = None
            tg.cancel_scope.cancel()
    lost = dict(watches.lost) if engine_loss.failed else {}
    if error is not None and (not lost or not isinstance(error, Exception)):
        raise error
    if error is not None:
        logger.error("The session also raised after the engine was lost: {}", error)
    return lost


@dataclasses.dataclass
class _Engines:
    """The engines this ``sieval run`` holds, by engine name, and what
    launching one takes."""

    deployer: LocalDeployer
    launched: dict[str, list[InferHandle]]
    handle_dir: Path
    result_dir: str | None
    keep_logs: int | None
    ready_timeout: float
    deployments: dict[str, Deployment] = dataclasses.field(default_factory=dict)

    async def start(
        self, name: str, plan: DeploymentPlan, commands: list[BackendCommand]
    ) -> Deployment:
        if name in self.launched:
            return self.deployments[name]
        self.deployments[name] = await _launch(
            name,
            plan,
            commands,
            deployer=self.deployer,
            launched=self.launched,
            handle_dir=self.handle_dir,
            result_dir=self.result_dir,
            keep_logs=self.keep_logs,
            ready_timeout=self.ready_timeout,
        )
        return self.deployments[name]

    async def stop(self, name: str) -> None:
        handles = self.launched.pop(name, None)
        self.deployments.pop(name, None)
        if handles is None:
            return
        log_user("Stopping inference for model: {}", name)
        await cleanup_model(
            name,
            handles,
            deployer=self.deployer,
            handle_path=self.handle_dir / f"{name}.json",
        )


async def _run_group(
    config_path: Path,
    config: Mapping[str, Any],
    group: _GroupLaunch,
    staged: _StagedPlan,
    engines: _Engines,
    *,
    resume: bool | str | Path | None,
    stop_override: str | None,
    keep_spanning: bool,
    spanning_segment: bool,
    reports: dict[str, object],
    **session: Any,
) -> tuple[dict[str, EngineExit], dict[str, int]]:
    """Run one serving group's session into *reports*, starting and stopping
    its engines at the barriers and watching each while it is up: a model
    starts before the session or at its ``from``, and stops at its ``until``
    or the session's end.

    *keep_spanning* keeps a model starting at a barrier up for the next group;
    *stop_override* ends this session at a barrier the groups share;
    *spanning_segment* says models starting there do not start in it.

    Returns the engines lost while samples needed them, by engine name, and
    the samples failed for them per task.
    """
    from sieval.cli.session import arun_session

    stop = stop_override or staged.stop
    serving = dataclasses.replace(
        group.serving(),
        barriers=staged.stages,
        stop_after=stop_override,
        lifecycles=staged.lifecycles_json(group.prepared),
    )
    needed = group.own & await _models_to_infer(
        config_path,
        config,
        group.prepared,
        group.plan_dicts,
        roles=staged.roles,
        resume=resume,
        serving=dataclasses.replace(serving, on_barrier=None),
        **session,
    )
    # A model starting at or after the stop never runs in this session.
    runnable = {
        root for root in needed if runs_before_stop(staged.lifecycles.get(root), stop)
    }
    upfront = {root for root in runnable if staged.starts(root) is None}
    deferred = runnable - upfront
    if runnable:
        await check_launch_gpu(required=True)
    for root in sorted(group.own - runnable):
        log_user("No work for model {} in this run; not starting it", group.names[root])
    realized: dict[str, Deployment] = {}
    for root in sorted(upfront):
        plan, commands = group.prepared[root]
        realized[root] = await engines.start(group.names[root], plan, commands)
    for root in sorted(deferred):
        # Up already, kept from an earlier group.
        name = group.names[root]
        if name in engines.launched:
            realized[root] = engines.deployments[name]

    engine_loss = EngineLoss(settle=_ENGINE_LOSS_SETTLE)
    watches = _Watches(engine_loss, engines.deployer)

    async def on_barrier(stage: str) -> None:
        for root in sorted(upfront | deferred):
            if staged.stops(root) == stage:
                # Unwatched first, so the stop is not a loss.
                await watches.stop(root)
                await engines.stop(group.names[root])
        if stage == stop:
            return  # nothing after the stop runs; nothing starts for it
        for root in sorted(deferred):
            if staged.starts(root) == stage:
                plan, commands = group.prepared[root]
                await engines.start(group.names[root], plan, commands)
                watches.start(root, engines.launched[group.names[root]])

    serving = dataclasses.replace(serving, on_barrier=on_barrier)
    unlaunched = set(group.prepared) - set(realized)
    session_call = functools.partial(
        arun_session,
        config_path,
        resume=resume,
        # `self_managed_endpoints` scopes the best-effort warning away
        # from endpoints we launched ourselves.
        self_managed_endpoints=frozenset(group.prepared),
        infer_plans=group.plan_dicts or None,
        realized_deployments=realized or None,
        unlaunched_deployments=_unlaunched(group.prepared, unlaunched) or None,
        deferred_roots=frozenset(deferred - set(realized)),
        infer_gpu=await _plan_gpu_text() if group.plan_dicts else None,
        serving=serving,
        engine_loss=engine_loss,
        **session,
    )
    try:
        lost = await _session_watching(
            session_call,
            {
                root: engines.launched[group.names[root]]
                for root in sorted(realized)
                if group.names[root] in engines.launched
            },
            engine_loss,
            deployer=engines.deployer,
            reports=reports,
            watches=watches,
        )
    finally:
        # The next group's engines need the GPUs this one holds. Stopped
        # after the watch ended, so a stop is not a loss.
        for root in sorted(upfront | deferred):
            if root in deferred and keep_spanning and not spanning_segment:
                continue
            await engines.stop(group.names[root])
    return {group.names[root]: e for root, e in lost.items()}, engine_loss.failed


async def _launch(
    model_name: str,
    plan: DeploymentPlan,
    commands: list[BackendCommand],
    *,
    deployer: LocalDeployer,
    launched: dict[str, list[InferHandle]],
    handle_dir: Path,
    result_dir: str | None,
    keep_logs: int | None,
    ready_timeout: float,
) -> Deployment:
    """Start one engine under *model_name*, wait until it is ready, and record
    its handles in *launched*."""
    log_user("Starting inference for model: {}", model_name)

    last_status = None
    last_log_time = 0.0

    def _progress(elapsed: float, status_value: str) -> None:
        nonlocal last_status, last_log_time
        if sys.stderr.isatty():
            logger.opt(raw=True).log(
                "USER",
                f"\r\x1b[K[{model_name}] Waiting... "
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

    # Claim → deploy → save handle (shared with infer start). The handle lives
    # under this process's own run-<pid>/ directory, so an `infer start` of the
    # same model name does not collide.
    try:
        new_handles, env = await launch_model(
            model_name,
            commands,
            backend=plan.backend,
            deployer=deployer,
            handle_path=handle_dir / f"{model_name}.json",
            log_dir=(Path(result_dir) / "logs" / model_name if result_dir else None),
            keep_logs=keep_logs,
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

    if env is not None:
        log_user(
            "  env: {} / CUDA {} / {} x {}",
            env.framework or "unknown",
            env.cuda_version or "?",
            env.gpu_model or "?",
            env.gpu_count,
        )
    return deployer.build_deployment(plan, new_handles)


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
        serving_group: Annotated[
            str | None,
            typer.Option(
                "--serving-group",
                help=(
                    "Run only the tasks of this serving group, a key `--dry-run` "
                    "lists under `serving`, so each group can run as a job of its "
                    "own. Default: every group, one after another."
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
        resumed = cli_resume(resume)
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
                    serving_group=serving_group,
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
                serving_group=serving_group,
            )

        try:
            with terminate_as_interrupt() as terminated:
                reports = anyio.run(_go)
        except KeyboardInterrupt:
            # Ctrl-C/SIGTERM is not a command failure: exit 130/143 without a
            # result payload. Everything else is left to `@cli_command`.
            sys.exit(terminated.exit_code)
        except EngineLostError as exc:
            render(
                CommandResult(
                    command="run", ok=False, data=exc.to_json(), error=str(exc)
                ),
                output,
            )
            raise typer.Exit(ENGINE_LOST_EXIT_CODE) from exc

        tasks_data = {
            task_name: {"report": report} for task_name, report in reports.items()
        }
        cmd_result = CommandResult(
            command="run",
            ok=True,
            data={"tasks": tasks_data},
        )
        render(cmd_result, output)
