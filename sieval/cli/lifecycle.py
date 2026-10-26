"""Model lifecycles: when each engine ``sieval run`` launches starts and stops.

``until: S`` stops an engine at the barrier on ``S``; ``from: S`` starts it
there. Lifecycle boundaries and ``stop_after`` are the session's barriers
(:mod:`sieval.core.runners.barriers`); the stretches between them are phases.
Everything here is planned before any engine starts: the barriers, who is alive
in each phase, whether they fit, and each engine's devices and port.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import dataclasses
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from sieval.cli.resolution import TASK_MODEL_ROLES, validate_named_config_map
from sieval.core.runners.barriers import (
    BARRIER_STAGES,
    ITERATION_BARRIERS,
    check_barrier_stage,
    stage_index,
)
from sieval.infer.backends.translator import BackendCommand

#: Least spacing between the base ports of engines alive at the same time; a
#: plan with more commands than this takes one port per command.
PORT_STEP = 10

_LIFECYCLE_KEYS = frozenset({"from", "until"})


class LifecycleError(ValueError):
    """A lifecycle, or the barriers it implies, cannot run as configured."""


class LifecycleFitError(LifecycleError):
    """Models alive in the same phase do not fit on the run's GPUs together."""


@dataclass(frozen=True)
class Lifecycle:
    """When one engine runs: from the barrier on *start* (the run's start when
    ``None``) to the barrier on *stop* (the run's end when ``None``)."""

    start: str | None = None
    stop: str | None = None

    def to_json(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.start is not None:
            out["from"] = self.start
        if self.stop is not None:
            out["until"] = self.stop
        return out


def parse_lifecycle(name: str, raw: object) -> Lifecycle:
    """*raw* as a :class:`Lifecycle`, refusing an unknown key or stage, ``from:
    report`` (nothing runs after it), and ``from`` at or after ``until``."""
    where = f"Model '{name}': 'lifecycle'"
    if not isinstance(raw, Mapping):
        raise LifecycleError(f"{where} must be a mapping of from/until")
    raw = cast(Mapping[str, object], raw)
    unknown = sorted(set(map(str, raw)) - _LIFECYCLE_KEYS)
    if unknown:
        raise LifecycleError(
            f"{where} has unknown key(s) {', '.join(unknown)}; expected from, until"
        )
    try:
        start = (
            check_barrier_stage(raw["from"], f"{where}.from") if "from" in raw else None
        )
        stop = (
            check_barrier_stage(raw["until"], f"{where}.until")
            if "until" in raw
            else None
        )
    except ValueError as exc:
        raise LifecycleError(str(exc)) from None
    if start == "report":
        raise LifecycleError(
            f"{where}.from is 'report': no stage after it calls a model"
        )
    if (
        start is not None
        and stop is not None
        and (stage_index(start) >= stage_index(stop))
    ):
        raise LifecycleError(f"{where}: from {start!r} is not before until {stop!r}")
    return Lifecycle(start, stop)


def model_lifecycles(config: Mapping[str, Any]) -> dict[str, Lifecycle]:
    """Each model's declared lifecycle, by ``models:`` name; validated."""
    models = validate_named_config_map("models", config.get("models", {}))
    out: dict[str, Lifecycle] = {}
    for name, model in models.items():
        if "lifecycle" not in model:
            continue
        if model.get("base") is not None:
            raise LifecycleError(
                f"Model '{name}': derived models cannot set 'lifecycle'; it is the "
                "root's engine. Set it on the base instead."
            )
        out[name] = parse_lifecycle(name, model["lifecycle"])
    return out


def session_stop_after(config: Mapping[str, Any]) -> str | None:
    """The top-level ``runner_config.stop_after``, validated; ``None`` if unset."""
    runner = config.get("runner_config") or {}
    if not isinstance(runner, Mapping) or runner.get("stop_after") is None:
        return None
    return check_barrier_stage(runner["stop_after"], "runner_config.stop_after")


def effective_barriers(
    config: Mapping[str, Any], lifecycles: Mapping[str, Lifecycle] | None = None
) -> dict[str, list[str]]:
    """Every barrier the session waits at, in pipeline order, each with what
    implied it: ``stop_after`` or ``<model>.lifecycle.from|until``."""
    if lifecycles is None:
        lifecycles = model_lifecycles(config)
    implied: dict[str, list[str]] = {}
    stop = session_stop_after(config)
    if stop is not None:
        implied.setdefault(stop, []).append("stop_after")
    for name in sorted(lifecycles):
        life = lifecycles[name]
        if life.start is not None:
            implied.setdefault(life.start, []).append(f"{name}.lifecycle.from")
        if life.stop is not None:
            implied.setdefault(life.stop, []).append(f"{name}.lifecycle.until")
    return {stage: implied[stage] for stage in sorted(implied, key=stage_index)}


def check_lifecycles_launched(
    lifecycles: Mapping[str, Lifecycle], launched: Iterable[str]
) -> None:
    """Refuse a lifecycle on a model ``sieval run`` does not launch."""
    served = set(launched)
    stray = sorted(set(lifecycles) - served)
    if stray:
        raise LifecycleError(
            "'lifecycle' applies to a model `sieval run` launches (a 'path' or "
            "'infer' without 'api_base'): " + ", ".join(stray)
        )


def check_iteration_barriers(
    config: Mapping[str, Any], barriers: Iterable[str]
) -> None:
    """Refuse a barrier on ``inference`` or ``postprocess`` when some task's
    ``max_iterations`` lets a sample go round again: it returns from
    ``feedback`` to ``inference``, past a barrier already crossed."""
    blocked = [s for s in barriers if s in ITERATION_BARRIERS]
    if not blocked:
        return
    tasks = validate_named_config_map("tasks", config.get("tasks", {}))
    top = config.get("runner_config") or {}
    iterating = []
    for name, task in tasks.items():
        runner = {**(top if isinstance(top, Mapping) else {})}
        runner.update(task.get("runner_config") or {})
        limit = runner.get("max_iterations")
        if isinstance(limit, int) and not isinstance(limit, bool) and limit > 1:
            iterating.append(f"{name} (max_iterations {limit})")
    if iterating:
        raise LifecycleError(
            "Task(s) "
            + ", ".join(iterating)
            + " can iterate: samples return from feedback to inference, so they "
            f"cannot pass a barrier on {', '.join(blocked)}; set max_iterations: 1 "
            "or drop the barrier"
        )


def role_models(config: Mapping[str, Any]) -> dict[str, set[str]]:
    """The ``models:`` entries each task names as ``grader`` or ``extractor``."""
    models = validate_named_config_map("models", config.get("models", {}))
    tasks = validate_named_config_map("tasks", config.get("tasks", {}))
    out: dict[str, set[str]] = {}
    for task_name, task in tasks.items():
        args = task.get("args") or {}
        if not isinstance(args, Mapping):
            continue
        for role in TASK_MODEL_ROLES:
            source = args.get(role)
            if isinstance(source, str) and source in models:
                out.setdefault(task_name, set()).add(source)
    return out


def root_model(config: Mapping[str, Any], name: str) -> str:
    models = config.get("models") or {}
    seen = {name}
    while isinstance(entry := models.get(name), Mapping) and isinstance(
        entry.get("base"), str
    ):
        name = entry["base"]
        if name in seen:
            break
        seen.add(name)
    return name


@dataclass(frozen=True)
class Phase:
    """The stretch of a run between two barriers: after *start* (the run's
    start when ``None``) until *stop* (its end when ``None``)."""

    start: str | None
    stop: str | None
    models: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"{self.start or 'start'} → {self.stop or 'end'}"


def plan_phases(
    barriers: Sequence[str],
    lifecycles: Mapping[str, Lifecycle],
    served: Iterable[str],
    stop_after: str | None = None,
) -> list[Phase]:
    """The phases that run, each with the served models alive in it.

    A run that stops after a barrier has no phase past it.
    """
    bounds: list[str | None] = [None, *barriers, None]
    last = (
        list(barriers).index(stop_after) + 1
        if stop_after is not None and stop_after in barriers
        else len(barriers) + 1
    )
    phases = []
    for k in range(last):
        alive = tuple(
            sorted(
                name
                for name in served
                if _alive(lifecycles.get(name, Lifecycle()), barriers, k)
            )
        )
        phases.append(Phase(bounds[k], bounds[k + 1], alive))
    return phases


def _alive(life: Lifecycle, barriers: Sequence[str], phase: int) -> bool:
    first = barriers.index(life.start) + 1 if life.start is not None else 0
    last = barriers.index(life.stop) if life.stop is not None else len(barriers)
    return first <= phase <= last


def visible_devices(environ: Mapping[str, str] | None = None) -> list[str] | None:
    """The inherited ``CUDA_VISIBLE_DEVICES`` entries; ``None`` when unset."""
    raw = (os.environ if environ is None else environ).get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


@dataclass(frozen=True)
class Placement:
    """Where one engine runs: its GPUs, as indices into the run's visible GPUs
    (``None`` when it is never alive with another engine, so it takes every
    GPU), and its port offset from the backend's default."""

    gpus: int
    devices: tuple[int, ...] | None
    port_offset: int


def place(
    phases: Sequence[Phase],
    gpus: Mapping[str, int],
    gpu_count: int | None,
    ports: Mapping[str, int] | None = None,
) -> dict[str, Placement]:
    """Disjoint devices and port ranges for models alive at the same time.

    *ports* is how many consecutive ports each model takes (1 when absent).
    Raises :class:`LifecycleFitError` when the models of a phase need more GPUs
    than the run has; with *gpu_count* unknown, nothing is checked and no
    devices are assigned.
    """
    together: set[str] = set()
    for phase in phases:
        if len(phase.models) > 1:
            together.update(phase.models)
    if gpu_count is not None:
        short = []
        for phase in phases:
            need = sum(gpus.get(name, 0) for name in phase.models)
            if need > gpu_count:
                detail = ", ".join(f"{n} {gpus.get(n, 0)}" for n in phase.models)
                short.append(
                    f"phase {phase.label}: {detail} need {need} GPU(s), "
                    f"the run has {gpu_count}"
                )
        if short:
            raise LifecycleFitError(
                "Models alive at the same time do not fit: "
                + "; ".join(short)
                + ". Pin a smaller tp/dp in infer.overrides, or give them "
                "lifecycles that do not overlap."
            )
    devices: dict[str, tuple[int, ...]] = {}
    if gpu_count is not None and together:
        devices = _assign(phases, together, gpus, gpu_count)
    offsets: dict[str, int] = {}
    next_offset = 0
    for name in sorted(together):
        offsets[name] = next_offset
        next_offset += max(PORT_STEP, (ports or {}).get(name, 1))
    out: dict[str, Placement] = {}
    for phase in phases:
        for name in phase.models:
            if name in out:
                continue
            out[name] = Placement(
                gpus=gpus.get(name, 0),
                devices=devices.get(name),
                port_offset=offsets.get(name, 0),
            )
    return out


def _assign(
    phases: Sequence[Phase],
    together: set[str],
    gpus: Mapping[str, int],
    gpu_count: int,
) -> dict[str, tuple[int, ...]]:
    """First fit: each model takes the lowest block of GPUs free in every phase
    it is alive in; a model keeps its block for its whole life."""
    alive = {
        name: [i for i, phase in enumerate(phases) if name in phase.models]
        for name in together
    }
    used: list[set[int]] = [set() for _ in phases]
    out: dict[str, tuple[int, ...]] = {}
    for name in sorted(together, key=lambda n: (alive[n][0], n)):
        need = gpus.get(name, 0)
        taken = set().union(*(used[i] for i in alive[name]))
        for start in range(gpu_count - need + 1):
            block = set(range(start, start + need))
            if not block & taken:
                out[name] = tuple(sorted(block))
                for i in alive[name]:
                    used[i] |= block
                break
        else:
            raise LifecycleFitError(
                f"Model '{name}' needs {need} contiguous GPU(s) free in every "
                f"phase it is alive in; none of the {gpu_count} are"
            )
    return out


def placed_commands(
    commands: Sequence[BackendCommand],
    placement: Placement | None,
    visible: Sequence[str] | None = None,
) -> list[BackendCommand]:
    """*commands* on their placement's devices and port; unchanged when the
    model takes every GPU on the default port. *visible* is the inherited
    ``CUDA_VISIBLE_DEVICES`` the device indices select from."""
    if placement is None or (placement.devices is None and not placement.port_offset):
        return list(commands)
    out = []
    for command in commands:
        args = list(command.cli_args)
        health = command.health_url
        if placement.port_offset and "--port" in args:
            at = args.index("--port") + 1
            old = args[at]
            new = str(int(old) + placement.port_offset)
            args[at] = new
            if health:
                health = health.replace(f":{old}/", f":{new}/")
        env = dict(command.env)
        if placement.devices is not None:
            devices = [
                visible[i] if visible is not None else str(i) for i in placement.devices
            ]
            env.setdefault("CUDA_VISIBLE_DEVICES", ",".join(devices))
        out.append(
            dataclasses.replace(command, cli_args=args, env=env, health_url=health)
        )
    return out


def phases_json(
    phases: Sequence[Phase],
    lifecycles: Mapping[str, Lifecycle],
    placements: Mapping[str, Placement],
) -> list[dict[str, object]]:
    return [
        {
            "after": phase.start,
            "until": phase.stop,
            "models": {
                name: {
                    "lifecycle": lifecycles.get(name, Lifecycle()).to_json(),
                    "gpus": placements[name].gpus if name in placements else None,
                    "devices": (
                        list(devices)
                        if name in placements
                        and (devices := placements[name].devices) is not None
                        else None
                    ),
                }
                for name in phase.models
            },
        }
        for phase in phases
    ]


def starts_at(life: Lifecycle | None) -> str | None:
    return None if life is None else life.start


def runs_before_stop(life: Lifecycle | None, stop_after: str | None) -> bool:
    """Whether a model with *life* has any phase in a run stopping after
    *stop_after*: one that starts at or after the stop never does."""
    if life is None or life.start is None or stop_after is None:
        return True
    return stage_index(life.start) < stage_index(stop_after)


__all__ = [
    "BARRIER_STAGES",
    "Lifecycle",
    "LifecycleError",
    "LifecycleFitError",
    "Phase",
    "Placement",
    "check_iteration_barriers",
    "check_lifecycles_launched",
    "effective_barriers",
    "model_lifecycles",
    "parse_lifecycle",
    "phases_json",
    "place",
    "placed_commands",
    "plan_phases",
    "role_models",
    "root_model",
    "runs_before_stop",
    "session_stop_after",
    "starts_at",
    "visible_devices",
]
