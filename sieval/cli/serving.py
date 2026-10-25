"""Reconcile serving with what tasks need: each task's context window, what a
model can serve (its ``context:`` block), and the serving groups a run launches.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from sieval.cli.resolution import config_task_candidates, validate_named_config_map
from sieval.infer.config import ParamValue
from sieval.infer.introspect import native_context_window
from sieval.infer.topology import DeploymentPlan

#: The engine parameter that sets the served context window, by backend.
CONTEXT_PARAMS: dict[str, str] = {"sglang": "context_length", "vllm": "max_model_len"}

#: The serving group that holds every recipe plan and every task an
#: external endpoint serves.
DEFAULT_GROUP = "default"

_MAX_TOKENS_KEYS = ("max_tokens", "max_completion_tokens")

#: Prompt room added to a ``max_tokens``-derived need: engines refuse
#: prompt + ``max_tokens`` > window. Longer prompts set ``min_context``.
PROMPT_HEADROOM = 4096


class ServingShortfallError(ValueError):
    """A task needs a longer context window than its model can serve."""


@dataclass(frozen=True, slots=True)
class ContextNeed:
    """The context window one task needs from its candidate root model."""

    root: str
    tokens: int
    source: str  # "min_context", "requires" or "max_tokens"


@dataclass(frozen=True, slots=True)
class ModelContext:
    """What a root model's ``context:`` block states."""

    native: int | None = None
    max: int | None = None
    scaling: Mapping[str, ParamValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ServedWindow:
    """The window a task's plan serves, reported by ``--dry-run``."""

    group: str
    window: int | None
    scaled: bool


@dataclass(frozen=True)
class ServingGroup:
    """Engines launched together, and the tasks they serve.

    *plans* holds one plan per served root model; *launches* are the roots
    some task of this group runs on. The default group also holds every task
    a model ``sieval run`` does not serve.
    """

    key: str
    plans: Mapping[str, DeploymentPlan]
    tasks: tuple[str, ...]
    windows: Mapping[str, ServedWindow]
    launches: frozenset[str]


@dataclass(frozen=True)
class SessionServing:
    """The serving group one session runs.

    *tasks* are the tasks it runs, ``None`` for all of them. *group* is its
    key, recorded in each served task's ``runs[].serving.group``. The plans go
    to *infer_plans_file*, so groups sharing a ``result_dir`` keep one each,
    with *recipes*: the recipe plans a scaled group's plans started from.
    """

    tasks: frozenset[str] | None = None
    group: str | None = None
    infer_plans_file: str = "infer_plans.yaml"
    recipes: Mapping[str, Any] = field(default_factory=dict)

    def runs(self, task_name: str) -> bool:
        return self.tasks is None or task_name in self.tasks


def parse_model_context(name: str, config: Mapping[str, Any]) -> ModelContext:
    """Validate a model's ``context:`` block."""
    raw = config.get("context")
    if raw is None:
        return ModelContext()
    if not isinstance(raw, Mapping):
        raise ValueError(f"Model '{name}' context must be a dictionary")
    unknown = sorted(set(raw) - {"native", "max", "scaling"})
    if unknown:
        raise ValueError(
            f"Model '{name}' context has unknown key(s): {', '.join(unknown)}; "
            "expected native, max, scaling"
        )
    native = _positive_int(raw.get("native"), f"Model '{name}' context.native")
    maximum = _positive_int(raw.get("max"), f"Model '{name}' context.max")
    scaling = raw.get("scaling") or {}
    if not isinstance(scaling, Mapping) or any(
        not isinstance(value, str | int | float | bool) for value in scaling.values()
    ):
        raise ValueError(
            f"Model '{name}' context.scaling must map engine parameters to scalars"
        )
    if native is not None and maximum is not None and maximum < native:
        raise ValueError(
            f"Model '{name}' context.max ({maximum}) is below context.native ({native})"
        )
    return ModelContext(native=native, max=maximum, scaling=dict(scaling))


def task_context_needs(config: Mapping[str, Any]) -> dict[str, ContextNeed]:
    """The context window each task needs, by task name.

    A need from ``max_tokens`` adds :data:`PROMPT_HEADROOM`; ``min_context``
    and a declared need already count the prompt. A task with none of the
    three is left out.
    """
    models = validate_named_config_map("models", config.get("models", {}))
    tasks = validate_named_config_map("tasks", config.get("tasks", {}))
    needs: dict[str, ContextNeed] = {}
    for task_name, (root, requirements) in config_task_candidates(config).items():
        task_config = tasks[task_name]
        override = _positive_int(
            task_config.get("min_context"), f"Task '{task_name}' min_context"
        )
        if override is not None:
            needs[task_name] = ContextNeed(root, override, "min_context")
            continue
        candidates: list[tuple[int, str]] = []
        if requirements.min_context is not None:
            candidates.append((requirements.min_context, "requires"))
        max_tokens = _effective_max_tokens(task_name, task_config, models)
        if max_tokens is not None:
            candidates.append((max_tokens + PROMPT_HEADROOM, "max_tokens"))
        if candidates:
            tokens, source = max(candidates)
            needs[task_name] = ContextNeed(root, tokens, source)
    return needs


def plan_window(plan: DeploymentPlan | Mapping[str, Any]) -> int | None:
    """The context window *plan* launches its engine with: the smallest across
    roles, since a prompt overflows on the first that refuses it. ``None`` when
    a role leaves it to the checkpoint's default."""
    if isinstance(plan, DeploymentPlan):
        params = [assignment.engine_params for assignment in plan.assignments]
    else:
        params = [
            assignment.get("engine_params") or {}
            if isinstance(assignment, Mapping)
            else {}
            for assignment in plan.get("assignments") or ()
        ]
    windows: list[int] = []
    for role in params:
        role_windows = [
            value
            for key, value in role.items()
            if key in CONTEXT_PARAMS.values()
            and isinstance(value, int)
            and not isinstance(value, bool)
        ]
        if not role_windows:
            return None
        windows.extend(role_windows)
    return min(windows) if windows else None


def served_window(
    plan: DeploymentPlan | Mapping[str, Any], native: int | None
) -> int | None:
    """The window *plan* serves: the one it sets, else the native one."""
    window = plan_window(plan)
    return window if window is not None else native


def check_context_fits(
    config: Mapping[str, Any],
    infer_plans: Mapping[str, Mapping[str, Any]] | None,
    tasks: frozenset[str] | None = None,
) -> None:
    """Refuse a task whose model cannot serve the window it needs.

    A model ``sieval run`` serves is checked against its plan; an endpoint
    against its root model's ``context_limit``. A model whose window is unknown
    is not checked: an overflow then shows per sample, as before. *tasks*
    limits the check to the tasks a session runs.
    """
    models = validate_named_config_map("models", config.get("models", {}))
    shortfalls: list[str] = []
    native_windows: dict[str, int | None] = {}
    for task_name, need in sorted(task_context_needs(config).items()):
        if tasks is not None and task_name not in tasks:
            continue
        root_config = models.get(need.root, {})
        declared = parse_model_context(need.root, root_config)
        plan = (infer_plans or {}).get(need.root)
        if plan is not None:
            if need.root not in native_windows:
                native_windows[need.root] = declared.native or _checkpoint_window(
                    plan.get("checkpoint")
                )
            window = served_window(plan, native_windows[need.root])
            what = "serves"
        else:
            window = _positive_int(
                root_config.get("context_limit"), f"Model '{need.root}' context_limit"
            )
            what = "declares context_limit"
        if window is not None and need.tokens > window:
            shortfalls.append(_shortfall(task_name, need, f"{what} {window}"))
    if shortfalls:
        raise ServingShortfallError(
            "Tasks need a longer context window than their model provides:\n"
            + "\n".join(shortfalls)
        )


def plan_serving_groups(
    config: Mapping[str, Any],
    base_plans: Mapping[str, DeploymentPlan],
) -> list[ServingGroup]:
    """Group the tasks by the plan that serves them, the default group first.

    *base_plans* are the recipe plans by root model. A need within the native
    window raises that plan; one within ``context.max`` gets the group
    ``<root>@<window>`` with ``context.scaling``. Raises
    :class:`ServingShortfallError` naming every task no plan can serve.
    """
    models = validate_named_config_map("models", config.get("models", {}))
    tasks = validate_named_config_map("tasks", config.get("tasks", {}))
    needs = task_context_needs(config)
    candidates = config_task_candidates(config)

    served: dict[str, list[str]] = {root: [] for root in base_plans}
    for task_name in tasks:
        candidate = candidates.get(task_name)
        if candidate is not None and candidate[0] in served:
            served[candidate[0]].append(task_name)

    default_plans: dict[str, DeploymentPlan] = {}
    windows: dict[str, ServedWindow] = {}
    scaled_groups: list[ServingGroup] = []
    shortfalls: list[str] = []
    launched: set[str] = set()
    for root, plan in base_plans.items():
        context = parse_model_context(root, models.get(root, {}))
        native = context.native or _checkpoint_window(plan.checkpoint)
        window = served_window(plan, native)
        pinned = _explicit_window(root, models.get(root, {}), plan.backend)
        raise_to: int | None = None
        scaled: list[str] = []
        for task_name in served[root]:
            need = needs.get(task_name)
            if need is None or window is None or need.tokens <= window:
                continue
            if pinned is not None:
                limit = f"pins {pinned[0]}: {pinned[1]}"
            elif native is None:
                limit = (
                    f"serves {window} and has no known native window "
                    "(set context.native)"
                )
            elif need.tokens <= native:
                raise_to = max(raise_to or 0, need.tokens)
                continue
            elif context.max is None:
                limit = f"supports at most its native {native} (no context.max)"
            elif need.tokens > context.max:
                limit = f"supports at most {context.max} (context.max)"
            elif not context.scaling:
                limit = (
                    f"needs scaling beyond its native {native}, but context "
                    "has no scaling parameters"
                )
            else:
                scaled.append(task_name)
                continue
            shortfalls.append(_shortfall(task_name, need, limit))

        default_plan = plan if raise_to is None else with_window(plan, raise_to, {})
        default_plans[root] = default_plan
        default_window = served_window(default_plan, native)
        stays = [task for task in served[root] if task not in scaled]
        if stays or not scaled:
            launched.add(root)
        for task_name in stays:
            windows[task_name] = ServedWindow(DEFAULT_GROUP, default_window, False)
        if scaled:
            scaled_window = max(needs[task].tokens for task in scaled)
            key = f"{root}@{scaled_window}"
            scaled_groups.append(
                ServingGroup(
                    key,
                    {root: with_window(plan, scaled_window, context.scaling)},
                    tuple(scaled),
                    {task: ServedWindow(key, scaled_window, True) for task in scaled},
                    frozenset({root}),
                )
            )
    if shortfalls:
        raise ServingShortfallError(
            "Tasks need a longer context window than their model can serve:\n"
            + "\n".join(shortfalls)
        )

    moved = {task for group in scaled_groups for task in group.tasks}
    default = ServingGroup(
        DEFAULT_GROUP,
        default_plans,
        tuple(task for task in tasks if task not in moved),
        windows,
        frozenset(launched),
    )
    return [default, *scaled_groups]


def with_window(
    plan: DeploymentPlan, window: int, scaling: Mapping[str, ParamValue]
) -> DeploymentPlan:
    """*plan* serving *window*, with the *scaling* engine parameters."""
    key = CONTEXT_PARAMS[plan.backend]
    return dataclasses.replace(
        plan,
        assignments=tuple(
            dataclasses.replace(
                assignment,
                engine_params={
                    **assignment.engine_params,
                    **scaling,
                    key: window,
                },
            )
            for assignment in plan.assignments
        ),
    )


def _shortfall(task: str, need: ContextNeed, limit: str) -> str:
    return (
        f"- task '{task}' needs {need.tokens} tokens ({_SOURCES[need.source]}); "
        f"model '{need.root}' {limit}"
    )


_SOURCES: dict[str, str] = {
    "min_context": "its min_context",
    "requires": "declared by the task",
    "max_tokens": f"its max_tokens plus {PROMPT_HEADROOM} for the prompt",
}


def _checkpoint_window(checkpoint: object) -> int | None:
    return native_context_window(checkpoint) if isinstance(checkpoint, str) else None


def _explicit_window(
    root: str, root_config: Mapping[str, Any], backend: str
) -> tuple[str, int] | None:
    """A window the user pinned, which the grouping never raises: an
    ``infer.overrides`` engine parameter, or a ``context_limit`` (which the
    session refuses unless the plan pins the same)."""
    infer = root_config.get("infer")
    overrides = infer.get("overrides") if isinstance(infer, Mapping) else None
    key = CONTEXT_PARAMS.get(backend)
    if isinstance(overrides, Mapping) and key is not None:
        for spelling in (key, key.replace("_", "-")):
            value = _positive_int(
                overrides.get(spelling), f"Model '{root}' infer.overrides.{spelling}"
            )
            if value is not None:
                return f"infer.overrides.{key}", value
    limit = root_config.get("context_limit")
    if isinstance(limit, int) and not isinstance(limit, bool):
        return "context_limit", limit
    return None


def _effective_max_tokens(
    task_name: str,
    task_config: Mapping[str, Any],
    models: Mapping[str, Mapping[str, Any]],
) -> int | None:
    """``max_tokens`` as the task's requests send it: its ``infer_args``, else
    the nearest model in its candidate's ``base:`` chain that sets it."""
    infer_args = task_config.get("infer_args")
    sources: list[Mapping[str, Any]] = []
    if isinstance(infer_args, Mapping):
        sources.append(infer_args)
    model_name = task_config.get("model")
    if model_name is None and len(models) == 1:
        model_name = next(iter(models))
    seen: set[str] = set()
    while isinstance(model_name, str) and model_name in models:
        if model_name in seen:
            break
        seen.add(model_name)
        args = models[model_name].get("args")
        if isinstance(args, Mapping):
            sources.append(args)
        model_name = models[model_name].get("base")
    for source in sources:
        for key in _MAX_TOKENS_KEYS:
            if source.get(key) is not None:
                return _positive_int(source[key], f"Task '{task_name}' {key}")
    return None


def _positive_int(value: object, where: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where} must be a positive integer, got {value!r}")
    return value
