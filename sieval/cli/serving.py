"""Reconcile serving with what tasks need: each task's context window, what a
model can serve (its ``context:`` block), the memory input scoring needs, and
the serving groups a run launches.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import dataclasses
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from sieval.cli.resolution import config_task_candidates, validate_named_config_map
from sieval.core.runners.barriers import BarrierHook
from sieval.infer.config import ParamValue
from sieval.infer.introspect import logit_shape, native_context_window
from sieval.infer.topology import DeploymentPlan

#: The engine parameter that sets the served context window, by backend.
CONTEXT_PARAMS: dict[str, str] = {"sglang": "context_length", "vllm": "max_model_len"}

#: The serving group that holds every recipe plan and every task an
#: external endpoint serves.
DEFAULT_GROUP = "default"

_MAX_TOKENS_KEYS = ("max_tokens", "max_completion_tokens")

#: Prompt room added to a ``max_tokens``-derived need: engines refuse
#: prompt + ``max_tokens`` > window. A ``max_tokens`` below the native window
#: is served at that window instead (:func:`_native_serves`). Longer prompts
#: set ``min_context``.
PROMPT_HEADROOM = 4096


#: The engine parameters the input-scoring headroom sets, by backend: the GPU
#: memory share taken up front, and the tokens of one prefill batch, which
#: bound the rows of logits it computes (SGLang charges every request of the
#: batch against it: ``rem_chunk_tokens`` in
#: https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/managers/schedule_policy.py#L846).
#: SGLang only; vLLM's prompt-logprob path has no measured rule yet.
MEMORY_PARAMS: dict[str, str] = {"sglang": "mem_fraction_static"}
PREFILL_PARAMS: dict[str, str] = {"sglang": "chunked_prefill_size"}

#: Bytes per scored logit beyond the model dtype's: SGLang's worst case holds
#: three fp32 vocabulary rows per token (``matrices_per_chunk`` in
#: https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/layers/logprob_processor.py#L561-L564).
SCORING_FP32_BYTES = 3 * 4

#: The most of one GPU's memory the reserve may take: a large vocabulary
#: shortens the prefill cap instead of shrinking the KV pool.
SCORING_RESERVE_SHARE = 0.10

#: The prefill cap is a multiple of this, unless one step exceeds the budget.
SCORING_PREFILL_STEP = 1024

#: The prefill cap never exceeds SGLang's own default on 80 to 160 GB GPUs.
SCORING_MAX_PREFILL = 8192


class ServingShortfallError(ValueError):
    """A task needs a longer context window than its model can serve."""


class ScoringHeadroomError(ValueError):
    """A pinned prefill cap needs more memory for its logits than the
    input-scoring budget allows."""


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
class ScoringHeadroom:
    """The memory reserved on one engine for the tasks that score input.

    *params* maps each changed engine parameter (``<role>.<param>`` on a plan
    with several roles) to its value before (``None``: unset) and after.
    *prefill_tokens* and *reserve_mib* are the largest role's, ``None`` when
    the reserve could not be sized; *reason* then says why.
    """

    tasks: tuple[str, ...]
    reason: str
    prefill_tokens: int | None = None
    reserve_mib: int | None = None
    params: Mapping[str, tuple[ParamValue | None, ParamValue]] = field(
        default_factory=dict
    )

    def to_json(self) -> dict[str, object]:
        return {
            "tasks": list(self.tasks),
            "prefill_tokens": self.prefill_tokens,
            "reserve_mib": self.reserve_mib,
            "params": {
                key: {"recipe": before, "served": after}
                for key, (before, after) in self.params.items()
            },
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ServingGroup:
    """Engines launched together, and the tasks they serve.

    *plans* holds one plan per served root model; *launches* are the roots
    some task of this group runs on. The default group also holds every task
    a model ``sieval run`` does not serve. *headroom* is the input-scoring
    reserve of each root some task of this group scores input on.
    """

    key: str
    plans: Mapping[str, DeploymentPlan]
    tasks: tuple[str, ...]
    windows: Mapping[str, ServedWindow]
    launches: frozenset[str]
    headroom: Mapping[str, ScoringHeadroom] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionServing:
    """The serving group one session runs.

    *tasks* are the tasks it runs, ``None`` for all of them. *group* is its
    key, recorded in each served task's ``runs[].serving.group``. The plans go
    to *infer_plans_file*, so groups sharing a ``result_dir`` keep one each,
    with *recipes*: the recipe plans this group's changed plans started from
    (scaled, or given input-scoring headroom).

    *barriers* and *on_barrier* are the stage barriers the session waits at and
    the hook awaited at each (:mod:`sieval.core.runners.barriers`); *stop_after*
    replaces the config's ``runner_config.stop_after`` when set. *lifecycles*
    are the launched models' lifecycles, recorded in ``runs[].serving``.
    """

    tasks: frozenset[str] | None = None
    group: str | None = None
    infer_plans_file: str = "infer_plans.yaml"
    recipes: Mapping[str, Any] = field(default_factory=dict)
    barriers: tuple[str, ...] = ()
    on_barrier: BarrierHook | None = None
    stop_after: str | None = None
    lifecycles: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

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
    and a declared need already count the prompt and are used as they are. A
    task with none of the three is left out. The serving side caps a
    ``max_tokens`` need at the native window (:func:`_native_serves`).
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
        native: int | None = None
        if plan is not None:
            if need.root not in native_windows:
                native_windows[need.root] = declared.native or _checkpoint_window(
                    plan.get("checkpoint")
                )
            native = native_windows[need.root]
            window = served_window(plan, native)
            what = "serves"
        else:
            window = _positive_int(
                root_config.get("context_limit"), f"Model '{need.root}' context_limit"
            )
            what = "declares context_limit"
        if window is not None and not _fits(need, window, native):
            shortfalls.append(_shortfall(task_name, need, f"{what} {window}", native))
    if shortfalls:
        raise ServingShortfallError(
            "Tasks need a longer context window than their model provides:\n"
            + "\n".join(shortfalls)
        )


def plan_serving_groups(
    config: Mapping[str, Any],
    base_plans: Mapping[str, DeploymentPlan],
    gpu_memory_mib: int | None = None,
) -> list[ServingGroup]:
    """Group the tasks by the plan that serves them, the default group first.

    *base_plans* are the recipe plans by root model. A need within the native
    window raises that plan, a ``max_tokens`` below native to the native
    window itself; one within ``context.max`` gets the group
    ``<root>@<window>`` with ``context.scaling``. A plan that serves a task
    scoring input then reserves memory for its logits
    (:func:`reserve_scoring_headroom`, sized for *gpu_memory_mib* per GPU).
    Raises :class:`ServingShortfallError` naming every task no plan can serve.
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

    scoring = {task for task, (_, req) in candidates.items() if req.input_scoring}

    def with_headroom(
        root: str, plan: DeploymentPlan, group_tasks: Iterable[str]
    ) -> tuple[DeploymentPlan, ScoringHeadroom | None]:
        scored = tuple(task for task in group_tasks if task in scoring)
        if not scored:
            return plan, None
        return reserve_scoring_headroom(
            plan,
            scored,
            longest_prompt=_longest_prompt(scored, needs, plan_window(plan)),
            gpu_memory_mib=gpu_memory_mib,
            pinned=_pinned_params(models.get(root, {}), plan.backend),
        )

    default_plans: dict[str, DeploymentPlan] = {}
    default_headroom: dict[str, ScoringHeadroom] = {}
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
            if need is None or window is None or _fits(need, window, native):
                continue
            if pinned is not None:
                limit = f"pins {pinned[0]}: {pinned[1]}"
            elif native is None:
                limit = (
                    f"serves {window} and has no known native window "
                    "(set context.native)"
                )
            elif _native_serves(need, native):
                raise_to = native
                continue
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
            shortfalls.append(_shortfall(task_name, need, limit, native))

        default_plan = plan if raise_to is None else with_window(plan, raise_to, {})
        default_window = served_window(default_plan, native)
        stays = [task for task in served[root] if task not in scaled]
        default_plan, headroom = with_headroom(root, default_plan, stays)
        default_plans[root] = default_plan
        if headroom is not None:
            default_headroom[root] = headroom
        if stays or not scaled:
            launched.add(root)
        for task_name in stays:
            windows[task_name] = ServedWindow(DEFAULT_GROUP, default_window, False)
        if scaled:
            scaled_window = max(needs[task].tokens for task in scaled)
            key = f"{root}@{scaled_window}"
            scaled_plan, headroom = with_headroom(
                root, with_window(plan, scaled_window, context.scaling), scaled
            )
            scaled_groups.append(
                ServingGroup(
                    key,
                    {root: scaled_plan},
                    tuple(scaled),
                    {task: ServedWindow(key, scaled_window, True) for task in scaled},
                    frozenset({root}),
                    {root: headroom} if headroom is not None else {},
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
        default_headroom,
    )
    return [default, *scaled_groups]


def reserve_scoring_headroom(
    plan: DeploymentPlan,
    tasks: tuple[str, ...],
    *,
    longest_prompt: int | None,
    gpu_memory_mib: int | None,
    pinned: frozenset[str] = frozenset(),
) -> tuple[DeploymentPlan, ScoringHeadroom]:
    """*plan* with GPU memory set aside for the logits input scoring computes.

    Per GPU (tensor parallelism gathers the full vocabulary on every rank)::

        row     = vocab_size * (dtype_bytes + SCORING_FP32_BYTES)
        budget  = SCORING_RESERVE_SHARE * gpu_bytes // row
        prefill = min(budget, longest_prompt, SCORING_MAX_PREFILL), up to a
                  SCORING_PREFILL_STEP, down again if that exceeds budget
        memory  = recipe_memory - prefill * row / gpu_bytes, floored to 0.01

    Each role keeps a smaller or pinned prefill cap, and its reserve follows
    that cap. A memory share pinned in *pinned* or left to the engine is not
    lowered. A missing fact (backend rule, vocabulary, GPU memory) leaves the
    plan as it is, and the reason says why.

    Raises :class:`ScoringHeadroomError` when a pinned cap's reserve exceeds
    the budget and the share would be lowered by it.
    """
    memory_key = MEMORY_PARAMS.get(plan.backend)
    prefill_key = PREFILL_PARAMS.get(plan.backend)
    if memory_key is None or prefill_key is None:
        return plan, ScoringHeadroom(
            tasks, f"unchanged: no headroom rule for backend {plan.backend!r}"
        )
    shape = logit_shape(plan.checkpoint)
    if shape is None:
        return plan, ScoringHeadroom(
            tasks, "unchanged: the checkpoint's vocab_size is unknown"
        )
    if gpu_memory_mib is None:
        return plan, ScoringHeadroom(tasks, "unchanged: the GPU memory is unknown")
    vocab, dtype_bytes = shape
    gpu_bytes = gpu_memory_mib * 2**20
    row = int(vocab * (dtype_bytes + SCORING_FP32_BYTES))
    budget = int(SCORING_RESERVE_SHARE * gpu_bytes) // row
    wanted = min(budget, longest_prompt or budget, SCORING_MAX_PREFILL)
    step = SCORING_PREFILL_STEP
    prefill = min(-(-wanted // step) * step, budget // step * step) or budget
    if prefill < 1:
        return plan, ScoringHeadroom(
            tasks, f"unchanged: one logit row of vocab {vocab} exceeds the budget"
        )

    def key(role: str, param: str) -> str:
        return param if len(plan.assignments) == 1 else f"{role}.{param}"

    params: dict[str, tuple[ParamValue | None, ParamValue]] = {}
    engines: list[dict[str, ParamValue]] = []
    caps: list[int] = []
    kept: set[str] = set()
    for assignment in plan.assignments:
        engine = dict(assignment.engine_params)
        cap = prefill
        current = _count(engine.get(prefill_key))
        if current is not None and (prefill_key in pinned or current <= prefill):
            cap = current
        else:
            engine[prefill_key] = prefill
            params[key(assignment.role, prefill_key)] = (current, prefill)
        memory = engine.get(memory_key)
        if memory_key in pinned:
            kept.add(f"{memory_key} kept: set by infer.overrides")
        elif isinstance(memory, bool) or not isinstance(memory, int | float):
            kept.add(f"{memory_key} left to the engine")
        elif cap > budget:
            raise ScoringHeadroomError(
                f"Input scoring by {', '.join(tasks)}: the {prefill_key} {cap} "
                f"pinned in infer.overrides needs {cap * row // 2**20} MiB per "
                f"GPU for its logits, over {SCORING_RESERVE_SHARE:.0%} of "
                f"{gpu_memory_mib} MiB. Pin {prefill_key} at most {budget}, "
                f"or pin {memory_key} as well."
            )
        else:
            lowered = math.floor(round((memory - cap * row / gpu_bytes) * 100, 6))
            if lowered / 100 < memory:
                engine[memory_key] = lowered / 100
                params[key(assignment.role, memory_key)] = (memory, lowered / 100)
        engines.append(engine)
        caps.append(cap)

    adjusted = dataclasses.replace(
        plan,
        assignments=tuple(
            dataclasses.replace(assignment, engine_params=engine)
            for assignment, engine in zip(plan.assignments, engines, strict=True)
        ),
    )
    prefill = max(caps)
    reserve = prefill * row
    reason = (
        f"input scoring by {', '.join(tasks)}: full-vocabulary logits of "
        f"{prefill} prefill tokens x {vocab} vocab x "
        f"({dtype_bytes:g}+{SCORING_FP32_BYTES}) B = {reserve // 2**20} MiB "
        f"per GPU, {reserve / gpu_bytes:.1%} of {gpu_memory_mib} MiB"
    )
    if kept:
        reason += "; " + "; ".join(sorted(kept))
    return adjusted, ScoringHeadroom(
        tasks, reason, prefill, reserve // 2**20, dict(sorted(params.items()))
    )


def _longest_prompt(
    tasks: Iterable[str], needs: Mapping[str, ContextNeed], window: int | None
) -> int | None:
    """The longest prompt *tasks* may score, as far as it is known: each one's
    context need, at most *window*; ``None`` when a task's is unknown."""
    bounds: list[int] = []
    for task in tasks:
        need = needs.get(task)
        known = [v for v in (need.tokens if need else None, window) if v is not None]
        if not known:
            return None
        bounds.append(min(known))
    return max(bounds) if bounds else None


def _pinned_params(root_config: Mapping[str, Any], backend: str) -> frozenset[str]:
    """The headroom parameters a root model's ``infer.overrides`` sets, which
    the reserve keeps."""
    infer = root_config.get("infer")
    overrides = infer.get("overrides") if isinstance(infer, Mapping) else None
    if not isinstance(overrides, Mapping):
        return frozenset()
    spelled = {str(key).replace("-", "_") for key in overrides}
    keys = {MEMORY_PARAMS.get(backend), PREFILL_PARAMS.get(backend)}
    return frozenset(key for key in keys if key is not None and key in spelled)


def _count(value: object) -> int | None:
    """*value* when it is a positive integer, else ``None``."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


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


def _native_serves(need: ContextNeed, native: int | None) -> bool:
    """Whether *native* serves *need* unscaled: a need from a ``max_tokens``
    below the native window. Such a task is never scaled, since scaling changes
    outputs; its window is raised to *native*, and the prompt gets the
    ``native - max_tokens`` tokens left. A longer prompt overflows per sample
    (``context_overflow``)."""
    return (
        need.source == "max_tokens"
        and native is not None
        and need.tokens - PROMPT_HEADROOM < native
    )


def _required(need: ContextNeed, native: int | None) -> int:
    """The smallest window that serves *need*: its tokens, capped at *native*
    when the native window serves it."""
    if native is not None and _native_serves(need, native):
        return min(need.tokens, native)
    return need.tokens


def _fits(need: ContextNeed, window: int, native: int | None) -> bool:
    return _required(need, native) <= window


def _shortfall(
    task: str, need: ContextNeed, limit: str, native: int | None = None
) -> str:
    tokens = _required(need, native)
    source = _SOURCES[need.source]
    if tokens != need.tokens:
        source = "its max_tokens, served at the native window"
    return (
        f"- task '{task}' needs {tokens} tokens ({source}); model '{need.root}' {limit}"
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
