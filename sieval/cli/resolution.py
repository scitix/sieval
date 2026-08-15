"""Config-string → class resolution, and the model-type derivation built on it.

:func:`derive_model_type` is shared by the eval session and infer recipe
resolution, so it lives here with the class-resolution helpers it stands on
rather than in ``leaderboard/session.py`` — importing the session to reach it
would point the dependency sideways across the two CLI subpackages.
``scripts/check_layer_imports.py`` rejects that edge.

The model-kind resolver consumes the provider-neutral requirements projection
from ``sieval.core``. It deliberately imports neither CLI subpackage: both the
eval composition root and infer recipe resolution depend on this module, never
on each other.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import copy
import hashlib
import importlib
import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from loguru import logger

from sieval.core.models.deployment import BINDING_RESOURCE_KEYS
from sieval.core.models.requirements import (
    AggregatedTaskRequirements,
    InlineModelBinding,
    InputKind,
    NamedModelBinding,
    RequirementContext,
    TaskModelRequirement,
    aggregate_task_requirements,
)
from sieval.core.types import JSONValue

# Registry for simple name lookups
DATASET_MODULE = "sieval.datasets"
TASK_MODULE = "sieval.tasks"

# Task arguments that describe a model binding rather than ordinary task data.
# Keep the pre-session config adapter and EvalSession composition root on this
# single role vocabulary so a newly supported role cannot work in only one CLI
# entry point.
TASK_MODEL_ROLES = ("grader", "extractor")
_TASK_MODEL_ROLE_SENTINELS: Mapping[str, frozenset[str]] = MappingProxyType(
    {"extractor": frozenset({"self"})}
)
COMPOSITION_OWNED_TASK_ARGS = frozenset({"dataset", "model", "models_by_role", "name"})
_INLINE_SECRET_KEYS = frozenset({"api_key", "authorization"})


def is_task_model_role_sentinel(role: str, source: object) -> bool:
    """Return whether ``source`` is an explicit sentinel admitted for ``role``."""

    return isinstance(source, str) and source in _TASK_MODEL_ROLE_SENTINELS.get(
        role, ()
    )


def validate_task_config_args(
    task_name: str,
    task_args: Mapping[str, object],
    *,
    task_class: type[Any] | None = None,
) -> None:
    """Reject composition-owned keys and role arguments a task cannot receive."""

    reserved = sorted(COMPOSITION_OWNED_TASK_ARGS & set(task_args))
    if reserved:
        raise ValueError(
            f"Task '{task_name}' args cannot set composition-owned constructor "
            "argument(s): " + ", ".join(reserved)
        )
    if task_class is None:
        return

    configured_roles = set(TASK_MODEL_ROLES) & set(task_args)
    if not configured_roles:
        return

    try:
        constructor_parameters = inspect.signature(task_class.__init__).parameters
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Cannot inspect {task_class.__name__}.__init__ for model-role arguments"
        ) from exc
    unsupported_roles = sorted(
        role for role in configured_roles if role not in constructor_parameters
    )
    if unsupported_roles:
        formatted = ", ".join(repr(role) for role in unsupported_roles)
        raise ValueError(
            f"Task '{task_name}' config supplies model role(s) {formatted}, but "
            f"{task_class.__name__} does not declare matching constructor arguments"
        )


def binding_resource_argument_paths(
    arguments: Mapping[str, object],
    *,
    allowed: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """Return binding-resource names misplaced on a request/default surface."""

    paths = {
        str(key)
        for key in arguments
        if isinstance(key, str) and key in BINDING_RESOURCE_KEYS and key not in allowed
    }
    for container_name in ("extra_body", "extra_wire_params"):
        nested = arguments.get(container_name)
        if not isinstance(nested, Mapping):
            continue
        paths.update(
            f"{container_name}.{key}"
            for key in nested
            if isinstance(key, str) and key in BINDING_RESOURCE_KEYS
        )
    return tuple(sorted(paths))


def _safe_inline_model_config(
    config: Mapping[str, object],
) -> dict[str, JSONValue]:
    """Copy an inline binding without putting raw credentials in setup data."""

    safe: dict[str, JSONValue] = {}
    for key, value in config.items():
        if not isinstance(key, str):
            raise TypeError("inline model config keys must be strings")
        if key in _INLINE_SECRET_KEYS:
            continue
        if key == "args" and isinstance(value, Mapping):
            value = {
                nested_key: nested_value
                for nested_key, nested_value in value.items()
                if nested_key not in _INLINE_SECRET_KEYS
            }
        safe[key] = cast(JSONValue, copy.deepcopy(value))
    return safe


def normalize_inline_model_binding(
    task_name: str,
    role: str,
    raw_config: Mapping[str, object],
) -> InlineModelBinding:
    """Normalize one inline role binding identically before and inside a session."""

    allowed_inline = frozenset(
        {
            "api_base",
            "api_key",
            "capabilities",
            "dialect",
            "engine",
            "max_retries",
            "model",
            "service_role",
        }
    )
    misplaced = binding_resource_argument_paths(
        raw_config,
        allowed=allowed_inline,
    )
    raw_args = raw_config.get("args", {})
    if not isinstance(raw_args, Mapping):
        raise ValueError(f"Task '{task_name}' inline {role} args must be a mapping")
    nested = binding_resource_argument_paths(
        cast(Mapping[str, object], raw_args),
        allowed=frozenset({"api_base", "api_key", "max_retries"}),
    )
    paths = (*misplaced, *(f"args.{path}" for path in nested))
    if paths:
        raise ValueError(
            f"Task '{task_name}' inline {role} places binding resource(s) "
            f"on an unsupported surface: {', '.join(paths)}"
        )

    requested_model_id = raw_config.get("model")
    if not isinstance(requested_model_id, str) or not requested_model_id:
        raise ValueError(
            f"Task '{task_name}' inline {role} model requires a non-empty 'model'"
        )
    safe_config = _safe_inline_model_config(raw_config)
    encoded = json.dumps(
        safe_config,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    binding_id = f"inline:{task_name}:{role}:{digest}"
    dialect = raw_config.get("dialect", "openai_chat")
    if not isinstance(dialect, str) or not dialect:
        raise ValueError(
            f"Task '{task_name}' inline {role} dialect must be a non-empty string"
        )
    if dialect == "sglang_legacy":
        raise ValueError(
            f"Task '{task_name}' inline {role} cannot use the named-model-only "
            "sglang_legacy bypass; configure a named model or use a bindable dialect"
        )
    return InlineModelBinding(
        binding_id=binding_id,
        root_deployment_key=f"deployment:{binding_id}",
        requested_model_id=requested_model_id,
        config=safe_config,
        dialect_id=dialect,
    )


def validate_task_model_requirements(
    task_class: type[Any],
    context: RequirementContext,
    raw_records: object,
) -> tuple[TaskModelRequirement, ...]:
    """Validate one task hook's complete projection of normalized bindings.

    The hook may describe each binding's requirements, but it may neither
    replace a normalized binding nor omit one supplied by configuration.  Both
    pre-session config resolution and :class:`EvalSession` call this function
    so dry-run and execution reject the same incomplete role projection.
    """

    hook_name = f"{task_class.__name__}.model_requirements_for()"
    if not isinstance(raw_records, tuple):
        raise TypeError(f"{hook_name} must return a tuple")

    records: list[TaskModelRequirement] = []
    declared_roles: set[str] = set()
    for record in raw_records:
        if not isinstance(record, TaskModelRequirement):
            raise TypeError(f"{hook_name} returned a non-TaskModelRequirement value")
        expected = context.model_bindings.get(record.role)
        if expected is None:
            raise ValueError(f"{hook_name} declared unknown model role {record.role!r}")
        if record.binding != expected:
            raise ValueError(
                f"{hook_name} changed the normalized binding for role {record.role!r}"
            )
        records.append(record)
        declared_roles.add(record.role)

    undeclared_roles = sorted(set(context.model_bindings) - declared_roles)
    if undeclared_roles:
        formatted_roles = ", ".join(repr(role) for role in undeclared_roles)
        raise ValueError(
            f"{hook_name} did not declare normalized model role(s): {formatted_roles}"
        )

    auxiliary_roles = declared_roles - {"candidate", "model"}
    if auxiliary_roles:
        constructor_parameters = inspect.signature(task_class.__init__).parameters
        if "models_by_role" not in constructor_parameters:
            raise ValueError(
                f"{hook_name} declares auxiliary model role(s) "
                f"{', '.join(sorted(auxiliary_roles))}, but "
                f"{task_class.__name__}.__init__ does not accept models_by_role"
            )
    return tuple(records)


def load_object_from_path(object_path: str, kind: str = "class") -> object:
    """
    Load a module attribute from a full path like 'pkg.module.Name'.

    *kind* only names the attribute in the error messages ("class", "function"),
    so the same import mechanism can be reported in the caller's vocabulary.
    """
    if "." not in object_path:
        raise ValueError(
            f"Invalid {kind} path: {object_path}. Expected format: "
            f"'module.{'ClassName' if kind == 'class' else 'name'}'"
        )

    module_name, attribute_name = object_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attribute_name)
    except ImportError as exc:
        if _is_missing_module_error(exc, module_name):
            raise ImportError(f"Could not import module '{module_name}'") from exc
        # Internal dependency missing — propagate the original error
        raise
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_name}' has no {kind} '{attribute_name}'"
        ) from e


def load_class_from_path(class_path: str) -> type:
    """
    Load a class from a full module path like 'sieval.core.datasets.AIME2024Dataset'.
    """
    loaded = load_object_from_path(class_path, "class")
    if not isinstance(loaded, type):
        raise ValueError(
            f"'{class_path}' resolved to {type(loaded).__name__}, not a class"
        )
    return loaded


def resolve_key_function(spec: str) -> Callable[..., object]:
    """Resolve ``'pkg.module.function'`` to the callable it names.

    The config counterpart of passing a function directly —
    ``dataset.filter(by=my_key, ...)`` in Python, ``by: {callable:
    'pkg.module.my_key'}`` in YAML. YAML cannot hold a function body, so it
    names one, exactly as ``class:`` names a class rather than defining it.

    A full dotted path is required: unlike a dataset or task class there is no
    registry to search a bare name in, and importing ``my_key`` from wherever
    it first appeared would make the resolved function depend on import order.
    """
    if not isinstance(spec, str):
        raise ValueError(f"Callable reference must be a string, got {spec!r}")
    if spec.startswith("."):
        raise ValueError(
            f"Relative import syntax is not supported: '{spec}'. "
            f"Use a full path ('pkg.module.my_key')"
        )
    if "." not in spec:
        raise ValueError(
            f"Invalid function path: {spec}. A callable must be given as a full "
            f"path ('pkg.module.{spec}'), since there is no module to search a "
            f"bare name in"
        )
    resolved = load_object_from_path(spec, "function")
    if not callable(resolved):
        raise ValueError(
            f"'{spec}' resolved to {type(resolved).__name__}, not a callable"
        )
    return resolved


def load_class_from_name(name: str, search_modules: list[str]) -> type:
    """Load a class by searching in multiple modules."""
    for module_name in search_modules:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, name):
                return getattr(module, name)
        except ImportError as exc:
            if not _is_missing_module_error(exc, module_name):
                raise
            continue

    raise ImportError(
        f"Could not find class '{name}' in any of: {search_modules}. "
        f"Use full path like 'my_module.{name}' for custom classes."
    )


def resolve_class(class_spec: str, search_modules: list[str]) -> type:
    """
    Resolve a class from either:
    - A full path: "sieval.core.datasets.AIME2024Dataset"
    - A simple name: "AIME2024Dataset"
    """
    if class_spec.startswith("."):
        raise ValueError(
            f"Relative import syntax is not supported: '{class_spec}'. "
            f"Use a simple name ('MyClass') or full path ('pkg.module.MyClass')"
        )
    if "." in class_spec:
        # Looks like a full path
        return load_class_from_path(class_spec)
    else:
        # Simple name, search in modules
        return load_class_from_name(class_spec, search_modules)


def resolve_dataset_class(class_spec: str) -> type:
    """Resolve a dataset class."""
    return resolve_class(class_spec, [DATASET_MODULE])


def _is_missing_module_error(exc: ImportError, target_module: str) -> bool:
    """Check if an ImportError is due to the target module itself not existing.

    Returns True only when the missing module IS the one we tried to import
    (i.e. the module simply doesn't exist).  Returns False when the target
    module exists but one of its internal dependencies is missing — in that
    case the error should be propagated so the user sees the real cause.
    """
    missing = getattr(exc, "name", None)
    if missing is None:
        return False
    # The target module itself is missing, or one of its parent packages
    return missing == target_module or target_module.startswith(f"{missing}.")


def resolve_task_class(class_spec: str) -> type:
    """Resolve a task class by searching in sieval.tasks submodules."""
    if class_spec.startswith("."):
        raise ValueError(
            f"Relative import syntax is not supported: '{class_spec}'. "
            f"Use a simple name ('MyClass') or full path ('pkg.module.MyClass')"
        )
    # For tasks, we need to search in submodules of sieval.tasks
    if "." in class_spec:
        return load_class_from_path(class_spec)

    # Try to find in sieval.tasks submodules
    try:
        tasks_module = importlib.import_module(TASK_MODULE)
        # Check if directly available (re-exported in __init__)
        if hasattr(tasks_module, class_spec):
            return getattr(tasks_module, class_spec)
    except ImportError as exc:
        if not _is_missing_module_error(exc, TASK_MODULE):
            raise
        # sieval.tasks itself doesn't exist — fall through to heuristic search

    # Search in submodules based on naming convention
    # e.g., "AIME2024ZeroShotGenTask" -> "aime_2024_0shot_gen"
    submodule_candidates = _guess_submodule_names(class_spec)
    for submodule in submodule_candidates:
        full_module = f"{TASK_MODULE}.{submodule}"
        try:
            module = importlib.import_module(full_module)
            if hasattr(module, class_spec):
                return getattr(module, class_spec)
        except ImportError as exc:
            if not _is_missing_module_error(exc, full_module):
                raise
            continue

    raise ImportError(
        f"Could not find task class '{class_spec}'. "
        f"Use full path like 'sieval.tasks.my_task.{class_spec}' for custom tasks."
    )


def validate_named_config_map(
    section_name: str,
    section_cfg: Any,
) -> dict[str, dict[str, Any]]:
    """Validate a config section is a ``name -> dict`` mapping, and return it.

    Shared so every entry point that reads a section reports the same error for
    the same malformed config. ``derive_model_type`` is now reached from the
    infer layer *before* an ``EvalSession`` exists, and full config validation
    only runs under ``--dry-run``, so without this a list-shaped ``tasks:``
    surfaced as an ``AttributeError`` from inside recipe resolution.
    """
    if not isinstance(section_cfg, dict):
        raise ValueError(
            f"'{section_name}' configuration must be a dictionary "
            "mapping names to config"
        )

    for item_name, item_cfg in section_cfg.items():
        if not isinstance(item_cfg, dict):
            raise ValueError(
                f"'{section_name}.{item_name}' configuration must be a dictionary"
            )

    return section_cfg


def derive_model_type(
    model_name: str,
    explicit_type: str | None,
    requirements: AggregatedTaskRequirements,
) -> Literal["chat", "gen"]:
    """Derive one deployment root's legacy kind from normalized evidence.

    ``TaskModelRequirement.requires.input`` is the sole task-side authority.
    When that evidence exists, legacy YAML ``type:`` is a checked assertion.
    With no input evidence, an explicit value remains a compatibility fallback;
    otherwise the historical ``chat`` default applies.

    The resolver never inspects task ``model_type`` metadata, model wrapper
    classes, dialects, engines, URLs, or provider response shapes.
    """

    if not isinstance(model_name, str) or not model_name:
        raise TypeError("model_name must be a non-empty string")
    if explicit_type not in (None, "chat", "gen"):
        raise ValueError(
            f"Model deployment root '{model_name}' has invalid type "
            f"{explicit_type!r}; expected 'chat' or 'gen'"
        )
    if not isinstance(requirements, AggregatedTaskRequirements):
        raise TypeError("requirements must be AggregatedTaskRequirements")

    kinds = requirements.input
    if len(kinds) > 1:
        evidence = "\n".join(
            "  - "
            + kind.value
            + ": "
            + ", ".join(sorted(requirements.input_sources.get(kind, ())))
            for kind in sorted(kinds, key=lambda item: item.value)
        )
        raise ValueError(
            f"Model deployment root '{model_name}' has conflicting normalized "
            f"input requirements:\n{evidence}\n"
            "Chat and completion consumers must use separate root model configs."
        )

    if kinds:
        input_kind = next(iter(kinds))
        derived_type: Literal["chat", "gen"] = (
            "chat" if input_kind is InputKind.CHAT else "gen"
        )
        if explicit_type is not None and explicit_type != derived_type:
            sources = ", ".join(sorted(requirements.input_sources.get(input_kind, ())))
            source_detail = f" from {sources}" if sources else ""
            raise ValueError(
                f"Model deployment root '{model_name}' declares type: "
                f"{explicit_type}, but normalized TaskRequirements require "
                f"'{derived_type}' ({input_kind.value}){source_detail}. Legacy "
                "type is a checked assertion when task evidence exists."
            )
        logger.info(
            "Derived model deployment root '{}' type as '{}' from normalized "
            "task requirements",
            model_name,
            derived_type,
        )
        return derived_type

    if explicit_type is not None:
        return cast(Literal["chat", "gen"], explicit_type)
    logger.info(
        "Using default type 'chat' for model deployment root '{}' with no input "
        "requirement evidence",
        model_name,
    )
    return "chat"


@dataclass(frozen=True)
class ConfigModelTypeResolution:
    """Legacy model kinds derived from normalized task requirement hooks."""

    model_types_by_root: Mapping[str, Literal["chat", "gen"]]
    model_types_by_config: Mapping[str, Literal["chat", "gen"]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_types_by_root",
            MappingProxyType(dict(self.model_types_by_root)),
        )
        object.__setattr__(
            self,
            "model_types_by_config",
            MappingProxyType(dict(self.model_types_by_config)),
        )


def resolve_config_model_types(
    config: Mapping[str, Any],
) -> ConfigModelTypeResolution:
    """Adapt YAML config into the normalized evidence accepted by the resolver.

    This is the recipe-layer adapter for callers that run before an
    :class:`EvalSession` exists. It invokes each task's
    ``model_requirements_for()`` hook with dialect-free bindings and then calls
    :func:`derive_model_type`; it never reads legacy ``Task.model_type``.

    Import/class failures remain fail-soft here because normal configuration
    validation reports them with richer context later. A successfully resolved
    hook is validated strictly so it cannot replace the supplied binding.
    """

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    models = validate_named_config_map("models", config.get("models", {}))
    tasks = validate_named_config_map("tasks", config.get("tasks", {}))
    raw_datasets = config.get("datasets", {})
    datasets = validate_named_config_map("datasets", raw_datasets)

    chains = {name: _config_model_chain(name, models) for name in models}
    bindings = {name: _config_named_binding(name, chains[name]) for name in models}
    records: list[TaskModelRequirement] = []

    for task_name, task_config in tasks.items():
        class_spec = task_config.get("class")
        if not isinstance(class_spec, str) or not class_spec:
            continue
        try:
            task_class = resolve_task_class(class_spec)
        except (ImportError, AttributeError):
            continue

        model_name = _config_task_model_name(task_name, task_config, models)
        candidate = bindings[model_name]
        context = _config_requirement_context(
            task_name,
            task_config,
            task_class,
            candidate,
            datasets,
        )
        requirement_hook = getattr(task_class, "model_requirements_for", None)
        if not callable(requirement_hook):
            continue
        task_records = validate_task_model_requirements(
            task_class,
            context,
            requirement_hook(context),
        )
        records.extend(task_records)

    by_root_bindings: dict[str, list[NamedModelBinding]] = {}
    for binding in bindings.values():
        by_root_bindings.setdefault(binding.root_deployment_key, []).append(binding)

    by_root: dict[str, Literal["chat", "gen"]] = {}
    by_config: dict[str, Literal["chat", "gen"]] = {}
    for root_key, root_bindings in by_root_bindings.items():
        root_name = chains[root_bindings[0].config_name][0][0]
        root_records = (
            record
            for record in records
            if isinstance(record.binding, NamedModelBinding)
            and record.binding.root_deployment_key == root_key
        )
        requirements = aggregate_task_requirements(root_records)
        explicit_type = _config_explicit_type_for_root(
            root_name,
            tuple(root_bindings),
            models,
        )
        model_type = derive_model_type(root_name, explicit_type, requirements)
        by_root[root_key] = model_type
        for binding in root_bindings:
            by_config[binding.config_name] = model_type

    return ConfigModelTypeResolution(by_root, by_config)


def _config_model_chain(
    model_name: str,
    models: Mapping[str, dict[str, Any]],
) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Return one root-to-leaf model inheritance chain."""

    chain: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()

    def visit(current: str) -> None:
        if current in seen:
            raise ValueError(
                "Circular model inheritance detected: "
                + " -> ".join([name for name, _ in chain] + [current])
            )
        seen.add(current)
        try:
            model_config = models[current]
        except KeyError as exc:
            raise ValueError(
                f"Model '{model_name}' references unknown base model '{current}'"
            ) from exc
        chain.append((current, model_config))
        base = model_config.get("base")
        if base is None:
            return
        if not isinstance(base, str) or not base:
            raise ValueError(f"Model '{current}' has invalid 'base' value: {base!r}")
        visit(base)

    visit(model_name)
    chain.reverse()
    return tuple(chain)


def _config_named_binding(
    model_name: str,
    chain: tuple[tuple[str, dict[str, Any]], ...],
) -> NamedModelBinding:
    root_name, root_config = chain[0]
    requested = root_config.get("name")
    if not requested:
        infer_config = root_config.get("infer")
        checkpoint = (
            infer_config.get("checkpoint")
            if isinstance(infer_config, Mapping)
            else None
        )
        if not checkpoint:
            checkpoint = root_config.get("path")
        requested = Path(checkpoint).name if checkpoint else root_name
    if not isinstance(requested, str) or not requested:
        raise ValueError(f"Model '{root_name}' has no usable requested model id")
    return NamedModelBinding(
        binding_id=f"model:{model_name}",
        root_deployment_key=f"model:{root_name}",
        requested_model_id=requested,
        config_name=model_name,
    )


def _config_task_model_name(
    task_name: str,
    task_config: Mapping[str, Any],
    models: Mapping[str, dict[str, Any]],
) -> str:
    model_ref = task_config.get("model")
    if model_ref is not None and not isinstance(model_ref, str):
        raise ValueError(f"Task '{task_name}': 'model' must be a string reference")
    if model_ref:
        if model_ref not in models:
            raise ValueError(
                f"Task '{task_name}' references unknown model '{model_ref}'"
            )
        return model_ref
    if len(models) == 1:
        return next(iter(models))
    if not models:
        raise ValueError(f"Task '{task_name}': no models defined in config")
    raise ValueError(
        f"Task '{task_name}': 'model' required when multiple models are defined"
    )


def _config_requirement_context(
    task_name: str,
    task_config: Mapping[str, Any],
    task_class: type[Any],
    candidate: NamedModelBinding,
    datasets: Mapping[str, dict[str, Any]],
) -> RequirementContext:
    raw_args = task_config.get("args")
    if raw_args is None:
        raw_args = {}
    if not isinstance(raw_args, Mapping):
        raise ValueError(f"Task '{task_name}' args must be a dictionary")
    validate_task_config_args(task_name, raw_args, task_class=task_class)
    task_args = copy.deepcopy(dict(raw_args))
    model_bindings: dict[str, NamedModelBinding | InlineModelBinding] = {
        "candidate": candidate
    }

    for role in TASK_MODEL_ROLES:
        role_source = task_args.get(role)
        if is_task_model_role_sentinel(role, role_source):
            # Preserve the sentinel as task data. The task requirement hook
            # projects only the candidate, and task construction resolves the
            # extractor after candidate ``infer_args`` have been applied.
            continue
        task_args.pop(role, None)
        if role_source is None:
            continue
        if not isinstance(role_source, Mapping):
            raise ValueError(
                f"Task '{task_name}' {role} must be "
                + (
                    "'self' or an inline mapping before launch"
                    if role == "extractor"
                    else "an inline mapping before launch"
                )
            )
        model_bindings[role] = normalize_inline_model_binding(
            task_name,
            role,
            role_source,
        )

    dataset_ref = task_config.get("dataset")
    if isinstance(dataset_ref, str):
        dataset_config: Mapping[str, Any] = datasets.get(dataset_ref, {})
    elif isinstance(dataset_ref, Mapping):
        dataset_config = dataset_ref
    else:
        # Model-kind selection remains fail-soft for an incomplete task; the
        # normal config validator reports the missing dataset later.
        dataset_config = {}

    infer_args = task_config.get("infer_args")
    if infer_args is None:
        infer_args = {}
    if not isinstance(infer_args, Mapping):
        raise ValueError(f"Task '{task_name}' infer_args must be a dictionary")
    misplaced_resources = binding_resource_argument_paths(infer_args)
    if misplaced_resources:
        raise ValueError(
            f"Task '{task_name}' infer_args cannot change binding resources: "
            + ", ".join(misplaced_resources)
        )
    return RequirementContext(
        model_bindings=model_bindings,
        task_args=cast(Mapping[str, JSONValue], task_args),
        dataset_config=cast(Mapping[str, JSONValue], copy.deepcopy(dataset_config)),
        infer_args=cast(Mapping[str, JSONValue], copy.deepcopy(dict(infer_args))),
    )


def _config_explicit_type_for_root(
    root_name: str,
    bindings: tuple[NamedModelBinding, ...],
    models: Mapping[str, dict[str, Any]],
) -> Literal["chat", "gen"] | None:
    declarations = {
        binding.config_name: models[binding.config_name]["type"]
        for binding in bindings
        if "type" in models[binding.config_name]
    }
    invalid = {
        name: value
        for name, value in declarations.items()
        if value not in ("chat", "gen")
    }
    if invalid:
        details = ", ".join(
            f"{name}={value!r}" for name, value in sorted(invalid.items())
        )
        raise ValueError(
            f"Model deployment root '{root_name}' has invalid type assertion(s): "
            f"{details}; expected 'chat' or 'gen'"
        )
    values = set(declarations.values())
    if len(values) > 1:
        details = ", ".join(
            f"{name}={value!r}" for name, value in sorted(declarations.items())
        )
        raise ValueError(
            f"Models sharing deployment root '{root_name}' declare conflicting "
            f"type assertions: {details}"
        )
    if not values:
        return None
    return cast(Literal["chat", "gen"], next(iter(values)))


def _guess_submodule_names(class_name: str) -> list[str]:
    """
    Guess possible submodule names from a class name.
    e.g. "AIME2024ZeroShotGenTask" -> ["aime_2024_0shot_gen", "aime_2024_zero_shot_gen"]
    """
    import re

    # Remove 'Task' suffix if present
    name = class_name
    if name.endswith("Task"):
        name = name[:-4]

    # Convert CamelCase to snake_case with proper handling:
    # 1. Handle consecutive capitals followed by lowercase (e.g., "AIME" -> "AIME_")
    # 2. Handle lowercase/digit followed by uppercase (e.g., "tGen" -> "t_Gen")
    # 3. Handle letters followed by digits (e.g., "AIME2024" -> "AIME_2024")
    # 4. Handle digits followed by letters (e.g., "2024Zero" -> "2024_Zero")

    # Step 1: Insert underscore between consecutive capitals and following lowercase
    s1 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    # Step 2: Insert underscore between lowercase/digit and uppercase
    s2 = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s1)
    # Step 3: Insert underscore between letters and digits
    s3 = re.sub(r"([A-Za-z])(\d)", r"\1_\2", s2)
    # Step 4: Insert underscore between digits and letters
    s4 = re.sub(r"(\d)([A-Za-z])", r"\1_\2", s3)

    snake = s4.lower()

    candidates = []

    # Primary candidate: with "0shot" style
    if "zero_shot" in snake:
        candidates.append(snake.replace("zero_shot", "0shot"))
    if "few_shot" in snake:
        candidates.append(snake.replace("few_shot", "kshot"))

    # Also include the original snake_case version
    candidates.append(snake)

    return candidates
