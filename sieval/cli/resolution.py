"""Config-string → class resolution, and the model-type derivation built on it.

:func:`derive_model_type` is shared by the eval session and infer recipe
resolution, so it lives here with the class-resolution helpers it stands on
rather than in ``leaderboard/session.py`` — importing the session to reach it
would point the dependency sideways across the two CLI subpackages.
``scripts/check_layer_imports.py`` rejects that edge.

It imports nothing from sieval (``sieval.tasks`` / ``sieval.datasets`` are
reached through ``importlib`` at call time), so it is a leaf both subpackages can
depend on. That is about the dependency graph, not load time: ``sieval/cli``'s
own ``__init__`` builds the whole app, so importing any ``sieval.cli`` submodule
already pulls in the session either way.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import importlib
from collections.abc import Mapping
from typing import Any

from loguru import logger

# Registry for simple name lookups
DATASET_MODULE = "sieval.datasets"
TASK_MODULE = "sieval.tasks"


def load_class_from_path(class_path: str) -> type:
    """
    Load a class from a full module path like 'sieval.core.datasets.AIME2024Dataset'.
    """
    if "." not in class_path:
        raise ValueError(
            f"Invalid class path: {class_path}. Expected format: 'module.ClassName'"
        )

    module_name, class_name = class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except ImportError as exc:
        if _is_missing_module_error(exc, module_name):
            raise ImportError(f"Could not import module '{module_name}'") from exc
        # Internal dependency missing — propagate the original error
        raise
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_name}' has no class '{class_name}'"
        ) from e


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
    tasks_cfg: Mapping[str, Mapping[str, Any]],
) -> str:
    """Decide whether a config's model is a ``"chat"`` or ``"gen"`` model.

    Priority:
      1. The config's explicit ``type``.
      2. The ``model_type`` declared by the tasks pointing at this model.
      3. Default to ``"chat"``.

    Both callers — the eval session (which model class to construct) and infer
    recipe resolution (which capability layer to serve) — must reach the same
    answer for one model, so the derivation is shared rather than read off
    ``type`` twice. Explicit-only would silently mean "instruct" for every
    config in the wild, since ``type`` is normally left to this inference.

    Args:
        model_name: Name of the model in config.
        explicit_type: Explicitly specified type from config, if any.
        tasks_cfg: The config's ``tasks`` mapping, used for inference.

    Returns:
        Model type: ``"chat"`` or ``"gen"``.

    Raises:
        ValueError: If tasks pointing at this model require conflicting types,
            or if ``tasks_cfg`` is not a ``name -> dict`` mapping.
    """
    # 1. User explicitly specified
    if explicit_type is not None:
        return explicit_type

    # 2. Infer from tasks
    tasks_cfg = validate_named_config_map("tasks", tasks_cfg)
    required_types: set[tuple[str, str]] = set()

    for task_name, task_cfg in tasks_cfg.items():
        if task_cfg.get("model") != model_name:
            continue

        # Resolve task class to check its model_type attribute
        task_class_spec = task_cfg.get("class")
        if not task_class_spec:
            continue

        try:
            task_class = resolve_task_class(task_class_spec)
            task_model_type = getattr(task_class, "model_type", None)

            if task_model_type is not None:
                required_types.add((task_name, task_model_type))
        except (ImportError, AttributeError):
            # If we can't resolve the task class yet, skip it
            # Validation will catch issues later
            continue

    # Check for conflicts
    unique_types = {t for _, t in required_types}

    if len(unique_types) > 1:
        # Conflicting requirements
        conflict_info = "\n".join(
            f"  - {task_name} requires '{model_type}'"
            for task_name, model_type in sorted(required_types)
        )
        raise ValueError(
            f"Model '{model_name}' is used by tasks requiring different types:\n"
            f"{conflict_info}\n"
            f"Please either:\n"
            f"  1. Explicitly specify 'type: chat' or 'type: gen' in model config\n"
            f"  2. Use separate models for different types"
        )

    if len(unique_types) == 1:
        # All tasks agree on the same type
        inferred_type = unique_types.pop()
        logger.info(
            "Inferred model '{}' type as '{}' from task requirements",
            model_name,
            inferred_type,
        )
        return inferred_type

    # 3. Default to "chat"
    logger.info("Using default type 'chat' for model '{}'", model_name)
    return "chat"


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
