"""
Config pre-validation and dry-run for ``sieval eval --dry-run``
and ``sieval run --dry-run``.

Provides three validation layers:
- **Schema validation** (``validate_eval_config``): static structure checks
- **Import validation** (``validate_eval_config_imports``): class import checks
- **Dry-run orchestration** (``run_dry_run``): structured result for CLI consumption

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import NotRequired, TypedDict

import yaml

from sieval.cli.leaderboard.session import (
    RootConfigDict,
    resolve_dataset_class,
    resolve_task_class,
)
from sieval.core.runners import TaskRunnerConfig
from sieval.core.tasks.consts import TaskAction

# Known top-level keys from RootConfigDict
_ROOT_KEYS: set[str] = set(RootConfigDict.__annotations__)

# Valid concurrency_limits keys — derived from TaskAction to stay in sync
_CONCURRENCY_KEYS: set[str] = {a.value for a in TaskAction}

# Valid dataset operations
_VALID_OPERATIONS: set[str] = {"slice", "shuffle", "repeat", "stratified_sample"}

# Operations renamed away from earlier names; map old -> new so stale configs
# get a migration hint instead of a bare "unknown operation".
_RENAMED_OPERATIONS: dict[str, str] = {
    "select": "slice",
}

# Valid TaskRunnerConfig field names
_RUNNER_CONFIG_FIELDS: set[str] = set(TaskRunnerConfig.__dataclass_fields__)


@dataclass
class ValidationResult:
    """Accumulated errors and warnings from config validation."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def merge(self, other: "ValidationResult") -> None:
        """Merge another result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


def _make_duplicate_key_checker(warnings: list[str]) -> Callable[..., dict]:
    """Create a construct_mapping override that records duplicate keys."""

    original = yaml.SafeLoader.construct_mapping

    def construct_mapping(loader, node, deep=False):
        loader.flatten_mapping(node)
        seen: set[str] = set()
        for key_node, _value_node in node.value:
            key = loader.construct_object(key_node, deep=False)
            if isinstance(key, str) and key in seen:
                mark = key_node.start_mark
                warnings.append(f"Duplicate key '{key}' at line {mark.line + 1}")
            if isinstance(key, str):
                seen.add(key)
        return original(loader, node, deep=deep)

    return construct_mapping


def load_yaml_with_duplicate_check(content: str) -> tuple[dict, list[str]]:
    """Parse YAML content, returning (parsed_dict, duplicate_key_warnings).

    Preserves last-wins behavior (consistent with ``yaml.safe_load``).
    """
    warnings: list[str] = []
    checker = _make_duplicate_key_checker(warnings)
    # Per-call subclass to avoid shared class-level state (thread-safe)
    loader_cls = type("_DuplicateKeyLoader", (yaml.SafeLoader,), {})
    loader_cls.construct_mapping = checker  # type: ignore[assignment]
    data = yaml.load(content, Loader=loader_cls)
    if data is None:
        data = {}
    return data, warnings


def _validate_structure(cfg: dict, result: ValidationResult) -> bool:
    """Validate top-level structure. Returns False if too broken to continue."""
    for key in cfg:
        if key not in _ROOT_KEYS:
            result.warnings.append(f"Unrecognized top-level key '{key}'")

    can_continue = True
    for section in ("models", "datasets", "tasks"):
        value = cfg.get(section, {})
        if not isinstance(value, dict):
            result.errors.append(
                f"'{section}' must be a dict, got {type(value).__name__}"
            )
            can_continue = False
        else:
            for item_name, item_cfg in value.items():
                if not isinstance(item_cfg, dict):
                    result.errors.append(
                        f"'{section}.{item_name}' must be a dict, "
                        f"got {type(item_cfg).__name__}"
                    )

    models = cfg.get("models", {})
    if isinstance(models, dict) and not models:
        result.warnings.append("'models' is empty — no model defined")

    tasks = cfg.get("tasks", {})
    if isinstance(tasks, dict) and not tasks:
        result.warnings.append("'tasks' is empty — nothing will run")

    return can_continue


def _validate_models(cfg: dict, result: ValidationResult) -> None:
    """Validate model configurations."""
    models = cfg.get("models", {})

    base_names: set[str] = set()
    derived: dict[str, str] = {}

    for name, mcfg in models.items():
        if not isinstance(mcfg, dict):
            continue  # already reported by _validate_structure
        has_name = "name" in mcfg
        has_base = "base" in mcfg

        if has_name and has_base:
            result.errors.append(f"Model '{name}': 'name' and 'base' cannot coexist")
            continue

        model_type = mcfg.get("type")
        if model_type is not None and model_type not in ("chat", "gen"):
            result.errors.append(
                f"Model '{name}': type must be 'chat' or 'gen', got '{model_type}'"
            )

        # `engine` selects the gen backend (mirrors _setup_models' guards; the
        # engine-on-non-gen check there uses the *inferred* type, so here we
        # can only flag the statically-decidable explicit `type: chat` case).
        if "engine" in mcfg:
            engine = mcfg.get("engine")
            if has_base:
                result.errors.append(
                    f"Model '{name}': derived models cannot set 'engine'; it is "
                    "inherited from the base model. Set it on the base instead."
                )
            elif engine not in ("vllm", "sglang"):
                result.errors.append(
                    f"Model '{name}': engine must be 'vllm' or 'sglang', got {engine!r}"
                )
            elif model_type == "chat":
                result.errors.append(
                    f"Model '{name}': 'engine' is only valid for type: gen, not 'chat'"
                )

        if has_base:
            base_ref = mcfg["base"]
            if not isinstance(base_ref, str) or not base_ref:
                result.errors.append(
                    f"Model '{name}': 'base' must be a non-empty string"
                )
            else:
                derived[name] = base_ref
        else:
            infer_dict = mcfg.get("infer")
            has_infer_dict = isinstance(infer_dict, dict)
            checkpoint = (infer_dict.get("checkpoint") or "") if has_infer_dict else ""
            path = mcfg.get("path") or ""

            # name is auto-injected from infer.checkpoint or path at
            # runtime (sieval run), so only error when neither source exists.
            if not has_name and not checkpoint and not path:
                result.errors.append(
                    f"Model '{name}': base model requires 'name' field"
                    " (or 'infer.checkpoint' / 'path' for auto-derivation)"
                )

            # `infer:` with nothing to serve and no api_base — `sieval run`
            # silently skips; surface it here so every entrypoint sees it.
            if (
                has_infer_dict
                and not mcfg.get("api_base")
                and not checkpoint
                and not path
            ):
                result.warnings.append(
                    f"Model '{name}': 'infer' section has no 'checkpoint' / "
                    "'path' and no 'api_base' — `sieval run` will skip "
                    "auto-serve. Add a checkpoint or set api_base explicitly."
                )

            base_names.add(name)

    all_model_names = set(models)
    for name, base_ref in derived.items():
        if base_ref not in all_model_names:
            result.errors.append(f"Model '{name}': base '{base_ref}' not found")

    resolved = set(base_names)
    pending = dict(derived)
    changed = True
    while changed and pending:
        changed = False
        for name in list(pending):
            if pending[name] in resolved:
                resolved.add(name)
                del pending[name]
                changed = True

    if pending:
        cycle_info = ", ".join(f"{n}->{pending[n]}" for n in sorted(pending))
        result.errors.append(f"Cyclic model dependencies: {cycle_info}")


def _validate_datasets(cfg: dict, result: ValidationResult) -> None:
    """Validate dataset configurations."""
    datasets = cfg.get("datasets", {})
    for name, dcfg in datasets.items():
        if not isinstance(dcfg, dict):
            continue  # already reported by _validate_structure
        if "class" not in dcfg:
            result.errors.append(f"Dataset '{name}': missing required field 'class'")
        operations = dcfg.get("operations")
        if operations is not None:
            if not isinstance(operations, list):
                result.errors.append(f"Dataset '{name}': 'operations' must be a list")
            else:
                _validate_operations(operations, name, result)


def _validate_operations(
    operations: list, dataset_name: str, result: ValidationResult
) -> None:
    """Validate dataset operation format."""
    for op in operations:
        if not isinstance(op, dict) or len(op) != 1:
            result.errors.append(
                f"Dataset '{dataset_name}': each operation must be a "
                f"single-key dict, got: {op!r}"
            )
            continue
        op_name = next(iter(op))
        if op_name in _RENAMED_OPERATIONS:
            result.errors.append(
                f"Dataset '{dataset_name}': operation '{op_name}' was renamed to "
                f"'{_RENAMED_OPERATIONS[op_name]}'; update your config."
            )
        elif op_name not in _VALID_OPERATIONS:
            result.errors.append(
                f"Dataset '{dataset_name}': unknown operation '{op_name}'. "
                f"Valid: {', '.join(sorted(_VALID_OPERATIONS))}"
            )
        op_args = op[op_name]
        if op_args is not None and not isinstance(op_args, dict):
            result.errors.append(
                f"Dataset '{dataset_name}': operation '{op_name}' "
                f"args must be a dict or null"
            )


def _validate_tasks(cfg: dict, result: ValidationResult) -> None:
    """Validate task configurations."""
    tasks = cfg.get("tasks", {})
    models = cfg.get("models", {})
    datasets = cfg.get("datasets", {})

    for name, tcfg in tasks.items():
        if not isinstance(tcfg, dict):
            continue  # already reported by _validate_structure
        class_spec = tcfg.get("class") or tcfg.get("task") or tcfg.get("task_class")
        if not class_spec:
            result.errors.append(f"Task '{name}': missing required field 'class'")

        ds_ref = tcfg.get("dataset")
        if isinstance(ds_ref, str):
            if ds_ref not in datasets:
                result.errors.append(f"Task '{name}': dataset '{ds_ref}' not found")
        elif isinstance(ds_ref, dict):
            if "class" not in ds_ref:
                result.errors.append(
                    f"Task '{name}': inline dataset missing 'class' field"
                )
            inline_ops = ds_ref.get("operations")
            if inline_ops is not None:
                if not isinstance(inline_ops, list):
                    result.errors.append(
                        f"Task '{name}': inline dataset 'operations' must be a list"
                    )
                else:
                    _validate_operations(inline_ops, f"{name}.dataset", result)
        elif ds_ref is None:
            result.errors.append(f"Task '{name}': missing required field 'dataset'")
        else:
            result.errors.append(f"Task '{name}': 'dataset' must be a string or dict")

        model_ref = tcfg.get("model")
        if model_ref is not None:
            if not isinstance(model_ref, str):
                result.errors.append(f"Task '{name}': 'model' must be a string")
            elif model_ref not in models:
                result.errors.append(f"Task '{name}': model '{model_ref}' not found")
        elif not models:
            result.errors.append(f"Task '{name}': no models defined in config")
        elif len(models) > 1:
            result.errors.append(
                f"Task '{name}': 'model' required when multiple models are defined"
            )


def _validate_runner_config(cfg: dict, result: ValidationResult) -> None:
    """Validate runner_config and concurrency_limits fields."""
    runner_cfg = cfg.get("runner_config")
    if runner_cfg is not None:
        if not isinstance(runner_cfg, dict):
            result.errors.append("'runner_config' must be a dict")
        else:
            for key in runner_cfg:
                if key not in _RUNNER_CONFIG_FIELDS:
                    result.warnings.append(f"runner_config: unknown field '{key}'")

    conc_limits = cfg.get("concurrency_limits")
    if isinstance(conc_limits, dict):
        for key in conc_limits:
            if key not in _CONCURRENCY_KEYS:
                result.warnings.append(f"concurrency_limits: unknown key '{key}'")


def _validate_unreferenced(cfg: dict, result: ValidationResult) -> None:
    """Warn about defined but unreferenced models/datasets."""
    tasks = cfg.get("tasks", {})
    models = cfg.get("models", {})
    datasets = cfg.get("datasets", {})

    referenced_models: set[str] = set()
    referenced_datasets: set[str] = set()

    for _name, tcfg in tasks.items():
        if not isinstance(tcfg, dict):
            continue
        model_ref = tcfg.get("model")
        if isinstance(model_ref, str):
            referenced_models.add(model_ref)
        ds_ref = tcfg.get("dataset")
        if isinstance(ds_ref, str):
            referenced_datasets.add(ds_ref)

    if len(models) == 1:
        referenced_models.update(models)

    for _name, mcfg in models.items():
        if not isinstance(mcfg, dict):
            continue
        base = mcfg.get("base")
        if isinstance(base, str):
            referenced_models.add(base)

    for name in models:
        if name not in referenced_models:
            result.warnings.append(
                f"Model '{name}' defined but not referenced by any task"
            )

    for name in datasets:
        if name not in referenced_datasets:
            result.warnings.append(
                f"Dataset '{name}' defined but not referenced by any task"
            )


def validate_eval_config(cfg: dict) -> ValidationResult:
    """Static schema validation — no imports, no instantiation."""
    result = ValidationResult()
    if not _validate_structure(cfg, result):
        return result
    _validate_models(cfg, result)
    _validate_datasets(cfg, result)
    _validate_tasks(cfg, result)
    _validate_runner_config(cfg, result)
    _validate_unreferenced(cfg, result)
    return result


def _is_builtin_class_spec(class_spec: str) -> bool:
    """Return True if the class spec refers to a built-in sieval class.

    Built-in specs are either simple names (resolved via ``sieval.datasets``
    or ``sieval.tasks``) or dotted paths rooted in the ``sieval.`` namespace.
    Third-party specs use a dotted path outside ``sieval.``.
    """
    if "." not in class_spec:
        return True  # simple name → searched in sieval.* modules
    return class_spec.startswith("sieval.")


def validate_eval_config_imports(cfg: dict) -> ValidationResult:
    """Import-level validation — tries to import task/dataset classes.

    Built-in class failures (simple names or ``sieval.*`` paths) are reported
    as **errors**. Third-party class failures (external dotted paths) degrade
    to **warnings** since the package may not be available in the current
    environment.

    Precondition: ``cfg`` must have passed ``validate_eval_config`` schema
    checks (i.e. ``tasks`` and ``datasets`` are dicts of dicts).
    """
    result = ValidationResult()

    task_specs: list[tuple[str, str]] = []  # (context_label, class_spec)
    dataset_specs: list[tuple[str, str]] = []

    tasks = cfg.get("tasks", {})
    datasets = cfg.get("datasets", {})
    if not isinstance(tasks, dict) or not isinstance(datasets, dict):
        return result

    for name, tcfg in tasks.items():
        if not isinstance(tcfg, dict):
            continue
        class_spec = tcfg.get("class") or tcfg.get("task") or tcfg.get("task_class")
        if class_spec:
            task_specs.append((f"task '{name}'", class_spec))
        ds_ref = tcfg.get("dataset")
        if isinstance(ds_ref, dict):
            ds_class = ds_ref.get("class")
            if ds_class:
                dataset_specs.append((f"task '{name}' inline dataset", ds_class))

    for name, dcfg in datasets.items():
        if not isinstance(dcfg, dict):
            continue
        ds_class = dcfg.get("class")
        if ds_class:
            dataset_specs.append((f"dataset '{name}'", ds_class))

    for label, spec in task_specs:
        try:
            resolve_task_class(spec)
        except Exception as exc:
            msg = f"Cannot import {label} class '{spec}': {exc}"
            if isinstance(exc, ValueError) or _is_builtin_class_spec(spec):
                result.errors.append(msg)
            else:
                result.warnings.append(msg)

    for label, spec in dataset_specs:
        try:
            resolve_dataset_class(spec)
        except Exception as exc:
            msg = f"Cannot import {label} class '{spec}': {exc}"
            if isinstance(exc, ValueError) or _is_builtin_class_spec(spec):
                result.errors.append(msg)
            else:
                result.warnings.append(msg)

    return result


# ---------------------------------------------------------------------------
# Dry-run orchestration
# ---------------------------------------------------------------------------


class DryRunCheck(TypedDict):
    """Single validation check result."""

    name: str
    ok: bool
    detail: NotRequired[str]
    warnings: NotRequired[list[str]]


class DryRunResult(TypedDict):
    """Structured result from run_dry_run."""

    checks: list[DryRunCheck]
    n_errors: int
    n_warnings: int


def run_dry_run(config: Path) -> DryRunResult:
    """Run config validation and return structured result."""
    checks: list[DryRunCheck] = []

    # 1. File existence
    if not config.exists():
        checks.append(
            {
                "name": "file_exists",
                "ok": False,
                "detail": f"Config file not found: {config}",
            }
        )
        return {"checks": checks, "n_errors": 1, "n_warnings": 0}

    checks.append({"name": "file_exists", "ok": True})

    # 2. YAML parse + duplicate key check
    content = config.read_text(encoding="utf-8")
    try:
        cfg, dup_warnings = load_yaml_with_duplicate_check(content)
    except yaml.YAMLError as exc:
        checks.append(
            {
                "name": "yaml_syntax",
                "ok": False,
                "detail": f"YAML syntax error: {exc}",
                "warnings": [],
            }
        )
        n_err = sum(1 for c in checks if not c["ok"])
        return {"checks": checks, "n_errors": n_err, "n_warnings": 0}

    checks.append(
        {
            "name": "yaml_syntax",
            "ok": True,
            "warnings": list(dup_warnings),
        }
    )

    # 3. Static schema validation
    result = validate_eval_config(cfg)

    # Structure summary
    models = cfg.get("models", {})
    datasets = cfg.get("datasets", {})
    tasks = cfg.get("tasks", {})
    detail = None
    if (
        isinstance(models, dict)
        and isinstance(datasets, dict)
        and isinstance(tasks, dict)
    ):
        detail = f"{len(models)} models, {len(datasets)} datasets, {len(tasks)} tasks"

    # 4. Import validation (skip if schema is broken)
    if result.ok:
        # Snapshot schema-specific warnings before merge adds import warnings.
        schema_warnings = list(result.warnings)

        import_result = validate_eval_config_imports(cfg)
        import_errors = list(import_result.errors)
        import_warnings = list(import_result.warnings)
        result.merge(import_result)

        schema_check: DryRunCheck = {"name": "schema", "ok": True}
        if detail:
            schema_check["detail"] = detail
        if schema_warnings:
            schema_check["warnings"] = schema_warnings
        checks.append(schema_check)

        import_ok = len(import_errors) == 0
        imports_check: DryRunCheck = {"name": "imports", "ok": import_ok}
        if not import_ok:
            imports_check["detail"] = "; ".join(import_errors)
        if import_warnings:
            imports_check["warnings"] = import_warnings
        checks.append(imports_check)
    else:
        schema_check: DryRunCheck = {"name": "schema", "ok": False}
        errors_detail = "; ".join(result.errors)
        if errors_detail:
            schema_check["detail"] = errors_detail
        schema_warnings = list(result.warnings)
        if schema_warnings:
            schema_check["warnings"] = schema_warnings
        checks.append(schema_check)

    n_err = sum(1 for c in checks if not c["ok"])
    all_warnings = list(dup_warnings) + list(result.warnings)
    n_warn = len(all_warnings)

    return {"checks": checks, "n_errors": n_err, "n_warnings": n_warn}
