"""
Config pre-validation and dry-run for ``sieval eval --dry-run``
and ``sieval run --dry-run``.

Provides four validation layers:
- **Schema validation** (``validate_eval_config``): static structure checks
- **Import validation** (``validate_eval_config_imports``): class import checks
- **Capability reconciliation**: task/binding compatibility before clients or launch
- **Dry-run orchestration** (``run_dry_run``): structured result for CLI consumption

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

import yaml

from sieval.cli._cli_model import CliModel, apply_cli_model
from sieval.cli._filter_spec import (
    check_arg_names,
    check_by,
    check_by_digest,
    check_values_source,
)
from sieval.cli.resolution import (
    binding_resource_argument_paths,
    model_config_extra,
    resolve_dataset_class,
    resolve_task_class,
    validate_configured_engine_id,
    validate_task_config_args,
)
from sieval.cli.session import EvalSession, RootConfigDict
from sieval.core.models import check_extra
from sieval.core.models.capabilities import (
    CAPABILITY_KEYS,
    CapabilityConfigError,
    DialectCapabilityStatus,
    Supported,
    normalize_capability_declarations,
)
from sieval.core.models.dialect_registry import (
    DialectNotImplemented,
    UnknownDialect,
    capability_decisions_for,
    get_dialect_spec,
)
from sieval.core.models.reconcile import ReconcileResult
from sieval.core.runners import TaskRunnerConfig
from sieval.core.tasks.consts import TaskAction
from sieval.core.tasks.python_requires import MissingPythonRequirementsError
from sieval.core.tasks.run_meta import STATED_SERVING_KEYS

# Known top-level keys from RootConfigDict
_ROOT_KEYS: set[str] = set(RootConfigDict.__annotations__)

# Valid concurrency_limits keys — derived from TaskAction to stay in sync
_CONCURRENCY_KEYS: set[str] = {a.value for a in TaskAction}

# Valid dataset operations
_VALID_OPERATIONS: set[str] = {
    "slice",
    "shuffle",
    "repeat",
    "filter",
    "stratified_sample",
}

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
    cast(Any, loader_cls).construct_mapping = checker
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


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonempty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _check_stated_serving(
    name: str, stated: object, *, launched: bool, result: ValidationResult
) -> None:
    """Refuse a root model's ``serving`` that is not a mapping of
    :data:`STATED_SERVING_KEYS`, or that describes an engine the run launches,
    whose serving SiEval records itself."""
    if not isinstance(stated, Mapping):
        return  # absent, or refused as not a mapping with the other root keys
    stated = cast(Mapping[str, object], stated)
    where = f"Model '{name}': 'serving'"
    if launched:
        result.errors.append(
            f"{where} describes an external endpoint; an engine `sieval run` "
            "launches records its own serving"
        )
        return
    unknown = sorted(set(stated) - set(STATED_SERVING_KEYS))
    if unknown:
        result.errors.append(
            f"{where} has unknown key(s) {', '.join(unknown)}; expected "
            f"{', '.join(STATED_SERVING_KEYS)}"
        )
    version = stated.get("engine_version")
    if version is not None and not _is_nonempty_str(version):
        result.errors.append(f"{where}.engine_version must be a non-empty string")


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

        if "service_role" in mcfg:
            service_role = mcfg.get("service_role")
            if not isinstance(service_role, str) or not service_role:
                result.errors.append(
                    f"Model '{name}': 'service_role' must be a non-empty string"
                )

        # `engine` is an optional identity assertion on the root deployment.
        # It is independent of the request dialect and legacy model type.
        if "engine" in mcfg:
            engine = mcfg.get("engine")
            if has_base:
                result.errors.append(
                    f"Model '{name}': derived models cannot set 'engine'; it is "
                    "inherited from the base model. Set it on the base instead."
                )
            else:
                try:
                    validate_configured_engine_id(
                        engine,
                        context=f"Model '{name}'",
                    )
                except (TypeError, ValueError) as exc:
                    result.errors.append(str(exc))

        # What the root serves: a derived model shares its weights and window.
        for key, kind, check in (
            ("context_limit", "a positive integer", _is_positive_int),
            ("weights", "a non-empty string", _is_nonempty_str),
            ("serving", "a mapping", lambda value: isinstance(value, dict)),
        ):
            if key not in mcfg:
                continue
            if has_base:
                result.errors.append(
                    f"Model '{name}': derived models cannot set '{key}'; it is "
                    "inherited from the base model. Set it on the base instead."
                )
            elif not check(mcfg[key]):
                result.errors.append(f"Model '{name}': '{key}' must be {kind}")

        raw_model_args = mcfg.get("args")
        for where, extra in (
            (f"Model '{name}' extra", mcfg.get("extra")),
            (
                f"Model '{name}' args.extra",
                raw_model_args.get("extra")
                if isinstance(raw_model_args, dict)
                else None,
            ),
        ):
            _check_model_extra(extra, where, result)
        try:
            model_config_extra(mcfg, f"Model '{name}'")
        except ValueError as exc:
            result.errors.append(str(exc))

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
            raw_args = mcfg.get("args")
            args_api_base = (
                raw_args.get("api_base") if isinstance(raw_args, dict) else None
            )
            if (
                has_infer_dict
                and not mcfg.get("api_base")
                and not args_api_base
                and not checkpoint
                and not path
            ):
                result.warnings.append(
                    f"Model '{name}': 'infer' section has no 'checkpoint' / "
                    "'path' and no 'api_base' — `sieval run` will skip "
                    "auto-serve. Add a checkpoint or set api_base explicitly."
                )

            # As `sieval run` decides: `args.api_base` does not stop a launch.
            _check_stated_serving(
                name,
                mcfg.get("serving"),
                launched=bool(checkpoint or path) and not mcfg.get("api_base"),
                result=result,
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


def _validate_capabilities(cfg: dict, result: ValidationResult) -> None:
    """Validate canonical dialect and capability declarations for each model.

    An explicit canonical dialect selects its descriptor and executable PR-1
    binder.  The temporary ``sglang_legacy`` pseudo-dialect is accepted without
    canonical capabilities until PR-5 replaces it.  If the dialect is omitted,
    only provider-neutral option shapes can be checked here; task-derived
    default selection remains composition's job.  This layer deliberately does
    not duplicate #59's model-kind resolver.
    """
    models = cfg.get("models", {})

    for name, mcfg in models.items():
        if not isinstance(mcfg, dict):
            continue  # already reported by _validate_structure

        has_dialect = "dialect" in mcfg
        raw_dialect = mcfg.get("dialect")
        dialect_id: str | None = None
        decisions = None

        if has_dialect:
            if not isinstance(raw_dialect, str) or not raw_dialect:
                result.errors.append(
                    f"Model '{name}': 'dialect' must be a non-empty string"
                )
                continue
            dialect_id = raw_dialect

        if dialect_id == "sglang_legacy":
            # TODO(PR-5): remove this pseudo-spec branch when the registered
            # sglang_native binder replaces the temporary legacy bypass.
            if "capabilities" not in mcfg:
                continue
            raw_capabilities = mcfg["capabilities"]
            if not isinstance(raw_capabilities, Mapping):
                result.errors.append(f"Model '{name}': capabilities must be a mapping")
            elif raw_capabilities:
                result.errors.append(
                    f"Model '{name}': the temporary sglang_legacy bypass "
                    "cannot declare canonical capabilities before the "
                    "sglang_native executable binder is available"
                )
            continue

        if dialect_id is not None:
            try:
                spec = get_dialect_spec(dialect_id)
                decisions = capability_decisions_for(dialect_id)
            except (UnknownDialect, DialectNotImplemented) as exc:
                result.errors.append(f"Model '{name}': {exc}")
                continue
            outcomes = spec.capability_outcomes
        else:
            # A task-derived chat/completion default is not available during
            # static schema validation.  An all-supported row invokes the same
            # typed parser without claiming dialect support; reconcile() repeats
            # the check against the selected dialect's real outcome row.
            outcomes = dict.fromkeys(
                CAPABILITY_KEYS,
                DialectCapabilityStatus.SUPPORTED,
            )
            dialect_id = "task-derived"

        if "capabilities" not in mcfg:
            continue

        try:
            declarations = normalize_capability_declarations(
                mcfg["capabilities"],
                dialect_id=dialect_id,
                outcomes=cast(Mapping[str, DialectCapabilityStatus | str], outcomes),
            )
        except CapabilityConfigError as exc:
            result.errors.append(f"Model '{name}': {exc}")
            continue

        if decisions is None:
            continue
        for key, declaration in declarations.items():
            if declaration is False:
                continue
            decision = decisions[key]
            if not isinstance(decision, Supported):
                # normalize_capability_declarations() already rejects enabled
                # unsupported outcomes; retain a defensive fail-loud branch if
                # registry metadata and executable decisions ever diverge.
                result.errors.append(
                    f"Model '{name}': dialect {dialect_id!r} has no executable "
                    f"binding for capability {key!r}"
                )
                continue
            try:
                decision.binding.validate_config(declaration)
            except (CapabilityConfigError, TypeError, ValueError) as exc:
                result.errors.append(
                    f"Model '{name}': invalid {key!r} configuration for "
                    f"dialect {dialect_id!r}: {exc}"
                )


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
        elif op_name == "filter" and isinstance(op_args, dict):
            _validate_filter_args(op_args, dataset_name, result)


def _validate_filter_args(
    op_args: dict, dataset_name: str, result: ValidationResult
) -> None:
    """Shape-check a ``filter`` operation.

    The checks are :mod:`sieval.cli._filter_spec`'s, which says what each one
    answers and what it leaves to the session. What is worth catching *here* is
    a config that could not run whatever the data turned out to be.
    """
    problem = check_by(op_args.get("by"))
    if problem is not None:
        result.errors.append(f"Dataset '{dataset_name}': {problem}")
    result.errors.extend(
        f"Dataset '{dataset_name}': {problem}"
        for problem in check_arg_names(op_args)
        + check_values_source(op_args)
        + check_by_digest(op_args)
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

        task_args = tcfg.get("args")
        if task_args is not None:
            if not isinstance(task_args, dict):
                result.errors.append(f"Task '{name}': 'args' must be a dictionary")
            else:
                # Resolve the class so the model-role check runs here too, not
                # only in the pre-launch path. Fail-soft: an unresolvable spec is
                # already reported by _validate_class_imports, and passing None
                # keeps the composition-owned-key half of the check working.
                task_class: type | None = None
                if isinstance(class_spec, str) and class_spec:
                    try:
                        task_class = resolve_task_class(class_spec)
                    except Exception:
                        task_class = None
                try:
                    validate_task_config_args(name, task_args, task_class=task_class)
                except (TypeError, ValueError) as exc:
                    result.errors.append(str(exc))

        infer_args = tcfg.get("infer_args")
        if infer_args is not None:
            if not isinstance(infer_args, dict):
                result.errors.append(
                    f"Task '{name}': 'infer_args' must be a dictionary"
                )
            else:
                misplaced = binding_resource_argument_paths(infer_args)
                if misplaced:
                    result.errors.append(
                        f"Task '{name}' infer_args cannot change binding resources: "
                        + ", ".join(misplaced)
                    )
                _check_model_extra(
                    infer_args.get("extra"), f"Task '{name}' infer_args.extra", result
                )


def _check_model_extra(extra: object, where: str, result: ValidationResult) -> None:
    """Report an ``extra`` whose documented keys are malformed
    (:func:`~sieval.core.models.check_extra`), such as a wrapper without
    ``{seq}``."""
    try:
        check_extra(extra, where)
    except ValueError as exc:
        result.errors.append(str(exc))


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
    """Validate static schema without imports or instantiation.

    Dialect selection and capability compatibility become authoritative only
    during prelaunch reconciliation; use :func:`run_dry_run` for that check.
    """
    result = ValidationResult()
    if not _validate_structure(cfg, result):
        return result
    _validate_models(cfg, result)
    _validate_capabilities(cfg, result)
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


def task_report_metrics_json(
    cfg: Mapping[str, Any], task: str
) -> dict[str, list[str]] | None:
    """*task*'s report metric names and patterns, ``None`` where unknown.

    See :mod:`sieval.core.tasks.report_metrics`. A class that does not resolve
    is unknown; the imports check reports it.

    Raises:
        TypeError: if the class's ``report_metrics`` is malformed.
    """
    from sieval.meta.report_metrics import task_report_metrics

    tasks = cfg.get("tasks")
    task_cfg = tasks.get(task) if isinstance(tasks, Mapping) else None
    spec = task_cfg.get("class") if isinstance(task_cfg, Mapping) else None
    if not isinstance(spec, str):
        return None
    try:
        task_cls = resolve_task_class(spec)
    except Exception:
        return None
    declared = task_report_metrics(task_cls)
    return declared.to_json() if declared is not None else None


# ---------------------------------------------------------------------------
# Dry-run orchestration
# ---------------------------------------------------------------------------


class DryRunCheck(TypedDict):
    """Single validation check result."""

    name: str
    ok: bool
    detail: NotRequired[str]
    warnings: NotRequired[list[str]]
    plan: NotRequired[dict[str, Any]]


class DryRunResult(TypedDict):
    """Structured result from run_dry_run."""

    checks: list[DryRunCheck]
    n_errors: int
    n_warnings: int


def _prelaunch_session(
    config: str | Path,
    *,
    model_override: str | None,
    resume: bool | str | Path | None,
    result_dir_override: str | None,
    deterministic_override: bool | None,
    infer_plans: Mapping[str, object] | None,
    invocation: str | None,
    self_managed_endpoints: frozenset[str],
    cli_model: CliModel | None,
) -> EvalSession:
    # Looked up at call time, so the session module stays the one seam.
    from sieval.cli.session import EvalSession, unwrap_proxies

    normalized_plans: dict[str, dict[str, Any]] | None = None
    if infer_plans is not None:
        normalized_plans = {}
        for model_name, plan in infer_plans.items():
            normalized = unwrap_proxies(plan)
            if not isinstance(normalized, dict):
                raise TypeError(
                    f"infer plan for {model_name!r} must normalize to a mapping"
                )
            normalized_plans[model_name] = normalized

    return EvalSession(
        config,
        model_override=model_override,
        resume=resume,
        result_dir_override=result_dir_override,
        deterministic_override=deterministic_override,
        infer_plans=normalized_plans,
        invocation=invocation,
        self_managed_endpoints=self_managed_endpoints,
        cli_model=cli_model,
    )


def prepare_prelaunch_reconciliation(
    config: str | Path,
    *,
    model_override: str | None = None,
    resume: bool | str | Path | None = None,
    result_dir_override: str | None = None,
    deterministic_override: bool | None = None,
    infer_plans: Mapping[str, object] | None = None,
    invocation: str | None = None,
    self_managed_endpoints: frozenset[str] = frozenset(),
    check_services: bool = False,
    check_python_requires: bool = False,
    cli_model: CliModel | None = None,
) -> ReconcileResult:
    """Run the shared pure prelaunch reconciliation and fail loud on errors.

    For a caller about to grade, *check_services* also checks each vendored
    service the run would start, and *check_python_requires* raises
    :class:`~sieval.core.tasks.python_requires.MissingPythonRequirementsError`
    for a task's unmet ``python_requires``. ``infer start`` asks for neither:
    it only serves a model.

    This root-level CLI seam keeps the ``infer`` subpackage off the eval
    engine.  It also converts typed infer-layer plans into the plain mapping
    :class:`EvalSession` consumes and persists.

    Exceptions deliberately propagate.  Only presentation boundaries such as
    :func:`run_dry_run` may turn them into structured diagnostics.
    """
    session = _prelaunch_session(
        config,
        model_override=model_override,
        resume=resume,
        result_dir_override=result_dir_override,
        deterministic_override=deterministic_override,
        infer_plans=infer_plans,
        invocation=invocation,
        self_managed_endpoints=self_managed_endpoints,
        cli_model=cli_model,
    )
    result = session.prepare_prelaunch(enforce_python_requires=check_python_requires)
    if check_services:
        # A vendored service that cannot start fails here, before any engine.
        from sieval.cli.services import check_vendored_services

        check_vendored_services(session.required_services)
    return result


def _dependencies_check(session: EvalSession) -> DryRunCheck:
    """Each task's declared ``python_requires`` against this interpreter."""
    unmet_by_task = session.unmet_python_requires
    plan = {
        task: [
            {"import": import_name, "requirement": requirement}
            for import_name, requirement in declared.items()
        ]
        for task, declared in session.python_requires.items()
        if declared
    }
    n_declared = sum(len(entries) for entries in plan.values())
    check: DryRunCheck = {"name": "dependencies", "ok": not unmet_by_task}
    if unmet_by_task:
        check["detail"] = str(MissingPythonRequirementsError(unmet_by_task))
        check["plan"] = {
            "unmet": {
                task: [
                    {
                        "import": entry.import_name,
                        "requirement": entry.requirement,
                        "reason": entry.reason,
                    }
                    for entry in unmet
                ]
                for task, unmet in unmet_by_task.items()
            },
            "declared": plan,
        }
    else:
        check["detail"] = f"{n_declared} declared Python packages installed"
        check["plan"] = {"unmet": {}, "declared": plan}
    return check


def _capability_reconcile_check(
    config: Path,
    *,
    model_override: str | None = None,
    resume: bool | str | Path | None = None,
    result_dir_override: str | None = None,
    deterministic_override: bool | None = None,
    infer_plans: Mapping[str, dict[str, Any]] | None = None,
    invocation: str | None = None,
    self_managed_endpoints: frozenset[str] = frozenset(),
    cli_model: CliModel | None = None,
) -> list[DryRunCheck]:
    """Run the same pure prelaunch reconcile used by normal execution.

    Returns the ``capability_reconcile`` check, then the ``dependencies`` check,
    which is absent when reconciliation fails (like any check not evaluated).
    """

    try:
        session = _prelaunch_session(
            config,
            model_override=model_override,
            resume=resume,
            result_dir_override=result_dir_override,
            deterministic_override=deterministic_override,
            infer_plans=infer_plans,
            invocation=invocation,
            self_managed_endpoints=self_managed_endpoints,
            cli_model=cli_model,
        )
        result = session.prepare_prelaunch(enforce_python_requires=False)
    except Exception as exc:
        return [
            {
                "name": "capability_reconcile",
                "ok": False,
                # Verbatim, so dry-run and run give one diagnostic contract.
                "detail": str(exc),
            }
        ]

    check: DryRunCheck = {
        "name": "capability_reconcile",
        "ok": True,
        "detail": (
            f"{len(result.binding_plans)} bindings, "
            f"{len(result.deployment_plans)} deployment roots"
        ),
        "plan": result.to_json_value(),
    }
    warnings = [
        f"{diagnostic.code}: {diagnostic.message}"
        for diagnostic in result.diagnostics
        if diagnostic.severity.value == "warning"
    ]
    if warnings:
        check["warnings"] = warnings
    return [check, _dependencies_check(session)]


def run_dry_run(
    config: Path,
    *,
    model_override: str | None = None,
    resume: bool | str | Path | None = None,
    result_dir_override: str | None = None,
    deterministic_override: bool | None = None,
    infer_plans: Mapping[str, dict[str, Any]] | None = None,
    invocation: str | None = None,
    self_managed_endpoints: frozenset[str] = frozenset(),
    cli_model: CliModel | None = None,
) -> DryRunResult:
    """Run config validation and return structured result.

    *cli_model* is added to the config first, shown as the ``cli_model`` check.
    """
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

    # 2b. The models: entry --checkpoint/--api-base becomes
    if cli_model is not None:
        try:
            if not isinstance(cfg, dict):
                raise ValueError("evaluation config must be a mapping")
            apply_cli_model(cfg, cli_model, model_override)
        except ValueError as exc:
            checks.append({"name": "cli_model", "ok": False, "detail": str(exc)})
            return {"checks": checks, "n_errors": 1, "n_warnings": len(dup_warnings)}
        checks.append(
            {"name": "cli_model", "ok": True, "plan": {"models": cfg["models"]}}
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

        # 5. Task/binding capability reconciliation. Import warnings here are
        # unresolved third-party classes, so composition cannot truthfully
        # inspect their TaskRequirements and the check remains unavailable.
        if import_ok and not import_warnings:
            prelaunch_checks = _capability_reconcile_check(
                config,
                model_override=model_override,
                resume=resume,
                result_dir_override=result_dir_override,
                deterministic_override=deterministic_override,
                infer_plans=infer_plans,
                invocation=invocation,
                self_managed_endpoints=self_managed_endpoints,
                cli_model=cli_model,
            )
            for prelaunch_check in prelaunch_checks:
                checks.append(prelaunch_check)
                result.warnings.extend(prelaunch_check.get("warnings", ()))
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
