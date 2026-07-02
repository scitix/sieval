"""
YAML-driven batch evaluation session.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import contextlib
import copy
import dataclasses
import importlib
import os
import shlex
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypedDict, cast

import anyio
import yaml
from anyio.to_thread import run_sync
from loguru import logger

from sieval.cli.leaderboard.card import AlignmentCard, load_card
from sieval.core.datasets import Dataset
from sieval.core.models import ChatModel, GenModel, Model
from sieval.core.runners import MultiTaskRunner, TaskRunnerConfig
from sieval.core.tasks.context import TaskAction
from sieval.core.types import JSONValue
from sieval.infer.topology.models import DETERMINISTIC_DEFAULT_SEED

# Registry for simple name lookups
DATASET_MODULE = "sieval.datasets"
TASK_MODULE = "sieval.tasks"

# ── Narrow scalar types for YAML configuration ──
# Mirrors sieval.infer.config.ParamValue but defined locally to keep core/
# free of infer imports.
_ParamValue = str | int | float | bool


# Type Definitions for YAML Configuration
class _InferDict(TypedDict, total=False):
    """YAML-level infer configuration for a model (inline in core/)."""

    backend: str
    recipe: str
    checkpoint: str
    overrides: dict[str, _ParamValue]


class _InferMetaDict(TypedDict, total=False):
    """User-declared inference environment metadata for audit (inline in core/)."""

    framework: str
    dtype: str
    tp: int
    gpu: str


class ModelConfigDict(TypedDict, total=False):
    name: str  # For base models
    type: Literal["chat", "gen"]  # "chat" or "gen" (default: "chat")
    base: str  # For derived models
    args: dict[str, Any]
    api_key: str
    api_base: str
    infer: _InferDict  # infer config for `sieval infer`
    infer_meta: _InferMetaDict  # infer metadata for audit


# Use functional syntax to support "class" key which is a Python keyword
DatasetConfigDict = TypedDict(
    "DatasetConfigDict",
    {
        "class": str,
        "path": str,
        "args": dict[str, Any],
        "operations": list[dict[str, dict[str, Any]]],
    },
    total=False,
)


TaskConfigDict = TypedDict(
    "TaskConfigDict",
    {
        "class": str,
        "dataset": str | DatasetConfigDict,
        "model": str,
        "args": dict[str, Any],
        "infer_args": dict[str, JSONValue],  # per-task overrides (scalar + structured)
        "runner_config": dict[str, Any],
    },
    total=False,
)


class AlignmentBlockDict(TypedDict):
    card: str


class RootConfigDict(TypedDict, total=False):
    deterministic: bool
    result_dir: str
    concurrency_limit: int
    concurrency_limits: dict[
        TaskAction | Literal["preprocess", "infer", "postprocess", "feedback"], int
    ]
    runner_config: dict[str, Any]
    models: dict[str, ModelConfigDict]
    datasets: dict[str, DatasetConfigDict]
    tasks: dict[str, TaskConfigDict]
    alignment: AlignmentBlockDict


def _split_header(text: str) -> tuple[str, str]:
    """Partition ``text`` into ``(header, body)`` at the comment header block
    written by ``_format_comment_header``.

    Anchored to the ``# ---...`` border pair so only OUR header is split off,
    not arbitrary user-added top-of-file comments. A leading border with no
    closing border is treated as malformed and yields ``("", text)`` so body
    comparison detects the tampering instead of silently succeeding. When no
    header is present, returns ``("", text)``. In all cases ``header + body``
    reconstructs ``text`` exactly.
    """
    lines = text.splitlines(keepends=True)
    if not lines or not lines[0].startswith("# -"):
        return "", text
    for i in range(1, len(lines)):
        if lines[i].startswith("# -"):
            end = i + 1
            if end < len(lines) and lines[end].strip() == "":
                end += 1
            return "".join(lines[:end]), "".join(lines[end:])
    return "", text


def _strip_header(text: str) -> str:
    """Return ``text`` with the ``_format_comment_header`` block removed.

    Thin wrapper over :func:`_split_header`; see it for border/malformed
    semantics.
    """
    return _split_header(text)[1]


def _diff_lines(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Return ``- <path>: <old> → <new>`` lines for every differing leaf.

    Walks two parsed-config mappings depth-first. Empty list means the two
    parse to the same structure (any textual difference was whitespace /
    formatting only).
    """
    diffs: list[str] = []

    def _walk(x: Any, y: Any, path: str) -> None:
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                _walk(x.get(k), y.get(k), f"{path}.{k}" if path else k)
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                diffs.append(f"- {path}: list length {len(x)} → {len(y)}")
            else:
                for i, (xv, yv) in enumerate(zip(x, y, strict=True)):
                    _walk(xv, yv, f"{path}[{i}]")
        elif x != y:
            diffs.append(f"- {path}: {x!r} → {y!r}")

    _walk(a, b, "")
    return diffs


def _diff_dicts(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Return a short human-readable hint describing which keys differ.

    Thin wrapper over :func:`_diff_lines`; reports up to 10 differing leaf
    paths and a "formatting only" sentinel when nothing differs.
    """
    lines = _diff_lines(a, b)
    if not lines:
        return "Diff: (whitespace / formatting only)"
    return "Diff:\n" + "\n".join(f"  {line}" for line in lines[:10])


def _brief_diff(existing: str, current: str) -> str:
    """Return a short human-readable hint describing which YAML keys differ.

    Falls back to a generic message if either body fails to parse — this
    can happen when the persisted file has been hand-edited into invalid
    YAML, and we don't want the parse error to mask the caller's Resume
    aborted RuntimeError.
    """
    try:
        e = yaml.safe_load(existing) or {}
        c = yaml.safe_load(current) or {}
    except yaml.YAMLError:
        return "Diff: (existing file is not valid YAML — cannot compute key-level diff)"
    return _diff_dicts(e, c)


# ── Resume strict-match field policy (must partition TaskRunnerConfig) ──
# Adjustable across --resume only if a field touches neither the sample data
# nor any persisted artifact: pure scheduling + console-only progress.
_THROUGHPUT_RUNNER_KEYS: frozenset[str] = frozenset(
    {
        "concurrency_limit",
        "concurrency_limits",
        "shard_read_concurrency",
        "shard_write_concurrency",
        "write_buffer_size",
        "write_buffer_flush_interval",
        # Console-only (tqdm bar + log cadence); not the progress.json dump.
        "show_progress",
        "progress_log_interval",
        "progress_log_pct_interval",
    }
)

# Must match: affect sample data, an on-disk artifact, or a recorded outcome's
# meaning — e.g. max_retries is the failure signal written into FAILED records;
# profile_*/detect_anomalies*/dump_progress write profiler/anomaly/progress files.
_STRICT_RUNNER_KEYS: frozenset[str] = frozenset(
    {
        "shard_samples",
        "record_each_stage",
        "record_type_metadata",
        "record_meta",
        "max_iterations",
        "deterministic",
        "max_retries",
        "profile_io",
        "profile_stages",
        "profile_usage",
        "detect_anomalies",
        "detect_anomalies_on_resume",
        "dump_progress",
        "progress_dump_interval",
    }
)

# Neither adjustable nor strict — listed only so the three buckets partition
# TaskRunnerConfig exactly (see test_every_field_classified_exactly_once). The
# strip removes result_dir at top level (reification injects it there); the rest
# are never reached because they don't survive into a persisted runner_config
# block: auto_resume is set by the orchestration layer at runtime, stage_meta
# hooks are non-serializable callables. A hand-authored runner_config field from
# this set that changed across a resume would still be compared strictly.
_NONMATCH_RUNNER_KEYS: frozenset[str] = frozenset(
    {
        "result_dir",
        "auto_resume",
        "stage_meta_hook",
        "stage_meta_hooks",
    }
)


def _strip_noncomparable_fields(cfg: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy ``cfg`` with resume-mutable fields removed, for comparison.

    Strips (input never mutated) top-level ``concurrency_limit`` /
    ``concurrency_limits`` / ``result_dir``, ``models.*.args.concurrency_limit``,
    and ``_THROUGHPUT_RUNNER_KEYS`` from every ``runner_config`` block.
    """
    out = copy.deepcopy(cfg)

    for key in ("concurrency_limit", "concurrency_limits", "result_dir"):
        out.pop(key, None)

    models = out.get("models")
    if isinstance(models, dict):
        for mcfg in models.values():
            if isinstance(mcfg, dict):
                args = mcfg.get("args")
                if isinstance(args, dict):
                    args.pop("concurrency_limit", None)

    # runner_config carries throughput knobs in two equivalent places: the
    # top-level defaults block (merged into every task) and per-task overrides.
    # Strip both identically.
    runner_config_blocks = [out.get("runner_config")]
    tasks = out.get("tasks")
    if isinstance(tasks, dict):
        runner_config_blocks.extend(
            tcfg.get("runner_config")
            for tcfg in tasks.values()
            if isinstance(tcfg, dict)
        )
    for rc in runner_config_blocks:
        if isinstance(rc, dict):
            for key in _THROUGHPUT_RUNNER_KEYS:
                rc.pop(key, None)

    return out


def resolve_deterministic(cli_override: bool | None, config: Mapping[str, Any]) -> bool:
    """Effective deterministic flag: monotone OR of YAML and CLI.

    The CLI flag is one-way — can force on, cannot downgrade YAML.
    """
    return bool(config.get("deterministic", False)) or bool(cli_override)


def unwrap_proxies(obj: Any) -> Any:
    """Recursively convert dataclasses / MappingProxyType to YAML-safe dicts/lists.

    Why we can't use ``dataclasses.asdict`` upstream: on Python 3.13 it
    invokes ``copy.deepcopy`` on "other" values, which raises
    ``TypeError: cannot pickle 'mappingproxy' object`` for any frozen
    ``MappingProxyType`` (used by ``RoleAssignment.engine_params`` via
    ``_freeze_dict``). This walker sidesteps both the pickle dependency
    and the leftover ``MappingProxyType`` nodes that a successful
    ``asdict`` would leave behind.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: unwrap_proxies(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        }
    if isinstance(obj, MappingProxyType):
        return {k: unwrap_proxies(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {k: unwrap_proxies(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [unwrap_proxies(v) for v in obj]
    return obj


def _reify_cli_overrides(
    cfg: dict[str, Any],
    *,
    deterministic: bool | None = None,
    model: str | None = None,
    result_dir: str | None = None,
) -> dict[str, Any]:
    """Apply CLI overrides onto a config dict in place, return the same dict.

    Mirrors EvalSession's runtime override behavior so that `sieval eval
    <persisted_effective_config>` with NO CLI args reproduces the session:
        --deterministic → root `deterministic: true` + setdefault(seed=0) on
                          each base model's `args`
        --model X       → overwrite `name: X` on every base model
        --result-dir D  → root `result_dir: D`
    Per-op seeds (dataset `shuffle.seed`, task `args.seed`) are not
    CLI-overridable; users edit YAML directly for those.
    """
    if deterministic:
        cfg["deterministic"] = True
        models = cfg.get("models") or {}
        if isinstance(models, dict):
            for mcfg in models.values():
                if not isinstance(mcfg, dict) or "base" in mcfg:
                    continue
                args = mcfg.setdefault("args", {})
                if isinstance(args, dict):
                    args.setdefault("seed", DETERMINISTIC_DEFAULT_SEED)

    if model is not None:
        models = cfg.get("models") or {}
        if isinstance(models, dict):
            for mcfg in models.values():
                if isinstance(mcfg, dict) and "base" not in mcfg:
                    mcfg["name"] = model

    if result_dir is not None:
        cfg["result_dir"] = result_dir

    return cfg


def _apply_endpoint_injection(
    cfg: dict[str, Any], endpoint_map: Mapping[str, str]
) -> dict[str, Any]:
    """Inject api_base / api_key / auto-filled name for locally-served models.

    For each model in ``endpoint_map``:
        - Set ``api_base`` to the given endpoint (always overrides).
        - If no ``api_key`` present at either top level or inside ``args``,
          set ``api_key: "local"`` placeholder (OpenAI client needs one).
        - If ``name`` is absent, derive from checkpoint basename (from
          ``infer.checkpoint`` or top-level ``path``).
    """
    models = cfg.get("models")
    if not isinstance(models, dict):
        return cfg

    for model_key, endpoint in endpoint_map.items():
        mcfg = models.get(model_key)
        if not isinstance(mcfg, dict):
            continue

        mcfg["api_base"] = endpoint

        if not mcfg.get("api_key"):
            args = mcfg.get("args") or {}
            if not (isinstance(args, dict) and args.get("api_key")):
                mcfg["api_key"] = "local"

        if "name" not in mcfg:
            checkpoint = ""
            infer_dict = mcfg.get("infer") or {}
            if isinstance(infer_dict, dict):
                checkpoint = infer_dict.get("checkpoint", "")
            if not checkpoint:
                checkpoint = mcfg.get("path", "")
            if checkpoint:
                mcfg["name"] = Path(checkpoint).name

    return cfg


def _format_comment_header(
    *,
    title: str,
    source_config: str,
    invocation: str,
    extra_lines: list[str] | None = None,
) -> str:
    """Return a YAML comment block capturing provenance for an audit file.

    Standard lines (always present):
        title @ sieval <version> at <ISO-8601 UTC>
        Invocation: <argv joined with spaces>
        Original source: <abs path of the user's YAML>

    ``extra_lines`` — caller-supplied free-form lines (no leading ``#``)
    inserted between ``Original source`` and the closing border. Callers
    use this for artifact-specific hints.

    ``yaml.safe_dump`` cannot preserve comments, so this header is
    prepended to the dumped body via string concatenation.
    """
    from sieval import __version__

    now = datetime.now(UTC).isoformat()
    border = "# " + "-" * 70
    lines = [
        border,
        f"# {title} sieval {__version__} at {now}",
        f"# Invocation: {invocation}",
        f"# Original source: {source_config}",
    ]
    if extra_lines:
        lines.extend(f"# {line}" for line in extra_lines)
    lines.extend([border, ""])
    return "\n".join(lines) + "\n"


def _append_resume_note(header: str, diff_lines: list[str]) -> str:
    """Insert a ``Resumed by …`` audit block into ``header``, before its border.

    Called when ``--resume`` rewrites a file because only resume-mutable fields
    changed. The original provenance survives and ``diff_lines`` is recorded with
    a timestamp, so the header accumulates the full lineage across resumes. The
    note sits inside the ``# ---`` border pair so :func:`_split_header` keeps
    treating the whole block as the header next time.

    Assumes a well-formed ``header`` (two borders) — the only kind the caller
    passes (from :func:`_format_comment_header` or :func:`_split_header`).
    """
    from sieval import __version__

    now = datetime.now(UTC).isoformat()
    note = [f"# Resumed by sieval {__version__} at {now}:\n"]
    note.extend(f"#   {line}\n" for line in diff_lines)

    lines = header.splitlines(keepends=True)
    borders = [i for i, line in enumerate(lines) if line.startswith("# -")]
    close = borders[-1]
    return "".join(lines[:close] + note + lines[close:])


def _warn_best_effort_deterministic(
    config: Mapping[str, Any],
    effective_deterministic: bool,
    self_managed_endpoints: frozenset[str] | set[str],
) -> None:
    """Warn when deterministic mode talks to engines we don't manage.

    For models reaching an externally-hosted ``api_base`` we can pin the
    per-request ``seed`` but cannot verify the remote engine runs batch-
    invariant kernels. Reproducibility is best-effort on those models.

    Only base models with their own ``api_base`` are listed; derived
    models that inherit ``api_base`` from a flagged base are covered
    transitively by the base's warning.
    """
    if not effective_deterministic:
        return
    external = sorted(
        name
        for name, cfg in (config.get("models") or {}).items()
        if isinstance(cfg, dict)
        and cfg.get("api_base")
        and name not in self_managed_endpoints
    )
    if external:
        logger.warning(
            "Deterministic mode is best-effort for model(s) {} — "
            "sieval pins `seed` in each request but cannot verify "
            "batch-invariant kernels on the remote engine. For guaranteed "
            "reproducibility, self-host via `sieval run` / "
            "`sieval infer start` with a local checkpoint.",
            external,
        )


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


class EvalSession:
    """
    YAML-based evaluation session.

    Example YAML structure:
    ```yaml
    result_dir: "./outputs/my-run"

    models:
      base_model:
        name: "gpt-4o"
        args:
          temperature: 0.0
        # infer:                  # optional, used by `sieval infer`
        #     backend: vllm
        #     checkpoint: /path/to/weights
        # infer_meta:             # optional, for result auditing
        #     framework: vllm==0.6.0
      math_model:
        base: base_model
        args:
          temperature: 0.7

    datasets:
      aime_2024:
        class: AIME2024Dataset  # or full path
        path: "./data/aime_2024"
        operations:
          - shuffle: {seed: 42}
          - slice: {num: 100}

    tasks:
      aime_2024_eval:
        class: AIME2024ZeroShotGenTask  # or full path
        dataset: aime_2024
        model: math_model
        args:
          k: 1
          n: 64
        infer_args:               # optional, per-task inference overrides
          max_tokens: 512         # overrides model's default
        runner_config:
          concurrency_limits:
            infer: 4
    ```
    """

    def __init__(
        self,
        config_path: str | Path,
        model_override: str | None = None,
        resume: bool = False,
        result_dir_override: str | None = None,
        deterministic_override: bool | None = None,
        endpoint_map: Mapping[str, str] | None = None,
        infer_plans: Mapping[str, dict[str, Any]] | None = None,
        invocation: str | None = None,
        self_managed_endpoints: frozenset[str] | set[str] = frozenset(),
    ):
        self.config_path = Path(config_path)
        self.model_override = model_override
        self.resume_override = resume
        self.result_dir_override = result_dir_override
        self.deterministic_override = deterministic_override
        self._endpoint_map: Mapping[str, str] = endpoint_map or {}
        self._infer_plans: Mapping[str, dict[str, Any]] | None = infer_plans
        # Snapshot at init time so every audit file this session writes
        # carries the same string. Library/test callers pass explicit; CLI
        # falls back to sys.argv.
        self.invocation: str = (
            invocation if invocation is not None else shlex.join(sys.argv)
        )

        with open(self.config_path, encoding="utf-8") as f:
            loaded_config: RootConfigDict | None = yaml.safe_load(f)
            if loaded_config is None:
                loaded_config = {}
            if not isinstance(loaded_config, dict):
                raise ValueError("Top-level YAML config must be a dictionary")

        # Pristine YAML — source of truth for deterministic / result_dir
        # resolution. Deep-copied so downstream in-place mutation on
        # ``loaded_config`` cannot leak in.
        self._raw_config: RootConfigDict = copy.deepcopy(loaded_config)

        # Optional alignment block. Card path is stored verbatim (relative
        # to ``config_path.parent``) in both raw and reified views so
        # ``effective_config.yaml`` stays portable across machines.
        alignment_card: AlignmentCard | None = None
        alignment_block = loaded_config.get("alignment")
        if alignment_block is not None:
            if not isinstance(alignment_block, dict):
                raise ValueError(
                    f"Leaderboard YAML {self.config_path} `alignment` must be a mapping"
                )
            if "card" not in alignment_block:
                raise ValueError(
                    f"Leaderboard YAML {self.config_path} has `alignment` block "
                    f"without required sub-key `alignment.card`"
                )
            unknown = set(alignment_block) - {"card"}
            if unknown:
                raise ValueError(
                    f"Leaderboard YAML {self.config_path} `alignment` has unknown "
                    f"keys: {sorted(unknown)} (only `card` is supported)"
                )
            card_rel = alignment_block["card"]
            if not isinstance(card_rel, str) or not card_rel:
                raise ValueError(
                    f"Leaderboard YAML {self.config_path} `alignment.card` must "
                    f"be a non-empty string"
                )
            card_path = (self.config_path.parent / card_rel).resolve()
            alignment_card = load_card(card_path)
        self.alignment_card: AlignmentCard | None = alignment_card

        # Raw + CLI reification, BEFORE endpoint injection — this is what
        # gets persisted to effective_config.yaml, so rerun via `sieval run`
        # re-launches services instead of connecting to a stale endpoint.
        reified = _reify_cli_overrides(
            # cast: ty rejects TypedDict → dict[str, Any] and its natural shims.
            cast(dict[str, Any], copy.deepcopy(loaded_config)),
            deterministic=deterministic_override,
            model=model_override,
            result_dir=result_dir_override,
        )
        self._reified_config: dict[str, Any] = copy.deepcopy(reified)

        # Runtime view = reified + endpoint injection (mutates ``reified``).
        # cast: helper is typed dict[str, Any] for mutation; narrow at the boundary.
        self.config: RootConfigDict = cast(
            RootConfigDict, _apply_endpoint_injection(reified, self._endpoint_map)
        )

        self.deterministic: bool = resolve_deterministic(
            deterministic_override, self._raw_config
        )

        _warn_best_effort_deterministic(
            self.config, self.deterministic, self_managed_endpoints
        )

        self.models: dict[str, Model] = {}
        self.datasets: dict[str, Dataset] = {}

        # Resolved lazily in `_init_runner` at the start of `_prepare_execution`.
        self.result_dir: str | None = None
        self.runner: MultiTaskRunner | None = None

    def _init_runner(self) -> None:
        self.result_dir = self.result_dir_override or self.config.get("result_dir")

        self.runner = MultiTaskRunner(
            result_dir=self.result_dir,
            concurrency_limit=self.config.get("concurrency_limit"),
            concurrency_limits=self.config.get("concurrency_limits"),
            deterministic=self.deterministic,
        )

    def _get_named_config_map(self, section_name: str) -> dict[str, dict[str, Any]]:
        """Get a config section and validate it is a name -> dict mapping."""
        section_cfg = self.config.get(section_name, {})
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

    @staticmethod
    def _normalize_dict(value: Any, field_name: str) -> dict[str, Any]:
        """Normalize optional dict fields, rejecting invalid shapes."""
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} must be a dictionary")
        return value.copy()

    @staticmethod
    def _normalize_list(value: Any, field_name: str) -> list[Any]:
        """Normalize optional list fields, rejecting invalid shapes."""
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(f"{field_name} must be a list")
        return value

    def _infer_model_type(self, model_name: str, explicit_type: str | None) -> str:
        """
        Infer the model type based on task requirements.

        Priority:
        1. User explicitly specifies type in config
        2. Infer from tasks that use this model
        3. Default to "chat"

        Args:
            model_name: Name of the model in config
            explicit_type: Explicitly specified type from config (if any)

        Returns:
            Model type: "chat" or "gen"

        Raises:
            ValueError: If tasks require conflicting model types
        """
        # 1. User explicitly specified
        if explicit_type is not None:
            return explicit_type

        # 2. Infer from tasks
        tasks_cfg = self._get_named_config_map("tasks")
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

    def _setup_models(self) -> None:
        """Initialize all models from config."""
        models_cfg = self._get_named_config_map("models")

        # First pass: create base models (those without 'base' key)
        for name, cfg in models_cfg.items():
            if "base" in cfg:
                continue

            model_name = self.model_override or cfg.get("name")
            if not model_name:
                raise ValueError(
                    f"Model '{name}' requires 'name' or use --model CLI arg"
                )

            args = self._normalize_dict(cfg.get("args"), f"Model '{name}' args")

            if self.deterministic:
                args.setdefault("seed", DETERMINISTIC_DEFAULT_SEED)

            # Support top-level api_key and api_base in YAML
            if "api_key" in cfg:
                args["api_key"] = cfg["api_key"]
            if "api_base" in cfg:
                args["api_base"] = cfg["api_base"]

            # Infer model type with priority: explicit > inferred from tasks > default
            explicit_type = cfg.get("type")
            model_type = self._infer_model_type(name, explicit_type)

            if model_type == "gen":
                self.models[name] = GenModel(model=model_name, **args)
            elif model_type == "chat":
                self.models[name] = ChatModel(model=model_name, **args)
            else:
                raise ValueError(
                    f"Model '{name}' has invalid type '{model_type}'. "
                    "Expected 'chat' or 'gen'"
                )
            logger.info(
                "Created model '{}' with type='{}' name='{}'",
                name,
                model_type,
                model_name,
            )

        # Second pass: create derived models (those with 'base' key)
        pending_derived = {
            name: cfg for name, cfg in models_cfg.items() if "base" in cfg
        }
        while pending_derived:
            resolved_any = False

            for name in list(pending_derived):
                cfg = pending_derived[name]
                base_name = cfg.get("base")
                if not isinstance(base_name, str) or not base_name:
                    raise ValueError(
                        f"Model '{name}' has invalid 'base' value: {base_name!r}"
                    )
                if base_name not in self.models:
                    continue

                base_model = self.models[base_name]
                args = self._normalize_dict(cfg.get("args"), f"Model '{name}' args")

                derived_api_key = cfg.get("api_key")
                derived_api_base = cfg.get("api_base")

                if derived_api_key is not None or derived_api_base is not None:
                    raise ValueError(
                        f"Derived model '{name}' cannot override api_key/api_base from "
                        f"base model '{base_name}'. Create a new base model instead."
                    )

                # Extract concurrency_limit separately for with_args
                concurrency_limit = args.pop("concurrency_limit", None)

                # Check if type conversion is needed
                target_type = cfg.get("type")
                if target_type:
                    # Convert to target type
                    if target_type == "gen":
                        new_model = base_model.as_type(GenModel)
                    elif target_type == "chat":
                        new_model = base_model.as_type(ChatModel)
                    else:
                        raise ValueError(
                            f"Model '{name}' has invalid type '{target_type}'. "
                            "Expected 'chat' or 'gen'"
                        )
                    logger.info(
                        "Created derived model '{}' from '{}' "
                        "with type conversion to '{}'",
                        name,
                        base_name,
                        target_type,
                    )
                else:
                    # No type conversion, just derive
                    new_model = base_model

                # Apply additional args (including concurrency_limit)
                if concurrency_limit is not None or args:
                    new_model = new_model.with_args(
                        concurrency_limit=concurrency_limit, **args
                    )
                    if concurrency_limit is not None:
                        logger.info(
                            "Derived model '{}' reserves {} from '{}'",
                            name,
                            concurrency_limit,
                            base_name,
                        )
                    else:
                        logger.info(
                            "Created derived model '{}' from '{}'",
                            name,
                            base_name,
                        )
                else:
                    logger.info("Created derived model '{}' from '{}'", name, base_name)

                self.models[name] = new_model
                del pending_derived[name]
                resolved_any = True

            if resolved_any:
                continue

            for name, cfg in pending_derived.items():
                base_name = cfg.get("base")
                if not isinstance(base_name, str) or not base_name:
                    raise ValueError(
                        f"Model '{name}' has invalid 'base' value: {base_name!r}"
                    )
                if base_name not in models_cfg:
                    raise ValueError(
                        f"Model '{name}' references unknown base model '{base_name}'"
                    )

            cycle_info = ", ".join(
                sorted(
                    f"{name}->{cfg.get('base')}"
                    for name, cfg in pending_derived.items()
                )
            )
            raise ValueError(
                "Unable to resolve derived models due to cyclic dependencies: "
                f"{cycle_info}"
            )

    def _check_over_subscription(self) -> None:
        """Check for over-subscription and warn if detected."""
        # Find all base models (those without parent_limiter)
        base_models = {
            name: model
            for name, model in self.models.items()
            if model._limiter is not None and model._parent_limiter is None
        }

        for base_name, base_model in base_models.items():
            # Find all derived models from this base
            children = [
                (name, m)
                for name, m in self.models.items()
                if getattr(m, "_parent_limiter", None) is base_model._limiter
            ]

            if not children:
                continue

            base_limiter = base_model._limiter
            if base_limiter is None:
                continue
            base_quota = base_limiter.total_tokens
            child_quotas = [
                (name, m._limiter.total_tokens)
                for name, m in children
                if m._limiter is not None
            ]

            if child_quotas:
                total_reserved = sum(quota for _, quota in child_quotas)
                if total_reserved > base_quota:
                    child_info = ", ".join(
                        f"{name}={quota}" for name, quota in child_quotas
                    )
                    logger.warning(
                        "Over-subscription detected for model '{}': "
                        "total quota={}, but derived models reserve "
                        "{} ({}). "
                        "Actual concurrency will be capped at {}.",
                        base_name,
                        base_quota,
                        total_reserved,
                        child_info,
                        base_quota,
                    )

    def _setup_datasets(self) -> None:
        """Initialize all datasets from config."""
        datasets_cfg = self._get_named_config_map("datasets")

        for name, cfg in datasets_cfg.items():
            # Resolve class
            class_spec = cfg.get("class")
            if not class_spec:
                raise ValueError(f"Dataset '{name}' requires 'class' field")

            ds_class = resolve_dataset_class(class_spec)

            # Instantiate dataset
            path = cfg.get("path")
            # Expand ${VAR} so ${SIEVAL_DATA_DIR}/drop resolves; scoped to
            # `datasets.*.path` only to avoid surprising users whose model
            # names or args contain `$`.
            if path is not None:
                path = os.path.expandvars(path)
            init_args = self._normalize_dict(cfg.get("args"), f"Dataset '{name}' args")

            try:
                dataset = ds_class(path, **init_args) if path else ds_class(**init_args)
            except FileNotFoundError as exc:
                # Also catches `datasets.exceptions.DataFilesNotFoundError`.
                from sieval.core.datasets.meta import get_dataset_meta

                try:
                    meta = get_dataset_meta(ds_class)
                except AttributeError:
                    # Undecorated user-custom class: no registered name →
                    # skip the hint (would point at an invalid command).
                    meta = None
                hint = (
                    f"\n\nHint: run `sieval dataset download {meta.name}` first, "
                    f"then retry."
                    if meta is not None
                    else ""
                )
                # Wrap in RuntimeError + chain via `from exc` so callers
                # still see the OSError attrs (.filename, .errno) through
                # __cause__. Reconstructing via `type(exc)(msg)` would
                # discard them via the 1-arg constructor path.
                raise RuntimeError(f"{type(exc).__name__}: {exc}{hint}") from exc

            # Apply operations
            operations = self._normalize_list(
                cfg.get("operations"), f"Dataset '{name}' operations"
            )
            dataset = self._apply_dataset_operations(dataset, operations, name)

            self.datasets[name] = dataset
            logger.info("Created dataset '{}' with class '{}'", name, class_spec)

    def _apply_dataset_operations(
        self,
        dataset: Dataset,
        operations: list[dict],
        dataset_name: str,
    ) -> Dataset:
        """Apply a sequence of operations to a dataset."""
        for op in operations:
            if not isinstance(op, dict) or len(op) != 1:
                raise ValueError(
                    f"Dataset '{dataset_name}': Invalid operation format. "
                    f"Expected dict with single key, got: {op}"
                )

            op_name, op_args_raw = next(iter(op.items()))
            if op_args_raw is None:
                op_args: dict[str, Any] = {}
            elif not isinstance(op_args_raw, dict):
                raise ValueError(
                    f"Dataset '{dataset_name}': Operation '{op_name}' args "
                    "must be a dictionary"
                )
            else:
                op_args = op_args_raw.copy()

            match op_name:
                case "select":
                    raise ValueError(
                        f"Dataset '{dataset_name}': operation 'select' was renamed "
                        f"to 'slice'; update your config."
                    )

                case "slice":
                    num = op_args.get("num", op_args.get("n"))
                    split = op_args.get("split", "test")
                    if num is None:
                        raise ValueError(
                            f"Dataset '{dataset_name}': 'slice' requires 'num'"
                        )
                    dataset = dataset.slice(num, split=split)
                    logger.debug(
                        "Dataset '{}': sliced to first {} samples", dataset_name, num
                    )

                case "shuffle":
                    seed = op_args.get("seed", 0)
                    split = op_args.get("split", "test")
                    dataset = dataset.shuffle(seed=seed, split=split)
                    logger.debug(
                        "Dataset '{}': shuffled with seed={}",
                        dataset_name,
                        seed,
                    )

                case "repeat":
                    times = op_args.get("times", op_args.get("n"))
                    split = op_args.get("split", "test")
                    if times is None:
                        raise ValueError(
                            f"Dataset '{dataset_name}': 'repeat' requires 'times'"
                        )
                    dataset = dataset.repeat(times, split=split)
                    logger.debug("Dataset '{}': repeated {} times", dataset_name, times)

                case "stratified_sample":
                    by = op_args.get("by")
                    num = op_args.get("num", op_args.get("n"))
                    per_group = op_args.get("per_group")
                    min_per_group = op_args.get("min_per_group")
                    if by is None:
                        raise ValueError(
                            f"Dataset '{dataset_name}': 'stratified_sample' "
                            f"requires 'by'"
                        )
                    if (num is None) == (per_group is None):
                        raise ValueError(
                            f"Dataset '{dataset_name}': 'stratified_sample' "
                            f"requires exactly one of 'num' or 'per_group'"
                        )
                    if per_group is not None and min_per_group is not None:
                        raise ValueError(
                            f"Dataset '{dataset_name}': 'stratified_sample' "
                            f"'min_per_group' cannot be combined with 'per_group'"
                        )
                    seed = op_args.get("seed", 0)
                    split = op_args.get("split", "test")
                    dataset = dataset.stratified_sample(
                        by,
                        num=num,
                        per_group=per_group,
                        min_per_group=min_per_group,
                        seed=seed,
                        split=split,
                    )
                    logger.debug(
                        "Dataset '{}': stratified-sampled by '{}' ({}, seed={})",
                        dataset_name,
                        by,
                        (
                            f"per_group={per_group}"
                            if per_group is not None
                            else f"num={num}, min_per_group={min_per_group}"
                        ),
                        seed,
                    )

                case _:
                    raise ValueError(
                        f"Dataset '{dataset_name}': Unknown operation '{op_name}'. "
                        f"Valid operations: slice, shuffle, repeat, "
                        f"stratified_sample"
                    )

        return dataset

    def _setup_tasks(self) -> None:
        """Initialize all tasks from config."""
        tasks_cfg = self._get_named_config_map("tasks")
        runner_defaults_raw = self.config.get("runner_config", {})
        if not isinstance(runner_defaults_raw, dict):
            raise ValueError("'runner_config' configuration must be a dictionary")
        runner_defaults = runner_defaults_raw

        for task_name, task_cfg in tasks_cfg.items():
            # Resolve task class
            task_spec = task_cfg.get("class")
            if not task_spec:
                raise ValueError(f"Task '{task_name}' requires 'class' field")

            task_class = resolve_task_class(task_spec)

            # Resolve dataset
            dataset = self._resolve_task_dataset(task_cfg, task_name)

            # Resolve model
            model = self._resolve_task_model(task_cfg, task_name)

            # Apply infer_args override (per-task inference parameter override)
            infer_args = self._normalize_dict(
                task_cfg.get("infer_args"),
                f"Task '{task_name}' infer_args",
            )
            if infer_args:
                model = model.with_args(**infer_args)
                logger.info(
                    "Task '{}': applied infer_args override {}",
                    task_name,
                    infer_args,
                )

            # Create task instance
            task_args = self._normalize_dict(
                task_cfg.get("args", {}),
                f"Task '{task_name}' args",
            )

            task = task_class(
                name=task_name,
                dataset=dataset,
                model=model,
                **task_args,
            )

            # Build runner config
            runner_config = self._build_runner_config(task_cfg, runner_defaults)

            assert self.runner is not None, "Runner not initialized"
            self.runner.add_task(task, config=runner_config)
            logger.info("Added task '{}' with class '{}'", task_name, task_spec)

    def _resolve_task_dataset(
        self, task_cfg: TaskConfigDict, task_name: str
    ) -> Dataset:
        """Resolve dataset for a task - either by reference or inline definition."""
        # Option 1: Reference to pre-defined dataset
        dataset_ref = task_cfg.get("dataset")
        if isinstance(dataset_ref, str):
            if dataset_ref not in self.datasets:
                raise ValueError(
                    f"Task '{task_name}' references unknown dataset '{dataset_ref}'"
                )
            return self.datasets[dataset_ref]

        # Option 2: Inline dataset definition
        if isinstance(dataset_ref, dict):
            class_spec = dataset_ref.get("class")
            if not class_spec:
                raise ValueError(
                    f"Task '{task_name}': inline dataset requires 'class' field"
                )

            ds_class = resolve_dataset_class(class_spec)
            path = dataset_ref.get("path")
            # Mirror the top-level expansion so inline `tasks.*.dataset.path`
            # resolves `${SIEVAL_DATA_DIR}` identically.
            if path is not None:
                path = os.path.expandvars(path)
            init_args = self._normalize_dict(
                dataset_ref.get("args"), f"Task '{task_name}' inline dataset args"
            )
            dataset = ds_class(path, **init_args) if path else ds_class(**init_args)

            # Apply operations if any
            operations = self._normalize_list(
                dataset_ref.get("operations"),
                f"Task '{task_name}' inline dataset operations",
            )
            dataset = self._apply_dataset_operations(
                dataset, operations, f"{task_name}.dataset"
            )

            return dataset

        raise ValueError(
            f"Task '{task_name}': 'dataset' must be a string reference or inline definition"  # noqa: E501
        )

    def _resolve_task_model(self, task_cfg: TaskConfigDict, task_name: str) -> Model:
        """Resolve model for a task."""
        model_ref = task_cfg.get("model")

        if model_ref is not None and not isinstance(model_ref, str):
            raise ValueError(f"Task '{task_name}': 'model' must be a string reference")

        # If no model specified, use default if only one exists
        if not model_ref:
            if len(self.models) == 1:
                return next(iter(self.models.values()))
            elif len(self.models) == 0:
                raise ValueError(f"Task '{task_name}': no models defined in config")
            else:
                raise ValueError(
                    f"Task '{task_name}': 'model' required when multiple models are defined"  # noqa: E501
                )

        if model_ref not in self.models:
            raise ValueError(
                f"Task '{task_name}' references unknown model '{model_ref}'"
            )

        return self.models[model_ref]

    def _build_runner_config(
        self, task_cfg: TaskConfigDict, defaults: dict[str, Any]
    ) -> TaskRunnerConfig:
        """Build TaskRunnerConfig from task config and defaults."""
        # Start with defaults
        cfg_dict = dict(defaults)

        # Override with task-specific config
        task_runner_cfg = self._normalize_dict(
            task_cfg.get("runner_config"), "Task 'runner_config'"
        )
        cfg_dict.update(task_runner_cfg)

        # Handle resume override
        if self.resume_override:
            cfg_dict["auto_resume"] = True

        # Filter to valid TaskRunnerConfig fields
        valid_fields = set(TaskRunnerConfig.__dataclass_fields__.keys())
        cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_fields}

        return TaskRunnerConfig(**cfg_dict)

    async def _prepare_execution(self) -> None:
        """Asynchronous preparation pipeline."""
        logger.info("Loading config from: {}", self.config_path)

        self._init_runner()

        # Wrap in to_thread: dataset download / heavy model init can block
        # the event loop otherwise.
        await run_sync(self._setup_models)
        await run_sync(self._check_over_subscription)
        await run_sync(self._setup_datasets)
        await run_sync(self._setup_tasks)

        assert self.runner is not None, "Runner not initialized"
        logger.info("Starting {} tasks", len(self.runner._runners))

    def _resolve_result_dir(self) -> str | None:
        """Resolve target result_dir before ``_init_runner`` has run.

        Persistence + resume checks fire before the runner is constructed,
        so they can't read ``self.result_dir``.
        """
        return self.result_dir_override or self._raw_config.get("result_dir")

    async def _persist_yaml_with_strict_resume(
        self,
        *,
        target_name: str,
        body: str,
        header: str,
        audit_label: str,
        mutable_strip: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """Atomically write ``header + body`` to ``result_dir/target_name``.

        Under ``--resume`` with an existing file: an identical body skips the
        rewrite (timestamps survive). With ``mutable_strip=None`` (e.g. infer
        plans) any other diff raises. Otherwise both bodies are parsed and
        compared with ``mutable_strip`` applied; a diff that vanishes
        (resume-mutable or formatting) is tolerated — the file is rewritten
        with the new body, and the original header gains an appended
        ``Resumed by …`` record of what changed — and any residual diff
        raises. ``RuntimeError`` is the only failure the caller observes; all
        else is best-effort and logged.
        """
        effective_result_dir = self._resolve_result_dir()
        if effective_result_dir is None:
            logger.warning(
                "No result_dir configured; skipping {} persistence", target_name
            )
            return

        result_path = anyio.Path(effective_result_dir)
        target = result_path / target_name

        write_header = header

        if self.resume_override and await target.exists():
            try:
                existing = await target.read_text(encoding="utf-8")
            except OSError as e:
                raise RuntimeError(
                    f"Resume aborted: cannot read existing {target}: {e}\n"
                    "Either:\n"
                    "  1. Remove the result_dir and start fresh\n"
                    f"  2. Ensure {target} is readable"
                ) from e

            existing_header, existing_body = _split_header(existing)

            if existing_body == body:
                logger.info("Resume: {} matches — skipping rewrite", target_name)
                return

            if mutable_strip is None:
                # Byte-for-byte strict (e.g. infer plans): any diff aborts.
                raise RuntimeError(
                    f"Resume aborted: {target} does not match current invocation.\n"
                    f"{_brief_diff(existing_body, body)}\n"
                    "Either:\n"
                    "  1. Remove the result_dir and start fresh\n"
                    f"  2. Match the invocation to the persisted {audit_label}"
                )

            # Parse both sides so YAML type coercion (tuple→list) and key
            # ordering can't cause a spurious mismatch.
            try:
                existing_cfg = yaml.safe_load(existing_body) or {}
                current_cfg = yaml.safe_load(body) or {}
            except yaml.YAMLError as e:
                raise RuntimeError(
                    f"Resume aborted: cannot parse existing {target} to verify "
                    f"match: {e}\n"
                    "Either:\n"
                    "  1. Remove the result_dir and start fresh\n"
                    f"  2. Restore {target} to valid YAML matching the persisted "
                    f"{audit_label}"
                ) from e

            # current_cfg is always a dict (we dump a mapping); a tampered file
            # may parse to a scalar/list. Refuse cleanly so mutable_strip can't
            # raise an opaque AttributeError instead of the documented RuntimeError.
            if not isinstance(existing_cfg, dict) or not isinstance(current_cfg, dict):
                raise RuntimeError(
                    f"Resume aborted: existing {target} is not a YAML mapping — "
                    "cannot verify match.\n"
                    "Either:\n"
                    "  1. Remove the result_dir and start fresh\n"
                    f"  2. Restore {target} to valid YAML matching the persisted "
                    f"{audit_label}"
                )

            stripped_existing = mutable_strip(existing_cfg)
            stripped_current = mutable_strip(current_cfg)
            if stripped_existing != stripped_current:
                raise RuntimeError(
                    f"Resume aborted: {target} does not match current invocation.\n"
                    f"{_diff_dicts(stripped_existing, stripped_current)}\n"
                    "Either:\n"
                    "  1. Remove the result_dir and start fresh\n"
                    f"  2. Match the invocation to the persisted {audit_label}"
                )

            # Only resume-mutable (or formatting) fields differ — rewrite with
            # the new body. When real fields changed (not just formatting) and
            # the file had a header, append a timestamped record of the change
            # so the header keeps the full resume lineage. Otherwise no note is
            # added: a formatting-only diff keeps the original header, and a
            # header-less file gets a fresh one (it had no lineage to extend).
            logger.info(
                "Resume: {} resume-mutable fields changed — updating file",
                target_name,
            )
            # The note records genuine resume-mutable changes only; result_dir is
            # a never-compared location field (reification injects it), so drop it
            # to keep it out of the audit trail.
            note_before = {k: v for k, v in existing_cfg.items() if k != "result_dir"}
            note_after = {k: v for k, v in current_cfg.items() if k != "result_dir"}
            change_lines = _diff_lines(note_before, note_after)
            if existing_header and change_lines:
                write_header = _append_resume_note(existing_header, change_lines)
            else:
                write_header = existing_header or header

        tmp_path = target.with_name(target.name + ".tmp")
        content = write_header + body

        try:
            await result_path.mkdir(parents=True, exist_ok=True)
            async with await anyio.open_file(tmp_path, "w", encoding="utf-8") as f:
                await f.write(content)
            await tmp_path.replace(target)
            logger.info("Saved {} to: {}", audit_label, target)
        except Exception as e:
            with contextlib.suppress(OSError):
                await tmp_path.unlink(missing_ok=True)
            logger.error("Failed to save {}: {}", target_name, e)

    async def _persist_effective_config(self) -> None:
        """Write effective_config.yaml to result_dir at session start.

        Dumps ``self._reified_config`` (raw YAML + CLI reification, minus
        endpoint injections). User-supplied ``api_base`` / ``api_key`` in
        the source YAML ARE preserved — only ``endpoint_map``-driven
        auto-injection is excluded, so ``sieval run <this file>`` can
        re-launch services from the preserved ``path`` / ``infer`` fields.
        """
        body = yaml.safe_dump(
            self._reified_config,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        header = _format_comment_header(
            title="Persisted by",
            source_config=str(self.config_path.resolve()),
            invocation=self.invocation,
            extra_lines=[
                "Reproduce:",
                "  sieval run <this file>",
                "    — universal; re-launches auto-served models",
                "  sieval eval <this file>",
                "    — only when every model already has api_base",
            ],
        )
        await self._persist_yaml_with_strict_resume(
            target_name="effective_config.yaml",
            body=body,
            header=header,
            audit_label="effective config",
            mutable_strip=_strip_noncomparable_fields,
        )

    async def _persist_infer_plans(self) -> None:
        """Write infer_plans.yaml to result_dir when the caller supplied plans.

        Rerun re-resolves plans from the ``infer:`` section of
        effective_config.yaml plus the installed sieval version, so this file
        is audit-only — not load-bearing for rerun. Under ``--resume`` it IS
        still part of the strict-match contract: a re-resolved plan that
        differs from the persisted one (different GPU fleet, different sieval
        version emitting a different recipe translation) means resuming would
        merge results from incompatible deployments.
        """
        if not self._infer_plans:
            return

        payload = {"models": dict(self._infer_plans)}
        body = yaml.safe_dump(
            payload,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        header = _format_comment_header(
            title="Persisted by",
            source_config=str(self.config_path.resolve()),
            invocation=self.invocation,
            extra_lines=[
                "Reference only: audit log of the DeploymentPlan used",
                "  for each served model. Re-resolved at runtime from",
                "  effective_config.yaml + installed sieval version.",
            ],
        )
        await self._persist_yaml_with_strict_resume(
            target_name="infer_plans.yaml",
            body=body,
            header=header,
            audit_label="infer plans",
        )

    async def arun(self) -> dict[str, Any]:
        """Run all configured tasks asynchronously."""
        # Persist BEFORE _prepare_execution so a resume-mismatch abort
        # fails fast — avoid paying the model/dataset load cost just to
        # discover the config doesn't match the persisted one.
        await self._persist_effective_config()
        await self._persist_infer_plans()
        await self._prepare_execution()
        if self.runner is None:
            raise RuntimeError("Runner not initialized")
        return await self.runner.arun()

    def run(self) -> dict[str, Any]:
        """Run all configured tasks synchronously (blocking)."""
        return anyio.run(self.arun)


async def arun_session(
    config: str | Path,
    model: str | None = None,
    resume: bool = False,
    result_dir: str | None = None,
    deterministic: bool | None = None,
    endpoint_map: Mapping[str, str] | None = None,
    infer_plans: Mapping[str, dict[str, Any]] | None = None,
    invocation: str | None = None,
    self_managed_endpoints: frozenset[str] | set[str] = frozenset(),
) -> dict[str, Any]:
    """Run tasks defined in a YAML configuration file asynchronously.

    Args:
        config: Path to the YAML configuration file.
        model: Override model name for all base models.
        resume: Enable auto-resume for all tasks.
        result_dir: Override result directory.
        deterministic: Monotone override. ``None`` defers to YAML, ``True``
            forces on, ``False`` is a no-op (cannot downgrade YAML).
        endpoint_map: ``{model_name: endpoint_url}`` injected at runtime.
            Not persisted to effective_config.yaml.
        infer_plans: ``{model_name: DeploymentPlan-dict}`` for audit-level
            persistence to infer_plans.yaml.
        invocation: Provenance string for audit headers. ``None`` falls back
            to ``sys.argv`` at ``EvalSession.__init__`` time.
        self_managed_endpoints: Names of models whose ``api_base`` points at
            a sieval-launched engine — scopes the best-effort deterministic
            warning to genuinely external endpoints.

    Returns:
        A dictionary mapping task names to their reports.
    """
    runner = EvalSession(
        config_path=config,
        model_override=model,
        resume=resume,
        result_dir_override=result_dir,
        deterministic_override=deterministic,
        endpoint_map=endpoint_map,
        infer_plans=infer_plans,
        invocation=invocation,
        self_managed_endpoints=self_managed_endpoints,
    )

    return await runner.arun()


def run_session(
    config: str | Path,
    model: str | None = None,
    resume: bool = False,
    result_dir: str | None = None,
    deterministic: bool | None = None,
    endpoint_map: Mapping[str, str] | None = None,
    infer_plans: Mapping[str, dict[str, Any]] | None = None,
    invocation: str | None = None,
    self_managed_endpoints: frozenset[str] | set[str] = frozenset(),
) -> dict[str, Any]:
    return anyio.run(
        arun_session,
        config,
        model,
        resume,
        result_dir,
        deterministic,
        endpoint_map,
        infer_plans,
        invocation,
        self_managed_endpoints,
    )
