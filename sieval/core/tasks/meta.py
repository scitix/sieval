"""
Task metadata value types, `@sieval_task` decorator, and registry.

v0.1 schema contract — partial freeze. Fields and values listed as frozen are
a consumer contract: renames, removals, or type changes require a
`meta/index.json` schema_version bump.

Frozen fields on TaskMeta:
    name, display_name, description, dataset (FK str), eval_mode, n_shot,
    deps_group, status.

Not frozen (may change within schema_version=1):
    tags values, model_type enum membership, reference_impl.notes,
    Python-side internals.

Frozen enum values:
    EvalMode: gen, ppl, clp.
    Status: stable, experimental, deprecated.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import importlib
import pkgutil
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal
from urllib.parse import urlparse

from sieval.core.datasets.meta import extract_sample_type

from .task import Task


class EvalMode(StrEnum):
    # GEN covers free-form generation / chat / letter-extracted MCQ /
    # executed code / LLM-judged — all driven by .generate().
    # PPL covers perplexity-based (logprobs).
    GEN = "gen"
    PPL = "ppl"
    # CLP: next-token conditional log-prob over a fixed set of option tokens
    # (single inference); vs PPL = full-sequence perplexity (n inferences).
    CLP = "clp"


@dataclass(frozen=True, slots=True)
class ReferenceImpl:
    """Alignment layer 1: points to the reference implementation this task mirrors."""

    source: str
    url: str
    notes: str = ""


Status = Literal["stable", "experimental", "deprecated"]


@dataclass(frozen=True, slots=True)
class TaskMeta:
    """Output of `@sieval_task`; do not construct directly — the `dataset`
    FK is resolved by the decorator from the sample-type generic."""

    name: str
    display_name: str
    description: str
    dataset: str  # FK → DatasetMeta.name
    eval_mode: EvalMode
    n_shot: int = 0
    tags: tuple[str, ...] = ()
    deps_group: str | None = None
    model_type: Literal["chat", "gen"] | None = None
    reference_impl: ReferenceImpl | None = None
    status: Status = "stable"


TASK_REGISTRY: dict[str, TaskMeta] = {}
_TASK_CLASSES: dict[str, type[Task]] = {}

_TASK_META_ATTR = "_sieval_task_meta"

# Per-host pinned-commit regex. Keyed on the lowercased hostname (no port,
# no www prefix). Any hostname matching one of these entries must satisfy
# the corresponding pattern; unknown hosts bypass the check.
_GITHUB_COM_PATTERN = re.compile(
    r"^https?://(?:www\.)?github\.com/[^/]+/[^/]+/(blob|tree|raw)/[0-9a-f]{7,40}/"
)
_PINNED_URL_PATTERNS: dict[str, re.Pattern[str]] = {
    "github.com": _GITHUB_COM_PATTERN,
    "www.github.com": _GITHUB_COM_PATTERN,
    "raw.githubusercontent.com": re.compile(
        r"^https?://raw\.githubusercontent\.com/[^/]+/[^/]+/[0-9a-f]{7,40}/"
    ),
    "gist.github.com": re.compile(
        r"^https?://gist\.github\.com/[^/]+/[0-9a-f]{7,40}/[0-9a-f]{7,40}"
    ),
}
_MAX_DESCRIPTION_LEN = 100


def _validate(
    name: str,
    display_name: str,
    description: str,
    n_shot: int,
    reference_impl: ReferenceImpl | None,
) -> None:
    if not name:
        raise ValueError("name must be non-empty")
    if not display_name:
        raise ValueError("display_name must be non-empty")
    if not description:
        raise ValueError("description must be non-empty")
    if len(description) > _MAX_DESCRIPTION_LEN:
        raise ValueError(
            f"description exceeds {_MAX_DESCRIPTION_LEN} chars ({len(description)})"
        )
    if n_shot < 0:
        raise ValueError(f"n_shot must be >= 0 (got {n_shot})")

    if reference_impl is not None:
        url = reference_impl.url
        parsed = urlparse(url)
        pattern = _PINNED_URL_PATTERNS.get(parsed.netloc)
        if pattern is not None and not pattern.match(url):
            raise ValueError(
                f"reference_impl.url {url!r} must be a pinned commit URL "
                f"(hex SHA in path) for host {parsed.netloc!r}; "
                f"mutable refs like 'main'/'master' are rejected"
            )


def sieval_task[T: type[Task]](
    *,
    name: str,
    display_name: str,
    description: str,
    eval_mode: EvalMode,
    n_shot: int = 0,
    tags: tuple[str, ...] = (),
    deps_group: str | None = None,
    model_type: Literal["chat", "gen"] | None = None,
    reference_impl: ReferenceImpl | None = None,
    status: Status = "stable",
) -> Callable[[T], T]:
    """Decorate a Task subclass to register its TaskMeta.

    Attaches the TaskMeta as `cls._sieval_task_meta` and stores it in
    `TASK_REGISTRY`. The `dataset` FK is resolved automatically by
    inspecting the Task's first generic argument (its sample TypedDict)
    and looking up the corresponding `@sieval_dataset`-decorated class.

    Also sets two runtime-facing class attrs: `cls.tags` (protocol set
    synthesized from `eval_mode` + `n_shot`, consumed by anomaly routing
    — distinct from the descriptive `tags` argument stored on
    `_sieval_task_meta`) and `cls.model_type`. Raises ValueError at import
    time for duplicate `name`.
    """
    _validate(
        name=name,
        display_name=display_name,
        description=description,
        n_shot=n_shot,
        reference_impl=reference_impl,
    )

    protocol_tags: frozenset[str] = frozenset(
        {eval_mode.value, "zero_shot" if n_shot == 0 else "few_shot"}
    )

    def decorator(cls: T) -> T:
        from sieval.core.datasets.meta import (
            DATASET_REGISTRY,
            SAMPLE_TO_DATASET,
            get_dataset_meta,
        )

        if name in DATASET_REGISTRY:
            raise ValueError(
                f"name {name!r} already registered as a Dataset; "
                f"names must be globally unique across Tasks and Datasets."
            )
        if name in TASK_REGISTRY:
            existing = TASK_REGISTRY[name]
            raise ValueError(
                f"Task name {name!r} already registered "
                f"(existing display_name={existing.display_name!r}); "
                f"task names must be globally unique."
            )

        # Reverse-lookup: sample type from generic base → Dataset class.
        sample_type = extract_sample_type(cls)
        dataset_cls = SAMPLE_TO_DATASET.get(sample_type)
        if dataset_cls is None:
            raise ValueError(
                f"No @sieval_dataset found for sample type "
                f"{sample_type.__qualname__!r}; register the Dataset first."
            )
        dataset_name = get_dataset_meta(dataset_cls).name

        meta = TaskMeta(
            name=name,
            display_name=display_name,
            description=description,
            dataset=dataset_name,
            eval_mode=eval_mode,
            n_shot=n_shot,
            tags=tags,
            deps_group=deps_group,
            model_type=model_type,
            reference_impl=reference_impl,
            status=status,
        )
        setattr(cls, _TASK_META_ATTR, meta)
        cls.tags = protocol_tags
        if model_type is not None:
            cls.model_type = model_type
        TASK_REGISTRY[name] = meta
        _TASK_CLASSES[name] = cls
        return cls

    return decorator


def get_task_meta(cls: type) -> TaskMeta:
    """Return the `TaskMeta` attached to `cls` (AttributeError if not registered)."""
    return getattr(cls, _TASK_META_ATTR)


def iter_task_metas() -> Iterator[TaskMeta]:
    """Iterate over all registered TaskMeta in insertion order."""
    return iter(TASK_REGISTRY.values())


def iter_task_entries() -> Iterator[tuple[type[Task], TaskMeta]]:
    """Iterate over registered (class, meta) pairs in insertion order."""
    for name, meta in TASK_REGISTRY.items():
        yield _TASK_CLASSES[name], meta


def get_task_class(name: str) -> type[Task]:
    """Return the Task subclass registered under *name*.

    Lazy-imports ``sieval.tasks.{name}`` on miss (decorator convention:
    one task per eponymous module) to avoid the full ``import_all_tasks()``
    cost. Raises ``KeyError`` if still unregistered after the import.

    After ``load_index()`` the ``_TASK_CLASSES`` cache is empty (the index is
    rebuilt from JSON and doesn't execute decorators), so the first call per
    *name* pays a real ``importlib.import_module`` cost. Acceptable for
    point lookups (``task show``); if a future caller does this per-row
    across the full registry, call ``import_all_tasks()`` once upfront.
    """
    if name not in _TASK_CLASSES:
        expected = f"{_TASKS_PACKAGE}.{name}"
        try:
            importlib.import_module(expected)
        except ModuleNotFoundError as e:
            # Suppress only the "no such task module" case. A nested
            # ModuleNotFoundError (task module's own `import` statement
            # failing) means the user has a real missing dependency —
            # surface it verbatim, or they'll debug a KeyError.
            if e.name != expected:
                raise
    return _TASK_CLASSES[name]


def lookup_task(name: str) -> "TaskMeta | None":
    """Look up a TaskMeta by registered name; None if not registered."""
    return TASK_REGISTRY.get(name)


def tasks_for_dataset(dataset_name: str) -> Iterator[TaskMeta]:
    """Yield all registered TaskMetas whose `dataset` FK matches `dataset_name`."""
    return (t for t in TASK_REGISTRY.values() if t.dataset == dataset_name)


_TASKS_PACKAGE = "sieval.tasks"


def import_all_tasks() -> None:
    """Import every ``sieval.tasks`` submodule to trigger registration.

    Dynamic import at call-time so no static ``core → tasks`` edge shows
    up in the layer-boundary AST check.
    """
    pkg = importlib.import_module(_TASKS_PACKAGE)
    for info in pkgutil.walk_packages(pkg.__path__, prefix=f"{_TASKS_PACKAGE}."):
        importlib.import_module(info.name)


def task_meta_to_dict(meta: TaskMeta) -> dict[str, Any]:
    """Convert a TaskMeta to a JSON-serializable dict."""
    return {
        "name": meta.name,
        "display_name": meta.display_name,
        "description": meta.description,
        "dataset": meta.dataset,
        "eval_mode": meta.eval_mode.value,
        "n_shot": meta.n_shot,
        "tags": list(meta.tags),
        "deps_group": meta.deps_group,
        "model_type": meta.model_type,
        "reference_impl": (
            {
                "source": meta.reference_impl.source,
                "url": meta.reference_impl.url,
                "notes": meta.reference_impl.notes,
            }
            if meta.reference_impl is not None
            else None
        ),
        "status": meta.status,
    }


def task_meta_from_dict(payload: dict[str, Any]) -> TaskMeta:
    """Reverse of ``task_meta_to_dict``: reconstruct from one index.json row.

    Pure deserializer — skips decorator-level validation (``_validate`` +
    sample-type dataset-FK lookup) since the index is release-authored.
    """
    ref = payload.get("reference_impl")
    reference_impl = (
        ReferenceImpl(
            source=ref["source"],
            url=ref["url"],
            notes=ref.get("notes", ""),
        )
        if ref is not None
        else None
    )
    return TaskMeta(
        name=payload["name"],
        display_name=payload["display_name"],
        description=payload["description"],
        dataset=payload["dataset"],
        eval_mode=EvalMode(payload["eval_mode"]),
        n_shot=payload.get("n_shot", 0),
        tags=tuple(payload.get("tags", ())),
        deps_group=payload.get("deps_group"),
        model_type=payload.get("model_type"),
        reference_impl=reference_impl,
        status=payload.get("status", "stable"),
    )
