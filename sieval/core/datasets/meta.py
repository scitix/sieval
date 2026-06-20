"""
Dataset metadata value types, `@sieval_dataset` decorator, and registry.

v0.1 schema contract — partial freeze. Fields and values listed as frozen are
a consumer contract: renames, removals, or type changes require a
`meta/index.json` schema_version bump.

Frozen fields on DatasetMeta:
    name, display_name, description, source (schemes hf:/url:), categories[].level1,
    deps_group, license.

Not frozen (may change within schema_version=1):
    tags values, Category.level2 vocabulary, Python-side internals.

Frozen enum values:
    Level1Category: LANGUAGE, KNOWLEDGE, LOGIC, MATHEMATICS, CODE, AGENT.

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

import importlib
import pkgutil
import re
import types
import typing
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse


class Level1Category(StrEnum):
    """Top-level task category; values mirror the YAML taxonomy verbatim."""

    LANGUAGE = "Language"
    KNOWLEDGE = "Knowledge"
    LOGIC = "Logic"
    MATHEMATICS = "Mathematics"
    CODE = "Code"
    AGENT = "Agent"


_VALID_LEVEL2: dict[Level1Category, frozenset[str]] = {
    Level1Category.LANGUAGE: frozenset(
        {
            "SentimentAnalysis",
            "IntentRecognition",
            "Translation",
            "Summarization",
            "SemanticUnderstanding",
            "ContentEvaluation",
            "CreativeWriting",
            "Dialogue",
            "InstructionFollowing",
        }
    ),
    Level1Category.KNOWLEDGE: frozenset(
        {
            "STEM",
            "Humanities",
            "SocialSciences",
            "CommonSense",
            "Multi-domain",
        }
    ),
    Level1Category.LOGIC: frozenset({"BasicLogic", "ComplexLogic", "TextualReasoning"}),
    Level1Category.MATHEMATICS: frozenset(
        {"ElementaryMath", "AdvancedMath", "AppliedMath", "CompetitionMath"}
    ),
    Level1Category.CODE: frozenset(
        {
            "MultiLanguageSupport",
            "CodeCompletion",
            "CodeQnA",
            "CodeGeneration",
            "CodingInterview",
        }
    ),
    Level1Category.AGENT: frozenset(
        {
            "ToolUseSimple",
            "ToolUseMulti",
            "DataAnalysisAgent",
            "SoftwareEngineeringAgent",
        }
    ),
}


@dataclass(frozen=True, slots=True)
class Category:
    """Task classification; `level2` is validated against `_VALID_LEVEL2[level1]`."""

    level1: Level1Category
    level2: str | None = None


# ---------------------------------------------------------------------------
# DatasetMeta, registries, and @sieval_dataset
# ---------------------------------------------------------------------------

_VALID_SCHEMES = ("hf:", "url:", "local:")
_MAX_DESCRIPTION_LEN = 100
_DATASET_META_ATTR = "_sieval_dataset_meta"

DATASET_REGISTRY: dict[str, "DatasetMeta"] = {}
SAMPLE_TO_DATASET: dict[type, type] = {}


@dataclass(frozen=True, slots=True)
class DatasetMeta:
    """Immutable metadata for a registered evaluation dataset.

    `deps_group` is the *loader-side* optional-deps group (extras needed to
    load/parse this dataset). Evaluator-side deps (scorers, parsers) live on
    `TaskMeta.deps_group` instead — the two are independent.
    """

    name: str
    display_name: str
    description: str
    source: tuple[str, ...]
    categories: tuple[Category, ...]
    tags: tuple[str, ...] = ()
    deps_group: str | None = None
    license: str | None = None
    checksums: tuple[tuple[str, str], ...] = ()


def _normalize_source(source: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Normalize decorator input (str | tuple | list) to the stored tuple form.

    A bare string on the decorator is a one-source convenience; ``DatasetMeta``
    stores a tuple uniformly so consumers never branch on origin.
    """
    if isinstance(source, str):
        return (source,)
    return tuple(source)


def _validate(
    name: str,
    display_name: str,
    description: str,
    categories: tuple[Category, ...],
    source: tuple[str, ...],
) -> None:
    """Validate DatasetMeta fields; raise ValueError on any violation."""
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
    if not categories:
        raise ValueError("at least one category required")

    for cat in categories:
        if cat.level2 is None:
            continue
        allowed = _VALID_LEVEL2[cat.level1]
        if cat.level2 not in allowed:
            raise ValueError(
                f"level2 {cat.level2!r} is not valid for "
                f"Level1Category.{cat.level1.name} ({cat.level1.value!r}); "
                f"valid level2 values: {sorted(allowed)}"
            )

    for src in source:
        if not any(src.startswith(s) for s in _VALID_SCHEMES):
            raise ValueError(
                f"source {src!r} must use scheme from "
                f"{_VALID_SCHEMES} (e.g. 'hf:org/name')"
            )

    _validate_url_basenames_unique(name, source)


def url_path_basename(url: str) -> str:
    """Path basename of *url* (``url:`` prefix pre-stripped); empty for
    trailing-slash paths — shared primitive with the URL downloader so the
    collision check here and the on-disk filename there never drift."""
    return urlparse(url).path.rsplit("/", 1)[-1]


def _validate_url_basenames_unique(name: str, source: tuple[str, ...]) -> None:
    """Reject duplicate basenames among url: sources in one dataset — two URLs
    sharing a basename would overwrite each other at ``<dest>/<name>/<basename>``."""
    basenames = [
        url_path_basename(src[len("url:") :])
        for src in source
        if src.startswith("url:")
    ]
    counter = Counter(basenames)
    duplicates = {b for b, count in counter.items() if count > 1}
    if duplicates:
        raise ValueError(
            f"url: sources in dataset {name!r} have colliding basenames: "
            f"{sorted(duplicates)}; each URL must produce a unique on-disk filename"
        )


_CHECKSUM_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _normalize_checksums(
    checksums: dict[str, str] | None,
) -> tuple[tuple[str, str], ...]:
    """Normalize the decorator's dict to a sorted tuple-of-pairs (mirrors
    ``_normalize_source``); keeps ``DatasetMeta`` immutable and ``index.json``
    deterministic."""
    if not checksums:
        return ()
    return tuple(sorted(checksums.items()))


def _validate_checksums(
    name: str,
    source: tuple[str, ...],
    checksums: tuple[tuple[str, str], ...],
) -> None:
    """Each value is a ``sha256:<hex>`` digest; each key is the basename of a
    declared ``url:`` source in this dataset."""
    if not checksums:
        return
    url_basenames = {
        url_path_basename(src[len("url:") :])
        for src in source
        if src.startswith("url:")
    }
    for basename, digest in checksums:
        if not _CHECKSUM_RE.match(digest):
            raise ValueError(
                f"dataset {name!r} checksum for {basename!r} must be "
                f"'sha256:<64 hex>', got {digest!r}"
            )
        if basename not in url_basenames:
            raise ValueError(
                f"dataset {name!r} checksum key {basename!r} is not the basename "
                f"of any declared url: source {sorted(url_basenames)}"
            )


def extract_sample_type(cls: type) -> type:
    """First concrete (non-TypeVar) generic arg on any parameterized base in
    the MRO. Shared by ``@sieval_dataset`` (``Dataset[TSample]``) and
    ``@sieval_task`` (``Task[TSample, ...]``).

    MRO-walking picks up a ``Dataset[...]`` / ``Task[...]`` declaration from
    an intermediate abstract base. The TypeVar filter rejects re-parameterized
    bases that would otherwise leak a bare TypeVar into the registry key.
    """
    for klass in cls.__mro__:
        for base in types.get_original_bases(klass):
            args = typing.get_args(base)
            if args and not isinstance(args[0], typing.TypeVar):
                return args[0]
    raise ValueError(
        f"Cannot determine sample type for {cls.__qualname__!r}: no concrete "
        "generic args found on any base. Ensure the class (or one of its bases) "
        "is declared as `BaseClass[YourSampleType]` with a concrete TypedDict "
        "argument (not a bare base or a type variable)."
    )


def sieval_dataset[T: type](
    *,
    name: str,
    display_name: str,
    description: str,
    source: str | tuple[str, ...] | list[str],
    categories: tuple[Category, ...],
    tags: tuple[str, ...] = (),
    deps_group: str | None = None,
    license: str | None = None,
    checksums: dict[str, str] | None = None,
) -> Callable[[T], T]:
    """Decorate a Dataset subclass to register its DatasetMeta.

    Attaches the DatasetMeta as `cls._sieval_dataset_meta` and stores it in
    `DATASET_REGISTRY`. Also registers the sample TypedDict → dataset class
    mapping in `SAMPLE_TO_DATASET` for reverse lookup by `@sieval_task`.
    Raises ValueError at import time for duplicate name or duplicate sample type.

    `source` accepts a bare string (one-source convenience), tuple, or list;
    it is normalized to a tuple and stored as such on the DatasetMeta.
    """
    normalized_source = _normalize_source(source)
    _validate(
        name=name,
        display_name=display_name,
        description=description,
        categories=categories,
        source=normalized_source,
    )
    normalized_checksums = _normalize_checksums(checksums)
    _validate_checksums(name, normalized_source, normalized_checksums)
    meta = DatasetMeta(
        name=name,
        display_name=display_name,
        description=description,
        source=normalized_source,
        categories=categories,
        tags=tags,
        deps_group=deps_group,
        license=license,
        checksums=normalized_checksums,
    )

    def decorator(cls: T) -> T:
        # Cross-registry uniqueness (delayed import to avoid module cycle)
        from sieval.core.tasks.meta import TASK_REGISTRY

        if name in TASK_REGISTRY:
            raise ValueError(
                f"name {name!r} already registered as a Task; "
                f"names must be globally unique across Tasks and Datasets."
            )
        if name in DATASET_REGISTRY:
            existing = DATASET_REGISTRY[name]
            raise ValueError(
                f"Dataset name {name!r} already registered "
                f"(existing display_name={existing.display_name!r}); "
                f"dataset names must be globally unique."
            )
        sample_type = extract_sample_type(cls)
        if sample_type in SAMPLE_TO_DATASET:
            existing_cls = SAMPLE_TO_DATASET[sample_type]
            raise ValueError(
                f"sample type {sample_type.__qualname__!r} already bound to "
                f"{existing_cls.__qualname__!r}; each sample TypedDict must map "
                f"to exactly one Dataset class."
            )
        setattr(cls, _DATASET_META_ATTR, meta)
        DATASET_REGISTRY[name] = meta
        SAMPLE_TO_DATASET[sample_type] = cls
        return cls

    return decorator


def get_dataset_meta(cls: type) -> DatasetMeta:
    """Return the DatasetMeta attached to *cls* (AttributeError if not registered)."""
    return getattr(cls, _DATASET_META_ATTR)


def iter_dataset_metas() -> Iterator[DatasetMeta]:
    """Iterate over all registered DatasetMeta in insertion order."""
    return iter(DATASET_REGISTRY.values())


def lookup_dataset(name: str) -> "DatasetMeta | None":
    """Look up a DatasetMeta by registered name; None if not registered."""
    return DATASET_REGISTRY.get(name)


def dataset_meta_to_dict(meta: DatasetMeta) -> dict[str, Any]:
    """Convert a DatasetMeta to a JSON-serializable dict.

    Wire-format contract (v0.1, frozen):
      - `source` is ALWAYS a list (single-source datasets become 1-element).
      - `license` is always present (may be null).
    """
    return {
        "name": meta.name,
        "display_name": meta.display_name,
        "description": meta.description,
        "source": list(meta.source),
        "categories": [
            {"level1": c.level1.value, "level2": c.level2} for c in meta.categories
        ],
        "tags": list(meta.tags),
        "deps_group": meta.deps_group,
        "license": meta.license,
        "checksums": dict(meta.checksums),
    }


def dataset_meta_from_dict(payload: dict[str, Any]) -> DatasetMeta:
    """Reverse of ``dataset_meta_to_dict``: reconstruct from one index.json row.

    Pure deserializer — skips decorator-level validation since the index is
    release-authored and trusted.
    """
    categories = tuple(
        Category(
            level1=Level1Category(c["level1"]),
            level2=c.get("level2"),
        )
        for c in payload["categories"]
    )
    return DatasetMeta(
        name=payload["name"],
        display_name=payload["display_name"],
        description=payload["description"],
        source=tuple(payload["source"]),
        categories=categories,
        tags=tuple(payload.get("tags", ())),
        deps_group=payload.get("deps_group"),
        license=payload.get("license"),
        checksums=tuple(sorted(payload.get("checksums", {}).items())),
    )


_DATASETS_PACKAGE = "sieval.datasets"


def import_all_datasets() -> None:
    """Import every ``sieval.datasets`` submodule to trigger registration.

    Dynamic import at call-time so no static ``core → datasets`` edge shows
    up in the layer-boundary AST check.
    """
    pkg = importlib.import_module(_DATASETS_PACKAGE)
    for info in pkgutil.walk_packages(pkg.__path__, prefix=f"{_DATASETS_PACKAGE}."):
        importlib.import_module(info.name)
