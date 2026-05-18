"""
Readiness probes for discovery CLI (`dataset show`, `task show`).

Computes the three-axis `ready` state (data, dataset-deps, task-deps) and a
structured `missing` list. Also owns the wire-format serialization helpers
so `dataset/render.py` and `task/render.py` don't duplicate them.

Pure logic — no I/O beyond what the handlers already expose.

AI-Generated Code - Claude Opus 4.7 (1M context) (Anthropic)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sieval.core.datasets.meta import DatasetMeta
from sieval.core.tasks.meta import TaskMeta
from sieval.datasets.downloaders import resolve as resolve_handler
from sieval.datasets.downloaders.resolver import extras_unsatisfied

MissingKind = Literal["data", "dataset-deps", "task-deps"]

# Ordered tuple of every MissingKind literal. Exported so rendering code can
# iterate / compute column widths without reflecting on __annotations__.
# Keep in sync with the Literal above (a static-check test pins the match).
MISSING_KINDS: tuple[MissingKind, ...] = ("data", "dataset-deps", "task-deps")


@dataclass(frozen=True, slots=True)
class MissingEntry:
    """One concrete remediation step the user can take.

    `kind="data"` populates `sources` (raw `source` strings from DatasetMeta);
    `kind="dataset-deps"` / `"task-deps"` populate `group` (extras group name)
    and `unmet` (verbatim output of `extras_unsatisfied`).
    """

    kind: MissingKind
    sources: tuple[str, ...] = ()
    group: str | None = None
    unmet: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    """Output of `evaluate_*_readiness`.

    `ready="yes"` → missing always empty.
    `ready="no"` → missing non-empty, entries in contract order (data →
        dataset-deps → task-deps).
    `ready="unknown"` → at least one axis was unresolvable (e.g. unknown
        scheme); missing may still contain entries from axes that *were*
        determinable.
    """

    ready: Literal["yes", "no", "unknown"]
    missing: tuple[MissingEntry, ...] = ()


def missing_entry_to_dict(e: MissingEntry) -> dict:
    """Flatten a MissingEntry to its wire-format shape (kind-specific keys)."""
    if e.kind == "data":
        return {"kind": "data", "sources": list(e.sources)}
    return {"kind": e.kind, "group": e.group, "unmet": list(e.unmet)}


def readiness_to_wire(r: ReadinessReport) -> tuple[str, list[dict]]:
    """Return (ready, missing_json_list) for embedding in CommandResult.data."""
    return r.ready, [missing_entry_to_dict(e) for e in r.missing]


def _probe_dataset(
    ds_meta: DatasetMeta, data_dir: Path
) -> tuple[list[MissingEntry], bool]:
    """Probe a dataset's data + dataset-deps axes independently.

    Returns `(entries, had_unknown)`. Entries are in contract order: `data`
    first, `dataset-deps` second. `had_unknown` is True iff any source
    scheme was unresolvable (handler not registered for its prefix).
    """
    entries: list[MissingEntry] = []
    had_unknown = False

    data_missing: list[str] = []
    for src in ds_meta.source:
        try:
            handler = resolve_handler(src)
        except NotImplementedError:
            had_unknown = True
            continue
        if not handler.is_downloaded(src, data_dir, ds_meta.name):
            data_missing.append(src)
    if data_missing:
        entries.append(MissingEntry(kind="data", sources=tuple(data_missing)))

    # Dataset-deps axis — runs regardless of the data-axis outcome.
    if ds_meta.deps_group:
        unmet = extras_unsatisfied(ds_meta.deps_group)
        if unmet:
            entries.append(
                MissingEntry(
                    kind="dataset-deps",
                    group=ds_meta.deps_group,
                    unmet=tuple(unmet),
                )
            )

    return entries, had_unknown


def evaluate_dataset_readiness(ds_meta: DatasetMeta, data_dir: Path) -> ReadinessReport:
    """Classify a dataset's readiness across data + dataset-deps axes."""
    entries, had_unknown = _probe_dataset(ds_meta, data_dir)
    if had_unknown:
        return ReadinessReport(ready="unknown", missing=tuple(entries))
    if entries:
        return ReadinessReport(ready="no", missing=tuple(entries))
    return ReadinessReport(ready="yes", missing=())


def evaluate_task_readiness(
    task_meta: TaskMeta,
    data_dir: Path,
    ds_meta: DatasetMeta,
) -> ReadinessReport:
    """Classify a task's readiness across dataset + task-deps axes.

    Dataset entries from `_probe_dataset` are concatenated verbatim into
    the task's `missing` list, preserving contract order (data →
    dataset-deps → task-deps).
    """
    entries, had_unknown = _probe_dataset(ds_meta, data_dir)

    if task_meta.deps_group:
        unmet = extras_unsatisfied(task_meta.deps_group)
        if unmet:
            entries.append(
                MissingEntry(
                    kind="task-deps",
                    group=task_meta.deps_group,
                    unmet=tuple(unmet),
                )
            )

    if had_unknown:
        return ReadinessReport(ready="unknown", missing=tuple(entries))
    if entries:
        return ReadinessReport(ready="no", missing=tuple(entries))
    return ReadinessReport(ready="yes", missing=())
