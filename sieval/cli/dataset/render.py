"""
CommandResult builders for sieval dataset list/show.

AI-Generated Code - Claude Opus 4.7 (Anthropic)
"""

from pathlib import Path

from sieval.cli._readiness import evaluate_dataset_readiness, readiness_to_wire
from sieval.cli.output import CommandResult
from sieval.core.datasets.meta import DatasetMeta
from sieval.core.tasks.meta import TaskMeta
from sieval.datasets.downloaders.hf import parse_hf_source


def _suggested_yaml_path(meta: DatasetMeta) -> str | None:
    """Return the literal string users should paste into YAML's `path:`.

    - Single HF source: the repo id (e.g. "HuggingFaceH4/aime_2024").
    - One or more URL sources: "${SIEVAL_DATA_DIR}/<meta.name>" — literal,
      unexpanded, so it survives per-environment portability.
    - Mixed sources (URL + HF): undefined, returns None.
    """
    url_sources = [s for s in meta.source if s.startswith("url:")]
    hf_sources = [s for s in meta.source if s.startswith("hf:")]
    if url_sources and not hf_sources:
        return "${SIEVAL_DATA_DIR}/" + meta.name
    if len(hf_sources) == 1 and not url_sources:
        return parse_hf_source(hf_sources[0]).repo_id
    return None


def _dataset_to_row(m: DatasetMeta, data_dir: Path) -> dict:
    domains = sorted({c.level1.value for c in m.categories})
    report = evaluate_dataset_readiness(m, data_dir)
    return {
        "name": m.name,
        "domain": "/".join(domains) or "-",
        "deps_group": m.deps_group or "-",
        "license": m.license or "unknown",
        "ready": report.ready,
    }


def render_dataset_list(
    metas: list[DatasetMeta],
    data_dir: Path,
) -> CommandResult:
    rows = [_dataset_to_row(m, data_dir) for m in metas]
    return CommandResult(command="dataset.list", ok=True, data=rows)


def render_dataset_show(
    m: DatasetMeta,
    tasks: list[TaskMeta],
    data_dir: Path,
) -> CommandResult:
    cats = ", ".join(
        f"{c.level1.value}/{c.level2}" if c.level2 else c.level1.value
        for c in m.categories
    )
    ready, missing = readiness_to_wire(evaluate_dataset_readiness(m, data_dir))
    detail = {
        "name": m.name,
        "display_name": m.display_name,
        "description": m.description,
        "source": list(m.source),
        "categories": cats,
        "tags": list(m.tags),
        "deps_group": m.deps_group,
        "license": m.license or "unknown",
        "ready": ready,
        "missing": missing,
        "suggested_path": _suggested_yaml_path(m),
        "tasks": [
            {
                "name": t.name,
                "eval_mode": t.eval_mode.value,
                "n_shot": t.n_shot,
                "status": t.status,
            }
            for t in tasks
        ],
    }
    return CommandResult(command="dataset.show", ok=True, data=detail)
