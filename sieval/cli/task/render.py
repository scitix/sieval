"""
CommandResult builders for sieval task list/show.

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

from pathlib import Path

from sieval.cli._readiness import evaluate_task_readiness, readiness_to_wire
from sieval.cli.output import CommandResult
from sieval.core.datasets.meta import DatasetMeta
from sieval.core.tasks.meta import TaskMeta, get_task_class

_QUICKSTART_EXAMPLE = "examples/quickstart.yaml"


def _task_to_row(t: TaskMeta, data_dir: Path, ds_meta: DatasetMeta | None) -> dict:
    # Orphan FK (index inconsistency): can't probe readiness → unknown.
    ready = (
        "unknown"
        if ds_meta is None
        else evaluate_task_readiness(t, data_dir, ds_meta).ready
    )
    return {
        "name": t.name,
        "dataset": t.dataset,
        "eval_mode": t.eval_mode.value,
        "n_shot": t.n_shot,
        "deps_group": t.deps_group or "-",
        "status": t.status,
        "ready": ready,
    }


def render_task_list(
    metas: list[TaskMeta],
    datasets: list[DatasetMeta],
    data_dir: Path,
) -> CommandResult:
    ds_by_name = {d.name: d for d in datasets}
    return CommandResult(
        command="task.list",
        ok=True,
        data=[_task_to_row(t, data_dir, ds_by_name.get(t.dataset)) for t in metas],
    )


def render_task_show(
    t: TaskMeta,
    ds: DatasetMeta | None,
    data_dir: Path,
) -> CommandResult:
    if ds is None:
        # Orphan FK (index inconsistency): readiness unresolvable.
        ready, missing = "unknown", []
    else:
        ready, missing = readiness_to_wire(evaluate_task_readiness(t, data_dir, ds))
    # Degrade to None when the task module's top-level imports fail (e.g.
    # unsatisfied `deps_group`): let the readiness report above deliver the
    # actionable diagnosis instead of crashing with a bare ModuleNotFoundError.
    try:
        suggested_class = get_task_class(t.name).__name__
    except ModuleNotFoundError:
        suggested_class = None

    data: dict = {
        "name": t.name,
        "display_name": t.display_name,
        "description": t.description,
        "dataset": t.dataset,
        "dataset_categories": (
            [
                f"{c.level1.value}/{c.level2}" if c.level2 else c.level1.value
                for c in ds.categories
            ]
            if ds
            else []
        ),
        "eval_mode": t.eval_mode.value,
        "n_shot": t.n_shot,
        "model_type": t.model_type,
        "deps_group": t.deps_group,
        "status": t.status,
        "reference_impl": (
            {
                "source": t.reference_impl.source,
                "url": t.reference_impl.url,
                "notes": t.reference_impl.notes,
            }
            if t.reference_impl
            else None
        ),
        "ready": ready,
        "missing": missing,
        "suggested_class": suggested_class,
    }
    if ready == "yes" and suggested_class is not None:
        data["run_hint"] = (
            f"sieval eval <config.yaml> "
            f"(template: {_QUICKSTART_EXAMPLE}, "
            f"set tasks.<task>.class: {suggested_class})"
        )
    return CommandResult(command="task.show", ok=True, data=data)
