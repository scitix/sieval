"""
sieval task {list, show} commands.

AI-Generated Code - Claude Sonnet 4.6 (Anthropic)
"""

from typing import Annotated

import typer

from sieval.cli.output import OutputFormat, render
from sieval.cli.task.render import render_task_list, render_task_show
from sieval.core.datasets.meta import Level1Category
from sieval.core.utils.logging import configure_logging
from sieval.core.utils.paths import resolve_data_dir
from sieval.meta import load_index

task_app = typer.Typer(help="Task discovery.")


@task_app.callback()
def _task_callback(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose (DEBUG) logging."),
    ] = False,
) -> None:
    configure_logging(verbose)


@task_app.command("list")
def list_cmd(
    dataset: Annotated[
        str | None, typer.Option("--dataset", help="Filter by dataset name.")
    ] = None,
    domain: Annotated[
        str | None, typer.Option("--domain", help="Filter by Level1Category.")
    ] = None,
    eval_mode: Annotated[
        str | None,
        typer.Option("--eval-mode", help="Filter by eval mode (gen/ppl/clp)."),
    ] = None,
    data_dir: Annotated[
        str | None, typer.Option("--data-dir", help="Override data directory.")
    ] = None,
    output: Annotated[OutputFormat, typer.Option("-o", "--output")] = OutputFormat.TEXT,
) -> None:
    """List registered tasks with dataset FK, eval mode, n-shot, deps, and status."""
    datasets, tasks = load_index()
    if dataset:
        tasks = [t for t in tasks if t.dataset == dataset]
    if domain:
        try:
            level1 = Level1Category(domain)
        except ValueError as e:
            valid = [c.value for c in Level1Category]
            raise typer.BadParameter(
                f"Unknown domain {domain!r}. Options: {valid}"
            ) from e
        datasets_by_name = {d.name: d for d in datasets}
        filtered = []
        for t in tasks:
            ds = datasets_by_name.get(t.dataset)
            if ds and any(c.level1 is level1 for c in ds.categories):
                filtered.append(t)
        tasks = filtered
    if eval_mode:
        tasks = [t for t in tasks if t.eval_mode.value == eval_mode]
    render(
        render_task_list(tasks, datasets, data_dir=resolve_data_dir(data_dir)),
        output,
    )


@task_app.command("show")
def show_cmd(
    name: Annotated[str, typer.Argument()],
    data_dir: Annotated[
        str | None, typer.Option("--data-dir", help="Override data directory.")
    ] = None,
    output: Annotated[OutputFormat, typer.Option("-o", "--output")] = OutputFormat.TEXT,
) -> None:
    """Show a task's full metadata plus its dataset's domain categories."""
    datasets, tasks = load_index()
    meta = next((t for t in tasks if t.name == name), None)
    if meta is None:
        typer.secho(
            f"Task {name!r} is not registered.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    ds = next((d for d in datasets if d.name == meta.dataset), None)
    render(render_task_show(meta, ds, data_dir=resolve_data_dir(data_dir)), output)
