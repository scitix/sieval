"""
Leaderboard CLI commands.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import shlex
import sys
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import anyio
import typer

from sieval.cli.output import CommandResult, OutputFormat, cli_error_message, render

from .catalog import scan_leaderboards
from .scanner import RunInfo, build_matrix, resolve_model_name, scan_runs

leaderboard_app = typer.Typer(
    name="leaderboard",
    help="Cross-run score aggregation and leaderboard display.",
    no_args_is_help=True,
)


def _resolve_run_models(runs: list[RunInfo]) -> list[RunInfo]:
    """Fill in missing model names from inference output (same as `report`)."""
    resolved: list[RunInfo] = []
    for run in runs:
        if run.model_name:
            resolved.append(run)
        else:
            resolved.append(replace(run, model_name=resolve_model_name(run.run_dir)))
    return resolved


@leaderboard_app.command()
def report(
    dirs: Annotated[
        list[Path] | None,
        typer.Argument(help="Directories to scan (default: ./outputs/)"),
    ] = None,
    output: Annotated[
        OutputFormat,
        typer.Option("-o", "--output", help="Output format"),
    ] = OutputFormat.TEXT,
    all_runs: Annotated[
        bool,
        typer.Option(
            "--all-runs", help="Keep all runs instead of latest per model/task"
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
) -> None:
    """Report a leaderboard table from evaluation outputs."""
    from sieval.core.utils.logging import configure_logging

    configure_logging(verbose)

    warnings: list[str] = []

    if dirs is None:
        dirs = [Path("outputs")]

    # Validate dirs exist; collect warnings for missing ones
    valid_dirs: list[Path] = []
    for d in dirs:
        if d.is_dir():
            valid_dirs.append(d)
        else:
            warnings.append(f"Directory not found, skipping: {d}")

    resolved_runs = _resolve_run_models(scan_runs(valid_dirs))
    matrix = build_matrix(resolved_runs, all_runs=all_runs)

    result = CommandResult(
        command="leaderboard.report",
        ok=True,
        data=dict(matrix),
        warnings=warnings or None,
    )
    render(result, output)


@leaderboard_app.command(name="ruler-avg")
def ruler_avg(
    dirs: Annotated[
        list[Path] | None,
        typer.Argument(help="Directories to scan (default: ./outputs/)"),
    ] = None,
    output: Annotated[
        OutputFormat,
        typer.Option("-o", "--output", help="Output format"),
    ] = OutputFormat.TEXT,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
) -> None:
    """RULER headline: mean of the 13 subtask scores, per context length.

    RULER's score is the unweighted average over its subtasks; the wide
    per-task matrix from `report` isn't that number. This collapses the
    ``ruler_*`` runs into the per-length mean (and an overall mean).
    """
    from sieval.core.utils.logging import configure_logging

    from ._ruler_avg import ruler_average

    configure_logging(verbose)

    warnings: list[str] = []
    if dirs is None:
        dirs = [Path("outputs")]
    valid_dirs: list[Path] = []
    for d in dirs:
        if d.is_dir():
            valid_dirs.append(d)
        else:
            warnings.append(f"Directory not found, skipping: {d}")

    resolved_runs = _resolve_run_models(scan_runs(valid_dirs))
    averages = ruler_average(
        [(r.model_name, r.task_name, r.report) for r in resolved_runs]
    )
    if not averages:
        warnings.append("No ruler_* runs with a numeric score found.")

    result = CommandResult(
        command="leaderboard.ruler_avg",
        ok=True,
        data={"models": averages},
        warnings=warnings or None,
    )
    render(result, output)


@leaderboard_app.command(name="list")
def list_cmd(
    directory: Annotated[
        Path,
        typer.Argument(help="Directory to scan (default: ./leaderboards/)"),
    ] = Path("leaderboards"),
    output: Annotated[
        OutputFormat,
        typer.Option("-o", "--output", help="Output format"),
    ] = OutputFormat.TEXT,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
) -> None:
    """List available leaderboards under the given directory."""
    from loguru import logger

    from sieval.core.utils.logging import configure_logging

    configure_logging(verbose)

    warnings: list[str] = []
    # Warn when the path isn't a directory (missing or a regular file) —
    # both cases fall through to an empty scan, and the user should see why.
    # An empty dir still falls through to "No leaderboards found" silently.
    if not directory.is_dir():
        msg = f"{directory}/ not found (cwd: {Path.cwd()})."
        logger.warning(msg)
        warnings.append(msg)

    summaries = scan_leaderboards(directory)

    rows = [
        {
            "name": s.name,
            "path": str(s.path),
            "models": s.models,
            "tasks": s.tasks,
            "alignment_card": s.alignment_card,
            "error": s.error,
        }
        for s in summaries
    ]

    result = CommandResult(
        command="leaderboard.list",
        ok=True,
        data={"leaderboards": rows},
        warnings=warnings or None,
    )
    render(result, output)


@leaderboard_app.command("run")
def run(
    ctx: typer.Context,
    config: Annotated[
        Path,
        typer.Argument(help="Path to YAML configuration file"),
    ],
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Override model name for all base models"),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume", "-r", help="Enable auto-resume for all tasks"),
    ] = False,
    result_dir: Annotated[
        str | None,
        typer.Option("--result-dir", help="Override result directory"),
    ] = None,
    deterministic: Annotated[
        bool | None,
        typer.Option(
            "--deterministic",
            help=(
                "Force deterministic inference mode on. "
                "Monotone: cannot disable a YAML-level `deterministic: true`."
            ),
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate config without running"),
    ] = False,
    output: Annotated[
        OutputFormat,
        typer.Option("-o", "--output", help="Output format"),
    ] = OutputFormat.TEXT,
) -> None:
    """Run evaluation tasks from a YAML config (model must be online)."""
    from sieval.core.utils.logging import configure_logging

    configure_logging(verbose)

    # The 'eval' shortcut and 'leaderboard run' share this exact callable;
    # the CommandResult.command reflects the entry point the user invoked so
    # JSON consumers can distinguish them. ctx.info_name is the invoked
    # command's own name ("eval" or "run") — more direct than walking up to
    # the parent group and comparing against the add_typer() string.
    command_name = "leaderboard.run" if ctx.info_name == "run" else "eval"
    dry_run_command = f"{command_name}.dry_run"

    if dry_run:
        from sieval.cli.validation import run_dry_run

        dry_result = run_dry_run(config)
        result = CommandResult(
            command=dry_run_command,
            ok=dry_result["n_errors"] == 0,
            data=dict(dry_result),
            error="Dry-run failed" if dry_result["n_errors"] > 0 else None,
        )
        render(result, output)
        if not result.ok:
            raise typer.Exit(1)
        return

    from sieval.cli.leaderboard.session import arun_session

    async def _run() -> dict:
        return await arun_session(
            config,
            model=model,
            resume=resume,
            result_dir=result_dir,
            deterministic=deterministic,
            invocation=shlex.join(sys.argv),
        )

    try:
        reports = anyio.run(_run)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        cmd_result = CommandResult(
            command=command_name, ok=False, error=cli_error_message(e)
        )
        render(cmd_result, output)
        raise typer.Exit(1) from e

    tasks_data = {
        task_name: {"report": report} for task_name, report in reports.items()
    }
    cmd_result = CommandResult(
        command=command_name,
        ok=True,
        data={"tasks": tasks_data},
    )
    render(cmd_result, output)
