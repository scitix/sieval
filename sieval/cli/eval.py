"""
``sieval eval`` — evaluate a YAML config against online model endpoints.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import shlex
import sys
from pathlib import Path
from typing import Annotated

import anyio
import typer

from sieval.cli._cli_model import ApiBaseOption, CliModel
from sieval.cli._resume_option import (
    ResumeFromOption,
    ResumeOption,
    cli_resume,
)
from sieval.cli.output import CommandResult, OutputFormat, cli_command, render
from sieval.cli.signals import terminate_as_interrupt


@cli_command
def eval_cmd(
    config: Annotated[
        Path,
        typer.Argument(help="Path to YAML configuration file"),
    ],
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Override model name for all base models"),
    ] = None,
    api_base: ApiBaseOption = None,
    resume: ResumeOption = None,
    result_dir: Annotated[
        str | None,
        typer.Option("--result-dir", help="Override result directory"),
    ] = None,
    resume_from: ResumeFromOption = None,
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
    resumed = cli_resume(resume, resume_from)
    # No --checkpoint: this command never launches an engine.
    cli_model = CliModel.from_options(checkpoint=None, api_base=api_base, model=model)

    if dry_run:
        from sieval.cli.validation import run_dry_run

        dry_result = run_dry_run(
            config,
            model_override=model,
            resume=resumed,
            result_dir_override=result_dir,
            deterministic_override=deterministic,
            invocation=shlex.join(sys.argv),
            cli_model=cli_model,
        )
        result = CommandResult(
            command="eval.dry_run",
            ok=dry_result["n_errors"] == 0,
            data=dict(dry_result),
            error="Dry-run failed" if dry_result["n_errors"] > 0 else None,
        )
        render(result, output)
        if not result.ok:
            raise typer.Exit(1)
        return

    from sieval.cli.session import arun_session

    async def _run() -> dict:
        return await arun_session(
            config,
            model=model,
            resume=resumed,
            result_dir=result_dir,
            deterministic=deterministic,
            invocation=shlex.join(sys.argv),
            cli_model=cli_model,
        )

    try:
        with terminate_as_interrupt() as terminated:
            reports = anyio.run(_run)
    except KeyboardInterrupt:
        # Ctrl-C/SIGTERM is not a command failure: exit 130/143 without a
        # result payload. Everything else is left to `@cli_command`.
        sys.exit(terminated.exit_code)

    tasks_data = {
        task_name: {"report": report} for task_name, report in reports.items()
    }
    cmd_result = CommandResult(
        command="eval",
        ok=True,
        data={"tasks": tasks_data},
    )
    render(cmd_result, output)
