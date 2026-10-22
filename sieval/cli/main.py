"""
CLI application root — entry point and command registration.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from typing import Annotated

import typer

from sieval.cli._resume_option import ResumeCommand
from sieval.cli.dataset import dataset_app
from sieval.cli.eval import eval_cmd
from sieval.cli.infer import infer_app
from sieval.cli.run import register_run_command
from sieval.cli.task import task_app

app = typer.Typer(
    name="sieval",
    help="SiEval — Model Delivery Quality Verification System",
    no_args_is_help=True,
)
app.add_typer(infer_app, name="infer")
app.add_typer(dataset_app, name="dataset")
app.add_typer(task_app, name="task")
register_run_command(app)

app.command(name="eval", cls=ResumeCommand)(eval_cmd)


def _version_callback(value: bool) -> None:
    if value:
        from sieval import __version__

        print(f"sieval {__version__}")  # noqa: T201
        raise typer.Exit()


@app.callback()
def _app_callback(
    version: Annotated[  # noqa: ARG001
        bool,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """SiEval — Model Delivery Quality Verification System."""


def main() -> None:
    app()
