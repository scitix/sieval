"""``--resume [DIR]`` for ``sieval run`` and ``sieval eval``.

Typer cannot declare an option whose value is optional, so
:class:`ResumeCommand` sets it up on the click option. It also keeps the forms
``--resume`` had as a flag: ``--resume cfg.yaml`` still names the config, and
``-rv`` is still ``-r -v``.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from pathlib import Path
from typing import Annotated

import click
import typer
from typer.core import TyperCommand

from sieval.core.runners import ResumeValue, fold_resume_aliases

#: Click's value for a bare ``--resume``: not empty (Typer makes that ``None``)
#: and never a path (no path holds NUL).
_IN_PLACE = "\x00"

RESUME_HELP = (
    "Resume each task in its own result_dir; with DIR, start each task from "
    "DIR/<task> of another run (read-only ok), or from DIR itself for a "
    "one-task config"
)

#: Only with :class:`ResumeCommand`, which supplies what Typer drops.
ResumeOption = Annotated[
    str | None,
    typer.Option(
        "--resume",
        "-r",
        metavar="[DIR]",
        help=RESUME_HELP,
    ),
]

ResumeFromOption = Annotated[
    Path | None,
    typer.Option(
        "--resume-from",
        metavar="DIR",
        help="Deprecated: use --resume DIR",
        hidden=True,
    ),
]


def cli_resume(resume: str | None, resume_from: Path | None) -> ResumeValue:
    """``True`` for a bare ``--resume``, a path for ``DIR``, else ``None``.

    Folds in the deprecated ``--resume-from``.
    """
    value: ResumeValue = (
        None if resume is None else True if resume == _IN_PLACE else Path(resume)
    )
    return fold_resume_aliases(
        value,
        resume_from=resume_from,
        new="--resume",
        auto="--resume",
        source="--resume-from",
    )


class ResumeCommand(TyperCommand):
    """A command with ``--resume [DIR]``."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for param in self.params:
            if param.name == "resume" and isinstance(param, click.Option):
                # Typer 0.24 drops both; private click attribute, so
                # TestParse.test_no_value_resumes_in_place guards it.
                param.flag_value = _IN_PLACE
                param._flag_needs_value = True

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        args = self._split_short_flags(args)
        return super().parse_args(ctx, self._unswallow(ctx, args))

    def _split_short_flags(self, args: list[str]) -> list[str]:
        """Split a group of short flags holding ``r`` (``-rv``, ``-vr``).

        As when ``-r`` was a flag: ``r`` inside the group resumes in place,
        ``r`` last is ``-r`` and may take the next token. ``-rDIR`` is left
        alone; ``-r`` glued to a value-taking option is refused.
        """
        flags: set[str] = set()
        short: set[str] = set()
        for p in self.params:
            if isinstance(p, click.Option) and p.name != "resume":
                for opt in p.opts:
                    if len(opt) == 2 and opt[0] == "-" and opt[1] != "-":
                        (flags if p.is_flag else short).add(opt[1])
        out: list[str] = []
        for i, token in enumerate(args):
            if token == "--":
                return out + args[i:]
            letters = token[1:]
            if (
                len(token) > 2
                and token[0] == "-"
                and token[1] != "-"
                and "r" in letters
                and all(c in flags or c == "r" for c in letters)
            ):
                last = len(letters) - 1
                out += [
                    ("-r" if j == last else f"--resume={_IN_PLACE}")
                    if c == "r"
                    else f"-{c}"
                    for j, c in enumerate(letters)
                ]
            elif token[:2] == "-r" and len(token) == 3 and token[2] in short:
                # `-ro json` reads as `--resume o`, not `-r -o json`.
                raise click.UsageError(
                    f"{token} is ambiguous: use --resume=DIR or -r -{token[2]}"
                )
            else:
                out.append(token)
        return out

    def _unswallow(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Give ``--resume X`` back to ``CONFIG`` when ``CONFIG`` is missing."""
        try:
            opts, _, _ = self.make_parser(ctx).parse_args(args=list(args))
        except click.UsageError:
            return args  # the real parse reports it
        value = opts.get("resume")
        missing = any(
            isinstance(p, click.Argument)
            and p.required
            and p.name is not None
            and not isinstance(opts.get(p.name), str)
            for p in self.params
        )
        if not missing or not isinstance(value, str) or value == _IN_PLACE:
            return args
        out = list(args)
        for i, token in enumerate(out[:-1]):
            if token in ("--resume", "-r") and out[i + 1] == value:
                out[i] = f"--resume={_IN_PLACE}"
                return out
        return args
