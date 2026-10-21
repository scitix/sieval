"""``--resume [DIR]``, shared by ``sieval run``, ``sieval eval`` and
``sieval leaderboard run``.

``--resume`` alone resumes each task in its own result_dir; ``--resume DIR``
starts each task from ``DIR/<task>`` of a recorded run (see
:meth:`sieval.cli.leaderboard.session.EvalSession._cli_resume_for`).
``--resume-from DIR`` is its deprecated alias.

The value is optional, click's ``is_flag=False, flag_value=...``, so click
takes the next token as ``DIR`` unless it starts with ``-``. Typer does not
expose that, so :class:`ResumeCommand` sets it on the click option. It also
keeps the forms ``--resume`` had as a flag: a token taken as ``DIR`` that
leaves ``CONFIG`` empty goes back to ``CONFIG`` (``sieval run --resume
cfg.yaml``), and ``-rv`` is still ``-r -v``.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from pathlib import Path
from typing import Annotated

import click
import typer
from typer.core import TyperCommand

from sieval.core.runners import ResumeValue, fold_resume_aliases

#: What click passes for ``--resume`` given without a value. Not empty, which
#: Typer turns into ``None``, and no path: a path holds no NUL.
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
    """``--resume [DIR]`` and its deprecated ``--resume-from DIR``, as one value.

    ``True`` for ``--resume`` alone, a path for a ``DIR``, ``None`` for
    neither. ``--resume-from`` warns, and is refused beside a different
    ``--resume DIR``.
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
    """A command with ``--resume [DIR]``; see the module docstring."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for param in self.params:
            if param.name == "resume" and isinstance(param, click.Option):
                # Typer 0.24 passes neither `flag_value` nor, by giving every
                # option a default, the inferred `_flag_needs_value` that
                # lets `--resume` stand alone.
                param.flag_value = _IN_PLACE
                param._flag_needs_value = True

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        args = self._split_short_flags(args)
        return super().parse_args(ctx, self._unswallow(ctx, args))

    def _split_short_flags(self, args: list[str]) -> list[str]:
        """A short group holding ``r``, such as ``-rv`` or ``-vr``, one by one.

        Only when every other letter is a flag, so ``-rDIR`` stays a value.
        Click would read the letters after ``r`` as ``DIR``; while ``-r`` was
        a flag ``-rv`` was ``-r -v``, so ``r`` inside a group resumes in place
        and ``r`` last is ``-r``, which may take the next token.
        """
        flags = {
            opt[1]
            for p in self.params
            if isinstance(p, click.Option) and p.is_flag
            for opt in p.opts
            if len(opt) == 2 and opt[0] == "-" and opt[1] != "-"
        }
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
            else:
                out.append(token)
        return out

    def _unswallow(self, ctx: click.Context, args: list[str]) -> list[str]:
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
