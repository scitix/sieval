"""One resume option: ``True`` resumes in place, a directory starts from it.

``resume`` is the single way to resume a task, on the CLI (``--resume [DIR]``),
in YAML (``runner_config.resume``) and in Python (``TaskRunnerConfig.resume``):

* ``True``: resume the task's own ``result_dir``.
* A directory: start from that recorded task directory, copied into an empty
  ``result_dir`` (:mod:`sieval.core.runners.stage_resume`). A directory that is
  the task's own ``result_dir`` resumes in place, as ``True`` does.
* ``False`` or ``None``: no resume.

:func:`resume_source` is the one place that tells a copy from an in-place
resume. The deprecated aliases ``auto_resume`` and ``resume_from`` fold into
``resume`` through :func:`fold_resume_aliases`, kept apart so dropping them is
a small diff.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import os
from pathlib import Path

from loguru import logger

type ResumeValue = bool | Path | None


def as_resume(value: object, where: str = "resume") -> ResumeValue:
    """*value* as a resume: a bool, a directory, or ``None``.

    A string or path-like names a directory, ``~`` expanded. An empty one, or
    any other type, is refused.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (str, os.PathLike)):
        text = os.fspath(value)
        if not text:
            raise ValueError(f"{where} must be true, false or a directory, not ''")
        return Path(text).expanduser()
    raise TypeError(
        f"{where} must be true, false or a directory, not {type(value).__name__}"
    )


def same_dir(a: Path, b: Path) -> bool:
    """Whether *a* and *b* name one directory; neither needs to exist."""
    return a.expanduser().resolve() == b.expanduser().resolve()


def resume_source(resume: object, own_dir: Path) -> Path | None:
    """The directory a resume copies from, or ``None``.

    ``None`` when *resume* resumes in place (``True``, or *own_dir* itself) or
    does not resume at all.
    """
    if not isinstance(resume, Path):
        return None
    return None if same_dir(resume, own_dir) else resume


# ----------------------------------------------------------------------
# Deprecated aliases, for one release: `auto_resume` and `resume_from`.
# ----------------------------------------------------------------------


def fold_resume_aliases(
    resume: ResumeValue,
    *,
    auto_resume: object = None,
    resume_from: object = None,
    new: str,
    auto: str,
    source: str,
) -> ResumeValue:
    """Fold the deprecated aliases into *resume*, warning once for each.

    *new*, *auto* and *source* name ``resume``, ``auto_resume`` and
    ``resume_from`` where the caller reads them, so a warning or refusal says
    what the user wrote. ``auto_resume`` maps to ``resume: true`` (or
    ``false``) and ``resume_from`` to ``resume: <dir>``, which wins over
    ``auto_resume`` as it did.

    *resume* and the aliases must agree. ``true`` agrees with a directory,
    which resumes too, and the directory is kept: ``--resume --resume-from
    DIR`` meant ``DIR`` before. Two directories agree only when they are one;
    ``false`` agrees only with ``false``. Anything else is refused.
    """
    alias: ResumeValue = None
    name = ""
    if auto_resume is not None:
        if not isinstance(auto_resume, bool):
            raise TypeError(f"{auto} must be true or false")
        _warn_once(f"{auto} is deprecated; use {new}: {_show(auto_resume)}")
        alias, name = auto_resume, auto
    if resume_from is not None and resume_from != "":
        if isinstance(resume_from, bool):
            raise TypeError(f"{source} must be a directory")
        _warn_once(f"{source} is deprecated; use {new} <dir>")
        alias, name = as_resume(resume_from, source), source
    if alias is None:
        return resume
    if resume is None:
        return alias
    merged = _merge(resume, alias)
    if merged is None:
        raise ValueError(
            f"{new} ({_show(resume)}) and its deprecated alias {name} "
            f"({_show(alias)}) disagree; set only {new}"
        )
    return merged


def _merge(a: bool | Path, b: bool | Path) -> bool | Path | None:
    """What *a* and *b* mean together, or ``None`` when they disagree."""
    if isinstance(a, Path) and isinstance(b, Path):
        return a if same_dir(a, b) else None
    if isinstance(a, Path) or isinstance(b, Path):
        other = b if isinstance(a, Path) else a
        return (a if isinstance(a, Path) else b) if other is True else None
    return a if a == b else None


#: Deprecations already reported, so a config read twice in one process (a
#: prelaunch check, then the run) warns once.
_warned: set[str] = set()


def _warn_once(message: str) -> None:
    if message not in _warned:
        _warned.add(message)
        logger.warning("{}", message)


def _show(value: object) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)
