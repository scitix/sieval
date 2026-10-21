"""The ``resume`` option: ``True`` resumes in place, a directory is copied from.

A directory that is the task's own ``result_dir`` resumes in place;
:func:`resume_source` is the one place that decides. The deprecated
``auto_resume`` / ``resume_from`` fold in via :func:`fold_resume_aliases`.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import os
from pathlib import Path

from loguru import logger

type ResumeValue = bool | Path | None


def as_resume(value: object, where: str = "resume") -> ResumeValue:
    """*value* as a bool, a ``~``-expanded directory, or ``None``."""
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
    """The directory to copy from; ``None`` for in place or no resume."""
    if not isinstance(resume, Path):
        return None
    return None if same_dir(resume, own_dir) else resume


# Deprecated aliases, removed next release.


def fold_resume_aliases(
    resume: ResumeValue,
    *,
    auto_resume: object = None,
    resume_from: object = None,
    new: str,
    auto: str,
    source: str,
) -> ResumeValue:
    """Fold the deprecated aliases into *resume*, warning once each.

    *new* / *auto* / *source* are the names the user wrote, for messages.
    ``resume_from`` wins over ``auto_resume``. The result must agree with
    *resume*: ``true`` with a directory (kept), a directory only with itself,
    ``false`` only with ``false``; otherwise refused.
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


#: Warned already: a config is read twice per process (prelaunch, then run).
_warned: set[str] = set()


def _warn_once(message: str) -> None:
    if message not in _warned:
        _warned.add(message)
        logger.warning("{}", message)


def _show(value: object) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)
