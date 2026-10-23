"""The ``resume`` option: ``True`` resumes in place, a directory is copied from.

A directory that is the task's own ``result_dir`` resumes in place;
:func:`resume_source` is the one place that decides.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import os
from pathlib import Path

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
