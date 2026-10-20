"""Atomic file writes: every file SiEval rewrites in a result directory.

A reader sees the old content or the new, never a torn file: the data goes to a
temporary file beside the target (``.<name>.<random>.tmp``, unique per call, so
two writers never share one), is fsynced, then replaces the target.

The replacement keeps an existing target's mode; a new file gets what ``open()``
would give it, ``0o666`` less the umask. ``mkstemp`` alone creates ``0o600``,
which readers under another uid on a shared volume cannot open.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import contextlib
import os
import stat
import tempfile
from pathlib import Path

import anyio.to_thread


def _process_umask() -> int:
    """The process umask, read once: ``os.umask`` can only be read by setting it."""
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


_UMASK = _process_umask()


def _replacement_mode(path: Path) -> int:
    try:
        return stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return 0o666 & ~_UMASK


def atomic_write(path: Path, data: bytes) -> None:
    """Replace *path* with *data* atomically. The parent directory must exist."""
    mode = _replacement_mode(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


async def aatomic_write(path: Path, data: bytes) -> None:
    """:func:`atomic_write` off the event loop."""
    await anyio.to_thread.run_sync(atomic_write, path, data)
