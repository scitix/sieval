"""
Unified logging configuration for SiEval.

Provides a single entry point for loguru configuration and a custom USER
level for user-facing output that replaces print().

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import sys
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, TextIO

from loguru import logger

if TYPE_CHECKING:
    # Message and Record are stub-only types (defined in loguru's .pyi, not
    # exposed at runtime), so they must stay behind TYPE_CHECKING.
    from loguru import Message, Record

_configured = False
_sink_id: int | None = None

# Custom level between INFO(20) and WARNING(30) — plain-text user output.
_USER_LEVEL_NO = 25

# loguru default format — keeps full color markup for all standard levels.
_LOGURU_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# USER level: plain-text, no timestamp/level prefix.
_USER_FORMAT = "{message}"


def _format(record: "Record") -> str:
    """Format by level: USER → plain message, others → loguru default."""
    if record["level"].name == "USER":
        return _USER_FORMAT + "\n"
    return _LOGURU_FORMAT + "\n"


def _make_tty_sink(stderr: TextIO) -> Callable[["Message"], None]:
    """Build a TTY sink that clears \\r progress lines before normal output.

    Progress lines emitted via ``logger.opt(raw=True)`` start with ``\\r``
    and are written verbatim.  When the next *normal* log line arrives the
    sink inserts a ``\\n`` first so the progress line is visually cleared.
    """
    from tqdm import tqdm

    has_cr = False

    def _sink(msg: "Message") -> None:
        nonlocal has_cr
        text = str(msg)
        if text.startswith("\r"):
            has_cr = True
            tqdm.write(text, file=stderr, end="")
        else:
            if has_cr:
                has_cr = False
                tqdm.write("", file=stderr)  # emit \n to clear progress line
            tqdm.write(text, file=stderr, end="")

    return _sink


def configure_logging(verbose: bool = False) -> None:
    """Configure loguru for the entire process.

    If called more than once, the later call's *verbose* flag takes effect
    (the previous sink is replaced).

    * TTY → sink via ``tqdm.write`` (avoids tqdm progress-bar corruption),
      colorized.  A ``\\r`` progress line is automatically cleared before
      the next normal log line.
    * Non-TTY → plain ``sys.stderr``, no ANSI, HuggingFace progress bars
      disabled.
    """
    global _configured, _sink_id

    if _sink_id is not None:
        logger.remove(_sink_id)

    if not _configured:
        logger.remove()  # remove loguru default handler on first call
        with suppress(ValueError):  # already registered if _configured was reset
            logger.level("USER", no=_USER_LEVEL_NO, color="<bold>")
        _configured = True

    is_tty = sys.stderr.isatty()
    log_level = "DEBUG" if verbose else "INFO"

    if is_tty:
        _sink_id = logger.add(
            _make_tty_sink(sys.stderr),
            format=_format,
            colorize=True,
            level=log_level,
        )
    else:
        _sink_id = logger.add(
            sys.stderr,
            format=_format,
            colorize=False,
            level=log_level,
        )
        try:
            import datasets

            datasets.disable_progress_bars()
        except ImportError:
            pass


def log_user(msg: str, *args, **kwargs) -> None:
    """Emit user-facing plain-text output via loguru USER level."""
    logger.log("USER", msg, *args, **kwargs)
