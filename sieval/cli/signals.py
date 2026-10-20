"""SIGTERM stops a command the way Ctrl+C does.

Python's default SIGTERM ends the process at once: no ``finally`` runs, so the
engines and services a run started outlive it. Under
:func:`terminate_as_interrupt` SIGTERM raises SIGINT instead, so every shielded
stop and flush runs, as it does for Ctrl+C.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import contextlib
import signal
import threading
from collections.abc import Iterator

#: Exit status of a command stopped by SIGTERM: ``128 + SIGTERM``.
TERMINATED_EXIT_CODE = 128 + signal.SIGTERM


class Terminated:
    """Whether SIGTERM arrived while :func:`terminate_as_interrupt` was active."""

    def __init__(self) -> None:
        self.received = False

    @property
    def exit_code(self) -> int:
        """143 after SIGTERM, else 130."""
        return TERMINATED_EXIT_CODE if self.received else 130


@contextlib.contextmanager
def terminate_as_interrupt() -> Iterator[Terminated]:
    """Turn SIGTERM into SIGINT for the duration of the block.

    A no-op off the main thread, where no signal handler can be installed.
    """
    state = Terminated()
    if threading.current_thread() is not threading.main_thread():
        yield state
        return
    previous_int = signal.getsignal(signal.SIGINT)

    def _handler(_signum: int, _frame: object) -> None:
        state.received = True
        # A command started with `&` from a script inherits SIGINT ignored, and
        # raising an ignored signal does nothing.
        if signal.getsignal(signal.SIGINT) is signal.SIG_IGN:
            signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.raise_signal(signal.SIGINT)

    previous = signal.signal(signal.SIGTERM, _handler)
    try:
        yield state
    finally:
        signal.signal(signal.SIGTERM, previous)
        if previous_int is not None:
            signal.signal(signal.SIGINT, previous_int)
