"""
SIGTERM stops a command the way Ctrl+C does: shielded cleanup runs.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import signal
import subprocess
import sys
import textwrap
import time

import pytest

from sieval.cli.signals import TERMINATED_EXIT_CODE, terminate_as_interrupt

_CHILD = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    import anyio

    from sieval.cli.signals import terminate_as_interrupt

    marker = Path(sys.argv[1])


    async def main():
        try:
            marker.write_text("started")
            await anyio.sleep_forever()
        finally:
            with anyio.CancelScope(shield=True):
                await anyio.sleep(0.05)
                marker.write_text("stopped")


    try:
        with terminate_as_interrupt() as terminated:
            anyio.run(main)
    except KeyboardInterrupt:
        sys.exit(terminated.exit_code)
    """
)


@pytest.mark.parametrize("sigint_ignored", [False, True])
def test_sigterm_runs_shielded_cleanup_and_exits_143(tmp_path, sigint_ignored):
    """Also when SIGINT is inherited ignored, as by `sieval run &` in a script."""
    marker = tmp_path / "marker"
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(marker)],
        preexec_fn=(
            (lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
            if sigint_ignored
            else None
        ),
    )
    try:
        deadline = time.monotonic() + 30
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.read_text() == "started"
        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=30) == TERMINATED_EXIT_CODE
    finally:
        if child.poll() is None:
            child.kill()
    assert marker.read_text() == "stopped"


def test_the_previous_handlers_are_restored():
    before = signal.getsignal(signal.SIGTERM)
    before_int = signal.getsignal(signal.SIGINT)
    with terminate_as_interrupt() as terminated:
        assert signal.getsignal(signal.SIGTERM) is not before
    assert signal.getsignal(signal.SIGTERM) is before
    assert signal.getsignal(signal.SIGINT) is before_int
    assert terminated.exit_code == 130


def test_off_the_main_thread_it_installs_nothing():
    import threading

    seen = {}

    def _worker():
        before = signal.getsignal(signal.SIGTERM)
        with terminate_as_interrupt():
            seen["same"] = signal.getsignal(signal.SIGTERM) is before

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()
    assert seen == {"same": True}
