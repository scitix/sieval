"""
Shared process management utilities for local inference backends (Unix only).

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import contextlib
import os
import signal


def pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is alive (not a zombie).

    Uses os.kill(pid, 0) which doesn't send a signal but checks existence.
    On Linux, also checks /proc/<pid>/status to exclude zombie processes,
    since os.kill(pid, 0) succeeds for zombies.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we lack permission — still alive
        return True

    # Try to reap zombie: if waitpid succeeds the process was our zombie child
    # — it's dead.
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
        if reaped == pid:
            return False
    except ChildProcessError:
        # Not our child — fall through to /proc check (Linux only)
        pass

    # /proc fallback for non-child zombies (e.g. re-parented to init).
    # NOTE: /proc is Linux-only; on macOS non-child zombies will fall
    # through to ``return True`` (reported as alive).  Acceptable since
    # macOS deployments are dev-only and zombies are rare in practice.
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("State:"):
                    return "Z" not in line
    except (FileNotFoundError, PermissionError):
        pass

    return True


def kill_process_group(pid: int, sig: signal.Signals) -> None:
    """Send *sig* to the process group led by *pid*.

    Falls back to signalling just *pid* if the pgid lookup fails.
    """
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass
    except OSError:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, sig)
