"""Tests for sieval.infer.backends.process — shared process utilities.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import os
import signal
from unittest.mock import patch

from sieval.infer.backends.process import kill_process_group, pid_alive


def test_pid_alive_current_process():
    assert pid_alive(os.getpid()) is True


def test_pid_alive_nonexistent():
    assert pid_alive(999999999) is False


def test_pid_alive_permission_error():
    with patch("sieval.infer.backends.process.os.kill", side_effect=PermissionError):
        assert pid_alive(12345) is True


def test_kill_process_group_sigterm():
    with (
        patch("sieval.infer.backends.process.os.getpgid", return_value=12345),
        patch("sieval.infer.backends.process.os.killpg") as mock_killpg,
    ):
        kill_process_group(12345, signal.SIGTERM)
        mock_killpg.assert_called_once_with(12345, signal.SIGTERM)


def test_kill_process_group_fallback_to_pid():
    with (
        patch("sieval.infer.backends.process.os.getpgid", side_effect=OSError),
        patch("sieval.infer.backends.process.os.kill") as mock_kill,
    ):
        kill_process_group(12345, signal.SIGTERM)
        mock_kill.assert_called_once_with(12345, signal.SIGTERM)
