"""Tests for sieval.core.utils.logging — unified logging configuration.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import sys
import types
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from sieval.core.utils import logging as logging_mod
from sieval.core.utils.logging import log_user


@pytest.fixture(autouse=True)
def _isolate_logging():
    """Reset module state between tests.

    This ensures that tests which forget to mock ``logger`` don't leak
    handler mutations into other tests.
    """
    logging_mod._configured = False
    logging_mod._sink_id = None
    yield
    logging_mod._configured = False
    logging_mod._sink_id = None


class TestConfigureLogging:
    def test_second_call_replaces_sink(self):
        with patch.object(logging_mod, "logger") as mock_logger:
            mock_logger.add.return_value = 1
            logging_mod.configure_logging(verbose=False)
            assert mock_logger.add.call_count == 1
            first_level = mock_logger.add.call_args.kwargs["level"]
            assert first_level == "INFO"

            mock_logger.add.return_value = 2
            logging_mod.configure_logging(verbose=True)
            # Second call removes the first sink and adds a new one
            assert mock_logger.add.call_count == 2
            second_level = mock_logger.add.call_args.kwargs["level"]
            assert second_level == "DEBUG"

    def test_non_tty_uses_stderr_no_color(self):
        fake_stderr = MagicMock()
        fake_stderr.isatty.return_value = False

        with (
            patch.object(logging_mod, "sys", wraps=sys) as mock_sys,
            patch.object(logging_mod, "logger") as mock_logger,
        ):
            mock_sys.stderr = fake_stderr
            logging_mod.configure_logging(verbose=False)

        mock_logger.add.assert_called_once()
        call_kwargs = mock_logger.add.call_args.kwargs
        assert call_kwargs["colorize"] is False
        assert call_kwargs["level"] == "INFO"

    def test_tty_uses_callable_sink(self):
        fake_stderr = MagicMock()
        fake_stderr.isatty.return_value = True

        with (
            patch.object(logging_mod, "sys", wraps=sys) as mock_sys,
            patch.object(logging_mod, "logger") as mock_logger,
        ):
            mock_sys.stderr = fake_stderr
            logging_mod.configure_logging(verbose=True)

        mock_logger.add.assert_called_once()
        sink = mock_logger.add.call_args.args[0]
        assert callable(sink)
        call_kwargs = mock_logger.add.call_args.kwargs
        assert call_kwargs["colorize"] is True
        assert call_kwargs["level"] == "DEBUG"

    def test_non_tty_disables_hf_progress_bars(self):
        fake_stderr = MagicMock()
        fake_stderr.isatty.return_value = False
        fake_datasets = types.SimpleNamespace(disable_progress_bars=MagicMock())

        with (
            patch.object(logging_mod, "sys", wraps=sys) as mock_sys,
            patch.object(logging_mod, "logger"),
            patch.dict(sys.modules, {"datasets": fake_datasets}),
        ):
            mock_sys.stderr = fake_stderr
            logging_mod.configure_logging(verbose=False)

        fake_datasets.disable_progress_bars.assert_called_once()

    def test_non_tty_ignores_datasets_import_error(self):
        fake_stderr = MagicMock()
        fake_stderr.isatty.return_value = False
        real_import = __import__

        def _fake_import(name, *args, **kwargs):
            if name == "datasets":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        with (
            patch.object(logging_mod, "sys", wraps=sys) as mock_sys,
            patch.object(logging_mod, "logger"),
            patch("builtins.__import__", side_effect=_fake_import),
        ):
            mock_sys.stderr = fake_stderr
            logging_mod.configure_logging(verbose=False)

    def test_registers_user_level(self):
        with (
            patch.object(logging_mod, "sys") as mock_sys,
            patch.object(logging_mod, "logger") as mock_logger,
        ):
            mock_sys.stderr.isatty.return_value = False
            mock_logger.add.return_value = 1
            logging_mod.configure_logging()

        mock_logger.level.assert_called_once_with("USER", no=25, color="<bold>")

    def test_user_level_registered_only_once(self):
        with (
            patch.object(logging_mod, "sys") as mock_sys,
            patch.object(logging_mod, "logger") as mock_logger,
        ):
            mock_sys.stderr.isatty.return_value = False
            mock_logger.add.return_value = 1
            logging_mod.configure_logging(verbose=False)
            mock_logger.add.return_value = 2
            logging_mod.configure_logging(verbose=True)

        # logger.level("USER", ...) should only be called once (first call)
        mock_logger.level.assert_called_once_with("USER", no=25, color="<bold>")


class TestFormatFunction:
    def test_user_level_plain_message(self):
        record = {"level": types.SimpleNamespace(name="USER")}
        result = logging_mod._format(record)  # type: ignore[arg-type]  # simplified record for test
        assert result == "{message}\n"
        assert "<level>" not in result

    def test_standard_level_uses_loguru_default_format(self):
        record = {"level": types.SimpleNamespace(name="INFO")}
        result = logging_mod._format(record)  # type: ignore[arg-type]  # simplified record for test
        assert "<green>{time:" in result
        assert "<level>{level:" in result
        assert "<level>{message}</level>" in result
        assert "<cyan>{name}</cyan>" in result


class TestTtySink:
    """Test the TTY sink returned by _make_tty_sink.

    loguru calls the sink with ``Message`` (a ``str`` subclass that only
    exists in the ``.pyi`` stub).  Tests pass plain ``str`` which is the
    runtime base class — hence the type-ignore comments below.
    """

    def test_cr_progress_cleared_before_normal_log(self):
        """A \\r progress line should be followed by \\n before normal output."""
        buf = StringIO()
        sink = logging_mod._make_tty_sink(buf)

        sink("\rProgress 50%")  # type: ignore[arg-type]
        sink("Normal log line\n")  # type: ignore[arg-type]

        output = buf.getvalue()
        # The \r progress should appear, then a newline, then the normal line
        assert "\rProgress 50%" in output
        assert "\n" in output.split("\rProgress 50%")[1]
        assert "Normal log line" in output

    def test_normal_lines_no_extra_newline(self):
        """Consecutive normal lines should not get extra newlines."""
        buf = StringIO()
        sink = logging_mod._make_tty_sink(buf)

        sink("Line 1\n")  # type: ignore[arg-type]
        sink("Line 2\n")  # type: ignore[arg-type]

        output = buf.getvalue()
        assert output == "Line 1\nLine 2\n"

    def test_consecutive_cr_lines_no_extra_newline(self):
        """Consecutive \\r progress lines should not insert extra newlines."""
        buf = StringIO()
        sink = logging_mod._make_tty_sink(buf)

        sink("\rProgress 50%")  # type: ignore[arg-type]
        sink("\rProgress 75%")  # type: ignore[arg-type]

        output = buf.getvalue()
        assert output == "\rProgress 50%\rProgress 75%"


class TestLogUser:
    def test_log_user_calls_logger_log(self):
        with patch.object(logging_mod, "logger") as mock_logger:
            log_user("hello {}", "world")
        mock_logger.log.assert_called_once_with("USER", "hello {}", "world")
