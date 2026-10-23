"""
``--resume [DIR]`` on ``sieval run`` and ``sieval eval``: every parse form.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from sieval.cli import app
from sieval.core.utils.logging import configure_logging

runner = CliRunner()


@pytest.fixture(autouse=True)
def _logging_configured():
    """Register the ``USER`` level even where a test patches the setup out."""
    configure_logging()


COMMANDS = [["run"], ["eval"]]


@pytest.fixture
def config(tmp_path: Path) -> str:
    path = tmp_path / "cfg.yaml"
    path.write_text("models: {}\ntasks: {}\n")
    return str(path)


def _invoke(command: list[str], args: list[str]) -> tuple[int, dict, str]:
    """Run *command*, returning its exit code, the call's kwargs and output."""
    run_all = AsyncMock(return_value={})
    arun = AsyncMock(return_value={})
    with (
        patch("sieval.cli.run._run_all", run_all),
        patch("sieval.cli.session.arun_session", arun),
    ):
        result = runner.invoke(app, [*command, *args])
    called = run_all if command == ["run"] else arun
    kwargs = dict(called.call_args.kwargs) if called.call_args else {}
    if called.call_args:
        kwargs["config"] = str(called.call_args.args[0])
    return result.exit_code, kwargs, result.output


@pytest.mark.parametrize("command", COMMANDS, ids=" ".join)
class TestParse:
    def test_absent_is_no_resume(self, command, config):
        code, kw, _ = _invoke(command, [config])
        assert code == 0
        assert kw["resume"] is None

    @pytest.mark.parametrize("flag", ["--resume", "-r"])
    def test_no_value_resumes_in_place(self, command, config, flag):
        code, kw, _ = _invoke(command, [config, flag])
        assert code == 0
        assert kw["resume"] is True

    @pytest.mark.parametrize("flag", ["--resume", "-r"])
    def test_a_value_is_a_directory(self, command, config, flag):
        code, kw, _ = _invoke(command, [config, flag, "/runs/old"])
        assert code == 0
        assert kw["resume"] == Path("/runs/old")
        assert kw["config"] == config

    def test_an_attached_value(self, command, config):
        code, kw, _ = _invoke(command, [config, "--resume=/runs/old"])
        assert code == 0
        assert kw["resume"] == Path("/runs/old")

    def test_an_option_after_it_is_not_its_value(self, command, config):
        code, kw, _ = _invoke(command, [config, "--resume", "--model", "m"])
        assert code == 0
        assert kw["resume"] is True
        assert kw["model"] == "m"

    @pytest.mark.parametrize("flag", ["--resume", "-r"])
    def test_the_positional_after_it_is_not_swallowed(self, command, config, flag):
        code, kw, _ = _invoke(command, [flag, config])
        assert code == 0
        assert kw["resume"] is True
        assert kw["config"] == config

    def test_a_value_then_the_positional(self, command, config):
        code, kw, _ = _invoke(command, ["--resume", "/runs/old", config])
        assert code == 0
        assert kw["resume"] == Path("/runs/old")
        assert kw["config"] == config

    def test_options_then_the_positional(self, command, config):
        code, kw, _ = _invoke(command, ["--resume", "-m", "m", config])
        assert code == 0
        assert kw["resume"] is True
        assert kw["model"] == "m"

    def test_grouped_with_a_short_flag(self, command, config):
        """``-rv`` was ``-r -v`` while ``-r`` was a flag, and stays so."""
        with patch("sieval.core.utils.logging.configure_logging") as logging:
            code, kw, _ = _invoke(command, [config, "-rv"])
        assert code == 0
        assert kw["resume"] is True
        if command == ["run"]:
            assert kw["verbose"] is True
        else:
            logging.assert_called_once_with(True)

    @pytest.mark.parametrize(
        "args",
        [["-vr", "{config}"], ["{config}", "-vr"], ["-vr", "-m", "m", "{config}"]],
    )
    def test_r_last_in_a_group(self, command, config, args):
        with patch("sieval.core.utils.logging.configure_logging"):
            code, kw, _ = _invoke(command, [a.format(config=config) for a in args])
        assert code == 0
        assert kw["resume"] is True
        assert kw["config"] == config

    def test_r_last_in_a_group_takes_a_value(self, command, config):
        with patch("sieval.core.utils.logging.configure_logging"):
            code, kw, _ = _invoke(command, [config, "-vr", "/runs/old"])
        assert code == 0
        assert kw["resume"] == Path("/runs/old")

    def test_an_attached_short_value(self, command, config):
        code, kw, _ = _invoke(command, [config, "-r/runs/old"])
        assert code == 0
        assert kw["resume"] == Path("/runs/old")

    @pytest.mark.parametrize("glued", ["-ro", "-rm"])
    def test_r_glued_to_a_value_option_is_refused(self, command, config, glued):
        """``-ro json`` would be ``--resume o``; refused rather than guessed."""
        code, kw, output = _invoke(command, [config, glued, "x"])
        assert code == 2
        assert kw == {}
        assert "ambiguous" in output

    def test_an_attached_value_is_never_the_positional(self, command, config):
        """``--resume=X`` names ``X`` as ``DIR`` outright; CONFIG is missing."""
        code, _, output = _invoke(command, [f"--resume={config}"])
        assert code == 2
        assert "Missing argument" in output

    def test_a_usage_error_still_reports(self, command, config):
        code, _, output = _invoke(command, [config, "--model"])
        assert code == 2
        assert "--model" in output

    def test_after_the_double_dash_nothing_is_split(self, command, config):
        code, _, _ = _invoke(command, [config, "--", "-rv"])
        assert code != 0

    def test_help_shows_the_optional_value(self, command):
        result = runner.invoke(app, [*command, "--help"], env={"COLUMNS": "200"})
        assert "[DIR]" in result.output
        assert "--resume-from" not in result.output


@pytest.mark.parametrize("command", COMMANDS, ids=" ".join)
def test_resume_from_was_removed(command, config):
    code, kw, output = _invoke(command, [config, "--resume-from", "/runs/old"])
    assert code == 2
    assert kw == {}
    assert "--resume-from" in output
