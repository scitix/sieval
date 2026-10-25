"""Tests for sieval.cli eval command (pure evaluation, model must be online).

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import json
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from sieval.cli import app

runner = CliRunner()


class TestEvalCommand:
    def test_eval_help(self):
        result = runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "yaml" in result.output.lower()

    def test_eval_accepts_options(self):
        result = runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--resume" in result.output
        assert "--dry-run" in result.output
        assert "--result-dir" in result.output
        assert "--verbose" in result.output

    def test_eval_emits_eval_command_field(self, tmp_path):
        config = tmp_path / "test.yaml"
        config.write_text("models: {}\ntasks: {}")

        mock_arun = AsyncMock(return_value={})

        with (
            patch("sieval.cli.session.arun_session", mock_arun),
            patch("sieval.core.utils.logging.configure_logging"),
        ):
            result = runner.invoke(app, ["eval", str(config), "-o", "json"])

        assert result.exit_code == 0
        assert mock_arun.call_count == 1
        assert json.loads(result.stdout)["command"] == "eval"

    def test_eval_dry_run_emits_eval_dry_run_command_field(self, tmp_path):
        config = tmp_path / "test.yaml"
        config.write_text("models: {}\ntasks: {}")

        dry = {"n_errors": 0, "n_warnings": 0, "checks": []}
        with (
            patch("sieval.cli.validation.run_dry_run", return_value=dry) as mock_dry,
            patch("sieval.core.utils.logging.configure_logging"),
        ):
            result = runner.invoke(
                app, ["eval", str(config), "--dry-run", "-o", "json"]
            )

        assert result.exit_code == 0
        assert mock_dry.call_count == 1
        assert json.loads(result.stdout)["command"] == "eval.dry_run"

    def test_leaderboard_group_is_gone(self):
        """#194: `sieval leaderboard run` moved to `sieval eval`; `report` and
        `list` were removed with cross-run aggregation."""
        result = runner.invoke(app, ["leaderboard", "run", "--help"])
        assert result.exit_code != 0
        assert "No such command 'leaderboard'" in result.output

    def test_result_dir_exists_hint_uses_cli_flags(self, tmp_path):
        from sieval.core.runners import ResultDirExistsError

        config = tmp_path / "test.yaml"
        config.write_text("models: {}\ntasks: {}")
        existing = tmp_path / "prior_run"

        mock_arun = AsyncMock(side_effect=ResultDirExistsError(existing))

        with (
            patch("sieval.cli.session.arun_session", mock_arun),
            patch("sieval.core.utils.logging.configure_logging"),
        ):
            result = runner.invoke(app, ["eval", str(config), "-o", "json"])

        assert result.exit_code != 0
        parsed = json.loads(result.stdout)
        assert parsed["ok"] is False
        assert parsed["command"] == "eval"
        error = parsed["error"]
        assert "--resume" in error
        assert "--result-dir" in error
        assert str(existing) in error
        assert "resume=True" not in error


class TestEvalRefusesAMissingPythonPackage:
    def test_exits_5_with_an_environment_refusal_line(self, tmp_path):
        """Refused before a dataset loads or a model client is built."""
        from sieval.cli.session import EvalSession
        from sieval.tasks.aime_2024_0shot_gen import AIME2024ZeroShotGenTask

        config = tmp_path / "c.yaml"
        config.write_text(
            "models:\n"
            "  m:\n"
            "    name: served\n"
            "    api_base: http://127.0.0.1:1/v1\n"
            "datasets:\n"
            "  d:\n"
            "    class: sieval.datasets.aime_2024.AIME2024Dataset\n"
            "tasks:\n"
            "  t:\n"
            "    class: sieval.tasks.aime_2024_0shot_gen.AIME2024ZeroShotGenTask\n"
            "    dataset: d\n"
            "    model: m\n"
            f"result_dir: {tmp_path / 'out'}\n",
            encoding="utf-8",
        )
        with (
            patch.object(
                AIME2024ZeroShotGenTask,
                "python_requires",
                {"sieval_no_such_mod": "nope>=1"},
            ),
            patch.object(
                EvalSession,
                "_setup_datasets",
                side_effect=AssertionError("datasets must not load"),
            ),
            patch.object(
                EvalSession,
                "_setup_models",
                side_effect=AssertionError("models must not be built"),
            ),
        ):
            result = runner.invoke(app, ["eval", str(config), "-o", "json"])

        assert result.exit_code == 5, result.output
        (line,) = [
            line
            for line in result.stderr.splitlines()
            if line.startswith("SIEVAL_REFUSAL ")
        ]
        payload = json.loads(line.removeprefix("SIEVAL_REFUSAL "))
        assert payload["kind"] == "environment"
        assert payload["reason"] == "python_requires"
        assert payload["task"] == "t"
        assert "nope>=1" in payload["message"]
