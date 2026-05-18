"""Tests for sieval.cli.output — formatted output helpers.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import json
from unittest.mock import patch

import yaml

from sieval.cli.output import (
    CommandResult,
    OutputFormat,
    _collapse_constant_columns,
    cli_error_message,
    render,
)
from sieval.core.runners import ResultDirExistsError


def _collect_log_user(mock_log_user):
    """Collect all positional args from log_user calls into a single string."""
    parts = []
    for call in mock_log_user.call_args_list:
        parts.extend(str(a) for a in call.args)
    return " ".join(parts)


class TestOutputFormat:
    def test_enum_values(self):
        assert OutputFormat.TEXT == "text"
        assert OutputFormat.JSON == "json"
        assert OutputFormat.YAML == "yaml"


class TestCommandResult:
    def test_creation_success(self):
        result = CommandResult(command="infer.list", ok=True, data=[{"model": "qwen"}])
        assert result.command == "infer.list"
        assert result.ok is True
        assert result.data == [{"model": "qwen"}]
        assert result.error is None
        assert result.warnings is None

    def test_creation_failure(self):
        result = CommandResult(
            command="infer.show", ok=False, error="No handle found for 'qwen'"
        )
        assert result.ok is False
        assert result.error == "No handle found for 'qwen'"

    def test_frozen(self):
        result = CommandResult(command="eval", ok=True)
        try:
            result.ok = False  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except AttributeError:
            pass


class TestCliErrorMessage:
    """CLI-layer translation of core exceptions to flag-aware user messages."""

    def test_result_dir_already_exists_surfaces_cli_flags(self, tmp_path):
        exc = ResultDirExistsError(tmp_path / "existing")
        msg = cli_error_message(exc)

        assert str(tmp_path / "existing") in msg
        assert "--resume" in msg
        assert "--result-dir" in msg
        assert "auto_resume=True" not in msg

    def test_generic_exception_falls_through_to_str(self):
        assert cli_error_message(RuntimeError("kaboom")) == "kaboom"


class TestRenderJson:
    def test_render_json_success(self, capsys):
        result = CommandResult(
            command="infer.list",
            ok=True,
            data=[{"model": "qwen3-4b", "phase": "running"}],
        )
        render(result, OutputFormat.JSON)
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["ok"] is True
        assert parsed["data"] == [{"model": "qwen3-4b", "phase": "running"}]
        assert "error" not in parsed

    def test_render_json_failure(self, capsys):
        result = CommandResult(
            command="infer.show",
            ok=False,
            error="No handle found for 'qwen'",
        )
        render(result, OutputFormat.JSON)
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["ok"] is False
        assert parsed["error"] == "No handle found for 'qwen'"
        assert "data" not in parsed

    def test_render_json_with_warnings(self, capsys):
        result = CommandResult(
            command="eval.dry_run",
            ok=True,
            data={"checks": []},
            warnings=["duplicate key: models.qwen"],
        )
        render(result, OutputFormat.JSON)
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["warnings"] == ["duplicate key: models.qwen"]

    def test_render_json_none_data_omitted(self, capsys):
        result = CommandResult(command="infer.stop", ok=True)
        render(result, OutputFormat.JSON)
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert "data" not in parsed


class TestRenderYaml:
    def test_render_yaml_success(self, capsys):
        result = CommandResult(
            command="infer.list",
            ok=True,
            data=[{"model": "qwen3-4b"}],
        )
        render(result, OutputFormat.YAML)
        out = capsys.readouterr().out
        parsed = yaml.safe_load(out)
        assert parsed["ok"] is True
        assert parsed["data"] == [{"model": "qwen3-4b"}]


class TestRenderTextInferList:
    def test_empty_list(self):
        result = CommandResult(command="infer.list", ok=True, data=[])
        with patch("sieval.cli.output.log_user") as mock:
            render(result, OutputFormat.TEXT)
        logged = _collect_log_user(mock)
        assert "No inference services found" in logged

    def test_with_services(self):
        result = CommandResult(
            command="infer.list",
            ok=True,
            data=[
                {
                    "model": "qwen3-4b",
                    "status": "Ready",
                    "endpoint": "http://localhost:8000/v1",
                },
                {"model": "llama3-8b", "status": "Pending", "endpoint": None},
            ],
        )
        with patch("sieval.cli.output.log_user") as mock:
            render(result, OutputFormat.TEXT)
        logged = _collect_log_user(mock)
        assert "qwen3-4b" in logged
        assert "Ready" in logged
        assert "llama3-8b" in logged
        assert "Pending" in logged


class TestRenderTextInferShow:
    def test_show_with_env(self):
        result = CommandResult(
            command="infer.show",
            ok=True,
            data={
                "model": "qwen3-4b",
                "status": "Ready",
                "backend": "vllm",
                "endpoint": "http://localhost:8000/v1",
                "handle_id": "12345",
                "metadata": {"role": "full"},
                "conditions": {"ready": {"status": True, "reason": ""}},
                "env": {
                    "framework": "vllm==0.8.3",
                    "image": "",
                    "cuda_version": "12.4",
                    "driver_version": "550.54.15",
                    "gpu_model": "A100",
                    "gpu_count": 8,
                    "gpu_topo": "NVLink",
                    "python_version": "3.12.3",
                    "extra": {},
                },
            },
        )
        with patch("sieval.cli.output.log_user") as mock:
            render(result, OutputFormat.TEXT)
        logged = _collect_log_user(mock)
        assert "qwen3-4b" in logged
        assert "Ready" in logged
        assert "vllm" in logged
        assert "12345" in logged
        assert "12.4" in logged
        assert "NVLink" in logged

    def test_show_without_env(self):
        result = CommandResult(
            command="infer.show",
            ok=True,
            data={
                "model": "qwen3-4b",
                "status": "Ready",
                "backend": "vllm",
                "endpoint": "http://localhost:8000/v1",
                "handle_id": "12345",
                "metadata": {},
                "conditions": {"ready": {"status": True, "reason": ""}},
                "env": None,
            },
        )
        with patch("sieval.cli.output.log_user") as mock:
            render(result, OutputFormat.TEXT)
        logged = _collect_log_user(mock)
        assert "qwen3-4b" in logged
        assert "Framework" not in logged


class TestRenderTextInferStop:
    def test_stop_success(self):
        result = CommandResult(
            command="infer.stop",
            ok=True,
            data={"model": "qwen3-4b", "stopped": True, "phase": "stopped"},
        )
        with patch("sieval.cli.output.log_user") as mock:
            render(result, OutputFormat.TEXT)
        logged = _collect_log_user(mock)
        assert "Stopped inference service" in logged
        assert "qwen3-4b" in logged

    def test_stop_failure(self):
        result = CommandResult(
            command="infer.stop",
            ok=True,
            data={
                "model": "qwen3-4b",
                "stopped": False,
                "phase": "running",
                "handle_id": "999",
            },
        )
        with (
            patch("sieval.cli.output.log_user"),
            patch("sieval.cli.output.logger") as mock_logger,
        ):
            render(result, OutputFormat.TEXT)
        assert mock_logger.error.called


class TestRenderTextInferStart:
    def test_start(self):
        result = CommandResult(
            command="infer.start",
            ok=True,
            data={
                "model": "qwen3-4b",
                "backend": "vllm",
                "endpoint": "http://localhost:8000/v1",
                "handle_id": "12345",
                "handle_path": "/root/.sieval/handles/qwen3-4b.json",
                "metadata": {"role": "full"},
            },
        )
        with patch("sieval.cli.output.log_user") as mock:
            render(result, OutputFormat.TEXT)
        logged = _collect_log_user(mock)
        assert "qwen3-4b" in logged
        assert "vllm" in logged
        assert "12345" in logged
        assert "/root/.sieval/handles/qwen3-4b.json" in logged


class TestRenderTextEval:
    def test_eval_results(self):
        result = CommandResult(
            command="eval",
            ok=True,
            data={
                "tasks": {
                    "mmlu": {"report": {"accuracy": 0.85}},
                    "gsm8k": {"report": {"accuracy": 0.72}},
                },
            },
        )
        with patch("sieval.cli.output.log_user") as mock:
            render(result, OutputFormat.TEXT)
        logged = _collect_log_user(mock)
        assert "RESULTS" in logged
        assert "mmlu" in logged
        assert "gsm8k" in logged


class TestRenderTextDryRun:
    def test_dry_run_pass(self):
        result = CommandResult(
            command="eval.dry_run",
            ok=True,
            data={
                "checks": [
                    {"name": "file_exists", "ok": True},
                    {"name": "yaml_syntax", "ok": True, "warnings": []},
                    {
                        "name": "schema",
                        "ok": True,
                        "detail": "3 models, 2 datasets, 4 tasks",
                    },
                    {"name": "imports", "ok": True},
                ],
                "n_errors": 0,
                "n_warnings": 0,
            },
        )
        with patch("sieval.cli.output.log_user") as mock:
            render(result, OutputFormat.TEXT)
        logged = _collect_log_user(mock)
        assert "Dry-run passed" in logged
        assert "3 models" in logged

    def test_dry_run_fail(self):
        result = CommandResult(
            command="eval.dry_run",
            ok=False,
            data={
                "checks": [
                    {"name": "file_exists", "ok": True},
                    {"name": "yaml_syntax", "ok": True, "warnings": ["dup key"]},
                    {
                        "name": "schema",
                        "ok": False,
                        "detail": "missing 'models' section",
                    },
                ],
                "n_errors": 1,
                "n_warnings": 1,
            },
            error="Dry-run failed",
        )
        with patch("sieval.cli.output.log_user") as mock:
            render(result, OutputFormat.TEXT)
        logged = _collect_log_user(mock)
        assert "Dry-run failed" in logged
        assert "error(s)" in logged


class TestRenderTextFallback:
    """Test the fallback path when no text renderer is registered."""

    def test_fallback_ok_with_data(self):
        result = CommandResult(command="unknown.cmd", ok=True, data={"key": "value"})
        with patch("sieval.cli.output.log_user") as mock:
            render(result, OutputFormat.TEXT)
        logged = _collect_log_user(mock)
        assert "key" in logged

    def test_fallback_error(self):
        result = CommandResult(command="unknown.cmd", ok=False, error="Something broke")
        with patch("sieval.cli.output.logger") as mock_logger:
            render(result, OutputFormat.TEXT)
        mock_logger.error.assert_called_once()


class TestRenderTextErrorPaths:
    """Test error-path branches in text renderers."""

    def test_infer_list_error(self):
        result = CommandResult(
            command="infer.list", ok=False, error="Handle dir missing"
        )
        with patch("sieval.cli.output.logger") as mock_logger:
            render(result, OutputFormat.TEXT)
        mock_logger.error.assert_called_once()

    def test_infer_show_error(self):
        result = CommandResult(command="infer.show", ok=False, error="No handle found")
        with patch("sieval.cli.output.logger") as mock_logger:
            render(result, OutputFormat.TEXT)
        mock_logger.error.assert_called_once()

    def test_infer_start_error(self):
        result = CommandResult(command="infer.start", ok=False, error="Deploy failed")
        with patch("sieval.cli.output.logger") as mock_logger:
            render(result, OutputFormat.TEXT)
        mock_logger.error.assert_called_once()

    def test_infer_dry_run_error(self):
        result = CommandResult(command="infer.dry_run", ok=False, error="Plan invalid")
        with patch("sieval.cli.output.logger") as mock_logger:
            render(result, OutputFormat.TEXT)
        mock_logger.error.assert_called_once()

    def test_eval_error(self):
        result = CommandResult(command="eval", ok=False, error="arun_session failed")
        with patch("sieval.cli.output.logger") as mock_logger:
            render(result, OutputFormat.TEXT)
        mock_logger.error.assert_called_once()

    def test_run_error(self):
        result = CommandResult(command="run", ok=False, error="Runtime error")
        with patch("sieval.cli.output.logger") as mock_logger:
            render(result, OutputFormat.TEXT)
        mock_logger.error.assert_called_once()


class TestRenderTextInferDryRun:
    def test_dry_run_text(self, capsys):
        result = CommandResult(
            command="infer.dry_run",
            ok=True,
            data={
                "model": "qwen3-4b",
                "command": ["vllm", "serve", "/models/qwen3-4b"],
                "health_check": "http://localhost:8000/health",
            },
        )
        render(result, OutputFormat.TEXT)
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["model"] == "qwen3-4b"


class TestRenderTextRun:
    def test_run_results(self):
        result = CommandResult(
            command="run",
            ok=True,
            data={
                "tasks": {
                    "mmlu": {"report": {"accuracy": 0.85}},
                },
            },
        )
        with patch("sieval.cli.output.log_user") as mock:
            render(result, OutputFormat.TEXT)
        logged = _collect_log_user(mock)
        assert "RESULTS" in logged
        assert "mmlu" in logged


class TestRenderEnvBlock:
    """Test _render_env_block with image field."""

    def test_env_with_image(self):
        result = CommandResult(
            command="infer.show",
            ok=True,
            data={
                "model": "qwen3-4b",
                "status": "Ready",
                "backend": "vllm",
                "endpoint": "http://localhost:8000/v1",
                "handle_id": "12345",
                "metadata": {},
                "conditions": {},
                "env": {
                    "framework": "vllm==0.8.3",
                    "image": "registry.example.com/vllm:latest",
                    "cuda_version": "12.4",
                    "driver_version": "550.54",
                    "gpu_model": "A100",
                    "gpu_count": 8,
                    "gpu_topo": "",
                    "python_version": "3.12.3",
                    "extra": {"nccl": "2.18"},
                },
            },
        )
        with patch("sieval.cli.output.log_user") as mock:
            render(result, OutputFormat.TEXT)
        logged = _collect_log_user(mock)
        assert "registry.example.com/vllm:latest" in logged
        assert "nccl" in logged


class TestRenderTextInferStartMetadata:
    """Test infer start text renderer without metadata."""

    def test_start_no_metadata(self):
        result = CommandResult(
            command="infer.start",
            ok=True,
            data={
                "model": "qwen3-4b",
                "backend": "vllm",
                "endpoint": "http://localhost:8000/v1",
                "handle_id": "12345",
                "handle_path": "/root/.sieval/handles/qwen3-4b.json",
                "metadata": {},
            },
        )
        with patch("sieval.cli.output.log_user") as mock:
            render(result, OutputFormat.TEXT)
        logged = _collect_log_user(mock)
        assert "qwen3-4b" in logged
        assert "Handle:" in logged


class TestRenderYamlNoDoubleNewline:
    def test_yaml_output_ends_with_single_newline(self, capsys):
        result = CommandResult(command="test", ok=True, data={"key": "val"})
        render(result, OutputFormat.YAML)
        out = capsys.readouterr().out
        assert out.endswith("\n")
        assert not out.endswith("\n\n")


class TestRenderTextTypeGuards:
    """Test isinstance guard branches that log type errors."""

    def _assert_type_error_logged(self, command: str, bad_data: object) -> None:
        result = CommandResult(command=command, ok=True, data=bad_data)  # type: ignore[arg-type]
        with patch("sieval.cli.output.logger") as mock_logger:
            render(result, OutputFormat.TEXT)
        mock_logger.error.assert_called_once()

    def test_infer_list_bad_type(self):
        self._assert_type_error_logged("infer.list", {"not": "a list"})

    def test_infer_show_bad_type(self):
        self._assert_type_error_logged("infer.show", [1, 2, 3])

    def test_infer_stop_bad_type(self):
        self._assert_type_error_logged("infer.stop", [1, 2, 3])

    def test_infer_start_bad_type(self):
        self._assert_type_error_logged("infer.start", [1, 2, 3])

    def test_infer_dry_run_bad_type(self):
        self._assert_type_error_logged("infer.dry_run", [1, 2, 3])

    def test_task_reports_bad_type(self):
        self._assert_type_error_logged("eval", [1, 2, 3])

    def test_dry_run_ok_bad_type(self):
        self._assert_type_error_logged("eval.dry_run", [1, 2, 3])


class TestRenderTextDryRunFailedCheckInSuccessPath:
    """The success path now renders ✗ for failed checks (defensive)."""

    def test_mixed_checks_render_both_symbols(self):
        result = CommandResult(
            command="eval.dry_run",
            ok=True,
            data={
                "checks": [
                    {"name": "file_exists", "ok": True},
                    {"name": "schema", "ok": False, "detail": "bad schema"},
                ],
                "n_errors": 0,
                "n_warnings": 0,
            },
        )
        with patch("sieval.cli.output.log_user") as mock:
            render(result, OutputFormat.TEXT)
        logged = _collect_log_user(mock)
        assert "✓" in logged
        assert "✗" in logged
        assert "bad schema" in logged


class TestCollapseConstantColumns:
    """Tests for _collapse_constant_columns helper."""

    def test_collapse_no_collapse_when_values_vary(self):
        rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]
        cols = [("A", "a"), ("B", "b")]
        visible, collapsed = _collapse_constant_columns(
            rows, cols, never_collapse=frozenset()
        )
        assert visible == [("A", "a")]
        assert collapsed == [("B", "b", "x")]

    def test_collapse_respects_never_collapse_set(self):
        """Columns whose key is in `never_collapse` stay visible even when constant."""
        rows = [{"name": "t1", "ready": "yes"}, {"name": "t2", "ready": "yes"}]
        cols = [("NAME", "name"), ("READY", "ready")]
        visible, collapsed = _collapse_constant_columns(
            rows, cols, never_collapse=frozenset({"name", "ready"})
        )
        # Both columns should stay — nothing collapses.
        assert visible == cols
        assert collapsed == []

    def test_collapse_single_row_no_collapse(self):
        """Single-row list is a degenerate case — collapsing all columns leaves
        nothing to render. Skip collapse entirely."""
        rows = [{"a": 1, "b": "x"}]
        cols = [("A", "a"), ("B", "b")]
        visible, collapsed = _collapse_constant_columns(
            rows, cols, never_collapse=frozenset()
        )
        assert visible == cols
        assert collapsed == []

    def test_collapse_empty_rows(self):
        """Zero-row list: helper should not error. Return cols unchanged — caller
        (the text renderer) handles the 'No X' message separately."""
        visible, collapsed = _collapse_constant_columns(
            [], [("A", "a")], never_collapse=frozenset()
        )
        assert visible == [("A", "a")]
        assert collapsed == []


def _format_log_user(mock_log_user) -> str:
    """Expand each log_user(template, *args) call into the final string."""
    lines = []
    for call in mock_log_user.call_args_list:
        args = call.args
        if not args:
            continue
        template = str(args[0])
        lines.append(template.format(*args[1:]))
    return "\n".join(lines)


class TestDatasetTaskListTextCollapse:
    """End-to-end collapse rendering on list commands — synthetic rows keep
    assertions independent of the real pilot's value distribution."""

    def test_dataset_list_constant_column_emits_footer(self):
        """Non-pinned constant column → one `HEADER: all VALUE` footer line."""
        rows = [
            {
                "name": "ds1",
                "domain": "A",
                "deps_group": "-",
                "license": "MIT",
                "ready": "yes",
            },
            {
                "name": "ds2",
                "domain": "B",
                "deps_group": "-",
                "license": "MIT",
                "ready": "no",
            },
        ]
        result = CommandResult(command="dataset.list", ok=True, data=rows)
        with patch("sieval.cli.output.log_user") as mock:
            render(result, OutputFormat.TEXT)
        logged = _format_log_user(mock)
        assert "DEPS_GROUP: all -" in logged
        assert "LICENSE: all MIT" in logged
        assert "NAME" in logged
        assert "READY" in logged

    def test_task_list_constant_column_emits_footer(self):
        rows = [
            {
                "name": "t1",
                "dataset": "ds1",
                "eval_mode": "gen",
                "n_shot": 0,
                "deps_group": "-",
                "status": "stable",
                "ready": "yes",
            },
            {
                "name": "t2",
                "dataset": "ds2",
                "eval_mode": "gen",
                "n_shot": 5,
                "deps_group": "-",
                "status": "stable",
                "ready": "no",
            },
        ]
        result = CommandResult(command="task.list", ok=True, data=rows)
        with patch("sieval.cli.output.log_user") as mock:
            render(result, OutputFormat.TEXT)
        logged = _format_log_user(mock)
        assert "EVAL_MODE: all gen" in logged
        assert "STATUS: all stable" in logged
        assert "NAME" in logged
        assert "READY" in logged

    def test_list_all_varying_columns_no_footer(self):
        """When every non-pinned column varies, no collapse footer appears."""
        rows = [
            {
                "name": "ds1",
                "domain": "A",
                "deps_group": "math",
                "license": "MIT",
                "ready": "yes",
            },
            {
                "name": "ds2",
                "domain": "B",
                "deps_group": "drop",
                "license": "Apache-2.0",
                "ready": "no",
            },
        ]
        result = CommandResult(command="dataset.list", ok=True, data=rows)
        with patch("sieval.cli.output.log_user") as mock:
            render(result, OutputFormat.TEXT)
        logged = _format_log_user(mock)
        assert ": all " not in logged
