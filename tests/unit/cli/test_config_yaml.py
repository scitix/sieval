"""Tests for sieval.cli._config_yaml — the evaluation config reader.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import ast
import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

import sieval.cli
from sieval.cli import app
from sieval.cli._config_yaml import DuplicateKeyError, load_eval_config
from sieval.cli.session import EvalSession

cli_runner = CliRunner()

# A second ``api_base`` under one model: last-wins would run on :8001.
_DUPLICATE_API_BASE = (
    "models:\n"
    "  m:\n"
    "    model_type: chat\n"
    "    api_base: http://localhost:8000/v1\n"
    "    api_base: http://localhost:8001/v1\n"
    "datasets: {}\n"
    "tasks: {}\n"
)
_DUPLICATE_MESSAGE = "duplicate key 'api_base' at line 5 (first at line 4)"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class TestLoadEvalConfig:
    def test_parses_as_safe_load(self, tmp_path):
        text = "models:\n  a:\n    name: foo\n  b: [1, 2]\nn: 3\n"
        assert load_eval_config(_write(tmp_path, text)) == yaml.safe_load(text)

    def test_empty_file_is_none(self, tmp_path):
        assert load_eval_config(_write(tmp_path, "")) is None

    def test_duplicate_top_level_key(self, tmp_path):
        path = _write(tmp_path, "models: {}\ntasks: {}\nmodels: {}\n")
        with pytest.raises(DuplicateKeyError) as info:
            load_eval_config(path)
        assert str(info.value) == (
            f"{path}: duplicate key 'models' at line 3 (first at line 1)"
        )

    def test_duplicate_nested_key_names_both_lines(self, tmp_path):
        with pytest.raises(DuplicateKeyError, match=r"line 5 \(first at line 4\)"):
            load_eval_config(_write(tmp_path, _DUPLICATE_API_BASE))

    def test_names_every_duplicate(self, tmp_path):
        path = _write(tmp_path, "a: 1\na: 2\nb:\n  c: 1\n  c: 2\n")
        with pytest.raises(DuplicateKeyError) as info:
            load_eval_config(path)
        message = str(info.value)
        assert "duplicate key 'a' at line 2" in message
        assert "duplicate key 'c' at line 5" in message

    def test_duplicate_key_is_a_value_error(self):
        # The CLI renders a ValueError's message as is.
        assert issubclass(DuplicateKeyError, ValueError)

    def test_same_key_in_sibling_mappings_is_not_a_duplicate(self, tmp_path):
        text = "models:\n  a:\n    name: x\n  b:\n    name: y\n"
        assert load_eval_config(_write(tmp_path, text)) == {
            "models": {"a": {"name": "x"}, "b": {"name": "y"}}
        }

    def test_merge_key_may_be_overridden(self, tmp_path):
        text = (
            "base: &base\n  api_base: http://a/v1\n  name: x\n"
            "m:\n  <<: *base\n  api_base: http://b/v1\n"
        )
        loaded = load_eval_config(_write(tmp_path, text))
        assert loaded == {
            "base": {"api_base": "http://a/v1", "name": "x"},
            "m": {"api_base": "http://b/v1", "name": "x"},
        }

    def test_two_merge_keys_are_not_duplicates(self, tmp_path):
        text = "a: &a {x: 1}\nb: &b {y: 2}\nm:\n  <<: *a\n  <<: *b\n"
        loaded = load_eval_config(_write(tmp_path, text))
        assert loaded == {"a": {"x": 1}, "b": {"y": 2}, "m": {"x": 1, "y": 2}}

    def test_merge_chain_through_a_nested_anchor_may_be_overridden(self, tmp_path):
        # The anchor is built after ``m`` merged from it: its node already
        # holds the inherited ``a``, which must not read as a repeat.
        text = (
            "base: &base {a: 1}\n"
            "defaults:\n  inner: &d\n    <<: *base\n    a: 2\n"
            "m:\n  <<: *d\n"
        )
        assert load_eval_config(_write(tmp_path, text)) == yaml.safe_load(text)

    def test_duplicate_after_a_merge_is_still_refused(self, tmp_path):
        text = "base: &base {x: 1}\nm:\n  <<: *base\n  y: 1\n  y: 2\n"
        with pytest.raises(DuplicateKeyError, match="'y' at line 5"):
            load_eval_config(_write(tmp_path, text))

    def test_keys_equal_after_construction_are_duplicates(self, tmp_path):
        # ``1`` and ``0x1`` both construct to the int 1.
        with pytest.raises(DuplicateKeyError, match="duplicate key 1 at line 2"):
            load_eval_config(_write(tmp_path, "1: a\n0x1: b\n"))

    def test_invalid_yaml_raises_yaml_error(self, tmp_path):
        with pytest.raises(yaml.YAMLError):
            load_eval_config(_write(tmp_path, "models:\n  - [invalid\n"))

    def test_loads_are_independent(self, tmp_path):
        dup = _write(tmp_path, "a: 1\na: 2\n")
        clean = tmp_path / "clean.yaml"
        clean.write_text("a: 1\nb: 2\n", encoding="utf-8")
        with pytest.raises(DuplicateKeyError):
            load_eval_config(dup)
        assert load_eval_config(clean) == {"a": 1, "b": 2}


class TestEveryEntryPointRefusesADuplicate:
    """A bare ``yaml.safe_load`` on any entry point fails here."""

    @pytest.mark.parametrize(
        "argv",
        [
            ["run"],
            ["run", "--dry-run"],
            ["eval"],
            ["eval", "--dry-run"],
        ],
        ids=["run", "run-dry-run", "eval", "eval-dry-run"],
    )
    def test_cli(self, tmp_path, argv):
        config = _write(tmp_path, _DUPLICATE_API_BASE)
        result = cli_runner.invoke(app, [*argv, str(config), "-o", "json"])
        assert result.exit_code == 1
        payload = json.loads(result.stdout)
        assert payload["ok"] is False
        assert _DUPLICATE_MESSAGE in json.dumps(payload)

    def test_dry_run_reports_a_failed_yaml_check(self, tmp_path):
        config = _write(tmp_path, _DUPLICATE_API_BASE)
        result = cli_runner.invoke(app, ["run", "--dry-run", str(config), "-o", "json"])
        checks = json.loads(result.stdout)["data"]["checks"]
        assert checks[-1] == {
            "name": "yaml_syntax",
            "ok": False,
            "detail": f"{config}: {_DUPLICATE_MESSAGE}",
            "warnings": [],
        }

    def test_eval_session(self, tmp_path):
        config = _write(tmp_path, _DUPLICATE_API_BASE)
        with pytest.raises(DuplicateKeyError, match=re.escape(_DUPLICATE_MESSAGE)):
            EvalSession(config_path=str(config))

    @pytest.mark.anyio
    async def test_run_refuses_before_planning_a_launch(self, tmp_path):
        # A later EvalSession refusal would come after engines started.
        from sieval.cli.run import _run_all

        config = _write(tmp_path, _DUPLICATE_API_BASE)
        with (
            patch("sieval.cli.run._prepare_launch_batch") as prepare,
            pytest.raises(DuplicateKeyError, match=re.escape(_DUPLICATE_MESSAGE)),
        ):
            await _run_all(config)
        prepare.assert_not_called()

    @pytest.mark.anyio
    async def test_run_dry_run_plans_nothing(self, tmp_path):
        from sieval.cli.run import _run_dry_run

        config = _write(tmp_path, _DUPLICATE_API_BASE)
        with patch("sieval.cli.run._prepare_launch_batch") as prepare:
            result = await _run_dry_run(config)
        prepare.assert_not_called()
        assert result["n_errors"] == 1
        assert result["launches"] is None


def test_no_other_yaml_load_in_the_cli():
    """A new ``yaml`` load under ``sieval/cli`` must say what it reads."""
    allowed = {
        # read_persisted_config and the effective_config diff: files sieval wrote.
        "session.py": 3,
        # Recipe files and `sieval infer` target files.
        "infer/recipe.py": 1,
        "infer/commands.py": 1,
    }
    cli = Path(sieval.cli.__file__).parent
    found: dict[str, int] = {}
    for path in sorted(cli.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "yaml"
                and node.func.attr in {"load", "safe_load", "full_load", "unsafe_load"}
            ):
                rel = path.relative_to(cli).as_posix()
                found[rel] = found.get(rel, 0) + 1
    assert found == allowed
