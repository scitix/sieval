"""Tests for `sieval infer start`'s capability-layer selection.

Checkpoint mode is the one entry point with no task context to derive the model
type from, so the operator declares it with ``--model-type``. These tests pin
the translation into the recipe's own vocabulary and the refusal to accept the
flag in YAML mode, where the config already answers the question.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from sieval.cli.infer import infer_app
from sieval.infer.topology.models import (
    DeploymentPlan,
    DeviceGroup,
    ParallelTopology,
    RoleAssignment,
    WellKnownRole,
)
from sieval.infer.topology.resolver import ResolveResult

runner = CliRunner()


def _failure_text(result) -> str:
    """Everywhere a failed command's message can surface, joined.

    A raised exception and rendered stdout are not interchangeable, and which
    one carries the message is a property of the CLI's error plumbing, not of
    the behavior under test. Today an escaping `ValueError` reaches
    `result.exception`; once command failures are funnelled through `render()`
    (scitix/sieval#85) the same message arrives on stdout with
    `result.exception` set to `SystemExit`. Asserting on both keeps these tests
    about the message, so they do not depend on which change lands first.
    """
    return f"{result.output}\n{result.exception!r}"


def _fake_plan() -> DeploymentPlan:
    return DeploymentPlan(
        checkpoint="/ckpt",
        backend="sglang",
        assignments=(
            RoleAssignment(
                role=WellKnownRole.FULL,
                devices=DeviceGroup(count=1, gpu_model="H100"),
                topology=ParallelTopology(tp=1, dp=1, pp=1),
                engine_params={"context_length": 32768},
            ),
        ),
    )


def _write_checkpoint(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"})
    )
    return path


class TestInferStartModelType:
    """`--model-type` selects the recipe capability layer in checkpoint mode."""

    def _capability_for(self, tmp_path: Path, args: list[str]) -> str:
        """Run `infer start --dry-run` and return the resolved capability."""
        resolve = AsyncMock(return_value=ResolveResult(plan=_fake_plan(), steps=()))
        translator = MagicMock()
        cmd = MagicMock()
        cmd.cli_args = ["sglang", "--model-path", "/ckpt"]
        cmd.health_url = "http://localhost:8000/health"
        cmd.env = {}
        translator.translate.return_value = [cmd]

        with (
            patch("sieval.cli.infer.commands.auto_resolve_plan", new=resolve),
            patch(
                "sieval.cli.infer.commands.get_translator",
                return_value=translator,
            ),
            patch("sieval.cli.infer.commands.validate_plan", return_value=[]),
        ):
            result = runner.invoke(infer_app, args)

        assert result.exit_code == 0, result.output
        resolve.assert_awaited_once()
        call = resolve.await_args
        assert call is not None
        return call.kwargs["capability"]

    def test_default_is_instruct(self, tmp_path: Path):
        ckpt = _write_checkpoint(tmp_path / "Qwen3-4B")
        assert (
            self._capability_for(tmp_path, ["start", str(ckpt), "--dry-run"])
            == "instruct"
        )

    def test_gen_selects_the_base_layer(self, tmp_path: Path):
        """The point of the flag: serve a base checkpoint without instruct params."""
        ckpt = _write_checkpoint(tmp_path / "Qwen3-4B-Base")
        capability = self._capability_for(
            tmp_path,
            ["start", str(ckpt), "--dry-run", "--model-type", "gen"],
        )
        assert capability == "base"

    def test_chat_selects_the_instruct_layer(self, tmp_path: Path):
        ckpt = _write_checkpoint(tmp_path / "Qwen3-4B")
        capability = self._capability_for(
            tmp_path,
            ["start", str(ckpt), "--dry-run", "--model-type", "chat"],
        )
        assert capability == "instruct"

    @pytest.mark.parametrize("bad", ["base", "instruct", "cht"])
    def test_recipe_vocabulary_and_typos_are_rejected(self, tmp_path: Path, bad: str):
        """The flag takes config words; `--model-type base` must not silently pass.

        `base` is the recipe layer's own name for what `gen` selects, so it is
        the likely slip — and accepting it as an unknown default would pick the
        *instruct* layer, the opposite of what was asked for.
        """
        ckpt = _write_checkpoint(tmp_path / "Qwen3-4B")
        result = runner.invoke(
            infer_app,
            ["start", str(ckpt), "--dry-run", "--model-type", bad],
        )
        assert result.exit_code != 0
        assert "expected 'chat' or 'gen'" in _failure_text(result)

    def test_rejected_in_yaml_mode(self, tmp_path: Path):
        """YAML already carries the type; accepting the flag would let them differ."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            yaml.safe_dump(
                {
                    "models": {
                        "m": {
                            "type": "chat",
                            "infer": {"backend": "sglang", "recipe": "qwen3-4b"},
                        }
                    }
                }
            )
        )

        result = runner.invoke(
            infer_app,
            ["start", str(cfg), "--dry-run", "--model-type", "gen"],
        )

        assert result.exit_code != 0
        # Typer renders the message inside a wrapped Rich panel, so match a
        # fragment short enough to survive line breaking.
        assert "applies to checkpoint mode" in _failure_text(result)
