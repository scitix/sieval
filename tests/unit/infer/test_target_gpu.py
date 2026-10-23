"""Tests for ``SIEVAL_TARGET_GPU``: declared GPUs in place of ``nvidia-smi``.

A plan resolved from a declaration must be the plan a node with those GPUs
resolves, byte for byte, since the serving recipe is part of the inference
identity a resume compares.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from sieval.cli.infer.recipe import resolve_infer_config
from sieval.cli.session import unwrap_proxies
from sieval.infer.introspect import (
    TARGET_GPU_ENV,
    GPUInfo,
    check_launch_gpu,
    declared_target_gpu,
    format_target_gpu,
    parse_target_gpu,
)
from sieval.infer.topology import plan_from_mapping
from sieval.infer.topology.resolver import auto_resolve_plan

# As `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`
# prints an 8-GPU H100 node.
H100 = GPUInfo(model="NVIDIA H100 80GB HBM3", count=8, memory_mib=81559)
A100_40G = GPUInfo(model="NVIDIA A100-SXM4-40GB", count=4, memory_mib=40960)

_RECIPE_GPU = "sieval.cli.infer.recipe.detect_local_gpu"
_RESOLVER_GPU = "sieval.infer.topology.resolver.detect_local_gpu"
_LAUNCH_GPU = "sieval.infer.introspect.detect_local_gpu"


def _qwen3_checkpoint(root: Path, name: str = "Qwen3-32B") -> Path:
    checkpoint = root / name
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "vocab_size": 151936,
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "intermediate_size": 25600,
                "torch_dtype": "bfloat16",
            }
        )
    )
    return checkpoint


class TestParseTargetGpu:
    def test_reads_the_fields_nvidia_smi_reports(self) -> None:
        assert parse_target_gpu("NVIDIA H100 80GB HBM3:8:81559") == H100

    def test_a_name_may_hold_colons(self) -> None:
        gpu = parse_target_gpu("vendor:card:2:1024")
        assert gpu == GPUInfo(model="vendor:card", count=2, memory_mib=1024)

    def test_round_trips_through_its_text(self) -> None:
        assert parse_target_gpu(format_target_gpu(H100)) == H100

    @pytest.mark.parametrize(
        "text",
        ["H100-80G:8", "H100", ":8:81559", "H100:0:81559", "H100:8:0", "H100:x:1"],
    )
    def test_refuses_anything_else_naming_the_form(self, text: str) -> None:
        with pytest.raises(ValueError, match=r"<name>:<count>:<memory MiB>"):
            parse_target_gpu(text)

    def test_unset_or_blank_declares_nothing(self) -> None:
        assert declared_target_gpu({}) is None
        assert declared_target_gpu({TARGET_GPU_ENV: "  "}) is None
        assert declared_target_gpu({TARGET_GPU_ENV: format_target_gpu(H100)}) == H100


class TestDeclaredEqualsDetected:
    """The plan from a declaration is the one detection on that node gives."""

    @pytest.mark.anyio
    @pytest.mark.parametrize("gpu", [H100, A100_40G], ids=["h100x8", "a100-40gx4"])
    async def test_path_only_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gpu: GPUInfo
    ) -> None:
        checkpoint = str(_qwen3_checkpoint(tmp_path))
        with patch(_RESOLVER_GPU, new=AsyncMock(return_value=gpu)):
            detected = await auto_resolve_plan(checkpoint, capability="instruct")

        monkeypatch.setenv(TARGET_GPU_ENV, format_target_gpu(gpu))
        nvidia_smi = AsyncMock(return_value=None)  # a CPU node
        with patch(_RESOLVER_GPU, new=nvidia_smi):
            declared = await auto_resolve_plan(checkpoint, capability="instruct")

        nvidia_smi.assert_not_awaited()
        assert declared.plan == detected.plan
        assert yaml.safe_dump(unwrap_proxies(declared.plan), sort_keys=False) == (
            yaml.safe_dump(unwrap_proxies(detected.plan), sort_keys=False)
        )

    @pytest.mark.anyio
    @pytest.mark.parametrize("gpu", [H100, A100_40G], ids=["h100x8", "a100-40gx4"])
    async def test_yaml_infer_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gpu: GPUInfo
    ) -> None:
        checkpoint = str(_qwen3_checkpoint(tmp_path))
        config = tmp_path / "cfg.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "models": {
                        "m": {"infer": {"backend": "vllm", "checkpoint": checkpoint}}
                    },
                    "tasks": {},
                }
            )
        )
        with patch(_RECIPE_GPU, new=AsyncMock(return_value=gpu)):
            detected = await resolve_infer_config(config, "m")

        monkeypatch.setenv(TARGET_GPU_ENV, format_target_gpu(gpu))
        nvidia_smi = AsyncMock(return_value=None)
        with patch(_RECIPE_GPU, new=nvidia_smi):
            declared = await resolve_infer_config(config, "m")

        nvidia_smi.assert_not_awaited()
        assert declared.plan == detected.plan
        # The hardware layer applied: an A100-40G and an H100 differ.
        assert declared.plan.assignments[0].engine_params

    @pytest.mark.anyio
    async def test_another_declaration_resolves_another_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The check above has teeth: the declaration reaches the plan."""
        checkpoint = str(_qwen3_checkpoint(tmp_path))
        plans = []
        for gpu in (H100, A100_40G):
            monkeypatch.setenv(TARGET_GPU_ENV, format_target_gpu(gpu))
            plans.append((await auto_resolve_plan(checkpoint)).plan)
        assert plans[0] != plans[1]

    @pytest.mark.anyio
    async def test_no_gpu_and_no_declaration_names_the_variable(
        self, tmp_path: Path
    ) -> None:
        checkpoint = str(_qwen3_checkpoint(tmp_path))
        with (
            patch(_RESOLVER_GPU, new=AsyncMock(return_value=None)),
            pytest.raises(RuntimeError, match=TARGET_GPU_ENV),
        ):
            await auto_resolve_plan(checkpoint)


class TestCheckLaunchGpu:
    @pytest.mark.anyio
    async def test_nothing_declared_checks_nothing(self) -> None:
        nvidia_smi = AsyncMock(return_value=None)
        with patch(_LAUNCH_GPU, new=nvidia_smi):
            await check_launch_gpu()
        nvidia_smi.assert_not_awaited()

    @pytest.mark.anyio
    async def test_the_declared_gpus_launch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(TARGET_GPU_ENV, format_target_gpu(H100))
        with patch(_LAUNCH_GPU, new=AsyncMock(return_value=H100)):
            await check_launch_gpu()

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("here", "named"),
        [(A100_40G, "NVIDIA A100-SXM4-40GB:4:40960"), (None, "no GPU")],
    )
    async def test_other_gpus_are_refused(
        self, monkeypatch: pytest.MonkeyPatch, here: GPUInfo | None, named: str
    ) -> None:
        monkeypatch.setenv(TARGET_GPU_ENV, format_target_gpu(H100))
        with (
            patch(_LAUNCH_GPU, new=AsyncMock(return_value=here)),
            pytest.raises(RuntimeError, match=named),
        ):
            await check_launch_gpu()

    @pytest.mark.anyio
    async def test_the_refusal_names_the_field_that_differs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # One SKU, another reported memory.total (e.g. ECC on).
        here = GPUInfo(model=H100.model, count=H100.count, memory_mib=80000)
        monkeypatch.setenv(TARGET_GPU_ENV, format_target_gpu(H100))
        with (
            patch(_LAUNCH_GPU, new=AsyncMock(return_value=here)),
            pytest.raises(RuntimeError) as refused,
        ):
            await check_launch_gpu()
        message = str(refused.value)
        assert f"memory MiB 80000 (declared {H100.memory_mib})" in message
        assert "name " not in message and "count " not in message


class TestPlanFromMapping:
    @pytest.mark.anyio
    async def test_a_recorded_plan_reads_back_equal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(TARGET_GPU_ENV, format_target_gpu(H100))
        plan = (await auto_resolve_plan(str(_qwen3_checkpoint(tmp_path)))).plan
        recorded = yaml.safe_load(yaml.safe_dump(unwrap_proxies(plan), sort_keys=False))
        assert plan_from_mapping(recorded) == plan

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ValueError):
            plan_from_mapping({"checkpoint": "c", "backend": "vllm"})
