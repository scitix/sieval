"""
Unit tests for sizing tp from the declared engine memory.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from loguru import logger

from sieval.infer.introspect import ModelIdentity
from sieval.infer.topology.memory import (
    MemoryDemand,
    memory_demand,
    memory_fit,
    size_tp,
)

_H100_MIB = 81920

# Qwen3-30B-A3B bf16 (scitix/sieval#251): 61 GB of weights, 48 layers × 4 KV
# heads × 128 head dim → 96 KiB of KV per token.
_QWEN3_30B_A3B = MemoryDemand(
    weight_bytes=61.06e9,
    fraction=0.87,
    context=32768,
    kv_layers=48,
    kv_heads=4,
    kv_head_dim=128,
)


@pytest.fixture
def warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    yield messages
    logger.remove(sink)


class TestMemoryFit:
    def test_kv_bytes_per_token(self) -> None:
        assert _QWEN3_30B_A3B.kv_bytes_per_token(1) == 96 * 1024

    def test_kv_heads_shard_then_replicate(self) -> None:
        assert _QWEN3_30B_A3B.kv_bytes_per_token(2) == 48 * 1024
        # tp=8 > 4 KV heads: each GPU still holds one head
        assert _QWEN3_30B_A3B.kv_bytes_per_token(8) == 24 * 1024

    def test_kv_unknown_without_shape(self) -> None:
        assert MemoryDemand(weight_bytes=1e9).kv_bytes_per_token(1) is None

    def test_qwen3_30b_a3b_fits_one_h100(self) -> None:
        fit = memory_fit(_QWEN3_30B_A3B, 1, _H100_MIB)
        assert fit.fits
        # 0.87 × 80 GiB = 74.7 GB; minus 61.1 GB of weights = 13.7 GB of KV
        assert fit.kv_capacity_tokens == pytest.approx(138_900, rel=0.01)
        assert fit.low_room  # fewer than 8 × 32k sequences

    def test_weights_alone_over_budget_do_not_fit(self) -> None:
        demand = MemoryDemand(weight_bytes=80e9, fraction=0.87)
        assert not memory_fit(demand, 1, _H100_MIB).fits
        assert memory_fit(demand, 2, _H100_MIB).fits

    def test_kv_of_one_sequence_must_fit_beside_weights(self) -> None:
        """Weights fit, but one full-context sequence of KV does not."""
        demand = MemoryDemand(
            weight_bytes=70e9,
            fraction=0.87,
            context=131072,
            kv_layers=64,
            kv_heads=8,
            kv_head_dim=128,
        )
        fit = memory_fit(demand, 1, _H100_MIB)
        assert fit.weight_bytes_per_gpu < fit.budget_bytes
        assert not fit.fits

    def test_no_fraction_means_whole_gpu(self) -> None:
        fit = memory_fit(MemoryDemand(weight_bytes=80e9), 1, _H100_MIB)
        assert fit.budget_bytes == _H100_MIB * 1024 * 1024
        assert fit.fits
        assert not fit.low_room  # no declared fraction, no room judgement
        assert "no memory fraction declared" in fit.describe()

    def test_describe_states_kv_room(self) -> None:
        text = memory_fit(_QWEN3_30B_A3B, 1, _H100_MIB).describe()
        assert "tp=1" in text
        assert "0.87 × GPU" in text
        assert "for context 32768" in text

    def test_describe_without_kv_shape(self) -> None:
        demand = MemoryDemand(weight_bytes=1e9, fraction=0.9)
        assert "KV not checked" in memory_fit(demand, 1, _H100_MIB).describe()


class TestSizeTp:
    def test_issue_251_qwen3_30b_a3b_on_one_h100(self, warnings: list[str]) -> None:
        """Proceeds with tp=1 and warns that concurrency will be low."""
        fit = size_tp(_QWEN3_30B_A3B, gpu_count=1, gpu_memory_mib=_H100_MIB)
        assert fit.tp == 1
        assert len(warnings) == 1
        assert "little KV room" in warnings[0]

    def test_smallest_power_of_two_that_fits(self) -> None:
        demand = MemoryDemand(weight_bytes=200e9, fraction=0.9)
        # 200 GB over 77.3 GB budgets → 3 GPUs, rounded up to 4
        assert size_tp(demand, gpu_count=8, gpu_memory_mib=_H100_MIB).tp == 4

    def test_roomy_fit_does_not_warn(self, warnings: list[str]) -> None:
        demand = MemoryDemand(
            weight_bytes=16e9,
            fraction=0.87,
            context=32768,
            kv_layers=36,
            kv_heads=8,
            kv_head_dim=128,
        )
        assert size_tp(demand, gpu_count=1, gpu_memory_mib=_H100_MIB).tp == 1
        assert warnings == []

    def test_stated_tp_is_used_as_given(self) -> None:
        fit = size_tp(
            _QWEN3_30B_A3B, gpu_count=8, gpu_memory_mib=_H100_MIB, stated_tp=4
        )
        assert fit.tp == 4

    def test_stated_tp_that_does_not_fit_is_refused(self) -> None:
        demand = MemoryDemand(weight_bytes=145e9, fraction=0.85)
        with pytest.raises(RuntimeError, match=r"tp=1 does not fit"):
            size_tp(demand, gpu_count=8, gpu_memory_mib=_H100_MIB, stated_tp=1)

    def test_more_gpus_than_present_is_refused(self) -> None:
        demand = MemoryDemand(weight_bytes=160e9, fraction=0.85)
        with pytest.raises(RuntimeError, match=r"requires tp=4 but only 2 GPUs"):
            size_tp(demand, gpu_count=2, gpu_memory_mib=_H100_MIB)

    def test_kv_alone_over_budget_never_fits(self) -> None:
        """A KV demand that does not shrink with tp is refused, not looped on."""
        demand = MemoryDemand(
            weight_bytes=1e9,
            fraction=0.5,
            context=10**9,
            kv_layers=1,
            kv_heads=1,
            kv_head_dim=128,
        )
        with pytest.raises(RuntimeError, match="at any tp"):
            size_tp(demand, gpu_count=8, gpu_memory_mib=_H100_MIB)


class TestMemoryDemand:
    def _checkpoint(self, tmp_path: Path, **extra: object) -> str:
        config = {
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "max_position_embeddings": 40960,
            "torch_dtype": "bfloat16",
            **extra,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 61_064_000_000}})
        )
        return str(tmp_path)

    def _identity(self) -> ModelIdentity:
        return ModelIdentity(
            architecture="Qwen3MoeForCausalLM",
            family="qwen3",
            param_billions=30.0,
            dtype="bfloat16",
        )

    def test_reads_sglang_params(self, tmp_path: Path) -> None:
        demand = memory_demand(
            checkpoint=self._checkpoint(tmp_path),
            identity=self._identity(),
            backend="sglang",
            params={"mem_fraction_static": 0.87, "context_length": 32768},
        )
        assert demand == MemoryDemand(
            weight_bytes=61_064_000_000,
            fraction=0.87,
            context=32768,
            kv_layers=48,
            kv_heads=4,
            kv_head_dim=128,
            kv_dtype_bytes=2.0,
        )

    def test_reads_vllm_params(self, tmp_path: Path) -> None:
        demand = memory_demand(
            checkpoint=self._checkpoint(tmp_path),
            identity=self._identity(),
            backend="vllm",
            # sglang's key is not vllm's: ignored
            params={
                "gpu_memory_utilization": 0.95,
                "max_model_len": 16384,
                "mem_fraction_static": 0.5,
            },
        )
        assert (demand.fraction, demand.context) == (0.95, 16384)

    def test_undeclared_fraction_and_context(self, tmp_path: Path) -> None:
        """No fraction is invented; the context is the checkpoint's native one."""
        demand = memory_demand(
            checkpoint=self._checkpoint(tmp_path),
            identity=self._identity(),
            backend="sglang",
            params={},
        )
        assert demand.fraction is None
        assert demand.context == 40960

    def test_fp8_kv_cache_halves_kv(self, tmp_path: Path) -> None:
        demand = memory_demand(
            checkpoint=self._checkpoint(tmp_path),
            identity=self._identity(),
            backend="vllm",
            params={"kv_cache_dtype": "fp8_e4m3"},
        )
        assert demand.kv_dtype_bytes == 1.0

    def test_weights_fall_back_to_param_count(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text(json.dumps({}))
        demand = memory_demand(
            checkpoint=str(tmp_path),
            identity=self._identity(),
            backend="sglang",
            params={},
        )
        assert demand.weight_bytes == 60e9
        assert demand.kv_layers is None
