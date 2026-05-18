"""
Tests for sieval.infer.introspect — model introspection and GPU detection.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sieval.infer.introspect import (
    ModelIdentity,
    QuantizationInfo,
    _estimate_params_billions,
    _extract_bits,
    _extract_size_from_name,
    _match_family,
    _refine_family_from_name,
    _resolve_param_billions,
    bytes_per_param,
    detect_local_gpu,
    extract_moe_info,
    introspect_checkpoint,
)


async def async_iter[T](items: list[T]) -> AsyncIterator[T]:
    """Helper: wrap a list into an async iterator for mocking ``async for``."""
    for item in items:
        yield item


# ===================================================================
# _match_family
# ===================================================================
class TestMatchFamily:
    def test_qwen3(self) -> None:
        assert _match_family("Qwen3ForCausalLM") == "qwen3"

    def test_qwen2(self) -> None:
        """Qwen2ForCausalLM → qwen2 (shared by Qwen2 and Qwen2.5)."""
        assert _match_family("Qwen2ForCausalLM") == "qwen2"

    def test_gpt_oss(self) -> None:
        """GptOssForCausalLM → gpt-oss (OpenAI's official slug)."""
        assert _match_family("GptOssForCausalLM") == "gpt-oss"

    def test_unknown_fallback(self) -> None:
        """Unknown architecture → cleaned lowercase."""
        result = _match_family("FooBarForCausalLM")
        assert result == "foobar"


class TestRefineFamilyFromName:
    """Qwen2.5 and Qwen2 share Qwen2ForCausalLM — refinement uses model name."""

    def test_qwen2_5_from_name_or_path(self) -> None:
        config = {"_name_or_path": "Qwen/Qwen2.5-7B-Instruct"}
        assert _refine_family_from_name("qwen2", "/models/x", config) == "qwen2.5"

    def test_qwen2_5_from_checkpoint_path(self) -> None:
        config: dict[str, object] = {}
        assert (
            _refine_family_from_name("qwen2", "/models/Qwen2.5-72B", config)
            == "qwen2.5"
        )

    def test_qwen2_stays_qwen2(self) -> None:
        config = {"_name_or_path": "Qwen/Qwen2-7B-Instruct"}
        assert _refine_family_from_name("qwen2", "/models/Qwen2-7B", config) == "qwen2"

    def test_non_qwen2_unchanged(self) -> None:
        config = {"_name_or_path": "Qwen/Qwen2.5-7B"}
        assert _refine_family_from_name("qwen3", "/models/x", config) == "qwen3"

    def test_qwen2_5_dash_separator(self) -> None:
        """Handles Qwen2-5 naming variant."""
        config = {"_name_or_path": "custom/Qwen2-5-14B"}
        assert _refine_family_from_name("qwen2", "/models/x", config) == "qwen2.5"

    def test_qwen2_5_underscore_separator(self) -> None:
        """Handles Qwen2_5 naming variant."""
        config = {"_name_or_path": "custom/Qwen2_5-32B"}
        assert _refine_family_from_name("qwen2", "/models/x", config) == "qwen2.5"


# ===================================================================
# _extract_size_from_name
# ===================================================================
class TestExtractSizeFromName:
    def test_standard_hf_names(self) -> None:
        assert _extract_size_from_name("Qwen3-8B") == 8.0
        assert _extract_size_from_name("Qwen3-0.5B") == 0.5
        assert _extract_size_from_name("Qwen2.5-72B") == 72.0

    def test_path_style(self) -> None:
        assert _extract_size_from_name("/models/Qwen3-8B") == 8.0
        assert _extract_size_from_name("Qwen/Qwen3-4B") == 4.0

    def test_suffix_after_size(self) -> None:
        assert _extract_size_from_name("Qwen3-8B-AWQ") == 8.0

    def test_no_size_label(self) -> None:
        assert _extract_size_from_name("my-custom-model") is None
        assert _extract_size_from_name("") is None


# ===================================================================
# _resolve_param_billions
# ===================================================================
class TestResolveParamBillions:
    def test_name_or_path_priority(self) -> None:
        """_name_or_path in config takes first priority."""
        config = {
            "_name_or_path": "Qwen/Qwen3-8B",
            "vocab_size": 151936,
            "hidden_size": 2560,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 9216,
        }
        result = _resolve_param_billions("/models/Qwen3-4B", config)
        # Should use _name_or_path (8B), not directory name (4B) or estimation
        assert result == 8.0

    def test_directory_name_fallback(self) -> None:
        """When _name_or_path has no size, falls back to directory name."""
        config = {
            "_name_or_path": "some/custom/path",
            "vocab_size": 151936,
            "hidden_size": 2560,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
        }
        result = _resolve_param_billions("/models/Qwen3-4B", config)
        assert result == 4.0

    def test_estimation_fallback(self) -> None:
        """When no name has size labels, falls back to estimation."""
        config = {
            "vocab_size": 151936,
            "hidden_size": 2560,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 9216,
        }
        result = _resolve_param_billions("/models/my-custom-model", config)
        # Should be estimated from config (Qwen3-4B-like dimensions)
        assert 3.0 < result < 6.0


# ===================================================================
# _estimate_params_billions
# ===================================================================
class TestEstimateParams:
    def test_qwen3_4b(self) -> None:
        """Qwen3-4B should estimate to roughly 4B params."""
        config = {
            "vocab_size": 151936,
            "hidden_size": 2560,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 9216,
        }
        estimate = _estimate_params_billions(config)
        # Should be in the ballpark of 4B (3-6B range)
        assert 3.0 < estimate < 6.0

    def test_qwen3_8b(self) -> None:
        """Qwen3-8B should estimate to roughly 8B params."""
        config = {
            "vocab_size": 151936,
            "hidden_size": 4096,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 12288,
        }
        estimate = _estimate_params_billions(config)
        assert 6.0 < estimate < 12.0

    def test_missing_fields_returns_zero(self) -> None:
        assert _estimate_params_billions({}) == 0.0
        assert _estimate_params_billions({"vocab_size": 32000}) == 0.0

    def test_gpt_oss_120b_moe(self) -> None:
        """gpt-oss-120b: 128 experts must be multiplied into MLP params."""
        config = {
            "vocab_size": 201088,
            "hidden_size": 2880,
            "num_hidden_layers": 36,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "intermediate_size": 2880,
            "num_local_experts": 128,
        }
        estimate = _estimate_params_billions(config)
        # Actual ~117B; must land in recipe bucket [100, 150).
        assert 100.0 < estimate < 150.0

    def test_gpt_oss_20b_moe(self) -> None:
        """gpt-oss-20b: 32 experts, ~21B total params."""
        config = {
            "vocab_size": 201088,
            "hidden_size": 2880,
            "num_hidden_layers": 24,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "intermediate_size": 2880,
            "num_local_experts": 32,
        }
        estimate = _estimate_params_billions(config)
        assert 15.0 < estimate < 30.0

    def test_qwen3_30b_a3b_moe(self) -> None:
        """Qwen3-30B-A3B: must use moe_intermediate_size, not dense width."""
        config = {
            "vocab_size": 151936,
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 6144,
            "moe_intermediate_size": 768,
            "num_experts": 128,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
        }
        estimate = _estimate_params_billions(config)
        # Actual ~30.5B.
        assert 25.0 < estimate < 40.0

    def test_qwen3_235b_a22b_moe(self) -> None:
        """Qwen3-235B-A22B: large MoE bucket."""
        config = {
            "vocab_size": 151936,
            "hidden_size": 4096,
            "num_hidden_layers": 94,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 12288,
            "moe_intermediate_size": 1536,
            "num_experts": 128,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
        }
        estimate = _estimate_params_billions(config)
        # Actual ~235B.
        assert 200.0 < estimate < 260.0


# ===================================================================
# introspect_checkpoint (async)
# ===================================================================
class TestIntrospectCheckpoint:
    @pytest.mark.anyio
    async def test_valid_checkpoint(self, tmp_path: Path) -> None:
        """Reads config.json and returns correct ModelIdentity."""
        config = {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "vocab_size": 151936,
            "hidden_size": 2560,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 9216,
            "torch_dtype": "bfloat16",
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        identity = await introspect_checkpoint(str(tmp_path))

        assert identity.architecture == "Qwen3ForCausalLM"
        assert identity.family == "qwen3"
        assert identity.param_billions > 0
        assert identity.dtype == "bfloat16"

    @pytest.mark.anyio
    async def test_no_config_json(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="config.json"):
            await introspect_checkpoint(str(tmp_path))

    @pytest.mark.anyio
    async def test_no_architectures(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "test"}))
        with pytest.raises(ValueError, match="architectures"):
            await introspect_checkpoint(str(tmp_path))

    @pytest.mark.anyio
    async def test_default_dtype(self, tmp_path: Path) -> None:
        """Missing torch_dtype defaults to float16."""
        config = {
            "architectures": ["Qwen2ForCausalLM"],
            "vocab_size": 151936,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        identity = await introspect_checkpoint(str(tmp_path))
        assert identity.dtype == "float16"

    @pytest.mark.anyio
    async def test_size_from_directory_name(self, tmp_path: Path) -> None:
        """Size extracted from directory name when _name_or_path absent."""
        model_dir = tmp_path / "Qwen3-8B"
        model_dir.mkdir()
        config = {
            "architectures": ["Qwen3ForCausalLM"],
            "vocab_size": 151936,
            "hidden_size": 4096,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 12288,
            "torch_dtype": "bfloat16",
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        identity = await introspect_checkpoint(str(model_dir))
        assert identity.param_billions == 8.0


# ===================================================================
# _extract_bits
# ===================================================================
class TestExtractBits:
    """Tests for _extract_bits() — unified bits extraction from quantization_config."""

    def test_awq_bits(self) -> None:
        config = {"quant_method": "awq", "bits": 4, "group_size": 128}
        assert _extract_bits(config) == 4

    def test_gptq_bits(self) -> None:
        config = {"quant_method": "gptq", "bits": 4, "group_size": 128}
        assert _extract_bits(config) == 4

    def test_gptq_8bit(self) -> None:
        config = {"quant_method": "gptq", "bits": 8}
        assert _extract_bits(config) == 8

    def test_hqq_nbits(self) -> None:
        config = {"quant_method": "hqq", "nbits": 4, "group_size": 64}
        assert _extract_bits(config) == 4

    def test_compressed_tensors_fp8(self) -> None:
        config = {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "weights": {"num_bits": 8, "type": "float", "symmetric": True},
                },
            },
        }
        assert _extract_bits(config) == 8

    def test_compressed_tensors_int4(self) -> None:
        config = {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "weights": {"num_bits": 4, "type": "int", "symmetric": True},
                },
            },
        }
        assert _extract_bits(config) == 4

    def test_bitsandbytes_4bit(self) -> None:
        config = {"quant_method": "bitsandbytes", "load_in_4bit": True}
        assert _extract_bits(config) == 4

    def test_bitsandbytes_8bit(self) -> None:
        config = {"quant_method": "bitsandbytes", "load_in_8bit": True}
        assert _extract_bits(config) == 8

    def test_fbgemm_fp8(self) -> None:
        config = {"quant_method": "fbgemm_fp8", "activation_scale_ub": 1200.0}
        assert _extract_bits(config) == 8

    def test_fp8_deepseek(self) -> None:
        config = {"quant_method": "fp8", "weight_block_size": [128, 128]}
        assert _extract_bits(config) == 8

    def test_mxfp4(self) -> None:
        """MXFP4 maps to 4-bit (approximated from ~4.25 bpw)."""
        config = {"quant_method": "mxfp4", "modules_to_not_convert": []}
        assert _extract_bits(config) == 4

    def test_unknown_method_fallback(self) -> None:
        config = {"quant_method": "some_future_method"}
        assert _extract_bits(config) == 8  # conservative fallback


# ===================================================================
# introspect_checkpoint — quantization extraction
# ===================================================================
class TestIntrospectQuantization:
    """Tests for quantization extraction in introspect_checkpoint."""

    @pytest.mark.anyio
    async def test_fp8_checkpoint(self, tmp_path: Path) -> None:
        config = {
            "architectures": ["Qwen3ForCausalLM"],
            "torch_dtype": "bfloat16",
            "vocab_size": 152064,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "quantization_config": {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            },
        }
        model_dir = tmp_path / "Qwen3-8B-FP8"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps(config))

        identity = await introspect_checkpoint(str(model_dir))

        assert identity.quantization is not None
        assert identity.quantization.quant_method == "fp8"
        assert identity.quantization.bits == 8
        assert identity.quantization.raw_config == config["quantization_config"]

    @pytest.mark.anyio
    async def test_awq_checkpoint(self, tmp_path: Path) -> None:
        config = {
            "architectures": ["Qwen3ForCausalLM"],
            "torch_dtype": "float16",
            "vocab_size": 152064,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
            },
        }
        model_dir = tmp_path / "Qwen3-8B-AWQ"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps(config))

        identity = await introspect_checkpoint(str(model_dir))

        assert identity.quantization is not None
        assert identity.quantization.quant_method == "awq"
        assert identity.quantization.bits == 4

    @pytest.mark.anyio
    async def test_no_quantization(self, tmp_path: Path) -> None:
        config = {
            "architectures": ["Qwen3ForCausalLM"],
            "torch_dtype": "bfloat16",
            "vocab_size": 152064,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
        }
        model_dir = tmp_path / "Qwen3-8B"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps(config))

        identity = await introspect_checkpoint(str(model_dir))

        assert identity.quantization is None


# ===================================================================
# bytes_per_param
# ===================================================================
class TestBytesPerParam:
    """Tests for bytes_per_param() — derive bytes/param from ModelIdentity."""

    def _make_identity(
        self,
        dtype: str = "bfloat16",
        quant: QuantizationInfo | None = None,
    ) -> ModelIdentity:
        return ModelIdentity(
            architecture="Qwen3ForCausalLM",
            family="qwen3",
            param_billions=8.0,
            dtype=dtype,
            quantization=quant,
        )

    def test_bf16_no_quant(self) -> None:
        assert bytes_per_param(self._make_identity("bfloat16")) == 2.0

    def test_fp16_no_quant(self) -> None:
        assert bytes_per_param(self._make_identity("float16")) == 2.0

    def test_fp32_no_quant(self) -> None:
        assert bytes_per_param(self._make_identity("float32")) == 4.0

    def test_fp64_no_quant(self) -> None:
        assert bytes_per_param(self._make_identity("float64")) == 8.0

    def test_unknown_dtype_defaults_to_2(self) -> None:
        assert bytes_per_param(self._make_identity("some_weird_dtype")) == 2.0

    def test_fp8_quantized(self) -> None:
        quant = QuantizationInfo(quant_method="fp8", bits=8, raw_config={})
        identity = self._make_identity(quant=quant)
        assert bytes_per_param(identity) == 1.0

    def test_int4_quantized(self) -> None:
        quant = QuantizationInfo(quant_method="awq", bits=4, raw_config={})
        identity = self._make_identity(quant=quant)
        assert bytes_per_param(identity) == 0.5

    def test_int8_quantized(self) -> None:
        quant = QuantizationInfo(quant_method="gptq", bits=8, raw_config={})
        identity = self._make_identity(quant=quant)
        assert bytes_per_param(identity) == 1.0

    def test_quant_overrides_dtype(self) -> None:
        """When quantized, bits from QuantizationInfo wins over dtype."""
        quant = QuantizationInfo(quant_method="awq", bits=4, raw_config={})
        identity = self._make_identity("float32", quant=quant)
        # float32 = 4 bytes, but quantized to 4-bit = 0.5 bytes
        assert bytes_per_param(identity) == 0.5


# ===================================================================
# detect_local_gpu (async — uses anyio.open_process)
# ===================================================================
class TestDetectLocalGPU:
    @pytest.mark.anyio
    async def test_nvidia_smi_success(self) -> None:
        """Parses nvidia-smi output correctly."""
        stdout_data = b"NVIDIA A100-SXM4-80GB, 81920\nNVIDIA A100-SXM4-80GB, 81920\n"

        mock_stdout = MagicMock()
        mock_stdout.__aiter__ = MagicMock(
            return_value=async_iter([stdout_data]),
        )
        mock_process = MagicMock()
        mock_process.stdout = mock_stdout
        mock_process.returncode = 0
        mock_process.wait = AsyncMock()

        with patch(
            "sieval.infer.introspect.anyio.open_process",
            new_callable=AsyncMock,
            return_value=mock_process,
        ):
            gpu = await detect_local_gpu()

        assert gpu is not None
        assert gpu.model == "NVIDIA A100-SXM4-80GB"
        assert gpu.count == 2
        assert gpu.memory_mib == 81920

    @pytest.mark.anyio
    async def test_nvidia_smi_multi_chunk(self) -> None:
        """Handles stdout split across multiple chunks."""
        chunk1 = b"NVIDIA A100-SXM4-80GB, 81920\n"
        chunk2 = b"NVIDIA A100-SXM4-80GB, 81920\n"
        chunk3 = b"NVIDIA A100-SXM4-80GB, 81920\n"

        mock_stdout = MagicMock()
        mock_stdout.__aiter__ = MagicMock(
            return_value=async_iter([chunk1, chunk2, chunk3]),
        )
        mock_process = MagicMock()
        mock_process.stdout = mock_stdout
        mock_process.returncode = 0
        mock_process.wait = AsyncMock()

        with patch(
            "sieval.infer.introspect.anyio.open_process",
            new_callable=AsyncMock,
            return_value=mock_process,
        ):
            gpu = await detect_local_gpu()

        assert gpu is not None
        assert gpu.count == 3

    @pytest.mark.anyio
    async def test_nvidia_smi_not_found(self) -> None:
        """nvidia-smi binary not on PATH → returns None."""
        with patch(
            "sieval.infer.introspect.anyio.open_process",
            new_callable=AsyncMock,
            side_effect=FileNotFoundError,
        ):
            assert await detect_local_gpu() is None

    @pytest.mark.anyio
    async def test_nvidia_smi_failure(self) -> None:
        """nvidia-smi returns non-zero exit code → returns None."""
        mock_stdout = MagicMock()
        mock_stdout.__aiter__ = MagicMock(
            return_value=async_iter([b""]),
        )
        mock_process = MagicMock()
        mock_process.stdout = mock_stdout
        mock_process.returncode = 1
        mock_process.wait = AsyncMock()

        with patch(
            "sieval.infer.introspect.anyio.open_process",
            new_callable=AsyncMock,
            return_value=mock_process,
        ):
            assert await detect_local_gpu() is None


# ===================================================================
# extract_moe_info
# ===================================================================
class TestExtractMoeInfo:
    """Tests for extract_moe_info() — MoE structure detection."""

    def test_non_moe_model(self) -> None:
        config = {"num_hidden_layers": 32}
        is_moe, num_experts, num_layers = extract_moe_info(config)
        assert is_moe is False
        assert num_experts is None
        assert num_layers == 32

    def test_moe_num_local_experts(self) -> None:
        """Standard HF key (Mixtral, Qwen3-MoE, etc.)."""
        config = {"num_hidden_layers": 64, "num_local_experts": 8}
        is_moe, num_experts, num_layers = extract_moe_info(config)
        assert is_moe is True
        assert num_experts == 8
        assert num_layers == 64

    def test_moe_num_experts(self) -> None:
        """Fallback key for some non-standard configs."""
        config = {"num_hidden_layers": 128, "num_experts": 256}
        is_moe, num_experts, num_layers = extract_moe_info(config)
        assert is_moe is True
        assert num_experts == 256
        assert num_layers == 128

    def test_single_expert_not_moe(self) -> None:
        """num_experts=1 is degenerate — should not be considered MoE."""
        config = {"num_hidden_layers": 32, "num_local_experts": 1}
        is_moe, num_experts, num_layers = extract_moe_info(config)
        assert is_moe is False
        assert num_experts is None
        assert num_layers == 32

    def test_num_local_experts_takes_priority(self) -> None:
        """When both keys present, num_local_experts wins (truthy first)."""
        config = {
            "num_hidden_layers": 64,
            "num_local_experts": 8,
            "num_experts": 16,
        }
        is_moe, num_experts, num_layers = extract_moe_info(config)
        assert is_moe is True
        assert num_experts == 8

    def test_empty_config(self) -> None:
        is_moe, num_experts, num_layers = extract_moe_info({})
        assert is_moe is False
        assert num_experts is None
        assert num_layers == 0
