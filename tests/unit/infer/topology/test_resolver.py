"""
Unit tests for topology resolver.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import pytest

from sieval.infer.topology.models import (
    HardwareEnv,
    ModelProfile,
    ResolveStep,
    UserHints,
    WellKnownRole,
)
from sieval.infer.topology.resolver import (
    _derive_topology,
    _overrides_to_hints,
    compute_tp,
    precision_key,
    resolve_simple,
)

# ---------- compute_tp ----------


class TestComputeTp:
    """Tests for compute_tp() — formula-based TP derivation."""

    def test_72b_bf16_on_80gb(self):
        """72B bf16 on 80GB → tp=4 (matches old resolve.compute_tp)."""
        tp = compute_tp(72.0, 2.0, 81920)
        assert tp == 4

    def test_72b_fp8_on_80gb(self):
        """72B fp8 on 80GB → tp=2."""
        tp = compute_tp(72.0, 1.0, 81920)
        assert tp == 2

    def test_8b_bf16_on_80gb(self):
        """8B bf16 on 80GB → tp=1."""
        tp = compute_tp(8.0, 2.0, 81920)
        assert tp == 1

    def test_degenerate_zero_params(self):
        tp = compute_tp(0.0, 2.0, 81920)
        assert tp == 1

    def test_degenerate_zero_bpp(self):
        tp = compute_tp(72.0, 0.0, 81920)
        assert tp == 1

    def test_degenerate_negative_params(self):
        tp = compute_tp(-1.0, 2.0, 81920)
        assert tp == 1

    def test_power_of_2_alignment(self):
        """Result is always a power of 2."""
        tp = compute_tp(200.0, 2.0, 81920)
        assert tp > 0
        assert (tp & (tp - 1)) == 0  # is power of 2

    def test_236b_bf16_on_80gb(self):
        """236B bf16 on 80GB → tp=16 (472GB / ~53GB per GPU ≈ 9 → next pow2 = 16)."""
        tp = compute_tp(236.0, 2.0, 81920)
        assert tp == 16


# ---------- _derive_topology ----------


class TestDeriveTopology:
    """Tests for _derive_topology() — topology derivation pipeline."""

    def _hw(self, gpu_count: int = 8, memory: int = 81920) -> HardwareEnv:
        return HardwareEnv(
            gpu_count=gpu_count,
            gpu_model="NVIDIA H100-SXM5-80GB",
            gpu_memory_mib=memory,
        )

    def test_basic_72b(self):
        """72B bf16 on 8x80GB → tp=4, dp=2."""
        model = ModelProfile(param_billions=72.0, num_layers=80)
        steps: list[ResolveStep] = []
        topo = _derive_topology(model, self._hw(), None, steps)
        assert topo.tp == 4
        assert topo.dp == 2
        assert topo.pp == 1
        assert len(steps) >= 2  # at least tp + dp steps

    def test_user_hints_override_tp(self):
        model = ModelProfile(param_billions=72.0, num_layers=80)
        hints = UserHints(tp=8)
        steps: list[ResolveStep] = []
        topo = _derive_topology(model, self._hw(), hints, steps)
        assert topo.tp == 8

    def test_user_hints_override_dp(self):
        model = ModelProfile(param_billions=72.0, num_layers=80)
        hints = UserHints(dp=4)
        steps: list[ResolveStep] = []
        topo = _derive_topology(model, self._hw(), hints, steps)
        assert topo.dp == 4

    def test_user_hints_pp(self):
        model = ModelProfile(param_billions=72.0, num_layers=80)
        hints = UserHints(pp=2)
        steps: list[ResolveStep] = []
        topo = _derive_topology(model, self._hw(), hints, steps)
        assert topo.pp == 2

    def test_user_hints_cp(self):
        model = ModelProfile(param_billions=72.0, num_layers=80)
        hints = UserHints(cp=4)
        steps: list[ResolveStep] = []
        topo = _derive_topology(model, self._hw(), hints, steps)
        assert topo.cp == 4

    def test_moe_auto_ep_sglang(self):
        """sglang MoE: EP ≤ TP, TP % EP == 0."""
        model = ModelProfile(
            param_billions=236.0,
            num_layers=64,
            is_moe=True,
            num_experts=128,
        )
        steps: list[ResolveStep] = []
        topo = _derive_topology(model, self._hw(), None, steps, backend="sglang")
        assert topo.ep is not None
        assert topo.ep > 1
        assert topo.ep <= topo.tp
        assert topo.tp % topo.ep == 0

    def test_moe_ep_capped_at_tp_sglang(self):
        """30B MoE (tp=2) on sglang must get ep=2, not ep=8."""
        model = ModelProfile(
            param_billions=30.0,
            num_layers=48,
            is_moe=True,
            num_experts=128,
            bytes_per_param=2.0,
        )
        steps: list[ResolveStep] = []
        topo = _derive_topology(model, self._hw(), None, steps, backend="sglang")
        assert topo.tp == 2
        assert topo.dp == 4
        assert topo.ep == 2  # largest divisor of tp=2 ≤ 128

    def test_moe_auto_ep_vllm(self):
        """vLLM MoE: EP = TP × DP (auto-computed by vLLM)."""
        model = ModelProfile(
            param_billions=236.0,
            num_layers=64,
            is_moe=True,
            num_experts=128,
        )
        steps: list[ResolveStep] = []
        # 236B bf16 → tp=16, dp=1 on 16 GPUs → ep = tp*dp = 16
        hw = self._hw(gpu_count=16)
        topo = _derive_topology(model, hw, None, steps, backend="vllm")
        assert topo.ep == topo.tp * topo.dp

    def test_moe_vllm_skips_ep_when_not_divisible(self):
        """vLLM: skip EP when num_experts % (TP×DP) != 0."""
        model = ModelProfile(
            param_billions=72.0,
            num_layers=80,
            is_moe=True,
            num_experts=5,  # 5 not divisible by tp*dp
        )
        steps: list[ResolveStep] = []
        topo = _derive_topology(model, self._hw(), None, steps, backend="vllm")
        assert topo.ep is None

    def test_moe_auto_dpa_sglang_deepseek(self):
        """sglang DPA auto-enabled only for deepseek family (MLA models)."""
        model = ModelProfile(
            param_billions=236.0,
            num_layers=64,
            is_moe=True,
            num_experts=128,
            family="deepseek",
        )
        # 236B bf16 → tp=16, need 32 GPUs for dp=2
        hw = self._hw(gpu_count=32)
        steps: list[ResolveStep] = []
        topo = _derive_topology(model, hw, None, steps, backend="sglang")
        assert topo.tp == 16
        assert topo.dp == 2
        assert topo.enable_dp_attention is True

    def test_moe_no_auto_dpa_unsupported_family(self):
        """sglang DPA NOT auto-enabled for unsupported families (mixtral)."""
        model = ModelProfile(
            param_billions=72.0,
            num_layers=80,
            is_moe=True,
            num_experts=8,
            family="mixtral",
        )
        hw = self._hw(gpu_count=8)
        steps: list[ResolveStep] = []
        topo = _derive_topology(model, hw, None, steps, backend="sglang")
        assert topo.enable_dp_attention is False

    def test_moe_no_auto_dpa_qwen3(self):
        """qwen3 is DPA-capable but NOT auto-enabled."""
        model = ModelProfile(
            param_billions=72.0,
            num_layers=80,
            is_moe=True,
            num_experts=8,
            family="qwen3",
        )
        hw = self._hw(gpu_count=8)
        steps: list[ResolveStep] = []
        topo = _derive_topology(model, hw, None, steps, backend="sglang")
        assert topo.dp == 2
        assert topo.enable_dp_attention is False

    def test_moe_auto_dpa_deepseek_small_tp(self):
        """DeepSeek MoE with tp=2, dp=4: auto-DPA even though tp < dp.

        Previously tp(2) % dp(4) != 0 blocked auto-enable. But sglang's
        constraint is total_tp(8) % dp(4) == 0 which always holds.
        """
        model = ModelProfile(
            param_billions=30.0,
            num_layers=48,
            is_moe=True,
            num_experts=128,
            bytes_per_param=2.0,
            family="deepseek",
        )
        hw = self._hw(gpu_count=8)
        steps: list[ResolveStep] = []
        topo = _derive_topology(model, hw, None, steps, backend="sglang")
        assert topo.tp == 2
        assert topo.dp == 4
        assert topo.enable_dp_attention is True

    def test_moe_no_auto_dpa_unknown_family(self):
        """sglang DPA NOT auto-enabled when family is None."""
        model = ModelProfile(
            param_billions=72.0,
            num_layers=80,
            is_moe=True,
            num_experts=8,
        )
        hw = self._hw(gpu_count=8)
        steps: list[ResolveStep] = []
        topo = _derive_topology(model, hw, None, steps, backend="sglang")
        assert topo.enable_dp_attention is False

    def test_vllm_no_auto_dpa(self):
        """vLLM does not support DPA — should never auto-enable it."""
        model = ModelProfile(
            param_billions=72.0,
            num_layers=80,
            is_moe=True,
            num_experts=8,
            family="qwen3",  # even supported family → no DPA on vLLM
        )
        hw = self._hw(gpu_count=8)
        steps: list[ResolveStep] = []
        topo = _derive_topology(model, hw, None, steps, backend="vllm")
        assert topo.enable_dp_attention is False

    def test_user_dpa_override_true(self):
        """User can force DPA on."""
        model = ModelProfile(param_billions=72.0, num_layers=80)
        hints = UserHints(enable_dp_attention=True)
        steps: list[ResolveStep] = []
        topo = _derive_topology(model, self._hw(), hints, steps)
        assert topo.enable_dp_attention is True

    def test_user_dpa_override_false(self):
        """User can force DPA off even for MoE."""
        model = ModelProfile(
            param_billions=72.0,
            num_layers=80,
            is_moe=True,
            num_experts=8,
        )
        hints = UserHints(enable_dp_attention=False)
        steps: list[ResolveStep] = []
        topo = _derive_topology(model, self._hw(), hints, steps)
        assert topo.enable_dp_attention is False

    def test_user_ep_override(self):
        """User-specified EP should skip auto-derivation."""
        model = ModelProfile(
            param_billions=72.0,
            num_layers=80,
            is_moe=True,
            num_experts=128,
        )
        hints = UserHints(ep=4)
        steps: list[ResolveStep] = []
        topo = _derive_topology(model, self._hw(), hints, steps)
        assert topo.ep == 4

    def test_non_moe_no_auto_ep(self):
        """Non-MoE model should not auto-derive EP."""
        model = ModelProfile(param_billions=72.0, num_layers=80)
        steps: list[ResolveStep] = []
        topo = _derive_topology(model, self._hw(), None, steps)
        assert topo.ep is None

    def test_steps_record_sources(self):
        """Steps should record auto vs user_override sources."""
        model = ModelProfile(param_billions=72.0, num_layers=80)
        hints = UserHints(tp=8)
        steps: list[ResolveStep] = []
        _derive_topology(model, self._hw(), hints, steps)
        auto_steps = [s for s in steps if s.source == "auto"]
        override_steps = [s for s in steps if s.source == "user_override"]
        assert len(auto_steps) >= 1  # at least formula tp
        assert len(override_steps) >= 1  # user tp override


# ---------- resolve_simple ----------


class TestResolveSimple:
    """Tests for resolve_simple() — Phase 1 entry point."""

    def _hw(self, gpu_count: int = 8) -> HardwareEnv:
        return HardwareEnv(
            gpu_count=gpu_count,
            gpu_model="NVIDIA H100-SXM5-80GB",
            gpu_memory_mib=81920,
        )

    def test_generates_valid_plan(self):
        model = ModelProfile(param_billions=72.0, num_layers=80)
        result = resolve_simple(
            checkpoint="/models/Qwen3-72B",
            backend="sglang",
            model=model,
            hw=self._hw(),
        )
        plan = result.plan
        assert plan.checkpoint == "/models/Qwen3-72B"
        assert plan.backend == "sglang"
        assert len(plan.assignments) == 1
        assert plan.assignments[0].role == WellKnownRole.FULL
        assert plan.validate() == []

    def test_gpu_insufficient_raises(self):
        """If derived topology needs more GPUs than available → RuntimeError."""
        model = ModelProfile(param_billions=236.0, num_layers=64)
        with pytest.raises(RuntimeError, match="GPUs"):
            resolve_simple(
                checkpoint="/models/large-model",
                backend="vllm",
                model=model,
                hw=self._hw(gpu_count=2),
            )

    def test_steps_recorded(self):
        model = ModelProfile(param_billions=72.0, num_layers=80)
        result = resolve_simple(
            checkpoint="/models/Qwen3-72B",
            backend="sglang",
            model=model,
            hw=self._hw(),
        )
        assert len(result.steps) >= 1
        assert any(s.field == "tp" for s in result.steps)

    def test_recipe_params_passed_through(self):
        """recipe_params should appear as engine_params on the assignment."""
        model = ModelProfile(param_billions=72.0, num_layers=80)
        result = resolve_simple(
            checkpoint="/models/Qwen3-72B",
            backend="sglang",
            model=model,
            hw=self._hw(),
            recipe_params={"dtype": "bfloat16", "max_model_len": 4096},
        )
        ep = result.plan.assignments[0].engine_params
        assert ep["dtype"] == "bfloat16"
        assert ep["max_model_len"] == 4096

    def test_no_recipe_params(self):
        """Without recipe_params, engine_params should be empty."""
        model = ModelProfile(param_billions=8.0, num_layers=36)
        result = resolve_simple(
            checkpoint="/models/Qwen3-8B",
            backend="vllm",
            model=model,
            hw=self._hw(),
        )
        assert result.plan.assignments[0].engine_params == {}

    def test_with_user_hints(self):
        model = ModelProfile(param_billions=72.0, num_layers=80)
        hints = UserHints(tp=2, dp=1)
        result = resolve_simple(
            checkpoint="/models/Qwen3-72B",
            backend="sglang",
            model=model,
            hw=self._hw(),
            hints=hints,
        )
        topo = result.plan.assignments[0].topology
        assert topo.tp == 2
        assert topo.dp == 1


# ---------- _overrides_to_hints ----------


class TestOverridesToHints:
    """Tests for _overrides_to_hints() — flat dict → UserHints."""

    def test_basic_tp_dp_pp(self):
        hints = _overrides_to_hints({"tp": 4, "dp": 2, "pp": 1})
        assert hints.tp == 4
        assert hints.dp == 2
        assert hints.pp == 1

    def test_ep(self):
        hints = _overrides_to_hints({"ep": 8})
        assert hints.ep == 8

    def test_cp(self):
        hints = _overrides_to_hints({"cp": 4})
        assert hints.cp == 4

    def test_enable_dp_attention(self):
        hints = _overrides_to_hints({"enable_dp_attention": True})
        assert hints.enable_dp_attention is True

    def test_framework_aliases_tp_size(self):
        hints = _overrides_to_hints({"tp_size": 4})
        assert hints.tp == 4

    def test_framework_aliases_tensor_parallel_size(self):
        hints = _overrides_to_hints({"tensor_parallel_size": 8})
        assert hints.tp == 8

    def test_framework_aliases_dp_size(self):
        hints = _overrides_to_hints({"dp_size": 2})
        assert hints.dp == 2

    def test_framework_aliases_data_parallel_size(self):
        hints = _overrides_to_hints({"data_parallel_size": 4})
        assert hints.dp == 4

    def test_framework_aliases_pp_size(self):
        hints = _overrides_to_hints({"pp_size": 2})
        assert hints.pp == 2

    def test_framework_aliases_ep_size(self):
        hints = _overrides_to_hints({"ep_size": 16})
        assert hints.ep == 16

    def test_framework_aliases_attn_cp_size(self):
        hints = _overrides_to_hints({"attn_cp_size": 2})
        assert hints.cp == 2

    def test_unrecognized_keys_ignored(self):
        """Keys not recognized as parallel dimensions are ignored."""
        hints = _overrides_to_hints({"dtype": "bfloat16", "max_model_len": 4096})
        assert hints.tp is None
        assert hints.dp is None

    def test_falsy_values_not_skipped(self):
        """Falsy values (0, False) must not be silently dropped."""
        hints = _overrides_to_hints({"tp": 0, "dp": 0, "enable_dp_attention": False})
        assert hints.tp == 0
        assert hints.dp == 0
        assert hints.enable_dp_attention is False

    def test_first_alias_wins(self):
        """When both generic and framework-specific keys present, first wins."""
        hints = _overrides_to_hints({"tp": 2, "tp_size": 4})
        assert hints.tp == 2

    def test_empty_dict(self):
        hints = _overrides_to_hints({})
        assert hints.tp is None
        assert hints.dp is None
        assert hints.pp is None
        assert hints.ep is None
        assert hints.cp is None
        assert hints.enable_dp_attention is None

    def test_float_truncation_warns(self):
        """Float values like 2.5 should be truncated to int with a warning."""
        from loguru import logger

        warnings: list[str] = []
        sink_id = logger.add(
            lambda msg: warnings.append(str(msg)),
            level="WARNING",
            filter=lambda record: record["level"].name == "WARNING",
        )
        try:
            hints = _overrides_to_hints({"tp": 2.5})
        finally:
            logger.remove(sink_id)
        assert hints.tp == 2
        assert any("truncating" in w for w in warnings)

    def test_float_whole_number_no_warning(self):
        """Float values like 4.0 should convert silently (no truncation)."""
        from loguru import logger

        warnings: list[str] = []
        sink_id = logger.add(
            lambda msg: warnings.append(str(msg)),
            level="WARNING",
            filter=lambda record: record["level"].name == "WARNING",
        )
        try:
            hints = _overrides_to_hints({"tp": 4.0})
        finally:
            logger.remove(sink_id)
        assert hints.tp == 4
        assert not any("truncating" in w for w in warnings)


# ---------- DP re-derivation when user overrides only TP ----------


class TestDpRederivation:
    """When user overrides only TP, DP should be recomputed from the new TP."""

    def _hw(self, gpu_count: int = 8) -> HardwareEnv:
        return HardwareEnv(
            gpu_count=gpu_count,
            gpu_model="NVIDIA H100-SXM5-80GB",
            gpu_memory_mib=81920,
        )

    def test_tp_override_recomputes_dp(self):
        """Setting tp=2 on 8 GPUs should give dp=4 (not stale dp from formula)."""
        model = ModelProfile(param_billions=72.0, num_layers=80)
        hints = UserHints(tp=2)
        steps: list[ResolveStep] = []
        topo = _derive_topology(model, self._hw(), hints, steps)
        assert topo.tp == 2
        assert topo.dp == 4  # 8 // 2
        assert topo.tp * topo.dp <= 8

    def test_tp_override_with_dp_override_no_rederive(self):
        """When user sets both tp and dp, dp should not be re-derived."""
        model = ModelProfile(param_billions=72.0, num_layers=80)
        hints = UserHints(tp=2, dp=1)
        steps: list[ResolveStep] = []
        topo = _derive_topology(model, self._hw(), hints, steps)
        assert topo.tp == 2
        assert topo.dp == 1

    def test_single_gpu(self):
        """1 GPU should give tp=1, dp=1."""
        model = ModelProfile(param_billions=0.5, num_layers=24)
        steps: list[ResolveStep] = []
        topo = _derive_topology(model, self._hw(gpu_count=1), None, steps)
        assert topo.tp == 1
        assert topo.dp == 1


# ---------- precision_key ----------


class TestPrecisionKey:
    def _make_identity(self, dtype="bfloat16", quant=None):
        from sieval.infer.introspect import ModelIdentity

        return ModelIdentity(
            architecture="Qwen3ForCausalLM",
            family="qwen3",
            param_billions=4.0,
            dtype=dtype,
            quantization=quant,
        )

    def test_bf16(self):
        assert precision_key(self._make_identity("bfloat16")) == "bf16"

    def test_fp16(self):
        assert precision_key(self._make_identity("float16")) == "fp16"

    def test_fp32(self):
        assert precision_key(self._make_identity("float32")) == "fp32"

    def test_unknown_dtype_defaults_bf16(self):
        assert precision_key(self._make_identity("int8")) == "bf16"

    def test_quantized_fp8(self):
        from sieval.infer.introspect import QuantizationInfo

        quant = QuantizationInfo(quant_method="fp8", bits=8, raw_config={})
        identity = self._make_identity(quant=quant)
        assert precision_key(identity) == "fp8"

    def test_quantized_int4(self):
        from sieval.infer.introspect import QuantizationInfo

        quant = QuantizationInfo(quant_method="gptq", bits=4, raw_config={})
        identity = self._make_identity(quant=quant)
        assert precision_key(identity) == "int4"

    def test_quantized_mxfp4(self):
        """MXFP4 has a dedicated precision key matching the yaml recipe."""
        from sieval.infer.introspect import QuantizationInfo

        quant = QuantizationInfo(quant_method="mxfp4", bits=4, raw_config={})
        identity = self._make_identity(quant=quant)
        assert precision_key(identity) == "mxfp4"


# ---------- auto_resolve_plan ----------


class TestAutoResolvePlan:
    @pytest.mark.anyio
    async def test_basic_auto_resolve(self, tmp_path):
        """auto_resolve_plan with a valid checkpoint and mocked GPU."""
        import json
        from unittest.mock import AsyncMock, patch

        from sieval.infer.introspect import GPUInfo
        from sieval.infer.topology.resolver import auto_resolve_plan

        checkpoint = tmp_path / "Qwen3-4B"
        checkpoint.mkdir()
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
        (checkpoint / "config.json").write_text(json.dumps(config))

        gpu = GPUInfo(model="NVIDIA H100-SXM5-80GB", count=8, memory_mib=81920)
        with patch(
            "sieval.infer.topology.resolver.detect_local_gpu",
            new_callable=AsyncMock,
            return_value=gpu,
        ):
            result = await auto_resolve_plan(str(checkpoint), backend="sglang")

        assert result.plan.backend == "sglang"
        assert result.plan.checkpoint == str(checkpoint)
        assert result.plan.validate() == []

    @pytest.mark.anyio
    async def test_no_gpu_raises(self, tmp_path):
        """auto_resolve_plan with no GPU should raise RuntimeError."""
        import json
        from unittest.mock import AsyncMock, patch

        from sieval.infer.topology.resolver import auto_resolve_plan

        checkpoint = tmp_path / "Model"
        checkpoint.mkdir()
        config = {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": 2560,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "torch_dtype": "bfloat16",
        }
        (checkpoint / "config.json").write_text(json.dumps(config))

        with (
            patch(
                "sieval.infer.topology.resolver.detect_local_gpu",
                new_callable=AsyncMock,
                return_value=None,
            ),
            pytest.raises(RuntimeError, match="No GPU detected"),
        ):
            await auto_resolve_plan(str(checkpoint))

    @pytest.mark.anyio
    async def test_with_overrides(self, tmp_path):
        """auto_resolve_plan with user overrides."""
        import json
        from unittest.mock import AsyncMock, patch

        from sieval.infer.introspect import GPUInfo
        from sieval.infer.topology.resolver import auto_resolve_plan

        checkpoint = tmp_path / "Model"
        checkpoint.mkdir()
        config = {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": 2560,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "torch_dtype": "bfloat16",
        }
        (checkpoint / "config.json").write_text(json.dumps(config))

        gpu = GPUInfo(model="NVIDIA H100-SXM5-80GB", count=8, memory_mib=81920)
        with patch(
            "sieval.infer.topology.resolver.detect_local_gpu",
            new_callable=AsyncMock,
            return_value=gpu,
        ):
            result = await auto_resolve_plan(
                str(checkpoint),
                backend="sglang",
                overrides={"tp": 2, "dtype": "float16"},
            )

        assert result.plan.assignments[0].topology.tp == 2
        # Non-topology overrides must land in engine_params (not be dropped)
        assert result.plan.assignments[0].engine_params.get("dtype") == "float16"

    @pytest.mark.anyio
    async def test_dash_form_overrides_normalized_at_entry(self, tmp_path):
        """Dash-form overrides normalize before the TOPO_KEYS filter and
        _overrides_to_hints see them.
        """
        import json
        from unittest.mock import AsyncMock, patch

        from sieval.infer.introspect import GPUInfo
        from sieval.infer.topology.resolver import auto_resolve_plan

        checkpoint = tmp_path / "Model"
        checkpoint.mkdir()
        config = {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": 2560,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "torch_dtype": "bfloat16",
        }
        (checkpoint / "config.json").write_text(json.dumps(config))

        gpu = GPUInfo(model="NVIDIA H100-SXM5-80GB", count=8, memory_mib=81920)
        with patch(
            "sieval.infer.topology.resolver.detect_local_gpu",
            new_callable=AsyncMock,
            return_value=gpu,
        ):
            result = await auto_resolve_plan(
                str(checkpoint),
                backend="vllm",
                overrides={"tensor-parallel-size": 2, "max-model-len": 4096},
            )

        assert result.plan.assignments[0].topology.tp == 2
        engine_params = result.plan.assignments[0].engine_params
        assert engine_params.get("max_model_len") == 4096
        assert "max-model-len" not in engine_params
        assert "tensor-parallel-size" not in engine_params
