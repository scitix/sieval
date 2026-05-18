"""Tests for the gpt-oss family recipe (sieval/infer/recipes/gpt_oss.yaml).

AI-Generated Code - Claude Opus 4.7 (Anthropic)
"""

from sieval.infer.recipes import Recipe, load_recipe, resolve_profile
from sieval.infer.recipes.registry import load_family_recipes


class TestGptOssRecipeShape:
    def test_family_loads_two_buckets(self) -> None:
        names = {r.name for r in load_family_recipes("gpt-oss")}
        assert names == {"gpt-oss-20b", "gpt-oss-120b"}

    def test_20b_size_range(self) -> None:
        recipe = load_recipe("gpt-oss-20b")
        assert isinstance(recipe, Recipe)
        assert recipe.size_range == (15.0, 30.0)

    def test_120b_size_range(self) -> None:
        assert load_recipe("gpt-oss-120b").size_range == (100.0, 150.0)

    def test_20b_has_both_hardware_tiers(self) -> None:
        recipe = load_recipe("gpt-oss-20b")
        assert "H100-80G" in recipe.profiles
        assert "H200-141G" in recipe.profiles

    def test_120b_has_both_hardware_tiers(self) -> None:
        recipe = load_recipe("gpt-oss-120b")
        assert "H100-80G" in recipe.profiles
        assert "H200-141G" in recipe.profiles

    def test_only_mxfp4_precision(self) -> None:
        h100 = load_recipe("gpt-oss-120b").profiles["H100-80G"]
        assert "mxfp4" in h100
        assert "bf16" not in h100
        assert "fp8" not in h100

    def test_both_frameworks_per_profile(self) -> None:
        h100_mxfp4 = load_recipe("gpt-oss-120b").profiles["H100-80G"]["mxfp4"]
        assert "vllm" in h100_mxfp4
        assert "sglang" in h100_mxfp4

    def test_tested_versions_present(self) -> None:
        tv = load_recipe("gpt-oss-120b").tested_versions
        assert "vllm" in tv
        assert "sglang" in tv

    def test_120b_known_issues(self) -> None:
        recipe = load_recipe("gpt-oss-120b")
        assert len(recipe.known_issues) >= 1
        joined = " ".join(recipe.known_issues).lower()
        assert "deterministic" in joined

    def test_20b_known_issues(self) -> None:
        recipe = load_recipe("gpt-oss-20b")
        assert len(recipe.known_issues) >= 1
        assert "deterministic" in " ".join(recipe.known_issues).lower()


class TestGptOss120bProfiles:
    def test_h100_vllm_profile(self) -> None:
        recipe = load_recipe("gpt-oss-120b")
        params = resolve_profile(recipe, "NVIDIA H100-SXM5-80GB", "mxfp4", "vllm")
        assert params is not None
        assert params["max_model_len"] == 131072
        assert params["gpu_memory_utilization"] == 0.85
        assert params["reasoning_parser"] == "openai"
        assert params["tool_call_parser"] == "openai"
        assert params["enable_auto_tool_choice"] is True
        assert params["max_num_batched_tokens"] == 8192
        assert params["max_cudagraph_capture_size"] == 2048
        assert params["stream_interval"] == 20
        assert params["no_enable_prefix_caching"] is True

    def test_h100_sglang_profile(self) -> None:
        recipe = load_recipe("gpt-oss-120b")
        params = resolve_profile(recipe, "NVIDIA H100-SXM5-80GB", "mxfp4", "sglang")
        assert params is not None
        assert params["context_length"] == 131072
        assert params["mem_fraction_static"] == 0.82
        assert params["reasoning_parser"] == "gpt-oss"
        assert params["tool_call_parser"] == "gpt-oss"

    def test_h200_vllm_profile(self) -> None:
        recipe = load_recipe("gpt-oss-120b")
        params = resolve_profile(recipe, "NVIDIA H200-141GB", "mxfp4", "vllm")
        assert params is not None
        assert params["gpu_memory_utilization"] == 0.90
        assert params["max_num_batched_tokens"] == 8192
        assert params["stream_interval"] == 20
        assert params["no_enable_prefix_caching"] is True

    def test_h200_sglang_profile(self) -> None:
        recipe = load_recipe("gpt-oss-120b")
        params = resolve_profile(recipe, "NVIDIA H200-141GB", "mxfp4", "sglang")
        assert params is not None
        assert params["mem_fraction_static"] == 0.85


class TestGptOss20bProfiles:
    def test_h100_vllm_profile(self) -> None:
        recipe = load_recipe("gpt-oss-20b")
        params = resolve_profile(recipe, "NVIDIA H100-SXM5-80GB", "mxfp4", "vllm")
        assert params is not None
        assert params["max_model_len"] == 131072
        assert params["gpu_memory_utilization"] == 0.90
        assert params["reasoning_parser"] == "openai"

    def test_h100_sglang_profile(self) -> None:
        recipe = load_recipe("gpt-oss-20b")
        params = resolve_profile(recipe, "NVIDIA H100-SXM5-80GB", "mxfp4", "sglang")
        assert params is not None
        assert params["mem_fraction_static"] == 0.85


class TestGptOssNegativeLookups:
    def test_bf16_not_defined(self) -> None:
        recipe = load_recipe("gpt-oss-120b")
        assert resolve_profile(recipe, "NVIDIA H100-SXM5-80GB", "bf16", "vllm") is None

    def test_a100_not_in_scope(self) -> None:
        recipe = load_recipe("gpt-oss-120b")
        assert resolve_profile(recipe, "NVIDIA A100-SXM4-80GB", "mxfp4", "vllm") is None

    def test_b200_not_in_scope(self) -> None:
        recipe = load_recipe("gpt-oss-120b")
        assert resolve_profile(recipe, "NVIDIA B200", "mxfp4", "vllm") is None
