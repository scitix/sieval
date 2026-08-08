"""Tests for sieval.infer.recipes.registry — recipe loading and layer resolution.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import io

import pytest
from loguru import logger

from sieval.infer.recipes import (
    Recipe,
    capability_model_type,
    check_tested_versions,
    list_recipes,
    load_recipe,
    resolve_capability_profile,
    resolve_hardware_profile,
)
from sieval.infer.recipes.registry import _parse_recipe


class TestListRecipes:
    def test_list_recipes(self) -> None:
        """Verify qwen3 recipes show up in the recipe list."""
        recipes = list_recipes()
        assert "qwen3-0.6b" in recipes
        assert "qwen3-8b" in recipes


class TestLoadRecipe:
    def test_load_recipe(self) -> None:
        """Verify loading qwen3-8b returns the expected two-layer structure."""
        recipe = load_recipe("qwen3-8b")
        assert isinstance(recipe, Recipe)
        assert recipe.known_issues == []

        # Hardware tiers present
        assert "H100-80G" in recipe.hardware
        assert "H200-141G" in recipe.hardware

        # H100-80G / bf16 / vllm has correct params
        h100_bf16_vllm = recipe.hardware["H100-80G"]["bf16"]["vllm"]
        assert h100_bf16_vllm["dtype"] == "bfloat16"
        assert h100_bf16_vllm["gpu_memory_utilization"] == 0.95
        assert h100_bf16_vllm["max_model_len"] == 32768

    def test_load_recipe_qwen3_06b(self) -> None:
        """Verify loading qwen3-0.6b returns expected structure."""
        recipe = load_recipe("qwen3-0.6b")
        assert isinstance(recipe, Recipe)
        assert recipe.size_range == (0.3, 1.0)
        assert "H100-80G" in recipe.hardware
        assert "H200-141G" in recipe.hardware

    def test_load_recipe_not_found(self) -> None:
        """Verify LookupError for nonexistent recipe."""
        with pytest.raises(LookupError, match="nonexistent-model"):
            load_recipe("nonexistent-model")

    def test_load_recipe_not_found_is_not_a_key_error(self) -> None:
        """The message is a sentence for the user, and `KeyError.__str__` is
        `repr(args[0])` — raising one here would reach the CLI double-quoted.
        `KeyError` subclasses `LookupError`, so the assertion above alone
        would not catch a regression to `KeyError`.
        """
        with pytest.raises(LookupError) as exc_info:
            load_recipe("nonexistent-model")

        assert not isinstance(exc_info.value, KeyError)
        assert str(exc_info.value).startswith("Recipe 'nonexistent-model' not found")

    def test_load_recipe_underscore_prefix_rejected(self) -> None:
        """Verify ValueError for underscore-prefixed recipe names (metadata keys).

        A reserved name is malformed input, not a lookup miss.
        """
        with pytest.raises(ValueError, match="must not start with '_'"):
            load_recipe("_metadata")

    def test_load_recipe_qwen3_235b_a22b_has_fp8_profile(self) -> None:
        """Verify qwen3-235b-a22b H100 has fp8 with correct max_model_len."""
        recipe = load_recipe("qwen3-235b-a22b")
        fp8_vllm = recipe.hardware["H100-80G"]["fp8"]["vllm"]
        assert fp8_vllm["max_model_len"] == 32768

    def test_hardware_overrides_removed(self) -> None:
        """Verify old fields don't exist on loaded recipe objects."""
        recipe = load_recipe("qwen3-8b")
        assert not hasattr(recipe, "frameworks")
        assert not hasattr(recipe, "hardware_overrides")
        assert not hasattr(recipe, "precision_overrides")


class TestParseRecipeLayers:
    """Tests for _parse_recipe with the hardware + capabilities structure."""

    def test_parse_hardware_structure(self) -> None:
        """Verify 4-level nested dict parsed correctly."""
        raw = {
            "hardware": {
                "H100-80G": {
                    "bf16": {
                        "vllm": {
                            "dtype": "bfloat16",
                            "gpu_memory_utilization": 0.95,
                            "max_model_len": 32768,
                        },
                        "sglang": {
                            "mem_fraction_static": 0.9,
                        },
                    },
                },
                "A100-80G": {
                    "bf16": {
                        "vllm": {
                            "dtype": "bfloat16",
                            "gpu_memory_utilization": 0.9,
                        },
                    },
                    "fp8": {
                        "vllm": {
                            "dtype": "fp8",
                            "max_model_len": 65536,
                        },
                    },
                },
            },
        }
        recipe = _parse_recipe("test-model", raw)
        assert "H100-80G" in recipe.hardware
        assert "A100-80G" in recipe.hardware
        assert "bf16" in recipe.hardware["H100-80G"]
        assert "vllm" in recipe.hardware["H100-80G"]["bf16"]
        assert recipe.hardware["H100-80G"]["bf16"]["vllm"]["dtype"] == "bfloat16"
        assert (
            recipe.hardware["H100-80G"]["bf16"]["vllm"]["gpu_memory_utilization"]
            == 0.95
        )
        assert recipe.hardware["A100-80G"]["fp8"]["vllm"]["dtype"] == "fp8"

    def test_parse_hardware_empty(self) -> None:
        """No hardware layer → empty dict."""
        raw: dict[str, object] = {"size_range": [6, 12]}
        recipe = _parse_recipe("test-8b", raw)
        assert recipe.hardware == {}

    def test_parse_preserves_other_fields(self) -> None:
        """known_issues and tested_versions survive alongside hardware."""
        raw = {
            "hardware": {
                "H100-80G": {
                    "bf16": {"vllm": {"dtype": "bfloat16"}},
                },
            },
            "known_issues": ["Some known issue"],
            "tested_versions": {"vllm": [">=0.8.0"]},
            "size_range": [6, 12],
        }
        recipe = _parse_recipe("test-8b", raw)
        assert recipe.known_issues == ["Some known issue"]
        assert recipe.tested_versions == {"vllm": [">=0.8.0"]}
        assert recipe.size_range == (6.0, 12.0)
        assert "H100-80G" in recipe.hardware

    def test_old_fields_ignored(self) -> None:
        """Old frameworks/hw_overrides/precision_overrides silently ignored."""
        raw = {
            "frameworks": {"vllm": {"dtype": "bfloat16"}},
            "hardware_overrides": {
                "A100-80G": {"vllm": {"gpu_memory_utilization": 0.9}},
            },
            "precision_overrides": {
                "fp8": {"A100-80G": {"vllm": {"max_model_len": 65536}}},
            },
            "hardware": {
                "H100-80G": {
                    "bf16": {"vllm": {"dtype": "bfloat16"}},
                },
            },
        }
        recipe = _parse_recipe("test-old", raw)
        # Old fields should not appear on Recipe
        assert not hasattr(recipe, "frameworks")
        assert not hasattr(recipe, "hardware_overrides")
        assert not hasattr(recipe, "precision_overrides")
        # The hardware layer should be parsed normally
        assert "H100-80G" in recipe.hardware


class TestResolveHardwareProfile:
    """Tests for resolve_hardware_profile."""

    @pytest.fixture
    def sample_recipe(self) -> Recipe:
        return Recipe(
            name="test-model",
            hardware={
                "H100-80G": {
                    "bf16": {
                        "vllm": {
                            "dtype": "bfloat16",
                            "gpu_memory_utilization": 0.95,
                            "max_model_len": 32768,
                        },
                        "sglang": {
                            "mem_fraction_static": 0.9,
                        },
                    },
                },
                "A100-80G": {
                    "bf16": {
                        "vllm": {
                            "dtype": "bfloat16",
                            "gpu_memory_utilization": 0.9,
                        },
                    },
                },
            },
        )

    def test_exact_match(self, sample_recipe: Recipe) -> None:
        """hw+prec+fw all match → correct params."""
        result = resolve_hardware_profile(
            sample_recipe, "NVIDIA H100-SXM5-80GB", "bf16", "vllm"
        )
        assert result is not None
        assert result["dtype"] == "bfloat16"
        assert result["gpu_memory_utilization"] == 0.95
        assert result["max_model_len"] == 32768

    def test_fuzzy_gpu_match(self, sample_recipe: Recipe) -> None:
        """'NVIDIA H100-SXM5-80GB' matches profile key 'H100-80G'."""
        result = resolve_hardware_profile(
            sample_recipe, "NVIDIA H100-SXM5-80GB", "bf16", "sglang"
        )
        assert result is not None
        assert result["mem_fraction_static"] == 0.9

    def test_precision_not_found_returns_none(self, sample_recipe: Recipe) -> None:
        """fp8 on H100 (only bf16 defined) → None."""
        result = resolve_hardware_profile(
            sample_recipe, "NVIDIA H100-SXM5-80GB", "fp8", "vllm"
        )
        assert result is None

    def test_gpu_not_found_returns_none(self, sample_recipe: Recipe) -> None:
        """Unknown GPU → None."""
        result = resolve_hardware_profile(
            sample_recipe, "NVIDIA V100-SXM2-32GB", "bf16", "vllm"
        )
        assert result is None

    def test_gpu_none_returns_none(self, sample_recipe: Recipe) -> None:
        """gpu_model=None → None."""
        result = resolve_hardware_profile(sample_recipe, None, "bf16", "vllm")
        assert result is None

    def test_precision_none_defaults_to_bf16(self, sample_recipe: Recipe) -> None:
        """precision=None → uses 'bf16'."""
        result = resolve_hardware_profile(
            sample_recipe, "NVIDIA H100-SXM5-80GB", None, "vllm"
        )
        assert result is not None
        assert result["dtype"] == "bfloat16"

    def test_framework_not_in_profile_returns_none(self, sample_recipe: Recipe) -> None:
        """'tensorrt' not in any profile → None."""
        result = resolve_hardware_profile(
            sample_recipe, "NVIDIA H100-SXM5-80GB", "bf16", "tensorrt"
        )
        assert result is None

    def test_empty_hardware_returns_none(self) -> None:
        """Empty hardware layer → None."""
        recipe = Recipe(name="empty", hardware={})
        result = resolve_hardware_profile(
            recipe, "NVIDIA H100-SXM5-80GB", "bf16", "vllm"
        )
        assert result is None


class TestTestedVersions:
    def test_recipe_has_tested_versions(self) -> None:
        """Verify qwen3-4b has tested_versions for both frameworks."""
        recipe = load_recipe("qwen3-4b")
        assert "vllm" in recipe.tested_versions
        assert isinstance(recipe.tested_versions["vllm"], list)
        assert "sglang" in recipe.tested_versions
        assert isinstance(recipe.tested_versions["sglang"], list)

    def test_recipe_tested_versions_are_specifiers(self) -> None:
        """Verify tested_versions values are valid PEP 440 specifiers."""
        recipe = load_recipe("qwen3-4b")
        for fw, specs in recipe.tested_versions.items():
            for spec in specs:
                assert spec.startswith((">=", "<=", "==", "!=", "~=", ">", "<")), (
                    f"Expected PEP 440 specifier for {fw}, got {spec!r}"
                )

    def test_all_recipes_have_tested_versions(self) -> None:
        """Verify all qwen3 recipes have tested_versions."""
        for name in [
            "qwen3-0.6b",
            "qwen3-1.7b",
            "qwen3-4b",
            "qwen3-8b",
            "qwen3-14b",
            "qwen3-30b-a3b",
            "qwen3-32b",
            "qwen3-72b",
            "qwen3-235b-a22b",
        ]:
            recipe = load_recipe(name)
            assert recipe.tested_versions, f"Recipe {name} has no tested_versions"

    def test_tested_versions_not_in_profiles(self) -> None:
        """tested_versions must NOT leak into profile params.

        _coerce_param would mangle lists into strings.
        """
        recipe = load_recipe("qwen3-4b")
        for _hw_key, prec_map in recipe.hardware.items():
            for _prec_key, fw_map in prec_map.items():
                for fw_params in fw_map.values():
                    assert "tested_versions" not in fw_params


class TestCheckTestedVersions:
    def test_version_satisfies(self) -> None:
        """Version within the tested range returns True."""
        assert check_tested_versions("vllm", "0.8.3", [">=0.8.0"]) is True

    def test_version_does_not_satisfy(self) -> None:
        """Version below the tested range returns False."""
        result = check_tested_versions("vllm", "0.1.0", [">=0.8.0"])
        assert result is False

    def test_single_item_with_and_specifiers(self) -> None:
        """A single item with comma-separated specifiers uses AND semantics."""
        assert check_tested_versions("sglang", "0.5.0", [">=0.4.0,<1.0"]) is True
        assert check_tested_versions("sglang", "1.0.0", [">=0.4.0,<1.0"]) is False

    def test_multiple_items_use_or_semantics(self) -> None:
        """Multiple list items use OR semantics — matching any one is enough."""
        # Official release matches first item
        assert (
            check_tested_versions("sglang", "0.5.0", [">=0.4.6.post1", "==0.0.0.dev1"])
            is True
        )
        # Dev build matches second item
        specs = [">=0.4.6.post1", "==0.0.0.dev1"]
        assert check_tested_versions("sglang", "0.0.0.dev1", specs) is True
        # Neither matches
        assert (
            check_tested_versions("sglang", "0.3.0", [">=0.4.6.post1", "==0.0.0.dev1"])
            is False
        )

    def test_dev_version_with_local_segment(self) -> None:
        """PEP 440 local segments are ignored by ==, so dev+local matches ==dev."""
        assert (
            check_tested_versions(
                "sglang",
                "0.0.0.dev1+gdce8b0606.d20260307",
                [">=0.4.6.post1", "==0.0.0.dev1"],
            )
            is True
        )

    def test_invalid_version_string(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Unparseable version string returns False and logs warning."""
        result = check_tested_versions("vllm", "not-a-version", [">=0.8.0"])
        assert result is False

    def test_invalid_specifier(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Invalid specifier returns False and logs warning."""
        result = check_tested_versions("vllm", "0.8.0", [">>>invalid"])
        assert result is False

    def test_invalid_specifier_skipped_others_still_match(self) -> None:
        """An invalid specifier is skipped; remaining specifiers are still checked."""
        assert check_tested_versions("vllm", "0.8.3", [">>>invalid", ">=0.8.0"]) is True


def _capture_logs(fn) -> str:
    """Capture loguru output during a call."""
    sink = io.StringIO()
    logger_id = logger.add(sink, format="{message}", level="WARNING")
    try:
        fn()
    finally:
        logger.remove(logger_id)
    return sink.getvalue()


_MARKER = "TEST-ISSUE-MARKER-1"


@pytest.fixture
def registry_with_issue(tmp_path, monkeypatch):
    """Patch _RECIPE_DIR to a tmp dir containing one recipe with known_issues."""
    import yaml as _yaml

    from sieval.infer.recipes import registry as reg_mod

    (tmp_path / "zz_test.yaml").write_text(
        _yaml.safe_dump(
            {
                "_family": "zz-test",
                "zz-test-1b": {
                    "size_range": [1.0, 2.0],
                    "known_issues": [_MARKER],
                    "hardware": {
                        "H100-80G": {
                            "bf16": {
                                "vllm": {"dtype": "bfloat16", "max_model_len": 4096},
                            },
                        },
                    },
                },
            }
        )
    )
    monkeypatch.setattr(reg_mod, "_RECIPE_DIR", tmp_path)
    return reg_mod


class TestKnownIssuesEmission:
    """Recipe.known_issues surfaces as logger.warning on lookup, not on iteration."""

    def test_match_recipe_emits(self, registry_with_issue) -> None:
        output = _capture_logs(lambda: registry_with_issue.match_recipe("zz-test", 1.5))
        assert _MARKER in output
        assert "known issue" in output.lower()

    def test_load_recipe_emits(self, registry_with_issue) -> None:
        output = _capture_logs(lambda: registry_with_issue.load_recipe("zz-test-1b"))
        assert _MARKER in output

    def test_match_recipe_miss_does_not_emit(self, registry_with_issue) -> None:
        # 99B is outside [1.0, 2.0) → no match
        reg = registry_with_issue
        output = _capture_logs(lambda: reg.match_recipe("zz-test", 99.0))
        assert _MARKER not in output

    def test_load_family_recipes_does_not_emit(self, registry_with_issue) -> None:
        reg = registry_with_issue
        output = _capture_logs(lambda: reg.load_family_recipes("zz-test"))
        assert _MARKER not in output

    def test_list_recipes_does_not_emit(self, registry_with_issue) -> None:
        output = _capture_logs(lambda: registry_with_issue.list_recipes())
        assert _MARKER not in output


class TestLegacyProfilesRejected:
    """The pre-split ``profiles`` key must fail loudly, not be read silently."""

    def test_profiles_key_rejected(self) -> None:
        raw = {
            "size_range": [6, 12],
            "profiles": {"H100-80G": {"bf16": {"vllm": {"dtype": "bfloat16"}}}},
        }
        with pytest.raises(ValueError, match="removed 'profiles' key"):
            _parse_recipe("test-8b", raw)

    def test_error_names_both_replacement_layers(self) -> None:
        """The migration hint must say where the two halves go."""
        with pytest.raises(ValueError) as exc:
            _parse_recipe("test-8b", {"profiles": {}})
        assert "hardware" in str(exc.value)
        assert "capabilities" in str(exc.value)

    def test_unknown_capability_model_type_rejected(self) -> None:
        raw = {"capabilities": {"chat": {"vllm": {"tool_call_parser": "hermes"}}}}
        with pytest.raises(ValueError, match="unknown capability model type"):
            _parse_recipe("test-8b", raw)


class TestCapabilityModelType:
    """Eval-config ``type`` → recipe capability key."""

    def test_gen_maps_to_base(self) -> None:
        assert capability_model_type("gen") == "base"

    def test_chat_maps_to_instruct(self) -> None:
        assert capability_model_type("chat") == "instruct"

    def test_none_defaults_to_instruct(self) -> None:
        """Matches the eval config's own default for an undeclared type."""
        assert capability_model_type(None) == "instruct"

    def test_recipe_vocabulary_is_rejected(self) -> None:
        """`type: base` must not silently select the *instruct* layer.

        The config vocabulary is chat/gen and the recipe's is instruct/base, so
        writing the recipe's word into a model config is the likely mistake —
        and defaulting it would pick the opposite layer of the one intended.
        """
        for bad in ("base", "instruct"):
            with pytest.raises(ValueError, match="expected 'chat' or 'gen'"):
                capability_model_type(bad)

    def test_unknown_type_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown model type"):
            capability_model_type("cht")


class TestResolveCapabilityProfile:
    @pytest.fixture
    def sample_recipe(self) -> Recipe:
        return _parse_recipe(
            "test-model",
            {
                "hardware": {"H100-80G": {"bf16": {"vllm": {"dtype": "bfloat16"}}}},
                "capabilities": {
                    "instruct": {
                        "vllm": {
                            "enable_auto_tool_choice": True,
                            "tool_call_parser": "hermes",
                        },
                        "sglang": {"tool_call_parser": "qwen"},
                    },
                    "base": {"vllm": {}, "sglang": {}},
                },
            },
        )

    def test_instruct_returns_parsers(self, sample_recipe: Recipe) -> None:
        assert resolve_capability_profile(sample_recipe, "instruct", "vllm") == {
            "enable_auto_tool_choice": True,
            "tool_call_parser": "hermes",
        }

    def test_base_returns_empty(self, sample_recipe: Recipe) -> None:
        assert resolve_capability_profile(sample_recipe, "base", "vllm") == {}

    def test_unknown_framework_returns_empty(self, sample_recipe: Recipe) -> None:
        assert resolve_capability_profile(sample_recipe, "instruct", "tensorrt") == {}

    def test_valid_but_undeclared_capability_returns_empty(
        self, sample_recipe: Recipe
    ) -> None:
        """A recipe may omit a layer entirely; that is not a caller error."""
        recipe = _parse_recipe(
            "instruct-only",
            {"capabilities": {"instruct": {"vllm": {"tool_call_parser": "hermes"}}}},
        )
        assert resolve_capability_profile(recipe, "base", "vllm") == {}

    def test_unknown_capability_rejected(self, sample_recipe: Recipe) -> None:
        """A key outside CAPABILITY_MODEL_TYPES is a caller bug, not a base model.

        Returning ``{}`` would make every typo — and every config-vocabulary
        word reaching the recipe layer by mistake — resolve to *no* capability
        params, i.e. silently to base-like serving. `auto_resolve_plan` takes
        this key as a public kwarg, so the mistake is reachable from outside.
        """
        for bad in ("nonexistent", "chat", "gen", ""):
            with pytest.raises(ValueError, match="Unknown recipe capability"):
                resolve_capability_profile(sample_recipe, bad, "vllm")

    def test_returns_a_copy(self, sample_recipe: Recipe) -> None:
        """Mutating the result must not corrupt the loaded recipe."""
        result = resolve_capability_profile(sample_recipe, "instruct", "vllm")
        result["tool_call_parser"] = "mutated"
        assert sample_recipe.capabilities["instruct"]["vllm"]["tool_call_parser"] == (
            "hermes"
        )


class TestShippedRecipeCapabilities:
    """The split must actually withhold instruct behavior from base models."""

    def test_instruct_gets_tool_call_parser(self) -> None:
        recipe = load_recipe("qwen2.5-72b")
        assert resolve_capability_profile(recipe, "instruct", "vllm") == {
            "enable_auto_tool_choice": True,
            "tool_call_parser": "hermes",
        }

    def test_base_gets_nothing(self) -> None:
        """Qwen2.5-72B-Base is the ARC / HellaSwag ppl comparison target."""
        recipe = load_recipe("qwen2.5-72b")
        assert resolve_capability_profile(recipe, "base", "vllm") == {}
        assert resolve_capability_profile(recipe, "base", "sglang") == {}

    def test_hardware_layer_carries_no_capability_params(self) -> None:
        """Parsers must not be left behind in any shipped hardware leaf."""
        capability_keys = {
            "tool_call_parser",
            "enable_auto_tool_choice",
            "reasoning_parser",
        }
        for name in list_recipes():
            recipe = load_recipe(name)
            for hw_key, prec_map in recipe.hardware.items():
                for precision, fw_map in prec_map.items():
                    for framework, params in fw_map.items():
                        leaked = capability_keys & set(params)
                        assert not leaked, (
                            f"{name} {hw_key}/{precision}/{framework} "
                            f"still carries {leaked}"
                        )

    def test_every_shipped_recipe_declares_both_model_types(self) -> None:
        """A missing `base` key would silently fall back to no capabilities,
        making an authoring omission indistinguishable from an intentional one."""
        for name in list_recipes():
            recipe = load_recipe(name)
            assert set(recipe.capabilities) == {"instruct", "base"}, name
