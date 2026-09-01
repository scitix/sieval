"""Tests for the A100 memory tiers declared in the qwen recipe files.

The A100-80G entries are *derived* from H100-80G, not measured: both cards hold
80GB, and capacity is what bounds ``max_model_len`` across these entries, so the
bf16 block is copied. That reasoning lives in the "Hardware notes" header of
each recipe file; these tests keep the copy from drifting out from under it.

AI-Generated Code - Claude Opus 5 (1M context) (Anthropic)
"""

from sieval.infer.recipes import list_recipes, load_recipe, resolve_hardware_profile

_A100_TIERS = frozenset({"A100-40G", "A100-80G"})


def _recipes_with_a100_80g() -> list[str]:
    return [n for n in list_recipes() if "A100-80G" in load_recipe(n).hardware]


class TestA100TierDerivation:
    def test_some_recipe_declares_the_tier(self) -> None:
        """Guard the sweeps below from passing on an empty set."""
        assert _recipes_with_a100_80g()

    def test_a100_80g_bf16_mirrors_h100_80g(self) -> None:
        """Equal capacity, equal bf16 block.

        If a real A100-80G measurement ever replaces the derivation, update the
        recipe files' header notes and this test together.
        """
        for name in _recipes_with_a100_80g():
            hardware = load_recipe(name).hardware
            assert hardware["A100-80G"]["bf16"] == hardware["H100-80G"]["bf16"], name

    def test_a100_80g_declares_no_fp8(self) -> None:
        """Ampere has no native FP8 — the same reason A100-40G declares none."""
        for name in _recipes_with_a100_80g():
            assert list(load_recipe(name).hardware["A100-80G"]) == ["bf16"], name

    def test_both_tiers_ship_wherever_either_does(self) -> None:
        """A recipe declaring one A100 tier declares both, or neither."""
        for name in list_recipes():
            tiers = _A100_TIERS & set(load_recipe(name).hardware)
            assert tiers in (frozenset(), _A100_TIERS), (name, tiers)


class TestA100TierResolution:
    def test_80gb_card_gets_the_80g_tier(self) -> None:
        """An 80GB A100 gets the 80G tier — not the 40G one, and not nothing.

        Before A100-80G shipped, the only A100 key was 40G, so these strings
        state a capacity no key agreed with and resolved to no params at all.
        """
        recipe = load_recipe("qwen3-30b-a3b")
        for gpu in ("NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"):
            result = resolve_hardware_profile(recipe, gpu, "bf16", "vllm")
            assert result == recipe.hardware["A100-80G"]["bf16"]["vllm"], gpu
            assert result != recipe.hardware["A100-40G"]["bf16"]["vllm"], gpu

    def test_40gb_card_still_gets_the_40g_tier(self) -> None:
        """The tier that already worked must not move."""
        recipe = load_recipe("qwen3-30b-a3b")
        result = resolve_hardware_profile(
            recipe, "NVIDIA A100-SXM4-40GB", "bf16", "vllm"
        )
        assert result == recipe.hardware["A100-40G"]["bf16"]["vllm"]

    def test_bare_a100_is_ambiguous_now_both_tiers_ship(self) -> None:
        """A bare 'NVIDIA A100' names no capacity, so the tier is a coin flip.

        This is the ambiguity guard costing a match that used to resolve: with
        only A100-40G declared, it was the sole candidate. Refusing is the
        intended trade — nvidia-smi reports the capacity for this card.
        """
        recipe = load_recipe("qwen3-30b-a3b")
        assert resolve_hardware_profile(recipe, "NVIDIA A100", "bf16", "vllm") is None

    def test_fp8_checkpoint_on_a100_80g_resolves_to_nothing(self) -> None:
        """No fp8 block means no params, rather than bf16 params for an fp8 run."""
        recipe = load_recipe("qwen3-30b-a3b")
        result = resolve_hardware_profile(
            recipe, "NVIDIA A100-SXM4-80GB", "fp8", "vllm"
        )
        assert result is None
