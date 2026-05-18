"""Tests for YAML-mode recipe auto-resolution in sieval.cli.infer.

Covers the decision matrix:
  - explicit recipe → load normally
  - no recipe + checkpoint introspectable + recipe matched
    → auto-select + WARNING
  - no recipe + checkpoint introspectable + no match + overrides
    → overrides only + WARNING
  - no recipe + checkpoint introspectable + no match + no overrides → error
  - no recipe + checkpoint not introspectable + overrides
    → overrides only + WARNING
  - no recipe + checkpoint not introspectable + no overrides → error
  - no recipe + no checkpoint + overrides → overrides only
  - no recipe + no checkpoint + no overrides → error

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from loguru import logger

from sieval.cli.infer.recipe import _resolve_recipe_params, resolve_infer_config
from sieval.infer.introspect import GPUInfo, ModelIdentity
from sieval.infer.recipes.registry import Recipe
from sieval.infer.topology.models import DeploymentPlan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Mock GPU used for profile matching in tests.
# "H100-80G" tokens must appear in the model string for fuzzy matching.
_MOCK_GPU = GPUInfo(model="NVIDIA H100-SXM5-80GB", count=1, memory_mib=81920)

# Patch target for detect_local_gpu — must patch on the consumer module
# since cli/infer.py binds the name at import time via `from ... import`.
_GPU_PATCH = "sieval.cli.infer.recipe.detect_local_gpu"


def _write_yaml(path: Path, models_cfg: dict) -> Path:
    """Write a minimal YAML config and return its path."""
    cfg = {"models": models_cfg}
    yaml_path = path / "config.yaml"
    yaml_path.write_text(yaml.dump(cfg))
    return yaml_path


def _write_qwen3_4b_checkpoint(model_dir: Path) -> None:
    """Write a Qwen3-4B-like config.json into model_dir."""
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
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(config))


def _write_unknown_arch_checkpoint(model_dir: Path) -> None:
    """Write a config.json with an unknown architecture."""
    config = {
        "architectures": ["TotallyNewModelForCausalLM"],
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "torch_dtype": "bfloat16",
    }
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(config))


def _get_engine_params(plan: DeploymentPlan) -> dict:
    """Extract engine_params from first assignment of a plan."""
    return dict(plan.assignments[0].engine_params)


def _get_all_params(plan: DeploymentPlan) -> dict:
    """Get all params (topology + engine) as a flat dict for assertions."""
    a = plan.assignments[0]
    params = dict(a.engine_params)
    topo = a.topology
    if topo.tp > 1:
        params["tp_size"] = topo.tp
    if topo.dp > 1:
        params["dp_size"] = topo.dp
    return params


# ---------------------------------------------------------------------------
# Case 1: explicit recipe (baseline — unchanged behaviour)
# ---------------------------------------------------------------------------


class TestExplicitRecipe:
    @pytest.mark.anyio
    async def test_explicit_recipe_loads(self, tmp_path: Path) -> None:
        """Explicit recipe name loads normally, no auto-resolve."""
        model_dir = tmp_path / "Qwen3-4B"
        _write_qwen3_4b_checkpoint(model_dir)

        yaml_path = _write_yaml(
            tmp_path,
            {
                "mymodel": {
                    "infer": {
                        "backend": "sglang",
                        "recipe": "qwen3-4b",
                        "checkpoint": str(model_dir),
                    },
                },
            },
        )
        with patch(_GPU_PATCH, new_callable=AsyncMock, return_value=_MOCK_GPU):
            name, plan, _env = await resolve_infer_config(yaml_path)
        assert name == "mymodel"
        assert isinstance(plan, DeploymentPlan)
        assert plan.backend == "sglang"
        # Recipe params should be populated (H100 bf16 profile matched)
        assert plan.assignments  # non-empty

    @pytest.mark.anyio
    async def test_explicit_recipe_no_gpu(self, tmp_path: Path) -> None:
        """Explicit recipe with no GPU detected → overrides used."""
        model_dir = tmp_path / "Qwen3-4B"
        _write_qwen3_4b_checkpoint(model_dir)

        yaml_path = _write_yaml(
            tmp_path,
            {
                "mymodel": {
                    "infer": {
                        "backend": "sglang",
                        "recipe": "qwen3-4b",
                        "checkpoint": str(model_dir),
                        "overrides": {"tp_size": 1, "dtype": "bfloat16"},
                    },
                },
            },
        )
        with patch(_GPU_PATCH, new_callable=AsyncMock, return_value=None):
            _, plan, _env = await resolve_infer_config(yaml_path)
        # No GPU → no profile match; identity resolved → dtype fallback
        params = _get_all_params(plan)
        assert params.get("dtype") == "bfloat16"

    @pytest.mark.anyio
    async def test_explicit_recipe_unmatched_gpu(self, tmp_path: Path) -> None:
        """Explicit recipe with a GPU that doesn't match any profile key."""
        model_dir = tmp_path / "Qwen3-4B"
        _write_qwen3_4b_checkpoint(model_dir)

        unknown_gpu = GPUInfo(model="NVIDIA RTX 4090-24GB", count=1, memory_mib=24576)
        yaml_path = _write_yaml(
            tmp_path,
            {
                "mymodel": {
                    "infer": {
                        "backend": "sglang",
                        "recipe": "qwen3-4b",
                        "checkpoint": str(model_dir),
                        "overrides": {"dtype": "bfloat16"},
                    },
                },
            },
        )
        with patch(
            _GPU_PATCH,
            new_callable=AsyncMock,
            return_value=unknown_gpu,
        ):
            _, plan, _env = await resolve_infer_config(yaml_path)
        # GPU doesn't match any profile → falls back to overrides
        engine_params = _get_engine_params(plan)
        assert engine_params.get("dtype") == "bfloat16"


# ---------------------------------------------------------------------------
# Case 2: no recipe + introspectable checkpoint + recipe matched
# ---------------------------------------------------------------------------


class TestAutoResolveRecipeMatched:
    @pytest.mark.anyio
    async def test_auto_selects_recipe_with_warning(self, tmp_path: Path) -> None:
        """Checkpoint introspection → recipe match → populated + WARNING."""
        model_dir = tmp_path / "Qwen3-4B"
        _write_qwen3_4b_checkpoint(model_dir)

        yaml_path = _write_yaml(
            tmp_path,
            {
                "mymodel": {
                    "infer": {
                        "backend": "sglang",
                        "checkpoint": str(model_dir),
                    },
                },
            },
        )

        # Capture loguru output via a temporary sink
        warnings: list[str] = []
        sink_id = logger.add(
            lambda msg: warnings.append(str(msg)),
            level="WARNING",
            filter=lambda record: record["level"].name == "WARNING",
        )
        try:
            with patch(
                _GPU_PATCH,
                new_callable=AsyncMock,
                return_value=_MOCK_GPU,
            ):
                name, plan, _env = await resolve_infer_config(yaml_path)
        finally:
            logger.remove(sink_id)

        assert name == "mymodel"
        assert isinstance(plan, DeploymentPlan)
        # Should have some engine params from recipe profile
        engine_params = _get_engine_params(plan)
        assert "dtype" in engine_params or "mem_fraction_static" in engine_params

        # Check the auto-selected recipe WARNING
        auto_msgs = [w for w in warnings if "Auto-selected" in w]
        assert auto_msgs, "Expected a WARNING about auto-selected recipe"
        assert "qwen3-4b" in auto_msgs[0]
        assert "To suppress this warning" in auto_msgs[0]

    @pytest.mark.anyio
    async def test_auto_resolve_with_overrides_merged(self, tmp_path: Path) -> None:
        """Auto-resolve recipe + user overrides → overrides take priority."""
        model_dir = tmp_path / "Qwen3-4B"
        _write_qwen3_4b_checkpoint(model_dir)

        yaml_path = _write_yaml(
            tmp_path,
            {
                "mymodel": {
                    "infer": {
                        "backend": "sglang",
                        "checkpoint": str(model_dir),
                        "overrides": {
                            "dtype": "float16",
                            "port": 9000,
                        },
                    },
                },
            },
        )

        with patch(
            _GPU_PATCH,
            new_callable=AsyncMock,
            return_value=_MOCK_GPU,
        ):
            _, plan, _env = await resolve_infer_config(yaml_path)

        # User overrides should win
        engine_params = _get_engine_params(plan)
        assert engine_params["dtype"] == "float16"
        assert engine_params["port"] == 9000


# ---------------------------------------------------------------------------
# Case 3: no recipe + introspectable + no match + overrides
# ---------------------------------------------------------------------------


class TestAutoResolveNoMatchWithOverrides:
    @pytest.mark.anyio
    async def test_unknown_family_with_overrides_proceeds(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Unknown model family + overrides → uses overrides only."""
        model_dir = tmp_path / "CustomModel"
        _write_unknown_arch_checkpoint(model_dir)

        yaml_path = _write_yaml(
            tmp_path,
            {
                "mymodel": {
                    "infer": {
                        "backend": "sglang",
                        "checkpoint": str(model_dir),
                        "overrides": {
                            "tp_size": 2,
                            "dtype": "bfloat16",
                        },
                    },
                },
            },
        )

        with patch(
            _GPU_PATCH,
            new_callable=AsyncMock,
            return_value=_MOCK_GPU,
        ):
            _, plan, _env = await resolve_infer_config(yaml_path)

        # Check that overrides are present
        all_params = _get_all_params(plan)
        assert all_params.get("dtype") == "bfloat16"
        # tp_size=2 should be reflected in topology or engine params
        topo = plan.assignments[0].topology
        engine = _get_engine_params(plan)
        assert topo.tp == 2 or engine.get("tp_size") == 2


# ---------------------------------------------------------------------------
# Case 4: no recipe + introspectable + no match + no overrides → error
# ---------------------------------------------------------------------------


class TestAutoResolveNoMatchNoOverrides:
    @pytest.mark.anyio
    async def test_unknown_family_no_overrides_raises(self, tmp_path: Path) -> None:
        """Unknown model family + no overrides → BadParameter."""
        model_dir = tmp_path / "CustomModel"
        _write_unknown_arch_checkpoint(model_dir)

        yaml_path = _write_yaml(
            tmp_path,
            {
                "mymodel": {
                    "infer": {
                        "backend": "sglang",
                        "checkpoint": str(model_dir),
                    },
                },
            },
        )

        from typer import BadParameter

        with (
            patch(
                _GPU_PATCH,
                new_callable=AsyncMock,
                return_value=_MOCK_GPU,
            ),
            pytest.raises(BadParameter, match="Available recipes"),
        ):
            await resolve_infer_config(yaml_path)


# ---------------------------------------------------------------------------
# Case 5: no recipe + not introspectable + overrides → overrides only
# ---------------------------------------------------------------------------


class TestNoIntrospectWithOverrides:
    @pytest.mark.anyio
    async def test_no_config_json_with_overrides_proceeds(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Missing config.json + overrides → uses overrides only."""
        fake_checkpoint = tmp_path / "no-config-model"
        fake_checkpoint.mkdir()
        # No config.json written

        yaml_path = _write_yaml(
            tmp_path,
            {
                "mymodel": {
                    "infer": {
                        "backend": "sglang",
                        "checkpoint": str(fake_checkpoint),
                        "overrides": {"tp_size": 4},
                    },
                },
            },
        )

        _, plan, _env = await resolve_infer_config(yaml_path)

        # tp_size=4 should be reflected
        all_params = _get_all_params(plan)
        topo = plan.assignments[0].topology
        assert topo.tp == 4 or all_params.get("tp_size") == 4


# ---------------------------------------------------------------------------
# Case 6: no recipe + not introspectable + no overrides → error
# ---------------------------------------------------------------------------


class TestNoIntrospectNoOverrides:
    @pytest.mark.anyio
    async def test_no_config_json_no_overrides_raises(self, tmp_path: Path) -> None:
        """Missing config.json + no overrides → BadParameter."""
        fake_checkpoint = tmp_path / "no-config-model"
        fake_checkpoint.mkdir()

        yaml_path = _write_yaml(
            tmp_path,
            {
                "mymodel": {
                    "infer": {
                        "backend": "sglang",
                        "checkpoint": str(fake_checkpoint),
                    },
                },
            },
        )

        from typer import BadParameter

        with pytest.raises(BadParameter, match="Cannot introspect"):
            await resolve_infer_config(yaml_path)


# ---------------------------------------------------------------------------
# Case 7: no recipe + no checkpoint + overrides → overrides as-is
# ---------------------------------------------------------------------------


class TestNoCheckpointWithOverrides:
    @pytest.mark.anyio
    async def test_overrides_only_no_checkpoint(self, tmp_path: Path) -> None:
        """No recipe, no checkpoint, just overrides → uses them."""
        yaml_path = _write_yaml(
            tmp_path,
            {
                "mymodel": {
                    "infer": {
                        "backend": "sglang",
                        "overrides": {
                            "tp_size": 2,
                            "dtype": "float16",
                        },
                    },
                },
            },
        )

        _, plan, _env = await resolve_infer_config(yaml_path)

        all_params = _get_all_params(plan)
        assert all_params.get("dtype") == "float16"
        topo = plan.assignments[0].topology
        assert topo.tp == 2 or all_params.get("tp_size") == 2


# ---------------------------------------------------------------------------
# Case 8: no recipe + no checkpoint + no overrides → error
# ---------------------------------------------------------------------------


class TestNoRecipeNoCheckpointNoOverrides:
    @pytest.mark.anyio
    async def test_nothing_specified_raises(self, tmp_path: Path) -> None:
        """No recipe, no checkpoint, no overrides → BadParameter."""
        yaml_path = _write_yaml(
            tmp_path,
            {
                "mymodel": {
                    "infer": {
                        "backend": "sglang",
                    },
                },
            },
        )

        from typer import BadParameter

        with pytest.raises(BadParameter, match="Available recipes"):
            await resolve_infer_config(yaml_path)


# ---------------------------------------------------------------------------
# Env passthrough
# ---------------------------------------------------------------------------


class TestEnvPassthrough:
    @pytest.mark.anyio
    async def test_env_returned_from_resolve(self, tmp_path: Path) -> None:
        """env section in YAML is returned as the third element."""
        model_dir = tmp_path / "Qwen3-4B"
        _write_qwen3_4b_checkpoint(model_dir)

        yaml_path = _write_yaml(
            tmp_path,
            {
                "mymodel": {
                    "infer": {
                        "backend": "sglang",
                        "recipe": "qwen3-4b",
                        "checkpoint": str(model_dir),
                        "env": {"NCCL_DEBUG": "INFO", "FOO": "bar"},
                    },
                },
            },
        )
        with patch(_GPU_PATCH, new_callable=AsyncMock, return_value=_MOCK_GPU):
            _, _, env = await resolve_infer_config(yaml_path)
        assert env == {"NCCL_DEBUG": "INFO", "FOO": "bar"}

    @pytest.mark.anyio
    async def test_env_empty_when_omitted(self, tmp_path: Path) -> None:
        """Omitting env section returns an empty dict."""
        model_dir = tmp_path / "Qwen3-4B"
        _write_qwen3_4b_checkpoint(model_dir)

        yaml_path = _write_yaml(
            tmp_path,
            {
                "mymodel": {
                    "infer": {
                        "backend": "sglang",
                        "recipe": "qwen3-4b",
                        "checkpoint": str(model_dir),
                    },
                },
            },
        )
        with patch(_GPU_PATCH, new_callable=AsyncMock, return_value=_MOCK_GPU):
            _, _, env = await resolve_infer_config(yaml_path)
        assert env == {}

    @pytest.mark.anyio
    async def test_non_string_env_values_coerced(self, tmp_path: Path) -> None:
        """YAML int/bool env values are coerced to str."""
        model_dir = tmp_path / "Qwen3-4B"
        _write_qwen3_4b_checkpoint(model_dir)

        yaml_path = _write_yaml(
            tmp_path,
            {
                "mymodel": {
                    "infer": {
                        "backend": "sglang",
                        "recipe": "qwen3-4b",
                        "checkpoint": str(model_dir),
                        "env": {"DEBUG": 1, "VERBOSE": True},
                    },
                },
            },
        )
        with patch(_GPU_PATCH, new_callable=AsyncMock, return_value=_MOCK_GPU):
            _, _, env = await resolve_infer_config(yaml_path)
        assert env == {"DEBUG": "1", "VERBOSE": "True"}


# ---------------------------------------------------------------------------
# Root-level deterministic propagation
# ---------------------------------------------------------------------------


class TestDeterministicPropagation:
    """resolve_infer_config reads root-level `deterministic: true` and sets
    plan.deterministic/seed so every entrypoint (sieval run, sieval infer
    start) inherits the YAML intent without the CLI layer having to mirror
    it separately."""

    @pytest.mark.anyio
    async def test_yaml_deterministic_true_sets_plan(self, tmp_path: Path) -> None:
        from sieval.infer.topology.models import DETERMINISTIC_DEFAULT_SEED

        model_dir = tmp_path / "Qwen3-4B"
        _write_qwen3_4b_checkpoint(model_dir)

        cfg = {
            "deterministic": True,
            "models": {
                "mymodel": {
                    "infer": {
                        "backend": "sglang",
                        "recipe": "qwen3-4b",
                        "checkpoint": str(model_dir),
                    },
                },
            },
        }
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(cfg))
        with patch(_GPU_PATCH, new_callable=AsyncMock, return_value=_MOCK_GPU):
            _, plan, _ = await resolve_infer_config(yaml_path)
        assert plan.deterministic is True
        assert plan.seed == DETERMINISTIC_DEFAULT_SEED

    @pytest.mark.anyio
    async def test_yaml_deterministic_false_leaves_plan_default(
        self, tmp_path: Path
    ) -> None:
        model_dir = tmp_path / "Qwen3-4B"
        _write_qwen3_4b_checkpoint(model_dir)

        cfg = {
            "deterministic": False,
            "models": {
                "mymodel": {
                    "infer": {
                        "backend": "sglang",
                        "recipe": "qwen3-4b",
                        "checkpoint": str(model_dir),
                    },
                },
            },
        }
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(cfg))
        with patch(_GPU_PATCH, new_callable=AsyncMock, return_value=_MOCK_GPU):
            _, plan, _ = await resolve_infer_config(yaml_path)
        assert plan.deterministic is False
        # seed defaults to DETERMINISTIC_DEFAULT_SEED even when non-deterministic;
        # translators only emit --seed when plan.deterministic is True.
        from sieval.infer.topology.models import DETERMINISTIC_DEFAULT_SEED

        assert plan.seed == DETERMINISTIC_DEFAULT_SEED

    @pytest.mark.anyio
    async def test_yaml_deterministic_missing_leaves_plan_default(
        self, tmp_path: Path
    ) -> None:
        from sieval.infer.topology.models import DETERMINISTIC_DEFAULT_SEED

        model_dir = tmp_path / "Qwen3-4B"
        _write_qwen3_4b_checkpoint(model_dir)

        yaml_path = _write_yaml(
            tmp_path,
            {
                "mymodel": {
                    "infer": {
                        "backend": "sglang",
                        "recipe": "qwen3-4b",
                        "checkpoint": str(model_dir),
                    },
                },
            },
        )
        with patch(_GPU_PATCH, new_callable=AsyncMock, return_value=_MOCK_GPU):
            _, plan, _ = await resolve_infer_config(yaml_path)
        assert plan.deterministic is False
        assert plan.seed == DETERMINISTIC_DEFAULT_SEED


# ---------------------------------------------------------------------------
# Key normalization in _resolve_recipe_params
# ---------------------------------------------------------------------------


class TestResolveRecipeParamsKeyNormalization:
    """Regression: key-form collisions between profile and user overrides must
    be resolved at merge time (user override wins; shadowed preset dropped).
    """

    @pytest.mark.anyio
    async def test_user_override_dash_form_wins_over_profile_underscore(
        self,
    ) -> None:
        identity = ModelIdentity(
            architecture="TestForCausalLM",
            family="test",
            param_billions=7.0,
            dtype="bfloat16",
        )
        recipe = Recipe(name="test-recipe")

        # Isolate merge logic: no GPU → skip formula TP/DP; stubbed profile → pin
        # exactly what profile contributes.
        with (
            patch(
                "sieval.cli.infer.recipe.detect_local_gpu",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "sieval.cli.infer.recipe.resolve_profile",
                return_value={"foo_bar": 0},
            ),
        ):
            result = await _resolve_recipe_params(
                identity,
                recipe,
                backend_name="vllm",
                overrides={"foo-bar": 42},
            )

        assert result == {"foo_bar": 42}

    @pytest.mark.anyio
    async def test_dash_form_override_is_normalized_when_fallback_engages(
        self,
    ) -> None:
        """Dash-form user overrides normalize even on the dtype fallback path."""
        identity = ModelIdentity(
            architecture="TestForCausalLM",
            family="test",
            param_billions=7.0,
            dtype="bfloat16",
        )
        recipe = Recipe(name="test-recipe")

        with (
            patch(
                "sieval.cli.infer.recipe.detect_local_gpu",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "sieval.cli.infer.recipe.resolve_profile",
                return_value=None,  # profile-less → dtype fallback engaged
            ),
        ):
            result = await _resolve_recipe_params(
                identity,
                recipe,
                backend_name="vllm",
                overrides={"max-model-len": 4096},
            )

        # Fallback injects identity.dtype; dash-form override normalizes to underscore.
        assert result == {"dtype": "bfloat16", "max_model_len": 4096}
