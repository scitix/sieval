"""
Recipe resolution logic for inference configuration.

Resolves YAML infer configs into DeploymentPlans by matching recipes,
introspecting checkpoints, and merging overrides.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import dataclasses
from pathlib import Path
from typing import NamedTuple

import typer
import yaml
from loguru import logger

from sieval.infer.config import ParamValue
from sieval.infer.introspect import (
    GPUInfo,
    ModelIdentity,
    bytes_per_param,
    detect_local_gpu,
    introspect_checkpoint_sync,
)
from sieval.infer.params import merge_params
from sieval.infer.recipes import (
    Recipe,
    list_recipes,
    load_family_recipes,
    load_recipe,
    match_recipe,
    resolve_profile,
)
from sieval.infer.topology.models import (
    CP_KEYS,
    DP_KEYS,
    DPA_KEYS,
    EP_KEYS,
    PP_KEYS,
    TOPO_KEYS,
    TP_KEYS,
    DeploymentPlan,
    DeviceGroup,
    ParallelTopology,
    RoleAssignment,
    WellKnownRole,
)
from sieval.infer.topology.resolver import compute_tp, precision_key


class ResolvedInferConfig(NamedTuple):
    """Result of resolve_infer_config: model identity, plan, and user env."""

    model_name: str
    plan: DeploymentPlan
    user_env: dict[str, str]


async def resolve_infer_config(
    yaml_path: Path,
    model_name: str | None = None,
) -> ResolvedInferConfig:
    """Load YAML and resolve a DeploymentPlan for a model.

    Recipe resolution strategy (in order):
      1. Explicit ``recipe`` field → load and merge with overrides.
      2. No recipe, but ``checkpoint`` exists → introspect checkpoint,
         attempt recipe auto-match.  Warn the user about the auto-selected
         recipe and list alternatives for the same family.
      3. No recipe match, but ``overrides`` present → use overrides only
         (supports custom/novel architectures).
      4. No recipe, no overrides, no checkpoint → error with available recipes.
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}

    models = cfg.get("models", {})
    if not models:
        raise typer.BadParameter("No models defined in config")

    # Resolve which model to serve
    if model_name is None:
        # Use first model with a 'infer' section
        for name, mcfg in models.items():
            if "infer" in mcfg:
                model_name = name
                break
        if model_name is None:
            raise typer.BadParameter(
                "No model has a 'infer' section. Specify --model or add infer config."
            )

    if model_name not in models:
        raise typer.BadParameter(f"Model {model_name!r} not found in config")

    mcfg = models[model_name]
    infer_dict = mcfg.get("infer", {})
    if not infer_dict:
        raise typer.BadParameter(f"Model {model_name!r} has no 'infer' section")

    backend_name = infer_dict.get("backend")
    if not backend_name:
        raise typer.BadParameter(f"Model {model_name!r} infer config missing 'backend'")

    checkpoint = infer_dict.get("checkpoint", "")
    overrides = infer_dict.get("overrides", {})
    raw_env = infer_dict.get("env") or {}
    user_env: dict[str, str] = {k: str(v) for k, v in raw_env.items()}

    # Recipe resolution
    recipe_params: dict[str, ParamValue] | None = None
    recipe_name = infer_dict.get("recipe")

    if recipe_name:
        # Case 1: explicit recipe
        recipe = load_recipe(recipe_name)
        recipe_params = await _resolve_with_recipe(
            recipe,
            checkpoint,
            backend_name,
            overrides,
        )
    elif checkpoint:
        # Case 2: no recipe, but checkpoint available → try auto-resolve
        recipe_params = await _try_auto_resolve_recipe(
            checkpoint=checkpoint,
            backend_name=backend_name,
            overrides=overrides,
            model_name=model_name,
        )
    elif overrides:
        # Case 3: no recipe, no checkpoint, but overrides → use as-is
        recipe_params = dict(overrides)
    else:
        # Case 4: nothing to work with → error
        available = list_recipes()
        raise typer.BadParameter(
            f"Model {model_name!r} has no 'recipe', 'checkpoint', "
            f"or 'overrides'. "
            f"Available recipes: {', '.join(available) or '(none)'}"
        )

    # Build DeploymentPlan from the resolved params
    plan = _build_plan_from_params(
        checkpoint=checkpoint,
        backend=backend_name,
        params=recipe_params or {},
    )

    # Mirror YAML-level `deterministic: true` into the plan so every
    # entrypoint (sieval run, sieval infer start) picks it up. CLI force-on
    # layers on top via monotone OR upstream.
    if bool(cfg.get("deterministic", False)):
        plan = dataclasses.replace(plan, deterministic=True)

    return ResolvedInferConfig(model_name, plan, user_env)


def _build_plan_from_params(
    *,
    checkpoint: str,
    backend: str,
    params: dict[str, ParamValue],
) -> DeploymentPlan:
    """Build a DeploymentPlan from flat params dict (YAML compatibility).

    Extracts topology-related keys (tp, dp, pp, ep, cp, enable_dp_attention)
    from params; remainder becomes engine_params.
    """
    # Extract parallel topology keys
    tp = 1
    dp = 1
    pp = 1
    ep = None
    cp = None
    enable_dpa = False

    for key, val in params.items():
        if key in TP_KEYS:
            tp = int(val)
        elif key in DP_KEYS:
            dp = int(val)
        elif key in PP_KEYS:
            pp = int(val)
        elif key in EP_KEYS:
            ep = int(val)
        elif key in CP_KEYS:
            cp = int(val)
        elif key in DPA_KEYS:
            enable_dpa = bool(val)

    # Everything else is engine_params
    engine_params = {k: v for k, v in params.items() if k not in TOPO_KEYS}

    topo = ParallelTopology(
        tp=tp,
        dp=dp,
        pp=pp,
        ep=ep,
        enable_dp_attention=enable_dpa,
        cp=cp,
    )

    assignment = RoleAssignment(
        role=WellKnownRole.FULL,
        devices=DeviceGroup(count=topo.gpu_count),
        topology=topo,
        engine_params=engine_params,
    )

    return DeploymentPlan(
        checkpoint=checkpoint,
        backend=backend,
        assignments=(assignment,),
    )


# Framework-specific TP/DP key mappings for YAML compatibility
_TP_KEYS: dict[str, str] = {
    "vllm": "tensor_parallel_size",
    "sglang": "tp_size",
}
_DP_KEYS: dict[str, str] = {
    "vllm": "data_parallel_size",
    "sglang": "dp_size",
}


async def _resolve_recipe_params(
    identity: ModelIdentity,
    recipe: Recipe,
    backend_name: str,
    overrides: dict[str, ParamValue],
) -> dict[str, ParamValue]:
    """Resolve engine params for a recipe.

    Pipeline: formula TP/DP → profile → overrides → safety check.

    Extracted from _resolve_with_recipe / _try_auto_resolve_recipe to
    eliminate duplication.
    """
    # Normalize overrides once up front so the dtype check below and the
    # final merge operate on a single canonical key form.
    overrides = merge_params(overrides or {})

    gpu = await detect_local_gpu()

    params: dict[str, ParamValue] = {}
    if gpu:
        bpp = bytes_per_param(identity)
        tp = compute_tp(identity.param_billions, bpp, gpu.memory_mib)
        tp_key = _TP_KEYS.get(backend_name)
        if tp_key:
            params[tp_key] = tp
        dp_key = _DP_KEYS.get(backend_name)
        if dp_key and tp > 0:
            dp = gpu.count // tp
            if dp > 1:
                params[dp_key] = dp

    # Profile overrides formula
    prec_key = precision_key(identity)
    gpu_model = gpu.model if gpu else None
    profile = resolve_profile(recipe, gpu_model, prec_key, backend_name)

    if profile is None and identity.dtype and "dtype" not in overrides:
        # No profile → fallback to model's intrinsic dtype (unless user overrode).
        params["dtype"] = identity.dtype

    params = merge_params(params, profile or {}, overrides)

    # Safety check
    if gpu:
        _check_tp_dp_safety(params, backend_name, gpu)

    return params


async def _resolve_with_recipe(
    recipe: Recipe,
    checkpoint: str,
    backend_name: str,
    overrides: dict[str, ParamValue],
) -> dict[str, ParamValue] | None:
    """Merge params for an already-loaded recipe via shared merge logic.

    Introspects the checkpoint (sync) for model identity and precision,
    detects GPU, then derives params.
    """

    identity = None
    if checkpoint:
        try:
            identity = introspect_checkpoint_sync(checkpoint)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning(
                "Cannot introspect checkpoint {!r} ({}). "
                "Recipe profile may be incomplete.",
                checkpoint,
                exc,
            )

    if identity is not None:
        return await _resolve_recipe_params(identity, recipe, backend_name, overrides)
    else:
        # No identity — can only use overrides
        return dict(overrides) if overrides else None


def _check_tp_dp_safety(
    params: dict[str, ParamValue],
    framework: str,
    gpu: GPUInfo,
) -> None:
    """Validate that tp × dp does not exceed GPU count."""
    tp_key = _TP_KEYS.get(framework, "")
    dp_key = _DP_KEYS.get(framework, "")
    final_tp = params.get(tp_key)
    final_dp = params.get(dp_key, 1)

    if isinstance(final_tp, int) and final_tp > gpu.count:
        raise RuntimeError(
            f"Model requires tp={final_tp} but only {gpu.count} "
            f"GPUs detected. "
            f"The model is too large for the available hardware."
        )
    if (
        isinstance(final_tp, int)
        and isinstance(final_dp, int)
        and final_tp * final_dp > gpu.count
    ):
        raise RuntimeError(
            f"tp={final_tp} × dp={final_dp} = "
            f"{final_tp * final_dp} GPUs "
            f"required but only {gpu.count} available."
        )


async def _try_auto_resolve_recipe(
    *,
    checkpoint: str,
    backend_name: str,
    overrides: dict[str, ParamValue],
    model_name: str,
) -> dict[str, ParamValue] | None:
    """Attempt to auto-resolve a recipe from checkpoint introspection.

    Decision matrix:
      - introspect OK + recipe matched → use recipe + overrides, WARNING
      - introspect OK + no match + overrides → use overrides only, WARNING
      - introspect OK + no match + no overrides → error
      - introspect failed + overrides → use overrides only, WARNING
      - introspect failed + no overrides → error
    """
    # Try to introspect the checkpoint
    try:
        identity = introspect_checkpoint_sync(checkpoint)
    except (FileNotFoundError, ValueError) as exc:
        # Cannot introspect — fall back based on overrides
        if overrides:
            logger.warning(
                "Cannot introspect checkpoint {!r} ({}). "
                "Proceeding with overrides only.",
                checkpoint,
                exc,
            )
            return dict(overrides)
        available = list_recipes()
        raise typer.BadParameter(
            f"Cannot introspect checkpoint {checkpoint!r} ({exc}), "
            f"and no 'recipe' or 'overrides' specified. "
            f"Available recipes: "
            f"{', '.join(available) or '(none)'}"
        ) from exc

    # Introspection succeeded — try recipe matching
    recipe = match_recipe(identity.family, identity.param_billions)

    if recipe is not None:
        # Matched — use shared merge logic
        params = await _resolve_recipe_params(identity, recipe, backend_name, overrides)

        family_recipes = load_family_recipes(identity.family)
        family_names = [r.name for r in family_recipes]
        logger.warning(
            "No 'recipe' specified for model {!r}. "
            "Auto-selected recipe {!r} "
            "(family={}, ~{:.1f}B params). "
            "Available recipes for {}: {}. "
            "To suppress this warning, add "
            "'recipe: {}' to your infer config.",
            model_name,
            recipe.name,
            identity.family,
            identity.param_billions,
            identity.family,
            ", ".join(family_names),
            recipe.name,
        )
        return params

    # No recipe match — fall back based on overrides
    all_recipes = list_recipes()
    if overrides:
        logger.warning(
            "No recipe matched for model {!r} "
            "(family={}, ~{:.1f}B params). "
            "Proceeding with overrides only. "
            "Available recipes: {}",
            model_name,
            identity.family,
            identity.param_billions,
            ", ".join(all_recipes) or "(none)",
        )
        return dict(overrides)

    # No recipe, no overrides — error
    raise typer.BadParameter(
        f"No recipe matched for model {model_name!r} "
        f"(family={identity.family!r}, "
        f"~{identity.param_billions:.1f}B params), "
        f"and no 'overrides' specified. "
        f"Available recipes: "
        f"{', '.join(all_recipes) or '(none)'}"
    )
