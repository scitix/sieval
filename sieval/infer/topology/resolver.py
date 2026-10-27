"""
Topology resolver: derive DeploymentPlan from model + hardware + user hints.

Sizes tp from the declared engine memory (see ``memory.py``) and merges
params into a DeploymentPlan.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from loguru import logger

from sieval.infer.config import ParamValue
from sieval.infer.introspect import (
    TARGET_GPU_ENV,
    LaunchGpuError,
    ModelIdentity,
    bytes_per_param,
    declared_target_gpu,
    detect_local_gpu,
    extract_moe_info,
    introspect_checkpoint_with_config,
)
from sieval.infer.params import merge_params
from sieval.infer.recipes import (
    match_recipe,
    resolve_capability_profile,
    resolve_hardware_profile,
)
from sieval.infer.topology.memory import (
    MemoryDemand,
    default_fraction,
    memory_demand,
    size_tp,
)
from sieval.infer.topology.models import (
    TOPO_KEYS,
    DeploymentPlan,
    DeviceGroup,
    HardwareEnv,
    ModelProfile,
    ParallelTopology,
    ResolveResult,
    ResolveStep,
    RoleAssignment,
    UserHints,
    WellKnownRole,
)

# Model families where DPA is auto-enabled (well-tested, officially recommended).
# Currently only DeepSeek (V2/V3/R1) which use MLA attention.
_SGLANG_DPA_AUTO_FAMILIES = frozenset({"deepseek"})

# Families that are DPA-capable (user can force DPA on without warning).
# sglang code-level supports these, but no official cookbook recommendation.
_SGLANG_DPA_CAPABLE_FAMILIES = frozenset(
    {"deepseek", "qwen2", "qwen3", "minimax", "kimi"}
)

# Quantization methods whose name IS the yaml precision key (not fp{bits}/int{bits}).
_NAMED_PRECISION_METHODS = frozenset({"mxfp4"})

# Float quantization methods — their precision key uses "fp" prefix
_FLOAT_QUANT_METHODS = frozenset(
    {
        "compressed-tensors",
        "fbgemm_fp8",
        "fp8",
        "quanto",
    }
)


def precision_key(identity: ModelIdentity) -> str:
    """Derive the profile precision lookup key from a ModelIdentity.

    Returns a key like "bf16", "fp16", "fp8", "int4", "fp32", "fp64".
    Always returns a string (never None).
    """
    if identity.quantization is not None:
        q = identity.quantization
        if q.quant_method in _NAMED_PRECISION_METHODS:
            return q.quant_method
        prefix = "fp" if q.quant_method in _FLOAT_QUANT_METHODS else "int"
        return f"{prefix}{q.bits}"
    dtype_map: dict[str, str] = {
        "bfloat16": "bf16",
        "float16": "fp16",
        "float32": "fp32",
        "float64": "fp64",
    }
    return dtype_map.get(identity.dtype, "bf16")


def _largest_divisor(n: int, upper_bound: int) -> int:
    """Return the largest divisor of *n* that is ≤ *upper_bound*.

    Used for sglang EP: find largest EP such that EP ≤ TP and TP % EP == 0
    and EP ≤ num_experts.
    """
    # Iterate from min(n, upper_bound) downward
    for d in range(min(n, upper_bound), 0, -1):
        if n % d == 0:
            return d
    return 1  # pragma: no cover — n >= 1 always has divisor 1


def _derive_topology(
    model: ModelProfile,
    hw: HardwareEnv,
    hints: UserHints | None,
    steps: list[ResolveStep],
    backend: str = "sglang",
    demand: MemoryDemand | None = None,
) -> ParallelTopology:
    """Derive ParallelTopology for a single FULL role.

    TP: the user's, else the smallest that fits *demand* (see
    :func:`size_tp`). Without a *demand*, *model*'s weights at the engine's
    default memory fraction.

    EP/DPA constraints are backend-specific:
      - **sglang**: EP ≤ TP, TP % EP == 0 (sglang computes tp_size // ep_size).
        DPA requires DP ≤ TP and TP % DP == 0.
      - **vllm**: EP is a boolean flag; vLLM auto-computes EP = TP × DP.
        num_experts % (TP × DP) == 0 required. DPA not supported.
    """
    # 1. TP
    if demand is None:
        demand = MemoryDemand(
            weight_bytes=model.param_billions * 1e9 * model.bytes_per_param,
            fraction=default_fraction(backend),
            fraction_declared=False,
        )
    stated_tp = hints.tp if hints else None
    fit = size_tp(
        demand,
        gpu_count=hw.gpu_count,
        gpu_memory_mib=hw.gpu_memory_mib,
        stated_tp=stated_tp,
    )
    tp = fit.tp
    steps.append(
        ResolveStep(
            field="tp",
            value=tp,
            reason=fit.describe(),
            source="user_override" if stated_tp is not None else "auto",
        )
    )

    # 2. DP: fill remaining GPUs
    dp = max(1, hw.gpu_count // tp)
    if dp > 1:
        steps.append(
            ResolveStep(
                field="dp",
                value=dp,
                reason=f"gpu_count({hw.gpu_count}) // tp({tp})",
                source="auto",
            )
        )

    # 3. User DP override
    if hints and hints.dp is not None:
        dp = hints.dp
        steps.append(
            ResolveStep(
                field="dp",
                value=dp,
                reason="user override",
                source="user_override",
            )
        )

    pp = hints.pp if hints and hints.pp is not None else 1
    if hints and hints.pp is not None:
        steps.append(
            ResolveStep(
                field="pp", value=pp, reason="user override", source="user_override"
            )
        )

    ep = hints.ep if hints and hints.ep is not None else None
    cp = hints.cp if hints and hints.cp is not None else None
    if hints and hints.cp is not None:
        steps.append(
            ResolveStep(
                field="cp", value=cp, reason="user override", source="user_override"
            )
        )

    # Auto EP for MoE models — backend-specific constraints
    if ep is None and model.is_moe and model.num_experts is not None:
        if backend == "vllm":
            # vLLM: EP is a boolean flag, EP_SIZE = TP × DP (auto-computed).
            # We just signal "enable EP" by setting ep to the auto value.
            # Constraint: num_experts must be divisible by TP × DP.
            auto_ep = tp * dp
            if model.num_experts % auto_ep == 0:
                ep = auto_ep
                steps.append(
                    ResolveStep(
                        field="ep",
                        value=ep,
                        reason=(
                            f"vLLM MoE auto: ep=tp*dp={auto_ep}, "
                            f"num_experts({model.num_experts}) % {auto_ep} == 0"
                        ),
                        source="auto",
                    )
                )
            else:
                steps.append(
                    ResolveStep(
                        field="ep",
                        value=None,
                        reason=(
                            f"vLLM MoE: skipped, num_experts({model.num_experts})"
                            f" % tp*dp({auto_ep}) != 0"
                        ),
                        source="auto",
                    )
                )
        elif backend == "sglang":
            # sglang: EP ≤ TP, TP % EP == 0.
            # Find largest divisor of TP that is ≤ num_experts.
            auto_ep = _largest_divisor(tp, model.num_experts)
            if auto_ep > 1:
                ep = auto_ep
                steps.append(
                    ResolveStep(
                        field="ep",
                        value=ep,
                        reason=(
                            f"sglang MoE auto: largest divisor of tp={tp}"
                            f" ≤ num_experts={model.num_experts} → ep={auto_ep}"
                        ),
                        source="auto",
                    )
                )

    enable_dpa = False
    if hints and hints.enable_dp_attention is not None:
        # User explicitly set DPA
        enable_dpa = hints.enable_dp_attention
        if enable_dpa and model.family not in _SGLANG_DPA_CAPABLE_FAMILIES:
            # sglang has no model-family validation for DPA — unsupported
            # models will silently produce wrong results.
            logger.warning(
                "enable_dp_attention forced on, but family={!r} is not in "
                "the supported list {}. sglang will NOT error but may "
                "produce silently incorrect outputs.",
                model.family,
                sorted(_SGLANG_DPA_CAPABLE_FAMILIES),
            )
        steps.append(
            ResolveStep(
                field="enable_dp_attention",
                value=enable_dpa,
                reason="user override",
                source="user_override",
            )
        )
    elif (
        backend == "sglang"
        and model.is_moe
        and dp > 1
        # No tp % dp check: sglang DPA constraint is total_tp % dp == 0,
        # and total_tp = tp * dp, so it is always satisfied.
        and model.family in _SGLANG_DPA_AUTO_FAMILIES
    ):
        # Auto DPA for sglang — only officially recommended families
        # (DeepSeek-V2/V3/R1 with MLA attention per sglang docs)
        enable_dpa = True
        steps.append(
            ResolveStep(
                field="enable_dp_attention",
                value=True,
                reason=(
                    f"MoE family={model.family} with dp={dp}"
                    f" (auto DPA for supported family)"
                ),
                source="auto",
            )
        )

    return ParallelTopology(
        tp=tp,
        dp=dp,
        pp=pp,
        ep=ep,
        enable_dp_attention=enable_dpa,
        cp=cp,
    )


def resolve_simple(
    *,
    checkpoint: str,
    backend: str,
    model: ModelProfile,
    hw: HardwareEnv,
    hints: UserHints | None = None,
    recipe_params: dict[str, ParamValue] | None = None,
    demand: MemoryDemand | None = None,
) -> ResolveResult:
    """Resolve a single-role (FULL) DeploymentPlan.

    This is the Phase 1 entry point. Multi-role (PD split) comes in Phase 2.
    """
    steps: list[ResolveStep] = []
    topology = _derive_topology(model, hw, hints, steps, backend=backend, demand=demand)

    # Safety check
    needed = topology.gpu_count
    if needed > hw.gpu_count:
        raise RuntimeError(
            f"Model requires {needed} GPUs (tp={topology.tp} × dp={topology.dp} "
            f"× pp={topology.pp}) but only {hw.gpu_count} available."
        )

    assignment = RoleAssignment(
        role=WellKnownRole.FULL,
        devices=DeviceGroup(
            count=needed,
            gpu_model=hw.gpu_model,
        ),
        topology=topology,
        engine_params=dict(recipe_params) if recipe_params else {},
    )

    plan = DeploymentPlan(
        checkpoint=checkpoint,
        backend=backend,
        assignments=(assignment,),
    )

    return ResolveResult(plan=plan, steps=tuple(steps))


async def auto_resolve_plan(
    checkpoint: str,
    *,
    backend: str = "sglang",
    overrides: dict[str, ParamValue] | None = None,
    capability: str = "instruct",
) -> ResolveResult:
    """Auto-resolve a DeploymentPlan from a checkpoint path.

    Same 5-step flow as the old resolve.auto_resolve(), returning a
    ResolveResult.

    Args:
        capability: Recipe capability key (``"instruct"`` / ``"base"``). Base
            checkpoints resolve without parser or tool-choice params. Defaults
            to ``"instruct"`` for standalone callers with no task context.
            This is the *recipe* vocabulary, not a config ``type:`` — pass a
            config type through
            :func:`sieval.infer.recipes.capability_model_type` first. An
            unrecognized value raises rather than resolving to no capabilities.
    """
    # Normalize keys at the entry: the TOPO_KEYS filter (step 4) and
    # _overrides_to_hints both look up specific underscore-form keys, so a
    # dash-form override like `tensor-parallel-size` would otherwise leak
    # into engine_params and the tp hint would come back None.
    overrides = merge_params(overrides or {})

    # 1. Introspect (read config.json once)
    identity, config_json = introspect_checkpoint_with_config(checkpoint)
    is_moe, num_experts, num_layers = extract_moe_info(config_json)

    model = ModelProfile(
        param_billions=identity.param_billions,
        num_layers=num_layers,
        is_moe=is_moe,
        num_experts=num_experts,
        bytes_per_param=bytes_per_param(identity),
        family=identity.family,
    )

    # 2. The declared GPUs, else this node's
    gpu = declared_target_gpu() or await detect_local_gpu()
    if gpu is None:
        raise LaunchGpuError(
            "no_gpu",
            "No GPU detected — cannot auto-resolve deployment plan. On a node "
            f"without the GPUs, declare them in {TARGET_GPU_ENV}.",
        )

    hw = HardwareEnv(
        gpu_count=gpu.count,
        gpu_model=gpu.model,
        gpu_memory_mib=gpu.memory_mib,
    )

    # 3. Match recipe → resolve hardware + capability layers → recipe_params
    recipe = match_recipe(identity.family, identity.param_billions)
    recipe_params: dict[str, ParamValue] | None = None
    if recipe is not None:
        prec_key = precision_key(identity)
        profile = resolve_hardware_profile(recipe, gpu.model, prec_key, backend)
        capabilities = resolve_capability_profile(recipe, capability, backend)
        # Capabilities last, matching `_resolve_recipe_params`; both sites feed
        # the same `infer_plans.yaml`, whose key order `--resume` compares
        # byte-for-byte. `merge_params` rather than a dict literal so both sites
        # emit one canonical key form — it normalizes each source, so a
        # dash-form recipe key cannot reach the plan through only one of them.
        # Note it does *not* flag a dash/underscore clash *across* sources: it
        # normalizes and the later source wins. Recipe YAML is underscore-only
        # by convention (`sieval/infer/params.py`), and a within-layer clash
        # still raises.
        if profile is not None or capabilities:
            recipe_params = merge_params(profile or {}, capabilities)

    # 4. Build user hints from overrides; non-topology keys become engine params
    hints: UserHints | None = None
    if overrides:
        hints = _overrides_to_hints(overrides)
        engine_overrides = {k: v for k, v in overrides.items() if k not in TOPO_KEYS}
        if engine_overrides:
            if recipe_params is not None:
                recipe_params.update(engine_overrides)
            else:
                recipe_params = engine_overrides

    # 5. Resolve, sizing tp from the memory the params declare
    demand = memory_demand(
        checkpoint=checkpoint,
        identity=identity,
        backend=backend,
        params=recipe_params or {},
    )
    return resolve_simple(
        checkpoint=checkpoint,
        backend=backend,
        model=model,
        hw=hw,
        hints=hints,
        recipe_params=recipe_params,
        demand=demand,
    )


def _first_present(d: dict[str, ParamValue], *keys: str) -> ParamValue | None:
    """Return the value of the first key present in *d*, or None.

    Uses ``key in d`` (presence check) instead of truthiness so that
    falsy values like ``0`` or ``False`` are not silently skipped.
    """
    for k in keys:
        if k in d:
            return d[k]
    return None


def _safe_int(val: ParamValue, name: str) -> int:
    """Convert a ParamValue to int, warning if a float is truncated."""
    if isinstance(val, float) and val != int(val):
        logger.warning(
            "Topology hint {!r} has non-integer value {!r}, truncating to {}",
            name,
            val,
            int(val),
        )
    return int(val)


def _overrides_to_hints(overrides: dict[str, ParamValue]) -> UserHints:
    """Convert flat override dict to UserHints.

    Recognizes tp/dp/pp/ep/enable_dp_attention/cp keys (framework-agnostic)
    and passes them through. Also accepts framework-specific aliases
    (tp_size, tensor_parallel_size, etc.) for backward compatibility.
    """
    tp_val = _first_present(overrides, "tp", "tp_size", "tensor_parallel_size")
    dp_val = _first_present(overrides, "dp", "dp_size", "data_parallel_size")
    pp_val = _first_present(overrides, "pp", "pp_size", "pipeline_parallel_size")
    ep_val = _first_present(overrides, "ep", "ep_size")
    cp_val = _first_present(overrides, "cp", "attn_cp_size")

    # Bool fields
    dpa_val = _first_present(overrides, "enable_dp_attention")

    return UserHints(
        tp=_safe_int(tp_val, "tp") if tp_val is not None else None,
        dp=_safe_int(dp_val, "dp") if dp_val is not None else None,
        pp=_safe_int(pp_val, "pp") if pp_val is not None else None,
        ep=_safe_int(ep_val, "ep") if ep_val is not None else None,
        enable_dp_attention=bool(dpa_val) if dpa_val is not None else None,
        cp=_safe_int(cp_val, "cp") if cp_val is not None else None,
    )
