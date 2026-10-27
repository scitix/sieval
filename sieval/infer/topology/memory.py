"""
Size tensor parallelism from the memory a plan declares.

The engine's own startup condition, checked before any GPU is allocated: one
GPU's share of the weights, plus the KV cache of one sequence at the declared
context, must fit in the declared memory fraction of that GPU. The fraction
(``mem_fraction_static`` / ``gpu_memory_utilization``) and the context
(``context_length`` / ``max_model_len``) are read from the resolved engine
params, so a user override beats the recipe. No ratio is assumed here: a plan
that declares no fraction is checked against the whole GPU, which only rules
out a tp the engine could never start with.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from dataclasses import dataclass

from loguru import logger

from sieval.infer.config import ParamValue
from sieval.infer.introspect import (
    ModelIdentity,
    bytes_per_param,
    checkpoint_weight_bytes,
    dtype_bytes,
    kv_shape,
    native_context_window,
)

# Per backend: the engine param holding the static memory fraction, and the
# one holding the context length.
_FRACTION_KEYS: dict[str, str] = {
    "sglang": "mem_fraction_static",
    "vllm": "gpu_memory_utilization",
}
_CONTEXT_KEYS: dict[str, str] = {
    "sglang": "context_length",
    "vllm": "max_model_len",
}

# Below this many full-context sequences of KV, a plan that fits is warned
# about: it starts, but serves little concurrency. Decides a warning only.
_LOW_KV_SEQUENCES = 8

# Upper bound of the smallest-tp search, so a degenerate demand terminates.
_MAX_TP = 1024


@dataclass(frozen=True, slots=True)
class MemoryDemand:
    """What one replica of a model needs, before it is split across GPUs.

    ``kv_layers``/``kv_heads``/``kv_head_dim`` are ``None`` together when the
    checkpoint's KV shape is unknown; ``context`` is ``None`` when neither the
    plan nor the checkpoint states one. Either way the KV term is dropped.
    ``fraction`` is ``None`` when the plan declares none.
    """

    weight_bytes: float
    fraction: float | None = None
    context: int | None = None
    kv_layers: int | None = None
    kv_heads: int | None = None
    kv_head_dim: int | None = None
    kv_dtype_bytes: float = 2.0

    def kv_bytes_per_token(self, tp: int) -> float | None:
        """KV bytes one token takes on one GPU at *tp*, or ``None`` if unknown.

        KV heads shard across tp, and are replicated once tp exceeds them.
        """
        if self.kv_layers is None or self.kv_heads is None or self.kv_head_dim is None:
            return None
        heads = max(1, self.kv_heads // tp)
        return self.kv_layers * heads * self.kv_head_dim * 2 * self.kv_dtype_bytes


@dataclass(frozen=True, slots=True)
class MemoryFit:
    """How a :class:`MemoryDemand` lands on one GPU at a given tp."""

    tp: int
    budget_bytes: float
    weight_bytes_per_gpu: float
    kv_bytes_per_token: float | None
    context: int | None
    fraction: float | None

    @property
    def kv_capacity_tokens(self) -> int | None:
        """Tokens of KV the budget holds after the weights; ``None`` if unknown."""
        if not self.kv_bytes_per_token:
            return None
        free = self.budget_bytes - self.weight_bytes_per_gpu
        return max(0, int(free // self.kv_bytes_per_token))

    @property
    def fits(self) -> bool:
        need = self.weight_bytes_per_gpu
        if self.kv_bytes_per_token and self.context:
            need += self.kv_bytes_per_token * self.context
        return need <= self.budget_bytes

    @property
    def low_room(self) -> bool:
        """It fits, but holds fewer than a few full-context sequences."""
        capacity = self.kv_capacity_tokens
        return (
            self.fits
            and self.fraction is not None
            and capacity is not None
            and self.context is not None
            and capacity < _LOW_KV_SEQUENCES * self.context
        )

    def describe(self) -> str:
        budget = (
            f"{self.fraction} × GPU = {_gb(self.budget_bytes)}"
            if self.fraction is not None
            else f"whole GPU (no memory fraction declared) = {_gb(self.budget_bytes)}"
        )
        parts = [
            f"tp={self.tp}: weights {_gb(self.weight_bytes_per_gpu)}/GPU",
            f"budget {budget}",
        ]
        capacity = self.kv_capacity_tokens
        if capacity is None:
            parts.append("KV not checked (shape unknown)")
        elif self.context is None:
            parts.append(f"KV room ~{capacity} tokens (no context declared)")
        else:
            parts.append(f"KV room ~{capacity} tokens for context {self.context}")
        return ", ".join(parts)


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


def memory_fit(demand: MemoryDemand, tp: int, gpu_memory_mib: int) -> MemoryFit:
    """Place *demand* on one GPU of *gpu_memory_mib* at *tp*."""
    gpu_bytes = gpu_memory_mib * 1024 * 1024
    fraction = demand.fraction
    return MemoryFit(
        tp=tp,
        budget_bytes=gpu_bytes * (fraction if fraction is not None else 1.0),
        weight_bytes_per_gpu=demand.weight_bytes / tp,
        kv_bytes_per_token=demand.kv_bytes_per_token(tp),
        context=demand.context,
        fraction=fraction,
    )


def size_tp(
    demand: MemoryDemand,
    *,
    gpu_count: int,
    gpu_memory_mib: int,
    stated_tp: int | None = None,
) -> MemoryFit:
    """The tp to serve *demand* with, and how it fits.

    A *stated_tp* is used as given; otherwise the smallest power of two that
    fits. Raises :class:`RuntimeError` when that tp does not fit or exceeds
    *gpu_count*, since the engine could not start; warns when it fits with
    little KV room.
    """
    if stated_tp is not None:
        fit = memory_fit(demand, stated_tp, gpu_memory_mib)
        if not fit.fits:
            raise RuntimeError(
                f"tp={stated_tp} does not fit the declared memory ({fit.describe()}). "
                f"The engine could not start; raise tp or the memory fraction, "
                f"or lower the context."
            )
    else:
        tp = 1
        fit = memory_fit(demand, tp, gpu_memory_mib)
        while not fit.fits and tp < _MAX_TP:
            tp *= 2
            fit = memory_fit(demand, tp, gpu_memory_mib)
        if not fit.fits:
            raise RuntimeError(
                f"The model does not fit the declared memory at any tp "
                f"({fit.describe()})."
            )
    if fit.tp > gpu_count:
        raise RuntimeError(
            f"Model requires tp={fit.tp} but only {gpu_count} GPUs detected "
            f"({fit.describe()})."
        )
    if fit.low_room:
        logger.warning(
            "The plan fits with little KV room, so concurrency will be low: {}",
            fit.describe(),
        )
    return fit


def memory_demand(
    *,
    checkpoint: str,
    identity: ModelIdentity,
    backend: str,
    params: dict[str, ParamValue],
) -> MemoryDemand:
    """Build the :class:`MemoryDemand` of *checkpoint* served with *params*.

    *params* are the resolved engine params (recipe merged with overrides).
    The context falls back to the checkpoint's native window, which is what
    the engine uses when none is declared.
    """
    weight_bytes = checkpoint_weight_bytes(checkpoint)
    if weight_bytes is None:
        weight_bytes = identity.param_billions * 1e9 * bytes_per_param(identity)

    fraction_key = _FRACTION_KEYS.get(backend)
    raw_fraction = params.get(fraction_key) if fraction_key else None
    fraction = (
        float(raw_fraction)
        if isinstance(raw_fraction, (int, float)) and not isinstance(raw_fraction, bool)
        else None
    )

    context_key = _CONTEXT_KEYS.get(backend)
    raw_context = params.get(context_key) if context_key else None
    context = (
        int(raw_context)
        if isinstance(raw_context, int) and not isinstance(raw_context, bool)
        else native_context_window(checkpoint)
    )

    kv_dtype = params.get("kv_cache_dtype")
    if isinstance(kv_dtype, str) and kv_dtype.startswith("fp8"):
        kv_dtype_bytes = 1.0
    else:
        dtype = params.get("dtype")
        kv_dtype_bytes = dtype_bytes(
            dtype if isinstance(dtype, str) else identity.dtype
        )

    shape = kv_shape(checkpoint)
    layers, heads, head_dim = shape if shape is not None else (None, None, None)
    return MemoryDemand(
        weight_bytes=weight_bytes,
        fraction=fraction,
        context=context,
        kv_layers=layers,
        kv_heads=heads,
        kv_head_dim=head_dim,
        kv_dtype_bytes=kv_dtype_bytes,
    )
