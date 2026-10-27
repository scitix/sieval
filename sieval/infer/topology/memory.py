"""
Size tensor parallelism from the memory a plan declares.

The engine's startup condition, checked before any GPU is allocated: one
GPU's share of the weights plus the KV cache of one sequence at the context
must fit in the memory fraction of that GPU. The fraction
(``mem_fraction_static`` / ``gpu_memory_utilization``) and the context
(``context_length`` / ``max_model_len``) come from the resolved engine
params, so a user override beats the recipe.

The fraction is treated as weights + KV on both backends. vLLM's also covers
activations and CUDA graphs, so a vLLM fit is slightly optimistic.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

from dataclasses import dataclass
from typing import Literal

from loguru import logger

from sieval.infer.config import ParamValue
from sieval.infer.introspect import (
    KvShape,
    ModelIdentity,
    bytes_per_param,
    checkpoint_weight_bytes,
    dtype_bytes,
    introspect_checkpoint_sync,
    kv_shape,
    native_context_window,
)

_FRACTION_KEYS: dict[str, str] = {
    "sglang": "mem_fraction_static",
    "vllm": "gpu_memory_utilization",
}
_CONTEXT_KEYS: dict[str, str] = {
    "sglang": "context_length",
    "vllm": "max_model_len",
}

# The fraction an engine uses when the plan states none. vLLM's is a constant;
# sglang computes one from the GPU and tp (about 0.80-0.84 on 40-80 GB GPUs up
# to tp=8), so its lower bound is assumed. An unknown backend gets the lowest.
_DEFAULT_FRACTIONS: dict[str, float] = {"vllm": 0.9, "sglang": 0.8}

# Fewer full-context sequences of KV room than this warns: the plan starts,
# but serves little concurrency.
_LOW_KV_SEQUENCES = 8

# Bounds the smallest-tp search when no tp fits.
_MAX_TP = 1024

# One text for the log line and the ``run --dry-run`` report.
_LOW_ROOM_WARNING = "The plan fits with little KV room, so concurrency will be low: {}"


def default_fraction(backend: str) -> float:
    """The memory fraction *backend* uses when none is declared."""
    return _DEFAULT_FRACTIONS.get(backend, min(_DEFAULT_FRACTIONS.values()))


@dataclass(frozen=True, slots=True)
class MemoryDemand:
    """One replica's memory, before it is split across GPUs.

    The KV term is dropped when *kv* or *context* is ``None``.
    """

    weight_bytes: float
    fraction: float
    fraction_declared: bool = True
    context: int | None = None
    kv: KvShape | None = None
    kv_dtype_bytes: float = 2.0

    def kv_bytes_per_token(self, tp: int) -> float | None:
        """KV bytes of one token on one GPU at *tp*; ``None`` if unknown.

        Sharded heads split across tp, rounding up: a GPU holds a whole head.
        """
        if self.kv is None:
            return None
        heads = -(-self.kv.heads // tp) if self.kv.sharded else self.kv.heads
        return self.kv.layers * heads * self.kv.width * self.kv_dtype_bytes


@dataclass(frozen=True, slots=True)
class MemoryFit:
    """How a :class:`MemoryDemand` lands on one GPU at *tp*."""

    tp: int
    budget_bytes: float
    weight_bytes_per_gpu: float
    kv_bytes_per_token: float | None
    context: int | None
    fraction: float
    fraction_declared: bool

    @property
    def kv_capacity_tokens(self) -> int | None:
        """Tokens of KV the budget holds beside the weights."""
        if not self.kv_bytes_per_token:
            return None
        free = self.budget_bytes - self.weight_bytes_per_gpu
        return max(0, int(free // self.kv_bytes_per_token))

    @property
    def full_context_sequences(self) -> int | None:
        """Full-context sequences the KV room holds."""
        capacity = self.kv_capacity_tokens
        if capacity is None or not self.context:
            return None
        return capacity // self.context

    @property
    def fits(self) -> bool:
        need = self.weight_bytes_per_gpu
        if self.kv_bytes_per_token and self.context:
            need += self.kv_bytes_per_token * self.context
        return need <= self.budget_bytes

    @property
    def low_room(self) -> bool:
        capacity = self.kv_capacity_tokens
        return (
            self.fits
            and capacity is not None
            and self.context is not None
            and capacity < _LOW_KV_SEQUENCES * self.context
        )

    def describe(self) -> str:
        source = "" if self.fraction_declared else " (engine default)"
        parts = [
            f"tp={self.tp}: weights {_gb(self.weight_bytes_per_gpu)}/GPU",
            f"budget {self.fraction}{source} × GPU = {_gb(self.budget_bytes)}",
        ]
        capacity = self.kv_capacity_tokens
        if capacity is None:
            parts.append("KV not checked (shape unknown)")
        elif self.context is None:
            parts.append(f"KV room ~{capacity} tokens (no context declared)")
        else:
            parts.append(
                f"KV room ~{capacity} tokens for context {self.context} "
                f"({self.full_context_sequences} full-context sequences)"
            )
        return ", ".join(parts)

    def low_room_warning(self) -> str | None:
        """The warning :func:`size_tp` logs for this fit; ``None`` when roomy."""
        return _LOW_ROOM_WARNING.format(self.describe()) if self.low_room else None

    def to_json(self, tp_source: Literal["stated", "sized"]) -> dict[str, object]:
        """This fit in the ``run --dry-run`` report. *tp_source* is ``stated``
        when the plan set tp, ``sized`` when :func:`size_tp` chose it."""
        return {
            "tp": self.tp,
            "tp_source": tp_source,
            "fraction": self.fraction,
            "fraction_declared": self.fraction_declared,
            "context": self.context,
            "kv_capacity_tokens": self.kv_capacity_tokens,
            "full_context_sequences": self.full_context_sequences,
            "fits": self.fits,
            "low_room": self.low_room,
            "detail": self.describe(),
        }


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


def memory_fit(demand: MemoryDemand, tp: int, gpu_memory_mib: int) -> MemoryFit:
    """Place *demand* on one GPU of *gpu_memory_mib* at *tp*."""
    return MemoryFit(
        tp=tp,
        budget_bytes=gpu_memory_mib * 1024 * 1024 * demand.fraction,
        weight_bytes_per_gpu=demand.weight_bytes / tp,
        kv_bytes_per_token=demand.kv_bytes_per_token(tp),
        context=demand.context,
        fraction=demand.fraction,
        fraction_declared=demand.fraction_declared,
    )


def size_tp(
    demand: MemoryDemand,
    *,
    gpu_count: int,
    gpu_memory_mib: int,
    stated_tp: int | None = None,
) -> MemoryFit:
    """The tp to serve *demand* with: *stated_tp*, else the smallest power of
    two that fits.

    Raises :class:`RuntimeError` when that tp does not fit or exceeds
    *gpu_count*; warns when it fits with little KV room.
    """
    if stated_tp is not None:
        if stated_tp < 1:
            raise RuntimeError(f"tp must be at least 1, got {stated_tp}.")
        fit = memory_fit(demand, stated_tp, gpu_memory_mib)
        if not fit.fits:
            raise RuntimeError(
                f"tp={stated_tp} does not fit the declared memory ({fit.describe()}). "
                f"Raise tp or the memory fraction, or lower the context."
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
    warning = fit.low_room_warning()
    if warning is not None:
        logger.warning("{}", warning)
    return fit


def memory_demand(
    *,
    checkpoint: str,
    identity: ModelIdentity,
    backend: str,
    params: dict[str, ParamValue],
) -> MemoryDemand:
    """The :class:`MemoryDemand` of *checkpoint* served with the resolved
    engine *params*.

    Undeclared, the fraction is the engine's default and the context the
    checkpoint's native window, as the engine would use.
    """
    weight_bytes = checkpoint_weight_bytes(checkpoint)
    if weight_bytes is None:
        weight_bytes = identity.param_billions * 1e9 * bytes_per_param(identity)

    fraction_key = _FRACTION_KEYS.get(backend)
    raw_fraction = params.get(fraction_key) if fraction_key else None
    if isinstance(raw_fraction, (int, float)) and not isinstance(raw_fraction, bool):
        fraction, declared = float(raw_fraction), True
    else:
        fraction, declared = default_fraction(backend), False

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

    return MemoryDemand(
        weight_bytes=weight_bytes,
        fraction=fraction,
        fraction_declared=declared,
        context=context,
        kv=kv_shape(checkpoint),
        kv_dtype_bytes=kv_dtype_bytes,
    )


def engine_fit(
    *,
    checkpoint: str,
    backend: str,
    tp: int,
    params: dict[str, ParamValue],
    gpu_memory_mib: int,
) -> MemoryFit | None:
    """How an engine launched with *params* at *tp* lands on one GPU.

    :func:`size_tp`'s check on the params the engine launches with, which a
    serving group's window or headroom may have changed since sizing.
    ``None`` when *checkpoint* cannot be introspected.
    """
    try:
        identity = introspect_checkpoint_sync(checkpoint)
    except (FileNotFoundError, ValueError):
        return None
    demand = memory_demand(
        checkpoint=checkpoint, identity=identity, backend=backend, params=params
    )
    return memory_fit(demand, tp, gpu_memory_mib)
