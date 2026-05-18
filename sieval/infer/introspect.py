"""
Introspect a HuggingFace checkpoint to extract model identity.

Reads config.json from a local model directory to determine architecture
family and approximate parameter count, which are used for recipe matching.

Two strategies for identifying model size:
  1. Path-based: extract size label (e.g. "8B", "72B") from directory name
     or _name_or_path field — fast and reliable for standard HF checkpoints.
  2. Estimation: compute approximate param count from config.json dimensions
     — fallback for renamed/custom checkpoints.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
import orjson
from loguru import logger

# Architecture class → recipe family mapping.
# Keys are regex patterns matched against HF config.json "architectures" entries.
#
# Note: Qwen2.5 shares architecture class Qwen2ForCausalLM with Qwen2.
# Disambiguation is handled in _refine_family_from_name() using _name_or_path.
_ARCHITECTURE_MAP: dict[str, str] = {
    r"Qwen3": "qwen3",
    r"Qwen2": "qwen2",
    r"GptOss": "gpt-oss",
}

# Regex to extract size label from model name/path.
# Matches patterns like "8B", "72b", "0.5B", "1.5B", "4b"
_SIZE_LABEL_RE = re.compile(r"(?:^|[-_/])(\d+(?:\.\d+)?)[Bb](?:[-_/]|$)")


@dataclass(frozen=True, slots=True)
class QuantizationInfo:
    """Quantization metadata extracted from config.json, preserved for audit."""

    quant_method: str  # "awq", "gptq", "compressed-tensors", "fp8", ...
    bits: int  # effective bits (unified extraction)
    raw_config: dict[str, Any]  # original quantization_config, kept intact


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Introspected model identity from config.json."""

    architecture: str  # raw HF architecture class, e.g. "Qwen3ForCausalLM"
    family: str  # normalized family name, e.g. "qwen3"
    param_billions: float  # approximate parameter count in billions
    dtype: str  # recommended dtype from config, e.g. "bfloat16"
    quantization: QuantizationInfo | None = None


def _extract_size_from_name(name: str) -> float | None:
    """Extract parameter count (in billions) from a model name or path.

    Matches patterns like "Qwen3-8B", "Qwen3-0.5B", "Qwen2.5-72B".
    Returns None if no size label found.
    """
    match = _SIZE_LABEL_RE.search(name)
    if match:
        return float(match.group(1))
    return None


def _estimate_params_billions(config: dict) -> float:
    """Estimate parameter count from HF config.json fields.

    Uses the standard dense-transformer parameter formula (see
    https://blog.eleuther.ai/transformer-math/ §"Parameter Counting").
    This is a rough estimate — good enough for recipe size-bucket
    matching, not for precise memory planning.  For exact counts,
    HuggingFace's ``accelerate estimate-memory`` loads the model on a
    ``meta`` device — but that requires the transformers library and
    the full model definition, which we intentionally avoid here.

    Used as fallback when the size label cannot be extracted from the
    model name.

    MoE width: prefer ``moe_intermediate_size`` when present (Qwen3-MoE
    style); fall back to ``intermediate_size`` (gpt-oss style).

    Note: counts output embeddings even if tied (overestimates by ~0.4B
    for large-vocab models), but this doesn't affect bucket matching.
    """
    vocab = config.get("vocab_size", 0)
    hidden = config.get("hidden_size", 0)
    n_layers = config.get("num_hidden_layers", 0)
    n_heads = config.get("num_attention_heads", 1)
    n_kv_heads = config.get("num_key_value_heads", n_heads)
    dense_intermediate = config.get("intermediate_size", 4 * hidden)
    num_experts = config.get("num_local_experts") or config.get("num_experts") or 1
    moe_intermediate = config.get("moe_intermediate_size")

    if not all([vocab, hidden, n_layers]):
        return 0.0

    head_dim = hidden // n_heads if n_heads else hidden
    kv_dim = n_kv_heads * head_dim

    # Embedding layers
    # input_embed + output_embed (often weight-tied, counted twice for simplicity)
    embed_params = vocab * hidden * 2

    # Attention projections: Q(h×h) + K(h×kv) + V(h×kv) + O(h×h)
    attn_params = hidden * hidden + hidden * kv_dim + hidden * kv_dim + hidden * hidden

    # MLP (SwiGLU): gate_proj + up_proj + down_proj, each hidden × width.
    # MoE layers add a small router term (hidden × num_experts).
    is_moe = num_experts > 1
    expert_width = moe_intermediate if moe_intermediate else dense_intermediate
    sparse_mlp = hidden * expert_width * 3 * num_experts + hidden * num_experts
    dense_mlp = hidden * dense_intermediate * 3

    if is_moe:
        # HF rule: layer is dense iff in mlp_only_layers OR
        # (layer_idx + 1) % decoder_sparse_step != 0.
        dense_set = set(config.get("mlp_only_layers") or [])
        sparse_step = config.get("decoder_sparse_step", 1) or 1
        n_dense = sum(
            1 for i in range(n_layers) if i in dense_set or (i + 1) % sparse_step != 0
        )
        n_sparse = n_layers - n_dense
        total_mlp = n_sparse * sparse_mlp + n_dense * dense_mlp
    else:
        total_mlp = n_layers * dense_mlp

    total = embed_params + n_layers * attn_params + total_mlp
    return total / 1e9


def _match_family(architecture: str) -> str:
    """Map an HF architecture class name to a recipe family."""
    for pattern, family in _ARCHITECTURE_MAP.items():
        if re.search(pattern, architecture):
            return family
    # Fallback: lowercase, strip "ForCausalLM" etc.
    cleaned = re.sub(r"For\w+$", "", architecture).lower()
    return cleaned


# Pattern to detect "Qwen2.5" in model name/path (case-insensitive).
# Qwen2.5 and Qwen2 share the same architecture class (Qwen2ForCausalLM),
# so we need the model name to disambiguate.
_QWEN2_5_NAME_RE = re.compile(r"qwen2[\.\-_]5", re.IGNORECASE)


def _refine_family_from_name(family: str, checkpoint: str, config: dict) -> str:
    """Refine family using model name when architecture alone is ambiguous.

    Qwen2.5 uses Qwen2ForCausalLM, so architecture-based detection returns
    "qwen2".  This checks _name_or_path and checkpoint path for "Qwen2.5"
    to upgrade to "qwen2.5".
    """
    if family != "qwen2":
        return family
    name_or_path = config.get("_name_or_path", "")
    if _QWEN2_5_NAME_RE.search(name_or_path):
        logger.debug(
            "Refined family qwen2 → qwen2.5 from _name_or_path: {}", name_or_path
        )
        return "qwen2.5"
    if _QWEN2_5_NAME_RE.search(checkpoint):
        logger.debug(
            "Refined family qwen2 → qwen2.5 from checkpoint path: {}", checkpoint
        )
        return "qwen2.5"
    return family


def _resolve_param_billions(checkpoint: str, config: dict) -> float:
    """Resolve parameter count using name extraction first, estimation as fallback.

    Priority:
      1. _name_or_path field in config.json (set by transformers at save time)
      2. Directory name of the checkpoint path
      3. Estimation from config.json dimensions
    """
    # Try _name_or_path from config (often contains HF model ID like "Qwen/Qwen3-8B")
    name_or_path = config.get("_name_or_path", "")
    if name_or_path:
        size = _extract_size_from_name(name_or_path)
        if size is not None:
            logger.debug(
                "Extracted size {:.1f}B from _name_or_path: {}",
                size,
                name_or_path,
            )
            return size

    # Try directory name (e.g. /models/Qwen3-8B → "Qwen3-8B")
    dir_name = Path(checkpoint).name
    size = _extract_size_from_name(dir_name)
    if size is not None:
        logger.debug("Extracted size {:.1f}B from directory name: {}", size, dir_name)
        return size

    # Fallback: estimate from config dimensions
    estimated = _estimate_params_billions(config)
    if estimated > 0:
        logger.debug("Estimated size {:.1f}B from config dimensions", estimated)
    return estimated


def _extract_bits(quant_config: dict[str, Any]) -> int:
    """Extract effective weight bits from a HuggingFace quantization_config.

    Different quant methods store bits in different fields.  This unifies
    extraction into a single int used by the TP formula.
    """
    method = quant_config.get("quant_method", "")

    # AWQ, GPTQ, AutoRound — top-level "bits" field
    if "bits" in quant_config:
        return int(quant_config["bits"])
    # HQQ — uses "nbits" instead
    if "nbits" in quant_config:
        return int(quant_config["nbits"])

    # compressed-tensors — dig into config_groups
    if method == "compressed-tensors":
        groups = quant_config.get("config_groups", {})
        for group in groups.values():
            if isinstance(group, dict):
                weights = group.get("weights")
                if isinstance(weights, dict) and "num_bits" in weights:
                    return int(weights["num_bits"])

    # BitsAndBytes — boolean flags
    if method == "bitsandbytes":
        if quant_config.get("load_in_4bit"):
            return 4
        if quant_config.get("load_in_8bit"):
            return 8

    # fbgemm_fp8, fp8 (DeepSeek-style) — always 8-bit
    if method in ("fbgemm_fp8", "fp8"):
        return 8

    # MXFP4 is ~4.25 effective bpw (E2M1 + shared E8M0 scale per block).
    # Approximating as 4 underestimates memory by ~6% — acceptable for
    # bucket-level TP decisions.
    if method == "mxfp4":
        return 4

    # Unknown method — conservative fallback
    logger.warning(
        "Unknown quant_method {!r}, assuming 8-bit weights",
        method,
    )
    return 8


def _extract_quantization(config: dict[str, Any]) -> QuantizationInfo | None:
    """Extract QuantizationInfo from a HuggingFace config.json dict.

    Returns None if no quantization_config is present.
    """
    qc = config.get("quantization_config")
    if not isinstance(qc, dict):
        return None
    method = qc.get("quant_method", "")
    if not method:
        return None
    bits = _extract_bits(qc)
    return QuantizationInfo(quant_method=str(method), bits=bits, raw_config=dict(qc))


def extract_moe_info(config: dict[str, Any]) -> tuple[bool, int | None, int]:
    """Extract MoE structure from HF config.json.

    Returns:
        (is_moe, num_experts, num_layers)

    ``num_local_experts`` is the standard HuggingFace key for MoE models
    (Mixtral, Qwen3-MoE, etc.). ``num_experts`` is a fallback for some
    non-standard configs. A model with exactly 1 expert is not considered
    MoE (degenerate case).
    """
    num_layers = config.get("num_hidden_layers", 0)
    num_experts = config.get("num_local_experts") or config.get("num_experts")
    is_moe = num_experts is not None and num_experts > 1
    return is_moe, num_experts if is_moe else None, num_layers


_DTYPE_BYTES: dict[str, float] = {
    "float64": 8.0,
    "float32": 4.0,
    "bfloat16": 2.0,
    "float16": 2.0,
}


def bytes_per_param(identity: ModelIdentity) -> float:
    """Compute bytes per parameter based on quantization or dtype.

    When quantized, uses QuantizationInfo.bits.  Otherwise falls back
    to the model's torch_dtype.  Default is 2.0 (float16/bfloat16).
    """
    if identity.quantization is not None:
        return identity.quantization.bits / 8
    return _DTYPE_BYTES.get(identity.dtype, 2.0)


def _read_checkpoint_config(checkpoint: str) -> tuple[dict[str, Any], str]:
    """Read and validate config.json from a checkpoint directory.

    Returns (config_dict, config_path_str).
    """
    config_path = Path(checkpoint) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.json found at {config_path}. "
            f"Is {checkpoint!r} a valid HuggingFace model directory?"
        )

    raw = config_path.read_bytes()
    config = orjson.loads(raw)

    architectures = config.get("architectures", [])
    if not architectures:
        raise ValueError(f"config.json at {config_path} has no 'architectures' field")

    return config, str(config_path)


def _identity_from_config(checkpoint: str, config: dict[str, Any]) -> ModelIdentity:
    """Build ModelIdentity from an already-read config dict."""
    architectures = config.get("architectures", [])
    architecture = architectures[0]
    family = _match_family(architecture)
    family = _refine_family_from_name(family, checkpoint, config)
    param_b = _resolve_param_billions(checkpoint, config)
    dtype = config.get("torch_dtype", "float16")
    quantization = _extract_quantization(config)

    logger.debug(
        "Introspected {}: family={}, ~{:.1f}B params, dtype={}",
        architecture,
        family,
        param_b,
        dtype,
    )

    return ModelIdentity(
        architecture=architecture,
        family=family,
        param_billions=param_b,
        dtype=dtype,
        quantization=quantization,
    )


def introspect_checkpoint_sync(checkpoint: str) -> ModelIdentity:
    """Read config.json from a local checkpoint and return ModelIdentity.

    Synchronous version for use in non-async contexts (e.g. YAML config
    resolution in the CLI layer).

    Args:
        checkpoint: Path to a local HuggingFace model directory.

    Raises:
        FileNotFoundError: If config.json does not exist.
        ValueError: If config.json is missing required fields.
    """
    config, _ = _read_checkpoint_config(checkpoint)
    return _identity_from_config(checkpoint, config)


def introspect_checkpoint_with_config(
    checkpoint: str,
) -> tuple[ModelIdentity, dict[str, Any]]:
    """Like :func:`introspect_checkpoint_sync` but also returns the raw config.

    Avoids re-reading config.json in callers that need both identity and
    the raw config dict (e.g. for MoE detection).
    """
    config, _ = _read_checkpoint_config(checkpoint)
    identity = _identity_from_config(checkpoint, config)
    return identity, config


async def introspect_checkpoint(checkpoint: str) -> ModelIdentity:
    """Read config.json from a local checkpoint and return ModelIdentity.

    Thin async wrapper around :func:`introspect_checkpoint_sync`.
    Reading a single small JSON file is fast enough to run inline.

    Args:
        checkpoint: Path to a local HuggingFace model directory.

    Raises:
        FileNotFoundError: If config.json does not exist.
        ValueError: If config.json is missing required fields.
    """
    return introspect_checkpoint_sync(checkpoint)


@dataclass(frozen=True, slots=True)
class GPUInfo:
    """Local GPU hardware information."""

    model: str  # e.g. "NVIDIA A100-SXM4-80GB"
    count: int  # number of GPUs
    memory_mib: int  # per-GPU memory in MiB


async def detect_local_gpu() -> GPUInfo | None:
    """Detect local GPU hardware via nvidia-smi (native async).

    Uses ``anyio.open_process`` to run nvidia-smi without blocking the
    event loop — no worker-thread wrapper needed.
    Returns None if nvidia-smi is not available or no GPUs found.
    """
    try:
        process = await anyio.open_process(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        logger.debug("nvidia-smi not found")
        return None

    try:
        with anyio.fail_after(10):
            chunks: list[bytes] = []
            assert process.stdout is not None  # guaranteed by stdout=PIPE
            async for chunk in process.stdout:
                chunks.append(chunk)
            stdout_bytes = b"".join(chunks)
            await process.wait()
    except TimeoutError:
        process.kill()
        await process.wait()
        logger.warning("nvidia-smi timed out")
        return None

    if process.returncode != 0:
        logger.debug("nvidia-smi failed (exit code {})", process.returncode)
        return None

    stdout = stdout_bytes.decode()
    lines = [line.strip() for line in stdout.strip().split("\n") if line.strip()]
    if not lines:
        return None

    # Parse first GPU line for model and memory
    parts = lines[0].split(", ")
    if len(parts) < 2:
        return None

    gpu_model = parts[0].strip()
    try:
        memory_mib = int(float(parts[1].strip()))
    except ValueError:
        memory_mib = 0

    info = GPUInfo(model=gpu_model, count=len(lines), memory_mib=memory_mib)
    logger.debug(
        "Detected GPU: {} x{} ({} MiB each)",
        info.model,
        info.count,
        info.memory_mib,
    )
    return info
