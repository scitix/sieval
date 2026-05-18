"""
Infer configuration data models and enums.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TypedDict

# Infer param values are YAML scalars
ParamValue = str | int | float | bool
# Metadata values used across InferHandle and InferConfig
MetadataValue = str | int | list[str]


class InferConfigDict(TypedDict, total=False):
    """YAML-level infer configuration for a model."""

    backend: str  # backend name (vllm, sglang, siflow)
    recipe: str  # reference recipe name
    checkpoint: str  # model weights path
    overrides: dict[str, ParamValue]  # user overrides on top of recipe
    env: dict[str, str]  # environment variables passed to the engine process
    # NOTE: user env overrides translator-set env (e.g. CUDA_VISIBLE_DEVICES).
    # Collisions are logged as warnings at injection time.


class InferMetaDict(TypedDict, total=False):
    """User-declared inference environment metadata for audit."""

    framework: str  # e.g. "vllm==0.6.0"
    dtype: str  # e.g. "bfloat16"
    tp: int  # tensor parallelism
    gpu: str  # GPU model and count
    image: str  # container image (e.g. "registry.example.com/sglang-rl:v3.2")


class InferPhase(Enum):
    """Coarse lifecycle phase — monotonically advances, never goes backwards."""

    PENDING = "pending"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class InferCondition:
    """A single condition (boolean status + human-readable reason)."""

    status: bool
    reason: str = ""


@dataclass
class InferHandle:
    """Handle returned by deployer, identifies a running infer instance."""

    backend: str  # backend that created this handle
    endpoint: str | None  # API endpoint (available when ready)
    handle_id: str  # backend-internal identifier (PID, job ID, etc.)
    metadata: dict[str, MetadataValue] = field(default_factory=dict)


@dataclass
class InferCommand:
    """Structured output: launch command + health check URL.

    Kept for backward compatibility; new code uses BackendCommand.
    """

    launch_cmd: list[str]
    # URL to probe for readiness (e.g., "http://localhost:8000/v1/health")
    health_url: str


@dataclass
class InferEnv:
    """Runtime environment information collected by a backend.

    Covers framework, driver, GPU hardware, and network — the key dimensions
    that affect inference reproducibility and diagnostics.
    Modeled after vLLM's collect_env and sichek's hardware checks.

    Fields:
        framework: Framework name + version (e.g., "vllm==0.8.3")
        image: Container image if running inside one (e.g.,
            "registry.example.com/sglang-rl:v3.2"); empty for bare-metal.
        cuda_version: CUDA runtime version (e.g., "12.4")
        driver_version: NVIDIA driver version (e.g., "550.54.15")
        gpu_model: GPU model name (e.g., "NVIDIA A100-SXM4-80GB")
        gpu_count: Number of GPUs visible to the process
        gpu_topo: GPU interconnect topology (e.g., "NVLink", "PCIe")
        python_version: Python version (e.g., "3.12.3")
        extra: Catch-all for backend-specific info not covered above
    """

    framework: str = ""
    image: str = ""
    cuda_version: str = ""
    driver_version: str = ""
    gpu_model: str = ""
    gpu_count: int = 0
    gpu_topo: str = ""
    python_version: str = ""
    extra: dict[str, str] = field(default_factory=dict)


# Legacy: InferConfig kept for backward compatibility during migration.
# New code should use DeploymentPlan instead.
@dataclass
class InferConfig:
    """Resolved infer configuration ready to pass to a backend.

    Deprecated: use DeploymentPlan from sieval.infer.topology instead.
    """

    backend: str
    checkpoint: str
    params: dict[str, ParamValue] = field(default_factory=dict)
    metadata: dict[str, MetadataValue] = field(default_factory=dict)
