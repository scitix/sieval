"""
Inference service orchestration.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from sieval.infer.config import (
    InferCommand,
    InferCondition,
    InferConfig,
    InferConfigDict,
    InferEnv,
    InferHandle,
    InferMetaDict,
    InferPhase,
    MetadataValue,
    ParamValue,
)
from sieval.infer.deployer import DeployError, DeployTimeoutError, LocalDeployer
from sieval.infer.introspect import QuantizationInfo, bytes_per_param
from sieval.infer.recipes import Recipe
from sieval.infer.topology import (
    DeploymentCapabilities,
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
from sieval.infer.topology.resolver import (
    auto_resolve_plan,
    compute_tp,
    precision_key,
    resolve_simple,
)

__all__ = [
    # Config types (kept for compatibility)
    "InferCommand",
    "InferConfig",
    "InferConfigDict",
    "InferEnv",
    "InferHandle",
    "InferMetaDict",
    "InferCondition",
    "InferPhase",
    "MetadataValue",
    "ParamValue",
    # Deployer
    "DeployError",
    "DeployTimeoutError",
    "LocalDeployer",
    # Introspect
    "QuantizationInfo",
    "bytes_per_param",
    # Recipes
    "Recipe",
    # Topology models
    "DeploymentCapabilities",
    "DeploymentPlan",
    "DeviceGroup",
    "HardwareEnv",
    "ModelProfile",
    "ParallelTopology",
    "ResolveResult",
    "ResolveStep",
    "RoleAssignment",
    "UserHints",
    "WellKnownRole",
    # Resolver
    "auto_resolve_plan",
    "compute_tp",
    "precision_key",
    "resolve_simple",
]
