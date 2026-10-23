"""
Inference service orchestration.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from sieval.core.models import (
    Deployment,
    DeploymentPlanProjection,
    DeploymentTopology,
    Engine,
    ServingFacts,
)
from sieval.infer.config import (
    InferCommand,
    InferCondition,
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
    deployment_plan_fingerprint,
    deployment_plan_projection,
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
    # Realized deployment types
    "Deployment",
    "DeploymentPlanProjection",
    "DeploymentTopology",
    "Engine",
    "ServingFacts",
    # Introspect
    "QuantizationInfo",
    "bytes_per_param",
    # Recipes
    "Recipe",
    # Topology models
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
    "deployment_plan_fingerprint",
    "deployment_plan_projection",
    # Resolver
    "auto_resolve_plan",
    "compute_tp",
    "precision_key",
    "resolve_simple",
]
