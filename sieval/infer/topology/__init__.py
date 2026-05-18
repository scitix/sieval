"""
Deployment topology: models, resolver, validation.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from sieval.infer.topology.models import (
    CP_KEYS,
    DP_KEYS,
    DPA_KEYS,
    EP_KEYS,
    PP_KEYS,
    TOPO_KEYS,
    TP_KEYS,
    DeploymentCapabilities,
    DeploymentPlan,
    DeviceGroup,
    HardwareEnv,
    ModelProfile,
    ParallelTopology,
    ResolveResult,
    ResolveStep,
    RoleAssignment,
    ScalingPolicy,
    ServiceBinding,
    UserHints,
    WellKnownRole,
)
from sieval.infer.topology.resolver import precision_key
from sieval.infer.topology.validator import validate_plan

__all__ = [
    "CP_KEYS",
    "DP_KEYS",
    "DPA_KEYS",
    "EP_KEYS",
    "PP_KEYS",
    "TOPO_KEYS",
    "TP_KEYS",
    "DeploymentCapabilities",
    "DeploymentPlan",
    "DeviceGroup",
    "HardwareEnv",
    "ModelProfile",
    "ParallelTopology",
    "ResolveResult",
    "ResolveStep",
    "RoleAssignment",
    "ScalingPolicy",
    "ServiceBinding",
    "UserHints",
    "WellKnownRole",
    "precision_key",
    "validate_plan",
]
