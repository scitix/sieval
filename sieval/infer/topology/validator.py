"""
Deployment plan validation.

Centralized validation for DeploymentPlan before translation and deployment.
Collects all errors rather than failing on the first one.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from sieval.infer.topology.models import DeploymentPlan, HardwareEnv


def validate_plan(
    plan: DeploymentPlan,
    hw: HardwareEnv | None = None,
) -> list[str]:
    """Validate a DeploymentPlan, returning all errors found.

    Combines structural validation (from plan.validate()) with
    optional hardware capacity checks.

    Args:
        plan: The plan to validate.
        hw: Optional hardware environment for capacity checks.

    Returns:
        List of error strings (empty = valid).
    """
    errors = plan.validate()

    # Hardware capacity check
    if hw is not None:
        total_needed = plan.total_gpus
        if total_needed > hw.gpu_count:
            errors.append(
                f"Plan requires {total_needed} total GPUs "
                f"but hardware only has {hw.gpu_count}"
            )

        # Multi-node checks
        if hw.nodes > 1 and hw.gpus_per_node is not None:
            total_node_gpus = hw.nodes * hw.gpus_per_node
            for a in plan.assignments:
                # Check per-PP-stage GPU count fits in a single node.
                # Each PP stage uses tp × dp GPUs.
                per_stage_gpus = a.topology.tp * a.topology.dp
                if per_stage_gpus > hw.gpus_per_node:
                    errors.append(
                        f"Role {a.role}: each PP stage needs {per_stage_gpus} GPUs "
                        f"(tp={a.topology.tp} x dp={a.topology.dp}) "
                        f"but only {hw.gpus_per_node} GPUs per node"
                    )
                role_total = a.topology.gpu_count * a.replicas
                if role_total > total_node_gpus:
                    errors.append(
                        f"Role {a.role}: needs {role_total} GPUs total "
                        f"({a.topology.gpu_count} x {a.replicas} replicas) "
                        f"but cluster only has {total_node_gpus} "
                        f"({hw.nodes} nodes x {hw.gpus_per_node} GPUs)"
                    )

    # Checkpoint / backend presence
    if not plan.checkpoint:
        errors.append("DeploymentPlan.checkpoint must not be empty")
    if not plan.backend:
        errors.append("DeploymentPlan.backend must not be empty")

    return errors
