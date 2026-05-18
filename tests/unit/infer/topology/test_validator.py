"""
Unit tests for topology validator.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from sieval.infer.topology.models import (
    DeploymentPlan,
    DeviceGroup,
    HardwareEnv,
    ParallelTopology,
    RoleAssignment,
)
from sieval.infer.topology.validator import validate_plan


class TestValidatePlan:
    """Tests for validate_plan() — centralized plan validation."""

    def _make_plan(
        self,
        tp: int = 4,
        dp: int = 1,
        gpu_count: int = 8,
        roles: tuple[str, ...] = ("full",),
    ) -> DeploymentPlan:
        assignments = tuple(
            RoleAssignment(
                role=role,
                devices=DeviceGroup(count=gpu_count),
                topology=ParallelTopology(tp=tp, dp=dp),
            )
            for role in roles
        )
        return DeploymentPlan(
            checkpoint="/models/test",
            backend="sglang",
            assignments=assignments,
        )

    def test_valid_plan_no_hw(self):
        plan = self._make_plan()
        assert validate_plan(plan) == []

    def test_valid_plan_with_hw(self):
        plan = self._make_plan(tp=4, dp=2, gpu_count=8)
        hw = HardwareEnv(gpu_count=8, gpu_model="H100", gpu_memory_mib=81920)
        assert validate_plan(plan, hw) == []

    def test_empty_assignments(self):
        plan = DeploymentPlan(
            checkpoint="/models/test",
            backend="sglang",
            assignments=(),
        )
        errors = validate_plan(plan)
        assert any("at least one assignment" in e for e in errors)

    def test_hw_capacity_exceeded(self):
        plan = self._make_plan(tp=4, dp=2, gpu_count=8)
        hw = HardwareEnv(gpu_count=4, gpu_model="H100", gpu_memory_mib=81920)
        errors = validate_plan(plan, hw)
        assert any("requires 8 total GPUs" in e for e in errors)

    def test_pd_roles_paired(self):
        plan = self._make_plan(roles=("prefill", "decode"))
        assert validate_plan(plan) == []

    def test_pd_roles_unpaired(self):
        plan = self._make_plan(roles=("prefill",))
        errors = validate_plan(plan)
        assert any("matching decode role" in e for e in errors)

    def test_empty_checkpoint(self):
        plan = DeploymentPlan(
            checkpoint="",
            backend="sglang",
            assignments=(
                RoleAssignment(
                    role="full",
                    devices=DeviceGroup(count=4),
                    topology=ParallelTopology(tp=4),
                ),
            ),
        )
        errors = validate_plan(plan)
        assert any("checkpoint must not be empty" in e for e in errors)

    def test_empty_backend(self):
        plan = DeploymentPlan(
            checkpoint="/models/test",
            backend="",
            assignments=(
                RoleAssignment(
                    role="full",
                    devices=DeviceGroup(count=4),
                    topology=ParallelTopology(tp=4),
                ),
            ),
        )
        errors = validate_plan(plan)
        assert any("backend must not be empty" in e for e in errors)

    def test_multi_node_pp_hint(self):
        """Multi-node without PP for a role needing > gpus_per_node → hint."""
        plan = DeploymentPlan(
            checkpoint="/models/test",
            backend="vllm",
            assignments=(
                RoleAssignment(
                    role="full",
                    devices=DeviceGroup(count=16),
                    topology=ParallelTopology(tp=16),
                ),
            ),
        )
        hw = HardwareEnv(
            gpu_count=16,
            gpu_model="H100",
            gpu_memory_mib=81920,
            nodes=2,
            gpus_per_node=8,
        )
        errors = validate_plan(plan, hw)
        assert any("PP stage needs" in e for e in errors)

    def test_multi_node_with_pp_ok(self):
        """Multi-node with PP fitting per node → no error."""
        plan = DeploymentPlan(
            checkpoint="/models/test",
            backend="vllm",
            assignments=(
                RoleAssignment(
                    role="full",
                    devices=DeviceGroup(count=16),
                    topology=ParallelTopology(tp=4, dp=1, pp=2),
                ),
            ),
        )
        hw = HardwareEnv(
            gpu_count=16,
            gpu_model="H100",
            gpu_memory_mib=81920,
            nodes=2,
            gpus_per_node=8,
        )
        # tp*dp=4 per PP stage ≤ 8 gpus_per_node, and pp=2 so no warning
        errors = validate_plan(plan, hw)
        # Should not flag the multi-node hint since pp > 1
        assert not any("PP stage" in e for e in errors)

    def test_propagates_topology_errors(self):
        """Invalid topology in assignment is reported."""
        plan = DeploymentPlan(
            checkpoint="/models/test",
            backend="sglang",
            assignments=(
                RoleAssignment(
                    role="full",
                    devices=DeviceGroup(count=8),
                    topology=ParallelTopology(tp=0, dp=1),
                ),
            ),
        )
        errors = validate_plan(plan)
        assert any("tp must be >= 1" in e for e in errors)

    def test_multi_node_replicas_exceed_cluster(self):
        """Replicas that collectively exceed cluster capacity are flagged."""
        plan = DeploymentPlan(
            checkpoint="/models/test",
            backend="vllm",
            assignments=(
                RoleAssignment(
                    role="full",
                    devices=DeviceGroup(count=8),
                    topology=ParallelTopology(tp=8),
                    replicas=3,
                ),
            ),
        )
        hw = HardwareEnv(
            gpu_count=24,
            gpu_model="H100",
            gpu_memory_mib=81920,
            nodes=2,
            gpus_per_node=8,
        )
        errors = validate_plan(plan, hw)
        assert any("replicas" in e for e in errors)

    def test_multi_node_pp_but_stage_exceeds_node(self):
        """PP > 1 but each PP stage (tp*dp) still exceeds gpus_per_node → error."""
        plan = DeploymentPlan(
            checkpoint="/models/test",
            backend="sglang",
            assignments=(
                RoleAssignment(
                    role="full",
                    devices=DeviceGroup(count=32),
                    topology=ParallelTopology(tp=16, dp=1, pp=2),
                ),
            ),
        )
        hw = HardwareEnv(
            gpu_count=32,
            gpu_model="H100",
            gpu_memory_mib=81920,
            nodes=4,
            gpus_per_node=8,
        )
        errors = validate_plan(plan, hw)
        # per-stage = tp*dp = 16 > 8 gpus_per_node
        assert any("PP stage" in e for e in errors)
