"""
Unit tests for deployment topology data models.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import pytest

from sieval.infer.topology.models import (
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

# ---------- ParallelTopology ----------


class TestParallelTopology:
    def test_gpu_count_tp_dp_pp(self):
        t = ParallelTopology(tp=4, dp=2, pp=1)
        assert t.gpu_count == 8

    def test_gpu_count_ep_does_not_add(self):
        """EP reuses TP x DP devices, should not increase gpu_count."""
        t = ParallelTopology(tp=4, dp=2, pp=1, ep=8)
        assert t.gpu_count == 8

    def test_gpu_count_cp_does_not_add(self):
        """CP reuses TP-group devices, should not increase gpu_count."""
        t = ParallelTopology(tp=4, dp=1, pp=1, cp=2)
        assert t.gpu_count == 4

    def test_gpu_count_all_dimensions(self):
        t = ParallelTopology(tp=2, dp=4, pp=2)
        assert t.gpu_count == 16

    def test_validate_valid_topology(self):
        t = ParallelTopology(tp=4, dp=2, pp=1, ep=8)
        assert t.validate() == []

    def test_validate_tp_less_than_1(self):
        t = ParallelTopology(tp=0, dp=1, pp=1)
        errors = t.validate()
        assert any("tp must be >= 1" in e for e in errors)

    def test_validate_dp_less_than_1(self):
        t = ParallelTopology(tp=1, dp=0, pp=1)
        errors = t.validate()
        assert any("dp must be >= 1" in e for e in errors)

    def test_validate_pp_less_than_1(self):
        t = ParallelTopology(tp=1, dp=1, pp=0)
        errors = t.validate()
        assert any("pp must be >= 1" in e for e in errors)

    def test_validate_ep_less_than_1(self):
        t = ParallelTopology(tp=1, dp=1, pp=1, ep=0)
        errors = t.validate()
        assert any("ep must be >= 1" in e for e in errors)

    def test_validate_ep_exceeds_tp_dp(self):
        t = ParallelTopology(tp=2, dp=2, pp=1, ep=5)
        errors = t.validate()
        assert any("ep(5) > tp*dp(4)" in e for e in errors)

    def test_validate_ep_at_boundary(self):
        """ep == tp*dp is valid."""
        t = ParallelTopology(tp=2, dp=2, pp=1, ep=4)
        assert t.validate() == []

    def test_validate_ep_le_tp_indivisible(self):
        """ep ≤ tp but tp % ep != 0 → error."""
        t = ParallelTopology(tp=4, dp=4, pp=1, ep=3)
        errors = t.validate()
        assert any("tp(4) % ep(3) != 0" in e for e in errors)

    def test_validate_ep_gt_tp_indivisible(self):
        """ep > tp but ep % tp != 0 → error (was silently accepted before fix)."""
        t = ParallelTopology(tp=4, dp=4, pp=1, ep=6)
        errors = t.validate()
        assert any("ep(6) % tp(4) != 0" in e for e in errors)

    def test_validate_tp_zero_with_ep_no_crash(self):
        """tp=0 with ep set must collect errors, not raise ZeroDivisionError."""
        t = ParallelTopology(tp=0, dp=1, pp=1, ep=2)
        errors = t.validate()
        assert any("tp must be >= 1" in e for e in errors)
        # EP block should be skipped when core dims are invalid
        assert not any("ZeroDivision" in e for e in errors)

    def test_validate_ep_gt_tp_divisible(self):
        """ep > tp and ep % tp == 0 → valid."""
        t = ParallelTopology(tp=4, dp=4, pp=1, ep=8)
        assert t.validate() == []

    def test_validate_cp_less_than_1(self):
        t = ParallelTopology(tp=1, dp=1, pp=1, cp=0)
        errors = t.validate()
        assert any("cp must be >= 1" in e for e in errors)

    def test_validate_dpa_no_tp_dp_constraint(self):
        """DPA no longer requires tp % dp == 0 (sglang constraint is tautological)."""
        t = ParallelTopology(tp=3, dp=2, pp=1, enable_dp_attention=True)
        assert t.validate() == []

    def test_validate_dpa_valid(self):
        """DPA with tp % dp == 0 is valid."""
        t = ParallelTopology(tp=4, dp=2, pp=1, enable_dp_attention=True)
        assert t.validate() == []

    def test_validate_dpa_dp_1_always_valid(self):
        """DPA with dp=1 never triggers the constraint (tp % 1 == 0)."""
        t = ParallelTopology(tp=3, dp=1, pp=1, enable_dp_attention=True)
        assert t.validate() == []

    def test_frozen(self):
        t = ParallelTopology(tp=4)
        with pytest.raises(AttributeError):
            t.tp = 8  # type: ignore[misc]

    def test_defaults(self):
        t = ParallelTopology()
        assert t.tp == 1
        assert t.dp == 1
        assert t.pp == 1
        assert t.ep is None
        assert t.enable_dp_attention is False
        assert t.cp is None


# ---------- DeviceGroup ----------


class TestDeviceGroup:
    def test_basic_construction(self):
        dg = DeviceGroup(count=8, gpu_model="NVIDIA H100-SXM5-80GB")
        assert dg.count == 8
        assert dg.gpu_model == "NVIDIA H100-SXM5-80GB"
        assert dg.host == "localhost"

    def test_frozen(self):
        dg = DeviceGroup(count=4)
        with pytest.raises(AttributeError):
            dg.count = 8  # type: ignore[misc]


# ---------- RoleAssignment ----------


class TestRoleAssignment:
    def test_validate_sufficient_gpus(self):
        ra = RoleAssignment(
            role="full",
            devices=DeviceGroup(count=8),
            topology=ParallelTopology(tp=4, dp=2),
        )
        assert ra.validate() == []

    def test_validate_insufficient_gpus(self):
        ra = RoleAssignment(
            role="full",
            devices=DeviceGroup(count=4),
            topology=ParallelTopology(tp=4, dp=2),
        )
        errors = ra.validate()
        assert any("needs 8 GPUs" in e for e in errors)

    def test_validate_with_replicas(self):
        """tp=2 x dp=1 x replicas=3 = 6 GPUs needed."""
        ra = RoleAssignment(
            role="full",
            devices=DeviceGroup(count=4),
            topology=ParallelTopology(tp=2, dp=1),
            replicas=3,
        )
        errors = ra.validate()
        assert any("needs 6 GPUs" in e for e in errors)

    def test_validate_replicas_less_than_1(self):
        ra = RoleAssignment(
            role="full",
            devices=DeviceGroup(count=8),
            topology=ParallelTopology(tp=4),
            replicas=0,
        )
        errors = ra.validate()
        assert any("replicas must be >= 1" in e for e in errors)

    def test_engine_params_default_empty(self):
        ra = RoleAssignment(
            role="full",
            devices=DeviceGroup(count=4),
            topology=ParallelTopology(tp=4),
        )
        assert ra.engine_params == {}

    def test_engine_params_populated(self):
        ra = RoleAssignment(
            role="full",
            devices=DeviceGroup(count=4),
            topology=ParallelTopology(tp=4),
            engine_params={"dtype": "bfloat16", "max_model_len": 4096},
        )
        assert ra.engine_params["dtype"] == "bfloat16"
        assert ra.engine_params["max_model_len"] == 4096

    def test_scaling_placeholder(self):
        sp = ScalingPolicy(min_replicas=1, max_replicas=4)
        ra = RoleAssignment(
            role="full",
            devices=DeviceGroup(count=4),
            topology=ParallelTopology(tp=4),
            scaling=sp,
        )
        assert ra.scaling is not None
        assert ra.scaling.max_replicas == 4

    def test_role_assignment_normalizes_dash_form_engine_params(self):
        """RoleAssignment normalizes dash-form engine_params keys to underscore."""
        assignment = RoleAssignment(
            role="prefill",
            devices=DeviceGroup(count=1),
            topology=ParallelTopology(tp=1, dp=1, pp=1),
            engine_params={"tensor-parallel-size": 1, "max-model-len": 4096},
        )
        assert dict(assignment.engine_params) == {
            "tensor_parallel_size": 1,
            "max_model_len": 4096,
        }

    def test_role_assignment_rejects_dash_underscore_collision(self):
        """RoleAssignment.__post_init__ raises ValueError on intra-dict collision."""
        with pytest.raises(ValueError, match="foo-bar.*foo_bar|foo_bar.*foo-bar"):
            RoleAssignment(
                role="prefill",
                devices=DeviceGroup(count=1),
                topology=ParallelTopology(tp=1, dp=1, pp=1),
                engine_params={"foo-bar": 1, "foo_bar": 2},
            )


# ---------- DeploymentPlan ----------


class TestDeploymentPlan:
    def _make_plan(self, *roles: str) -> DeploymentPlan:
        assignments = tuple(
            RoleAssignment(
                role=role,
                devices=DeviceGroup(count=4),
                topology=ParallelTopology(tp=4),
            )
            for role in roles
        )
        return DeploymentPlan(
            checkpoint="/models/test",
            backend="sglang",
            assignments=assignments,
        )

    def test_is_disaggregated_full_only(self):
        plan = self._make_plan(WellKnownRole.FULL)
        assert plan.is_disaggregated is False

    def test_is_disaggregated_pd_split(self):
        plan = self._make_plan(WellKnownRole.PREFILL, WellKnownRole.DECODE)
        assert plan.is_disaggregated is True

    def test_total_gpus_single_role(self):
        plan = DeploymentPlan(
            checkpoint="/models/test",
            backend="vllm",
            assignments=(
                RoleAssignment(
                    role="full",
                    devices=DeviceGroup(count=8),
                    topology=ParallelTopology(tp=4, dp=2),
                ),
            ),
        )
        assert plan.total_gpus == 8

    def test_total_gpus_multi_role(self):
        plan = DeploymentPlan(
            checkpoint="/models/test",
            backend="sglang",
            assignments=(
                RoleAssignment(
                    role="prefill",
                    devices=DeviceGroup(count=4),
                    topology=ParallelTopology(tp=4),
                ),
                RoleAssignment(
                    role="decode",
                    devices=DeviceGroup(count=4),
                    topology=ParallelTopology(tp=2, dp=2),
                ),
            ),
        )
        assert plan.total_gpus == 8

    def test_total_gpus_with_replicas(self):
        plan = DeploymentPlan(
            checkpoint="/models/test",
            backend="vllm",
            assignments=(
                RoleAssignment(
                    role="full",
                    devices=DeviceGroup(count=6),
                    topology=ParallelTopology(tp=2),
                    replicas=3,
                ),
            ),
        )
        assert plan.total_gpus == 6

    def test_validate_empty_assignments(self):
        plan = DeploymentPlan(
            checkpoint="/models/test",
            backend="vllm",
            assignments=(),
        )
        errors = plan.validate()
        assert any("at least one assignment" in e for e in errors)

    def test_validate_pd_paired(self):
        plan = self._make_plan(WellKnownRole.PREFILL, WellKnownRole.DECODE)
        assert plan.validate() == []

    def test_validate_prefill_without_decode(self):
        plan = self._make_plan(WellKnownRole.PREFILL)
        errors = plan.validate()
        assert any("matching decode role" in e for e in errors)

    def test_validate_decode_without_prefill(self):
        plan = self._make_plan(WellKnownRole.DECODE)
        errors = plan.validate()
        assert any("matching prefill role" in e for e in errors)

    def test_validate_propagates_assignment_errors(self):
        plan = DeploymentPlan(
            checkpoint="/models/test",
            backend="vllm",
            assignments=(
                RoleAssignment(
                    role="full",
                    devices=DeviceGroup(count=1),
                    topology=ParallelTopology(tp=4, dp=2),
                ),
            ),
        )
        errors = plan.validate()
        assert any("needs 8 GPUs" in e for e in errors)

    def test_frozen(self):
        plan = self._make_plan("full")
        with pytest.raises(AttributeError):
            plan.checkpoint = "/other"  # type: ignore[misc]

    # -------- deterministic / seed field plumbing --------

    def test_deterministic_true_with_seed_is_ok(self):
        plan = DeploymentPlan(
            checkpoint="/models/test",
            backend="vllm",
            assignments=(
                RoleAssignment(
                    role="full",
                    devices=DeviceGroup(count=1),
                    topology=ParallelTopology(tp=1),
                ),
            ),
            deterministic=True,
            seed=0,
        )
        assert plan.deterministic is True
        assert plan.seed == 0

    def test_non_deterministic_defaults(self):
        from sieval.infer.topology.models import DETERMINISTIC_DEFAULT_SEED

        plan = DeploymentPlan(
            checkpoint="/models/test",
            backend="vllm",
            assignments=(
                RoleAssignment(
                    role="full",
                    devices=DeviceGroup(count=1),
                    topology=ParallelTopology(tp=1),
                ),
            ),
        )
        assert plan.deterministic is False
        # seed always defaults to DETERMINISTIC_DEFAULT_SEED; the translators
        # gate --seed emission on plan.deterministic, not seed-is-set.
        assert plan.seed == DETERMINISTIC_DEFAULT_SEED


# ---------- DeploymentCapabilities ----------


class TestDeploymentCapabilities:
    def test_basic_construction(self):
        caps = DeploymentCapabilities(
            api_base="http://localhost:8000/v1",
            is_disaggregated=False,
            roles=("full",),
            total_gpus=4,
        )
        assert caps.api_base == "http://localhost:8000/v1"
        assert caps.is_disaggregated is False
        assert caps.roles == ("full",)
        assert caps.total_gpus == 4
        assert caps.endpoints == {}
        assert caps.metrics_url is None

    def test_frozen(self):
        caps = DeploymentCapabilities(
            api_base="http://localhost:8000/v1",
            is_disaggregated=False,
            roles=("full",),
            total_gpus=4,
        )
        with pytest.raises(AttributeError):
            caps.api_base = "other"  # type: ignore[misc]


# ---------- ModelProfile ----------


class TestModelProfile:
    def test_basic_construction(self):
        mp = ModelProfile(param_billions=72.0, num_layers=80)
        assert mp.param_billions == 72.0
        assert mp.num_layers == 80
        assert mp.is_moe is False
        assert mp.num_experts is None
        assert mp.bytes_per_param == 2.0

    def test_moe_fields(self):
        mp = ModelProfile(
            param_billions=236.0,
            num_layers=64,
            is_moe=True,
            num_experts=128,
            bytes_per_param=1.0,
        )
        assert mp.is_moe is True
        assert mp.num_experts == 128


# ---------- HardwareEnv ----------


class TestHardwareEnv:
    def test_basic_construction(self):
        hw = HardwareEnv(
            gpu_count=8,
            gpu_model="NVIDIA H100-SXM5-80GB",
            gpu_memory_mib=81920,
        )
        assert hw.gpu_count == 8
        assert hw.nodes == 1
        assert hw.gpus_per_node is None


# ---------- UserHints ----------


class TestUserHints:
    def test_all_none_defaults(self):
        h = UserHints()
        assert h.optimize_for is None
        assert h.disaggregation is None
        assert h.role_overrides is None
        assert h.tp is None
        assert h.dp is None
        assert h.pp is None
        assert h.ep is None
        assert h.enable_dp_attention is None
        assert h.cp is None

    def test_enable_dp_attention_field(self):
        h = UserHints(enable_dp_attention=True)
        assert h.enable_dp_attention is True

    def test_cp_field(self):
        h = UserHints(cp=4)
        assert h.cp == 4

    def test_role_overrides_type(self):
        h = UserHints(role_overrides={"prefill": {"dtype": "bfloat16"}})
        assert h.role_overrides is not None
        assert h.role_overrides["prefill"]["dtype"] == "bfloat16"


# ---------- ScalingPolicy ----------


class TestScalingPolicy:
    def test_defaults(self):
        sp = ScalingPolicy()
        assert sp.min_replicas == 1
        assert sp.max_replicas is None
        assert sp.scale_unit_gpus is None
        assert sp.max_disruption == "hot"

    def test_frozen(self):
        sp = ScalingPolicy()
        with pytest.raises(AttributeError):
            sp.min_replicas = 2  # type: ignore[misc]


# ---------- ResolveStep / ResolveResult ----------


class TestResolveStepAndResult:
    def test_resolve_step_construction(self):
        step = ResolveStep(field="tp", value=4, reason="compute_tp(72B)", source="auto")
        assert step.field == "tp"
        assert step.value == 4
        assert step.source == "auto"

    def test_resolve_result_construction(self):
        plan = DeploymentPlan(
            checkpoint="/models/test",
            backend="vllm",
            assignments=(
                RoleAssignment(
                    role="full",
                    devices=DeviceGroup(count=8),
                    topology=ParallelTopology(tp=4, dp=2),
                ),
            ),
        )
        step = ResolveStep(field="tp", value=4, reason="test", source="auto")
        result = ResolveResult(plan=plan, steps=(step,))
        assert result.plan is plan
        assert len(result.steps) == 1


# ---------- ServiceBinding ----------


class TestServiceBinding:
    def test_basic_construction(self):
        sb = ServiceBinding(
            kind="kv_cache",
            provider="simm",
            address="10.0.1.100:30001",
        )
        assert sb.kind == "kv_cache"
        assert sb.options == {}


# ---------- WellKnownRole ----------


class TestWellKnownRole:
    def test_role_constants(self):
        assert WellKnownRole.FULL == "full"
        assert WellKnownRole.PREFILL == "prefill"
        assert WellKnownRole.DECODE == "decode"
        assert WellKnownRole.ENCODER == "encoder"
        assert WellKnownRole.ROUTER == "router"
