"""
Unit tests for backend translators (vLLM + SGLang).

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

import pytest
from loguru import logger

from sieval.infer.backends.sglang_translator import SglangTranslator
from sieval.infer.backends.translator import BackendCommand, inject_user_env
from sieval.infer.backends.vllm_translator import VllmTranslator
from sieval.infer.topology.models import (
    DETERMINISTIC_DEFAULT_SEED,
    DeploymentPlan,
    DeviceGroup,
    ParallelTopology,
    RoleAssignment,
)


def _make_plan(
    backend: str = "vllm",
    checkpoint: str = "/models/Qwen3-72B",
    tp: int = 1,
    dp: int = 1,
    pp: int = 1,
    ep: int | None = None,
    enable_dp_attention: bool = False,
    cp: int | None = None,
    gpu_count: int = 8,
    engine_params: dict | None = None,
    roles: tuple[str, ...] = ("full",),
    host: str = "localhost",
    deterministic: bool = False,
    seed: int = DETERMINISTIC_DEFAULT_SEED,
) -> DeploymentPlan:
    topo = ParallelTopology(
        tp=tp,
        dp=dp,
        pp=pp,
        ep=ep,
        enable_dp_attention=enable_dp_attention,
        cp=cp,
    )
    assignments = tuple(
        RoleAssignment(
            role=role,
            devices=DeviceGroup(count=gpu_count, host=host),
            topology=topo,
            engine_params=dict(engine_params) if engine_params else {},
        )
        for role in roles
    )
    return DeploymentPlan(
        checkpoint=checkpoint,
        backend=backend,
        assignments=assignments,
        deterministic=deterministic,
        seed=seed,
    )


# ===================== VllmTranslator =====================


class TestVllmTranslator:
    def test_full_role_tp4(self):
        plan = _make_plan(backend="vllm", tp=4)
        cmds = VllmTranslator().translate(plan)
        assert len(cmds) == 1
        cmd = cmds[0]
        assert "--tensor-parallel-size" in cmd.cli_args
        idx = cmd.cli_args.index("--tensor-parallel-size")
        assert cmd.cli_args[idx + 1] == "4"

    def test_dp2(self):
        plan = _make_plan(backend="vllm", dp=2)
        cmds = VllmTranslator().translate(plan)
        cmd = cmds[0]
        assert "--data-parallel-size" in cmd.cli_args
        idx = cmd.cli_args.index("--data-parallel-size")
        assert cmd.cli_args[idx + 1] == "2"

    def test_pp2(self):
        plan = _make_plan(backend="vllm", pp=2)
        cmds = VllmTranslator().translate(plan)
        cmd = cmds[0]
        assert "--pipeline-parallel-size" in cmd.cli_args

    def test_ep_as_bool_flag(self):
        """vLLM EP is a bool flag: --enable-expert-parallel (no value)."""
        plan = _make_plan(backend="vllm", ep=8)
        cmds = VllmTranslator().translate(plan)
        cmd = cmds[0]
        assert "--enable-expert-parallel" in cmd.cli_args
        # No value after the flag
        idx = cmd.cli_args.index("--enable-expert-parallel")
        if idx + 1 < len(cmd.cli_args):
            assert not cmd.cli_args[idx + 1].isdigit()

    def test_ep_1_no_flag(self):
        """ep=1 should not enable expert parallelism (no-op)."""
        plan = _make_plan(backend="vllm", ep=1)
        cmds = VllmTranslator().translate(plan)
        cmd = cmds[0]
        assert "--enable-expert-parallel" not in cmd.cli_args

    def test_cp_dual_flags(self):
        """vLLM uses separate prefill/decode CP flags."""
        plan = _make_plan(backend="vllm", cp=2)
        cmds = VllmTranslator().translate(plan)
        cmd = cmds[0]
        assert "--prefill-context-parallel-size" in cmd.cli_args
        assert "--decode-context-parallel-size" in cmd.cli_args
        idx_p = cmd.cli_args.index("--prefill-context-parallel-size")
        idx_d = cmd.cli_args.index("--decode-context-parallel-size")
        assert cmd.cli_args[idx_p + 1] == "2"
        assert cmd.cli_args[idx_d + 1] == "2"

    def test_dpa_warning_and_skip(self):
        """vLLM doesn't support DPA — should not add a flag."""
        plan = _make_plan(backend="vllm", enable_dp_attention=True)
        cmds = VllmTranslator().translate(plan)
        cmd = cmds[0]
        assert "--enable-dp-attention" not in cmd.cli_args

    def test_default_port_8000(self):
        plan = _make_plan(backend="vllm")
        cmds = VllmTranslator().translate(plan)
        cmd = cmds[0]
        idx = cmd.cli_args.index("--port")
        assert cmd.cli_args[idx + 1] == "8000"

    def test_health_url(self):
        plan = _make_plan(backend="vllm")
        cmds = VllmTranslator().translate(plan)
        cmd = cmds[0]
        assert cmd.health_url == "http://localhost:8000/health"
        assert cmd.backend == "vllm"

    def test_health_url_custom_host(self):
        plan = _make_plan(backend="vllm", host="10.0.1.5")
        cmds = VllmTranslator().translate(plan)
        cmd = cmds[0]
        assert cmd.health_url == "http://10.0.1.5:8000/health"

    def test_engine_params_dtype(self):
        plan = _make_plan(backend="vllm", engine_params={"dtype": "bfloat16"})
        cmds = VllmTranslator().translate(plan)
        cmd = cmds[0]
        assert "--dtype" in cmd.cli_args
        idx = cmd.cli_args.index("--dtype")
        assert cmd.cli_args[idx + 1] == "bfloat16"

    def test_engine_params_bool_true(self):
        plan = _make_plan(backend="vllm", engine_params={"trust_remote_code": True})
        cmds = VllmTranslator().translate(plan)
        cmd = cmds[0]
        assert "--trust-remote-code" in cmd.cli_args

    def test_engine_params_bool_false(self):
        """Bool False should not produce a flag."""
        plan = _make_plan(backend="vllm", engine_params={"trust_remote_code": False})
        cmds = VllmTranslator().translate(plan)
        cmd = cmds[0]
        assert "--trust-remote-code" not in cmd.cli_args

    def test_checkpoint_in_args(self):
        plan = _make_plan(backend="vllm", checkpoint="/models/test-model")
        cmds = VllmTranslator().translate(plan)
        cmd = cmds[0]
        assert "/models/test-model" in cmd.cli_args

    def test_tp1_not_in_args(self):
        """tp=1 should not produce a --tensor-parallel-size flag."""
        plan = _make_plan(backend="vllm", tp=1)
        cmds = VllmTranslator().translate(plan)
        cmd = cmds[0]
        assert "--tensor-parallel-size" not in cmd.cli_args

    def test_dp1_not_in_args(self):
        """dp=1 should not produce a --data-parallel-size flag."""
        plan = _make_plan(backend="vllm", dp=1)
        cmds = VllmTranslator().translate(plan)
        cmd = cmds[0]
        assert "--data-parallel-size" not in cmd.cli_args


# ===================== SglangTranslator =====================


class TestSglangTranslator:
    def test_full_role_tp4(self):
        plan = _make_plan(backend="sglang", tp=4)
        cmds = SglangTranslator().translate(plan)
        assert len(cmds) == 1
        cmd = cmds[0]
        assert "--tp" in cmd.cli_args
        idx = cmd.cli_args.index("--tp")
        assert cmd.cli_args[idx + 1] == "4"

    def test_dp2(self):
        plan = _make_plan(backend="sglang", dp=2)
        cmds = SglangTranslator().translate(plan)
        cmd = cmds[0]
        assert "--dp-size" in cmd.cli_args
        idx = cmd.cli_args.index("--dp-size")
        assert cmd.cli_args[idx + 1] == "2"

    def test_pp2(self):
        plan = _make_plan(backend="sglang", pp=2)
        cmds = SglangTranslator().translate(plan)
        cmd = cmds[0]
        assert "--pp-size" in cmd.cli_args

    def test_ep_as_int(self):
        """SGLang EP uses --ep-size with an int value (not bool like vLLM)."""
        plan = _make_plan(backend="sglang", ep=8)
        cmds = SglangTranslator().translate(plan)
        cmd = cmds[0]
        assert "--ep-size" in cmd.cli_args
        idx = cmd.cli_args.index("--ep-size")
        assert cmd.cli_args[idx + 1] == "8"

    def test_ep_1_no_flag(self):
        """ep=1 should not emit --ep-size (no-op, same as no EP)."""
        plan = _make_plan(backend="sglang", ep=1)
        cmds = SglangTranslator().translate(plan)
        cmd = cmds[0]
        assert "--ep-size" not in cmd.cli_args

    def test_enable_dp_attention(self):
        """DPA: --tp should be total GPUs (tp*dp), not per-group."""
        plan = _make_plan(backend="sglang", tp=2, dp=4, enable_dp_attention=True)
        cmds = SglangTranslator().translate(plan)
        cmd = cmds[0]
        assert "--enable-dp-attention" in cmd.cli_args
        # --tp = total GPUs: 2 * 4 = 8
        idx = cmd.cli_args.index("--tp")
        assert cmd.cli_args[idx + 1] == "8"
        # --dp-size still present
        idx_dp = cmd.cli_args.index("--dp-size")
        assert cmd.cli_args[idx_dp + 1] == "4"

    def test_dpa_tp1_dp4(self):
        """DPA with per-group tp=1: --tp emitted as total GPUs."""
        plan = _make_plan(backend="sglang", tp=1, dp=4, enable_dp_attention=True)
        cmds = SglangTranslator().translate(plan)
        cmd = cmds[0]
        assert "--enable-dp-attention" in cmd.cli_args
        idx = cmd.cli_args.index("--tp")
        assert cmd.cli_args[idx + 1] == "4"  # 1 * 4

    def test_dpa_tp4_dp1(self):
        """DPA with dp=1: --tp emitted, no --dp-size."""
        plan = _make_plan(backend="sglang", tp=4, dp=1, enable_dp_attention=True)
        cmds = SglangTranslator().translate(plan)
        cmd = cmds[0]
        assert "--enable-dp-attention" in cmd.cli_args
        idx = cmd.cli_args.index("--tp")
        assert cmd.cli_args[idx + 1] == "4"  # 4 * 1
        assert "--dp-size" not in cmd.cli_args

    def test_no_dpa_tp_unchanged(self):
        """Without DPA, --tp is per-group (unchanged behavior)."""
        plan = _make_plan(backend="sglang", tp=2, dp=4, enable_dp_attention=False)
        cmds = SglangTranslator().translate(plan)
        cmd = cmds[0]
        assert "--enable-dp-attention" not in cmd.cli_args
        idx = cmd.cli_args.index("--tp")
        assert cmd.cli_args[idx + 1] == "2"  # per-group, not 8

    def test_default_port_30000(self):
        plan = _make_plan(backend="sglang")
        cmds = SglangTranslator().translate(plan)
        cmd = cmds[0]
        idx = cmd.cli_args.index("--port")
        assert cmd.cli_args[idx + 1] == "30000"

    def test_cp4(self):
        plan = _make_plan(backend="sglang", cp=4)
        cmds = SglangTranslator().translate(plan)
        cmd = cmds[0]
        assert "--attn-cp-size" in cmd.cli_args
        idx = cmd.cli_args.index("--attn-cp-size")
        assert cmd.cli_args[idx + 1] == "4"

    def test_checkpoint_passed(self):
        plan = _make_plan(backend="sglang", checkpoint="/models/qwen3")
        cmds = SglangTranslator().translate(plan)
        cmd = cmds[0]
        assert "--model-path" in cmd.cli_args
        idx = cmd.cli_args.index("--model-path")
        assert cmd.cli_args[idx + 1] == "/models/qwen3"

    def test_engine_params_passthrough(self):
        plan = _make_plan(
            backend="sglang",
            engine_params={"dtype": "bfloat16", "max_model_len": 4096},
        )
        cmds = SglangTranslator().translate(plan)
        cmd = cmds[0]
        assert "--dtype" in cmd.cli_args
        assert "--max-model-len" in cmd.cli_args

    def test_engine_params_bool_true(self):
        plan = _make_plan(
            backend="sglang",
            engine_params={"trust_remote_code": True},
        )
        cmds = SglangTranslator().translate(plan)
        cmd = cmds[0]
        assert "--trust-remote-code" in cmd.cli_args

    def test_engine_params_bool_false(self):
        """Bool engine params with False should not appear."""
        plan = _make_plan(
            backend="sglang",
            engine_params={"trust_remote_code": False},
        )
        cmds = SglangTranslator().translate(plan)
        cmd = cmds[0]
        assert "--trust-remote-code" not in cmd.cli_args

    def test_health_url(self):
        plan = _make_plan(backend="sglang")
        cmds = SglangTranslator().translate(plan)
        cmd = cmds[0]
        assert cmd.health_url == "http://localhost:30000/health"

    def test_tp1_not_in_args(self):
        plan = _make_plan(backend="sglang", tp=1)
        cmds = SglangTranslator().translate(plan)
        cmd = cmds[0]
        assert "--tp" not in cmd.cli_args

    def test_multi_assignment_port_offset(self):
        """Multiple assignments get incremented ports."""
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
        cmds = SglangTranslator().translate(plan)
        assert len(cmds) == 2
        # First: port 30000
        idx0 = cmds[0].cli_args.index("--port")
        assert cmds[0].cli_args[idx0 + 1] == "30000"
        assert cmds[0].role == "prefill"
        # Second: port 30001
        idx1 = cmds[1].cli_args.index("--port")
        assert cmds[1].cli_args[idx1 + 1] == "30001"
        assert cmds[1].role == "decode"


# ===================== get_translator registry =====================


class TestGetTranslator:
    def test_vllm_via_registry(self):
        from sieval.infer.backends import get_translator

        translator = get_translator("vllm")
        assert isinstance(translator, VllmTranslator)

    def test_sglang_via_registry(self):
        from sieval.infer.backends import get_translator

        translator = get_translator("sglang")
        assert isinstance(translator, SglangTranslator)

    def test_unknown_raises(self):
        from sieval.infer.backends import get_translator

        with pytest.raises(KeyError, match="Unknown backend"):
            get_translator("tensorrt-llm")

    def test_vllm_protocol(self):
        translator = VllmTranslator()
        assert hasattr(translator, "translate")

    def test_sglang_protocol(self):
        translator = SglangTranslator()
        assert hasattr(translator, "translate")


# ===================== inject_user_env =====================


class TestInjectUserEnv:
    def test_injects_env_into_commands(self) -> None:
        cmds = [BackendCommand(cli_args=["serve"])]
        inject_user_env(cmds, {"FOO": "bar"})
        assert cmds[0].env == {"FOO": "bar"}

    def test_noop_on_empty_env(self) -> None:
        cmds = [BackendCommand(cli_args=["serve"], env={"EXISTING": "1"})]
        inject_user_env(cmds, {})
        assert cmds[0].env == {"EXISTING": "1"}

    def test_string_values_passed_through(self) -> None:
        """Values are already coerced to str by resolve_infer_config."""
        cmds = [BackendCommand(cli_args=["serve"])]
        inject_user_env(cmds, {"DEBUG": "1", "VERBOSE": "True"})
        assert cmds[0].env == {"DEBUG": "1", "VERBOSE": "True"}

    def test_overrides_existing_with_warning(self) -> None:
        cmds = [BackendCommand(cli_args=["serve"], env={"CUDA_VISIBLE_DEVICES": "0,1"})]
        warnings: list[str] = []
        sink_id = logger.add(lambda msg: warnings.append(msg), level="WARNING")
        try:
            inject_user_env(cmds, {"CUDA_VISIBLE_DEVICES": "2,3", "FOO": "bar"})
        finally:
            logger.remove(sink_id)
        assert cmds[0].env["CUDA_VISIBLE_DEVICES"] == "2,3"
        assert cmds[0].env["FOO"] == "bar"
        assert any("CUDA_VISIBLE_DEVICES" in w for w in warnings)

    def test_locked_env_key_override_raises(self) -> None:
        """Overriding a translator-locked env key raises (no silent weakening)."""
        cmds = [
            BackendCommand(
                cli_args=["serve"],
                env={"VLLM_BATCH_INVARIANT": "1"},
                locked_env_keys=frozenset({"VLLM_BATCH_INVARIANT"}),
            )
        ]
        with pytest.raises(ValueError, match="VLLM_BATCH_INVARIANT"):
            inject_user_env(cmds, {"VLLM_BATCH_INVARIANT": "0"})

    def test_locked_env_allows_unrelated_user_keys(self) -> None:
        """Locked keys don't block unrelated user env from being merged."""
        cmds = [
            BackendCommand(
                cli_args=["serve"],
                env={"VLLM_BATCH_INVARIANT": "1"},
                locked_env_keys=frozenset({"VLLM_BATCH_INVARIANT"}),
            )
        ]
        inject_user_env(cmds, {"CUDA_VISIBLE_DEVICES": "0,1"})
        assert cmds[0].env["CUDA_VISIBLE_DEVICES"] == "0,1"
        assert cmds[0].env["VLLM_BATCH_INVARIANT"] == "1"


# ===================== VllmDeterministic =====================


class TestVllmDeterministic:
    def test_deterministic_injects_env_var(self):
        plan = _make_plan(backend="vllm", deterministic=True, seed=0)
        translator = VllmTranslator()
        commands = translator.translate(plan)
        assert len(commands) == 1
        assert commands[0].env.get("VLLM_BATCH_INVARIANT") == "1"

    def test_non_deterministic_no_env_var(self):
        plan = _make_plan(backend="vllm")
        translator = VllmTranslator()
        commands = translator.translate(plan)
        assert len(commands) == 1
        assert "VLLM_BATCH_INVARIANT" not in commands[0].env

    def test_deterministic_pins_engine_seed(self):
        """Under deterministic mode, vLLM translator pins --seed from plan.seed."""
        plan = _make_plan(backend="vllm", deterministic=True, seed=0)
        translator = VllmTranslator()
        commands = translator.translate(plan)
        args = commands[0].cli_args
        assert "--seed" in args
        # seed value immediately follows the flag
        assert args[args.index("--seed") + 1] == "0"

    def test_non_deterministic_no_engine_seed(self):
        """Without deterministic, --seed is not injected (engine default kicks in)."""
        plan = _make_plan(backend="vllm")
        translator = VllmTranslator()
        commands = translator.translate(plan)
        assert "--seed" not in commands[0].cli_args

    def test_user_engine_params_skip_pinned_seed(self):
        """YAML `overrides: {seed: 42}` makes the translator skip pinning.

        We deliberately avoid emitting both --seed 0 and --seed 42 so the
        behavior doesn't depend on upstream argparse "last wins" semantics.
        """
        plan = _make_plan(
            backend="vllm",
            deterministic=True,
            seed=0,
            engine_params={"seed": 42},
        )
        translator = VllmTranslator()
        commands = translator.translate(plan)
        args = commands[0].cli_args
        seed_indices = [i for i, a in enumerate(args) if a == "--seed"]
        # Only the user-provided seed appears — no pinned duplicate.
        assert len(seed_indices) == 1
        assert args[seed_indices[0] + 1] == "42"

    def test_plan_seed_flows_to_pinned_seed(self):
        """`plan.seed=42` (from --seed CLI) is honored as the engine seed."""
        plan = _make_plan(backend="vllm", deterministic=True, seed=42)
        commands = VllmTranslator().translate(plan)
        args = commands[0].cli_args
        assert args[args.index("--seed") + 1] == "42"

    def test_batch_invariant_key_is_locked(self):
        """Deterministic vLLM command declares VLLM_BATCH_INVARIANT as locked."""
        plan = _make_plan(backend="vllm", deterministic=True, seed=0)
        commands = VllmTranslator().translate(plan)
        assert "VLLM_BATCH_INVARIANT" in commands[0].locked_env_keys

    def test_non_deterministic_has_no_locked_keys(self):
        """Non-deterministic command has no locked env keys."""
        plan = _make_plan(backend="vllm")
        commands = VllmTranslator().translate(plan)
        assert commands[0].locked_env_keys == frozenset()


# ===================== SglangDeterministic =====================


class TestSglangDeterministic:
    def test_deterministic_injects_flag(self):
        plan = _make_plan(backend="sglang", deterministic=True, seed=0)
        translator = SglangTranslator()
        commands = translator.translate(plan)
        assert len(commands) == 1
        assert "--enable-deterministic-inference" in commands[0].cli_args

    def test_non_deterministic_no_flag(self):
        plan = _make_plan(backend="sglang")
        translator = SglangTranslator()
        commands = translator.translate(plan)
        assert len(commands) == 1
        assert "--enable-deterministic-inference" not in commands[0].cli_args

    def test_deterministic_pins_engine_seed(self):
        """Under deterministic mode, SGLang pins --random-seed from plan.seed."""
        plan = _make_plan(backend="sglang", deterministic=True, seed=0)
        translator = SglangTranslator()
        commands = translator.translate(plan)
        args = commands[0].cli_args
        assert "--random-seed" in args
        assert args[args.index("--random-seed") + 1] == "0"

    def test_non_deterministic_no_engine_seed(self):
        """Without deterministic, --random-seed is not injected."""
        plan = _make_plan(backend="sglang")
        translator = SglangTranslator()
        commands = translator.translate(plan)
        assert "--random-seed" not in commands[0].cli_args

    def test_user_engine_params_skip_pinned_seed(self):
        """YAML `overrides: {random_seed: 42}` makes the translator skip pinning.

        We deliberately avoid emitting both --random-seed 0 and --random-seed 42
        so the behavior doesn't depend on upstream argparse "last wins" semantics.
        """
        plan = _make_plan(
            backend="sglang",
            deterministic=True,
            seed=0,
            engine_params={"random_seed": 42},
        )
        translator = SglangTranslator()
        commands = translator.translate(plan)
        args = commands[0].cli_args
        seed_indices = [i for i, a in enumerate(args) if a == "--random-seed"]
        # Only the user-provided seed appears — no pinned duplicate.
        assert len(seed_indices) == 1
        assert args[seed_indices[0] + 1] == "42"

    def test_user_engine_params_dash_form_skips_pinned_seed(self):
        """User sets `random-seed` (dash form); translator should skip the pinned
        default. Works because `RoleAssignment.__post_init__` normalizes the key
        to `random_seed` before the translator sees it, so the guard
        `"random_seed" not in assignment.engine_params` correctly detects the
        user's override.
        """
        plan = _make_plan(
            backend="sglang",
            deterministic=True,
            seed=0,
            engine_params={"random-seed": 42},
        )
        commands = SglangTranslator().translate(plan)
        args = commands[0].cli_args
        seed_indices = [i for i, a in enumerate(args) if a == "--random-seed"]
        assert len(seed_indices) == 1
        assert args[seed_indices[0] + 1] == "42"

    def test_plan_seed_flows_to_pinned_seed(self):
        """`plan.seed=42` (from --seed CLI) is honored as the engine seed."""
        plan = _make_plan(backend="sglang", deterministic=True, seed=42)
        commands = SglangTranslator().translate(plan)
        args = commands[0].cli_args
        assert args[args.index("--random-seed") + 1] == "42"
