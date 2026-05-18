"""
vLLM backend translator.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from loguru import logger

from sieval.infer.backends.translator import BackendCommand, BackendTranslator
from sieval.infer.topology.models import DeploymentPlan, RoleAssignment

_DEFAULT_PORT = 8000


class VllmTranslator(BackendTranslator):
    """Translate DeploymentPlan into vLLM launch commands."""

    def translate(self, plan: DeploymentPlan) -> list[BackendCommand]:
        commands: list[BackendCommand] = []
        for i, assignment in enumerate(plan.assignments):
            cmd = self._translate_assignment(plan, assignment, port_offset=i)
            commands.append(cmd)
        return commands

    def _translate_assignment(
        self,
        plan: DeploymentPlan,
        assignment: RoleAssignment,
        port_offset: int = 0,
    ) -> BackendCommand:
        topo = assignment.topology
        port = _DEFAULT_PORT + port_offset

        args = [
            "vllm",
            "serve",
            plan.checkpoint,
            "--port",
            str(port),
        ]

        # Parallel topology → CLI flags
        if topo.tp > 1:
            args.extend(["--tensor-parallel-size", str(topo.tp)])
        if topo.dp > 1:
            args.extend(["--data-parallel-size", str(topo.dp)])
        if topo.pp > 1:
            args.extend(["--pipeline-parallel-size", str(topo.pp)])
        if topo.ep is not None and topo.ep > 1:
            args.append("--enable-expert-parallel")
        if topo.cp is not None:
            # vLLM uses separate flags for prefill/decode CP;
            # Phase 1 (single role) uses the same value for both
            args.extend(["--prefill-context-parallel-size", str(topo.cp)])
            args.extend(["--decode-context-parallel-size", str(topo.cp)])
        if topo.enable_dp_attention:
            # vLLM doesn't support DPA — warn and skip
            logger.warning("vLLM does not support --enable-dp-attention, ignoring")

        if plan.deterministic and "seed" not in assignment.engine_params:
            args.extend(["--seed", str(plan.seed)])

        # Engine params (recipe profile: dtype, max_model_len, etc.)
        for key, value in assignment.engine_params.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    args.append(flag)
            else:
                args.extend([flag, str(value)])

        health_url = f"http://{assignment.devices.host}:{port}/health"

        # Batch-invariant kernel path; locked because it's part of the
        # deterministic contract. Ref: Thinking Machines, "Defeating
        # Nondeterminism in LLM Inference" (2025).
        if plan.deterministic:
            env: dict[str, str] = {"VLLM_BATCH_INVARIANT": "1"}
            locked_env_keys: frozenset[str] = frozenset({"VLLM_BATCH_INVARIANT"})
        else:
            env = {}
            locked_env_keys = frozenset()

        return BackendCommand(
            cli_args=args,
            backend="vllm",
            host=assignment.devices.host,
            role=assignment.role,
            health_url=health_url,
            env=env,
            locked_env_keys=locked_env_keys,
        )
