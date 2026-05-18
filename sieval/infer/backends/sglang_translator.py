"""
SGLang backend translator.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from sieval.infer.backends.translator import BackendCommand, BackendTranslator
from sieval.infer.topology.models import DeploymentPlan, RoleAssignment

_DEFAULT_PORT = 30000


class SglangTranslator(BackendTranslator):
    """Translate DeploymentPlan into SGLang launch commands."""

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
            "sglang",
            "serve",
            "--model-path",
            plan.checkpoint,
            "--port",
            str(port),
        ]

        # Parallel topology → CLI flags
        if topo.enable_dp_attention:
            # DPA semantics: sglang --tp = total GPUs, --dp-size = DP groups.
            # sglang internally computes attention TP = --tp / --dp-size.
            sglang_tp = topo.tp * topo.dp
            args.extend(["--tp", str(sglang_tp)])
            if topo.dp > 1:
                args.extend(["--dp-size", str(topo.dp)])
            args.append("--enable-dp-attention")
        else:
            # Non-DPA: --tp is per-group TP (standard semantics)
            if topo.tp > 1:
                args.extend(["--tp", str(topo.tp)])
            if topo.dp > 1:
                args.extend(["--dp-size", str(topo.dp)])
        if topo.pp > 1:
            args.extend(["--pp-size", str(topo.pp)])
        if topo.ep is not None and topo.ep > 1:
            args.extend(["--ep-size", str(topo.ep)])
        if topo.cp is not None:
            args.extend(["--attn-cp-size", str(topo.cp)])

        if plan.deterministic:
            # Batch-invariant kernels. Ref: Thinking Machines, "Defeating
            # Nondeterminism in LLM Inference" (2025).
            args.append("--enable-deterministic-inference")
            if "random_seed" not in assignment.engine_params:
                args.extend(["--random-seed", str(plan.seed)])

        # Engine params (recipe profile: dtype, max_model_len, etc.)
        for key, value in assignment.engine_params.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    args.append(flag)
            else:
                args.extend([flag, str(value)])

        health_url = f"http://{assignment.devices.host}:{port}/health"

        return BackendCommand(
            cli_args=args,
            backend="sglang",
            host=assignment.devices.host,
            role=assignment.role,
            health_url=health_url,
        )
