"""
Backend translator protocol and command data model.

Translators convert a DeploymentPlan into backend-specific launch commands.
They are stateless -- all information comes from the DeploymentPlan.

AI-Generated Code - Claude Opus 4.6 (Anthropic)
"""

from dataclasses import dataclass, field
from typing import Protocol

from loguru import logger

from sieval.infer.topology.models import DeploymentPlan


@dataclass
class BackendCommand:
    """One executable backend launch command."""

    cli_args: list[str]
    backend: str = ""  # engine name ("sglang", "vllm")
    host: str = "localhost"
    env: dict[str, str] = field(default_factory=dict)
    role: str = "full"
    working_dir: str | None = None
    gpu_ids: list[int] | None = None
    health_url: str | None = None
    # Env keys the translator declares non-overridable (e.g. determinism
    # contract bits). `inject_user_env` raises when user env touches any
    # of these instead of silently weakening the contract.
    locked_env_keys: frozenset[str] = field(default_factory=frozenset)


def inject_user_env(commands: list[BackendCommand], user_env: dict[str, str]) -> None:
    """Merge user-specified env vars into backend commands.

    User env intentionally overrides translator-set values (escape hatch for
    novel configurations).  Collisions are logged as warnings so operators
    notice when infra-critical keys like CUDA_VISIBLE_DEVICES are shadowed.
    Keys declared in ``locked_env_keys`` are off-limits and raise instead.
    """
    if not user_env:
        return
    for cmd in commands:
        locked = cmd.locked_env_keys & user_env.keys()
        if locked:
            raise ValueError(
                "Cannot override translator-locked env var(s): "
                f"{sorted(locked)}. Disable the feature that locked them "
                "(e.g. turn off deterministic mode) instead of overriding."
            )
        collisions = cmd.env.keys() & user_env.keys()
        if collisions:
            logger.warning(
                "User env overrides translator-set keys: {}",
                ", ".join(sorted(collisions)),
            )
        cmd.env.update(user_env)


class BackendTranslator(Protocol):
    """Translate a DeploymentPlan into backend-specific commands."""

    def translate(self, plan: DeploymentPlan) -> list[BackendCommand]: ...
