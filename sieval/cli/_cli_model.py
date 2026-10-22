"""``--checkpoint`` / ``--api-base``: the model of a config with no ``models:``.

The flag becomes that config's one ``models:`` entry, so it is recorded in
``effective_config.yaml`` and a rerun of that file needs no flags.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import os
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Annotated

import typer

CheckpointOption = Annotated[
    str | None,
    typer.Option(
        "--checkpoint",
        help="Checkpoint to launch and evaluate, for a config with no 'models:'.",
    ),
]

ApiBaseOption = Annotated[
    str | None,
    typer.Option(
        "--api-base",
        help="Endpoint to evaluate, for a config with no 'models:'. Needs --model.",
    ),
]


@dataclass(frozen=True)
class CliModel:
    """The model ``--checkpoint`` or ``--api-base`` names. Exactly one is set."""

    checkpoint: str | None = None
    api_base: str | None = None

    def __post_init__(self) -> None:
        if (self.checkpoint is None) == (self.api_base is None):
            raise ValueError("CliModel takes exactly one of checkpoint and api_base")
        for flag, value in (
            ("--checkpoint", self.checkpoint),
            ("--api-base", self.api_base),
        ):
            if value is not None and not value:
                raise ValueError(f"{flag} must not be empty")

    @classmethod
    def from_options(
        cls, *, checkpoint: str | None, api_base: str | None, model: str | None
    ) -> "CliModel | None":
        """The model the flags name, or ``None`` when neither is given.

        A checkpoint is made absolute, so the recorded config reruns from any
        directory.
        """
        if checkpoint is not None and api_base is not None:
            raise ValueError(
                "--checkpoint and --api-base name two models; give one of them"
            )
        if checkpoint is None and api_base is None:
            return None
        if api_base is not None and not model:
            raise ValueError(
                "--api-base needs --model to name the model the endpoint serves"
            )
        if checkpoint:
            checkpoint = os.path.abspath(os.path.expanduser(checkpoint))
        return cls(checkpoint=checkpoint, api_base=api_base)

    def entry(self, model: str | None) -> tuple[str, dict]:
        """The ``models:`` key and entry; *model* (``--model``) is ``name:``."""
        if self.checkpoint is not None:
            entry: dict = {"path": self.checkpoint}
        else:
            entry = {"api_base": self.api_base}
        if model:
            entry["name"] = model
        key = PurePosixPath(model or self.checkpoint or "").name
        if not key:
            raise ValueError("Cannot name a model entry; pass --model")
        return key, entry


def apply_cli_model(cfg: dict, cli_model: CliModel | None, model: str | None) -> dict:
    """Add *cli_model* as the only ``models:`` entry of *cfg*, in place.

    Raises:
        ValueError: *cfg* already defines models.
    """
    if cli_model is None:
        return cfg
    existing = cfg.get("models")
    if existing:
        names = ", ".join(sorted(existing)) if isinstance(existing, dict) else "…"
        flag = "--checkpoint" if cli_model.checkpoint is not None else "--api-base"
        raise ValueError(
            f"{flag} is for a config with no 'models:', but this config "
            f"defines {names}. Drop {flag} or the config's 'models:'."
        )
    key, entry = cli_model.entry(model)
    cfg["models"] = {key: entry}
    return cfg
